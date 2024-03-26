import ast
import pandas as pd

from pathlib import Path
from typing import Union

from tenet.data.schema import ContainerCommand
from tenet.handlers.plugin import PluginHandler
from arepo.models.data import DatasetModel
from arepo.models.vcs.symbol import FunctionModel
from arepo.utils import get_digest


class JSCodeShiftHandler(PluginHandler):
    """
        JSCodeShift plugin
    """

    class Meta:
        label = "jscodeshift"

    def __init__(self, **kw):
        super().__init__(**kw)

    def set_sources(self):
        self.set('dataset_path', self.output)

    def get_sinks(self):
        self.get('raw_files_path')

    def run(self, dataset: DatasetModel, image_name: str = "epicosy/securityaware:jscodeshift",
            **kwargs) -> Union[pd.DataFrame, None]:
        """
            runs the plugin
        """
        workdir = Path('/tmp/jscodeshift')
        workdir.mkdir(parents=True, exist_ok=True)
        self.path = workdir
        self.app.extend('workdir', workdir)
        self.app.extend('bind', workdir)
        container = self.container_handler.run(image_name=image_name)
        all_fn_bounds = []

        for v in dataset.vulnerabilities:
            for c in v.commits:
                if c.kind == 'parent':
                    self.app.log.warning(f"Commit {c.sha} is a parent commit")
                    continue

                if len(c.parents) == 0:
                    self.app.log.warning(f"Commit {c.sha} has no parent")
                    continue

                for cf in c.files:
                    if len(cf.functions) > 0:
                        self.app.log.warning(f"File {cf.filename} in commit {c.id} already has function boundaries")

                        for f in cf.functions:
                            all_fn_bounds.append({
                                'project': c.repository.name,
                                'fpath': cf.filename,
                                'func_id': f.id,
                                'start_line': f.start_line,
                                'start_col': f.start_col,
                                'end_line': f.end_line,
                                'end_col': f.end_col,
                                'size': f.size
                            })

                        continue

                    repo_path = f"{c.repository.owner}/{c.repository.name}"
                    output_path = workdir / repo_path / c.sha / cf.filename
                    file_content, _ = self.github_handler.get_file_from_raw_url(cf.raw_url, output_path)
                    file_content_lines = file_content.splitlines()

                    if not output_path.exists():
                        self.app.log.error(f"File {cf.filename} not found in {output_path}")
                        continue

                    fn_boundaries_file = output_path.parent / 'output.txt'

                    # TODO: fix the node name
                    if not fn_boundaries_file.exists():
                        cmd = ContainerCommand(org=f"jscodeshift -p -s -d -t /js-fn-rearrange/transforms/outputFnBoundary.js {output_path.parent}")
                        # TODO: fix the working dir
                        self.container_handler.working_dir = output_path.parent
                        self.container_handler.run_cmds(container.id, [cmd])
                        self.container_handler.working_dir = workdir

                    if not fn_boundaries_file.exists():
                        self.app.log.error(f"jscodeshift output file {fn_boundaries_file} not found")
                        continue

                    if not fn_boundaries_file.stat().st_size > 0:
                        self.app.log.error(f"jscodeshift output file {fn_boundaries_file} is empty")
                        continue

                    outputs = fn_boundaries_file.open(mode='r').readlines()
                    fn_bounds = []

                    for line in outputs:
                        clean_line = line.replace("'", '')
                        fn_dict = ast.literal_eval(clean_line)
                        # TODO: change to pass the function type to the function model
                        fn_bounds.extend(fn_dict['fnExps'])
                        fn_bounds.extend(fn_dict['fnDec'])
                        fn_bounds.extend(fn_dict['fnArrow'])

                    session = self.app.db.get_session()

                    for fn in fn_bounds:
                        # TODO: fix the parsing of the output
                        start_line, start_col, end_line, end_col = [int(el) for el in fn.split(',')]
                        url_path = f"{repo_path}/blob/{c.sha}/{cf.filename}#L{start_line}-L{end_line}"
                        fn_id = get_digest(url_path)

                        has_fn = session.query(FunctionModel).filter_by(id=fn_id).first()

                        if has_fn:
                            self.app.log.warning(f"Function {fn_id} already exists")
                            continue

                        size = end_line - start_line
                        name = file_content_lines[start_line - 1][:start_col].strip()
                        content = '\n'.join(file_content_lines[start_line - 1:end_line])
                        fn_model = FunctionModel(id=fn_id, name=name, start_line=start_line, end_line=end_line,
                                                 start_col=start_col, end_col=end_col, size=size, content=content,
                                                 commit_file_id=cf.id)
                        session.add(fn_model)
                        session.commit()
                        all_fn_bounds.append({
                            'project': c.repository.name,
                            'fpath': cf.filename,
                            'func_id': fn_id,
                            'start_line': start_line,
                            'start_col': start_col,
                            'end_line': end_line,
                            'end_col': end_col,
                            'size': size
                        })

        self.container_handler.stop(container)

        # TODO: refactor the code in the if block
        if all_fn_bounds:
            df = pd.DataFrame(all_fn_bounds)

            return df

        return None


def load(app):
    app.handler.register(JSCodeShiftHandler)
