import ast
import pandas as pd

from pathlib import Path
from typing import Union
from tqdm import tqdm

from securityaware.data.diff import FunctionBoundary, InlineDiff
from securityaware.data.schema import ContainerCommand
from securityaware.handlers.plugin import PluginHandler


class JSCodeShiftHandler(PluginHandler):
    """
        JSCodeShift plugin
    """

    class Meta:
        label = "jscodeshift"

    def __init__(self, **kw):
        super().__init__(**kw)
        self.fn_boundaries = None

    def run(self, dataset: pd.DataFrame, image_name: str = "jscodeshift", single_unsafe_fn: bool = False,
            **kwargs) -> Union[pd.DataFrame, None]:
        """
            runs the plugin
        """
        self.fn_boundaries_file = (self.path / 'output.txt')
        self.set('fn_boundaries_file', self.fn_boundaries_file)
        self.set('dataset_path', self.output)

        if not self.get('raw_files_path'):
            self.app.log.warning(f"raw files path not instantiated.")
            return None

        raw_files_path = Path(str(self.get('raw_files_path')).replace(str(self.app.workdir), str(self.app.bind)))

        if not self.fn_boundaries_file.exists():
            # TODO: fix the node name
            container = self.container_handler.run(image_name=image_name, node_name=self.node.name)
            cmd = ContainerCommand(org=f"jscodeshift -p -s -d -t /js-fn-rearrange/transforms/outputFnBoundary.js {raw_files_path}")
            self.container_handler.run_cmds(container.id, [cmd])
            self.container_handler.stop(container)

        try:
            if not self.fn_boundaries_file.exists():
                self.app.log.error(f"jscodeshift output file {self.fn_boundaries_file} not found")
                return None

            if not self.fn_boundaries_file.stat().st_size > 0:
                self.app.log.error(f"jscodeshift output file {self.fn_boundaries_file} is empty")
                return

        except TypeError as te:
            self.app.log.error(te)
            self.app.log.warning(f"jscodeshift output file not instantiated.")
            return None

        outputs = self.fn_boundaries_file.open(mode='r').readlines()
        self.fn_boundaries = {}
        raw_files_path = str(raw_files_path).replace(str(self.app.workdir), str(self.app.bind))

        for line in outputs:
            clean_line = line.replace("'", '')
            fn_dict = ast.literal_eval(clean_line)
            fn_path = fn_dict['path'].replace(raw_files_path + '/', '')
            del fn_dict['path']
            self.fn_boundaries[fn_path] = fn_dict

        # TODO: fix this drop of columns

        if 'sim_ratio' in dataset.columns:
            dataset = dataset.drop(columns=['sim_ratio'])
        if 'rule_id' in dataset.columns:
            dataset = dataset.drop(columns=['rule_id'])

        for (owner, project, version, fpath), rows in tqdm(dataset.groupby(['owner', 'project', 'version', 'fpath'])):
            self.multi_task_handler.add(group_inline_diff=rows, path=str(Path(owner, project, version, fpath)),
                                        owner=owner, project=project, version=version, fpath=fpath)
        self.multi_task_handler(func=self.convert_bound)

        fn_bounds = self.multi_task_handler.results(expand=True)

        # TODO: refactor the code in the if block
        if fn_bounds:
            df = pd.DataFrame(fn_bounds)

            # Remove duplicates
            df = df.drop_duplicates(ignore_index=True)
            df = df.reset_index().rename(columns={'index': 'func_id'})
            # df["n_mut"] = [0] * df.shape[0]

            if single_unsafe_fn:
                safe = df[df['label'] == 'safe']
                unsafe = df[df['label'] == 'unsafe'].groupby(['project', 'fpath', 'label']).filter(lambda x: len(x) < 2)

                return pd.concat([safe, unsafe]).sort_values(by=['func_id'])

            return df

        return None

    def convert_bound(self, group_inline_diff: pd.Series, path: str, owner: str, version: str, project: str, fpath: str):
        """
            Finds the function boundaries for the code snippet.
        """
        fn_bounds = []

        if path not in self.fn_boundaries:
            self.app.log.error(f"file {path} not found in jscodeshift output")
            return None

        fn_boundaries = self.fn_boundaries[path]
        fn_decs, fn_exps = FunctionBoundary.parse_fn_inline_diffs(fn_boundaries, owner=owner, project=project,
                                                                  version=version, fpath=fpath)

        for index, row in group_inline_diff.to_dict('index').items():
            inline_diff = InlineDiff(**row)
            self.app.log.info(f'Matching inline diff {inline_diff} with {len(fn_decs)} fn decs and {len(fn_exps)} fn exps')
            fn_bound = None

            for fn_dec in fn_decs:
                if fn_dec.is_contained(inline_diff):
                    fn_dec.label = inline_diff.label
                    fn_bound = fn_dec

            if fn_bound:
                fn_bounds.append(fn_bound.to_dict(ftype='fn_dec'))
                continue

            for fn_exp in fn_exps:
                if fn_exp.is_contained(inline_diff):
                    fn_exp.label = inline_diff.label
                    fn_bound = fn_exp

            if fn_bound:
                fn_bounds.append(fn_bound.to_dict(ftype='fn_exp'))

        return fn_bounds


def load(app):
    app.handler.register(JSCodeShiftHandler)
