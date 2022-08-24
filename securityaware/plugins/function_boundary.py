"""
    Plugin for getting the function boundary from snippets of code
"""
import time

import pandas as pd
import ast
from pathlib import Path
from typing import Union, List

from securityaware.core.exc import Found
from securityaware.core.rearrange.convert_bound import parse_fn_bound
from securityaware.data.diff import InlineDiff, FunctionBoundary
from securityaware.data.runner import Task, Runner
from securityaware.data.schema import ContainerCommand
from securityaware.handlers.plugin import PluginHandler
from securityaware.handlers.runner import ThreadPoolWorker


class FunctionBoundaryHandler(PluginHandler):
    """
        FunctionBoundary plugin
    """

    class Meta:
        label = "function_boundary"

    def __init__(self, **kw):
        super().__init__(**kw)
        self.fn_boundaries = None

    def run(self, dataset: pd.DataFrame, **kwargs) -> Union[pd.DataFrame, None]:
        """
            :param dataset: data from the previous cell
        """
        self.set('dataset_path', self.output)

        if not self.get('files_path'):
            self.app.log.warning(f"raw files path not instantiated.")
            return None

        raw_files_path = self.get('files_path')

        try:
            jscodeshift_output_file = Path(self.get('jscodeshift_output'))

            if not jscodeshift_output_file.exists():
                self.app.log.error(f"jscodeshift output file {jscodeshift_output_file} not found")
                return None

            if not jscodeshift_output_file.stat().st_size > 0:
                self.app.log.error(f"jscodeshift output file {jscodeshift_output_file} is empty")
                return

        except TypeError as te:
            self.app.log.error(te)
            self.app.log.warning(f"jscodeshift output file not instantiated.")
            return None

        outputs = jscodeshift_output_file.open(mode='r').readlines()
        self.fn_boundaries = {}
        raw_files_path = str(raw_files_path).replace(str(self.app.workdir), str(self.app.bind))

        for line in outputs:
            clean_line = line.replace("'", '')
            fn_dict = ast.literal_eval(clean_line)
            fn_path = fn_dict['path'].replace(raw_files_path + '/', '')
            del fn_dict['path']
            self.fn_boundaries[fn_path] = fn_dict

        runner_data = Runner()
        threads = self.app.get_config('local_threads')
        tasks = []
        dataset.rename(columns={'fpath': 'file_path'}, inplace=True)

        for (project, fpath), rows in dataset.groupby(['project', 'file_path']):
            task = Task()
            task['id'] = (rows.index[0], rows.index[-1])
            task['group_inline_diff'] = rows
            task['path'] = str(Path(project, fpath))
            task['project'] = project
            task['fpath'] = fpath
            tasks.append(task)

        worker = ThreadPoolWorker(runner_data, tasks=tasks, threads=threads, logger=self.app.log,
                                  func=self.convert_bound_task)
        worker.start()

        while worker.is_alive():
            time.sleep(1)

        fn_bounds = []

        for res in runner_data.finished:
            if 'result' in res and res['result']:
                for fn_bound in res['result']:
                    fn_bounds.append(fn_bound.to_list())

        if fn_bounds:
            df = pd.DataFrame(fn_bounds, columns=['project', 'fpath', 'sline', 'scol', 'eline', 'ecol', 'label', 'ftype'])

            # Remove duplicates
            df = df.drop_duplicates(ignore_index=True)
            df = df.reset_index().rename(columns={'index': 'func_id'})
            # df["n_mut"] = [0] * df.shape[0]

            return df

        return None

    def convert_bound_task(self, task: Task):
        """
            Maps arguments to the call
        """
        return self.convert_bound(**task.assets)

    def convert_bound(self, group_inline_diff: pd.Series, path: str, project: str, fpath: str, **kwargs):
        """
            Finds the function boundaries for the code snippet.
        """
        fn_bounds = []

        if path not in self.fn_boundaries:
            self.app.log.error(f"file {path} not found in jscodeshift output")
            return None

        fn_boundaries = self.fn_boundaries[path]
        fn_decs, fn_exps = FunctionBoundary.parse_fn_inline_diffs(fn_boundaries, project=project, fpath=fpath)

        for index, row in group_inline_diff.to_dict('index').items():
            inline_diff = InlineDiff(**row)
            self.app.log.info(f'Matching inline diff {inline_diff} with {len(fn_decs)} fn decs and {len(fn_exps)} fn exps')
            fn_bound = None

            for fn_dec in fn_decs:
                if fn_dec.is_contained(inline_diff):
                    fn_dec.label = inline_diff.label
                    fn_bound = fn_dec

            if fn_bound:
                fn_bounds.append(fn_bound)
                continue

            for fn_exp in fn_exps:
                if fn_exp.is_contained(inline_diff):
                    fn_exp.label = inline_diff.label
                    fn_bound = fn_exp

            if fn_bound:
                fn_bounds.append(fn_bound)

        return fn_bounds


def load(app):
    app.handler.register(FunctionBoundaryHandler)
