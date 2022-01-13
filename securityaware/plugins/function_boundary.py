"""
    Plugin for getting the function boundary from snippets of code
"""
import time

import pandas as pd

from pathlib import Path
from typing import Union

from securityaware.core.rearrange.convert_bound import transform_inline_diff, parse_fn_bound
from securityaware.data.diff import InlineDiff
from securityaware.data.runner import Task, Runner
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
        self.output_fn = None
        self.code_dir = None

    def run(self, dataset: pd.DataFrame, output_fn: str = "",
            **kwargs) -> Union[pd.DataFrame, None]:
        """
            :param node: context
            :param dataset: data from the previous cell
            :param files_path: path to the files from the previous cell
            :param output_fn: `outputFnBoundary.js` file location.
        """
        self.output_fn = Path(output_fn)
        self.set('dataset_path', self.output)

        if not self.output_fn.exists():
            self.app.log.error(f"Transform JS file {output_fn} not found")
            return

        if dataset is None:
            return None

        runner_data = Runner()
        threads = self.app.get_config('local_threads')

        dataset.rename(columns={"fpath": "file_path"}, inplace=True)
        tasks = []
        files_path = self.get('files_path')

        for (project, file_path), rows in dataset.groupby(['project', 'file_path']):
            # self.app.log.info(f"Creating task for record. {index}/{total}")
            code_path = Path(files_path, f"{project}/{file_path}")

            if code_path.exists():
                task = Task()
                task['id'] = (rows.index[0], rows.index[-1])
                task['group_inline_diff'] = rows
                task['code_path'] = code_path
                tasks.append(task)

        worker = ThreadPoolWorker(runner_data, tasks=tasks, threads=threads, logger=self.app.log,
                                  func=self.convert_bound_task)
        worker.start()

        while worker.is_alive():
            time.sleep(1)

        fn_bounds = [fn_bound.to_list() for res in runner_data.finished for fn_bound in res['result'] if
                     'result' in res and res['result']]

        if fn_bounds:
            df = pd.DataFrame(fn_bounds, columns=['project', 'fpath', 'sline', 'scol', 'eline', 'ecol', 'label'])

            # Remove duplicates
            df = df.drop_duplicates(ignore_index=True)
            df = df.reset_index().rename(columns={'index': 'func_id'})
            df["n_mut"] = [0] * df.shape[0]

            return df

        return None

    def convert_bound_task(self, task: Task):
        """
            Maps arguments to the call
        """
        return self.convert_bound(group_inline_diff=task['group_inline_diff'], code_path=task['code_path'])

    def convert_bound(self, group_inline_diff: pd.Series, code_path: Path):
        """
            Finds the function boundaries for the code snippet.
        """
        fn_bounds = []

        for index, row in group_inline_diff.to_dict('index').items():
            inline_diff = InlineDiff(**row)

            # if fn_bounds and inline_diff.is_same(fn_bounds[-1]) and inline_diff.inbounds(fn_bounds[-1]):
            #    self.app.log.info(f"Skipped {index}.")
            # else:
            self.app.log.info(f"Transforming inline diff on row {index}")

            fn_bound_str = transform_inline_diff(transform_path=str(self.output_fn), code_path=str(code_path),
                                                 inline_diff=inline_diff)
            if fn_bound_str:
                fn_bound = parse_fn_bound(fn_bound_str, inline_diff)

                if not fn_bound:
                    self.app.log.error("Error when applying jscodeshift: row {}\n{},{}\n{}".format(
                        index, inline_diff.project, inline_diff.file_path, "\n".join(fn_bound_str)))
                else:
                    fn_bounds.append(fn_bound)

        return fn_bounds


def load(app):
    app.handler.register(FunctionBoundaryHandler)
