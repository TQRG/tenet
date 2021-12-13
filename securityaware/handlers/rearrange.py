import time
import pandas as pd

from pathlib import Path
from cement import Handler

from securityaware.core.interfaces import HandlersInterface
from securityaware.core.rearrange.convert_bound import parse_fn_bound, transform_inline_diff
from securityaware.data.diff import InlineDiff
from securityaware.data.runner import Runner, Task
from securityaware.handlers.runner import ThreadPoolWorker


class RearrangeHandler(HandlersInterface, Handler):
    class Meta:
        label = 'rearrange'

    def __call__(self, csv_in: Path, output_fn: Path, code_dir: Path):
        """
            :param csv_in: Input csv file.
            :param output_fn: `outputFnBoundary.js` file location.
            :param code_dir: Raw code directory.
        """

        runner_data = Runner()
        threads = self.app.get_config('local_threads')

        dataframe = pd.read_csv(csv_in)
        dataframe.rename(columns={"fpath": "file_path"}, inplace=True)
        tasks = []

        for (project, file_path), rows in dataframe.groupby(['project', 'file_path']):
            # self.app.log.info(f"Creating task for record. {index}/{total}")
            code_path = code_dir / f"{project}/{file_path}"

            if code_path.exists():
                task = Task()
                task['id'] = (rows.index[0], rows.index[-1])
                task['group_inline_diff'] = rows
                task['code_path'] = code_path
                task['transform_path'] = output_fn
                tasks.append(task)

        worker = ThreadPoolWorker(runner_data, tasks=tasks, threads=threads, logger=self.app.log,
                                  func=self.convert_bound_task)
        worker.start()

        while worker.is_alive():
            time.sleep(1)

        return runner_data

    def convert_bound_task(self, task: Task):
        return self.convert_bound(group_inline_diff=task['group_inline_diff'],  code_path=task['code_path'],
                                  transform_path=task['transform_path'])

    def convert_bound(self, group_inline_diff: pd.Series, code_path: Path, transform_path: Path):
        fn_bounds = []

        for index, row in group_inline_diff.to_dict('index').items():
            inline_diff = InlineDiff(**row)

            #if fn_bounds and inline_diff.is_same(fn_bounds[-1]) and inline_diff.inbounds(fn_bounds[-1]):
            #    self.app.log.info(f"Skipped {index}.")
            #else:
            self.app.log.info(f"Transforming inline diff on row {index}")

            fn_bound_str = transform_inline_diff(transform_path=str(transform_path), code_path=str(code_path),
                                                 inline_diff=inline_diff)
            if fn_bound_str:
                fn_bound = parse_fn_bound(fn_bound_str, inline_diff)

                if not fn_bound:
                    self.app.log.error("Error when applying jscodeshift: row {}\n{},{}\n{}".format(
                        index, inline_diff.project, inline_diff.file_path, "\n".join(fn_bound_str)))
                else:
                    fn_bounds.append(fn_bound)

        return fn_bounds
