import time
import pandas as pd
import tqdm

from pathlib import Path
from typing import Union, Tuple, List

from securityaware.core.diff_labeller.labeler import Labeler as DiffLabeler
from securityaware.core.exc import SecurityAwareWarning
from securityaware.data.diff import Entry, DiffBlock, InlineDiff
from securityaware.data.runner import Runner, Task
from securityaware.handlers.plugin import PluginHandler
from securityaware.handlers.runner import ThreadPoolWorker


class Labeler(PluginHandler):
    """
        Labels plugin
    """

    class Meta:
        label = "labeler"

    def __init__(self, **kw):
        super().__init__(**kw)
        self.multi_label = False
        self.file_size_limit = None

    def get_file_str(self, file: Path) -> str:
        """
            Returns the content in the files
        """

        if file.exists():
            if self.file_size_limit and file.stat().st_size > self.file_size_limit:
                raise ValueError(f"File {file} size {file.stat().st_size} greater than limit")
            self.app.log.info(f"Reading {file}")
            with file.open(mode="r") as bf:
                return bf.read()

        raise ValueError(f"File {file} not found.")

    def run(self, dataset: pd.DataFrame, multi_label: bool = False, file_size_limit: int = None,
            **kwargs) -> Union[pd.DataFrame, None]:
        """
            runs the plugin
        """

        self.set('dataset', str(self.output))

        if self.output.exists():
            return pd.read_csv(str(self.output))

        out_dir = self.output.parent

        self.multi_label = multi_label

        if file_size_limit:
            self.file_size_limit = file_size_limit

        runner_data = Runner()
        threads = self.app.get_config('local_threads')
        tasks = []
        total = len(dataset)

        self.app.log.info(f"Creating {total} tasks")

        for i, entry in tqdm.tqdm(dataset.iterrows()):
            task = Task()
            task['id'] = i
            task['entry'] = Entry(a_proj=entry['a_proj'], b_proj=entry['b_proj'], a_file=Path(entry['a_file']),
                                  diff_block=DiffBlock(start=entry['start'], a_path=entry['a_path'],
                                                       b_path=entry['b_path']), b_file=Path(entry['b_file']),
                                  label=entry['label'])
            task['output_dir'] = out_dir / 'inline'
            tasks.append(task)

        worker = ThreadPoolWorker(runner_data, tasks=tasks, threads=threads, logger=self.app.log,
                                  func=self.to_inline_task)
        worker.start()

        while worker.is_alive():
            time.sleep(1)

        # Save similarity ratio as a csv file in the same output folder
        with self.output.open(mode="w") as out, \
                (out_dir / f"{self.output.stem}.ratios.csv").open(mode="w") as ratio_file:

            out.write("project,fpath,sline,scol,eline,ecol,label\n")
            ratio_file.write("project_a,fpath_a,project_b,fpath_b,sim_ratio\n")

            for res in runner_data.finished:
                if 'result' in res and res['result']:
                    if not res['result'][0]:
                        continue
                    ratio_file.write(f"{res['result'][1]}\n")
                    for inline_diff in res['result'][0]:
                        out.write(f"{inline_diff}\n")

        return pd.read_csv(str(self.output))

    def to_inline_task(self, task: Task):
        return self.to_inline(entry=task['entry'], output_dir=task['output_dir'])

    def to_inline(self, entry: Entry, output_dir: Path) -> Tuple[List[InlineDiff], float]:
        try:
            a_str = self.get_file_str(file=entry.a_file)
            b_str = self.get_file_str(file=entry.b_file)
            # Perform pretty-printing and diff comparison
            labeler = DiffLabeler(a_proj=entry.a_proj, b_proj=entry.b_proj, diff_block=entry.diff_block, a_str=a_str,
                                  b_str=b_str, inline_proj_dir=output_dir / f"{entry.a_proj}_{entry.b_proj}")
            # Save the pretty printed inline diffs
            labeler.pretty_printing()
            labeler(unsafe_label='unsafe' if not self.multi_label else entry.label)

            try:
                # Calc and check similarity ratio
                labeler.calc_sim_ratio()
                return labeler.inline_diffs, labeler.sim_ratio
            except SecurityAwareWarning as saw:
                self.app.log.warning(str(saw))

            del labeler

        except (AssertionError, ValueError) as e:
            self.app.log.error(str(e))

        return [], 0


def load(app):
    app.handler.register(Labeler)
