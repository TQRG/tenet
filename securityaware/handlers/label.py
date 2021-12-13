from typing import List, Tuple

import time
import pandas as pd

from pathlib import Path
from cement import Handler
from tqdm import tqdm

from securityaware.core.interfaces import HandlersInterface
from securityaware.core.diff_labeller.misc import check_or_create_dir
from securityaware.data.diff import Entry, DiffBlock, InlineDiff
from securityaware.data.runner import Task, Runner
from securityaware.handlers.runner import ThreadPoolWorker
from securityaware.core.diff_labeller.labeler import Labeler
from securityaware.core.exc import SecurityAwareWarning


class LabelHandler(HandlersInterface, Handler):
    class Meta:
        label = 'label'

    def __init__(self, **kw):
        super().__init__(**kw)
        self.multi_label = False
        self.file_size_limit = None

    def get_file_str(self, file: Path) -> str:
        if file.exists():
            if self.file_size_limit and file.stat().st_size > self.file_size_limit:
                raise ValueError(f"File {file} size {file.stat().st_size} greater than limit")
            self.app.log.info(f"Reading {file}")
            with file.open(mode="r") as bf:
                return bf.read()

    def __call__(self, output_dir: Path, dataset: Path):
        check_or_create_dir(output_dir)
        runner_data = Runner()
        threads = self.app.get_config('local_threads')
        dataframe = pd.read_csv(dataset)
        tasks = []
        total = len(dataframe)

        self.app.log.info(f"Creating {total} tasks")

        for i, entry in tqdm(dataframe.iterrows()):
            task = Task()
            task['id'] = i
            task['entry'] = Entry(a_proj=entry['a_proj'], b_proj=entry['b_proj'], a_file=Path(entry['a_file']),
                                  diff_block=DiffBlock(start=entry['start'], a_path=entry['a_path'],
                                                       b_path=entry['b_path']), b_file=Path(entry['b_file']),
                                  label=entry['label'])
            task['output_dir'] = output_dir / 'inline'
            tasks.append(task)

        worker = ThreadPoolWorker(runner_data, tasks=tasks, threads=threads, logger=self.app.log,
                                  func=self.to_inline_task)
        worker.start()

        while worker.is_alive():
            time.sleep(1)

        return runner_data

    def to_inline_task(self, task: Task):
        return self.to_inline(entry=task['entry'], output_dir=task['output_dir'])

    def to_inline(self, entry: Entry, output_dir: Path) -> Tuple[List[InlineDiff], float]:
        try:
            a_str = self.get_file_str(file=entry.a_file)
            b_str = self.get_file_str(file=entry.b_file)

            # Perform pretty-printing and diff comparison
            labeler = Labeler(a_proj=entry.a_proj, b_proj=entry.b_proj, diff_block=entry.diff_block, a_str=a_str,
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
