import pandas as pd
import tqdm

from pathlib import Path
from typing import Union

from securityaware.core.diff_labeller.labeler import Labeler as DiffLabeler
from securityaware.data.diff import Entry, DiffBlock
from securityaware.handlers.plugin import PluginHandler


class Labeler(PluginHandler):
    """
        Labels plugin
    """

    class Meta:
        label = "labeler"

    def __init__(self, **kw):
        super().__init__(**kw)
        self.file_size_limit = None
        self.inline_dir = None
        self.sim_ratio_thresh = None

    def set_dirs(self):
        self.inline_dir = self.output.parent / 'inline'

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

    def run(self, dataset: pd.DataFrame, file_size_limit: int = None, sim_ratio_tresh: float = 0.8,
            **kwargs) -> Union[pd.DataFrame, None]:
        """
            runs the plugin
        """
        if not self.get('raw_files_path'):
            self.app.log.warning(f"Raw files path not instantiated.")
            return None

        if not self.get('raw_files_path').exists():
            self.app.log.warning(f"Train data file not found.")
            return None

        self.sim_ratio_thresh = sim_ratio_tresh
        self.set('dataset', str(self.output))
        self.set_dirs()

        if file_size_limit:
            self.file_size_limit = file_size_limit

        self.app.log.info(f"Creating {len(dataset)} tasks")

        for i, entry in tqdm.tqdm(dataset.iterrows()):
            diff_block = DiffBlock(start=entry['start'], a_path=entry['a_path'], b_path=entry['b_path'])
            entry = Entry(a_version=entry['a_version'], b_version=entry['b_version'], label=entry['label'],
                          diff_block=diff_block, owner=entry['owner'], project=entry['project'])
            self.multi_task_handler.add(entry=entry)

        self.multi_task_handler(func=self.to_inline)
        inline_diffs = self.multi_task_handler.results(expand=True)

        if inline_diffs:
            return pd.DataFrame.from_dict(inline_diffs)

        return None

    def to_inline(self, entry: Entry) -> list:
        inline_proj_dir = self.inline_dir / f"{entry.owner}_{entry.project}_{entry.a_version}_{entry.b_version}"

        try:
            a_str = self.get_file_str(file=self.get('raw_files_path') / entry.full_a_path)
            b_str = self.get_file_str(file=self.get('raw_files_path') / entry.full_b_path)
            # Perform pretty-printing and diff comparison
            labeler = DiffLabeler(entry=entry, a_str=a_str, b_str=b_str, inline_proj_dir=inline_proj_dir)
            labeler(unsafe_label=entry.label)

            # Calc and check similarity ratio
            if labeler.calc_sim_ratio() < self.sim_ratio_thresh:
                self.app.log.warning(labeler.warning())
                return []

            return [il.to_dict(sim_ratio=round(labeler.sim_ratio, 3)) for il in labeler.inline_diffs]

        except (AssertionError, ValueError, IndexError) as e:
            # TODO: fix the IndexError
            self.app.log.error(f"{inline_proj_dir}_{entry.diff_block.a_path} {e}")

        return []


def load(app):
    app.handler.register(Labeler)
