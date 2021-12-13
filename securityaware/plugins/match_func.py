import pandas as pd

from typing import Union
from pathlib import Path
from securityaware.handlers.plugin import PluginHandler


class FilterFunctions(PluginHandler):
    """
        FilterFunctions plugin
    """

    class Meta:
        label = "filter_functions"

    def __init__(self, **kw):
        super().__init__(**kw)
        self.diff_dataset = None

    def run(self, node: dict, cell: dict, dataset: pd.DataFrame, files_path: Path, child_dataset: str = "",
            **kwargs) -> Union[pd.DataFrame, None]:
        """
            Filters the alerts functions between parent and child commits
        """

        diff_dataset_path = Path(diff_dataset)

        if not diff_dataset_path.exists():
            self.app.log.error(f"Diff dataset {diff_dataset} not found")
            return

        self.diff_dataset = self.load_dataset(diff_dataset_path, terminate=True)

        if self.diff_dataset.empty:
            self.app.log.error(f"Diff dataset is empty")
            return None

        self.diff_dataset['s_location'] = self.diff_dataset.apply(lambda r: f"{r.project},{r.fpath},{r.sline}", axis=1)

