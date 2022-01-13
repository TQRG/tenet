from pathlib import Path

import pandas as pd
from typing import Union

from tqdm import tqdm

from securityaware.handlers.plugin import PluginHandler


class MatchLocation(PluginHandler):
    """
        MatchLocation plugin
    """

    class Meta:
        label = "match_location"

    def __init__(self, **kw):
        super().__init__(**kw)
        self.diff_dataset = None

    def run(self, dataset: pd.DataFrame, offset: int = 1, negate: bool = False,
            add_proj: bool = True, **kwargs) -> Union[pd.DataFrame, None]:
        """
            Matches the location of the alerts with the diff dataset

            :param node: context
            :param dataset: data from the previous cell
            :param negate: looks for locations that do not match
            :param add_proj: adds the project name to the file path
            :param offset: considers locations with the offset (locs = [start_line + i for i in range(offset)])
        """

        if dataset is None:
            return None

        diff_dataset_path = Path(self.get('diff_dataset'))

        if not diff_dataset_path.exists():
            self.app.log.error(f"Diff dataset {diff_dataset_path} not found")
            return

        self.diff_dataset = pd.read_csv(str(diff_dataset_path))

        if self.diff_dataset.empty:
            self.app.log.error(f"Diff dataset is empty")
            return None

        if add_proj:
            self.diff_dataset['location'] = self.diff_dataset.apply(lambda r: f"{r.project},{r.fpath},{r.sline}", axis=1)
        else:
            self.diff_dataset['location'] = self.diff_dataset.apply(lambda r: f"{r.fpath},{r.sline}", axis=1)

        filtered_dataset = []

        for i, row in tqdm(dataset.iterrows()):
            if add_proj:
                files_path = f"{row.project},{row.fpath}"
            else:
                files_path = row.fpath

            if self.has_locations(file_path=files_path, line=row.sline, offset=offset, negate=negate):
                filtered_dataset.append(row)

        if filtered_dataset:
            return pd.DataFrame(filtered_dataset, columns=list(dataset.columns.values))

        return None

    def has_locations(self, file_path: str, line: int, offset: int = 1, negate: bool = False):
        """
            Lookup if alert location matches diff location
        """
        for i in range(offset):
            if not self.diff_dataset[self.diff_dataset['location'] == f"{file_path},{line + i}"].empty:
                if negate:
                    return False
                return True

        if negate:
            return True

        return False


def load(app):
    app.handler.register(MatchLocation)
