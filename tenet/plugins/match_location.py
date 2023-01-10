import itertools
from pathlib import Path

import pandas as pd
from typing import Union

from tqdm import tqdm

from tenet.handlers.plugin import PluginHandler


class MatchLocation(PluginHandler):
    """
        MatchLocation plugin
    """

    class Meta:
        label = "match_location"

    def __init__(self, **kw):
        super().__init__(**kw)
        self.diff_dataset = None
        self.ver_dataset = None

    def run(self, dataset: pd.DataFrame, offset: int = 1, negate: bool = False, **kwargs) -> Union[pd.DataFrame, None]:
        """
            Matches the location of the alerts with the diff dataset

            :param dataset: data from the previous cell
            :param negate: looks for locations that do not match
            :param offset: considers locations with the offset (locs = [start_line + i for i in range(offset)])
        """

        if dataset is None:
            return None

        diff_dataset_path = Path(self.get('diff_dataset'))
        versions_dataset_path = Path(self.get('ver_dataset'))

        if not diff_dataset_path.exists():
            self.app.log.error(f"Diff dataset {diff_dataset_path} not found")
            return

        if not versions_dataset_path.exists():
            self.app.log.error(f"Versions dataset {versions_dataset_path} not found")
            return

        self.diff_dataset = pd.read_csv(str(diff_dataset_path))
        self.ver_dataset = pd.read_csv(str(versions_dataset_path))

        if self.diff_dataset.empty:
            self.app.log.error(f"Diff dataset is empty")
            return None

        if self.ver_dataset.empty:
            self.app.log.error(f"Version dataset is empty")
            return None

        # TODO: simplify this
        self.diff_dataset['location'] = self.diff_dataset.apply(lambda r: f"{r.project},{r.fpath},{r.sline}", axis=1)
        dataset['location'] = dataset.apply(lambda r: f"{r.project},{r.fpath}", axis=1)
        self.ver_dataset['parent'] = self.ver_dataset.apply(lambda r: f"{r.a_proj},{r.a_path}", axis=1)
        self.ver_dataset['child'] = self.ver_dataset.apply(lambda r: f"{r.b_proj},{r.b_path}", axis=1)

        filtered_dataset = []
        pairs = {}
        to_check = {}

        for i, row in tqdm(dataset.iterrows()):
            pair = self.get_pair(row.location)

            if not pair:
                continue

            # check for matching pairs of alerts by project and file name
            if pair['parent'] in pairs:
                to_check[pair['parent']] = pair['child']
            else:
                pairs[pair['parent']] = pair['child']

        for parent, child in to_check.items():
            # remove pair from parent
            del pairs[parent]

            parent_rows = dataset.index[dataset['location'] == parent].tolist()
            child_row = dataset.index[dataset['location'] == child].to_list()
            permutations = list(itertools.product(parent_rows, child_row))

            if len(permutations) == 0:
                self.app.log.warning(f"Skipping {parent}: 0 permutations.")
                continue

            self.app.log.info(f"Scanning with offset of {6} lines: {len(permutations)} permutations.")

            for parent_row, child_row in permutations:
                if self.is_actioanble(parent_row=dataset.iloc[parent_row],
                                      child_row=dataset.iloc[child_row], offset=offset):
                    filtered_dataset.append(dataset.iloc[parent_row])
                    filtered_dataset.append(dataset.iloc[child_row])
                    break

        # add the rest of the alerts
        for parent, child in pairs.items():
            parent_res = dataset.index[dataset['location'] == parent]
            child_res = dataset.index[dataset['location'] == child]

            if parent_res.empty:
                child_row, = child_res
                filtered_dataset.append(dataset.iloc[child_row])
            else:
                parent_row, = parent_res
                filtered_dataset.append(dataset.iloc[parent_row])

        if filtered_dataset:
            dataset = pd.DataFrame(filtered_dataset, columns=list(dataset.columns.values))
            del dataset['location']
            return dataset

        return None

    def get_pair(self, file_path: str):
        res = self.ver_dataset.index[self.ver_dataset['parent'] == file_path]

        if not res.empty:
            is_parent, = res
            return {'parent': self.ver_dataset.iloc[is_parent].parent,
                    'child': self.ver_dataset.iloc[is_parent].child}
        else:
            res = self.ver_dataset.index[self.ver_dataset['child'] == file_path]
            if not res.empty:
                is_child, = res
                return {'parent': self.ver_dataset.iloc[is_child].parent,
                        'child':  self.ver_dataset.iloc[is_child].child}

            self.app.log.warning(f"{file_path} not found.")
            return None

    def is_actioanble(self, parent_row: pd.Series, child_row: pd.Series, offset: int = 1):
        """
            Lookup if alert location matches diff location in parent but not in child.
        """
        child_loc = f"{child_row.project}{child_row.fpath}"
        parent_loc = f"{parent_row.project}{parent_row.fpath}"

        for i in range(offset):
            if f"{parent_loc}{parent_row.sline}" == f"{child_loc}{child_row.sline + i}":
                self.app.log.debug(f"Matched {parent_loc}{parent_row.sline} {child_loc}{child_row.sline + i}")
                return False

            if f"{parent_loc}{parent_row.sline + 1}" == f"{child_loc}{child_row.sline}":
                self.app.log.debug(f"Matched {parent_loc}{parent_row.sline + 1} {child_loc}{child_row.sline}")
                return False

        return True


def load(app):
    app.handler.register(MatchLocation)
