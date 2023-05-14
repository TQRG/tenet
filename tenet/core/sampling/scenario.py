import ast
from abc import abstractmethod, ABC

import pandas as pd

from typing import Union, Tuple, Hashable

from tenet.core.diff_labeller.changes import Changes
from tenet.core.exc import TenetError
from tenet.utils.misc import df_init


def get_line_numbers(patch):
    changes = Changes(patch)
    blocks = changes.split_changes()

    for idx, block in enumerate(blocks, start=0):
        sline = changes.get_sdel(idx)
        changes.set_block(block)
        changes.split_block()
        changes.get_deleted_lines(sline)

    for idx, block in enumerate(blocks, start=0):
        sline = changes.get_sadd(idx)
        changes.set_block(block)
        changes.split_block()
        changes.get_added_lines(sline)

    return changes.deleted_lines, changes.added_lines


def get_status(file_changes: dict):
    if 'status' in file_changes:
        return file_changes['status']
    else:
        # some changes may do not have status, therefore look into the patch.
        # Heuristics: If there is no line added, then the patch is only removing lines.
        if '\n+' not in file_changes['patch']:
            return "removed"


def parse_file_changes(file_changes: dict, commit_sha: str, vuln_commit_hash: str) -> dict:
    info = {'additions': file_changes['additions'], 'deletions': file_changes['deletions'],
            'changes': file_changes['changes'], 'raw_url_fix': file_changes['raw_url'],
            }

    if 'patch' in file_changes.keys() and file_changes['patch']:
        lines_deleted, lines_added = get_line_numbers(file_changes['patch'])
        info['locs_deleted'] = str(lines_deleted)
        info['locs_added'] = str(lines_added)

    info['status'] = get_status(file_changes)
    info['raw_url_vuln'] = file_changes['raw_url'].replace(commit_sha, vuln_commit_hash)

    return info


def parse_file_in_row(row: pd.Series, file: str) -> dict:
    target_cols = ['dataset', 'project', 'owner', 'project_url', 'before_first_fix_commit', 'last_fix_commit', 'vuln_id',
                   'cwe_id', 'score', 'bf_class', 'operation']
    row_info = row[target_cols].copy().rename({'before_first_fix_commit': 'vuln_commit_hash', 'raw_url': 'raw_url_vuln',
                                               'last_fix_commit': 'fixed_commit_hash'}).to_dict()
    row_info['file_path'] = file

    return row_info


def get_repo_path(project_url: str):
    url_split = project_url.split("/")
    return url_split[-2], url_split[-1]


class Scenario(ABC):
    def __init__(self, df: pd.DataFrame, scenario_type: str):
        self.df = df
        self.stype = scenario_type

        if 'project' not in df.columns:
            self.df['owner'], self.df['project'] = zip(*self.df['project_url'].apply(get_repo_path))

        new_dataset = []

        # TODO: fix this
        if 'files' in self.df.columns:
            for i, r in self.df.iterrows():
                dict_row = r.to_dict()
                del dict_row['files']

                try:
                    if isinstance(r.files, str):
                        instance = ast.literal_eval(r.files.strip())
                    else:
                        instance = r.files
                except SyntaxError:
                    continue

                for f, values in instance.items():
                    dict_row['file_path'] = f
                    dict_row['file_changes'] = values
                    new_dataset.append(dict_row)

            self.df = pd.DataFrame(new_dataset)

    @abstractmethod
    def generate(self, **kwargs):
        pass

    @abstractmethod
    def augment(self, **kwargs):
        pass

    @abstractmethod
    def remove_changes(self, **kwargs):
        pass

    @abstractmethod
    def drop_duplicates(self, **kwargs):
        pass


class NegativeScenario(Scenario):
    def __init__(self, df: pd.DataFrame, scenario_type: str, negative: pd.DataFrame):
        Scenario.__init__(self, df, scenario_type)

        if negative is None:
            raise TenetError(f"'negative' dataset is None")

        if negative.empty:
            raise TenetError(f"'negative' dataset is empty")

        self.negative = negative
        self.negative['used'] = [0] * len(negative)
        self.keywords = None

    def augment(self, n: int):
        if n is not None:
            if n < 1:
                raise TenetError(f"'n' should be equal or greater than 1")

            for i in range(n):
                print(f"Augmenting dataset ({i+1}/3)...")
                self.generate(augment=True)

    def get_files_pairs(self, project: str) -> Tuple[Union[pd.DataFrame, pd.Series], Union[pd.DataFrame, pd.Series]]:
        """
            Gets samples of vulnerable and non vulnerable files for a given project
        """
        # get vulnerable files for each project
        vulnerable_files = self.df[self.df["project"] == project]

        # get non-vulnerable files for each project from negative dataset
        non_vulnerable_files = self.negative[(self.negative["project"] == project) & (self.negative['used'] == 0)]

        if len(vulnerable_files) > len(non_vulnerable_files):
            print(project, len(vulnerable_files), len(non_vulnerable_files))

        return vulnerable_files, non_vulnerable_files

    def filter_by_file_path(self, target_df: Union[pd.DataFrame, pd.Series], file_path: str) \
            -> Union[pd.DataFrame, pd.Series]:
        if self.keywords is not None:
            heuristics = (target_df['file_path'] != file_path) & (~target_df['message'].str.match(self.keywords))
        else:
            heuristics = target_df['file_path'] != file_path

        return target_df[heuristics]

    def randomize_negative(self, target: Union[pd.DataFrame, pd.Series], source: pd.Series) \
            -> Union[pd.DataFrame, pd.Series]:
        randomized = target.sample(n=1)
        self.negative.at[randomized.index[0], 'used'] = 1
        source.at[randomized.index[0], 'used'] = 1

        return randomized

    def generate_for_vulnerable_file(self, idx: Hashable, file_path: str, non_vulnerable_samples: pd.Series,
                                     project_url: str, project: str, augment: bool = False):
        """
            Generates non vulnerable samples given vulnerable file
        """
        to_pick_from = self.filter_by_file_path(non_vulnerable_samples, file_path)

        if len(to_pick_from) > 0:
            for _, row in self.randomize_negative(source=non_vulnerable_samples, target=to_pick_from).iterrows():
                negative_sample_info = row.copy().rename({'file_path': 'non_vuln_file_path',
                                                          'sha': 'non_vuln_commit_hash',
                                                          'non_vuln_raw_url': 'raw_url_non_vuln'}).to_dict()

                if augment:
                    augment_data = {'dataset': 'non-vuln', 'project_url': project_url, 'project': project}
                    negative_sample_info.update(augment_data)
                    self.df = pd.concat([self.df, df_init(negative_sample_info)], ignore_index=True)
                else:
                    for k, v in negative_sample_info.items():
                        self.df.at[idx, k] = v
        else:
            print(f"No more data available for {project}...")

    def generate(self, augment: bool = False):
        # iterate over each project

        for project in self.df['project'].unique():
            vulnerable_samples, non_vulnerable_samples = self.get_files_pairs(project)

            for i, file in vulnerable_samples.iterrows():
                self.generate_for_vulnerable_file(file_path=file['file_path'], project_url=file['project_url'], idx=i,
                                                  augment=augment, project=file['project'],
                                                  non_vulnerable_samples=non_vulnerable_samples)

    def drop_duplicates(self, **kwargs):
        pass

    def remove_changes(self, **kwargs):
        pass


class Random(NegativeScenario):
    def __init__(self, df: pd.DataFrame, negative: pd.DataFrame):
        NegativeScenario.__init__(self, df, "random", negative)
        self.negative = self.negative.drop_duplicates()
        self.negative["used"] = 0


class Controlled(NegativeScenario):
    def __init__(self, df: pd.DataFrame, negative: pd.DataFrame, keywords: list):
        NegativeScenario.__init__(self, df, "controlled", negative)
        self.keywords = '|'.join(keywords)
        self.negative = self.negative[pd.notnull(self.negative['message'])]
        self.negative = self.negative.drop_duplicates()
        self.negative["used"] = 0


class Fix(Scenario):
    def __init__(self, df: pd.DataFrame, extensions: list = None):
        Scenario.__init__(self, df, "fix")
        self.extensions = extensions

    def parse_file_changes_in_row(self, row: pd.Series, file: str, file_changes: dict) -> Union[pd.DataFrame, None]:
        # if extensions provided
        if self.extensions:
            extension = file.split('.')[-1].lower()

            # pick only the files with changes with given extension
            if extension not in self.extensions:
                return None

        if 'tests/' in file or 'test/' in file:
            return None
        elif 'test' in file.split('/')[-1].lower():
            return None

        changes_info = parse_file_changes(file_changes, row['commit_sha'], row['before_first_fix_commit'])
        row_info = parse_file_in_row(row, file=file)
        row_info.update(changes_info)

        return df_init(row_info)

    def generate(self):
        # iterate over projects
        frames = []

        for project in self.df['project'].unique():
            # get over bigvul commits
            for _, row in self.df[self.df['project'] == project].iterrows():
                # skip commits with no files changed
                if not pd.notna(row["file_path"]):
                    continue

                # todo: optimize loops
                new_row = self.parse_file_changes_in_row(row, row['file_path'], file_changes=row["file_changes"])

                if new_row is not None:
                    frames.append(new_row)

        self.df = pd.concat(frames, ignore_index=True)

    def remove_changes(self):
        # remove renamed and removed files
        renamed_files = (self.df['status'] != 'renamed')
        removed_files = (self.df['status'] != 'removed')
        self.df = self.df[(renamed_files) & (removed_files)]

    def drop_duplicates(self):
        # drop duplicates
        keys = list(self.df.keys())
        keys.remove("dataset")
        keys.remove("score")
        keys.remove("cwe_id")
        keys.remove("project")
        self.df = self.df.drop_duplicates(subset=keys, keep="last")

    def augment(self, n: int, **kwargs):
        pass
