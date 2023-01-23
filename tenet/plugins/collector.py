import pandas as pd

from pathlib import Path
from typing import Union

from github import BadCredentialsException
from tqdm import tqdm

from tenet.core.exc import TenetError
from tenet.core.plotter import Plotter
from tenet.data.diff import Entry
from tenet.handlers.plugin import PluginHandler
from tenet.core.diff_labeller.misc import check_or_create_dir


class Collector(PluginHandler):
    """
        Collector plugin
    """

    class Meta:
        label = "collector"

    def __init__(self, **kw):
        super().__init__(**kw)
        self.diff_dir: Path = None

    def set_sources(self):
        self.set('files_path', Path(self.path, 'files'))

    def get_sinks(self):
        pass

    def run(self, dataset: pd.DataFrame, max_size: int = None, **kwargs) -> Union[pd.DataFrame, None]:
        """
            runs the plugin
        """
        cols = ["owner", "project", "fixed_commit_hash", "vuln_commit_hash", "cwe_id"]

        if not all([col in dataset.columns for col in cols]):
            raise TenetError(f"Missing columns. (Dataset must contain {cols})")

        check_or_create_dir(self.sources['files_path'])

        # Save diff texts as files in the directory
        self.diff_dir = Path(self.path, 'diffs')
        check_or_create_dir(self.diff_dir)

        # Select columns: repo, sha_list, parents and ensure no duplicates
        dataset = dataset.drop_duplicates().reset_index(drop=True)

        self.app.log.info(f"Creating {len(dataset)} tasks.")
        for i, proj in tqdm(dataset.iterrows()):
            self.multi_task_handler.add(project=proj['project'], fix_sha=proj['fixed_commit_hash'], owner=proj['owner'],
                                        vuln_sha=proj['vuln_commit_hash'], label=proj['cwe_id'])

        self.multi_task_handler(func=self.parse_diffs)
        diffs = self.multi_task_handler.results(expand=True)

        if diffs:
            df = pd.DataFrame.from_dict(diffs)

            if max_size:
                self.app.log.info(f"Size before filter: {len(df)}")
                df = df[df.apply(lambda x: (x['a_file_size'] + x['b_file_size']) / 2 < max_size, axis=1)]
                self.app.log.info(f"Size after filter: {len(df)}")

            return df

        return None

    def plot(self, dataset: pd.DataFrame, **kwargs):
        if 'a_file_size' in dataset.columns and 'b_file_size' in dataset.columns:
            Plotter(self.path).histogram_columns(dataset, columns=['a_file_size', 'b_file_size'], y_label='Occurrences',
                                                 x_label='File size (bytes)', labels=['Fix', 'Parent'], bins=20)

    def parse_diffs(self, owner: str, project: str, fix_sha: str, vuln_sha: str, label: str):
        try:
            repo = self.github_handler.get_repo(owner=owner, project=project)
        except BadCredentialsException as bde:
            self.app.log.error(f"could not get repo {project}: {bde}")
            return None

        fix_commit = self.github_handler.get_commit(repo, commit_sha=fix_sha)
        vuln_commit = self.github_handler.get_commit(repo, commit_sha=vuln_sha)
        diff_file = self.diff_dir / f"{owner}_{project}_{vuln_sha}_{fix_sha}.txt"

        if fix_commit and vuln_commit:
            # Get the diff string using GitHub API
            diff_text = self.github_handler.get_diff(commit=fix_commit, output_path=diff_file)

            # Parse the diff string and collect diff information
            self.app.log.info(f"Parsing {diff_file}")
            diff_blocks = self.github_handler.get_blocks_from_diff(diff_text=diff_text)

            entries = []
            for diff_block in diff_blocks:
                entry = Entry(owner=owner, project=project, a_version=vuln_sha, b_version=fix_sha, label=label,
                              diff_block=diff_block)
                # Get the contents of the two files using GitHub API
                a_output_file = self.sources['files_path'] / entry.full_a_path
                _, entry.a_file_size = self.github_handler.get_file_from_commit(commit=vuln_commit,
                                                                                repo_file_path=diff_block.a_path,
                                                                                output_path=a_output_file)
                b_output_file = self.sources['files_path'] / entry.full_b_path
                _, entry.b_file_size = self.github_handler.get_file_from_commit(commit=fix_commit,
                                                                                repo_file_path=diff_block.b_path,
                                                                                output_path=b_output_file)

                entries.append(entry.to_dict())

            return entries
        return None


def load(app):
    app.handler.register(Collector)
