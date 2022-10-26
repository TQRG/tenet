import pandas as pd

from pathlib import Path
from typing import Union

from github import BadCredentialsException
from tqdm import tqdm

from securityaware.core.exc import SecurityAwareError
from securityaware.core.plotter import Plotter
from securityaware.data.diff import Entry
from securityaware.handlers.plugin import PluginHandler
from securityaware.core.diff_labeller.misc import check_or_create_dir


class Collector(PluginHandler):
    """
        Collector plugin
    """

    class Meta:
        label = "collector"

    def __init__(self, **kw):
        super().__init__(**kw)
        self.files_dir: Path = None
        self.diff_dir: Path = None

    def set_dirs(self):
        # TODO: refactor this
        self.files_dir = Path(self.path, 'files')
        check_or_create_dir(self.files_dir)
        self.set('files_path', self.files_dir)

        # Save diff texts as files in the directory
        self.diff_dir = Path(self.path, 'diffs')
        check_or_create_dir(self.diff_dir)

    def run(self, dataset: pd.DataFrame, token: str = None, max_size: int = None) -> Union[pd.DataFrame, None]:
        """
            runs the plugin
        """
        cols = ["repo", "fix_sha", "parent_sha", "cwe", "commits"]

        if not all([col in dataset.columns for col in cols]):
            raise SecurityAwareError(f"Missing columns. (Dataset must contain {cols})")

        self.github_handler.token = token
        self.set_dirs()
        self.set('dataset', self.output)

        # Filter focus where commits==1
        dataset = dataset[dataset["commits"] == 1]

        # Select columns: repo, sha_list, parents and ensure no duplicates
        dataset = dataset[cols].drop_duplicates().reset_index(drop=True)

        self.app.log.info(f"Creating {len(dataset)} tasks.")
        for i, proj in tqdm(dataset.iterrows()):
            self.multi_task_handler.add(repo_path=proj.repo, fix_sha=proj.fix_sha, parent_sha=proj.parent_sha,
                                        label=proj['cwe'])

        self.multi_task_handler(func=self.parse_diffs)
        diffs = self.multi_task_handler.results(expand=True)

        if diffs:
            df = pd.DataFrame.from_dict(diffs)

            if 'a_file_size' in df.columns and 'b_file_size' in df.columns:
                Plotter(self.path).histogram_columns(df, columns=['a_file_size', 'b_file_size'], y_label='Occurrences',
                                                     x_label='File size (bytes)',  labels=['Fix', 'Parent'], bins=20)
            if max_size:
                self.app.log.info(f"Size before filter: {len(df)}")
                df = df[df.apply(lambda x: (x['a_file_size'] + x['b_file_size']) / 2 < max_size, axis=1)]
                self.app.log.info(f"Size after filter: {len(df)}")

            return df

        return None

    def parse_diffs(self, repo_path: str, fix_sha: str, parent_sha: str, label: str):
        try:
            repo = self.github_handler.git_api.get_repo(repo_path)
        except BadCredentialsException as bde:
            self.app.log.error(f"could not get repo {repo_path}: {bde}")
            return None

        fix_hash, parent_hash = self.github_handler.parse_commit_sha([fix_sha, parent_sha])
        fix_commit = self.github_handler.get_commit(repo, commit_sha=fix_hash)
        parent_commit = self.github_handler.get_commit(repo, commit_sha=parent_hash)
        owner, project = repo_path.split('/')
        diff_file = self.diff_dir / f"{owner}_{project}_{parent_hash}_{fix_hash}.txt"

        if fix_commit and parent_commit:
            # Get the diff string using GitHub API
            diff_text = self.github_handler.get_diff(commit=fix_commit, output_path=diff_file)

            # Parse the diff string and collect diff information
            self.app.log.info(f"Parsing {diff_file}")
            diff_blocks = self.github_handler.get_blocks_from_diff(diff_text=diff_text)

            entries = []
            for diff_block in diff_blocks:
                entry = Entry(owner=owner, project=project, a_version=parent_hash, b_version=fix_hash, label=label,
                              diff_block=diff_block)
                # Get the contents of the two files using GitHub API
                a_output_file = self.files_dir / entry.full_a_path
                _, entry.a_file_size = self.github_handler.get_file_from_commit(commit=parent_commit,
                                                                                repo_file_path=diff_block.a_path,
                                                                                output_path=a_output_file)
                b_output_file = self.files_dir / entry.full_b_path
                _, entry.b_file_size = self.github_handler.get_file_from_commit(commit=fix_commit,
                                                                                repo_file_path=diff_block.b_path,
                                                                                output_path=b_output_file)

                entries.append(entry.to_dict())

            return entries
        return None


def load(app):
    app.handler.register(Collector)
