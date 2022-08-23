import time

import pandas as pd
import requests
import ast

from pathlib import Path
from typing import Union

from securityaware.handlers.plugin import PluginHandler
from securityaware.core.diff_labeller.parser import DiffParser
from securityaware.core.diff_labeller.misc import check_or_create_dir, safe_write
from securityaware.data.runner import Task, Runner
from securityaware.handlers.runner import ThreadPoolWorker
from github import Github, GithubException


class GithubCollector(PluginHandler):
    """
        Github collector plugin
    """

    class Meta:
        label = "github_collector"

    def __init__(self, **kw):
        super().__init__(**kw)
        self.token = None

    def run(self, dataset: pd.DataFrame, token: str = None) -> Union[pd.DataFrame, None]:
        """
            runs the plugin
        """
        self.token = token
        files_dir = Path(self.path, 'files')
        self.set('files_path', files_dir)
        self.set('dataset', self.output)

        # Save diff texts as files in the directory
        check_or_create_dir(files_dir)

        runner_data = Runner()
        threads = self.app.get_config('local_threads')

        # Select columns: repo, sha_list, parents
        # Ensure no duplicates
        tasks = []
        total = len(dataset)
        # Drop unnamed columns
        dataset.drop(columns=[c for c in dataset.columns if 'Unnamed:' in c], inplace=True)

        for i, row in dataset.iterrows():
            self.app.log.info(f"Creating task for repo {row['project_name'] + row['file_path']}. {i}/{total}")
            task = Task()
            task['id'] = i
            task['row'] = row
            task['files_dir'] = files_dir
            tasks.append(task)

        worker = ThreadPoolWorker(runner_data, tasks=tasks, threads=threads, logger=self.app.log,
                                  func=self.parse_diffs_task)
        worker.start()

        while worker.is_alive():
            time.sleep(1)

        rows = {i + j: row.to_dict() for i, res in enumerate(runner_data.finished) for j, row in
                 enumerate(res['result']) if 'result' in res and res['result']}

        if rows:
            return pd.DataFrame.from_dict(rows, orient='index')

        return None

    def parse_diffs_task(self, task: Task):
        return self.parse_diffs(row=task['row'], files_dir=task['files_dir'])

    def parse_diffs(self, row: pd.Series, files_dir: Path):
        git_api = Github(self.token)
        repo = row.project.replace('https://github.com/', '')
        repo_path = f"https://github.com/{repo_path}"
        repo = git_api.get_repo(repo)
        #repo_path = repo.full_name.replace("/", "_")

        try:
            parent_commit = repo.get_commit(sha=row.commit_hash)
        except ValueError:
            self.app.log.error(f"Parent commit {row.commit_hash} for repo {repo.name} unavailable")
        else:
            file_path = files_dir / repo_path / row.commit_hash / row.file_path
            parent_proj = f"{repo_path}_{row.commit_hash}"

            # Get the diff string using GitHub API
            self.app.log.info(f"Requesting {parent_commit.raw_data['html_url']}")

            if not file_path.exists():
                commit_url = f"{repo_path}/raw/{row.commit_hash.replace('#diff-', '')}/{row.file_path}"
                fstr = requests.get(commit_url).text
                diff_text = requests.get(f"{parent_commit.raw_data['html_url']}")

                if diff_text:
                    # Save the diff string as a file

                    with open(diff_file, "w") as tmp_file:
                        self.app.log.info(f"Writing diff to {parent_proj}_{fix_proj}_diff")
                        safe_write(tmp_file, diff_text, f"{parent_proj}_{fix_proj}_diff")


def load(app):
    app.handler.register(GithubCollector)
