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
        diff_dir = Path(self.path, 'diffs')
        self.set('files_path', files_dir)
        self.set('dataset', self.output)

        # Save diff texts as files in the directory
        check_or_create_dir(files_dir)
        check_or_create_dir(diff_dir)

        runner_data = Runner()
        threads = self.app.get_config('local_threads')

        # Filter focus where commits==1
        dataset = dataset[dataset["commits"] == 1]
        # Select columns: repo, sha_list, parents
        # Ensure no duplicates
        dataset = dataset[["repo", "sha_list", "parents", "cwe_id"]].drop_duplicates().reset_index(drop=True)
        tasks = []
        total = len(dataset)

        for i, proj in dataset.iterrows():
            self.app.log.info(f"Creating task for repo {proj['repo']}. {i}/{total}")
            task = Task()
            task['id'] = i
            task['repo'] = proj["repo"]
            task['fix_sha'] = proj['sha_list']
            task['parent_sha'] = proj['parents']
            task['diff_dir'] = diff_dir
            task['label'] = proj['cwe_id'].replace('\r\n', '|')
            tasks.append(task)

        worker = ThreadPoolWorker(runner_data, tasks=tasks, threads=threads, logger=self.app.log,
                                  func=self.parse_diffs_task)
        worker.start()

        while worker.is_alive():
            time.sleep(1)

        diffs = {i + j: diff.to_dict() for i, res in enumerate(runner_data.finished) for j, diff in
                 enumerate(res['result']) if 'result' in res and res['result']}

        if diffs:
            return pd.DataFrame.from_dict(diffs, orient='index')

        return None

    def parse_diffs_task(self, task: Task):
        return self.parse_diffs(repo=task['repo'], fix_sha=task['fix_sha'], parent_sha=task['parent_sha'],
                                diff_dir=task['diff_dir'], label=task['label'])

    def parse_diffs(self, repo: str, diff_dir: Path, fix_sha: str, parent_sha: str, label: str):
        git_api = Github(self.token)
        repo = git_api.get_repo(repo)
        repo_path = repo.full_name.replace("/", "_")

        # Ignore unavailable commits
        try:
            fix_hash = ast.literal_eval(fix_sha)[0].split("#")[0].split("?")[0]
            fix_commit = repo.get_commit(sha=fix_hash)
        except (ValueError, GithubException):
            self.app.log.error(f"Commit {fix_sha} for repo {repo.name} unavailable: ")
        else:
            # Get diff only from the first parent if the fix is a merge commit
            parent_hash = ast.literal_eval(parent_sha)[0][0]

            try:
                parent_commit = repo.get_commit(sha=parent_hash)
            except ValueError:
                self.app.log.error(f"Parent commit {parent_hash} for repo {repo.name} unavailable")
            else:
                diff_file = diff_dir / f"{repo_path}_{parent_hash}_{fix_hash}.txt"
                parent_proj = f"{repo_path}_{parent_hash}"
                fix_proj = f"{repo_path}_{fix_hash}"

                # Get the diff string using GitHub API
                self.app.log.info(f"Requesting {fix_commit.raw_data['html_url']}.diff")
                if diff_file.exists():
                    self.app.log.info(f"Reading {diff_file}")
                    with diff_file.open(mode='r') as df:
                        diff_text = df.read()
                else:
                    diff_text = requests.get(f"{fix_commit.raw_data['html_url']}.diff").text

                    if diff_text:
                        # Save the diff string as a file

                        with open(diff_file, "w") as tmp_file:
                            self.app.log.info(f"Writing diff to {parent_proj}_{fix_proj}_diff")
                            safe_write(tmp_file, diff_text, f"{parent_proj}_{fix_proj}_diff")

                if diff_text:
                    # Parse the diff string and collect diff information
                    diff_parser = DiffParser(diff_text=diff_text, a_proj=parent_proj, b_proj=fix_proj,
                                             parent=parent_commit, fix=fix_commit, logger=self.app.log)
                    self.app.log.info(f"Parsing {fix_commit.raw_data['html_url']}.diff")
                    diff_parser.parse(extensions=self.app.get_config('proj_ext'))

                    return diff_parser(files_dir=diff_dir.parent / 'files', label=label)


def load(app):
    app.handler.register(GithubCollector)
