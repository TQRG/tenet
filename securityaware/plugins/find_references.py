import ast
import re
import pandas as pd

from github import Github, GithubException
from github.GithubException import UnknownObjectException
from typing import Union

from securityaware.handlers.plugin import PluginHandler


class FindReferencesHandler(PluginHandler):
    """
        Separate plugin
    """

    def __init__(self, **kw):
        super().__init__(**kw)
        self.token = None

    class Meta:
        label = "find_references"

    def run(self, dataset: pd.DataFrame, token: str = None, column: str = None, **kwargs) -> Union[pd.DataFrame, None]:
        """
            runs the plugin
        """
        self.token = token
        self.set('dataset', self.output)

        if not column:
            self.app.log.warning("Must specify 'column' name")
            return None

        if column not in dataset.columns:
            self.app.log.warning(f"Column {column} not found in DataFrame")
            return None

        filtered_dataset = []
        git_api = Github(self.token)
        git_repo_regex = '(?P<host>(git@|https://)([\w\.@]+)(/|:))(?P<owner>[\w,\-,\_]+)/(?P<repo>[\w,\-,\_]+)(.git){0,1}((/){0,1})'
        fix_hashes = []
        parent_hashes = []
        commits = []
        repos = []
        projects = []

        for i, el in dataset.iterrows():
            refs = ast.literal_eval(el[column])
            commit_fix_hashes = []
            commit_parent_hashes = []
            commit_repos = ''
            commit_project = ''

            for ref in refs:
                match = re.match(pattern='https://github\.com(?:/[^/]+)*/commit/([0-9a-f]{40})', string=ref)

                if match:
                    commit_fix_sha = match.group(1)
                    match = re.match(pattern=git_repo_regex, string=ref)
                    repo_name = match['repo']
                    repo_owner = match['owner']
                    repo_path = f"{repo_owner}/{repo_name}"
                    self.app.log.info(f"Parsing {commit_fix_sha}")

                    try:
                        repo = git_api.get_repo(repo_path)
                        commit_fix_sha = repo.get_commit(sha=commit_fix_sha)
                    except (UnknownObjectException, GithubException):
                        self.app.log.error(f"Commit {commit_fix_sha} for repo {repo_path} not found")
                        continue

                    commit_project = repo_name
                    commit_repos = repo_path
                    commit_fix_hashes.append(commit_fix_sha.sha)
                    commit_parent_hashes.append(commit_fix_sha.parents[0].sha)

            if len(commit_fix_hashes) > 0:
                filtered_dataset.append(el)
                repos.append(commit_repos)
                projects.append(commit_project)
                fix_hashes.append(str(commit_fix_hashes))
                parent_hashes.append(str(commit_parent_hashes))
                commits.append(len(commit_fix_hashes))

        new_dataset = pd.DataFrame(filtered_dataset, columns=list(dataset.columns.values))
        new_dataset.rename(columns={'id': 'cve_id'})
        new_dataset.drop(columns=[column])
        new_dataset['project'] = projects
        new_dataset['repo'] = repos
        new_dataset['fix_sha'] = fix_hashes
        new_dataset['parent_sha'] = parent_hashes
        new_dataset['commits'] = commits

        return new_dataset


def load(app):
    app.handler.register(FindReferencesHandler)
