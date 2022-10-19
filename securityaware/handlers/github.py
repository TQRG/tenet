import ast
from pathlib import Path

import requests
from cement import Handler
from github import Github
from typing import Union, List, Tuple

from github.Commit import Commit
from github.GithubException import GithubException
from github.Repository import Repository

from securityaware.core.diff_labeller.misc import safe_write
from securityaware.core.interfaces import HandlersInterface
from securityaware.data.diff import DiffBlock


class GithubHandler(HandlersInterface, Handler):
    """
        Github handler abstraction
    """
    class Meta:
        label = 'github'

    def __init__(self, **kw):
        super().__init__(**kw)
        self._git_api: Github = None
        self._token: str = None

    @property
    def git_api(self):
        return self._git_api

    @git_api.setter
    def git_api(self, value: Github):
        if self._git_api is None:
            self._git_api = value

    @property
    def token(self):
        return self._token

    @token.setter
    def token(self, value: str):
        self._token = value

        if self._git_api is None:
            self._git_api = Github(self._token)

    @staticmethod
    def parse_commit_sha(commit_sha: Union[list, str]) -> list:
        commit_hashes = []

        if isinstance(commit_sha, str):
            commit_sha = [commit_sha]

        for cs in commit_sha:
            # Get diff only from the first parent if the fix is a merge commit
            commit_hash = ast.literal_eval(cs)[0]

            if isinstance(commit_hash, list):
                commit_hash = commit_hash[0]

            commit_hashes.append(commit_hash.split("#")[0].split("?")[0])

        return commit_hashes

    def get_commit(self, repo: Repository, commit_sha: str) -> Union[Commit, None]:
        # Ignore unavailable commits
        try:
            return repo.get_commit(sha=commit_sha)
        except (ValueError, GithubException):
            self.app.log.error(f"Commit {commit_sha} for repo {repo.name} unavailable: ")
            return None

    def get_diff(self, commit: Commit, output_path: Path) -> str:

        if output_path.exists():
            self.app.log.info(f"{output_path} exists, reading...")

            with output_path.open(mode='r') as df:
                return df.read()

        self.app.log.info(f"Requesting {commit.raw_data['html_url']}.diff")
        diff_text = requests.get(f"{commit.raw_data['html_url']}.diff").text

        if output_path.exists() and output_path.stat().st_size != 0 and diff_text:
            # Save the diff string as a file

            with output_path.open(mode="w") as tmp_file:
                self.app.log.info(f"Writing diff to {output_path}")
                safe_write(tmp_file, diff_text, str(output_path.parents))

        return diff_text

    def get_file_from_commit(self, repo_file_path: str, commit: Commit, output_path: Path) -> Tuple[str, int]:
        if output_path.exists() and output_path.stat().st_size != 0:
            self.app.log.info(f"{output_path} exists, reading...")

            with output_path.open(mode='r') as f:
                f_str = f.read()
        else:
            url = f"{commit.html_url}/{repo_file_path}".replace('commit', 'raw')
            self.app.log.info(f"Requesting {url}")
            f_str = requests.get(url).text

            self.app.log.info(f"Writing {output_path}")
            output_path.parent.mkdir(exist_ok=True, parents=True)

            with output_path.open(mode="w") as f:
                f.write(f_str)

        return f_str, output_path.stat().st_size

    def get_blocks_from_diff(self, diff_text: str, extensions: list = None) -> List[DiffBlock]:
        """
        Parses the input diff string and returns a list of result entries.

        :param diff_text: The input git diff string in unified diff format.
        :param extensions: file to include from the diff based on extensions.
        :return: A list of entries resulted from the input diff to be appended to the output csv file.
        """

        if not diff_text:
            return []

        # Look for a_path
        lines = diff_text.splitlines()
        diff_path_bound = [line_id for line_id in range(len(lines)) if lines[line_id].startswith("--- ")]
        num_paths = len(diff_path_bound)
        diff_path_bound.append(len(lines))
        blocks = []

        extensions = extensions if extensions else self.app.get_config('proj_ext')

        for path_id in range(num_paths):
            # Only look for a_paths with the interested file extensions
            for ext in extensions:
                if lines[diff_path_bound[path_id]].endswith(ext):
                    # Only consider file modification, ignore file additions for now
                    block_start = diff_path_bound[path_id]
                    if not lines[block_start + 1].startswith("+++ "):
                        self.app.log.warning(f"Skipping block {block_start + 1} missing +++")
                        continue

                    # Ignore file deletions for now
                    if not lines[block_start + 1].endswith(" /dev/null"):
                        # Format of the "---" and "+++" lines:
                        # --- a/<a_path>
                        # +++ b/<b_path>
                        diff_block = DiffBlock(start=block_start, a_path=lines[block_start][len("--- a/"):],
                                               b_path=lines[block_start + 1][len("+++ b/"):])

                        # Do not include diff in the test files
                        if "test" in diff_block.a_path or "test" in diff_block.b_path:
                            continue

                        blocks.append(diff_block)

        return blocks
