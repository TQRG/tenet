import ast
import threading

import numpy as np
import pandas as pd
import requests
import sys

from collections import deque
from pathlib import Path

from cement import Handler
from github import Github
from typing import Union, List, Tuple

from github.Commit import Commit
from github.GithubException import GithubException, RateLimitExceededException, UnknownObjectException
from github.Repository import Repository

from securityaware.core.diff_labeller.misc import safe_write
from securityaware.core.exc import SecurityAwareError
from securityaware.core.interfaces import HandlersInterface
from securityaware.data.dataset import CommitMetadata, ChainMetadata
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
        self._tokens: deque = None
        self.lock = threading.Lock()

    def has_rate_available(self):
        git_api_rate_limit = self.git_api.get_rate_limit()
        used = git_api_rate_limit.core.limit - git_api_rate_limit.core.remaining

        return used > 0

    @property
    def git_api(self):
        with self.lock:
            if self._git_api is None:
                self._git_api = Github(self._tokens[0])
                self._tokens.rotate(-1)

            return self._git_api

    @git_api.deleter
    def git_api(self):
        with self.lock:
            self._git_api = None

    @property
    def tokens(self):
        self._tokens.rotate(-1)
        return self._tokens[0]

    @tokens.setter
    def tokens(self, value: Union[str, list]):
        if isinstance(value, str):
            value = []

        self._tokens = deque(value, maxlen=len(value))

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

    def get_commit(self, repo: Repository, commit_sha: str, raise_err: bool = False) -> Union[Commit, None]:
        # Ignore unavailable commits
        try:
            self.app.log.info(f"Getting commit {commit_sha}")
            return repo.get_commit(sha=commit_sha)
        except (ValueError, GithubException):
            err_msg = f"Commit {commit_sha} for repo {repo.name} unavailable: "
        except RateLimitExceededException as rle:
            del self.git_api

            if not self.has_rate_available():
                # TODO: must update repo before calling
                return repo.get_commit(sha=commit_sha)

            err_msg = f"Rate limit exhausted: {rle}"
        except Exception:
            err_msg = f"Unexpected error {sys.exc_info()}"

        if raise_err:
            raise SecurityAwareError(err_msg)

        self.app.log.error(err_msg)

        return None

    def get_repo(self, owner: str, project: str, raise_err: bool = False) -> Union[Repository, None]:
        repo_path = '{}/{}'.format(owner, project)

        try:
            self.app.log.info(f"Getting repo {repo_path}")
            return self.git_api.get_repo(repo_path)
        except RateLimitExceededException as rle:
            del self.git_api

            if not self.has_rate_available():
                return self.git_api.get_repo(repo_path)

            err_msg = f"Rate limit exhausted: {rle}"

        except UnknownObjectException:
            err_msg = f"Repo not found. Skipping {owner}/{project} ..."
        except Exception:
            err_msg = f"Unexpected error {sys.exc_info()}"

        if raise_err:
            raise SecurityAwareError(err_msg)

        self.app.log.error(err_msg)

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

    def normalize_sha(self, chain: list):
        new_chain = []

        for commit_str in chain:

            # FIXME: WTF? Find why...
            if "//commit" in commit_str:
                commit_str = commit_str.replace("//commit", "/commit")

            if "pull/" in commit_str:
                owner, project, _, _, _, sha = commit_str.split("/")[3::]
            else:
                owner, project, _, sha = commit_str.split("/")[3:7]

            if len(sha) != 40:
                repo = self.get_repo(owner=owner, project=project)

                if not repo:
                    continue

                commit = self.get_commit(repo, sha.strip())

                if not commit:
                    continue

                new_chain.append(f"https://github.com/{owner}/{project}/commit/{commit.commit.sha}")
            else:
                new_chain.append(commit_str)

        return set(new_chain)

    def sort_chain(self, repo: Repository, chain: Union[str, set]) -> Tuple[Union[list, None], Union[list, None]]:
        if isinstance(chain, str):
            chain = list(eval(chain))

        df = pd.DataFrame()

        for commit_str in chain:
            commit = self.get_commit(repo, commit_sha=commit_str.split('/')[-1].strip())

            if not commit:
                return None, None

            author = commit.commit.author
            df = df.append({'commit': commit, 'datetime': author.date}, ignore_index=True)

        df = df.drop_duplicates()
        df = df.sort_values(by='datetime', ascending=True)

        # chain of commit and datetime
        return list(df['commit']), list(df['datetime'])

    def get_repo_from_link(self, repo_link: str, raise_err: bool = False) -> Union[Repository, None]:
        # get owner and project
        if not pd.notna(repo_link):
            return None

        owner, project = repo_link.split('/')[3::]

        return self.get_repo(owner, project, raise_err=raise_err)

    @staticmethod
    def get_commit_parents(commit: Commit) -> set:
        return set([c.sha for c in commit.commit.parents])

    @staticmethod
    def get_commit_comments(commit: Commit) -> dict:
        comments, count = {}, 1

        for comment in commit.get_comments():
            comments[f'com_{count}'] = {
                'author': comment.user.login,
                'datetime': comment.created_at.strftime("%m/%d/%Y, %H:%M:%S"),
                'body': comment.body.strip()
            }
            count += 1

        return comments

    @staticmethod
    def get_commit_files(commit: Commit) -> dict:
        files = {}

        for f in commit.files:
            files[f.filename] = {
                'additions': f.additions,
                'deletions': f.deletions,
                'changes': f.changes,
                'status': f.status,
                'raw_url': f.raw_url,
                'patch': f.patch.strip() if f.patch else None
            }

        return files

    def get_commit_metadata(self, commit: Commit) -> CommitMetadata:
        comments = self.get_commit_comments(commit)
        files = self.get_commit_files(commit)
        stats = {'additions': commit.stats.additions, 'deletions': commit.stats.deletions, 'total': commit.stats.total}

        return CommitMetadata(author=commit.commit.author.name.strip(), message=commit.commit.message.strip(),
                              comments=str(comments) if len(comments) > 0 else np.nan, files=files, stats=stats)

    def get_chain_metadata(self, commit_sha: str, chain_ord: list, chain_datetime: list) -> ChainMetadata:
        chain_ord_sha = [commit.commit.sha for commit in chain_ord]
        self.app.log.info(f"Chain order: {chain_ord_sha}")

        chain_idx = chain_ord_sha.index(commit_sha)
        parents = self.get_commit_parents(chain_ord[-1])

        commit_datetime = chain_datetime[chain_idx].strftime("%m/%d/%Y, %H:%M:%S")
        commit_metadata = self.get_commit_metadata(chain_ord[chain_idx])

        return ChainMetadata(commit_metadata=commit_metadata, chain_ord=chain_ord_sha, before_first_fix_commit=parents,
                             chain_ord_pos=chain_idx + 1, last_fix_commit=chain_ord[-1].commit.sha,
                             commit_sha=commit_sha, commit_datetime=commit_datetime)

    def get_project_metadata(self, project: str, chains: list, commits: list, indexes: list) -> Union[pd.DataFrame, None]:
        repo = self.get_repo_from_link(project)

        if not repo:
            return None

        self.app.log.info(f"Getting the metadata from project {repo.name}...")
        chain_metadata_entries = []

        for idx, chain, commit_sha in zip(indexes, chains, commits):
            chain_ord, chain_datetime = self.sort_chain(repo, chain)

            if not chain_ord and not chain_datetime:
                self.app.log.info(f"Skipping {commit_sha} ...")
                continue

            chain_metadata = self.get_chain_metadata(commit_sha=commit_sha, chain_ord=chain_ord,
                                                     chain_datetime=chain_datetime)
            chain_metadata_dict = chain_metadata.to_dict(flatten=True)
            chain_metadata_dict.update({'index': idx})
            chain_metadata_entries.append(chain_metadata_dict)

        if chain_metadata_entries:
            return pd.DataFrame(chain_metadata_entries).set_index('index')

        return None
