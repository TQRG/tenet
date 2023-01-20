import ast
import random
import threading

import numpy as np
import pandas as pd
import requests
import sys

from collections import deque
from pathlib import Path

from cement import Handler
from github import Github
from typing import Union, List, Tuple, Dict

from github.Commit import Commit
from github.GithubException import GithubException, RateLimitExceededException, UnknownObjectException
from github.PaginatedList import PaginatedList
from github.Repository import Repository

from tenet.core.diff_labeller.misc import safe_write
from tenet.core.exc import TenetError
from tenet.core.interfaces import HandlersInterface
from tenet.data.dataset import CommitMetadata, ChainMetadata
from tenet.data.diff import DiffBlock


class LocalGitFile:
    def __init__(self, url: str, path: Path, short: Path, content: str = None, tag: str = None):
        self.url = url
        self.path = path
        self.short = short
        self.content = content
        self.tag = tag

    def download(self):
        try:
            print(f'Requesting {self.tag}: {self.url}')
            request = requests.get(self.url)

            if request.status_code == 200:
                self.content = request.text
            else:
                print(f'Request code {request.status_code}')

        except ConnectionError:
            print("ConnectionError")

    def write(self) -> int:
        if not self.content:
            print(f"No file content for {self.path}")
            return -1
            # raise ValueError(f"No file content for {self.path}")

        self.path.parent.mkdir(exist_ok=True, parents=True)

        with self.path.open(mode='w') as f:
            print(f'Writing to  {self.path}')
            return f.write(self.content)

    def read(self):
        if not self.content:
            #print(f'Reading from {self.path}')

            if not self.path.exists() or self.path.stat().st_size == 0:
                if self.tag:
                    print(self.tag)
                self.download()
                self.write()
            else:
                with self.path.open(mode='r') as f:
                    self.content = f.read()

        return self.content


class PaginatedListRandomPicker:
    """
        Wrapper of PaginatedList that randomly picks pages
    """
    def __init__(self, paginated_list: PaginatedList, n_pages: int):
        self._paginated_list = paginated_list
        self._n_pages = n_pages
        self._visited = []

    def __call__(self, *args, **kwargs) -> Tuple[list, int]:
        if len(self._visited) >= self._n_pages:
            raise TenetError(f"PaginatedListRandomPicker visited all pages")

        rand_page = random.randint(0, self._n_pages - 1)

        while rand_page in self._visited:
            rand_page = random.randint(0, self._n_pages - 1)

        commits = self._paginated_list.get_page(rand_page)
        self._visited.append(rand_page)

        return commits, rand_page


class RandomCommitFilesLookup:
    def __init__(self, excluded_files: set, excluded_commits: set, target_extensions: list = None):
        self._excluded_files = excluded_files if excluded_files else {}
        self._excluded_commits = excluded_commits if excluded_files else {}
        self._target_extensions = target_extensions
        self._visited = []

    def __len__(self):
        return len(self._visited)

    def coverage(self):
        coverage = set(self._visited)
        coverage.update(self._excluded_commits)

        return len(coverage)

    @property
    def excluded_files(self):
        return self._excluded_files

    @property
    def excluded_commits(self):
        return self._excluded_commits

    def check_file(self, file) -> bool:
        check = file.filename not in self._excluded_files and 'test' not in file.filename and file.status == 'modified'

        if self._target_extensions:
            return check and file.filename.split('.')[-1].lower() in self._target_extensions

        return check

    def get_commit_files(self, commit: Commit) -> list:
        return [file for file in commit.files if self.check_file(file)]

    def __call__(self, commits: List[Commit], files_limit: int = None) -> Dict[Commit, list]:
        # filter excluded commits
        commits = [commit for commit in commits if commit.sha not in self._excluded_commits]

        if len(commits) == 0:
            raise TenetError(f"No available commits after filtering excluded commits ")

        commit_files_pair = {}
        total_files = 0
        picked = []

        while len(picked) != len(commits):
            commit_idx = random.randint(0, len(commits) - 1)

            while commit_idx in picked:
                commit_idx = random.randint(0, len(commits) - 1)

            picked.append(commit_idx)
            random_commit = commits[commit_idx]
            commit_files_pair[random_commit] = self.get_commit_files(random_commit)
            total_files += len(commit_files_pair[random_commit])
            self._visited.append(random_commit.sha)

            if total_files > files_limit:
                break

        return commit_files_pair


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
        # Size of the paginated list returned by get_commits, may change
        self.paginated_list_size = 30
        self._sw_types = None

    def has_rate_available(self):
        return self.git_api.get_rate_limit().core.remaining > 0

    @property
    def sw_types(self):
        if self._sw_types is None:
            self._sw_types = {}
            # normalize owner and project names
            for owner, projects in self.app.config.get_section_dict('sw_type').items():
                normalized_owner = owner.strip().lower()
                self._sw_types[normalized_owner] = {}

                for project, sw_type in projects.items():
                    normalized_project = project.strip().lower()
                    self._sw_types[normalized_owner][normalized_project] = sw_type

        return self._sw_types

    def get_sw_type(self, owner: str, project: str) -> Union[str, None]:
        if owner is None or pd.isna(owner):
            return 'unk'

        if project is None or pd.isna(project):
            return 'unk'

        normalized_owner = owner.strip().lower()
        normalized_project = project.strip().lower()

        if normalized_owner in self.sw_types and normalized_project in self.sw_types[normalized_owner]:
            return self.sw_types[normalized_owner][normalized_project]

        return 'unk'

    @property
    def git_api(self):
        with self.lock:
            if not self._tokens:
                tokens = self.app.pargs.tokens.split(',')
                self._tokens = deque(tokens, maxlen=len(tokens))

            if not self._git_api:
                self._git_api = Github(self._tokens[0])
                self._tokens.rotate(-1)

            count = 0
            while not self._git_api.get_rate_limit().core.remaining > 0:
                if count == len(self._tokens):
                    raise TenetError(f"Tokens exhausted")
                self._git_api = Github(self._tokens[0])
                self._tokens.rotate(-1)
                count += 1

            return self._git_api

    @git_api.deleter
    def git_api(self):
        with self.lock:
            self._git_api = None

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
            err_msg = f"Rate limit exhausted: {rle}"
        #except Exception:
        #    err_msg = f"Unexpected error {sys.exc_info()}"

        if raise_err:
            raise TenetError(err_msg)

        self.app.log.error(err_msg)

        return None

    def get_repo(self, owner: str, project: str, raise_err: bool = False) -> Union[Repository, None]:
        repo_path = '{}/{}'.format(owner, project)

        try:
            self.app.log.info(f"Getting repo {repo_path}")
            return self.git_api.get_repo(repo_path)
        except RateLimitExceededException as rle:
            err_msg = f"Rate limit exhausted: {rle}"
        except UnknownObjectException:
            err_msg = f"Repo not found. Skipping {owner}/{project} ..."
        #except Exception:
        #    err_msg = f"Unexpected error {sys.exc_info()}"

        if raise_err:
            raise TenetError(err_msg)

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
        # TODO: we want this too be more flexible, adapt
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

    @staticmethod
    def get_commit_parents(commit: Commit) -> set:
        return set([c.sha for c in commit.commit.parents])

    def get_commit_comments(self, commit: Commit, raise_err: bool = False) -> dict:
        comments, count = {}, 1
        err_msg = None

        try:
            for comment in commit.get_comments():
                comments[f'com_{count}'] = {
                    'author': comment.user.login,
                    'datetime': comment.created_at.strftime("%m/%d/%Y, %H:%M:%S"),
                    'body': comment.body.strip()
                }
                count += 1
        except RateLimitExceededException as rle:
            err_msg = f"Rate limit exhausted: {rle}"
        #except Exception:
        #    err_msg = f"Unexpected error {sys.exc_info()}"

        if err_msg:
            if raise_err:
                raise TenetError(err_msg)

            self.app.log.error(err_msg)

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

    def get_commit_metadata(self, commit: Commit, include_comments: bool = True) -> CommitMetadata:
        comments = self.get_commit_comments(commit) if include_comments else {}
        files = self.get_commit_files(commit)
        stats = {'additions': commit.stats.additions, 'deletions': commit.stats.deletions, 'total': commit.stats.total}

        return CommitMetadata(author=commit.commit.author.name.strip(), message=commit.commit.message.strip(),
                              comments=str(comments) if len(comments) > 0 else np.nan, files=files, stats=stats)

    def get_chain_metadata(self, commit_sha: str, chain_ord: list, chain_datetime: list, raise_err: bool = False,
                           include_comments: bool = True) -> Union[ChainMetadata, None]:

        chain_ord_sha = []
        err_msg = None

        for commit in chain_ord:
            try:
                chain_ord_sha.append(commit.commit.sha)
            except RateLimitExceededException as rle:
                err_msg = f"Rate limit exhausted: {rle}"
                break
            #except Exception:
            #    err_msg = f"Unexpected error {sys.exc_info()}"
            #    break

        # self.app.log.info(f"Chain order: {chain_ord_sha}")

        if err_msg:
            if raise_err:
                raise TenetError(err_msg)

            self.app.log.error(err_msg)

        if len(chain_ord) != len(chain_ord_sha):
            return None

        try:
            chain_idx = chain_ord_sha.index(commit_sha)
        except ValueError as ve:
            self.app.log.error(f"{ve}")
            return None

        parents = list(self.get_commit_parents(chain_ord[-1]))

        commit_datetime = chain_datetime[chain_idx].strftime("%m/%d/%Y, %H:%M:%S")
        commit_metadata = self.get_commit_metadata(chain_ord[chain_idx], include_comments=include_comments)

        return ChainMetadata(commit_metadata=commit_metadata, chain_ord=chain_ord_sha, before_first_fix_commit=parents,
                             chain_ord_pos=chain_idx + 1, last_fix_commit=chain_ord[-1].commit.sha,
                             commit_sha=commit_sha, commit_datetime=commit_datetime)

    def get_project_metadata(self, project: str, chains: list, commits: list, indexes: list, save_path: Path,
                             include_comments: bool = True, drop_patch: bool = False) -> Union[pd.DataFrame, None]:

        # get owner and project
        if not pd.notna(project):
            return None

        owner, project_name = project.split('/')[3::]
        project_path = save_path / project_name

        if project_path.exists() and ChainMetadata.has_commits(project_path, commits):
            # in this case we do not need to query the api as the metadata has been saved previously
            self.app.log.info(f"Found metadata for {owner}/{project_name}")
            repo = None
        else:
            repo = self.get_repo(owner, project_name, raise_err=False)

            if not repo:
                return None

        self.app.log.info(f"Getting the metadata for {len(commits)} commits from project {project_name}...")
        chain_metadata_entries = []

        for idx, chain, commit_sha in zip(indexes, chains, commits):
            commit_metadata_path = project_path / f"{commit_sha}.json"
            commit_metadata_path.parent.mkdir(parents=True, exist_ok=True)
            chain_metadata_dict = ChainMetadata.load(path=commit_metadata_path)

            if not chain_metadata_dict:
                if repo is None:
                    repo = self.get_repo(owner, project_name, raise_err=False)

                    if repo is None:
                        break

                chain_ord, chain_datetime = self.sort_chain(repo, chain)

                if not chain_ord and not chain_datetime:
                    self.app.log.info(f"Skipping {commit_sha} ...")
                    continue

                chain_metadata = self.get_chain_metadata(commit_sha=commit_sha, chain_ord=chain_ord,
                                                         chain_datetime=chain_datetime, include_comments=include_comments)
                if chain_metadata is None:
                    continue

                chain_metadata_dict = chain_metadata.save(path=commit_metadata_path)

            if drop_patch and 'patch' in chain_metadata_dict:
                del chain_metadata_dict['patch']
            chain_metadata_dict.update({'index': idx})
            chain_metadata_entries.append(chain_metadata_dict)

        if chain_metadata_entries:
            return pd.DataFrame(chain_metadata_entries).set_index('index')

        return None

    def get_repo_commits(self, repo: Repository, raise_err: bool = False) \
            -> Union[Tuple[PaginatedList, int, int], Tuple[None, None, None]]:
        """
            Parses the input diff string and returns a list of result entries.

            :param repo: the Repository object
            :param raise_err: flag to raise caught exceptions
            :return: 2-sized-tuple with the PaginatedList commits and the numbers of pages.
        """
        try:
            repo_commits = repo.get_commits()
            total_commits = repo_commits.totalCount
            pages = (total_commits // self.paginated_list_size) + 1
            self.app.log.info(f"{repo.full_name} has {total_commits} commits (~{pages} pages).")
            return repo_commits, pages, total_commits
        except RateLimitExceededException as rle:
            err_msg = f"Rate limit exhausted: {rle}"
        except UnknownObjectException:
            err_msg = f"Failed to get {repo.full_name} repository commits..."
        #except Exception:
        #    err_msg = f"Unexpected error {sys.exc_info()}"

        if raise_err:
            raise TenetError(err_msg)

        self.app.log.error(err_msg)

        return None, None, None

    def repo_random_files_lookup(self, repo: Repository, target_files_count: int = 50, excluded_files: set = None,
                                 excluded_commits: set = None,  target_extensions: list = None,
                                 max_commits: int = None) \
            -> Tuple[pd.DataFrame, str]:
        self.app.log.info(f"Searching {repo.full_name} for {target_files_count} files...")
        paginated_commits, n_pages, total = self.get_repo_commits(repo=repo)

        if paginated_commits is None:
            return pd.DataFrame(), "error"

        paginated_list_random_picker = PaginatedListRandomPicker(paginated_commits, n_pages=n_pages)
        random_commit_file_lookup = RandomCommitFilesLookup(excluded_files=excluded_files,
                                                            target_extensions=target_extensions,
                                                            excluded_commits=excluded_commits)
        collected_files = []
        early_stopping = 0
        empty_commits = []

        while len(collected_files) < target_files_count:
            if max_commits is not None and early_stopping >= max_commits:
                self.app.log.warning(f"Early stopping after looking up {early_stopping} commits")
                return pd.DataFrame(collected_files + empty_commits), "stop"

            try:
                commits, rand_page = paginated_list_random_picker()
            except TenetError as sae:
                self.app.log.warning(sae)

                if random_commit_file_lookup.coverage() == total:
                    return pd.DataFrame(collected_files + empty_commits), "visited"
                else:
                    return pd.DataFrame(collected_files + empty_commits), "stop"

            self.app.log.info(f"Searching into page {rand_page}: {len(collected_files)}/{target_files_count}...")

            try:
                for commit, files in random_commit_file_lookup(commits, files_limit=target_files_count).items():
                    random_commit_file_lookup.excluded_commits.update({commit.sha})
                    early_stopping += 1

                    if len(files) == 0:
                        empty_commits.append({'sha': commit.sha, 'file_path': None, 'non_vuln_raw_url': None,
                                              'message': None})
                        continue

                    self.app.log.info(f"Found {len(files)} files for {commit.sha}...")

                    for file in files:
                        collected_files.append({'sha': commit.sha, 'file_path': file.filename,
                                                'non_vuln_raw_url': file.raw_url, 'message': commit.commit.message})
            except TenetError as sae:
                self.app.log.warning(sae)
                continue

        return pd.DataFrame(collected_files + empty_commits), "success"
