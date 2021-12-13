from pathlib import Path
from typing import List

import requests
from cement.core.log import LogHandler
from github import Commit

from securityaware.data.diff import DiffBlock, Entry


class DiffParser:
    def __init__(self, diff_text: str, a_proj: str, b_proj: str, parent: Commit, fix: Commit, logger: LogHandler):
        """
        Parses the input diff string and returns a list of result entries.
        :param diff_text: The input git diff string in unified diff format.
        :param a_proj: The project id of the parent commit: <owner>_<project_name>_<sha>.
        :param b_proj: The project id of the later commit: <owner>_<project_name>_<sha>.
        :param parent: The Commit object of the parent commit.
        :param fix: The Commit object of the later commit.
        :return: A list of entries resulted from the input diff to be appended to the output csv file.
        """
        self.logger = logger
        self.diff_text = diff_text
        self.a_proj = a_proj
        self.b_proj = b_proj
        self.parent = parent
        self.fix = fix

        # Look for a_path
        self.lines = diff_text.splitlines()
        self.diff_path_bound = [line_id for line_id in range(len(self.lines)) if self.lines[line_id].startswith("--- ")]
        self.num_paths = len(self.diff_path_bound)
        self.diff_path_bound.append(len(self.lines))
        self.diff_blocks: List[DiffBlock] = []

    def add_diff_block(self, path_id: int):
        # Only consider file modification, ignore file additions for now
        block_start = self.diff_path_bound[path_id]
        assert self.lines[block_start + 1].startswith("+++ ")

        # Ignore file deletions for now
        if not self.lines[block_start + 1].endswith(" /dev/null"):
            # Format of the "---" and "+++" lines:
            # --- a/<a_path>
            # +++ b/<b_path>
            diff_block = DiffBlock(start=block_start, a_path=self.lines[block_start][len("--- a/"):],
                                   b_path=self.lines[block_start + 1][len("+++ b/"):])

            # Do not include diff in the test files
            if not ("test" in diff_block.a_path or "test" in diff_block.b_path):
                self.diff_blocks.append(diff_block)

    def parse(self, extensions: List[str]):
        if not self.diff_blocks:
            for path_id in range(self.num_paths):
                # Only look for a_paths with the interested file extensions
                for ext in extensions:
                    if self.lines[self.diff_path_bound[path_id]].endswith(ext):
                        # Add diff block
                        self.add_diff_block(path_id)

    def __call__(self, files_dir: Path, label: str):
        entries = []

        for diff_block in self.diff_blocks:
            # Get the contents of the two files using GitHub API
            a_file = files_dir / self.a_proj / diff_block.a_path
            b_file = files_dir / self.b_proj / diff_block.b_path

            self.download_file_str(file=a_file,
                                   url=f"{self.parent.html_url}/{diff_block.a_path}".replace('commit', 'raw'))

            self.download_file_str(file=b_file,
                                   url=f"{self.fix.html_url}/{diff_block.b_path}".replace('commit', 'raw'))
            entries.append(Entry(a_proj=self.a_proj, b_proj=self.b_proj, diff_block=diff_block, a_file=a_file,
                                 b_file=b_file, label=label))

        return entries

    def download_file_str(self, file: Path, url: str) -> Path:
        self.logger.info(f"Requesting {url}")
        f_str = requests.get(url).text

        self.logger.info(f"Writing {file}")
        file.parent.mkdir(exist_ok=True, parents=True)
        with file.open(mode="w") as f:
            f.write(f_str)

        return file
