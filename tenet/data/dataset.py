import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union


@dataclass
class CommitMetadata:
    author: str
    message: str
    files: dict
    stats: dict
    comments: dict

    def to_dict(self):
        return {'author': self.author, 'message': self.message, 'files': self.files, 'comments': self.comments,
                'stats': self.stats}


@dataclass
class ChainMetadata:
    commit_metadata: CommitMetadata
    commit_sha: str
    chain_ord: list
    before_first_fix_commit: list
    last_fix_commit: str
    chain_ord_pos: int
    commit_datetime: str

    def to_dict(self, flatten: bool = False):
        chain_metadata = {'before_first_fix_commit': self.before_first_fix_commit, 'chain_ord_pos': self.chain_ord_pos,
                          'last_fix_commit': self.last_fix_commit, 'commit_datetime': self.commit_datetime,
                          'commit_sha': self.commit_sha, 'chain_ord': self.chain_ord}

        commit_metadata = self.commit_metadata.to_dict()

        if flatten:
            chain_metadata.update(commit_metadata)
        else:
            chain_metadata.update({'commit_metadata': commit_metadata})

        return chain_metadata

    def save(self, path: Path) -> dict:
        content = self.to_dict(flatten=True)

        with path.open(mode='w') as j:
            json.dump(content, j)

        return content

    @staticmethod
    def load(path: Path) -> Union[dict, None]:
        if path.exists():
            with path.open(mode='r') as j:
                return json.load(j)

        return None

    @staticmethod
    def has_commits(path: Path, lookup: list) -> bool:
        if path.exists():
            return all([f.stem in lookup for f in path.iterdir()])
        return False


@dataclass
class Dataset:
    rows: List = field(default_factory=lambda: [])

    def __len__(self):
        return len(self.rows)

    def __str__(self):
        return '\n'.join([str(row) for row in self.rows])

    def add(self, row):
        self.rows.append(row)

    def write(self, path: Path):
        with path.open(mode="w", newline='') as dataset_file:
            for sample in self.rows:
                dataset_file.write(f"{sample}\n")


@dataclass
class XYDataset:
    x: Dataset = Dataset()
    y: Dataset = Dataset()

    def __len__(self):
        return len(self.x)

    def __str__(self):
        return f"x:{self.x}\ny:{self.y}"

    def add(self, x_row, y_row):
        self.x.add(x_row)
        self.y.add(y_row)

    def write(self, path: Path, delimiter: str, offset: bool, headers: str = None):
        with path.open(mode="w", newline='') as dataset_file:
            if headers:
                dataset_file.write(headers + "\n")
            for x, y in zip(self.x.rows, self.y.rows):
                if offset:
                    dataset_file.write(f"{y},{delimiter.join(x)}\n")
                else:
                    dataset_file.write(f"{delimiter.join(x)}\n")

        return path
