from dataclasses import dataclass
from pathlib import Path


@dataclass
class InlineDiff:
    project: str
    file_path: str
    sline: int
    scol: int
    eline: int
    ecol: int
    label: str

    def is_same(self, other):
        return self.project == other.project and self.file_path == other.file_path and self.label == other.label

    def inbounds(self, other):
        if self.sline > other.eline:
            return False

        if other.sline > self.eline:
            return False

        if self.sline == other.sline:
            if self.eline == other.eline:
                return self.scol >= other.scol and self.ecol <= other.ecol
            if self.eline < other.eline:
                return self.scol >= other.scol

        if self.sline > other.sline:
            if self.eline < other.eline:
                return True
            if self.eline == other.eline:
                return self.ecol <= other.ecol

        return False

    def to_list(self):
        return [self.project, self.file_path, self.sline, self.scol, self.eline, self.ecol, self.label]

    def __str__(self):
        return f"{self.project},{self.file_path},{self.sline},{self.scol},{self.eline},{self.ecol},{self.label}"


@dataclass
class DiffBlock:
    start: int
    a_path: str
    b_path: str


@dataclass
class Entry:
    a_proj: str
    b_proj: str
    diff_block: DiffBlock
    a_file: Path
    b_file: Path
    label: str

    def __str__(self):
        return f"{self.a_proj},{self.b_proj},{self.diff_block.start},{self.diff_block.a_path}," \
               f"{self.diff_block.b_path},{self.a_file},{self.b_file},{self.label}"
