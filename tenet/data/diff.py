import pandas as pd

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Change:
    content: str
    number: int
    type: str

    @property
    def start_col(self):
        # TODO: This is a temporary solution. Need to find a better way to calculate the start column
        line_with_spaces = self.content.expandtabs(4)
        return (len(line_with_spaces) - len(line_with_spaces.lstrip())) + 1

    @property
    def end_col(self):
        return len(self.content) - 1


@dataclass
class Addition(Change):
    type: str = 'addition'


@dataclass
class Deletion(Change):
    type: str = 'deletion'


@dataclass
class InlineDiff:
    owner: str
    project: str
    version: str
    fpath: str
    sline: int
    scol: int
    eline: int
    ecol: int
    label: str
    pair_hash: str

    def is_null(self):
        return pd.isnull(self.sline) and pd.isnull(self.scol) and pd.isnull(self.eline) and pd.isnull(self.ecol)

    def is_same(self, other):
        return self.owner == other.owner and self.project == other.project and self.version == other.version and \
               self.fpath == other.fpath and self.label == other.label

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

    def to_dict(self, **kwargs):
        il_dict = {'owner': self.owner, 'version': self.version, 'project': self.project, 'fpath': self.fpath,
                   'sline': self.sline, 'scol': self.scol, 'eline': self.eline, 'ecol': self.ecol, 'label': self.label,
                   'pair_hash': self.pair_hash}

        if kwargs:
            il_dict.update(kwargs)

        return il_dict

    def to_list(self):
        return [self.owner, self.project, self.version, self.fpath, self.sline, self.scol, self.eline, self.ecol,
                self.label, self.pair_hash]

    def __str__(self):
        return f"{self.owner},{self.project},{self.version},{self.fpath},{self.sline},{self.scol},{self.eline}," \
               f"{self.ecol},{self.label},{self.pair_hash}"


@dataclass
class FunctionBoundary(InlineDiff):
    ftype: str = None
    label: str = ''
    pair_hash: str = ''

    @staticmethod
    def parse_fn_inline_diffs(fn_boundaries: dict, owner: str, project: str, version: str, fpath: str):
        def get_fn_bound(fn_bound_str: str, ftype: str):
            sline, scol, eline, ecol = fn_bound_str.split(",")
            return FunctionBoundary(sline=int(sline), scol=int(scol), eline=int(eline), ecol=int(ecol), ftype=ftype,
                                    owner=owner, project=project, version=version, fpath=fpath)

        fn_decs = [get_fn_bound(fn_dec, 'fnDec') for fn_dec in fn_boundaries['fnDec']] if 'fnDec' in fn_boundaries else []
        fn_exps = [get_fn_bound(fn_exp, 'fnExp') for fn_exp in fn_boundaries['fnExps']] if 'fnExps' in fn_boundaries else []

        return fn_decs, fn_exps

    def to_list(self):
        return super().to_list() + [self.ftype]

    def is_contained(self, other):
        start_ok = (self.sline < other.sline) or (self.sline == other.sline and self.scol <= other.scol)
        end_ok = (self.eline > other.eline) or (self.eline == other.eline and self.ecol >= other.ecol)

        return start_ok and end_ok

    def __str__(self):
        return super().__str__() + ',' + self.ftype


@dataclass
class DiffBlock:
    start: int
    a_path: str
    b_path: str

    def to_dict(self):
        return {"start": self.start, "a_path": self.a_path, "b_path": self.b_path}


@dataclass
class Entry:
    project: str
    owner: str
    a_version: str
    b_version: str
    diff_block: DiffBlock
    label: str
    a_file_size: int = None
    b_file_size: int = None

    @property
    def full_a_path(self):
        return Path(self.owner, self.project, self.a_version, self.diff_block.a_path)

    @property
    def full_b_path(self):
        return Path(self.owner, self.project, self.b_version, self.diff_block.b_path)

    def to_dict(self):
        diff_block_dict = self.diff_block.to_dict()
        diff_block_dict.update({"owner": self.owner, "project": self.project, "a_version": self.a_version,
                                "b_version": self.b_version, "label": self.label, 'a_file_size': self.a_file_size,
                                'b_file_size': self.b_file_size})

        return diff_block_dict

    def __str__(self):
        return f"{self.owner},{self.project},{self.a_version},{self.b_version},{self.diff_block.start}," \
               f"{self.diff_block.a_path},{self.diff_block.b_path},{self.a_file_size},{self.b_file_size},{self.label}"
