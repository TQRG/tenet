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
class FunctionBoundary(InlineDiff):
    ftype: str = None
    label: str = ''

    @staticmethod
    def parse_fn_inline_diffs(fn_boundaries: dict, project: str, fpath: str):
        def get_fn_bound(fn_bound_str: str, ftype: str):
            sline, scol, eline, ecol = fn_bound_str.split(",")
            return FunctionBoundary(sline=int(sline), scol=int(scol), eline=int(eline), ecol=int(ecol), ftype=ftype,
                                    project=project, file_path=fpath)

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


@dataclass
class Entry:
    a_proj: str
    b_proj: str
    diff_block: DiffBlock
    a_file: Path
    b_file: Path
    label: str

    def to_dict(self):
        return {"a_proj": self.a_proj, "b_proj": self.b_proj, "a_file": self.a_file, "b_file": self.b_file,
                "label": self.label, "a_path": self.diff_block.a_path, "b_path": self.diff_block.b_path,
                "start": self.diff_block.start}

    def __str__(self):
        return f"{self.a_proj},{self.b_proj},{self.diff_block.start},{self.diff_block.a_path}," \
               f"{self.diff_block.b_path},{self.a_file},{self.b_file},{self.label}"
