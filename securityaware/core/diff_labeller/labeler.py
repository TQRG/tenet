import difflib
import jsbeautifier

from pathlib import Path
from typing import List, Tuple

from securityaware.core.diff_labeller.misc import check_or_create_dir, get_range_offset, get_row_diff_range
from securityaware.data.diff import InlineDiff, Entry


class Labeler:
    def __init__(self, entry: Entry, a_str: str, b_str: str, inline_proj_dir: Path):
        """
            :param entry: Diff block entry.
            :param a_str: The contents of file A.
            :param b_str: The contents of file B.
        """
        self.entry = entry
        self.a_del_line_cnt = 0
        self.b_add_line_cnt = 0
        self.inline_diffs: List[InlineDiff] = []
        self.sim_ratio = None
        self.inline_proj_dir = inline_proj_dir

        a_path_mod = self.entry.diff_block.a_path.replace('/', '_')
        self.a_path_file = inline_proj_dir / f"a_{a_path_mod}.txt"
        b_path_mod = self.entry.diff_block.b_path.replace('/', '_')
        self.b_path_file = inline_proj_dir / f"b_{b_path_mod}.txt"
        self.inline_file = inline_proj_dir / f"{a_path_mod}_inline.txt"

        self.a_formatted = None
        self.b_formatted = None
        self.b_formatted_range = None
        self.a_formatted_range = None
        self.inline_diff_text = None
        self.a_formatted_lines = None
        self.b_formatted_lines = None
        self.size_a_lines = None
        self.size_b_lines = None

        self.pretty_printing(a_str, b_str)
        self.get_inline_diffs(a_str, b_str)

    def diff_bound(self) -> Tuple[List[int], List[str], int]:
        inline_lines = self.inline_diff_text.splitlines()
        # Look for diffs
        diff_bound = [inline_id for inline_id, inline_line in enumerate(inline_lines) if inline_line.startswith("@@ ")]
        num_diffs = len(diff_bound)
        diff_bound.append(len(inline_lines))

        return diff_bound, inline_lines, num_diffs

    def get_inline_diffs(self, a_str: str, b_str: str):
        self.a_formatted_range = get_range_offset(a_str, self.a_formatted)
        self.b_formatted_range = get_range_offset(b_str, self.b_formatted)
        self.a_formatted_lines = self.a_formatted.splitlines(keepends=True)
        self.b_formatted_lines = self.b_formatted.splitlines(keepends=True)
        self.size_a_lines = len(self.a_formatted_lines)
        self.size_b_lines = len(self.b_formatted_lines)

        # TODO: handler long file names
        # Save the pretty printed inline diffs
        if self.inline_file.exists():
            with self.inline_file.open(mode="r") as ilf:
                print(f"Reading {self.inline_file}")
                self.inline_diff_text = ilf.read()
        else:
            self.inline_diff_text = "".join(difflib.unified_diff(self.a_formatted_lines, self.b_formatted_lines))

            with self.inline_file.open(mode='w') as diff_temp:
                print(f"Writing {self.inline_file}")
                diff_temp.write(self.inline_diff_text)

    def pretty_printing(self, a_str: str, b_str: str):
        """
                Performs pretty-printing on the whole files A and B
        """
        opts = jsbeautifier.default_options()
        opts.indent_size = 0
        check_or_create_dir(self.inline_proj_dir)

        if not self.a_path_file.exists():
            self.a_formatted = jsbeautifier.beautify(a_str, opts)

            with self.a_path_file.open(mode="w") as a_temp:
                print(f"Writing {self.a_path_file}")
                a_temp.write(self.a_formatted)

        else:
            with self.a_path_file.open(mode="r") as apf:
                print(f"Reading {self.a_path_file}")
                self.a_formatted = apf.read()

        if not self.b_path_file.exists():
            self.b_formatted = jsbeautifier.beautify(b_str, opts)

            with self.b_path_file.open(mode="w") as b_temp:
                print(f"Writing {self.b_path_file}")
                b_temp.write(self.b_formatted)

        else:
            with self.b_path_file.open(mode="r") as bpf:
                print(f"Reading {self.b_path_file}")
                self.b_formatted = bpf.read()

    def add_inline_diffs(self, owner: str, project: str, version: str, file_path: str, formatted_range: List,
                         linenos: List[int], label: str):
        """
            Adds the InlineDiff instances.
            :param owner: The repository owner.
            :param project: The respective project.
            :param version: The commit sha.
            :param file_path: The filepath of the target file.
            :param formatted_range: The InlineDiff instance.
            :param linenos: The InlineDiff instance.
            :param label: The InlineDiff instance.
        """
        for i, inlineno in enumerate(linenos):
            inline_diff = InlineDiff(owner=owner, project=project, version=version, fpath=file_path, label=label,
                                     sline=formatted_range[inlineno - 1][0], scol=formatted_range[inlineno - 1][1],
                                     eline=formatted_range[inlineno - 1][2], ecol=formatted_range[inlineno - 1][3])

            self.inline_diffs.append(inline_diff)

    def __call__(self, unsafe_label: str) -> List[InlineDiff]:
        """
            Gets the corresponding row/column index range in the original text and returns a list of result entries.
            :return: A list of InlineDiff instances resulted from this inline diff.
        """
        if not self.inline_diffs:
            diff_bound, inline_lines, num_diffs = self.diff_bound()

            for diff_id in range(num_diffs):
                a_del_linenos, b_add_linenos, _, _ = get_row_diff_range(inline_lines, diff_bound, diff_id)
                self.a_del_line_cnt += len(a_del_linenos)
                self.b_add_line_cnt += len(b_add_linenos)

                self.add_inline_diffs(owner=self.entry.owner, project=self.entry.project, version=self.entry.a_version,
                                      file_path=self.entry.diff_block.a_path, label=unsafe_label,
                                      linenos=a_del_linenos, formatted_range=self.a_formatted_range)

                self.add_inline_diffs(owner=self.entry.owner, project=self.entry.project, version=self.entry.b_version,
                                      file_path=self.entry.diff_block.b_path, label='safe',
                                      linenos=b_add_linenos, formatted_range=self.b_formatted_range)

        return self.inline_diffs

    def calc_sim_ratio(self) -> float:
        """
            Computes and records the similarity ratio between the two files from the parent and the later
            commit at line-level (after pretty-printing). Prints out a notification message if the ratio is
            lower than a specified threshold.
            # >>> Ratio = 1. - (a_only + b_only) / (a_total + b_total)
            #     a_only: The number of deleted lines in file A.
            #     b_only: The number of added lines in file B.
            #     a_total: The total number of lines in file A.
            #     b_total: The total number of lines in file B.
        """
        if not self.sim_ratio:
            total_lines = self.size_a_lines + self.size_b_lines
            self.sim_ratio = 1. - (self.a_del_line_cnt + self.b_add_line_cnt) / total_lines

        return self.sim_ratio

    def __str__(self):
        return f"{self.entry.a_version},{self.entry.diff_block.a_path},{self.entry.b_version}," \
               f"{self.entry.diff_block.b_path},{self.sim_ratio}"

    def warning(self) -> str:
        return f"\nNote: line-level similarity ratio = {self.sim_ratio}, a:-{self.a_del_line_cnt}/{self.size_a_lines}," \
               f"b:+{self.b_add_line_cnt}/{self.size_b_lines}\na: {self.entry.a_version},{self.entry.diff_block.a_path}\n" \
               f"b: {self.entry.b_version},{self.entry.diff_block.b_path}\n "
