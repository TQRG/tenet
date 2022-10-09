import string
import sys
import numpy as np

from itertools import accumulate
from pathlib import Path
from typing import TextIO, List, Tuple

included_chars = set(string.printable) - set(string.whitespace)


def check_or_create_dir(dirname: Path) -> None:
    """
    Creates a directory specified by the input if it does not exist, or terminates the program if
    the directory name collides with an existing file.
    :param dirname: The path of the directory to be created.
    """
    if not dirname.exists():
        dirname.mkdir(parents=True)
    elif not dirname.is_dir():
        print("Error: {} is not a folder.".format(dirname), file=sys.stderr)
        sys.exit(1)


def safe_write(outfile: TextIO, data: str, proj: str) -> None:
    """
    Calls the write method of the specified TextIO object safely, ignoring any UnicodeEncodeError.
    :param outfile: The TextIO object of the file to be written to.
    :param data: The text to be written to the file.
    :param proj: A string containing the project id of the current commit.
    """
    try:
        outfile.write(data)
    except UnicodeEncodeError as e:
        print(proj + ": ", end="")
        print(e)


def get_row_diff_range(lines: List[str], diff_bound: List[int],
                       diff_id: int) -> Tuple[List[int], List[int], List[str], List[str]]:
    """
    Parses the diff text chunks, gets the line numbers and the contents where diffs occur.
    :param lines: A list of diff texts split by lines.
    :param diff_bound: A list of the starting line number of each diff chunk.
    :param diff_id: The index of the current diff chunk.
    :return: A 4-tuple of (line numbers where deletions happen in file A,
                           line numbers where additions happen in file B,
                           contents of the deleted lines in file A,
                           contents of the added lines in file B)
    """
    # Format of the diff range line:
    # @@ -<a_start>,<a_lines> +<b_start>,<b_lines> @@ [<func>]
    diff_info = [x.strip() for x in lines[diff_bound[diff_id]].split("@@")
                 if x.strip()]

    # if len(diff_info) > 1:
    #     diff_func = diff_info[-1]
    line_info = diff_info[0].split()
    a_lineno = int(line_info[0].split(",")[0].strip("-"))
    b_lineno = int(line_info[1].split(",")[0].strip("+"))
    diff_start = diff_bound[diff_id]
    diff_end = diff_bound[diff_id + 1]
    a_del_linenos = []
    b_add_linenos = []
    a_del_lines = []
    b_add_lines = []

    for line_id in range(diff_start + 1, diff_end):
        if lines[line_id].startswith("-"):
            ln = lines[line_id][len("-"):]
            # Ignore comments or newline changes in the diff
            if ln.strip() and not (ln.strip().startswith("//") or ln.strip().startswith("/*")):
                a_del_linenos.append(a_lineno)
                a_del_lines.append(ln)
            a_lineno += 1
        elif lines[line_id].startswith("+"):
            ln = lines[line_id][len("+"):]
            if ln.strip() and not (ln.strip().startswith("//") or ln.strip().startswith("/*")):
                b_add_linenos.append(b_lineno)
                b_add_lines.append(ln)
            b_lineno += 1
        elif lines[line_id].startswith(" "):
            a_lineno += 1
            b_lineno += 1
        # Ignore "\ No newline at end of file"
    return a_del_linenos, b_add_linenos, a_del_lines, b_add_lines


def shift_range(orig_one_line: str, formatted: str):
    # Remove and re-insert excluded characters (whitespaces and non-ascii)
    formatted_lines = formatted.splitlines()
    formatted_char_cnt = [len([ch for ch in ln if ch in included_chars]) for ln in formatted_lines]
    formatted_range = np.array(list(accumulate(formatted_char_cnt)))
    excluded_total = [pos for pos, ch in enumerate(orig_one_line) if ch not in included_chars]

    for excluded_pos in excluded_total:
        increments = np.zeros_like(formatted_range)
        # get the index of the elements greater than excluded_pos
        gt_excluded_pos = np.argwhere(formatted_range >= excluded_pos)
        # increment the elements
        increments[gt_excluded_pos] += 1
        # to apply the increments
        formatted_range += increments

    return list(formatted_range)


def get_range_offset(orig: str, formatted: str) -> List[Tuple[int, int, int, int]]:
    """
    Matches the pretty-printed text with the original text, gets the corresponding row and
    column range in the original text for each line in the pretty-printed text.
    :param orig: The original text before pretty-printing.
    :param formatted: The pretty-printed text of the original text.
    :return: A list of 4-tuple of (row_start, col_start, row_end, col_end) in the original text
             for each line in the pretty-printed text.
    """

    orig_one_line = "".join(orig.splitlines())
    formatted_range = shift_range(orig_one_line, formatted)

    if formatted_range:
        if formatted_range[-1] != len(orig_one_line):
            raise ValueError('Size of formatted_range != size of orig_one_line')

        # Convert column_end pos to (column_start, column_end)
        for i in range(len(formatted_range) - 1, 0, -1):
            formatted_range[i] = (formatted_range[i - 1] + 1, formatted_range[i])
        formatted_range[0] = (1, formatted_range[0])

        orig_line_widths = [len(x) for x in orig.splitlines()]
        orig_line_range = [0] + list(accumulate(orig_line_widths))

        # Update orig pos for each line in the pretty-printed text
        orig_row = 1

        for r in range(len(formatted_range)):
            while formatted_range[r][0] > orig_line_range[orig_row]:
                orig_row += 1
            r_start = orig_row
            c_start = formatted_range[r][0] - orig_line_range[orig_row - 1]
            while formatted_range[r][1] > orig_line_range[orig_row]:
                orig_row += 1
            r_end = orig_row
            c_end = formatted_range[r][1] - orig_line_range[orig_row - 1]
            formatted_range[r] = (r_start, c_start, r_end, c_end)
    return formatted_range
