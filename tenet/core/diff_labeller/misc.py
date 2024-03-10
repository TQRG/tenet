import string
import sys
import numpy as np

from tenet.data.diff import Change, Deletion, Addition
from itertools import accumulate
from pathlib import Path
from typing import TextIO, List, Tuple


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


def get_row_diff_range(lines: List[str], diff_bound: List[int], diff_id: int) -> List[Change]:
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
    diff_info = [x.strip() for x in lines[diff_bound[diff_id]].split("@@") if x.strip()]

    # if len(diff_info) > 1:
    #     diff_func = diff_info[-1]
    line_info = diff_info[0].split()
    a_lineno = int(line_info[0].split(",")[0].strip("-"))
    b_lineno = int(line_info[1].split(",")[0].strip("+"))
    diff_start = diff_bound[diff_id]
    diff_end = diff_bound[diff_id + 1]

    changes = []

    for line_id in range(diff_start + 1, diff_end):
        if lines[line_id].startswith("-"):
            ln = lines[line_id][len("-"):]
            # Ignore comments or newline changes in the diff
            if ln.strip() and not (ln.strip().startswith("//") or ln.strip().startswith("/*")):
                changes.append(Deletion(content=ln, number=a_lineno))
            a_lineno += 1
        elif lines[line_id].startswith("+"):
            ln = lines[line_id][len("+"):]
            if ln.strip() and not (ln.strip().startswith("//") or ln.strip().startswith("/*")):
                changes.append(Addition(content=ln, number=b_lineno))
            b_lineno += 1
        elif lines[line_id].startswith(" "):
            a_lineno += 1
            b_lineno += 1
        # Ignore "\ No newline at end of file"

    return changes


def shift_range(orig_one_line: str, formatted: str, included_chars: set):
    """
    Shifts the character ranges in a formatted string based on excluded characters.

    Args:
        orig_one_line (str): Original line from which the formatting was done.
        formatted (str): Formatted string.
        included_chars (set): Set of characters to include.

    Returns:
        list: List of shifted character ranges in the formatted string.
    """
    # Split the formatted string into lines and count included characters in each line
    formatted_lines = formatted.splitlines()
    formatted_char_count = [len([ch for ch in line if ch in included_chars]) for line in formatted_lines]

    # Calculate cumulative sum of included characters for each line
    formatted_range = np.array(list(accumulate(formatted_char_count)))

    # Compute how much to shift the range in each line of the formatted text
    for pos, ch in enumerate(orig_one_line):
        if ch not in included_chars:
            # Create an array to store the amount of increment needed for each line
            increments = np.zeros_like(formatted_range)

            # Get the indices of the elements greater than or equal to the current position
            gt_excluded_pos = np.argwhere(formatted_range >= pos)

            # Increment the elements at those indices
            increments[gt_excluded_pos] += 1

            # Apply the increments to shift the ranges
            formatted_range += increments

    return list(formatted_range)


def convert_column_pos_to_range(positions):
    ranges = []

    # Construct ranges between adjacent positions
    for i in range(1, len(positions)):
        start = positions[i - 1] + 1
        end = positions[i]
        ranges.append((start, end))

    # Add the initial range from 1 to the first position
    initial_range = (1, positions[0])
    ranges.insert(0, initial_range)

    return ranges


def map_format_to_original_ranges(formatted_ranges, original_line_ranges):
    """
    Maps formatted ranges to corresponding original ranges in a text.

    Args:
        formatted_ranges (list of tuples): Ranges in the formatted text.
        original_line_ranges (list of ints): Line ranges in the original text.

    Returns:
        list of tuples: Original ranges corresponding to each formatted range.
    """
    mapping = []
    current_line = 1

    for i, format_range in enumerate(formatted_ranges):
        initial_line = current_line
        print(i+1, format_range, original_line_ranges[current_line])
        # Find the corresponding line in the original text for the start of the range
        if format_range[0] > original_line_ranges[current_line]:
            for current_line in range(current_line, len(original_line_ranges)):
                if format_range[0] <= original_line_ranges[current_line]:
                    break

        original_start_line = current_line
        original_start_column = format_range[0] - original_line_ranges[current_line - 1]

        # Find the corresponding line in the original text for the end of the range
        if format_range[1] > original_line_ranges[current_line]:
            for current_line in range(current_line, len(original_line_ranges)):
                if format_range[1] <= original_line_ranges[current_line]:
                    break

        original_end_line = current_line
        original_end_column = format_range[1] - original_line_ranges[current_line - 1]

        # Append the mapped range to the result
        mapping.append((original_start_line, original_start_column, original_end_line, original_end_column))
        print(initial_line == current_line)

    return mapping


# TODO: fix this function to make it work properly (unsure if shift or mapping is wrong)
def get_range_offset(orig: str, formatted: str) -> List[Tuple[int, int, int, int]]:
    """
    Matches the pretty-printed text with the original text, gets the corresponding row and
    column range in the original text for each line in the pretty-printed text.
    :param orig: The original text before pretty-printing.
    :param formatted: The pretty-printed text of the original text.
    :return: A list of 4-tuple of (row_start, col_start, row_end, col_end) in the original text
             for each line in the pretty-printed text.
    """
    orig_lines = orig.splitlines()
    orig_one_line = "".join(orig_lines)
    orig_line_widths = [len(x) for x in orig_lines]
    included_chars = set(string.printable) - set(string.whitespace)
    column_pos = shift_range(orig_one_line, formatted, included_chars)

    if column_pos:
        if column_pos[-1] != len(orig_one_line):
            raise ValueError('Size of formatted_range != size of orig_one_line')

        # Convert column_end pos to (column_start, column_end)
        column_range = convert_column_pos_to_range(column_pos)
        orig_line_range = [0] + list(accumulate(orig_line_widths))

        # Update orig pos for each line in the pretty-printed text
        return map_format_to_original_ranges(column_range, orig_line_range)

    return []
