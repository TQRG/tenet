import difflib

from typing import List

from tenet.core.diff_labeller.misc import get_range_offset, get_row_diff_range
from tenet.data.diff import Change


class Labeler:
    def __init__(self, a_str: str, b_str: str, prettify: bool = False, extension: str = None):
        """
            :param a_str: The contents of file A.
            :param b_str: The contents of file B.
        """
        self.a_str = a_str
        self.b_str = b_str
        self.prettify = prettify
        self.extension = extension  # TODO: extract the extension from the file name

        self._a_formatted = None
        self._b_formatted = None
        self._b_formatted_range = None
        self._a_formatted_range = None

        self._diff_lines = None
        self._diff_bound = None

    @property
    def diff_lines(self) -> List[str]:
        if not self._diff_lines:
            if self.prettify:
                self._diff_lines = list(difflib.unified_diff(self.a_formatted.splitlines(keepends=True),
                                                             self.b_formatted.splitlines(keepends=True)))
            else:
                self._diff_lines = list(difflib.unified_diff(self.a_str.splitlines(keepends=True),
                                                             self.b_str.splitlines(keepends=True)))

        return self._diff_lines

    @property
    def diff_bound(self) -> List[int]:
        if not self._diff_bound:
            self._diff_bound = [idx for idx, line in enumerate(self.diff_lines) if line.startswith("@@ ")]
            self._diff_bound.append(len(self.diff_lines))

        return self._diff_bound

    @property
    def num_diffs(self):
        return len(self.diff_bound) - 1

    @property
    def a_formatted(self):
        if not self._a_formatted:
            self._a_formatted = self.get_pretty_printed(self.a_str)

        return self.a_formatted

    @property
    def b_formatted(self):
        if not self._b_formatted:
            self._b_formatted = self.get_pretty_printed(self.b_str)

        return self._b_formatted

    @property
    def a_formatted_range(self):
        if not self._a_formatted_range:
            self._a_formatted_range = get_range_offset(self.a_str, self.a_formatted)

        return self._a_formatted_range

    @property
    def b_formatted_range(self):
        if not self._b_formatted_range:
            self._b_formatted_range = get_range_offset(self.b_str, self.b_formatted)

        return self._b_formatted_range

    def get_pretty_printed(self, code: str):
        """
                Performs pretty-printing on the whole files A and B
        """
        if self.extension in ['js', '.js', '.ts', '.tsx', '.jsx']:
            import jsbeautifier

            opts = jsbeautifier.default_options()
            opts.indent_size = 0

            return jsbeautifier.beautify(code, opts)
        else:
            # TODO: To be implemented for other programming languages
            return code

    def parse_inline_diff(self, change: Change):
        """
            Adds the InlineDiff instances.
            :param change: The Change instance.
        """

        output = {'type': change.type}

        if self.prettify:
            idx = change.number - 1
            formatted_range = self.b_formatted_range if change.type == 'addition' else self.a_formatted_range
            output.update({'sline': formatted_range[idx][0], 'scol': formatted_range[idx][1],
                           'eline': formatted_range[idx][2], 'ecol': formatted_range[idx][3]})
        else:
            output.update({'sline': change.number, 'scol': change.start_col,
                           'eline': change.number, 'ecol': change.end_col})

        return output

    def __call__(self, beautify: bool = False) -> List[dict]:
        """
            Gets the corresponding row/column index range in the original text and returns a list of result entries.
            :return: A list of InlineDiff instances resulted from this inline diff.
        """

        results = []

        for diff_id in range(self.num_diffs):
            for change in get_row_diff_range(self.diff_lines, self.diff_bound, diff_id):
                results.append(self.parse_inline_diff(change))

        return results
