import difflib
import random
import re
import statistics
from typing import Union, List, Tuple

import pyparsing
import code_tokenize as ctok

from cement import Handler
from code_tokenize.tokens import ASTToken

from securityaware.core.interfaces import HandlersInterface


class CodeParserHandler(HandlersInterface, Handler):
    """
        Plugin handler abstraction
    """
    class Meta:
        label = 'code_parser'

    def __init__(self, **kw):
        super().__init__(**kw)
        self._comment_filter = None
        self._code_ext = None
        self._lang_map = {'js': 'javascript', 'py': 'python', 'h': 'c'}
        self._accepted_langs = ['java', 'javascript', 'go', 'ruby', 'php', 'python', 'c', 'cpp']

    @property
    def code_ext(self):
        if self._code_ext is None:
            self._code_ext = self.app.get_config('proj_ext')[0]

        return self._code_ext

    @code_ext.setter
    def code_ext(self, value: str):
        self._code_ext = value

    @property
    def comment_filter(self):
        if self._comment_filter is None:
            if self.code_ext in ['js', 'cpp', 'cc', 'hpp', 'hh', 'c++', 'cxx']:
                self._comment_filter = pyparsing.cppStyleComment.suppress()
            elif self.code_ext == 'java':
                self._comment_filter = pyparsing.javaStyleComment.suppress()
            elif self.code_ext in ['c', 'h']:
                # TODO: pyparsing.cStyleComment.suppress() does not work for single line comments
                self._comment_filter = pyparsing.cppStyleComment.suppress()
            elif self.code_ext == 'py':
                self._comment_filter = pyparsing.pythonStyleComment.suppress()
            else:
                raise ValueError(f'Comment filter not available for {self.code_ext} extension')

        return self._comment_filter

    @comment_filter.deleter
    def comment_filter(self):
        self._comment_filter = None

    def filter_comments(self, code: str, extension: str = None):
        try:
            if extension:
                extension = extension.lower()

                if self._code_ext != extension:
                    del self.comment_filter
                    self._code_ext = extension

            return self.comment_filter.transformString(code)
        except (pyparsing.exceptions.ParseException, KeyError) as e:
            self.app.log.error(e)
            return code

    def tokenize(self, code: str, clean: bool = False, string: bool = True) -> Union[List[str], List[ASTToken]]:
        # Tokenize and remove comments
        if clean:
            code = self.filter_comments(code)

        # TODO: verify accepted languages
        if self.code_ext not in self._accepted_langs:
            lang = self._lang_map[self.code_ext]
        else:
            lang = self.code_ext

        tokens = ctok.tokenize(code, lang=lang, syntax_error="ignore")

        if string:
            str_tokens = []
            # TODO: fix try catch, there is an offset of one for some samples
            try:
                for t in tokens:
                    str_tokens.append(str(t))
            except IndexError:
                self.app.log.error(f"Only could get {len(str_tokens)} of {len(tokens)}")

            return str_tokens

        return tokens

    def match_snippet(self, match, flines):
        if not match:
            self.app.log.warning("No match")
            return None

        size = len(match.groups())
        snippet = ''

        for i in range(1, size + 1):
            start, end = match.group(i).split(',')
            snippet += '\n'.join(flines[int(start): int(end)]) + '\n'
        #        print(start, end, snippet)

        return snippet

    def get_pair_snippet(self, fix_file_content: str, vuln_file_content: str) \
            -> Tuple[Union[None, str], Union[None, int], Union[None, str], Union[None, int]]:

        if fix_file_content and vuln_file_content:
            vuln_lines = vuln_file_content.splitlines()
            fix_lines = fix_file_content.splitlines()

            diff = '\n'.join(difflib.context_diff(vuln_lines, fix_lines))

            match = re.search("\*\*\* (\d+,\d+) \*\*\*\*", diff)
            size_vuln_lines = len(vuln_lines)
            vuln_str = self.match_snippet(match, vuln_lines)

            match = re.search("--- (\d+,\d+) ----", diff)
            size_fix_lines = len(fix_lines)
            fix_str = self.match_snippet(match, fix_lines)

            #if self.real:
            #    sequence_matcher = difflib.SequenceMatcher(None, vuln_lines, fix_lines)
            #    chunks = ['\n'.join(vuln_lines[a: a + size]) for a, b, size in sequence_matcher.get_matching_blocks()]

            return vuln_str, size_vuln_lines, fix_str, size_fix_lines

        return None, None, None, None

    @staticmethod
    def get_non_vuln_snippet(non_vuln_file_content: str, size_fix_lines: int = None, size_vuln_lines: int = None):
        if not non_vuln_file_content:
            return None

        # here, the row contains only non_vuln_files info
        non_vuln_lines = non_vuln_file_content.splitlines()
        size_non_vuln = len(non_vuln_lines)

        # TODO: this should be done differently
        if size_vuln_lines is None:
            size_vuln_lines = random.randint(0, round(size_non_vuln * 0.05))

        if size_fix_lines is None:
            size_fix_lines = random.randint(0, round(size_non_vuln * 0.25))

        # Size of non vuln snippet equal to the mean of the vuln/fix sizes
        size = round(statistics.mean([size_fix_lines, size_vuln_lines]))

        if size_non_vuln > size:
            start = random.randint(0, size_non_vuln - size)
        elif size_non_vuln > size_vuln_lines:
            start = random.randint(0, size_non_vuln - size_vuln_lines)
        elif size_non_vuln > size_fix_lines:
            start = random.randint(0, size_non_vuln - size_fix_lines)
        else:
            start = random.randint(0, round(size_non_vuln/2))

        return '\n'.join(non_vuln_lines[start: start+size])

#    def get_contents(self):
#        if self.non_vuln_file:
#            if not self.vuln_file:
#                return None, None, self.non_vuln_file.read()
#            return self.vuln_file.read(), self.fix_file.read(), self.non_vuln_file.read()

#        return self.vuln_file.read(), self.fix_file.read(), None
