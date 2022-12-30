from typing import Union, List

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
        self._code_language = None
        self._lang_map = {'js': 'javascript', 'py': 'python', 'h': 'c'}

    @property
    def code_language(self):
        if self._code_language is None:
            self._code_language = self.app.get_config('proj_ext')[0]

        return self._code_language

    @property
    def comment_filter(self):
        if self._comment_filter is None:
            if self.code_language in ['js', 'cpp', 'cc', 'h']:
                self._comment_filter = pyparsing.cppStyleComment.suppress()
            elif self.code_language == 'java':
                self._comment_filter = pyparsing.javaStyleComment.suppress()
            elif self.code_language == 'c':
                self._comment_filter = pyparsing.cStyleComment.suppress()
            elif self.code_language == 'py':
                self._comment_filter = pyparsing.pythonStyleComment.suppress()
            else:
                raise ValueError(f'Comment filter not available for {self.code_language} language')

        return self._comment_filter

    def filter_comments(self, code: str):
        try:
            return self.comment_filter.transformString(code)
        except (pyparsing.exceptions.ParseException, KeyError) as e:
            self.app.log.error(e)
            return code

    def tokenize(self, code: str, clean: bool = False, string: bool = True) -> Union[List[str], List[ASTToken]]:
        # Tokenize and remove comments
        if clean:
            code = self.filter_comments(code)
        # TODO: verify accepted languages
        lang = self._lang_map[self.code_language]
        tokens = ctok.tokenize(code, lang=lang, syntax_error="ignore")

        if string:
            return [str(t) for t in tokens]

        return tokens
