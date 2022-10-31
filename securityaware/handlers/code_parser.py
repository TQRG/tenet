import pyparsing

from cement import Handler
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

    @property
    def comment_filter(self):
        if self._comment_filter is None:
            code_language = self.app.get_config('proj_ext')[0]

            if code_language in ['js', 'cpp', 'cc', 'h']:
                self._comment_filter = pyparsing.cppStyleComment.suppress()
            elif code_language == 'java':
                self._comment_filter = pyparsing.javaStyleComment.suppress()
            elif code_language == 'c':
                self._comment_filter = pyparsing.cStyleComment.suppress()
            elif code_language == 'py':
                self._comment_filter = pyparsing.pythonStyleComment.suppress()
            else:
                raise ValueError(f'Comment filter not available for {code_language} language')

        return self._comment_filter

    def filter_comments(self, code: str):
        try:
            return self.comment_filter.transformString(code)
        except (pyparsing.exceptions.ParseException, KeyError) as e:
            self.app.log.error(e)
            return code

    @staticmethod
    def tokenize(code: str):
        # Tokenize, remove comments and blank lines
        import codeprep.api.text as cp_text
        return cp_text.nosplit(code, no_spaces=True)
