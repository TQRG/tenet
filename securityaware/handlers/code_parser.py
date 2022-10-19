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
        self.comment_filter = None

    def filter_comments(self, code: str):
        if self.comment_filter is None:
            code_language = self.app.get_config('proj_ext')[0]

            if code_language in ['js', 'cpp', 'cc', 'h']:
                self.comment_filter = pyparsing.cppStyleComment.suppress()
            elif code_language == 'java':
                self.comment_filter = pyparsing.javaStyleComment.suppress()
            elif code_language == 'c':
                self.comment_filter = pyparsing.cStyleComment.suppress()
            elif code_language == 'py':
                self.comment_filter = pyparsing.pythonStyleComment.suppress()
            else:
                raise ValueError(f'Comment filter not available for {code_language} language')
        try:
            return self.comment_filter.transformString(code)
        except (pyparsing.exceptions.ParseException, KeyError) as e:
            self.app.log.error(e)
            return None
