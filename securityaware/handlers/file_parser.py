from typing import Union

import pandas as pd
import numpy as np

from pathlib import Path

from cement import Handler
from securityaware.core.interfaces import HandlersInterface


class FileParserHandler(HandlersInterface, Handler):
    """
        Plugin handler abstraction
    """
    class Meta:
        label = 'file_parser'

    def __init__(self, **kw):
        super().__init__(**kw)
        # TODO: add this into configuration file
        self.ext_map = {
            "Objective-C": {"m", "mm"}, "Java": {"java", "jsp", "jspf"}, "Scala": {"scala"},
            "C/C++": {"c", "cc", "cxx", "cpp", "hpp", "c++", "h", "hh", "cppm", "ixx", "cp", "inl"},
            "Groovy": {"groovy"},
            "PHP": {"php", "tpl", "inc", "ctp", "phpt", "phtml"}, "JavaScript": {"js", "jsx", "coffee"},
            "Python": {"py"},
            "Config files": {"lock", "gradle", "json", "config", "yaml", "conf"}, "Ruby": {"rb"},
            "HTML": {"html", "erb"},
            "Perl": {"pm"}, "Go": {"go"}, "Lua": {"lua"}, "Erlang": {"erl"}, "C#": {"cs"}, "Rust": {"rust"},
            "Vala": {"vala"},
            "SQL": {"sql"}, "XML": {"xml"}, "Shell": {"sh"},
        }

    @staticmethod
    def get_extension(file: str):
        return Path(file).suffix.replace('.', '').lower()

    def get_files_extension(self, files: Union[str, list]):
        if isinstance(files, str):
            files = eval(files)

            if isinstance(files, dict):
                files = files.keys()

        extensions = set([self.get_extension(file) for file in files if len(file.split(".")) > 1])

        return extensions if len(extensions) > 0 else np.nan

    def get_extension_mapping(self, val: str):
        for key, value in self.ext_map.items():
            if val in value:
                return key

    def get_language(self, extensions):
        if not pd.notna(extensions):
            return np.nan

        languages = []

        for ext in extensions:
            ext_map = self.get_extension_mapping(ext)

            if ext_map is not None:
                languages.append(ext_map)

        return set(languages) if len(languages) > 0 else np.nan
