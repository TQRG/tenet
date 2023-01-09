import functools
import shutil

import pandas as pd
import requests

from typing import Union, Tuple
from abc import abstractmethod
from urllib.parse import urlparse

from tqdm import tqdm
from requests import Response
from pathlib import Path

from tenet.handlers.code_parser import CodeParserHandler
from tenet.handlers.container import ContainerHandler
from tenet.handlers.cwe_list import CWEListHandler
from tenet.handlers.file_parser import FileParserHandler
from tenet.handlers.github import GithubHandler
from tenet.handlers.node import NodeHandler
from tenet.handlers.runner import MultiTaskHandler
from tenet.handlers.sampling import SamplingHandler


class PluginHandler(NodeHandler):
    """
        Plugin handler abstraction
    """
    class Meta:
        label = 'plugin'

    def __init__(self, **kw):
        super().__init__(**kw)

        self._multi_task_handler: MultiTaskHandler = None
        self._code_parser_handler: CodeParserHandler = None
        self._container_handler: ContainerHandler = None
        self._github_handler: GithubHandler = None
        self._cwe_list_handler: CWEListHandler = None
        self._file_parser_handler: FileParserHandler = None
        self._sampling_handler: SamplingHandler = None

    @property
    def github_handler(self):
        if not self._github_handler:
            self._github_handler = self.app.handler.get('handlers', 'github', setup=True)
        return self._github_handler

    @github_handler.deleter
    def github_handler(self):
        self._github_handler = None

    @property
    def cwe_list_handler(self):
        if not self._cwe_list_handler:
            self._cwe_list_handler = self.app.handler.get('handlers', 'cwe_list', setup=True)
        return self._cwe_list_handler

    @cwe_list_handler.deleter
    def cwe_list_handler(self):
        self._cwe_list_handler = None

    @property
    def multi_task_handler(self):
        if not self._multi_task_handler:
            self._multi_task_handler = self.app.handler.get('handlers', 'multi_task', setup=True)
        return self._multi_task_handler

    @multi_task_handler.deleter
    def multi_task_handler(self):
        self._multi_task_handler = None

    @property
    def code_parser_handler(self):
        if not self._code_parser_handler:
            self._code_parser_handler = self.app.handler.get('handlers', 'code_parser', setup=True)
        return self._code_parser_handler

    @code_parser_handler.deleter
    def code_parser_handler(self):
        self._code_parser_handler = None

    @property
    def container_handler(self):
        if not self._container_handler:
            self._container_handler = self.app.handler.get('plugins', 'container', setup=True)
            # TODO: fix this
            self._container_handler.path = self.path
        return self._container_handler

    @container_handler.deleter
    def container_handler(self):
        self._container_handler = None

    @property
    def file_parser_handler(self):
        if not self._file_parser_handler:
            self._file_parser_handler = self.app.handler.get('handlers', 'file_parser', setup=True)
        return self._file_parser_handler

    @file_parser_handler.deleter
    def file_parser_handler(self):
        self._file_parser_handler = None

    @property
    def sampling_handler(self):
        if not self._sampling_handler:
            self._sampling_handler = self.app.handler.get('handlers', 'sampling', setup=True)
        return self._sampling_handler

    @sampling_handler.deleter
    def sampling_handler(self):
        self._sampling_handler = None

    @abstractmethod
    def run(self, dataset: pd.DataFrame, **kwargs) -> Union[pd.DataFrame, None]:
        """
            Executes plugin

            :param dataset: dataframe with the dataset resulting from the previous node
            :return: dataframe with the processed dataset
        """
        pass

    def plot(self, dataset: pd.DataFrame, **kwargs):
        """
            Plots resulting dataframe

            :param dataset: dataframe with the dataset resulting from the previous node
            :return: dataframe with the processed dataset
        """
        pass

    def convert_labels(self, labels: pd.Series):
        label_map = self.app.get_config('labels_map')
        return labels.apply(lambda l: label_map[l])

    def download_file_from_url(self, url: str, extract: bool = False) -> Union[Tuple[Response, Path], None]:

        if not 'http' in url:
            self.app.lof.warning(f"URL {url} is not valid.")
            return None

        file_path = self.path / Path(urlparse(url).path).name
        extract_file_path = self.path / file_path.stem
        response = requests.get(url, stream=True, allow_redirects=True)

        if response.status_code != 200:
            self.app.log.error(f"Request to {url} returned status code {response.status_code}")
            return None

        total_size_in_bytes = int(response.headers.get('Content-Length', 0))

        if file_path.exists() and file_path.stat().st_size == total_size_in_bytes:
            self.app.log.warning(f"File {file_path} exists. Skipping download...")
        else:
            desc = "(Unknown total file size)" if total_size_in_bytes == 0 else ""
            response.raw.read = functools.partial(response.raw.read, decode_content=True)  # Decompress if needed

            with tqdm.wrapattr(response.raw, "read", total=total_size_in_bytes, desc=desc) as r_raw:
                with file_path.open("wb") as f:
                    shutil.copyfileobj(r_raw, f)

        if extract:
            if not extract_file_path.exists():
                self.app.log.info(f"Extracting file {extract_file_path}...")
                shutil.unpack_archive(file_path, self.path)

            return response, extract_file_path

        return response, file_path
