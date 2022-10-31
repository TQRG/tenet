import pandas as pd

from typing import Union
from abc import abstractmethod

from securityaware.handlers.code_parser import CodeParserHandler
from securityaware.handlers.container import ContainerHandler
from securityaware.handlers.cwe_list import CWEListHandler
from securityaware.handlers.github import GithubHandler
from securityaware.handlers.node import NodeHandler
from securityaware.handlers.runner import MultiTaskHandler


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
            self._container_handler = self.app.handler.get('handlers', 'container', setup=True)
            # TODO: fix this
            self._container_handler.path = self.path
        return self._container_handler

    @container_handler.deleter
    def container_handler(self):
        self._container_handler = None

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
