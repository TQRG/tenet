from abc import abstractmethod
from pathlib import Path
from typing import Union

import pandas as pd

from securityaware.handlers.node import NodeHandler


class PluginHandler(NodeHandler):
    """
        Plugin handler abstraction
    """
    class Meta:
        label = 'plugin'

    @abstractmethod
    def run(self, dataset: pd.DataFrame, **kwargs) -> Union[pd.DataFrame, None]:
        """
            Executes plugin

            :param dataset: dataframe with the dataset resulting from the previous node
            :return: dataframe with the processed dataset
        """
        pass
