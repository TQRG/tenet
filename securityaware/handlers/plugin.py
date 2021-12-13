from abc import abstractmethod
from pathlib import Path
from typing import Union

import pandas as pd
from cement import Handler

from securityaware.core.interfaces import HandlersInterface


class PluginHandler(HandlersInterface, Handler):
    class Meta:
        label = 'plugin'

    def load_dataset(self, path: Path, terminate: bool = False):
        """
            Loads the dataset

            :param path: path to the dataset
            :param terminate: flag to terminate program execution if dataset not found (default: False)
            :return: pandas dataframe
        """

        if path.is_file and path.suffix == '.csv':
            return pd.read_csv(str(path))

        self.app.log.error("dataset not found")

        if terminate:
            exit(1)

        return None


    @abstractmethod
    def run(self, node: dict, cell: dict, dataset: pd.DataFrame, files_path: Path,
            **kwargs) -> Union[pd.DataFrame, None]:
        pass
