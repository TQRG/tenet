from pathlib import Path
from typing import Any

import pandas as pd
from cement import Handler

from securityaware.core.exc import Skip
from securityaware.core.interfaces import HandlersInterface
from securityaware.data.schema import Edge


class NodeHandler(HandlersInterface, Handler):
    class Meta:
        label = 'container'

    def __init__(self, **kw):
        super().__init__(**kw)
        self.path = None
        self.output = None
        self.edge = None
        self.node = None

    def load_dataset(self, suffix: str = '.csv', terminate: bool = False):
        """
            Loads the dataset

            :param suffix: dataset suffix
            :param terminate: flag to terminate program execution if dataset not found (default: False)
            :return: pandas dataframe
        """

        if self.output and self.output.is_file and self.output.suffix == suffix:
            return pd.read_csv(str(self.output))

        self.app.log.error("dataset not found")

        if terminate:
            exit(1)

        return None

    def get(self, attr: str):
        return self.app.connectors[self.edge.name, attr]

    def set(self, attr: str, value: Any, skip: bool = True):
        self.app.connectors[self.edge.name, attr] = value

        if self.has_dataset and self.app.connectors.has_source(self.edge.name) and \
                self.app.connectors.has_values(self.edge.name) and skip:
            raise Skip(f"Connectors for source \"{self.edge.name}\" are instantiated and exist.")

    def load(self, edge: Edge, dataset_name: str):
        """
            Loads all the related paths to the workflow

            :param edge: edge object
        """

        paths = {p.name: p for p in self.app.workdir.iterdir()}

        if edge.name in paths:
            self.app.log.info(f"Loading node {edge.name}")
            self.path = str(paths[edge.name])
            # TODO: include correct datasets, and add the layer as well
        else:
            self.app.log.info(f"Making directory for {edge.name}.")
            self.path = str(self.app.workdir / edge.name)
            Path(self.path).mkdir()

        self.output = Path(self.path, f"{dataset_name}.{self.edge.name}.csv")

    @property
    def has_dataset(self):
        """
            Checks whether output exists.
        """

        return self.output.exists()

    @property
    def is_skippable(self):
        """
            Checks whether node execution can be skipped, that is, output exists and node does not have dependencies.
        """
        return self.has_dataset and not self.app.connectors.has_source(self.edge.name)

    def __str__(self):
        return self.edge.name if self.edge else ""