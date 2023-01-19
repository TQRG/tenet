from pathlib import Path
from typing import Any

import pandas as pd
from cement import Handler

from tenet.core.exc import Skip, TenetError
from tenet.core.interfaces import PluginsInterface
from tenet.data.plugin import Sources, Sinks
from tenet.data.schema import Node


class NodeHandler(PluginsInterface, Handler):
    class Meta:
        label = 'container'

    def __init__(self, **kw):
        super().__init__(**kw)
        self.path = None
        self.output = None
        self.node = None
        self.edge = None
        self.sources = Sources()
        self.sinks = Sinks()

    def load_dataset(self, suffix: str = '.csv', terminate: bool = False):
        """
            Loads the dataset

            :param suffix: dataset suffix
            :param terminate: flag to terminate program execution if dataset not found (default: False)
            :return: pandas dataframe
        """

        if self.output and self.output.is_file and self.output.suffix == suffix and self.output.exists():
            try:
                return pd.read_csv(str(self.output))
            except pd.errors.EmptyDataError as ede:
                self.app.log.warning(f"Empty Data Error: {ede}")
                return pd.DataFrame()

        self.app.log.error("dataset not found")

        if terminate:
            exit(1)

        return None

    def check_sink(self, name: str):
        if not self.sinks[name]:
            raise TenetError(f"Sink '{name}' of '{self.node.name}' node not instantiated")

        # check if path exists
        if isinstance(self.sinks[name], Path) and not self.sinks[name].exists():
            raise TenetError(f"Path '{self.sinks[name]}' for sink '{name}' of '{self.node.name}' node not found.")

    def get(self, attr: str, default: Any = None):
        try:
            self.sinks[attr] = self.app.connectors[self.node.name, attr]
            self.check_sink(attr)
            return self.app.connectors[self.node.name, attr]
        except KeyError as ke:
            if not default:
                raise TenetError(f"Sink '{ke}' of '{self.node.name}' node not found")

            self.sinks[attr] = default

    def set(self, attr: str, value: Any):
        self.app.connectors[self.node.name, attr] = value
        self.sources[attr] = value

    def load(self, node: Node, dataset_name: str, ext: str = '.csv'):
        """
            Loads all the related paths to the workflow

            :param node: node object
            :param dataset_name: name of the dataset
            :param ext: extension of the dataset
        """

        self.path = self.app.workdir / node.layer / node.name

        if self.path.exists():
            self.app.log.info(f"Loading node {node.name}")
            self.path = Path(self.path)
            # TODO: include correct datasets, and add the layer as well
        else:
            self.app.log.info(f"Making directory for {node.name}.")
            self.path.mkdir(parents=True, exist_ok=True)

        self.output = self.path / f"{dataset_name}{ext}"

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
        return self.has_dataset and len(self.sources) == 0

    def __str__(self):
        return self.node.name if self.node else ""
