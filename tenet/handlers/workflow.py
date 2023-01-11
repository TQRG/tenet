import pandas as pd
import pandas.errors
import tqdm
import traceback
import csv 

from cement import Handler
from pathlib import Path

from tenet.core.exc import Skip, TenetError
from tenet.core.interfaces import HandlersInterface
from tenet.data.schema import Edge, Plugin
from tenet.handlers.plugin import PluginHandler


class WorkflowHandler(HandlersInterface, Handler):
    """
        Workflow handler to execute the pipeline.
    """

    class Meta:
        label = 'workflow'

    def __init__(self, **kw):
        super().__init__(**kw)
        self.traversal = {}

    def load(self, dataset_path: Path):
        """
            Loads and initializes plugins
        """

        for node, edge in tqdm.tqdm(self.app.pipeline.walk(), desc="Initializing plugins", colour='blue'):
            if not edge:
                edge = Edge(name=node.name, node=node.name)

            self.app.log.info(f"Traversing edge {edge.name}")

            node_handler = self.app.get_plugin_handler(node.name)
            node_handler.load(edge, dataset_name=dataset_path.stem)

            node_handler.node = node
            node_handler.edge = edge
            self.traversal[edge.name] = node_handler

        self.app.log.info("Linking nodes")
        connectors = self.app.pipeline.link(self.traversal)
        self.app.extend('connectors', connectors)

    def __call__(self, dataset_path: Path):
        dataframe = pd.read_csv(str(dataset_path), sep='\t' if dataset_path.suffix == '.tsv' else ',')
        # Instantiates the connectors for the nodes

        if dataframe is None:
            self.app.log.error(f"Could not load dataset. Not found.")
            exit(1)

        for _, node_handler in tqdm.tqdm(self.traversal.items(), desc="Executing pipeline", colour='green'):
            self.app.log.info(f"Running node {node_handler}")

            if node_handler.is_skippable:
                self.app.log.info(f"{node_handler.edge.name}: dataset {node_handler.output} exists.")
                dataframe = pd.read_csv(node_handler.output)
                self.app.log.info(f"{node_handler.edge.name} plotting...")
                if not self.app.pargs.suppress_plot:
                    node_handler.plot(dataframe)
                continue

            kwargs = node_handler.node.kwargs.copy()

            if node_handler.edge.kwargs:
                kwargs.update(node_handler.edge.kwargs)

            try:
                dataframe = node_handler.run(dataset=dataframe, **kwargs)

                if dataframe is not None:
                    dataframe.to_csv(str(node_handler.output), index=False)
                    self.app.log.warning(f"Saving dataset {node_handler.output}.")
                else:
                    self.app.log.warning(f"Node {node_handler} returned no dataframe. Stopping execution.")
                    break
                if not self.app.pargs.suppress_plot:
                    node_handler.plot(dataframe)
            except Skip as se:
                self.app.log.warning(f"{se} Skipping {node_handler}.")
                dataframe = node_handler.load_dataset()
                if not self.app.pargs.suppress_plot:
                    node_handler.plot(dataframe)
                continue
            except TenetError:
                self.app.log.error(f"Plugin '{node_handler.node.name}' raised exception with {traceback.format_exc()}\nStopped execution.")
                exit(1)
