from typing import Tuple

import pandas as pd
import tqdm

from cement import Handler
from pathlib import Path

from tenet.core.exc import Skip, TenetError
from tenet.core.interfaces import HandlersInterface
from tenet.data.schema import Edge
from collections import OrderedDict


class WorkflowHandler(HandlersInterface, Handler):
    """
        Workflow handler to execute the pipeline.
    """

    class Meta:
        label = 'workflow'

    def __init__(self, **kw):
        super().__init__(**kw)
        self.traversal = OrderedDict()

    def load(self, workflow_name, dataset_path: Path, layers: list):
        """
            Loads and initializes plugins
        """
        self.app.log.info(f"Loading {layers} layers for {workflow_name} workflow")
        for node, edge in tqdm.tqdm(self.app.pipeline.walk(layers), desc="Initializing plugins", colour='blue'):
            if not edge:
                edge = Edge(name=node.name, node=node.name)

            self.app.log.info(f"Traversing edge {edge.name}")

            node_handler = self.app.get_plugin_handler(node.name)
            node_handler.load(edge, dataset_name=dataset_path.stem)

            node_handler.node = node
            node_handler.edge = edge
            self.traversal[edge.name] = node_handler

        self.app.log.info("Linking nodes")
        # Instantiates the connectors for the nodes
        if hasattr(self.app, 'connectors'):
            self.app.connectors = self.app.pipeline.link(workflow_name, self.app.executed_edges, self.app.connectors)
        else:
            self.app.extend('connectors', self.app.pipeline.link(workflow_name, self.traversal))

    def __call__(self, dataset_path: Path) -> Tuple[pd.DataFrame, Path]:
        dataframe = pd.read_csv(str(dataset_path), sep='\t' if dataset_path.suffix == '.tsv' else ',')

        if dataframe.empty:
            raise TenetError(f"Dataset is empty.")

        while tqdm.tqdm(self.traversal, desc="Executing pipeline", colour='green'):
            node_name, node_handler = self.traversal.popitem(last=False)

            self.app.log.info(f"Running node {node_handler}")

            if node_handler.is_skippable:
                self.app.log.info(f"{node_handler.edge.name}: dataset {node_handler.output} exists.")
                dataframe = pd.read_csv(node_handler.output)
                dataset_path = node_handler.output

                if not self.app.pargs.suppress_plot:
                    self.app.log.info(f"{node_handler.edge.name} plotting...")
                    node_handler.plot(dataframe)
            else:
                kwargs = node_handler.node.kwargs.copy()

                if node_handler.edge.kwargs:
                    kwargs.update(node_handler.edge.kwargs)

                try:
                    dataframe = node_handler.run(dataset=dataframe, **kwargs)
                    dataset_path = node_handler.output

                    if dataframe is not None:
                        dataframe.to_csv(str(node_handler.output), index=False)
                        self.app.log.info(f"Saving dataset {node_handler.output}.")
                    else:
                        raise TenetError(f"Node {node_handler} returned no dataframe. Stopping execution.")
                    if not self.app.pargs.suppress_plot:
                        node_handler.plot(dataframe)
                except Skip as se:
                    self.app.log.warning(f"{se} Skipping {node_handler}.")
                    dataframe = node_handler.load_dataset()
                    dataset_path = node_handler.output

                    if not self.app.pargs.suppress_plot:
                        node_handler.plot(dataframe)

            self.app.executed_edges[node_name] = node_handler

        return dataframe, dataset_path
