import pandas as pd
import tqdm

from typing import Tuple, List

from cement import Handler
from pathlib import Path

from tenet.core.exc import Skip, TenetError
from tenet.core.interfaces import HandlersInterface
from tenet.data.schema import Node, DAGPath, ParallelLayer
from collections import OrderedDict

from tenet.handlers.plugin import PluginHandler


class WorkflowHandler(HandlersInterface, Handler):
    """
        Workflow handler to execute the pipeline.
    """

    class Meta:
        label = 'workflow'

    def __init__(self, **kw):
        super().__init__(**kw)
        self.current_nodes = OrderedDict()
        self.previous_node = None

    def _load(self, dataset_path: Path, path: DAGPath):
        """
            Loads and initializes plugins
        """

        self.app.log.info(f"Loading nodes in {path} path...")

        for node in tqdm.tqdm(path.flatten(), desc="Initializing plugins", colour='blue'):
            if node.name not in self.current_nodes:
                node_handler = self.app.get_plugin_handler(node.plugin)
                node_handler.load(node, dataset_name=dataset_path.stem)
                node_handler.node = node
            else:
                node_handler = self.current_nodes[node.name]

            self.current_nodes[node.name] = node_handler

    def _exec_node(self, node_handler: PluginHandler, dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, Path]:
        self.app.log.info(f"Running node {node_handler}")
        node_handler.get_sinks()
        node_handler.set_sources()

        if node_handler.is_skippable:
            self.app.log.info(f"{node_handler.node.name}: dataset {node_handler.output} exists.")
            dataframe = pd.read_csv(node_handler.output)
            dataset_path = node_handler.output

            return dataframe, dataset_path

        kwargs = node_handler.node.kwargs.copy()

        if node_handler.node.kwargs:
            kwargs.update(node_handler.node.kwargs)

        if node_handler.has_dataset and node_handler.sources.is_init() and \
                self.app.connectors.has_values(node_handler.node.name):

            self.app.log.warning(f"Connectors for source \"{node_handler.node.name}\" are instantiated and exist.")
            self.app.log.warning(f"Skipping {node_handler}.")
            dataframe = node_handler.load_dataset()
            dataset_path = node_handler.output
        else:
            dataframe = node_handler.run(dataset=dataframe, **kwargs)
            dataset_path = node_handler.output

            if dataframe is not None:
                dataframe.to_csv(str(node_handler.output), index=False)
                self.app.log.info(f"Saving dataset {node_handler.output}.")
            else:
                raise TenetError(f"Node {node_handler} returned no dataframe. Stopping execution.")

        self.previous_node = node_handler.node

        return dataframe, dataset_path

    def _exec_layer(self, layer: ParallelLayer, dataframe: pd.DataFrame, dataset_path: Path) \
            -> Tuple[pd.DataFrame, Path]:
        # TODO: verify connection between edges in tracks
        tracks_outcome = []
        layer_dataset_path = self.app.workdir / layer.name / dataset_path.name

        if not layer_dataset_path.exists():
            for t_name, edges in layer.tracks.items():
                self.app.log.info(f"Executing track {t_name}")
                track_outcome = [(dataframe, dataset_path)]

                for edge in edges:
                    for node in edge.to_list():
                        # TODO: change; here we skip the previous executed node that leaded to this layer
                        if node == self.previous_node:
                            continue
                        node_handler = self.current_nodes[node.name]
                        track_dataframe, track_dataset_path = self._exec_node(node_handler, track_outcome[-1][0])
                        track_outcome.append((track_dataframe, track_dataset_path))

                        if not self.app.pargs.suppress_plot:
                            self.app.log.info(f"{node_handler.node.name} plotting...")
                            node_handler.plot(dataframe)

                tracks_outcome.append(track_outcome[-1])

            dataframe = pd.concat([d for d, dp in tracks_outcome], ignore_index=True)
            dataframe.to_csv(str(layer_dataset_path), index=False)
            self.app.log.info(f"{len(dataframe)} entries after merging {list(layer.tracks.keys())} tracks;")
        else:
            # TODO: should load the sources from executed nodes
            self.app.log.warning(f"Dataset for '{layer.name}' layer exists, loading {layer_dataset_path} ...")
            dataframe = pd.read_csv(str(layer_dataset_path))

        self.previous_node = None

        return dataframe, layer_dataset_path

    def __call__(self, dataset_path: Path, dag_path: DAGPath) -> Tuple[pd.DataFrame, Path]:
        self._load(dataset_path, dag_path)
        dataframe = pd.read_csv(str(dataset_path), sep='\t' if dataset_path.suffix == '.tsv' else ',')

        if dataframe.empty:
            raise TenetError(f"Dataset is empty.")

        for el in tqdm.tqdm(dag_path.nodes, desc="Executing pipeline", colour='green'):
            if not isinstance(el, Node):
                dataframe, dataset_path = self._exec_layer(el, dataframe, dataset_path)
            else:
                node_handler = self.current_nodes[el.name]
                dataframe, dataset_path = self._exec_node(node_handler, dataframe)

                if not self.app.pargs.suppress_plot:
                    self.app.log.info(f"{node_handler.node.name} plotting...")
                    node_handler.plot(dataframe)

        return dataframe, dataset_path
