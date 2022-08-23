from pathlib import Path

import pandas as pd
import pandas.errors
import tqdm
import traceback
from cement import Handler

from securityaware.core.exc import Skip
from securityaware.core.interfaces import HandlersInterface
from securityaware.data.schema import Edge, Plugin
from securityaware.handlers.plugin import PluginHandler


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

            # self.app.log.info(f"Traversing edge {edge.name}")

            if isinstance(node, Plugin):
                node_handler = self.app.get_plugin_handler(node.name)
                node_handler.load(edge, dataset_name=dataset_path.stem)
            else:
                node_handler = self.app.handler.get('handlers', 'container', setup=True)
                # TODO: fix this abomination

                if not node.output:
                    node_handler.load(edge, dataset_name='output', ext='')
                else:
                    node_handler.load(edge, dataset_name=node.output, ext='')

                node.output = node_handler.output

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

            if isinstance(node_handler, PluginHandler):
                if node_handler.is_skippable:
                    self.app.log.info(f"{node_handler.edge.name}: dataset {node_handler.output} exists.")
                    dataframe = pd.read_csv(node_handler.output)
                    continue

                kwargs = node_handler.node.kwargs

                if node_handler.edge.kwargs:
                    kwargs.update(node_handler.edge.kwargs)

                try:
                    dataframe = node_handler.run(dataset=dataframe, **kwargs)
                except Skip as se:
                    self.app.log.warning(f"{se} Skipping {node_handler}.")
                    dataframe = node_handler.load_dataset()
                    continue
                except Exception:
                    self.app.log.error(f"Plugin '{node_handler.node.name}' raised exception with {traceback.format_exc()}\nStopped execution.")
                    exit(1)

                if dataframe is not None:
                    dataframe.to_csv(str(node_handler.output), index=False)
                    self.app.log.warning(f"Saving dataset {node_handler.output}.")
                else:
                    self.app.log.warning(f"Node {node_handler} returned no dataframe. Stopping execution.")
                    break
            else:
                skip = False

                try:
                    node_handler.parse()
                except Skip as se:
                    self.app.log.warning(str(se))
                    skip = True

                if node_handler.find_output():
                    self.app.log.warning(f"Loading existing output dataset from {node_handler.output}")
                    try:
                        dataframe = node_handler.load_dataset(suffix=node_handler.output.suffix)
                    except pandas.errors.ParserError as pe:
                        self.app.log.warning(str(pe))
                    continue

                if skip:
                    continue

                exec_status, cmds = node_handler.run()

                if not exec_status:
                    break

                if not node_handler.output.exists():
                    with node_handler.output.open(mode='w') as f:
                        outputs = '\n'.join([cmd.output for cmd in cmds])
                        f.write(outputs)
