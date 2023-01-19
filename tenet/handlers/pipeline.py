import yaml

from pathlib import Path
from cement import Handler
from schema import SchemaError
from collections import OrderedDict
from typing import List

from tenet.core.exc import TenetError
from tenet.core.interfaces import HandlersInterface
from ..data.schema import parse_pipeline, Connectors, Layer, ParallelLayer, Node, Workflow, Edge


class PipelineHandler(HandlersInterface, Handler):
    """
        Workflow handler to execute the pipeline.
    """

    class Meta:
        label = 'pipeline'

    def __init__(self, **kw):
        super().__init__(**kw)
        self.edges = {}
        self.nodes = {}
        self.workflows = {}

    def _init_nodes(self, nodes: list):
        for node in nodes:
            for k, v in node.items():
                self.nodes[k] = Node(name=k, **v)

    def _init_edges(self, edges: list):
        for edge_el in edges:
            for e_name, edge in edge_el.items():
                self.edges[e_name] = Edge(name=e_name, at=self.nodes[edge['at']], to=self.nodes[edge['to']])

    def _init_workflows(self, workflows: dict):
        for w_name, workflow in workflows.items():
            layers = OrderedDict()

            for layer in workflow:
                if layer not in self.layers:
                    raise TenetError(f"Layer {layer} not defined")

                layers[layer] = self.layers[layer]

            self.workflows[w_name] = Workflow(name=w_name, layers=layers)

    def _init_layers(self, layers):
        self.layers = {}

        for l_name, l_edges in layers.items():
            if any(isinstance(x, list) for x in l_edges):
                self.layers[l_name] = ParallelLayer(tracks={i: [self.edges[e] for e in x] for i, x in enumerate(l_edges)},
                                                    name=l_name)
            else:
                self.layers[l_name] = Layer(edges={le: self.edges[le] for le in l_edges}, name=l_name)

    def _init_connectors(self):
        self.app.extend('connectors', Connectors())

        for name, node in self.nodes.items():
            self.app.connectors.init_sources(name)
            self.app.connectors.init_sinks(name)

    def _init_links(self, links):
        """
            creates the respective connectors
        """
        self.app.log.info("Linking nodes")
        parsed_links = {}

        for at_node, at_node_links in links.items():
            for at_node_source, to_node_links in at_node_links.items():
                for to_node_sink, to_nodes in to_node_links.items():
                    for to_node in to_nodes:
                        if at_node not in parsed_links:
                            parsed_links[at_node] = {}
                        if to_node not in parsed_links[at_node]:
                            parsed_links[at_node][to_node] = {}

                        parsed_links[at_node][to_node][at_node_source] = to_node_sink

        for at_node, to_nodes in parsed_links.items():
            for to_node, at_node_links in to_nodes.items():
                self.app.connectors.init_links(at_node=at_node, to_node=to_node, links=at_node_links)

    def load(self, path: Path):
        """
            Loads and initializes pipeline
        """

        with path.open(mode="r") as stream:
            try:
                self.app.log.info(f"Parsing pipeline file: {path}")
                pipeline = parse_pipeline(yaml.safe_load(stream))
            except SchemaError as se:
                raise TenetError(str(se))

        self._init_nodes(pipeline['nodes'])
        self._init_edges(pipeline['edges'])
        self._init_layers(pipeline['layers'])
        self._init_workflows(pipeline['workflows'])

        self._init_connectors()
        self._init_links(pipeline['links'])

    def get_nodes(self, nodes: str) -> List[Node]:
        return [self.nodes[node] for node in nodes]

    def get_nodes_from_edges(self, edges: List[Edge]) -> List[Node]:
        nodes = []
        for edge in edges:
            for n in edge.to_list():
                if self.nodes[n] not in nodes:
                    nodes.append(self.nodes[n])

        return nodes
