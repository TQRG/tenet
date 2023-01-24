import networkx as nx

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Tuple, Callable, List, Union, OrderedDict

from schema import Schema, And, Optional, Or

from tenet.core.exc import TenetError
from tenet.utils.misc import random_id


_nodes = Schema(And([{str: {'plugin': str, Optional('kwargs', default={}): dict}}]))
_edges = Schema(And([{str: {'at': str, 'to': str, Optional('links', default={}): dict}}]))
_links = Schema(And({str: {str: {str: [str]}}}))


@dataclass
class Connector:
    """
        Data class representing node connections
    """
    source: str
    sink: str
    id: str = random_id()
    links: Dict[str, str] = field(default_factory=lambda: {})
    attrs: Dict[str, Any] = field(default_factory=lambda: {})

    def __getitem__(self, key: str):
        for source_attr, sink_attr in self.links.items():
            if key == sink_attr:
                return self.attrs[sink_attr]
            # in case the key is used inside the same plugin
            if key == source_attr:
                return self.attrs[source_attr]

        raise ValueError(f"Key '{key}' for sink '{self.sink}' not found in links. Set it in '{self.source}' source.")

    def __setitem__(self, key: str, value: Any):
        # TODO: check if 'if' is necessary
        # if key in self.attrs:
        self.attrs[key] = value

    def is_init(self) -> bool:
        """
            Attributes exist and are instantiated
        """

        for attr, value in self.attrs.items():
            if value is not None:
                if isinstance(value, (str, Path)):
                    if Path(value).exists():
                        continue
                    return False
            else:
                return False

        return True

    """
    def match(self, attr: str, kind: Any = None):
        if kind and hasattr(self, attr):
            return isinstance(getattr(self, attr), kind)
        return hasattr(self, attr)
    """


@dataclass
class ContainerCommand:
    """
        Data object representing the container command.
    """
    org: str
    parsed: str = None
    skip: bool = True
    parse_fn: Callable = None
    tag: str = None

    def __str__(self):
        return self.parsed if self.parsed else self.org


@dataclass
class Node:
    """
        Data object representing connections between nodes
    """
    name: str
    plugin: str
    layer: str = None
    kwargs: dict = field(default_factory=lambda: {})


@dataclass
class Edge:
    """
        Data object representing connections between nodes
    """
    name: str
    at: Node
    to: Node
    links: dict = field(default_factory=lambda: {})

    def to_list(self) -> list:
        return [self.at, self.to]


@dataclass
class Layer:
    name: str
    edges: Dict[str, Edge]

    def to_list(self,):
        return [(e.at, e.to) for e in self.edges.values()]

    def get_nodes(self):
        return [n for e in self.edges.values() for n in [e.at, e.to]]

    def set_nodes_layer(self, at_node: bool = True, to_node: bool = True):
        for edge in self.edges.values():
            if at_node and not edge.at.layer:
                edge.at.layer = self.name
            if to_node and not edge.to.layer:
                edge.to.layer = self.name


@dataclass
class ParallelLayer:
    name: str
    tracks: Dict[str, List[Edge]]

    def to_list(self) -> list:
        return [(e.at, e.to) for t, v in self.tracks.items() for e in v]

    def set_nodes_layer(self, at_node: bool = True, to_node: bool = False):
        for track in self.tracks.values():
            for edge in track:
                # init layer for 'at nodes' only
                if to_node and not edge.to.layer:
                    edge.to.layer = self.name
                if at_node and not edge.at.layer:
                    edge.at.layer = self.name

    def get_nodes(self):
        nodes = []
        for t, v in self.tracks.items():
            for e in v:
                if e.at not in nodes:
                    nodes.append(e.at)
                if e.to not in nodes:
                    nodes.append(e.to)

        return nodes


@dataclass
class DAGPath:
    nodes: List[Union[Node, ParallelLayer]]

    def flatten(self) -> List[Node]:
        res = []
        last_node = None
        for el in self.nodes:
            if isinstance(el, ParallelLayer):
                res.extend([n for n in el.get_nodes() if n != last_node])
            else:
                last_node = el
                res.append(el)

        return res

    def __str__(self):
        return f"{' -> '.join([el.name for el in self.nodes])}"


@dataclass
class Workflow:
    name: str
    layers: OrderedDict[str, Union[Layer, ParallelLayer]]
    _nodes: Dict[str, Node] = field(default_factory=lambda: {})

    def __str__(self):
        return f"{' -> '.join([el for el in self.layers])}"

    @property
    def nodes(self):
        if not self._nodes:
            last_layer = self.layers[next(reversed(self.layers))]

            if isinstance(last_layer, ParallelLayer):
                last_layer.set_nodes_layer(at_node=False, to_node=True)

            for i, (l_name, layer) in enumerate(self.layers.items()):
                layer.set_nodes_layer()
                for node in layer.get_nodes():
                    if node.name not in self._nodes:
                        self._nodes[node.name] = node

        return self._nodes

    def get_graph(self) -> nx.DiGraph:
        print(f"Building graph for layers {list(self.layers.keys())}")
        # set nodes
        _ = self.nodes
        graph = nx.DiGraph()

        for i, (l_name, layer) in enumerate(self.layers.items()):
            if isinstance(layer, ParallelLayer):
                graph.add_node(l_name, type='layer', layer=l_name)
                for track in layer.tracks.values():
                    if graph.has_node(track[-1].at.name):
                        graph.add_edge(track[-1].at.name, l_name)

                    if i == 0 and track[-1].to.layer != l_name and not graph.has_node(track[-1].to.name):
                        graph.add_node(track[-1].to.name, type='node', layer=track[-1].to.layer)
                        graph.add_edge(l_name, track[-1].to.name)
            else:
                for e_name, edge in layer.edges.items():
                    graph.add_node(edge.at.name, type='node', layer=l_name)
                    graph.add_node(edge.to.name, type='node', layer=l_name)
                    graph.add_edge(edge.at.name, edge.to.name)

        return graph

    def get_all_paths(self, graph: nx.DiGraph) -> List[DAGPath]:
        if len(graph.edges) == 0:
            for n, data in graph.nodes(data=True):
                if data['type'] == 'layer':
                    return [DAGPath([self.layers[n]])]
                else:
                    return [DAGPath([self.nodes[n]])]

        nodes = {k: v['type'] for k, v in graph.nodes(data=True)}
        roots = [node for node in graph.nodes if graph.in_degree(node) == 0]
        leaves = [node for node in graph.nodes if graph.out_degree(node) == 0]
        paths = []

        for root in roots:
            for leaf in leaves:
                for path in nx.all_simple_paths(graph, root, leaf):
                    paths.append(DAGPath([self.nodes[n] if nodes[n] == 'node' else self.layers[n] for n in path]))

        return paths


@dataclass
class Connectors:
    """
        Maps sources and sinks through connectors
    """
    sinks: Dict[str, Dict[str, Connector]] = field(default_factory=lambda: {})
    sources: Dict[str, Dict[str, Connector]] = field(default_factory=lambda: {})

    def init_sinks(self, name: str):
        if name not in self.sinks:
            self.sinks[name] = {}

    def init_sources(self, name: str):
        if name not in self.sources:
            self.sources[name] = {}

    def init_links(self, at_node: str, to_node: str, links: dict):
        connector = Connector(source=at_node, sink=to_node, attrs={}, links=links)

        if at_node not in self.sources:
            raise TenetError(f"'{at_node}' 'at node' not found in sources")

        self.sources[at_node][to_node] = connector

        if to_node not in self.sources:
            raise TenetError(f"'{to_node}' node in the links of '{at_node}' node not found.")

        self.sources[to_node][at_node] = connector

    def __getitem__(self, sink_key: Tuple[str, str]):
        sink, key = sink_key

        # todo: check this code if is used
        for connector in self.sinks[sink].values():
            if connector.sink == sink:
                for k, v in connector.links.items():
                    if connector.sink == sink and key == v:
                        return connector[v]

        for connector in self.sources[sink].values():
            if connector.sink == sink:
                for k, v in connector.links.items():
                    if key == v:
                        return connector[v]

    def __setitem__(self, source_attr: Tuple[str, str], value: Any):
        source, attr = source_attr

        for name, connector in self.sources[source].items():
            if attr in connector.links:
                connector[connector.links[attr]] = value

    def has_values(self, source: str) -> bool:
        """
            Look if the number of instantiated sinks is equal to the number of sinks
        """
        return len(self.sources[source]) == len([con for _, con in self.sources[source].items() if con.is_init()])

    def has_sink(self, node: str):
        """
            Checks whether the node is a sink
        """
        return node in self.sinks

    def has_source(self, node: str, attr: str = None):
        """
            Checks whether the node is a source
        """
        if attr:
            for connector in self.sources[node].values():
                if attr in connector.attrs:
                    return True
            return False
        return node in self.sources and self.sources[node]


def parse_pipeline(yaml: dict):
    """
        Parse the yaml file into piepline

        :param yaml: dictionary from the yaml file
        :return: Pipeline object
    """

    return Schema(And({'nodes': _nodes, 'edges': _edges, 'links': _links,
                       'layers': Schema(And({str: Or(list, [list])})),
                       'workflows': Schema(And({str: list}))})).validate(yaml)
