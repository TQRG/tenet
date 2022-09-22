import re

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, AnyStr, Any, Tuple, Union
from schema import Schema, And, Use, Optional, Or

from securityaware.core.exc import SecurityAwareError
from securityaware.utils.misc import random_id

_container = Schema(
    And({'name': str, 'image': str, Optional('output', default=None): str,
         'cmds': Schema(And([str], Use(lambda cmds: [ContainerCommand(org=cmd) for cmd in cmds])))},
        Use(lambda c: Container(**c))))

_plugin = Schema(
    And({'name': str, Optional('kwargs', default={}): dict},
        Use(lambda p: Plugin(**p))))

_edges = Schema(
    And([{str: {'node': str, Optional('sinks', default={}): dict, Optional('sources', default=[]): list,
                Optional('placeholders', default={}): dict, Optional('kwargs', default={}): dict}}],
        Use(lambda els: {k: Edge(name=k, sinks=v['sinks'], sources=v['sources'], kwargs=v['kwargs'], node=v['node'],
                                 placeholders={t: Placeholder(tag=t, value=p, node=k) for t, p in v['placeholders'].items()}) for el in els for k, v in el.items()}))
)

_layers = Schema(And({str: _edges}, Use(lambda l: {k: Layer(edges=v) for k, v in l.items()})))

_nodes = Schema(
    And(
        [Or({'container': _container}, {'plugin': _plugin})],
        Use(lambda n: {v.name: v for el in n for k, v in el.items()})
    )
)


@dataclass
class Connector:
    """
        Data class representing edge connections
    """
    source: str
    sink: str
    id: str = random_id()
    links: Dict[str, str] = field(default_factory=lambda: {})
    attrs: Dict[str, Any] = field(default_factory=lambda: {})

    def map_placeholders(self):
        """
            Looks for placeholders in the attributes and returns them with their associated values.
        """
        matches = {}

        for source_attr, sink_attr in self.links.items():
            if re.search("(p\d+)", sink_attr):
                matches[sink_attr] = self.attrs[source_attr]

        return matches

    def __getitem__(self, key: str):
        for source_attr, sink_attr in self.links.items():

            if key == sink_attr:
                return self.attrs[source_attr]
            # in case the key is used inside the same plugin
            if key == source_attr:
                return self.attrs[source_attr]

        raise ValueError(f'Key {key} for sink {self.sink} not found in links. Set it in {self.source} source.')

    def __setitem__(self, key: str, value: Any):
        if key in self.attrs:
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
class Edge:
    """
        Data object representing connections between nodes
    """
    name: str
    node: str
    sinks: dict = field(default_factory=lambda: {})
    sources: list = field(default_factory=lambda: [])
    placeholders: dict = field(default_factory=lambda: {})
    kwargs: dict = field(default_factory=lambda: {})

    @property
    def connectors(self):
        cons = {}
        cons.update(self.sinks)
        # cons.update({l: k for k, v in self.sources.items() for l in v})

        return cons


@dataclass
class Plugin:
    """
        Code extensions
    """
    name: str
    kwargs: dict = field(default_factory=lambda: {})


@dataclass
class Placeholder:
    tag: str
    node: str
    value: Any

    def __str__(self):
        return f"{self.node} - {self.tag}={self.value}"


@dataclass
class ContainerCommand:
    """
        Data object representing the container command.
    """
    org: str
    parsed: str = None
    skip: bool = True
    placeholders: Dict[str, Placeholder] = field(default_factory=lambda: {})

    def get_placeholders(self):
        return {t: p.value for t, p in self.placeholders.items()}

    def __str__(self):
        return self.parsed if self.parsed else self.org


@dataclass
class Container:
    """
        Docker container
    """
    name: str
    image: str
    cmds: List[ContainerCommand]
    output: str

    def find_placeholders(self, skip: bool = True):
        """
            Looks up for and returns the placeholders in the commands.
        """
        matches = []

        def match_placeholder(string: str):
            match = re.findall("\{(p\d+)\}", string)

            if match:
                matches.extend(match)

            return match

        for cmd in self.cmds:
            if match_placeholder(cmd.org):
                cmd.skip = skip

        match_placeholder(str(self.output))

        return set(matches)


@dataclass
class Layer:
    edges: Dict[str, Edge]

    def traverse(self, nodes: Dict[str, Union[Plugin, Container]]):
        return [(nodes[edge.node], edge) for edge in self.edges.values()]


@dataclass
class Connectors:
    """
        Maps sources and sinks through connectors
    """
    sinks: Dict[str, Dict[str, Connector]] = field(default_factory=lambda: {})
    sources: Dict[str, Dict[str, Connector]] = field(default_factory=lambda: {})

    def __getitem__(self, sink_key: Tuple[str, str]):
        sink, key = sink_key

        for connector in self.sinks[sink].values():
            if key in connector.links.values():
                return connector[key]

        for connector in self.sources[sink].values():
            if key in connector.links.keys():
                return connector[key]

    def __setitem__(self, source_attr: Tuple[str, str], value: Any):
        source, attr = source_attr

        for name, connector in self.sources[source].items():
            connector[attr] = value

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
        return node in self.sources


@dataclass
class Pipeline:
    """
        Data object representing the pipeline
    """
    layers: dict
    nodes: dict
    workflow: list

    def unpack(self):
        return {name: edge for l_name, layer in self.layers.items() if l_name in self.workflow for name, edge in layer.edges.items()}

    def match(self, placeholders: list, sink_attrs: list):
        """
            Checks whether the placeholders match sink attributes.
        """
        for attr in sink_attrs:
            if attr not in placeholders:
                return False
        return True

    def link(self, node_handlers: dict) -> Connectors:
        """
            creates the respective connectors
        """
        connectors = Connectors()
        sources = {}

        for _, edge in self.unpack().items():
            if edge.sources:
                if edge.name not in connectors.sources:
                    connectors.sources[edge.name] = {}

                # If node has attributes, init connector with the value of the attributes
                sources[edge.name] = {attr: getattr(node_handlers[edge.name], attr, None) for attr in edge.sources}

            for source, links in edge.sinks.items():
                if source == edge.name:
                    if isinstance(self.nodes[edge.node], Container):
                        placeholders = self.nodes[edge.node].find_placeholders(skip=False)

                        if not self.match(placeholders, list(links.values())):
                            raise ValueError(f"Placeholders are not matching attributes in the container commands.")
                    else:
                        raise ValueError(f"Plugin {source} cannot reference itself.")

                if source not in sources:
                    raise SecurityAwareError(f"source {source} must be defined before sink {edge.name}")

                if source in sources:
                    connector = Connector(source=source, sink=edge.name, attrs=sources[source], links=links)
                    connectors.sources[source][edge.name] = connector

                    if edge.name not in connectors.sinks:
                        connectors.sinks[edge.name] = {}

                    connectors.sinks[edge.name][source] = connector

        return connectors

    def walk(self):
        """
            Walks the edges and returns list with the traversal. Initializes edge connectors.
        """
        # TODO: traversal include parallel execution
        traversal = []

        for el in self.workflow:
            print(f'Traversing layer {el}')
            if el in self.layers:
                traversal.extend(self.layers[el].traverse(self.nodes))
            elif el in self.nodes:
                traversal.append((self.nodes[el], None))
            else:
                raise ValueError(f"{el} not found.")

        return traversal


def parse_pipeline(yaml: dict) -> Pipeline:
    """
        Parse the yaml file into piepline

        :param yaml: dictionary from the yaml file
        :return: Pipeline object
    """

    return Schema(And({'nodes': _nodes, 'layers': _layers, 'workflow': list},
                      Use(lambda pipe: Pipeline(**pipe)))).validate(yaml)
