from dataclasses import dataclass, field
from typing import Dict, List, AnyStr

from schema import Schema, And, Use, Optional, Or


@dataclass
class Cell:
    """
        Basic operation unit
    """
    name: str
    force: bool


@dataclass
class Plugin(Cell):
    """
        Code extensions
    """
    label: str
    args: dict = field(default_factory=lambda: {})


@dataclass
class Container(Cell):
    """
        Docker container
    """
    image: str
    cmds: List[AnyStr]


@dataclass
class Layer:
    """
        Sequential set of plugins/containers
    """
    input: str
    cells: List[Cell]


@dataclass
class Pipeline:
    prepare: Dict[str, Cell]
    model: dict

    @property
    def nodes(self):
        nodes = {}

        if self.prepare:
            nodes['prepare'] = self.prepare

        if self.model:
            nodes['model'] = self.model

        return nodes


def parse_pipeline(yaml: dict) -> Pipeline:
    """
        Parse the yaml file into piepline

        :param yaml: dictionary from the yaml file
        :return: Pipeline object
    """
    container = Schema(And({'name': str, 'image': str, Optional('force', default=False): bool, 'cmds': [str]},
                           Use(lambda c: Container(**c))))
    plugin = Schema(And({'label': str, 'name': str, Optional('force', default=False): bool,
                         Optional('args', default={}): dict}, Use(lambda p: Plugin(**p))))
    layers = Schema(
        And(
            {str: {
                'input': str,
                'cells': And([Or({'container': container}, {'plugin': plugin})],
                             Use(lambda cells: [v for c in cells for k, v in c.items()]))
            }}, Use(lambda l: {k: Layer(**v) for k, v in l.items()})))

    return Schema(And({'prepare': layers, Optional('model', default=None): dict},
                      Use(lambda pipe: Pipeline(**pipe)))).validate(yaml)
