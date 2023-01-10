from dataclasses import dataclass


@dataclass
class Plugin:
    name: str
    loaded: bool
