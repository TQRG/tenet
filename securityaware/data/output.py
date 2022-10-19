from typing import AnyStr, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class CommandData:
    # env: dict = None
    args: str
    return_code: int = 0
    duration: float = 0
    start: datetime = None
    end: datetime = None
    output: AnyStr = None
    error: AnyStr = None
    timeout: bool = False
    parsed_output: Any = None
    tag: str = None
