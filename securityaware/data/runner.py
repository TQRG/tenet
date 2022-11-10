from dataclasses import dataclass, field
from datetime import datetime
from typing import List, AnyStr, Any, Dict


@dataclass
class Store:
    assets: dict = field(default_factory=lambda: {})

    def __getitem__(self, key: str):
        return self.assets[key]

    def __setitem__(self, key: str, value):
        self.assets[key] = value

    def __iter__(self):
        return iter(self.assets)

    def keys(self):
        return self.assets.keys()

    def items(self):
        return self.assets.items()

    def values(self):
        return self.assets.values()


@dataclass
class Task(Store):
    id: int = None
    status: str = None
    start_date: datetime = None
    end_date: datetime = None
    err: AnyStr = None
    result: Any = None

    def update_id(self, idx: int):
        self.id = idx

    def to_dict(self):
        return {'id': self.id, 'status': self.status, 'start_date': self.start_date, 'end_date': self.end_date,
                'err': self.err, 'result': self.result}

    def duration(self):
        if self.start_date and self.end_date:
            return (self.end_date - self.start_date).total_seconds()
        return 0

    def has_started(self):
        return self.status == "Started"

    def start(self):
        self.start_date = datetime.now()
        self.status = "Started"

    def wait(self):
        self.status = "Waiting"

    def error(self, msg: AnyStr):
        self.status = "Error"
        self.err = msg
        self.end_date = datetime.now()

    def done(self):
        if self.has_started():
            self.status = "Done"
        else:
            self.status = "Finished"

        self.end_date = datetime.now()


@dataclass
class Runner:
    pointer: int = 1
    tasks: List[Task] = field(default_factory=lambda: [])
    finished: List[Task] = field(default_factory=lambda: [])
    running: Dict[int, Task] = field(default_factory=lambda: {})
    waiting: List[Task] = field(default_factory=lambda: [])

    def __len__(self):
        return len(self.tasks)

    def results(self, skip_none: bool = True):
        return [task.result for task in self.finished if not (skip_none and (task.result is None))]

    def reset(self):
        self.tasks = []
        self.pointer = 1
        self.finished = []
        self.running = {}
        self.waiting = []

    def add(self, task: Task):
        task.id = self.pointer
        self.tasks.append(task)
        self.pointer += 1

    def done(self, task: Task):
        task.done()
        self.finished += [task]
        del self.running[task.id]
