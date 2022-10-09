from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Dataset:
    rows: List = field(default_factory=lambda: [])

    def __len__(self):
        return len(self.rows)

    def __str__(self):
        return '\n'.join([str(row) for row in self.rows])

    def add(self, row):
        self.rows.append(row)

    def write(self, path: Path):
        with path.open(mode="w", newline='') as dataset_file:
            for sample in self.rows:
                dataset_file.write(f"{sample}\n")


@dataclass
class XYDataset:
    x: Dataset = Dataset()
    y: Dataset = Dataset()

    def __len__(self):
        return len(self.x)

    def __str__(self):
        return f"x:{self.x}\ny:{self.y}"

    def add(self, x_row, y_row):
        self.x.add(x_row)
        self.y.add(y_row)

    def write(self, path: Path, delimiter: str, offset: bool, headers: str = None):
        with path.open(mode="w", newline='') as dataset_file:
            if headers:
                dataset_file.write(headers + "\n")
            for x, y in zip(self.x.rows, self.y.rows):
                if offset:
                    dataset_file.write(f"{y},{delimiter.join(x)}\n")
                else:
                    dataset_file.write(f"{delimiter.join(x)}\n")

        return path
