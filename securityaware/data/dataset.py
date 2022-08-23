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

    def write(self, path: Path, delimiter: str, headers: str = None):
        with path.open(mode="w", newline='') as dataset_file:
            if headers:
                dataset_file.write(headers + "\n")
            for x, y in zip(self.x.rows, self.y.rows):
                #dataset_file.write(f"{y}{delimiter}{x}\n")
                dataset_file.write(f"{delimiter.join(x)}\n")

        return path


@dataclass
class SplitDataset:
    train: XYDataset
    val: XYDataset
    test: XYDataset

    def write(self, out_dir: Path, delimiter: str, headers: str = None):
        print(len(self.train), len(self.test), len(self.val))
        return self.train.write(out_dir / "train.raw.txt", delimiter, headers), \
               self.val.write(out_dir / "validation.raw.txt", delimiter, headers), \
               self.test.write(out_dir / "test.raw.txt", delimiter, headers)
