import json
from pathlib import Path


def to_offset(data_file: Path, output_file: Path):
    offset_dict = {}

    with data_file.open(mode='rb') as f, output_file.open(mode="w") as out:
        number_of_lines = 0
        for _ in f:
            number_of_lines += 1
        print(number_of_lines)

        f.seek(0)
        for lineno in range(number_of_lines):
            offset_dict[lineno] = f.tell()
            f.readline()

        json.dump(offset_dict, out)
