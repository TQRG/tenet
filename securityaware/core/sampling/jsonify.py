from typing import Tuple

import jsonlines
import csv
from pathlib import Path
from tqdm import tqdm
import re

labels_dict = {
    "safe": 0,
    "unsafe": 1
}

comment_regex = r"(([\"'])(?:\\[\s\S]|.)*?\2|\/(?![*\/])(?:\\.|\[(?:\\.|.)\]|.)*?\/)|\/\/.*?$|\/\*[\s\S]*?\*\/"
subst = "\\1"
# TODO: fix the blacklist problem by removing comments from code in a different way
blacklist = ["vis.js", "highlight.js", "swagger-ui-bundle.js", "dhtmlx.js", "worker-lua.js", "worker-javascript.js",
             "jquery.inputmask.bundle.js", "jquery-ui.js", "highlight.pack.js", "docs/jquery.js", "c3.min.js",
             "customize-controls.min.js"]


def to_dict(data_file: Path, replace: Tuple[str, str]):
    functions_dict = {}

    with data_file.open(mode='r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in tqdm(reader):
            print(row)
            fpath = Path(row["fpath"].replace(replace[0], replace[1]))
            print(fpath)
            func_features = [int(row["sline"]), int(row["scol"]), int(row["eline"]), int(row["ecol"]), row["label"]]

            if fpath not in functions_dict:
                functions_dict[fpath] = [func_features]
            else:
                functions_dict[fpath].append(func_features)

    return functions_dict


def to_json(files_path: Path, functions: dict, output_filename: Path):
    with jsonlines.open(output_filename, mode='a') as output_file:
        for filename in tqdm(functions):
            target_file = files_path / filename

            if not target_file.exists():
                print(f"{target_file} not found")
                continue

            with target_file.open(encoding="latin-1") as code_file:
                print(f"Processing {filename}...")
                lines = code_file.readlines()
                for i, function in enumerate(functions[filename]):
                    print(f"\tProcessing function {i+1}")
                    sline = function[0]
                    scol = function[1]
                    eline = function[2]
                    ecol = function[3]
                    label = function[4]

                    code = lines[(sline - 1):eline]

                    if (len(code) > 1):
                        code[0] = code[0][scol:]
                        code[-1] = code[-1][:ecol]

                    else:
                        code[0] = code[0][scol:ecol]
                    code = ''.join(code)

                    if (not any(s in str(filename) for s in blacklist)) and (
                            not all(s in str(filename) for s in ["config", "mhjx_MhJx"])):
                        print("regex")
                        code = re.sub(comment_regex, subst, code, 0, re.MULTILINE)

                    output_file.write({"code": code,
                                       # "id": id,
                                       "label": labels_dict[label]  # ,
                                       # "hash": hashlib.md5(code.encode()).hexdigest()
                                       })
