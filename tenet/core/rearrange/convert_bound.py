import re
import subprocess

from typing import List, Union

from tenet.data.diff import InlineDiff

fn_bound_pattern = re.compile(r"[0-9]+,[0-9]+,[0-9]+,[0-9]+$")


def transform_inline_diff(code_path: str, inline_diff: InlineDiff,
                          transform_path: str = "transforms/outputFnBoundary") -> List[str]:
    # Set -d option for dry-run to avoid corrupting code files
    loc_arg = f"--loc={inline_diff.sline},{inline_diff.scol},{inline_diff.eline},{inline_diff.ecol}"
    bound_result = subprocess.run(["jscodeshift", "-p", "-s", "-d", "-t", transform_path, code_path, loc_arg],
                                  stdout=subprocess.PIPE)

    return bound_result.stdout.decode("utf-8").splitlines()
