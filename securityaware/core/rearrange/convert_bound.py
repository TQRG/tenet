import re
import subprocess

from typing import List, Union

from securityaware.data.diff import InlineDiff

fn_bound_pattern = re.compile(r"[0-9]+,[0-9]+,[0-9]+,[0-9]+$")


def transform_inline_diff(code_path: str, inline_diff: InlineDiff,
                          transform_path: str = "transforms/outputFnBoundary") -> List[str]:
    # Set -d option for dry-run to avoid corrupting code files
    loc_arg = f"--loc={inline_diff.sline},{inline_diff.scol},{inline_diff.eline},{inline_diff.ecol}"
    bound_result = subprocess.run(["jscodeshift", "-p", "-s", "-d", "-t", transform_path, code_path, loc_arg],
                                  stdout=subprocess.PIPE)

    return bound_result.stdout.decode("utf-8").splitlines()


def parse_fn_bound(fn_bound_str: List[str], inline_diff: InlineDiff) -> Union[InlineDiff, None]:
    if fn_bound_pattern.match(fn_bound_str[0]) is not None:
        fn_boundary = fn_bound_str[0].split(",")

        return InlineDiff(project=inline_diff.project, file_path=inline_diff.file_path, sline=int(fn_boundary[0]),
                          scol=int(fn_boundary[1]), eline=int(fn_boundary[2]), ecol=int(fn_boundary[3]),
                          label=inline_diff.label)
    return None
