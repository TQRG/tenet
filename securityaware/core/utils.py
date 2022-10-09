# Remove comments, annotations, import and empty lines from the content of a file
import re


def clean_code(code: str):
    """
        Removes empty lines, inline comments, single and multi line comments in C/C++/JS code.
    """
    code = code.encode("utf-8", "replace").decode("utf-8")

    # Remove multi-line comments
    pattern = r"^\s*/\*(.*?)\*/"
    code = re.sub(pattern, "", code, flags=re.DOTALL | re.MULTILINE)

    # Remove inline comments within /* and */
    pattern = r"/\*(.*?)\*/"
    code = re.sub(pattern, "", code)
    # TODO: Make this work
    # Remove single inline comments //
    pattern = r";\s*/\/\\s*.*"
    code = re.sub(pattern, "", code)
    # Remove single-line comments
    pattern = r"^\s*//.*"
    code = re.sub(pattern, "", code, flags=re.MULTILINE)

    # Remove empty lines
    pattern = r"^\s*[\r\n]"
    code = re.sub(pattern, "", code, flags=re.MULTILINE)

    return code
