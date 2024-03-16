import ast
import re
import traceback
from typing import Union
from bs4 import BeautifulSoup


def remove_code_blocks(text):
    # Regular expression pattern to match code blocks
    code_block_pattern = re.compile(r'```[^```]+\```')

    # Remove code blocks from the text
    text_without_code_blocks = re.sub(code_block_pattern, '', text)

    return text_without_code_blocks


def remove_links(text):
    # Regular expression pattern to match URLs
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

    # Remove URLs from the text
    text_without_links = re.sub(url_pattern, '', text)

    return text_without_links


def remove_html_tags(text):
    # Create a BeautifulSoup object from the input text
    soup = BeautifulSoup(text, 'html.parser')

    # Remove all HTML tags from the parsed text
    text_without_tags = soup.get_text()

    return text_without_tags


def extract_dictionary(content: str) -> Union[dict, None]:
    try:
        # match dictionary in the message content string
        matches = re.findall(r'(\{[^{}]+\})', content)

        # and convert it to a dictionary
        if matches:
            try:
                response = ast.literal_eval(matches[0])

            except (SyntaxError, ValueError):
                return None

            # put every key in lower case
            return {k.lower(): v for k, v in response.items()}

        return None

    except Exception:
        print(traceback.format_exc())

        return None
