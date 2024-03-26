from typing import Union, List, Dict

from tenet.core.openai.helpers import remove_links, remove_html_tags, remove_code_blocks

REQUEST_TOKENS_LIMIT = 2000
GET_REPOSITORY_SOFTWARE_TYPE = [{"role": "system",
                                 "content": "Given the name, description, and read me of a repository, identify its "
                                            "Software Type. The Software Type can be either: Web Application, Utility, "
                                            "Server, Operating System, Browser, Framework, or Middleware. 'Web "
                                            "Application' are web-based/cloud-based software used for content "
                                            "management, information management, transactions, and more. 'Utility' are "
                                            "standalone apps including productivity tools, creativity software, "
                                            "antivirus programs, scripts, and non-web clients. 'Server' are system "
                                            "servers such as cloud-based, database, email, proxy, web, FTP, DNS, load "
                                            "balancers, and network monitors. 'Operating System' are firmware, device "
                                            "drivers, virtual machines, and all types of operating systems. 'Browser' "
                                            "are all types of web browsers used to access and navigate websites and "
                                            "web applications. 'Framework' are software components like libraries, "
                                            "plugins, and extensions that provide a foundation for building "
                                            "applications or systems. 'Middleware' are enterprise transaction "
                                            "platforms like message queuing, object storage, and identity management "
                                            "systems that facilitate communication between different software. "
                                            "Respond in this format: {'software_type': 'value'}"}
                                ]

LABEL_PATCH_ROOT_CAUSE = [{"role": "system",
                           "content": "Your goal is to identify the improper state within the vulnerable code and "
                                      "pinpoint the specific element causing the vulnerability. An improper state is "
                                      "defined by an (operation, operand_1, ···, operand_n) tuple, for which at least "
                                      "one element is improper. The process involves analyzing the provided code "
                                      "snippet and determining the operation and its associated operands in the "
                                      "vulnerable code. The code lines starting with '-' refer to a deletion and '+' "
                                      "to an addition. The response must be formatted into a concise tuple structure, "
                                      "including the index of the improper element in the tuple (numeration starts at "
                                      "0), followed by the operation and any operands involved in the improper state. "
                                      "This format would be: (index_of_improper_element, '{operation}', '{operand_1}',"
                                      " ..., '{operand_n}') "
                           }]

# todo: make the prompt label with dif changes and return operation, sink, improper operand
# try defining what operation, sink, improper operand are 
# understand how to make the instruction to focus on the vulnerable code/deletions and how to use 
# the additions/fix to capture what is wrong in the vulnerable code, fix contains the edit operations,
# for example, method replacement/operation (html() -> text()), this tells that html should not be used with unsanitized
# user input/ improper operand.
 
 # todo: provide additions and deletions and make the LLM reason about it.
LABEL_PATCH_ROOT_CAUSE_CHANGES = [{"role": "system",
                           "content": "Your goal is to identify the operation, sink, and source. "
                                      "operation is the action that inserts/appends the source to the sink, which can bring vulnerability to the sink. "
                                      "sink is the DOM element where the vulnerability manifests."
                                      "source is the tainted variable that can have dangerous element, "
                                      "and if these are rendered into the sink by the means of the operation, " 
                                      "it can enable the vulnerability. "
                                      "The text below shows the deletions and additions. "
                                      "deletions are vulnerable code that were deleted and replaced with the code in the additions to fix the vulnerability. "
                                      "The response must be formatted into a concise tuple structure of code in the, "
                                      "This format would be: (operation, sink, source ). Note that operation, sink, source are all code in the deletions. "
                           }]


def get_repository_software_type(name: str, description: str, read_me: str, margin: int = 5, prompt: bool = False,
                                 description_size_limit: int = 450, max_allowed_size: int = 450) \
        -> Union[str, List[Dict[str, str]]]:
    messages = GET_REPOSITORY_SOFTWARE_TYPE.copy()
    prompt_size = len(messages[0]['content'])
    max_completion_size = 50

    if len(description) > description_size_limit:
        description = description[:description_size_limit]

    content = f"Name: {name}\nDescription: {description}"
    allowed_size = REQUEST_TOKENS_LIMIT - prompt_size - max_completion_size - len(content) - margin

    # Remove code blocks from read me
    read_me = remove_code_blocks(read_me)
    # Remove HTML elements from read me
    read_me = remove_html_tags(read_me)
    # Remove URLs from the read me
    read_me = remove_links(read_me)

    if allowed_size > max_allowed_size:
        allowed_size = max_allowed_size

    if len(read_me) > allowed_size:
        read_me = read_me[:allowed_size]

    content += f"\nRead me: {read_me}"
    messages.append({"role": "user", "content": content})

    if not prompt:
        return messages

    return "\n".join([f"{m['content']}" for m in messages])


def label_patch_root_cause(diff: str, cwe_id: str, margin: int = 5, prompt: bool = False, max_allowed_size: int = 1000):
    messages = LABEL_PATCH_ROOT_CAUSE.copy()
    task_msg = f"This task entails examining the code patch for a {cwe_id} vulnerability."
    messages[0]['content'] = task_msg + " " + messages[0]['content']
    content = f"The code patch is the following:\n`{diff}`\nRespond only with the tuple in the specified format."
    prompt_size = len(messages[0]['content']) + len(content)
    max_completion_size = 100

    allowed_size = REQUEST_TOKENS_LIMIT - prompt_size - max_completion_size - len(diff) - margin

    if allowed_size > max_allowed_size:
        raise ValueError(f"Request size exceeds the maximum allowed size of {REQUEST_TOKENS_LIMIT} characters")

    messages.append({"role": "user", "content": content})
    print("messages")
    print(messages)
    if not prompt:
        return messages

    return "\n".join([f"{m['content']}" for m in messages])


# todo: 
# make this work with changes and new prompt
def label_patch_root_cause_changes(additions: list, deletions: list, cwe_id: str, margin: int = 5, prompt: bool = False, max_allowed_size: int = 1000):
    messages = LABEL_PATCH_ROOT_CAUSE_CHANGES.copy()
    task_msg = f"This task entails examining the code patch for a {cwe_id} vulnerability."
    messages[0]['content'] = task_msg + " " + messages[0]['content']
    changes =  "**Deletions**:" + " ".join([line for line in deletions]) + "\n**Additions**:\n" + " ".join([line for line in additions])
    content = f"The additions and deletions are:\n`{changes}`\nRespond only with the tuple in the specified format."
    
    
    prompt_size = len(messages[0]['content']) + len(content)
    max_completion_size = 100

    allowed_size = REQUEST_TOKENS_LIMIT - prompt_size - max_completion_size - len(changes) - margin

    if allowed_size > max_allowed_size:
        raise ValueError(f"Request size exceeds the maximum allowed size of {REQUEST_TOKENS_LIMIT} characters")

    messages.append({"role": "user", "content": content})
    if not prompt:
        return messages

    return "\n".join([f"{m['content']}" for m in messages])
