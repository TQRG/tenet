import time
import openai

from typing import Union
from cement import Handler

from tenet.core.interfaces import HandlersInterface
from tenet.core.openai.prompts import get_repository_software_type, label_patch_root_cause
from tenet.core.openai.helpers import extract_dictionary


class OpenAIHandler(HandlersInterface, Handler):
    class Meta:
        label = 'openai'

    def __init__(self, **kw):
        super().__init__(**kw)
        self.timed_out = False
        self._models = None

    @property
    def models(self):
        if self._models is None:
            self._models = [m.id for m in openai.models.list()]

        return self._models

    def handle_rate_limit_error(self, timeout: float = 60.0):
        if self.timed_out:
            self.app.log.warning(f"API exhausted. Terminating.")
            exit(1)

        self.timed_out = True
        self.app.log.warning(f"Rate limit hit. Sleeping for {timeout} seconds.")
        time.sleep(timeout)

    def _check_model(self, model: str) -> bool:
        if model not in self.models:
            print(f"Model {model} not found.")
            return False
        return True

    def create_chat_completion(self, model: str, messages: list, max_tokens: int = 100, delay: float = 1.5, **kwargs) \
            -> Union[openai.ChatCompletion, None]:

        if not self._check_model(model):
            return None

        completion = None

        try:
            completion = openai.chat.completions.create(model=model, max_tokens=max_tokens, n=1, messages=messages,
                                                        **kwargs)
        except openai.RateLimitError as e:
            self.app.log.error(f"Rate limit error: {e}")
            self.handle_rate_limit_error(delay)
        except openai.OpenAIError as e:
            self.app.log.error(f"Error while generating text: {e}")
        except ValueError as e:
            self.app.log.error(f"Error while generating text: {e}")

        if delay:
            time.sleep(delay)

        return completion

    def _setup(self, app):
        import os
        super(OpenAIHandler, self)._setup(app)
        self.app.log.info(f"Setting up OpenAIHandler...")
        try:
            openai.api_key = os.environ['OPENAI_TOKEN']
        except AttributeError:
            self.app.log.error(f"OpenAI token not set. Provide the token in the OPENAI_TOKEN environment variable.")
            exit(1)

    def label_diff(self, model: str, diff: str, cwe_id: str, delay: float = 1.5, **kwargs) \
            -> Union[openai.ChatCompletion, None]:
        messages = label_patch_root_cause(diff=diff, cwe_id=cwe_id)
        return self.create_chat_completion(model=model, messages=messages, max_tokens=100, delay=delay, **kwargs)

    def generate_software_type(self, model: str, name: str, description: str, read_me: str, delay: float = 1.5) \
            -> Union[str, None]:
        messages = get_repository_software_type(name=name, description=description, read_me=read_me)
        completion = self.create_chat_completion(model=model, messages=messages, max_tokens=50, delay=delay)

        if completion:
            result = extract_dictionary(completion.choices[0].text)
            if result:
                for column in ['software_type', 'Software Type', 'Software_Type', 'software type', 'Software type',
                               'Software_type', 'software Type', 'software_Type']:
                    if column in result:
                        return result[column]

        return None
