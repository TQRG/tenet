import pandas as pd
import openai

from typing import Union
from ast import literal_eval

from tenet.handlers.plugin import PluginHandler
from arepo.models.data import DatasetModel, WeaknessModel, CompletionModel
from arepo.models.common.vulnerability import VulnerabilityModel


def is_tuple(content: str):
    # check if the completion is a tuple
    try:
        improper_state = literal_eval(content)

        if isinstance(improper_state, tuple):
            return True
    except ValueError as e:
        pass

    return False


class AILabeler(PluginHandler):
    """
        AILabeler plugin
    """

    class Meta:
        label = "ai_labeler"

    def __init__(self, **kw):
        super().__init__(**kw)

    def set_sources(self):
        pass

    def get_sinks(self):
        pass

    def run(self, dataset: DatasetModel, **kwargs) -> Union[pd.DataFrame, None]:
        session = self.app.db.get_session()
        weaknesses = []

        for vuln in dataset.vulnerabilities:
            self.app.log.info(f"Processing vulnerability {vuln.id}")
            # skip if in weaknesses table
            vuln_weaknesses = session.query(WeaknessModel).filter(WeaknessModel.vulnerability_id == vuln.id).all()

            if vuln_weaknesses:
                self.app.log.info(f"Vulnerability {vuln.id} already processed")
                weaknesses.extend(vuln_weaknesses)
                continue

            cwe_id = self.get_vuln_cwe(vuln)
            patch = self.get_patches(vuln)

            if not cwe_id or not patch:
                self.app.log.error(f"Vulnerability {vuln.id} has no CWE or patch")
                continue

            completion = self.openai_handler.label_diff(model='gpt-3.5-turbo', diff=patch, cwe_id=f"CWE-{cwe_id}")

            if not completion:
                self.app.log.error(f"Vulnerability {vuln.id} has no completion")
                continue

            self.save_completion(completion)

            if is_tuple(completion.completion):
                weakness = WeaknessModel(completion_id=completion.id, vulnerability_id=vuln.id,
                                         tuple=completion.completion)
                weakness.save()
                weaknesses.append(weakness)
            else:
                self.app.log.error(f"Completion's {completion.id} text is not a tuple")

        return weaknesses

    def get_vuln_cwe(self, vulnerability_model: VulnerabilityModel) -> Union[int, None]:
        if not vulnerability_model.cwes:
            self.app.log.error(f"Vulnerability {vulnerability_model.id} has no CWE associated")
            return None

        if len(vulnerability_model.cwes) > 1:
            self.app.log.error(f"Vulnerability {vulnerability_model.id} has more than one CWE associated")
            return None

        return vulnerability_model.cwes[0].id

    def get_patches(self, vulnerability_model: VulnerabilityModel) -> Union[str, None]:
        # TODO: update to make all the checks dynamic and return a list of patches

        commits = [c for c in vulnerability_model.commits if c.kind != 'parent']

        if not commits:
            self.app.log.error(f"Vulnerability {vulnerability_model.id} has no commits associated")
            return None

        if len(commits) > 1:
            self.app.log.error(f"Vulnerability {vulnerability_model.id} has more than one commit associated")
            return None

        commit_files = [cf for cf in commits[0].files]

        if not commit_files:
            self.app.log.error(f"Vulnerability {vulnerability_model.id} has no patch file associated")
            return None

        if len(commit_files) > 1:
            self.app.log.error(f"Vulnerability {vulnerability_model.id} has more than one patch file associated")
            return None

        patch = commit_files[0].patch

        if not patch:
            self.app.log.error(f"Vulnerability {vulnerability_model.id} has no patch diff associated")
            return None

        if len(patch) > 1000:
            self.app.log.error(f"Vulnerability {vulnerability_model.id} has a patch diff larger than 1000 characters")
            return None

        return patch

    def save_completion(self, completion: openai.ChatCompletion) -> CompletionModel:
        session = self.app.db.get_session()
        completion_model = CompletionModel(id=completion.id, model=completion.model, object=completion.object,
                                           created=completion.created, prompt=str(completion.message),
                                           completion=completion.completion, finish_reason=completion.finish_reason,
                                           prompt_tokens=completion.usage["prompt_tokens"],
                                           total_tokens=completion.usage["total_tokens"],
                                           completion_tokens=completion.usage["completion_tokens"])
        session.add(completion_model)
        session.commit()

        return completion_model

    def plot(self, dataset: pd.DataFrame, **kwargs):
        pass


def load(app):
    app.handler.register(AILabeler)
