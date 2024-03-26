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


class AILabelerChanges(PluginHandler):
    """
        AILabelerChanges plugin
    """

    class Meta:
        label = "ai_labeler_changes"

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

            # if vuln_weaknesses:
            #     self.app.log.info(f"Vulnerability {vuln.id} already processed")
            #     weaknesses.extend(vuln_weaknesses)
            #     continue

            cwe_id = self.get_vuln_cwe(vuln)
            changes = self.get_changes(vuln)
            print("changes")
            print(changes)

            if not cwe_id or not changes:
                self.app.log.error(f"Vulnerability {vuln.id} has no CWE or changes")
                continue


            # todo: take from changes(additions and deletions) and pass them to the prompt
            # additions = [ c for c in changes if c.type == "addition"]
            # deletions = [ c for c in changes if c.type == "deletion"]

            additions = [ c.content for c in changes if c.type == "addition"]
            deletions = [ c.content for c in changes if c.type == "deletion"]
            print("add")
            print(additions)
            prompt, completion = self.openai_handler.label_changes(model='gpt-3.5-turbo', additions = additions, deletions = deletions, cwe_id=f"CWE-{cwe_id}")

            if not completion:
                self.app.log.error(f"Vulnerability {vuln.id} has no completion")
                continue

            self.save_completion(completion,prompt)

            gpt_reply_content = completion.choices[0].message.content.replace("`", "'")

            if is_tuple(gpt_reply_content):
                weakness = WeaknessModel(completion_id=completion.id, vulnerability_id=vuln.id,
                                         tuple=gpt_reply_content)
                session.add(weakness)
                session.commit()
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

    def get_changes(self, vulnerability_model: VulnerabilityModel) -> Union[str, None]:
        commits = [c for c in vulnerability_model.commits if c.kind != 'parent']

        if not commits:
            self.app.log.error(f"Vulnerability {vulnerability_model.id} has no commits associated")
            return None

        if len(commits) > 1:
            self.app.log.error(f"Vulnerability {vulnerability_model.id} has more than one commit associated")
            return None
        print("commit files")
        print(commits)
        commit_files = [cf for cf in commits[0].files]

        if not commit_files:
            self.app.log.error(f"Vulnerability {vulnerability_model.id} has no patch file associated")
            return None

        if len(commit_files) > 1:
            self.app.log.error(f"Vulnerability {vulnerability_model.id} has more than one patch file associated")
            return None

        diff_blocks = commit_files[0].diff_blocks

        print("dif block")
        print(diff_blocks)

        if not diff_blocks:
            self.app.log.error(f"Vulnerability {vulnerability_model.id} has no diff_blocks associated")
            return None

        if len(diff_blocks) > 2:
            self.app.log.error(f"Vulnerability {vulnerability_model.id} has more than two  diff_blocks")
            return None
        
        changes = diff_blocks[0].changes
        print("changes")
        print(changes)

        if commit_files[0].additions== 0 or commit_files[0].deletions == 0:
            self.app.log.error(f"Vulnerability {vulnerability_model.id} has no addition or deletion")
            return None
        
        return changes

    def save_completion(self, completion: openai.ChatCompletion, prompt: str ) -> CompletionModel:
        session = self.app.db.get_session()
        print("completion")
        print(completion)
        
        completion_model = CompletionModel(id=completion.id, model=completion.model, object=completion.object,
                                           created=completion.created, prompt=prompt[0]["content"],
                                           completion=completion.choices[0].message.content,
                                           finish_reason=completion.choices[0].finish_reason,
                                           prompt_tokens=completion.usage.prompt_tokens,
                                           total_tokens=completion.usage.total_tokens,
                                           completion_tokens=completion.usage.completion_tokens)
        session.add(completion_model)
        session.commit()

        return completion_model

    def plot(self, dataset: pd.DataFrame, **kwargs):
        pass


def load(app):
    app.handler.register(AILabelerChanges)
