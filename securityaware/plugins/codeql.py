from pathlib import Path
from typing import Union

import pandas as pd
import jq
import json

from securityaware.core.exc import SecurityAwareError
from securityaware.data.schema import ContainerCommand
from securityaware.handlers.plugin import PluginHandler


class CodeQLExtractLabelsHandler(PluginHandler):
    """
        Separate plugin
    """
    class Meta:
        label = "codeql"

    def __init__(self, **kw):
        super().__init__(**kw)
        self.language_mapping = {'js': 'javascript', 'c++': 'cpp', 'c#': 'csharp', 'py': 'python', 'go': 'go',
                                 'java': 'java', 'rb': 'ruby', 'swift': 'swift'}
        self.cwes = [20, 22, 73, 78, 79, 89, 94, 1004, 116, 117, 1275, 134, 178, 200, 201, 209, 295, 300, 312,
                     313, 326, 327, 338, 346, 347, 352, 367, 377, 384, 400, 451, 502, 506, 598, 601, 611, 614, 640,
                     643, 730, 754, 770, 776, 798, 807, 829, 830, 834, 843, 862, 912, 915, 916, 918]

    def parse_language(self, language: str) -> str:
        if language.lower() in self.language_mapping.values():
            return language.lower()

        if language.lower() in self.language_mapping:
            return self.language_mapping[language.lower()]

        raise SecurityAwareError(f"Programming language {language} not available.")

    def parse_cwe(self, cwe: str):
        if 'CWE-' in cwe.upper():
            cwe_number = int(cwe.upper().split('CWE-')[-1])
        else:
            cwe_number = int(cwe)

        if cwe_number not in self.cwes:
            raise SecurityAwareError(f"CWE-{cwe_number} not available.")

        return f'CWE-{cwe_number:03}'

    def run(self, dataset: pd.DataFrame, image_name: str = 'codeql', language: str = 'javascript',
            target_cwes: list = None, parent_files_only: bool = True, add_fix_files: bool = True,
            **kwargs) -> Union[pd.DataFrame, None]:
        """
            run CodeQL and extracts the labels from its report
        """

        codeql_db_path = self.container_handler.working_dir / 'db'
        report_file = self.path / f"{self.output.stem}_report.json"
        report_file_container = self.container_handler.working_dir / f"{self.output.stem}_report.json"
        self.set('report_file_path', report_file)
        self.set('codeql_db_path', codeql_db_path)

        if not self.get('raw_files_path'):
            self.app.log.warning(f"raw files path not instantiated.")
            return None

        raw_files_path = Path(str(self.get('raw_files_path')).replace(str(self.app.workdir), str(self.app.bind)))

        if not target_cwes:
            target_cwes = dataset.label.unique().tolist()

        if not report_file.exists():
            # TODO: fix the node name
            container = self.container_handler.run(image_name=image_name, node_name=self.node.name)
            language = self.parse_language(language)
            cwes = [self.parse_cwe(cwe) for cwe in target_cwes]
            create_cmd = ContainerCommand(org=f"codeql database create {codeql_db_path}")
            create_cmd.org += f" --threads={self.app.threads} --language={language} --source-root={raw_files_path} 2>&1"
            analyze_cmd = ContainerCommand(org=f"codeql database analyze {codeql_db_path} --format=sarif-latest")
            analyze_cmd.org += f" --threads={self.app.threads} --output={report_file_container}"

            for cwe in cwes:
                analyze_cmd.org += f" /codeql-home/codeql-repo/javascript/ql/src/Security/{cwe}"

            analyze_cmd.org += f" 2>&1"
            self.container_handler.run_cmds(container.id, [create_cmd, analyze_cmd])
            self.container_handler.stop(container)

        if not report_file.exists():
            self.app.log.warning(f"CodeQL report file not found.")
            return None

        parent_commits = dataset['a_version'].to_list()
        missing_parent_commits = []
        new_dataset = []

        with report_file.open(mode='r') as rf:
            json_report = json.loads(rf.read())
            result = jq.all(r'.runs[0].results', json_report)

            for i, data_point in enumerate(jq.iter(r'.[]', result[0])):
                # TODO: run this in parallel
                self.app.log.info(f"Processing data point {i}")
                vuln_entry = jq.all(r'.', data_point)[0]
                fpath = jq.first(r'.locations[0].physicalLocation.artifactLocation.uri', vuln_entry)

                if fpath.endswith('.js'):
                    rule_id = jq.first(r'.ruleId', vuln_entry)
                    sline = jq.first(r'.locations[0].physicalLocation.region.startLine', vuln_entry)
                    scol = jq.first(r'.locations[0].physicalLocation.region.startColumn', vuln_entry)
                    eline = jq.first(r'.locations[0].physicalLocation.region.endLine', vuln_entry)
                    ecol = jq.first(r'.locations[0].physicalLocation.region.endColumn', vuln_entry)

                    if eline is None or eline == "null":
                        eline = sline

                    owner, project, version, *file = fpath.replace(str(self.app.bind)+'/', '').split('/')

                    if parent_files_only and (version not in parent_commits):
                        continue

                    # keep missing commits
                    parent_commits.remove(version)
                    new_dataset.append({'owner': owner, 'project': project, 'version': version, 'fpath': '/'.join(file),
                                        'sline': sline, 'scol': scol, 'eline': eline, 'ecol': ecol, 'label': 'unsafe',
                                        'rule_id': rule_id})

        if add_fix_files:
            for i, row in dataset.iterrows():
                new_dataset.append({'owner': row.owner, 'project': row.project, 'version': row['b_version'],
                                    'fpath': row['b_path'], 'sline': None, 'scol': None, 'eline': None,
                                    'ecol': None, 'label': 'safe', 'rule_id': None})

                if parent_files_only and (row['a_version'] in parent_commits):
                    new_dataset.append({'owner': row.owner, 'project': row.project, 'version': row['a_version'],
                                        'fpath': row['a_path'], 'sline': None, 'scol': None, 'eline': None,
                                        'ecol': None, 'label': 'safe', 'rule_id': None})

        if new_dataset:
            return pd.DataFrame(new_dataset)

        return None


def load(app):
    app.handler.register(CodeQLExtractLabelsHandler)
