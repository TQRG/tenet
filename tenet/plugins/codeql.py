import ast

import pandas as pd
import jq
import json
import tqdm

from typing import Union, Any
from pathlib import Path

from tenet.core.exc import TenetError
from tenet.data.schema import ContainerCommand
from tenet.handlers.plugin import PluginHandler


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
        self.rule_id_cwe_mapping = {'js/xss-through-dom': 'CWE-79', 'js/untrusted-data-to-external-api': 'CWE-20',
                                    'js/html-constructed-from-input': 'CWE-79', 'js/xss': 'CWE-79',
                                    'js/regex/missing-regexp-anchor': 'CWE-20', 'js/unsafe-jquery-plugin': 'CWE-79',
                                    'js/path-injection': 'CWE-22', 'js/incomplete-url-scheme-check': 'CWE-20',
                                    'js/useless-regexp-character-escape': 'CWE-20', 'js/overly-large-range': 'CWE-20',
                                    'js/incomplete-url-substring-sanitization': 'CWE-20', 'js/reflected-xss': 'CWE-79',
                                    'js/biased-cryptographic-random': 'CWE-327', 'js/server-crash': 'CWE-730',
                                    'js/remote-property-injection': 'CWE-400', 'js/unnecessary-use-of-cat': 'CWE-78',
                                    'js/client-side-unvalidated-url-redirection': 'CWE-601', 'js/stored-xss': 'CWE-79',
                                    'js/disabling-certificate-validation': 'CWE-295', 'js/request-forgery': 'CWE-918',
                                    'js/server-side-unvalidated-url-redirection': 'CWE-601', 'js/zipslip': 'CWE-22',
                                    'js/incorrect-suffix-check': 'CWE-20', 'js/weak-cryptographic-algorithm': 'CWE-327',
                                    'js/file-access-to-http': 'CWE-200', 'js/exposure-of-private-files': 'CWE-200',
                                    'js/code-injection': 'CWE-94'
                                    }

        self.cwes = [20, 22, 73, 78, 79, 89, 94, 1004, 116, 117, 1275, 134, 178, 200, 201, 209, 295, 300, 312,
                     313, 326, 327, 338, 346, 347, 352, 367, 377, 384, 400, 451, 502, 506, 598, 601, 611, 614, 640,
                     643, 730, 754, 770, 776, 798, 807, 829, 830, 834, 843, 862, 912, 915, 916, 918]
        self.parent_files = True
        self.fix_files = True
        self.drop_unavailable = False
        self.drop_cwes = []

    def set_sources(self):
        self.set('report_file_path', self.path / f"{self.output.stem}_report.json")
        self.set('codeql_db_path', self.container_handler.working_dir / 'db')

    def get_sinks(self):
        self.get('raw_files_path')

    def parse_language(self, language: str) -> str:
        if language.lower() in self.language_mapping.values():
            return language.lower()

        if language.lower() in self.language_mapping:
            return self.language_mapping[language.lower()]

        raise TenetError(f"Programming language {language} not available.")

    def parse_cwe(self, cwe: str) -> Union[str, None]:
        if 'CWE-' in cwe.upper():
            cwe_number = int(cwe.upper().split('CWE-')[-1])
        else:
            cwe_number = int(cwe)

        if cwe_number not in self.cwes:
            if not self.drop_unavailable:
                raise TenetError(f"CWE-{cwe_number} not available.")
            return None

        return f'CWE-{cwe_number:03}'

    def parse_target_cwes(self, cwes: Any):
        cwes = ast.literal_eval(cwes)

        if not isinstance(cwes, str):
            return [self.parse_cwe(cwe) for cwe in cwes]

        return [self.parse_cwe(cwes)]

    def run(self, dataset: pd.DataFrame, image_name: str = 'codeql', language: str = 'javascript',
            target_cwes: list = None, parent_files: bool = True, fix_files: bool = True, drop_unavailable: bool = False,
            drop_cwes: list = None, **kwargs) -> Union[pd.DataFrame, None]:
        """
            run CodeQL and extracts the labels from its report
            :param dataset: dataset with diff blocks
            :param image_name: name of the CodeQL image
            :param language: programming language of target code to scan
            :param target_cwes: list of CWEs to analyze for
            :param parent_files: flag to include warnings for parent files
            :param fix_files: flag to include warnings for fix files
            :param drop_unavailable: flag to drop target CWEs not covered by CodeQL
            :param drop_cwes: list of cwes to drop, in case codeql raises some error for them
        """
        report_file_container = self.container_handler.working_dir / f"{self.output.stem}_report.json"

        self.drop_unavailable = drop_unavailable
        if drop_cwes:
            for cwe in drop_cwes:
                if cwe in self.cwes:
                    self.cwes.remove(cwe)


        raw_files_path = Path(str(self.sinks['raw_files_path']).replace(str(self.app.workdir), str(self.app.bind)))

        if not target_cwes:
            target_cwes = dataset.label.unique().tolist()

        if not self.sources['report_file_path'].exists():
            # TODO: fix the node name
            container = self.container_handler.run(image_name=image_name)
            language = self.parse_language(language)
            cwes = [parsed_cwe for cwe in target_cwes for parsed_cwe in self.parse_target_cwes(cwe) if parsed_cwe is not None]
            create_cmd = ContainerCommand(org=f"codeql database create {self.sources['codeql_db_path']}")
            create_cmd.org += f" --threads={self.app.threads} --language={language} --source-root={raw_files_path} 2>&1"
            analyze_cmd = ContainerCommand(org=f"codeql database analyze {self.sources['codeql_db_path']} --format=sarif-latest")
            analyze_cmd.org += f" --threads={self.app.threads} --output={report_file_container}"

            for cwe in cwes:
                analyze_cmd.org += f" /codeql-home/codeql-repo/javascript/ql/src/Security/{cwe}"

            analyze_cmd.org += f" 2>&1"
            self.container_handler.run_cmds(container.id, [create_cmd, analyze_cmd])
            self.container_handler.stop(container)

        if not self.sources['report_file_path'].exists():
            self.app.log.warning(f"CodeQL report file not found.")
            return None

        with self.sources['report_file_path'].open(mode='r') as rf:
            json_report = json.loads(rf.read())
            result = jq.all(r'.runs[0].results', json_report)
            data_points = jq.iter(r'.[]', result[0])

            for data_point in tqdm.tqdm(data_points):
                self.multi_task_handler.add(data_point=data_point)

            self.multi_task_handler(func=self.parse_data_point)

        new_dataset = self.multi_task_handler.results()

        if new_dataset:
            df = pd.DataFrame.from_dict(new_dataset)
            commits = []

            if parent_files:
                commits.extend(dataset['a_version'].to_list())
            if fix_files:
                commits.extend(dataset['b_version'].to_list())

            return df[df.version.isin(commits)]

        return None

    def parse_data_point(self, data_point):
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

            owner, project, version, *file = fpath.replace(str(self.app.bind) + '/', '').split('/')
            cwe_id = self.rule_id_cwe_mapping.get(rule_id, None)

            if cwe_id is None:
                self.app.log.warning(f"No cwe found for rule {rule_id}")
                cwe_id = 'unsafe'

            return {'owner': owner, 'project': project, 'version': version, 'fpath': '/'.join(file), 'sline': sline,
                    'scol': scol, 'eline': eline, 'ecol': ecol, 'label': cwe_id, 'rule_id': rule_id,
                    'pair_hash': None}

        return None


def load(app):
    app.handler.register(CodeQLExtractLabelsHandler)
