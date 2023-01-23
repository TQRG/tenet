from pathlib import Path

import pandas as pd
from tqdm import tqdm

from typing import Union, Tuple

from tenet.core.diff_labeller.misc import check_or_create_dir
from tenet.core.exc import TenetError
from tenet.handlers.github import LocalGitFile
from tenet.handlers.plugin import PluginHandler


class Transform(PluginHandler):
    """
        Transform plugin
    """

    class Meta:
        label = "transform"

    def __init__(self, **kw):
        """
            runs the plugin
            Args:
            fix (bool): flag to add fix file
            snippet (bool): flag to transform samples to snippet granularity
        """
        super().__init__(**kw)
        self.fix = False

    def set_sources(self):
        self.set('clean_files_path', Path(self.path, 'files'))

    def get_sinks(self):
        self.get('files_path')

    def run(self, dataset: pd.DataFrame, fix: bool = False) -> Union[pd.DataFrame, None]:
        """
            runs the plugin
            Args:
            fix (bool): flag to add fix file
            snippet (bool): flag to transform samples to snippet granularity
        """
        check_or_create_dir(self.sources['clean_files_path'])
        self.fix = fix

        for row in tqdm(dataset.to_dict(orient='records')):
            self.multi_task_handler.add(row=row)

        self.multi_task_handler(func=self.transform)
        rows = self.multi_task_handler.results(expand=True)

        if rows:
            parsed = pd.DataFrame(rows)

            return parsed

        return None

    def get_triplet(self, row: dict) -> Tuple[LocalGitFile, LocalGitFile, LocalGitFile]:
        project_path = self.sinks['files_path'] / row['project']

        if 'raw_url_vuln' in row and not pd.isnull(row['raw_url_vuln']):
            vuln_file = LocalGitFile(url=row['raw_url_vuln'], short=Path(row['file_path']), tag='vuln',
                                     path=project_path / row['vuln_commit_hash'] / row['file_path'])
        else:
            vuln_file = None

        if 'raw_url_fix' in row and not pd.isnull(row['raw_url_fix']):
            fix_file = LocalGitFile(url=row['raw_url_fix'], short=Path(row['file_path']), tag='fix',
                                    path=project_path / row['fixed_commit_hash'] / row['file_path'])
        else:
            fix_file = None

        if 'raw_url_non_vuln' in row and not pd.isnull(row['raw_url_non_vuln']):
            non_vuln_file = LocalGitFile(url=row["raw_url_non_vuln"], short=Path(row['non_vuln_file_path']),
                                         tag='non-vuln',
                                         path=project_path / row['non_vuln_commit_hash'] / row['non_vuln_file_path'])
        else:
            non_vuln_file = None

        return vuln_file, fix_file, non_vuln_file

    def transform(self, row: dict):
        rows = []
        vuln_file, fix_file, non_vuln_file = self.get_triplet(row)

        if not vuln_file and not fix_file and not non_vuln_file:
            raise TenetError("Empty row")

        fix_file_content = self.get_clean_content(project=row['project'], commit=row['fixed_commit_hash'],
                                                  file=fix_file)
        vuln_file_content = self.get_clean_content(project=row['project'], commit=row['vuln_commit_hash'],
                                                   file=vuln_file)
        non_vuln_file_content = self.get_clean_content(project=row['project'], commit=row['non_vuln_commit_hash'],
                                                       file=non_vuln_file)

        vuln_str, size_vuln_lines, fix_str, size_fix_lines = self.code_parser_handler.get_pair_snippet(fix_file_content,
                                                                                                       vuln_file_content)
        non_vuln_str = self.code_parser_handler.get_non_vuln_snippet(non_vuln_file_content, size_fix_lines,
                                                                     size_vuln_lines)
        del row['dataset']

        if vuln_str and len(vuln_str.strip()) > 0:
            new_row = row.copy()
            new_row.update({'input': vuln_str, 'LOC': len(vuln_str.splitlines()), 'label': 'unsafe', 'tag': 'vuln'})
            rows.append(new_row)

        if self.fix and fix_str and len(fix_str.strip()) > 0:
            new_row = row.copy()
            new_row.update({'input': fix_str, 'LOC': len(fix_str.splitlines()), 'label': 'safe', 'cwe_id': None,
                            'score': None, 'tag': 'fix'})
            rows.append(new_row)
        if non_vuln_file and len(non_vuln_str.strip()) > 0:
            new_row = row.copy()
            new_row.update({'input': non_vuln_str, 'LOC': len(non_vuln_str.splitlines()), 'label': 'safe',
                            'cwe_id': None, 'score': None, 'tag': 'non-vuln'})
            rows.append(new_row)

        return rows

    def get_clean_content(self, project: str, commit: str, file: LocalGitFile):
        if file is None:
            return None

        clean_file = self.sources['clean_files_path'] / project / commit / file.short

        if clean_file.exists():
            return clean_file.open(mode='r').read()

        elif file and file.path.exists():
            clean_content = self.code_parser_handler.filter_comments(file.read(),
                                                                     extension=file.short.suffix.split('.')[-1])

            if len(clean_content.strip()) > 0:
                clean_file.parent.mkdir(parents=True, exist_ok=True)
                with clean_file.open(mode='w') as f:
                    f.write(clean_content)
                    return clean_content

        return None


def load(app):
    app.handler.register(Transform)
