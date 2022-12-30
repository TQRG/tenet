from pathlib import Path

import pandas as pd
from tqdm import tqdm

from typing import Union

from securityaware.core.diff_labeller.changes import Triplet
from securityaware.core.sampling.scenario import Sample
from securityaware.handlers.github import LocalGitFile
from securityaware.handlers.plugin import PluginHandler


class Transform(PluginHandler):
    """
        Transform plugin
    """

    class Meta:
        label = "transform"

    def __init__(self, **kw):
        super().__init__(**kw)
        self.files_path = None
        self.real = None
        self.snippet = None
        self.fix = None

    def run(self, dataset: pd.DataFrame, real: bool = False, fix: bool = False, snippet: bool = False) \
            -> Union[pd.DataFrame, None]:
        """
            runs the plugin
            Args:
            real (bool): flag to add the other parts of file as non-vuln snippets
            fix (bool): flag to add fix file
            snippet (bool): flag to transform samples to snippet granularity
        """
        self.files_path = self.get('files_path')
        self.real = real
        self.fix = fix
        self.snippet = snippet

        for row in tqdm(dataset.to_dict(orient='records')):
            self.multi_task_handler.add(row=row)

        self.multi_task_handler(func=self.transform)
        rows = self.multi_task_handler.results(expand=True)

        if rows:
            parsed = pd.DataFrame(rows)

            return parsed

        return None

    def transform(self, row: dict):
        project_path = self.files_path / row['project_name']

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

        if 'raw_url_non_vuln' in row:
            non_vuln_file = LocalGitFile(url=row["raw_url_non_vuln"], short=Path(row['non_vuln_file_path']),
                                         tag='non-vuln',
                                         path=project_path / row['non_vuln_commit_hash'] / row['non_vuln_file_path'])
        else:
            non_vuln_file = None

        triplet = Triplet(vuln_file=vuln_file, fix_file=fix_file, non_vuln_file=non_vuln_file, real=self.real)
        sample = Sample(row, snippet=self.snippet, triplet=triplet, fix=self.fix)

        return [r for r in sample() if r is not None]


def load(app):
    app.handler.register(Transform)
