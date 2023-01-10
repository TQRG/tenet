import pandas as pd

from pathlib import Path
from typing import Union
from tqdm import tqdm

from tenet.handlers.github import LocalGitFile
from tenet.handlers.plugin import PluginHandler
from tenet.core.diff_labeller.misc import check_or_create_dir


class Download(PluginHandler):
    """
        Download plugin
    """

    class Meta:
        label = "download"

    def __init__(self, **kw):
        super().__init__(**kw)
        self.files_dir: Path = None

    def run(self, dataset: pd.DataFrame, tokens: list = None) -> Union[pd.DataFrame, None]:
        """
            runs the plugin
        """
        self.files_dir = Path(self.path, 'files')
        check_or_create_dir(self.files_dir)
        self.set('files_path', self.files_dir)
        self.github_handler.tokens = tokens
        self.set('dataset', self.output)

        self.app.log.info(f"Creating {len(dataset)} tasks.")

        for row in tqdm(dataset.to_dict(orient='records')):
            self.multi_task_handler.add(row=row)

        self.multi_task_handler(func=self.download)
        rows = self.multi_task_handler.results()

        if rows:
            return pd.DataFrame(rows)

        return None

    def download(self, row: dict):
        project_path = self.files_dir / row['project_name']
        if 'raw_url_vuln' in row and not pd.isnull(row['raw_url_vuln']):
            vuln_file = LocalGitFile(url=row['raw_url_vuln'], short=Path(row['file_path']), tag='vuln',
                                     path=project_path / row['vuln_commit_hash'] / row['file_path'])

            if not vuln_file.path.exists():
                vuln_file.download()
                if vuln_file.write() < 1:
                    self.app.log.warning(f"No file content for {row['vuln_id']}")
                    return None

        if 'raw_url_fix' in row and not pd.isnull(row['raw_url_fix']):
            fix_file = LocalGitFile(url=row['raw_url_fix'], short=Path(row['file_path']), tag='fix',
                                    path=project_path / row['fixed_commit_hash'] / row['file_path'])

            if not fix_file.path.exists():
                fix_file.download()
                if fix_file.write() < 1:
                    self.app.log.warning(f"No file content for {row['vuln_id']}")
                    return None

        if 'raw_url_non_vuln' in row:
            non_vuln_file = LocalGitFile(url=row["raw_url_non_vuln"], short=Path(row['non_vuln_file_path']), tag='non-vuln',
                                         path=project_path / row['non_vuln_commit_hash'] / row['non_vuln_file_path'])
            if not non_vuln_file.path.exists():
                non_vuln_file.download()
                if non_vuln_file.write() < 1:
                    self.app.log.warning(f"No file content for {row['vuln_id']}")
                    return None

        return row


def load(app):
    app.handler.register(Download)
