import time

import pandas as pd
import requests

from pathlib import Path
from typing import Union

from securityaware.handlers.plugin import PluginHandler
from securityaware.core.diff_labeller.misc import check_or_create_dir, safe_write
from securityaware.data.runner import Task, Runner
from securityaware.handlers.runner import ThreadPoolWorker


class FileCollector(PluginHandler):
    """
        Github collector plugin
    """

    class Meta:
        label = "file_collector"

    def __init__(self, **kw):
        super().__init__(**kw)

    def run(self, dataset: pd.DataFrame, **kwargs) -> Union[pd.DataFrame, None]:
        """
            runs the plugin
        """
        files_dir = Path(self.path, 'files')
        self.set('files_path', files_dir)
        self.set('dataset', self.output)

        # Save diff texts as files in the directory
        check_or_create_dir(files_dir)

        runner_data = Runner()
        threads = self.app.get_config('local_threads')

        # Select columns: repo, sha_list, parents
        # Ensure no duplicates
        tasks = []
        total = len(dataset)
        # Drop unnamed columns
        dataset.drop(columns=[c for c in dataset.columns if 'Unnamed:' in c], inplace=True)

        for i, row in dataset.iterrows():
            self.app.log.info(f"Creating task for repo {row['project_name'] + row['file_path']}. {i}/{total}")
            task = Task()
            task['id'] = i
            task['row'] = row
            task['files_dir'] = files_dir
            tasks.append(task)

        worker = ThreadPoolWorker(runner_data, tasks=tasks, threads=threads, logger=self.app.log,
                                  func=self.parse_diffs_task)
        worker.start()

        while worker.is_alive():
            time.sleep(1)

        rows = [res['result'] for res in runner_data.finished if 'result' in res and res['result']]

        if rows:
            return pd.DataFrame(rows, columns=dataset.columns)

        return None

    def parse_diffs_task(self, task: Task):
        return self.parse_diffs(row=task['row'], files_dir=task['files_dir'])

    def parse_diffs(self, row: pd.Series, files_dir: Path):
        file_path = files_dir / row.project_name / row.commit_hash / row.file_path
        commit_url = f"{row.project}/raw/{row.commit_hash}/{row.file_path}"

        # Get the diff string using GitHub API
        self.app.log.info(f"Requesting {commit_url}")

        if not file_path.exists():
            file_path.parent.mkdir(exist_ok=True, parents=True)
            fstr = requests.get(commit_url).text

            if fstr:
                # Save the diff string as a file

                with open(file_path, "w") as tmp_file:
                    self.app.log.info(f"Writing file to {file_path}")
                    safe_write(tmp_file, fstr, f"{row.project_name}_{row.commit_hash}")

        row.file_path = f"{row.project_name}/{row.commit_hash}/{row.file_path}"
        return row.to_list()


def load(app):
    app.handler.register(FileCollector)
