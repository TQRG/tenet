from pathlib import Path

import pandas as pd

from typing import Union

from securityaware.handlers.plugin import PluginHandler


class FilterHandler(PluginHandler):
    """
        Separate plugin
    """

    class Meta:
        label = "filter"

    def is_file_within(self, file: Path, limit: int) -> bool:
        """
            Returns the whether the file is within the specified size.
        """

        if file.exists():
            if limit and file.stat().st_size > limit:
                self.app.log.warning(f"File {file} size {file.stat().st_size} greater than limit")
                return False
            return True
        self.app.log.warning(f"File {file} not found.")
        return False

    def copy_files(self, a_proj: str, a_path: str, a_file: Path, b_proj: Path, b_path: Path, b_file: Path,
                   dest_path: Path):
        a_new_file = dest_path / a_proj / a_path
        a_new_file.parent.mkdir(parents=True, exist_ok=True)

        with a_file.open(mode="r") as rf, a_new_file.open(mode="w") as wf:
            self.app.log.info(f"Writing {a_file} to {a_new_file}.")
            wf.write(rf.read())

        b_new_file = dest_path / b_proj / b_path
        b_new_file.parent.mkdir(parents=True, exist_ok=True)

        with b_file.open(mode="r") as rf, b_new_file.open(mode="w") as wf:
            self.app.log.info(f"Writing {b_file} to {b_new_file}.")
            wf.write(rf.read())

    def run(self, dataset: pd.DataFrame, file_size: int = None, **kwargs) -> Union[pd.DataFrame, None]:
        """
            runs the plugin
        """
        self.set('dataset', self.output)
        out_files_path = Path(self.path, 'files')
        self.set('files_path', out_files_path)

        if file_size:
            filtered_dataset = []

            for i, row in dataset.iterrows():
                if self.is_file_within(Path(row.a_file), file_size) and self.is_file_within(Path(row.b_file),
                                                                                            file_size):
                    filtered_dataset.append(row)
                    self.copy_files(a_proj=row.a_proj, a_path=row.a_path, a_file=row.a_file,
                                    b_proj=row.b_proj, b_path=row.b_path, b_file=row.b_file,
                                    dest_path=out_files_path)

            if filtered_dataset:
                return pd.DataFrame(filtered_dataset, columns=list(dataset.columns.values))

            return None

        for i, row in dataset.iterrows():
            self.copy_files(a_proj=row.a_proj, a_path=row.a_path, a_file=row.a_file,
                            b_proj=row.b_proj, b_path=row.b_path, b_file=row.b_file,
                            dest_path=out_files_path)

        return dataset


def load(app):
    app.handler.register(FilterHandler)
