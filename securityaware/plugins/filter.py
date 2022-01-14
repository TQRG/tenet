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

    def run(self, dataset: pd.DataFrame, file_size: int = None, **kwargs) -> Union[pd.DataFrame, None]:
        """
            runs the plugin
        """
        self.set('dataset', self.output)

        if file_size:
            filtered_dataset = []

            for i, row in dataset.iterrows():
                if self.is_file_within(Path(row.a_file), file_size) and self.is_file_within(Path(row.b_file), file_size):
                    filtered_dataset.append(row)

            if filtered_dataset:
                return pd.DataFrame(filtered_dataset, columns=list(dataset.columns.values))

            return None

        return dataset


def load(app):
    app.handler.register(FilterHandler)
