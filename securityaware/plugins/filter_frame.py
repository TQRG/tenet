from pathlib import Path
import pandas as pd

from typing import Union, Any

from securityaware.handlers.plugin import PluginHandler


class FilterFrameHandler(PluginHandler):
    """
        Separate plugin
    """

    class Meta:
        label = "filter_frame"

    def run(self, dataset: pd.DataFrame, column: str = None, value: Any = None,  **kwargs) -> Union[pd.DataFrame, None]:
        """
            runs the plugin
        """
        self.set('dataset', self.output)

        if not column:
            self.app.log("Must specify 'column' name")
            return None

        if column not in dataset.columns:
            self.app.log(f"Column {column} not found in DataFrame")
            return None

        return dataset[dataset[column] == value]


def load(app):
    app.handler.register(FilterFrameHandler)
