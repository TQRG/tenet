import pandas as pd

from typing import Union

from tenet.handlers.plugin import PluginHandler


class FilterFnSizeHandler(PluginHandler):
    """
        Separate plugin
    """

    class Meta:
        label = "filter_fn_size"

    def run(self, dataset: pd.DataFrame, size: int = 100, **kwargs) -> Union[pd.DataFrame, None]:
        """
            runs the plugin
        """

        if 'sline' not in dataset.columns:
            self.app.log(f"Column 'sline' with the start line of the function not found in DataFrame")
            return None

        if 'eline' not in dataset.columns:
            self.app.log(f"Column 'eline' with the end line of the function not found in DataFrame")
            return None

        filtered = dataset[dataset.apply(lambda row: row.eline - row.sline < size, axis=1)]
        self.app.log.info(f"Dataset size is {len(filtered)} after applying filter.")

        return filtered


def load(app):
    app.handler.register(FilterFnSizeHandler)
