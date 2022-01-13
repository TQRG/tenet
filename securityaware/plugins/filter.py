import pandas as pd

from typing import Union

from securityaware.handlers.plugin import PluginHandler


class FilterHandler(PluginHandler):
    """
        Separate plugin
    """
    class Meta:
        label = "filter"

    def run(self, dataset: pd.DataFrame, **kwargs) -> Union[pd.DataFrame, None]:
        """
            runs the plugin
        """

        return dataset


def load(app):
    app.handler.register(FilterHandler)
