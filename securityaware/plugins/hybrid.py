import pandas as pd

from typing import Union
from securityaware.handlers.plugin import PluginHandler


class HybridHandler(PluginHandler):
    """
        HybridHandler plugin
    """

    class Meta:
        label = "hybrid"

    def __init__(self, **kw):
        super().__init__(**kw)

    def run(self, dataset: pd.DataFrame, inner_join: bool = False, **kwargs) -> Union[pd.DataFrame, None]:
        """
            runs the plugin
            :param inner_join: flag to perform inner join between diff labelled and static labelled datasets, if false,
            full outer join if performed
        """

        diff_labelled_data_path = self.get('diff_labelled_data_path')
        static_labelled_data_path = self.get('static_labelled_data_path')

        if not diff_labelled_data_path:
            self.app.log.error(f"Diff labelled data path not instantiated")
            return None

        if not static_labelled_data_path:
            self.app.log.error(f"Static labelled data path not instantiated")
            return None

        diff_labelled_data = pd.read_csv(str(diff_labelled_data_path))
        static_labelled_data = pd.read_csv(str(static_labelled_data_path))

        # label the safe labels in the diff dataset as unsafe given
        unsafe_static_labelled_fns = static_labelled_data[static_labelled_data.label == 'unsafe'].hash.to_list()

        if inner_join:
            for i, row in diff_labelled_data[diff_labelled_data.label == 'unsafe'].iterrows():
                if row.label == 'unsafe' and row.hash not in unsafe_static_labelled_fns:
                    self.app.log.info(f"Updating label for {row.fpath}")
                    diff_labelled_data.at[i, 'label'] = 'safe'
        else:
            for i, row in diff_labelled_data[diff_labelled_data.hash.isin(unsafe_static_labelled_fns)].iterrows():
                if row.label != 'unsafe':
                    self.app.log.info(f"Updating label for {row.fpath}")
                    diff_labelled_data.at[i, 'label'] = 'unsafe'
            self.app.log.info(f"Unsafe fns: {len(diff_labelled_data[diff_labelled_data.label == 'unsafe'])}")

        return diff_labelled_data


def load(app):
    app.handler.register(HybridHandler)
