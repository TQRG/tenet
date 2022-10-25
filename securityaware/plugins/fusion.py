import pandas as pd

from typing import Union
from securityaware.handlers.plugin import PluginHandler


class FusionHandler(PluginHandler):
    """
        Data fusion plugin
    """

    class Meta:
        label = "fusion"

    def __init__(self, **kw):
        super().__init__(**kw)

    def run(self, dataset: pd.DataFrame, sa_labels: bool = False, da_labels: bool = False,
            **kwargs) -> Union[pd.DataFrame, None]:
        """
            runs the plugin
            :param da_labels: flag to return datasets with only diff labels
            :param sa_labels: flag to return datasets with only static labels
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
        diff_labelled_data.reset_index(inplace=True, drop=True)
        diff_labelled_data.rename(columns={'label': 'da_label'}, inplace=True)
        static_labelled_data.reset_index(inplace=True, drop=True)
        static_labelled_data.rename(columns={'label': 'sa_label'}, inplace=True)

        diff_data_hashes = diff_labelled_data.hash.to_list()
        static_data_hashes = static_labelled_data.hash.to_list()

        # fusing diff and static datasets into a single dataset
        common_fns = pd.merge(diff_labelled_data, static_labelled_data[['hash', 'sa_label']], on='hash', how='inner')

        self.app.log.info(f"Common functions {len(common_fns)}")
        self.app.log.info(f"\tDiff safe functions: {len(common_fns[common_fns['da_label'] == 'safe'])}, "
                          f"Diff unsafe functions: {len(common_fns[common_fns['da_label'] == 'unsafe'])}")
        self.app.log.info(f"\tStatic safe functions: {len(common_fns[common_fns['sa_label'] == 'safe'])}, "
                          f"Static unsafe functions: {len(common_fns[common_fns['sa_label'] == 'unsafe'])}")

        diff_uncommon_fns = diff_labelled_data[~diff_labelled_data.hash.isin(static_data_hashes)]
        diff_safe_uncommon_fns = diff_uncommon_fns[diff_uncommon_fns['da_label'] == 'safe']
        diff_unsafe_uncommon_fns = diff_uncommon_fns[diff_uncommon_fns['da_label'] == 'unsafe']
        self.app.log.info(f"Diff analysis uncommon functions {len(diff_uncommon_fns)} "
                          f"({len(diff_safe_uncommon_fns)} safe, {len(diff_unsafe_uncommon_fns)} unsafe)")

        diff_safe_uncommon_fns['sa_label'] = diff_safe_uncommon_fns['da_label'].copy()
        diff_unsafe_uncommon_fns['sa_label'] = diff_unsafe_uncommon_fns['da_label'].copy().apply(lambda x: 'safe')

        static_uncommon_fns = static_labelled_data[~static_labelled_data.hash.isin(diff_data_hashes)]
        static_safe_uncommon_fns = static_uncommon_fns[static_uncommon_fns['sa_label'] == 'safe']
        static_unsafe_uncommon_fns = static_uncommon_fns[static_uncommon_fns['sa_label'] == 'unsafe']
        self.app.log.info(f"Static analysis uncommon functions {len(static_uncommon_fns)} "
                          f"({len(static_safe_uncommon_fns)} safe, {len(static_unsafe_uncommon_fns)} unsafe)")

        static_safe_uncommon_fns['da_label'] = static_safe_uncommon_fns['sa_label'].copy()
        static_unsafe_uncommon_fns['da_label'] = static_unsafe_uncommon_fns['sa_label'].copy().apply(lambda x: 'safe')

        dataset = pd.concat([common_fns, diff_safe_uncommon_fns, diff_unsafe_uncommon_fns, static_safe_uncommon_fns,
                             static_unsafe_uncommon_fns])

        dataset.reset_index(inplace=True)
        self.app.log.info(f"Total functions after fusion: {len(dataset)}")

        if sa_labels:
            dataset.drop(columns=['da_label'], inplace=True)
            return dataset.rename(columns={'sa_label': 'label'})

        if da_labels:
            dataset.drop(columns=['sa_label'], inplace=True)
            return dataset.rename(columns={'da_label': 'label'})

        return dataset


def load(app):
    app.handler.register(FusionHandler)
