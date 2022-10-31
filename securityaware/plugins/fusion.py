import pandas as pd

from typing import Union, Tuple
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
        self.set_labels(diff_labelled_data, label_type='diff')
        self.set_labels(static_labelled_data, label_type='sa')

        common_fns = self.get_common_fns(diff_labelled_data, static_labelled_data)
        common_hashes = common_fns.hash.to_list()

        diff_safe_uncommon_fns, diff_unsafe_uncommon_fns = self.get_uncommon_fns(diff_labelled_data,
                                                                                 target_analysis='diff_label',
                                                                                 dest_analysis='sa_label',
                                                                                 common_hashes=common_hashes)

        static_safe_uncommon_fns, static_safe_uncommon_fns = self.get_uncommon_fns(static_labelled_data,
                                                                                   target_analysis='sa_label',
                                                                                   dest_analysis='diff_label',
                                                                                   common_hashes=common_hashes)

        dataset = pd.concat([common_fns, diff_safe_uncommon_fns, diff_unsafe_uncommon_fns, static_safe_uncommon_fns,
                             static_safe_uncommon_fns])

        dataset.reset_index(inplace=True)
        self.app.log.info(f"Total functions after fusion: {len(dataset)}")

        if sa_labels:
            dataset.drop(columns=['da_label'], inplace=True)
            return dataset.rename(columns={'sa_label': 'label'})

        if da_labels:
            dataset.drop(columns=['sa_label'], inplace=True)
            return dataset.rename(columns={'da_label': 'label'})

        return dataset

    def set_labels(self, labelled_data: pd.DataFrame, label_type: str):
        self.app.log.info(f"Initial size of {label_type}: {len(labelled_data)}")
        labelled_data.reset_index(inplace=True, drop=True)
        labelled_data.rename(columns={'label': f"{label_type}_label"}, inplace=True)
        labelled_data.rename(columns={'cwe': f"{label_type}_cwe"}, inplace=True)
        labelled_data.rename(columns={'sfp': f"{label_type}_sfp"}, inplace=True)

    def get_common_fns(self, diff_labelled_data: pd.DataFrame, static_labelled_data: pd.DataFrame) -> pd.DataFrame:
        # fusing diff and static datasets into a single dataset
        common_fns = pd.merge(diff_labelled_data, static_labelled_data[['hash', 'sa_label', 'sa_cwe', 'sa_sfp']],
                              on='hash', how='inner')

        # drop duplicates on hash
        common_fns.drop_duplicates(subset=['hash'], inplace=True)

        self.app.log.info(f"Common functions {len(common_fns)}")
        self.app.log.info(f"\tDiff safe functions: {len(common_fns[common_fns['diff_label'] == 'safe'])}, "
                          f"Diff unsafe functions: {len(common_fns[common_fns['diff_label'] == 'unsafe'])}")
        self.app.log.info(f"\tStatic safe functions: {len(common_fns[common_fns['sa_label'] == 'safe'])}, "
                          f"Static unsafe functions: {len(common_fns[common_fns['sa_label'] == 'unsafe'])}")

        return common_fns

    def get_uncommon_fns(self, labelled_data: pd.DataFrame, target_analysis: str, dest_analysis: str,
                         common_hashes: list) -> Tuple[pd.DataFrame, pd.DataFrame]:
        uncommon_fns = labelled_data[~labelled_data.hash.isin(common_hashes)]
        safe_uncommon_fns = uncommon_fns[uncommon_fns[target_analysis] == 'safe']
        unsafe_uncommon_fns = uncommon_fns[uncommon_fns[target_analysis] == 'unsafe']
        self.app.log.info(f"{target_analysis} uncommon functions {len(uncommon_fns)} "
                          f"({len(safe_uncommon_fns)} safe, {len(unsafe_uncommon_fns)} unsafe)")

        safe_uncommon_fns[dest_analysis] = safe_uncommon_fns[target_analysis].copy()
        unsafe_uncommon_fns[dest_analysis] = unsafe_uncommon_fns[target_analysis].copy().apply(lambda x: 'safe')

        return safe_uncommon_fns, unsafe_uncommon_fns


def load(app):
    app.handler.register(FusionHandler)
