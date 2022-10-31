import pandas as pd

from typing import Union, Tuple
from pathlib import Path

from securityaware.core.sampling.balance import split_data, stratified_pair_hash
from securityaware.handlers.plugin import PluginHandler


class SamplingHandler(PluginHandler):
    """
        Sampling plugin
    """

    class Meta:
        label = "sampling"

    def __init__(self, **kw):
        super().__init__(**kw)
        self.token = None
        self.balance_techniques = ['stratified_pair_hash']

    def __call__(self, technique: str, seed: int, dataset: pd.DataFrame) \
            -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # TODO: adapt rest of techniques
        '''
                if technique == 'over':
                    return oversampling(x, y, seed)
                elif technique == 'disj_smote':
                    if offset:
                        return disjoint_smote_hash(x, y, seed)
                    else:
                        return disjoint_smote(x, y, seed)
                elif technique == 'disj_over':
                    if offset:
                        return disjoint_hash(x, y, seed)
                    else:
                        return disjoint(x, y, seed)
                elif technique == 'unique':
                    if offset:
                        return unique_hash(x, y, seed)
                    else:
                        return unique(x, y, seed)
                elif technique == '1_to_1':
                    return one_one_ratio(x, y, seed)
                elif technique == 'random_undersampling':
                    return random_undersampling(x, y, seed)
        '''

        if technique == 'stratified_pair_hash':
            return stratified_pair_hash(dataset=dataset, seed=seed)

        self.app.log.info('No balancing technique applied.')
        return split_data(dataset=dataset, seed=seed)

    def run(self, dataset: pd.DataFrame, technique: str = "", seed: int = 0, only_single: bool = False,
            target_primary_sfp: int = None, only_multiple: bool = False, **kwargs) -> Union[pd.DataFrame, None]:
        """
            runs the plugin

            :param only_single: flag for considering only single unsafe functions in a file
            :param only_multiple: flag for considering only multiple unsafe functions in a file
        """
        # TODO: change these sets into something simpler
        train_data_path = Path(self.path, 'train.csv')
        val_data_path = Path(self.path, 'val.csv')
        test_data_path = Path(self.path, 'test.csv')

        self.set('train_data', train_data_path)
        self.set('val_data', val_data_path)
        self.set('test_data', test_data_path)

        if technique and technique not in self.balance_techniques:
            self.app.log.warning(f"Technique must be one of the following: {self.balance_techniques}")
            return None

        self.app.log.info((f"Sampling with {technique}.\n" if technique else "") + f"Saving results to {self.path}")
        self.app.log.info(f"Dataset has {len(dataset)} samples.")

        if target_primary_sfp:
            if target_primary_sfp not in self.cwe_list_handler.sfp_primary_ids:
                self.app.log.error(f"Could not found SFP ID {target_primary_sfp}")
                return None

            if 'sfp' not in dataset.columns:
                self.app.log.error(f"'sfp' column with primary sfp type not found in the dataset")
                return None

            for i, row in dataset[dataset.label == 'unsafe'].iterrows():
                if isinstance(row.sfp, str):
                    if all([int(sfp) != target_primary_sfp for sfp in row.sfp.split('|')]):
                        self.app.log.info(f"Changed label for {row.sfp} to safe")
                        dataset.loc[i, 'label'] = 'safe'
                elif int(row.sfp) != target_primary_sfp:
                    self.app.log.info(f"Changed label for {row.sfp} to safe")
                    dataset.loc[i, 'label'] = 'safe'

        if only_single:
            for g, rows in dataset[dataset.label == 'unsafe'].groupby(['owner', 'project', 'version']):
                if len(rows) > 1:
                    for i, row in rows.iterrows():
                        dataset.loc[i, 'label'] = 'safe'

        elif only_multiple:
            for g, rows in dataset[dataset.label == 'unsafe'].groupby(['owner', 'project', 'version']):
                if len(rows) == 1:
                    for i, row in rows.iterrows():
                        dataset.loc[i, 'label'] = 'safe'

        if len(dataset[dataset.label == 'unsafe']) == 0:
            self.app.log.warning(f"No samples with 'unsafe' label in the dataset")

        train, val, test = self.__call__(dataset=dataset, seed=seed, technique=technique)

        self.app.log.info("Writing split to files...")
        self.app.log.info(f"Train: {len(train)} ({train.label.value_counts()})\n"
                          f"Val.: {len(val)} ({val.label.value_counts()})\n"
                          f"Test: {len(test)} ({test.label.value_counts()})")

        train.to_csv(str(train_data_path))
        val.to_csv(str(val_data_path))
        test.to_csv(str(test_data_path))

        return dataset


def load(app):
    app.handler.register(SamplingHandler)
