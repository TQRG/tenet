import pandas as pd

from typing import Union, Tuple
from pathlib import Path

from tenet.core.sampling.balance import split_data, stratified_pair_hash, stratified_column, stratified_k_fold
from tenet.handlers.plugin import PluginHandler


class SamplingHandler(PluginHandler):
    """
        Sampling plugin
    """

    class Meta:
        label = "sampling"

    def __init__(self, **kw):
        super().__init__(**kw)
        self.token = None
        self.balance_techniques = ['stratified_pair_hash', 'stratified_column', 'stratified_k_fold']

    def set_sources(self):
        if 'k_fold' in self.node.kwargs:
            self.set('train_data', [self.path / f'train_{i}.csv' for i in range(0, self.node.kwargs['k_fold'])])
            self.set('val_data', [self.path / f'val_{i}.csv' for i in range(0, self.node.kwargs['k_fold'])])
            self.set('test_data', [self.path / f'test_{i}.csv' for i in range(0, self.node.kwargs['k_fold'])])
        else:
            self.set('train_data', self.path / 'train.csv')
            self.set('val_data', self.path / 'val.csv')
            self.set('test_data', self.path / 'test.csv')

    def get_sinks(self):
        pass

    def __call__(self, technique: str, seed: int, dataset: pd.DataFrame, undersample_safe: float = None,
                 column: str = None, k_fold: int = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
            return stratified_pair_hash(dataset=dataset, seed=seed, undersample_safe=undersample_safe)

        if technique == 'stratified_column':
            return stratified_column(dataset=dataset, column=column, seed=seed)

        self.app.log.info('No balancing technique applied.')
        return split_data(dataset=dataset, seed=seed)

    def run(self, dataset: pd.DataFrame, technique: str = "", seed: int = 0, undersample_safe: float = None,
            stratified_column: str = None, k_fold: int = None, columns: list = None, **kwargs) \
            -> Union[pd.DataFrame, None]:
        """
            runs the plugin
            :param only_single: flag for considering only single unsafe functions in a file
            :param only_multiple: flag for considering only multiple unsafe functions in a file
        """

        if technique and technique not in self.balance_techniques:
            self.app.log.warning(f"Technique must be one of the following: {self.balance_techniques}")
            return None

        self.app.log.info((f"Sampling with {technique}.\n" if technique else "") + f"Saving results to {self.path}")
        self.app.log.info(f"Dataset has {len(dataset)} samples.")

        if len(dataset[dataset.label == 'unsafe']) == 0:
            self.app.log.warning(f"No samples with 'unsafe' label in the dataset")

        if technique == 'stratified_k_fold':
            folds = stratified_k_fold(dataset=dataset, n_splits=k_fold, columns=columns, seed=seed)

            for (train_idxs, test_idxs), train_path, test_path in zip(folds, self.sources['train_data'], self.sources['test_data']):
                dataset.iloc[train_idxs].to_csv(str(train_path))
                dataset.iloc[test_idxs].to_csv(str(test_path))
        else:
            train, val, test = self.__call__(dataset=dataset, seed=seed, technique=technique,
                                             undersample_safe=undersample_safe, column=stratified_column)

            self.app.log.info("Writing split to files...")
            self.app.log.info(f"Train: {len(train)} ({train.label.value_counts()})\n"
                              f"Val.: {len(val)} ({val.label.value_counts()})\n"
                              f"Test: {len(test)} ({test.label.value_counts()})")

            train.to_csv(str(self.sources['train_data']))
            val.to_csv(str(self.sources['val_data']))
            test.to_csv(str(self.sources['test_data']))

        return dataset


def load(app):
    app.handler.register(SamplingHandler)
