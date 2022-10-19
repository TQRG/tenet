import pandas as pd

from typing import Union
from pathlib import Path

from securityaware.core.sampling.balance import oversampling, disjoint_smote, disjoint_smote_hash, \
    random_undersampling, split_data, one_one_ratio, disjoint, disjoint_hash, unique, unique_hash
from securityaware.data.dataset import XYDataset, Dataset

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
        self.balance_techniques = ['over', 'disj_over', 'disj_smote', 'unique', 'undersampling', 'stratified', '1_to_1']

    def run(self, dataset: pd.DataFrame, technique: str = "", seed: int = 0, offset: bool = False,
            **kwargs) -> Union[pd.DataFrame, None]:
        """
            runs the plugin
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

        train, val, test = split_data(dataset=dataset, seed=seed)

        self.app.log.info("Writing split to files...")
        self.app.log.info(f"Train: {len(train)}, Val.: {len(val)}, Test: {len(test)}")

        train.to_csv(str(train_data_path))
        val.to_csv(str(val_data_path))
        test.to_csv(str(test_data_path))

        return dataset

    '''
    # TODO: adapt rest of techniques
        def __call__(self, technique: str, seed: int, dataset: pd.DataFrame) \
                -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            # TODO: adapt rest of techniques
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
            else:

            self.app.log.info('No balancing technique applied.')
            return split_data(dataset=dataset, seed=seed)
    '''


def load(app):
    app.handler.register(SamplingHandler)
