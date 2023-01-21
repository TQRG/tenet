import numpy as np
import pandas as pd

from typing import Union

from tenet.handlers.plugin import PluginHandler


class Attributes(PluginHandler):
    """
        Data attributes plugin
    """

    class Meta:
        label = "attributes"

    def __init__(self, **kw):
        super().__init__(**kw)

    def set_sources(self):
        self.set('train_data', [self.path / path.name for path in self.sinks['train_data']])

    def get_sinks(self):
        self.get('train_data')

    def run(self, dataset: pd.DataFrame, technique: str = "", seed: int = 0, target_bf_class: str = None,
            drop_ratio: float = None, drop_tag: str = None, target_cwe: int = None, target_project: str = None,
            target_software_type: str = None, **kwargs) -> Union[pd.DataFrame, None]:

        for i, (train_data_in, train_data_out) in enumerate(zip(self.sinks['train_data'], self.sources['train_data'])):
            train_data = pd.read_csv(str(train_data_in))

            if drop_ratio and 'tag' in train_data and drop_tag in train_data['tag'].unique():
                if drop_ratio < 1:
                    train_data_tag = train_data[train_data['tag'] == drop_tag]
                    drop_indices = np.random.choice(train_data_tag.index, round(drop_ratio * len(train_data_tag)),
                                                    replace=False)
                    train_data = train_data.drop(drop_indices)
                else:
                    train_data = train_data[train_data['tag'] != drop_tag]

            if 'tag' in train_data:
                self.app.log.info(f"{train_data['tag'].value_counts()}")

            if target_bf_class:
                # TODO: check if BF-class exists

                if 'bf_class' not in train_data.columns:
                    self.app.log.error(f"'bf_class' column with primary bf_class type not found in the dataset")
                    return None

                not_target_bf_class = train_data[train_data['label'] == 'unsafe' and (train_data['bf_class'] != target_bf_class)]
                not_target_safe = train_data[(train_data['label'] == 'safe') & (train_data['vuln_commit_hash'].isin(not_target_bf_class['vuln_commit_hash']))]
                train_data = train_data.drop(not_target_bf_class.index)
                train_data = train_data.drop(not_target_safe.index)

            if target_cwe:
                # TODO: check if CWE-ID exists

                if 'cwe_id' not in train_data.columns:
                    self.app.log.error(f"'cwe_id' column not found in the dataset")
                    return None

                train_data['cwe_id'] = train_data['cwe_id'].apply(lambda x: self.cwe_list_handler.parse_cwe_id(x))

                not_target_cwe = train_data[(train_data['label'] == 'unsafe') & (train_data['cwe_id'] != target_cwe)]
                not_target_safe = train_data[(train_data['label'] == 'safe') & (train_data['vuln_commit_hash'].isin(not_target_cwe['vuln_commit_hash']))]
                train_data = train_data.drop(not_target_cwe.index)
                train_data = train_data.drop(not_target_safe.index)

            if target_project:
                if target_project not in train_data['project'].unique():
                    self.app.log.error(f"No '{target_project}' project in the training data")
                    return None

                not_project_cwe = train_data[train_data['project'] != target_project]
                train_data = train_data.drop(not_project_cwe.index)

            if target_software_type:
                # TODO: get categories from handlers
                if target_software_type not in ['utility', 'middleware', 'framework', 'operating system',
                                                'web application', 'server', 'browser', 'unk']:
                    self.app.log.error(f"Could not find '{target_software_type}' software type")
                    return None

                if 'sw_type' not in train_data.columns:
                    self.app.log.error(f"'sw_type' column not found in the dataset")
                    return None

                not_sw_type = train_data[train_data['sw_type'] != target_software_type]
                train_data = train_data.drop(not_sw_type.index)

            if len(train_data[train_data.label == 'unsafe']) == 0:
                self.app.log.warning(f"No samples with 'unsafe' label in the dataset")

            train_data.to_csv(str(train_data_out))

        return dataset


def load(app):
    app.handler.register(Attributes)
