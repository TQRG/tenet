import pandas as pd

from typing import Union
from securityaware.handlers.plugin import PluginHandler


class HybridHandler(PluginHandler):
    """
        Hybrid fusion plugin
    """

    class Meta:
        label = "hybrid"

    def __init__(self, **kw):
        super().__init__(**kw)

    def run(self, dataset: pd.DataFrame, inner_join: bool = False, **kwargs) -> Union[pd.DataFrame, None]:
        """
            runs the plugin
            :param inner_join: flag to perform inner join between diff labelled and static labelled datasets, if false,
            performs full outer join
        """

        hybrid_label = 'safe' if inner_join else 'unsafe'
        join_type = 'inner' if inner_join else 'outer'
        dataset['label'] = dataset['diff_label']
        dataset['cwe'] = dataset['diff_cwe']
        dataset['sfp'] = dataset['diff_sfp']

        for i, row in dataset[(dataset['diff_label'] == 'unsafe') | (dataset['sa_label'] == 'unsafe')].iterrows():
            if row['diff_label'] == 'unsafe' and row['sa_label'] == 'safe':
                self.app.log.info(f"Updating label for {row.hash}")
                dataset.at[i, 'label'] = hybrid_label
                dataset.at[i, 'cwe'] = row['diff_cwe']
                dataset.at[i, 'sfp'] = row['diff_sfp']
            elif row['sa_label'] == 'unsafe' and row['diff_label'] == 'safe':
                self.app.log.info(f"Updating label for {row.hash}")
                dataset.at[i, 'label'] = hybrid_label
                dataset.at[i, 'cwe'] = row['sa_cwe']
                dataset.at[i, 'sfp'] = row['sa_sfp']
            elif row['sa_label'] == 'unsafe' and row['diff_label'] == 'unsafe':
                if row['sa_cwe'] != row['diff_cwe']:
                    dataset.at[i, 'cwe'] = f"{row['diff_cwe']}|{row['sa_cwe']}"
                elif row['diff_sfp'] != row['sa_sfp']:
                    dataset.at[i, 'sfp'] = f"{row['diff_sfp']}|{row['sa_sfp']}"
                else:
                    dataset.at[i, 'cwe'] = row['diff_cwe']

        self.app.log.info(f"Total unsafe fns from hybrid {join_type} join: {len(dataset[dataset.label == 'unsafe'])}")

        dataset.drop(columns=['diff_label', 'diff_cwe', 'diff_sfp', 'sa_label', 'sa_cwe', 'sa_sfp'], inplace=True)

        return dataset


def load(app):
    app.handler.register(HybridHandler)
