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
        dataset['label'] = dataset['da_label']

        for i, row in dataset[(dataset['da_label'] == 'unsafe') | (dataset['sa_label'] == 'unsafe')].iterrows():
            if row['da_label'] == 'unsafe' and row['sa_label'] == 'safe':
                self.app.log.info(f"Updating label for {row.fpath}")
                dataset.at[i, 'label'] = hybrid_label
            elif row['sa_label'] == 'unsafe' and row['da_label'] == 'safe':
                self.app.log.info(f"Updating label for {row.fpath}")
                dataset.at[i, 'label'] = hybrid_label

        self.app.log.info(f"Total unsafe fns from hybrid {join_type} join: {len(dataset[dataset.label == 'unsafe'])}")

        dataset.drop(columns=['da_label'], inplace=True)
        dataset.drop(columns=['sa_label'], inplace=True)

        return dataset


def load(app):
    app.handler.register(HybridHandler)
