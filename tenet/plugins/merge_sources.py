import pandas as pd

from typing import Union

from tenet.handlers.plugin import PluginHandler


class MergeSources(PluginHandler):
    """
        MergeSources plugin
    """

    class Meta:
        label = "merge_sources"

    def run(self, dataset: pd.DataFrame, **kwargs) -> Union[pd.DataFrame, None]:
        """
            runs the plugin
        """
        sources = [connector.source for connector in self.get_sinks().values() if 'dataset' in connector.links]

        if sources:
            dataset_paths = [self.app.executed_edges[s].output for s in sources if s in self.app.executed_edges]
            datasets = [pd.read_csv(d, index_col=False) for d in dataset_paths]

            if datasets:
                df = pd.concat(datasets, ignore_index=True)
                self.app.log.info(f"Total number of entries: {len(df)}")
                return df

        return None
   

def load(app):
    app.handler.register(MergeSources)
