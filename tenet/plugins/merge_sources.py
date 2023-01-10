import pandas as pd

from typing import Union

from tenet.handlers.plugin import PluginHandler


class MergeSources(PluginHandler):
    """
        MergeSources plugin
    """

    class Meta:
        label = "merge_sources"

    def run(self, 
            dataset: pd.DataFrame, 
            **kwargs) -> Union[pd.DataFrame, None]:
        """
            runs the plugin
        """
        print(kwargs['nvd_path'])
        print(kwargs['osv_path'])
        df_osv = pd.read_csv(f"{self.app.bind}/{kwargs['osv_path']}")
        df_nvd = pd.read_csv(f"{self.app.bind}/{kwargs['nvd_path']}")
        df = pd.concat([df_osv, df_nvd], ignore_index=True)
        self.app.log.info(f"Total number of entries: {len(df)}")
        return df
   

def load(app):
    app.handler.register(MergeSources)
