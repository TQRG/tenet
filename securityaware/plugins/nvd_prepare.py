import pandas as pd

from typing import Union

from tqdm import tqdm

from securityaware.handlers.plugin import PluginHandler
from securityaware.utils.misc import split_github_commits, clean_github_commits, project_from_chain, \
    parse_published_date, transform_to_commits


class NVDPrepare(PluginHandler):
    """
        NVDPrepare plugin
    """

    class Meta:
        label = "nvd_prepare"

    def run(self, dataset: pd.DataFrame, token: str = None, **kwargs) -> Union[pd.DataFrame, None]:
        """
            runs the plugin
        """
        dataset.rename(inplace=True, columns={'cve_id': 'vuln_id', 'cwes': 'cwe_id', 'commits': 'chain',
                                              'description': 'summary', 'impact': 'score'})
        dataset = dataset[['vuln_id', 'cwe_id', 'score', 'chain', 'summary', 'published_date']]
        dataset = self.normalize(dataset)
        self.github_handler.token = token

        for idx, row in tqdm(dataset.iterrows()):
            self.multi_task_handler.add(chain=row['chain']).update_id(idx)

        self.multi_task_handler(func=self.github_handler.normalize_sha)
        df_normalized_sha = self.multi_task_handler.get_tasks(as_frame=True)
        df_normalized_sha = df_normalized_sha[['result']].rename(columns={'result': 'chain'})

        dataset.drop(columns=['chain'], inplace=True)
        dataset = pd.merge(dataset, df_normalized_sha, left_index=True, right_index=True)
        dataset = dataset.dropna(subset=['chain'])
        self.app.log.info(f"Entries (after nan drop): {len(dataset)}")

        dataset = transform_to_commits(dataset)

        return dataset

    def normalize(self, df: pd.DataFrame):
        self.app.log.info("Normalizing NVD ...")
        df['chain'] = df['chain'].apply(lambda x: split_github_commits(x))
        self.app.log.info(f"Size after split {len(df)}")

        df['chain'] = df['chain'].apply(lambda x: clean_github_commits(x))
        self.app.log.info(f"Entries after clean {len(df)}")

        df = df.dropna(subset=['chain'])
        self.app.log.info(f"Entries (After duplicates): {len(df)}")

        df['chain_len'] = df['chain'].apply(lambda x: len(x))
        df['project'] = df['chain'].apply(lambda x: project_from_chain(x))
        df['published_date'] = df['published_date'].apply(lambda x: parse_published_date(x))

        return df


def load(app):
    app.handler.register(NVDPrepare)
