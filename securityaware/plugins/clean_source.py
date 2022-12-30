import pandas as pd


from typing import Union

from securityaware.handlers.plugin import PluginHandler
from securityaware.core.plotter import Plotter


class CleanSource(PluginHandler):
    """
        CleanSource plugin
    """
    class Meta:
        label = "clean_source"

    def run(self, dataset: pd.DataFrame, projects_blacklist: list = None, dataset_name: str = None,
            drop_multi_cwe: bool = False, drop_unk_cwe: bool = False, **kwargs) -> Union[pd.DataFrame, None]:
        """Cleans and filters the sources
        dataset.
        Args:
            projects_blacklist (list): list of projects to exclude from dataset
            drop_multi_cwe (bool): flag to drop CVE samples with multiple CWE-IDs
            drop_unk_cwe (bool): flag to drop CVE samples with unknown CWE-IDs (e.g., NVD-CWE-Other, NVD-CWE-noinfo)
        """

        dataset['message'] = dataset['message'].apply(lambda x: self.msg(x))
        no_merge = ~dataset['message'].str.contains("merge message")
        cwe = dataset['cwe_id'].notnull()
        commit = dataset['last_fix_commit'].notnull()

        # filters only the patches with
        df = dataset[(cwe) & (commit) & (no_merge)]
        self.app.log.info(f"Size before filtering by patches {len(df)}")
        df['before_first_fix_commit'] = df['before_first_fix_commit'].apply(lambda x: list(eval(x)))
        df['before_first_fix_commit'] = df['before_first_fix_commit'].apply(lambda x: x[0] if x else None)
        df = df[~df['before_first_fix_commit'].isnull()]
        self.app.log.info(f"Size after filtering by patches {len(df)}")

        if drop_multi_cwe:
            initial_size = len(df)
            df = df[df.apply(lambda x:  len(eval(x['cwe_id'])) == 1, axis=1)]
            self.app.log.warning(f"Dropped {initial_size - len(df)} samples with multiple CWE-IDs")

        if drop_unk_cwe:
            # TODO: drop considering a whitelist
            initial_size = len(df)
            unk_cwes = ['NVD-CWE-Other', 'NVD-CWE-noinfo', 'Unknown']
            df = df[df.apply(lambda x:  all([el not in unk_cwes for el in list(eval(x['cwe_id']))]), axis=1)]
            self.app.log.warning(f"Dropped {initial_size - len(df)} samples with unknown CWE-ID")

        df.rename(columns={'project': 'project_url'}, inplace=True)

        df['bf_class'] = [None] * len(df)
        df['operation'] = [None] * len(df)

        for i, row in df.iterrows():
            df.at[i, 'bf_class'], df.at[i, 'operation'] = self.cwe_list_handler.find_bf_class(row['cwe_id'])

        if dataset_name is None:
            dataset_name = self.output.stem

        # remove deprecated repo
        for proj in projects_blacklist:
            df = df[~df['commit_href'].str.contains(proj)]

        cols = ["vuln_id", "cwe_id", "dataset", "score", "published_date", "project_url", "commit_href", "commit_sha",
                "before_first_fix_commit", "last_fix_commit", "commit_datetime", "files", 'language', 'bf_class', 'operation']

        df['dataset'] = [dataset_name] * len(df)
        return df[cols]

    @staticmethod
    def msg(x):
        if pd.notna(x):
            return x.lower()
        else:
            return ''

    def plot(self, dataset: pd.DataFrame, **kwargs):
        top_10_cwe_without_bf = list(dataset[dataset['bf_class'].isnull()]['cwe_id'].value_counts().head(10).keys())
        self.app.log.info(f"Top 10 CWE IDs without BF Class: {top_10_cwe_without_bf}")

        dataset = dataset[~dataset['bf_class'].isnull()]
        self.app.log.info(f"Entries with BF class: {len(dataset)}")

        Plotter(self.path).bar_labels(dataset, column='cwe_id', y_label='Occurrences', x_label='CWE-ID')
        Plotter(self.path).bar_labels(dataset, column='bf_class', y_label='Occurrences', x_label='BF Class')


def load(app):
    app.handler.register(CleanSource)
