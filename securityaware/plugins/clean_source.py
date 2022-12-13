import pandas as pd


from typing import Union

from securityaware.handlers.plugin import PluginHandler


class CleanSource(PluginHandler):
    """
        CleanSource plugin
    """
    class Meta:
        label = "clean_source"

    def run(self, dataset: pd.DataFrame, projects_blacklist: list = None, dataset_name: str = None,
            **kwargs) -> Union[pd.DataFrame, None]:
        """Cleans and filters the sources
        dataset.
        Args:
            projects_blacklist (list): list of projects to exclude from dataset
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
        df['cwe_id'] = df['cwe_id'].apply(lambda x: ','.join(list(eval(x))))
        df.rename(columns={'project': 'project_url'}, inplace=True)

        if dataset_name is None:
            dataset_name = self.output.stem

        # remove deprecated repo
        for proj in projects_blacklist:
            df = df[~df['commit_href'].str.contains(proj)]

        cols = ["vuln_id", "cwe_id", "dataset", "score", "published_date", "project_url", "commit_href", "commit_sha",
                "before_first_fix_commit", "last_fix_commit", "commit_datetime", "files", 'language']

        df['dataset'] = [dataset_name] * len(df)
        return df[cols]

    @staticmethod
    def msg(x):
        if pd.notna(x):
            return x.lower()
        else:
            return ''


def load(app):
    app.handler.register(CleanSource)
