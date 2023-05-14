import pandas as pd

from typing import Union

from tqdm import tqdm

from tenet.core.plotter import Plotter
from tenet.handlers.plugin import PluginHandler
from tenet.utils.misc import split_github_commits, clean_github_commits, project_from_chain, \
    parse_published_date, transform_to_commits, join


class OSVprepare(PluginHandler):
    """
        OSVPrepare plugin
    """

    class Meta:
        label = "osv_prepare"

    def __init__(self, **kw):
        super().__init__(**kw)

    def set_sources(self):
        self.set('metadata_path', self.path / 'metadata')
        self.set('normalized_path', self.path / f'{self.output.stem}_normalized.csv')

    def get_sinks(self):
        pass

    def run(self, dataset: pd.DataFrame, metadata: bool = True, language: bool = True, extension: bool = True,
            include_comments: bool = True, drop_patch: bool = True, **kwargs) \
            -> Union[pd.DataFrame, None]:
        """
            runs the plugin
        """

        if not self.sources['normalized_path'].exists():
            dataset = dataset.rename(columns={'commits': 'chain'})
            dataset['summary'] = dataset.apply(lambda x: join(x['summary'], x['details']), axis=1)
            dataset = dataset[['vuln_id', 'cwe_id', 'score', 'chain', 'summary', 'published_date']]
            dataset['dataset'] = "OSV"
            dataset = self.normalize(dataset)

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
            dataset.to_csv(str(self.sources['normalized_path']))
        else:
            dataset = pd.read_csv(str(self.sources['normalized_path']))

        self.app.log.info(f"Size after normalization: {len(dataset)}")

        if metadata:
            del self.multi_task_handler
            for project, rows in tqdm(dataset.groupby(['project'])):
                self.multi_task_handler.add(project=project, chains=rows['chain'].to_list(), indexes=rows.index,
                                            include_comments=include_comments, commits=rows['commit_sha'].to_list(),
                                            save_path=self.sources['metadata_path'], drop_patch=drop_patch)

            self.multi_task_handler(func=self.github_handler.get_project_metadata)

            try:
                metadata_df = pd.concat(self.multi_task_handler.results())
            except ValueError as ve:
                self.app.log.error(ve)
                return None

            dataset.drop(columns=['commit_sha'], inplace=True)
            dataset = pd.merge(dataset, metadata_df, left_index=True, right_index=True)

            self.app.log.info(f"Size after merging with metadata: {len(dataset)}")

        if extension:
            dataset["files_extension"] = dataset["files"].apply(lambda x: self.file_parser_handler.get_files_extension(x))

        if language:
            dataset = self.add_language(dataset)

        return dataset

    def add_language(self, df: pd.DataFrame):
        if 'file_extension' not in df:
            df["files_extension"] = df["files"].apply(lambda x: self.file_parser_handler.get_files_extension(x))

        df["language"] = df["files_extension"].apply(lambda x: self.file_parser_handler.get_language(x))

        return df

    def normalize(self, df: pd.DataFrame):
        self.app.log.info("Normalizing OSV ...")
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

    def plot(self, dataset: pd.DataFrame, **kwargs):
        # add bf_class
        dataset['bf_class'] = [None] * len(dataset)

        for i, row in dataset.iterrows():
            bf_class = self.cwe_list_handler.find_bf_class(row['cwe_id'])

            if bf_class:
                dataset.at[i, 'bf_class'] = bf_class
        top_10_cwe_without_bf = list(dataset[dataset['bf_class'].isnull()]['cwe_id'].value_counts().head(10).keys())
        self.app.log.info(f"Top 10 CWE IDs without BF Class: {top_10_cwe_without_bf}")

        dataset = dataset[~dataset['bf_class'].isnull()]
        self.app.log.info(f"Entries with BF class: {len(dataset)}")
        top_5_languages = list(dataset.language.value_counts().head(5).keys())
        dataset = dataset[dataset.language.isin(top_5_languages)]
        self.app.log.info(f"Entries for top 5 Languages ({top_5_languages}): {len(dataset)}")

        dataset['language'] = dataset[~dataset.language.isnull()]['language'].apply(lambda x: str(x) if not pd.isna(x) else None)
        languages = dataset[~dataset['language'].isna()].language.unique()
        languages = [l for ls in languages for l in eval(ls)]
        languages = list(set(languages))
        node_labels = list(dataset['bf_class'].unique()) + list(languages)

        sources = []
        targets = []
        values = []
        link_labels = []

        for name, group in dataset.groupby(['bf_class', 'language']):
            source, target = name
            languages = eval(target)

            for language in languages:
                sources.append(node_labels.index(source))
                targets.append(node_labels.index(language))
                values.append(len(group))
                link_labels.append(source)

        Plotter(path=self.path).sankey(sources=sources, targets=targets, values=values, link_labels=link_labels,
                                       node_labels=node_labels, title="BF Class Top 5 Languages", opacity=0.6)


def load(app):
    app.handler.register(OSVprepare)
