import ast

import pandas as pd

from typing import Union

from tqdm import tqdm

from securityaware.handlers.plugin import PluginHandler
from securityaware.core.plotter import Plotter


class Generate(PluginHandler):
    """
        Generate plugin
    """

    class Meta:
        label = "generate"

    def __init__(self, **kw):
        super().__init__(**kw)
        self.negative_df_path = None
        self.lookup_df_path = None
        self.extensions = []
        self.target_files_factor = 3
        self.max_commits = None
        self.early_stopping_limit = None

    def run(self, dataset: pd.DataFrame, scenario: str = 'fix', augment: int = None, tokens: list = None,
            target_files_factor: int = 3, languages: list = None,  max_commits: int = None,
            **kwargs) -> Union[pd.DataFrame, None]:
        """Generates sampling strategy

        Args:
            scenario (str): one of the following scenarios fix, random, controlled
            languages (list): list of target programming languages
            augment (int): factor for augmenting the negative classes in the dataset
            tokens (list): GitHub API tokens
            max_commits (int): limit of commits to search
            target_files_factor (int): the factor used to gather target files for the negative dataset,
                                        e.g., #collected_files = #project_files * target_files_factor
        """
        self.app.log.info(f'{scenario} scenario...')
        langs = self.set_extensions(languages)
        self.app.log.info(f'Initial size: {len(dataset)}')
        if langs:
            new_dataset = []
            for idx, row in dataset.iterrows():
                if pd.isna(row['language']):
                    continue

                if ast.literal_eval(row['language']).intersection(langs):
                    new_dataset.append(row.to_dict())

            dataset = pd.DataFrame(new_dataset)
            self.app.log.info(f"Size after selecting samples for {langs}: {len(dataset)}")

        self.target_files_factor = target_files_factor
        self.max_commits = max_commits

        # TODO: pass through command line
        self.github_handler.tokens = tokens
        negative = self.generate_negative(dataset=dataset) if scenario.lower() != 'fix' else None
        scn = self.sampling_handler.get_scenario(dataset, scenario=scenario, negative=negative,
                                                 extension=self.extensions)
        self.app.log.info(f"Generating {scenario} scenario...")
        scn.generate()
        self.app.log.info(f"Cleaning {scenario}...")
        scn.remove_changes()
        scn.drop_duplicates()
        self.app.log.info(f"Augmenting {scenario}...")
        scn.augment(augment)
        scn.df['scenario'] = [scenario] * len(scn.df)

        return scn.df

    def set_extensions(self, languages: list):
        langs = []
        if languages:
            self.extensions = []
            extensions_mapping = {k.lower(): v for k, v in self.file_parser_handler.extension_mapping.items()}

            for lang in languages:
                if lang.lower() in extensions_mapping:
                    langs.append(lang)
                    self.extensions.extend(extensions_mapping[lang.lower()])
                else:
                    self.app.log.warning(f"Could not find mapping for language {lang}. Skipping...")

        return langs

    def update_lookup(self, status: str, lookup_df: pd.DataFrame, project: str, project_url: str) -> pd.DataFrame:
        project_lookup_idx = lookup_df.index[lookup_df['project_url'] == project_url] if not lookup_df.empty else []

        if len(project_lookup_idx) > 0:
            lookup_df.at[project_lookup_idx[0], 'status'] = status
        else:
            lookup = [{'project_name': project, 'project_url': project_url, 'status': status}]
            lookup_df = pd.concat([lookup_df, pd.DataFrame(lookup)], ignore_index=True)

        lookup_df.to_csv(str(self.lookup_df_path), index=False)

        return lookup_df

    def generate_negative(self, dataset: pd.DataFrame):
        # generate negatives for random and controlled scenarios
        self.negative_df_path = self.path / 'negative.csv'
        self.lookup_df_path = self.path / 'lookup.csv'

        if self.negative_df_path.exists():
            negative_samples = pd.read_csv(self.negative_df_path)
            #return negative_samples
        else:
            negative_samples = pd.DataFrame()

        if self.lookup_df_path.exists():
            lookup_df = pd.read_csv(self.lookup_df_path)
            visited_projects = lookup_df[lookup_df['status'].isin(['success', 'visited'])]['project_url'].unique()
        else:
            lookup_df = pd.DataFrame()
            visited_projects = []

        if len(visited_projects) > 0:
            self.app.log.info(f"Skipping {len(visited_projects)} visited projects")

        for project_url, group in tqdm(dataset[~dataset['project_url'].isin(visited_projects)].groupby('project_url')):
            owner, project = project_url.split("/")[3:5]
            repo = self.github_handler.get_repo(owner=owner, project=project)

            if repo is None:
                continue

            project_files = set([f for gf in group.files for f in ast.literal_eval(gf).keys()])
            target_files_count = len(project_files) * self.target_files_factor
            excluded_commits = set(list(group['before_first_fix_commit']))
            excluded_commits.update(set(list(group['last_fix_commit'])))

            if not negative_samples.empty:
                project_negative_samples = negative_samples[negative_samples['project_url'] == project_url]
                collected_samples = project_negative_samples[~project_negative_samples['file_path'].isnull()]

                if len(collected_samples) > 0:
                    self.app.log.info(f"Found {len(collected_samples)} files previously collected files for {project}")
                    target_files_count -= len(collected_samples)

                if len(project_negative_samples) > 0:
                    excluded_commits.update(set(project_negative_samples.sha.to_list()))

            samples, status = self.github_handler.repo_random_files_lookup(repo=repo, excluded_files=project_files,
                                                                           target_extensions=self.extensions,
                                                                           target_files_count=target_files_count,
                                                                           max_commits=self.max_commits,
                                                                           excluded_commits=excluded_commits)

            if samples.empty:
                # TODO: verify if this can happen, should not
                lookup_df = self.update_lookup('empty', lookup_df=lookup_df, project=project, project_url=project_url)
                continue

            samples['project_name'] = [project] * len(samples)
            samples['project_url'] = [project_url] * len(samples)

            negative_samples = pd.concat([negative_samples, samples], ignore_index=True)
            negative_samples.to_csv(str(self.negative_df_path), index=False)
            lookup_df = self.update_lookup(status, lookup_df=lookup_df, project=project, project_url=project_url)

        negative_samples = negative_samples[~negative_samples['file_path'].isnull()]
        return negative_samples

    def plot(self, dataset: pd.DataFrame, **kwargs):
        dataset = dataset[~dataset['bf_class'].isnull()]
        self.app.log.info(f"Entries with BF class: {len(dataset)}")

        Plotter(self.path, fig_size=(20,10)).bar_labels(dataset, column='cwe_id', y_label='Occurrences',
                                                        x_label='CWE-ID')
        Plotter(self.path).bar_labels(dataset, column='bf_class', y_label='Occurrences', x_label='BF Class')
        Plotter(self.path, fig_size=(20, 10)).bar_labels(dataset, column='project_name', y_label='Occurrences',
                                                         x_label='Project', top_n=50)


def load(app):
    app.handler.register(Generate)
