import json
from pathlib import Path

import pandas as pd
import numpy as np

from typing import Union, Tuple, List
from tqdm import tqdm

from securityaware.handlers.plugin import PluginHandler
from securityaware.utils.misc import split_commits, filter_references, get_source, normalize_commits,\
    filter_commits_by_source
from securityaware.core.plotter import Plotter


class NVDSource(PluginHandler):
    """
        NVDSource plugin
    """

    class Meta:
        label = "nvd_source"

    def plot(self, dataset: pd.DataFrame, **kwargs):
        """ Print commits statistics. """

        # get number of commits involved in each patch
        dataset['n_commits'] = dataset['code_refs'].transform(lambda x: len(x))

        self.app.log.info(f"Plotting bar chart with number of commits for each fix")
        Plotter(path=self.path).bar_labels(df=dataset, column='n_commits', y_label='Count', x_label='#commits',
                                           bar_value_label=False, rotate_labels=False)

        # get commits source
        sources = []
        for source in dataset['code_refs'].transform(get_source):
            sources += source

        # iterate over the different sources
        for source in set(sources):
            n_source = len([s for s in sources if s == source])
            if source == 'bitbucket':
                print(f"{source}\t{n_source}\t\t{(n_source / len(sources)) * 100:.2f}%")
            else:
                print(f"{source}\t\t{n_source}\t\t{(n_source / len(sources)) * 100:.2f}%")

    def run(self, dataset: pd.DataFrame, years: Union[Tuple, List] = (2002, 2022), normalize: bool = True,
            code_source: str = 'github',
            base_url: str = 'https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-{year}.json.zip',
            **kwargs) -> Union[pd.DataFrame, None]:
        """
            runs the plugin
        """

        if isinstance(years, tuple):
            if len(years) > 2 or len(years) < 2:
                self.app.log.error(f"'years' argument must be a tuple of size 2 (min_year, max_year) or a list.")
                return None

            years = range(years[0], years[1]+1, 1)

        if '{year}' not in base_url:
            self.app.lof.error("Base URL must include {year} field")
            return None

        # download from source and extract
        for year in tqdm(years):
            url = base_url.format(year=year)
            self.multi_task_handler.add(url=url, extract=True)

        self.multi_task_handler(func=self.download_file_from_url)
        results = self.multi_task_handler.results()
        del self.multi_task_handler

        # parse json files into a single dataframe
        for _, file_path in tqdm(results):
            self.multi_task_handler.add(file_path=file_path)

        self.multi_task_handler(func=self.parse_json)
        results = self.multi_task_handler.results()

        df = pd.concat(results, ignore_index=True)
        self.app.log.info(f"Initial DataFrame size: {len(df)}")
        # drop rows without refs
        df = df.dropna(subset=['refs'])
        self.app.log.info(f"Size after nan refs drop: {len(df)}")

        if normalize:
            df = self.normalize(df, code_source)

        return df

    def normalize(self, df: pd.DataFrame, code_source: str):
        # normalize refs
        df['refs'] = df['refs'].apply(lambda ref: split_commits(ref))
        # drop cases with no refs
        df = df.dropna(subset=['refs'])
        self.app.log.info(f"Size after null refs drop: {len(df)}")
        df = filter_references(df)
        self.app.log.info(f"Size after filtering refs: {len(df)}")
        df = normalize_commits(df)
        self.app.log.info(f"Size after normalizing refs: {len(df)}")

        if code_source:
            df = filter_commits_by_source(df, source=code_source)

        return df

    def parse_json(self, file_path: Path) -> pd.DataFrame:
        df_data = []
        self.app.log.info(f"Parsing {file_path}...")
        df_file_path = self.path / f"{file_path.stem}.csv"

        if df_file_path.exists():
            return pd.read_csv(str(df_file_path))

        with file_path.open(mode='r') as f:
            cve_ids = json.load(f)["CVE_Items"]

            for cve in cve_ids:
                cve_data = {
                    "cve_id": self.get_cve(cve),
                    "cwes": self.get_cwe_ids(cve),
                    "description": self.get_description(cve),
                    "severity": self.get_severity(cve),
                    "exploitability": self.get_exploitability(cve),
                    "impact": self.get_impact(cve),
                    "published_date": self.get_published_date(cve),
                    "last_modified_date": self.get_last_modified_date(cve),
                    "refs": self.get_references(cve),
                }
                df_data.append(cve_data)

            df = pd.DataFrame(df_data)
            self.app.log.info(f"Saving to {df_file_path}")
            df.to_csv(str(df_file_path))

            return df

    @staticmethod
    def get_cwe_ids(cve):
        cwes = set()
        for data in cve["cve"]["problemtype"]["problemtype_data"]:
            for cwe in data["description"]:
                cwes.add(cwe["value"])
        return str(cwes) if len(cwes) > 0 else np.nan

    @staticmethod
    def get_cve(data: pd.DataFrame):
        return data["cve"]["CVE_data_meta"]["ID"]

    @staticmethod
    def get_description(data):
        return data["cve"]["description"]["description_data"][0]["value"]

    @staticmethod
    def get_published_date(data):
        return data["publishedDate"]

    @staticmethod
    def get_last_modified_date(data):
        return data["lastModifiedDate"]

    @staticmethod
    def get_severity(data):
        if data["impact"]:
            if "baseMetricV2" in data["impact"].keys():
                return data["impact"]["baseMetricV2"]["severity"]
        return np.nan

    @staticmethod
    def get_exploitability(data):
        if data["impact"]:
            if "baseMetricV2" in data["impact"].keys():
                return data["impact"]["baseMetricV2"]["exploitabilityScore"]
        return np.nan

    @staticmethod
    def get_impact(data):
        if data["impact"]:
            if "baseMetricV2" in data["impact"].keys():
                return data["impact"]["baseMetricV2"]["impactScore"]
        return np.nan

    @staticmethod
    def get_references(data):
        refs = set()
        for ref in data["cve"]["references"]["reference_data"]:
            refs.add(ref["url"])
        return str(refs) if len(refs) > 0 else np.nan


def load(app):
    app.handler.register(NVDSource)
