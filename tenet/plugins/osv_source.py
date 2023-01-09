import os
import zipfile
import glob
import shutil

import pandas as pd
import numpy as np

from pathlib import Path
from typing import Union
from tqdm import tqdm
from google.cloud import storage

from tenet.handlers.plugin import PluginHandler
from tenet.utils.misc import split_commits, filter_references, normalize_commits,\
    filter_commits_by_source, load_json_file, create_df, get_source
from tenet.core.plotter import Plotter
from tenet.utils.ghsa import dump


class OSVSource(PluginHandler):
    """
        OSVSource plugin
    """

    class Meta:
        label = "osv_source"
        
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


    def run(self, 
            dataset: pd.DataFrame, 
            normalize: bool = True,
            code_source: str = 'github',
            **kwargs) -> Union[pd.DataFrame, None]:
        """
            runs the plugin
        """
        
        token, df_ecosystems = kwargs['tokens'][-1], []
        # set folder for GHSA dump
        ecosystem = 'GHSA'
        ghsa_metadata_path = Path(self.path / ecosystem)
        self.app.log.info(f"Making directory for {str(ghsa_metadata_path)}.")
        if not os.path.exists(ghsa_metadata_path):
            ghsa_metadata_path.mkdir()
        # get GitHub advisories (GHSA)
        self.app.log.info(f"Downloading {ecosystem} metadata...")
        dump(ghsa_metadata_path, token)

        # df_metadata_path = self.path / kwargs['dataset_name']
        ghsa_results = self.parse_json_ghsa(fin_path=ghsa_metadata_path)
        df_ecosystems.append(ghsa_results)
        self.app.log.info(f"Found {len(ghsa_results)} vulnerabilities from {ecosystem}.")
        shutil.rmtree(ghsa_metadata_path)
        
        client = storage.Client.create_anonymous_client()
        bucket = client.bucket('osv-vulnerabilities')
        for ecosystem in kwargs['ecosystems']:
            # get ecosystem blob
            object_name = f"{ecosystem}/all.zip"
            blob = bucket.blob(object_name)
            # download blob to local folder
            self.app.log.info(f"Downloading {ecosystem} metadata...")
            blob_zip_file_path = Path(f"{self.path}/{ecosystem}.zip")
            blob.download_to_filename(blob_zip_file_path)
            
            # creata output dir
            ecosystem_metadata_path = Path(self.path / ecosystem)
            if not os.path.exists(ecosystem_metadata_path):
                ecosystem_metadata_path.mkdir()

            # unzip data
            with zipfile.ZipFile(blob_zip_file_path,"r") as zip_ref:
                zip_ref.extractall(ecosystem_metadata_path)
                
            # remove ecosystem.zip
            blob_zip_file_path.unlink()
            
            ecosystem_results = self.parse_json_ecosystem(
                    ecosystem, 
                    fin_path=ecosystem_metadata_path 
                    ) 
            df_ecosystems.append(ecosystem_results)
            self.app.log.info(f"Found {len(ecosystem_results)} vulnerabilities from {ecosystem}.")
            shutil.rmtree(ecosystem_metadata_path)
        
        df = pd.concat(df_ecosystems, ignore_index=True)
        self.app.log.info(f"Initial DataFrame size: {len(df)}")
        df['refs'] = df['refs'].apply(lambda ref: np.nan if ref == 'nan' else ref)
        # drop rows without refs
        df = df.dropna(subset=['refs'])
        self.app.log.info(f"Size after nan refs drop: {len(df)}")
        
        if normalize:
            df = self.normalize(df, code_source)
        
        return df
    
    # TODO: This function is common to NVD and OSV;
    # what should we do? the plot function should 
    # also be repeated.
    def normalize(self, df: pd.DataFrame, code_source: str):
        # normalize refs
        df['refs'] = df['refs'].apply(lambda ref: split_commits(ref))
        # drop cases with no refs
        df = df.dropna(subset=['refs'])
        self.app.log.info(f"Size after nan refs drop: {len(df)}")
        df = filter_references(df)
        self.app.log.info(f"Size after filtering refs: {len(df)}")
        df = normalize_commits(df)
        self.app.log.info(f"Size after normalizing refs: {len(df)}")

        if code_source:
            df = filter_commits_by_source(df, source=code_source)

        return df
    
    def parse_json_ecosystem(self, ecosystem: str, fin_path: Path) -> pd.DataFrame:
        """Processes vulnerability reports per ecosystem.
        Args:
            ecosystem (string): ecosystem name.
            fout (string): path to save the vulnerabilities data.
        """

        def get_ranges(data):
            """Get the range of code repository that are
            affected by the vulnerability.
            Args:
                data (dict): vulnerability data.
            Returns:
                ranges: if exists, else return NaN
            """
            if "affected" in data.keys():
                if "ranges" in data["affected"][0].keys():
                    return data["affected"][0]["ranges"]     
            return []

        def normalize_ref(repo):
            return repo.replace(".git", "") if ".git" in repo else repo

        def get_commits(data):
            """Get commits that introduce and fix the
            vulnerbility.
            Args:
                data (dict): vulnerability data.
            Returns:
                set: commits that introduce the vulnerability
                set: commits that fix the vulnerability
            """
            introduced, fixed, ranges = set(), set(), get_ranges(data)
                        
            if ranges:
                for range in ranges:
                    if range["type"] == "GIT":
                        for event in range["events"]:
                            if "introduced" in event.keys():
                                if event["introduced"] != "0":
                                    repo = normalize_ref(range["repo"])
                                    introduced.add(f"{repo}/commit/{event['introduced']}")
                            elif "fixed" in event.keys():
                                repo = normalize_ref(range["repo"])
                                fixed.add(f"{repo}/commit/{event['fixed']}")
            return (
                introduced if introduced else np.nan,
                fixed if fixed else np.nan,
            )

        def get_severity(data):
            if "database_specific" in data.keys():
                if "severity" in data["database_specific"].keys():
                    return data["database_specific"]["severity"]
            if "affected" in data.keys():
                if "ecosystem_specific" in data["affected"][0].keys():
                    eco_spec = data["affected"][0]["ecosystem_specific"]
                    if "severity" in eco_spec.keys():
                        return eco_spec["severity"]
            return np.nan

        def get_cwes(data):
            cwes = set()
            if "database_specific" in data.keys():
                if "cwe_ids" in data["database_specific"].keys():
                    cwe_ids = data["database_specific"]["cwe_ids"]
                    if cwes:
                        return str(set(cwe_ids))
            if "affected" in data.keys():
                if "database_specific" in data["affected"][0].keys():
                    db_spec = data["affected"][0]["database_specific"]
                    if "cwes" in db_spec.keys():
                        for cwe in db_spec["cwes"]:
                            cwes.add(cwe["cweId"])
            return cwes if cwes else np.nan

        def get_score(data):
            if "affected" in data.keys():
                if "database_specific" in data["affected"][0].keys():
                    db_spec = data["affected"][0]["database_specific"]
                    if "cvss" in db_spec.keys():
                        if db_spec["cvss"] and type(db_spec["cvss"]) != str:
                            if "score" in db_spec["cvss"].keys():
                                return db_spec["cvss"]["score"]
            return np.nan

        def get_aliases(data):
            if 'aliases' in data.keys():
                aliases = data['aliases']
                if aliases:
                    return str(set(aliases))
            return np.nan

        df, first = None, True

        # iterate over the ecosystem vuln reports
        for report_path in glob.glob(f"{str(fin_path)}/*.json"):

            # load json file
            data = load_json_file(report_path)

            refs = self.get_references(data)
            introduced, fixed = get_commits(data)
            if pd.notna(fixed) and pd.notna(refs):
                refs = set.union(eval(refs), fixed)

            vuln_data = {
                "ecosystem": ecosystem,
                "vuln_id": self.get_field(data, "id"),
                "aliases": get_aliases(data),
                "summary": self.get_field(data, "summary"),
                "details": self.get_field(data, "details"),
                "modified_date": self.get_field(data, "modified"),
                "published_date": self.get_field(data, "published"),
                "severity": get_severity(data),
                "score": get_score(data),
                "cwe_id": get_cwes(data),
                "refs": str(refs),
                "introduced": str(introduced),
            }
            
            if first:
                df, first = create_df(vuln_data), False
            else:
                df = pd.concat([df, create_df(vuln_data)], ignore_index=True)

        return df

    
    def parse_json_ghsa(self, fin_path: Path, ecosystem="GHSA") -> pd.DataFrame:
        """Processes GHSA vulnerability reports.
            Args:
                fout_path (string): Path to save the vulnerabilities data.
        """

        def get_cwes(data):
            """Get weakness ids from vulnerability.
            Args:
                data (dict): vulnerability report.
            Returns:
                set: set of weakness ids in report.
            """
            cwes = set()
            for cwe in data["cwes"]["nodes"]:
                cwes.add(cwe["cweId"])
            return str(cwes) if len(cwes) > 0 else np.nan

        def get_aliases(data):
            """Get aliases of vulnerability.
            Args:
                data (dict): vulnerability report.
            Returns:
                set: set of aliases in report.
            """
            aliases = set()
            for id in data["identifiers"]:
                if id["value"] != data["ghsaId"]:
                    aliases.add(id["value"])
            return str(aliases) if len(aliases) > 0 else np.nan

        def get_score(data):
            """Get vulnerability score.
            Args:
                data (dict): vulnerability report.
            Returns:
                float: vulnerability score.
            """
            return data["cvss"]["score"]
        
        df, first = None, True
        
        # get reports available for GHSA ecosystem
        reports = [
            os.path.join(fin_path, f) for f in os.listdir(fin_path) \
                if os.path.isfile(os.path.join(fin_path, f))
        ]
        
        # iterate over the ecosystem vulns
        for file_path in tqdm(reports):

            # load json file
            data = load_json_file(file_path)

            vuln_data = {
                "ecosystem": ecosystem,
                "vuln_id": data["ghsaId"],
                "summary": self.get_field(data, "summary").strip(),
                "details": self.get_field(data, "description").strip(),
                "aliases": get_aliases(data),
                "modified_date": self.get_field(data, "updatedAt"),
                "published_date": self.get_field(data, "publishedAt"),
                "severity": self.get_field(data, "severity"),
                "score": get_score(data),
                "cwe_id": get_cwes(data),
                "refs": self.get_references(data),
            }

            if first:
                df, first = create_df(vuln_data), False
            else:
                df = pd.concat([df, create_df(vuln_data)], ignore_index=True)

        return df
    
    @staticmethod
    def get_field(data, field):
        """Verifies if field exists in the vulnerability report.
        Args:
            data (dict): vulnerability report.
            field (string): field to verify.
        Returns:
            string: value from field in data.
        """
        return data[field] if field in data.keys() else np.nan
                
    @staticmethod
    def get_references(data):
        """Gets references from vulnerability report.
        Args:
            data (dict): vulnerability report.
        Returns:
            set: set of references in report.
        """
        if "references" in data.keys():
            refs = set([ref["url"] for ref in data["references"] if "url" in ref.keys()])
            if refs: return str(refs)
        return np.nan


def load(app):
    app.handler.register(OSVSource)
