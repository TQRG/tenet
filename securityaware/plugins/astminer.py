import hashlib
import re
import pandas as pd
import numpy as np

from pathlib import Path
from typing import Union
from tqdm import tqdm

from securityaware.core.exc import SecurityAwareError
from securityaware.core.plotter import Plotter
from securityaware.data.schema import ContainerCommand
from securityaware.handlers.plugin import PluginHandler

re_plain = '(?P<label>(\w+)),(?P<hash>(\w+)),(?P<pair_hash>(\w*)),(?P<fpath>([\w\/\.\_\-]+)),(?P<sline>(\d+)),(?P<scol>(\d+)),(?P<eline>(\d+)),(?P<ecol>(\d+))'
re_context_paths = '(?P<fpath>([\w\/\.\_-]+))\_(?P<sline>(\d+))\_(?P<scol>(\d+))\_(?P<eline>(\d+))\_(?P<ecol>(\d+)) (?P<label>(\w+)) (?P<hash>(\w+)) (?P<pair_hash>(\w*)) (?P<context_paths>([\w ,|]+))'


class ASTMinerHandler(PluginHandler):
    """
        ASTMiner plugin
    """

    class Meta:
        label = "astminer"

    def __init__(self, **kw):
        super().__init__(**kw)
        self.path_context_file = None
        self.extract_cp = None
        self.raw_files_path = None

    def run(self, dataset: pd.DataFrame, extract_cp: bool = True, image_name: str = "astminer",
            max_old_space_size: int = 8192, mutations: bool = False, **kwargs) -> Union[pd.DataFrame, None]:
        """
            runs the plugin
        """
        self.extract_cp = extract_cp
        self.path_context_file = (self.path / 'path_contexts.c2s')
        self.set('path_contexts', self.path_context_file)
        self.set('dataset', self.output)

        if not self.get('raw_fn_bounds_file'):
            self.app.log.warning(f"path to dataset with fn boundaries not instantiated.")
            return None

        self.raw_files_path = Path(str(self.get('raw_files_path')).replace(str(self.app.workdir), str(self.app.bind)))

        if not self.path_context_file.exists():
            # TODO: fix the node name
            container = self.container_handler.run(image_name=image_name, node_name=self.node.name)
            self.run_astminer(container, max_old_space_size, mutations)

        results = self.parse_path_context_file()

        if not results:
            raise AssertionError('Could not parse path context file')

        df = self.convert_to_dataframe(results)
        return self.add_cwe_ids(df, labels_df=dataset)

    def plot(self, dataset: pd.DataFrame, **kwargs):
        unsafe_samples = dataset[dataset.label == 'unsafe']

        if 'cwe' in dataset.columns:
            Plotter(self.path).bar_labels(unsafe_samples, column='cwe', y_label='Occurrences', x_label='CWE-ID')
        if 'sfp' in dataset.columns:
            unsafe_samples['sfp'] = unsafe_samples.sfp.apply(lambda x: self.cwe_list_handler.get_sfp_title(x))
            Plotter(self.path).bar_labels(unsafe_samples, column='sfp', y_label='Occurrences', x_label='SFP Cluster')
        if 'cp_size' in dataset.columns:
            Plotter(self.path).histogram_pairs(dataset, column='cp_size', x_label='Context paths size',
                                               filter_outliers=True)
        if 'fsize' in dataset.columns:
            Plotter(self.path).histogram_pairs(dataset, column='fsize', x_label='Function size', filter_outliers=True)

    def run_astminer(self, container, max_old_space_size: int, mutations: bool):
        export_cmd = ContainerCommand(org=f"export NODE_OPTIONS=\"--max-old-space-size={max_old_space_size}\"")

        raw_fn_bounds_file = Path(
            str(self.get('raw_fn_bounds_file')).replace(str(self.app.workdir), str(self.app.bind)))

        max_old_space_size_gb = round(max_old_space_size / 1024)
        max_mem = f"-Xmx{max_old_space_size_gb}g"
        max_size = f"-Xms{max_old_space_size_gb}g"

        astminer_jar_path = "../../../astminer/build/shadow/astminer.jar"
        astminer_cmd = ContainerCommand(org=f"java -jar {max_size} {max_mem} {astminer_jar_path}")

        astminer_cmd.org += ' ' + ('code2vec' if self.extract_cp else 'codebert')
        astminer_cmd.org += f" {self.raw_files_path} {self.container_handler.working_dir} {raw_fn_bounds_file} {1 if mutations else 0}"

        self.container_handler.run_cmds(container.id, [export_cmd, astminer_cmd])
        self.container_handler.stop(container)

    def parse_full_file_path(self, fpath: str):
        fpath_split = fpath.replace(str(self.raw_files_path), '').split('/')

        if fpath_split[0] == '':
            fpath_split = fpath_split[1:]

        owner, project, version, *file, = fpath_split

        return owner, project, version, '/'.join(file)

    def parse_path_context_file(self):
        results = []
        self.app.log.info(f"Parsing output file {self.path / 'path_contexts.c2s'}")

        with (self.path / 'path_contexts.c2s').open(mode='r') as output:
            reg_exp = re_context_paths if self.extract_cp else re_plain

            for i, line in enumerate(output.read().splitlines()):
                match = re.match(pattern=reg_exp, string=line)

                if match:
                    results.append(match.groupdict())
                else:
                    raise ValueError(f"Could not match line {i} {line}")

        return results

    @staticmethod
    def compute_missing_hash(row) -> str:
        if row['pair_hash'] == '' or pd.isnull(row['pair_hash']) or pd.isna(row['pair_hash']):
            row_str = f"{row.owner}_{row.project}_{row.version}_{row.fpath}_{row.sline}_{row.scol}_{row.eline}_{row.ecol}"
            return hashlib.md5(row_str.encode()).hexdigest()

        return row['pair_hash']

    def add_cwe_ids(self, dataset: pd.DataFrame, labels_df: pd.DataFrame):
        labels_df.rename(columns={'label': "cwe"}, inplace=True)

        unsafe_labels_df = labels_df[labels_df["cwe"] != 'safe']
        # add primary software fault pattern clusters
        unsafe_labels_df['sfp'] = unsafe_labels_df.cwe.apply(lambda x: self.cwe_list_handler.find_primary_sfp_cluster(x, only_id=True))
        before_match_cwes = len(unsafe_labels_df)
        merge_on = ['owner', 'project', 'version', 'fpath', 'sline', 'scol', 'eline', 'ecol']
        unsafe_labels_df = unsafe_labels_df[merge_on + ['cwe', 'sfp']]

        df = pd.merge(dataset, unsafe_labels_df, on=merge_on,  how='left')
        after_match_cwes = len(df[~df["cwe"].isnull()])
        difference = before_match_cwes - after_match_cwes
        df.drop_duplicates(inplace=True)

        if difference > 0:
            self.app.log.warning(f"Could not match {difference} unsafe functions.")

        self.app.log.info(f"After labels match: {len(df)}")

        return df

    def convert_to_dataframe(self, results: list):
        df = pd.DataFrame(results)
        loc_cols = ['sline', 'scol', 'eline', 'ecol']

        if not all([col in df.columns for col in loc_cols]):
            raise SecurityAwareError(f"Columns not found {loc_cols} in the astminer output.")

        df.drop_duplicates(subset=['hash'], inplace=True)
        self.app.log.info(f'Initial size: {len(results)} | Without duplicates: {len(df)}')

        df['owner'], df['project'], df['version'], df['fpath'] = zip(*df.fpath.apply(self.parse_full_file_path))
        df['input'] = [None] * len(df)
        df['pair_hash'] = df.apply(self.compute_missing_hash, axis=1)

        # Convert function locations to integer
        df[loc_cols] = df[loc_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.int64)
        df = self.get_functions(df)

        if not df.empty and 'input' in df.columns:
            df['fsize'] = df.apply(lambda r: len(r.input.splitlines()), axis=1)

        if self.extract_cp:
            # Calculate context paths size
            df['cp_size'] = df.apply(lambda r: len(r['context_paths'].split()), axis=1)

        return df

    def extract_functions(self, target_file: Path, rows: pd.Series):
        with target_file.open(encoding="latin-1") as code_file:
            self.app.log.info(f"Processing {target_file} rows {(rows.index[0], rows.index[-1])}")
            lines = code_file.readlines()

            for idx, row in rows.iterrows():
                code = lines[(row.sline - 1):row.eline]

                if len(code) > 1:
                    code[0] = code[0][row.scol:]
                    code[-1] = code[-1][:row.ecol]
                else:
                    code[0] = code[0][row.scol:row.ecol]

                code = ''.join(code)
                rows.loc[idx, 'input'] = code

                if code is None:
                    self.app.log.warning(f"Could not parse function {row.hash}")
                    continue

        return rows[~rows.input.isnull()]

    def get_functions(self, funcs_df: pd.DataFrame):
        initial_size = len(funcs_df)
        funcs_df.reset_index(inplace=True)

        for target_file, rows in tqdm(funcs_df.groupby(['owner', 'project', 'version', 'fpath'])):
            target_file = self.get('raw_files_path') / '/'.join(target_file)

            if not target_file.exists():
                self.app.log.warning(f"{target_file} not found")
                continue

            self.multi_task_handler.add(target_file=target_file, rows=rows)

        self.multi_task_handler(func=self.extract_functions)

        funcs_df = pd.concat(self.multi_task_handler.results()).sort_index()

        if initial_size != len(funcs_df):
            self.app.log.warning(f"Dataframe does not have initial functions: {initial_size} -> {len(funcs_df)}")

        return funcs_df


def load(app):
    app.handler.register(ASTMinerHandler)
