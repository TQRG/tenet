import re
import pandas as pd
import numpy as np

from pathlib import Path
from typing import Union
from tqdm import tqdm

from securityaware.core.plotter import Plotter
from securityaware.data.schema import ContainerCommand
from securityaware.handlers.plugin import PluginHandler

re_plain = '(?P<label>(\w+)),(?P<hash>(\w+)),(?P<fpath>([\w\/\.\_\-]+)),(?P<sline>(\d+)),(?P<scol>(\d+)),(?P<eline>(\d+)),(?P<ecol>(\d+))'
re_context_paths = '(?P<fpath>([\w\/\.\_-]+))\_(?P<sline>(\d+))\_(?P<scol>(\d+))\_(?P<eline>(\d+))\_(?P<ecol>(\d+)) (?P<label>(\w+)) (?P<hash>(\w+)) (?P<input>([\w ,|]+))'


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

    def run(self, dataset: pd.DataFrame, extract_cp: bool = True, image_name: str = "astminer",
            max_old_space_size: int = 8192, mutations: bool = False, **kwargs) -> Union[pd.DataFrame, None]:
        """
            runs the plugin
        """
        self.extract_cp = extract_cp
        self.path_context_file = (self.path / 'path_contexts.c2s')
        self.set('path_contexts', self.path_context_file)

        if not self.get('raw_fn_bounds_file'):
            self.app.log.warning(f"path to dataset with fn boundaries not instantiated.")
            return None

        if not self.path_context_file.exists():
            # TODO: fix the node name
            container = self.container_handler.run(image_name=image_name, node_name=self.node.name)
            export_cmd = ContainerCommand(org=f"export NODE_OPTIONS=\"--max-old-space-size={max_old_space_size}\"")
            astminer_cmd = ContainerCommand(org="java -jar -Xms4g -Xmx4g ../../../astminer/build/shadow/astminer.jar")
            astminer_cmd.org += ' ' + ('code2vec' if extract_cp else 'codebert')
            raw_files_path = Path(str(self.get('raw_files_path')).replace(str(self.app.workdir), str(self.app.bind)))
            raw_fn_bounds_file = Path(str(self.get('raw_fn_bounds_file')).replace(str(self.app.workdir), str(self.app.bind)))
            astminer_cmd.org += f" {raw_files_path} {self.container_handler.working_dir} {raw_fn_bounds_file} {1 if mutations else 0}"
            self.container_handler.run_cmds(container.id, [export_cmd, astminer_cmd])
            self.container_handler.stop(container)

        return self.parse_path_context_file()

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

        if not results:
            raise AssertionError('Could not parse path context file')

        df = pd.DataFrame(results)
        df.drop_duplicates(inplace=True)
        self.app.log.info(f'Initial size: {len(results)} | Without duplicates: {len(df)}')

        df['fpath'] = df.fpath.apply(lambda x: x.replace(str(self.app.bind), str(self.app.workdir)))

        if 'input' not in df.columns:
            df['input'] = [None] * len(df)

        loc_cols = ['sline', 'scol', 'eline', 'ecol']

        if all([col in df.columns for col in loc_cols]):
            # Convert function locations to integer
            df[loc_cols] = df[loc_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.int64)

            if not self.extract_cp:
                df = self.get_functions(df)

                if not df.empty and 'input' in df.columns:
                    df['fsize'] = df.apply(lambda r: len(r.input.splitlines()), axis=1)
            else:
                # Calculate function size
                df['fsize'] = df.apply(lambda r: 1 if r.sline == r.eline else r.eline - r.sline, axis=1)
                # Calculate context paths size
                df['cp_size'] = df.apply(lambda r: len(r.input.split()), axis=1)

        if 'cp_size' in df.columns:
            Plotter(self.path).histogram_pairs(df, column='cp_size', x_label='Context paths size', filter_outliers=True)
        if 'fsize' in df.columns:
            Plotter(self.path).histogram_pairs(df, column='fsize', x_label='Function size', filter_outliers=True)

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
                code = self.code_parser_handler.filter_comments(code)
                rows.loc[idx, 'input'] = code

                if code is None:
                    self.app.log.warning(f"Could not parse function {row.hash}")
                    continue

        return rows[~rows.input.isnull()]

    def get_functions(self, funcs_df: pd.DataFrame):
        initial_size = len(funcs_df)
        funcs_df.reset_index(inplace=True)

        for target_file, rows in tqdm(funcs_df.groupby(['fpath'])):
            target_file = Path(target_file)

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
