from pathlib import Path
from typing import Union

import pandas as pd

from securityaware.handlers.plugin import PluginHandler


class SeparateHandler(PluginHandler):
    """
        Separate plugin
    """
    class Meta:
        label = "separate"

    def run(self, node: dict, cell: dict, **kwargs) -> Union[pd.DataFrame, None]:
        """
            runs the plugin
        """
        files_path = cell['path'] / 'files'
        buggy_path = files_path / 'buggy'
        patch_path = files_path / 'patch'

        if 'raw' not in node:
            return None

        dataset = self.load_dataset(node['raw'][''])

        buggy_files = []
        patch_files = []

        for i, row in dataset.iterrows():
            a_new_file = buggy_path / row.a_proj / Path(row.a_path).parent / Path(row.a_file).name
            a_new_file.parent.mkdir(parents=True, exist_ok=True)

            with Path(row.a_file).open(mode="r") as rf, a_new_file.open(mode="w") as wf:
                self.app.log.info(f"Writing {row.a_file} to {a_new_file}.")
                wf.write(rf.read())

            b_new_file = patch_path / row.b_proj / Path(row.b_path).parent / Path(row.b_file).name
            b_new_file.parent.mkdir(parents=True, exist_ok=True)

            with Path(row.b_file).open(mode="r") as rf, b_new_file.open(mode="w") as wf:
                self.app.log.info(f"Writing {row.b_file} to {b_new_file}.")
                wf.write(rf.read())

            buggy_files.append(str(a_new_file))
            patch_files.append(str(b_new_file))

        dataset['a_file'] = buggy_files
        dataset['b_file'] = patch_files

        return dataset

    def post(self, node: dict, cell: dict) -> Union[pd.DataFrame, None]:
        if 'func' not in node:
            return None

        func_unsafe_dataset = pd.read_csv(str(node['func']['path'] / 'xss_advisories.codeql.func.csv'))

        del func_unsafe_dataset['func_id']
        del func_unsafe_dataset['n_mut']

        return func_unsafe_dataset


#    def post(self, path: Path, name: str, dataset: pd.DataFrame) -> pd.DataFrame:
        #unsafe_dataset = pd.read_csv(str(path / 'cwe-079.labels.csv'))
        #unsafe_dataset['ext'] = ['unsafe'] * len(unsafe_dataset)
        #unsafe_dataset.rename(columns={'ext': 'label'}, inplace=True)
        #projects, fpaths = [], []

        #for i, row in unsafe_dataset.iterrows():
        #    project = row.fpath.split('/')[0]
        #    fpath = row.fpath.replace(f"{project}/", '')
        #    projects.append(project)
        #    fpaths.append(fpath)

        #unsafe_dataset.project = projects
        #unsafe_dataset.fpath = fpaths

        #func_dataset = pd.read_csv(str(path.parent / 'func' / 'xss_advisories.func.csv'))
        #del func_dataset['func_id']
        #del func_dataset['n_mut']
        #safe_dataset = func_dataset[func_dataset['label'] == 'safe']
        #safe_dataset = safe_dataset[safe_dataset['fpath'].isin(func_unsafe_dataset['fpath'])]
        #return unsafe_dataset
        #return pd.concat([func_unsafe_dataset, safe_dataset])