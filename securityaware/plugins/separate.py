import pandas as pd

from pathlib import Path
from typing import Union

from securityaware.handlers.plugin import PluginHandler


class SeparateHandler(PluginHandler):
    """
        Separate plugin
    """
    class Meta:
        label = "separate"

    def run(self, dataset: pd.DataFrame, **kwargs) -> Union[pd.DataFrame, None]:
        """
            runs the plugin
        """

        out_files_path = Path(self.node.path, 'files')
        buggy_path = out_files_path / 'vuln'
        patch_path = out_files_path / 'patch'

        buggy_files = []
        patch_files = []

        for i, row in dataset.iterrows():
            a_new_file = buggy_path / row.a_proj / row.a_path
            a_new_file.parent.mkdir(parents=True, exist_ok=True)
            a_path = Path(self.get('files_path'), row.a_proj, row.a_path)

            with a_path.open(mode="r") as rf, a_new_file.open(mode="w") as wf:
                self.app.log.info(f"Writing {row.a_file} to {a_new_file}.")
                wf.write(rf.read())

            b_new_file = patch_path / row.b_proj / row.b_path
            b_new_file.parent.mkdir(parents=True, exist_ok=True)
            b_path = Path(self.get('files_path'), row.b_proj, row.b_path)

            with b_path.open(mode="r") as rf, b_new_file.open(mode="w") as wf:
                self.app.log.info(f"Writing {row.b_file} to {b_new_file}.")
                wf.write(rf.read())

            buggy_files.append(str(a_new_file))
            patch_files.append(str(b_new_file))

        dataset['a_file'] = buggy_files
        dataset['b_file'] = patch_files

        return dataset


def load(app):
    app.handler.register(SeparateHandler)
