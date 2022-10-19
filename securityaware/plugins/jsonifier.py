import jsonlines
import pandas as pd

from typing import Union
from pathlib import Path

from securityaware.core.sampling.offset import to_offset
from securityaware.handlers.plugin import PluginHandler


class JSONifierHandler(PluginHandler):
    """
        JSONLinesHandler plugin
    """

    class Meta:
        label = "jsonifier"

    def __init__(self, **kw):
        super().__init__(**kw)

    def run(self, dataset: pd.DataFrame, offset: bool = False, **kwargs) -> Union[pd.DataFrame, None]:
        """
            runs the plugin
        """

        train_data_path = self.get('train_data_path')
        val_data_path = self.get('val_data_path')
        test_data_path = self.get('test_data_path')
        raw_files_path_str = self.get('raw_files_path')

        if not train_data_path:
            self.app.log.error(f"Train data path not instantiated")
            return None

        if not val_data_path:
            self.app.log.error(f"Val data path not instantiated")
            return None

        if not test_data_path:
            self.app.log.error(f"Test data path not instantiated")
            return None

        if not raw_files_path_str:
            self.app.log.error(f"Raw files path not instantiated")
            return None

        raw_files_path = Path(raw_files_path_str)

        if not raw_files_path.exists():
            self.app.log.warning(f"{raw_files_path} not found")
            return None

        split_files = [train_data_path, val_data_path, test_data_path]
        jsonlines_files = {}
        offset_files = {}

        for file in split_files:
            jsonlines_path = f'{file.stem}_lines'
            jsonlines_files[jsonlines_path] = Path(self.path, f'{file.stem}.jsonl')
            self.set(jsonlines_path, jsonlines_files[jsonlines_path])

        if offset:
            for file in split_files:
                offset_path = f'{file.stem}_offset_path'
                offset_files[offset_path] = Path(self.path, f'{file.stem}_offset_dict.json')
                self.set(offset_path, offset_files[offset_path])

        for jsonlines_file, split_file in zip(jsonlines_files.values(), split_files):
            self.app.log.info(f"======== Building JSONLines for {split_file}... ========")

            funcs_df = pd.read_csv(split_file)
            funcs_df.rename(columns={'input': 'code'}, inplace=True)
            funcs_df['label'] = self.convert_labels(funcs_df['label'])

            with jsonlines.open(str(jsonlines_file), mode='a') as output_file:
                for i, row in funcs_df.iterrows():
                    output_file.write({'code': row.code, 'label': row.label})

        if offset:
            for jsonlines_file, offset_file in zip(jsonlines_files.values(), offset_files.values()):
                to_offset(jsonlines_file, offset_file)

        return dataset


def load(app):
    app.handler.register(JSONifierHandler)
