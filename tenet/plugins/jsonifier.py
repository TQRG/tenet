import jsonlines
import pandas as pd

from typing import Union
from pathlib import Path

from tenet.core.sampling.offset import to_offset
from tenet.handlers.plugin import PluginHandler


class JSONifierHandler(PluginHandler):
    """
        JSONLinesHandler plugin
    """

    class Meta:
        label = "jsonifier"

    def __init__(self, **kw):
        super().__init__(**kw)
        self.jsonlines_files = {}
        self.offset_files = {}

    def set_sources(self):
        for name, file in self.sinks.items():
            if name == 'raw_files_path':
                continue
            jsonlines_path = f'{file.stem}_lines'
            self.jsonlines_files[jsonlines_path] = Path(self.path, f'{file.stem}.jsonl')
            self.set(jsonlines_path, self.jsonlines_files[jsonlines_path])

        if 'offset' in self.node.kwargs and self.node.kwargs['offset']:
            for name, file in self.sinks.items():
                if name == 'raw_files_path':
                    continue
                offset_path = f'{file.stem}_offset_path'
                self.offset_files[offset_path] = Path(self.path, f'{file.stem}_offset_dict.json')
                self.set(offset_path, self.offset_files[offset_path])

    def get_sinks(self):
        self.get('train_data_path')
        self.get('val_data_path')
        self.get('test_data_path')
        self.get('raw_files_path')

    def run(self, dataset: pd.DataFrame, offset: bool = False, **kwargs) -> Union[pd.DataFrame, None]:
        """
            runs the plugin
        """

        for jsonlines_file, (name, split_file) in zip(self.jsonlines_files.values(), self.sinks.items()):
            if name == 'raw_files_path':
                continue
            self.app.log.info(f"======== Building JSONLines for {split_file}... ========")

            funcs_df = pd.read_csv(split_file)
            funcs_df.rename(columns={'input': 'code'}, inplace=True)
            funcs_df['code'] = funcs_df.code.apply(lambda x: self.code_parser_handler.filter_comments(x))
            funcs_df['label'] = self.convert_labels(funcs_df['label'])

            with jsonlines.open(str(jsonlines_file), mode='a') as output_file:
                for i, row in funcs_df.iterrows():
                    output_file.write({'code': row.code, 'label': row.label})

        if offset:
            for jsonlines_file, offset_file in zip(self.jsonlines_files.values(), self.offset_files.values()):
                to_offset(jsonlines_file, offset_file)

        return dataset


def load(app):
    app.handler.register(JSONifierHandler)
