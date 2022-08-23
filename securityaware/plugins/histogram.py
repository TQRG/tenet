import csv
import pandas as pd

from typing import Union
from pathlib import Path
from tqdm import tqdm

from securityaware.core.sampling.jsonify import to_json, to_dict
from securityaware.core.sampling.offset import to_offset
from securityaware.core.sampling.balance import oversampling, disjoint_smote, disjoint_smote_hash, \
    random_undersampling, split_data, one_one_ratio, disjoint, disjoint_hash, unique, unique_hash
from securityaware.data.dataset import XYDataset, Dataset, SplitDataset
from securityaware.handlers.plugin import PluginHandler


class SamplingHandler(PluginHandler):
    """
        Separate plugin
    """

    class Meta:
        label = "sampling"

    def __init__(self, **kw):
        super().__init__(**kw)
        self.token = None
        self.balance_techniques = ['over', 'disj_over', 'disj_smote', 'unique', 'undersampling', 'stratified', '1_to_1']

    def __call__(self, technique: str, seed: int, x: Dataset, y: Dataset, offset: bool = False) -> SplitDataset:
        if technique == 'over':
            return oversampling(x, y, seed)
        elif technique == 'disj_smote':
            if offset:
                return disjoint_smote_hash(x, y, seed)
            else:
                return disjoint_smote(x, y, seed)
        elif technique == 'disj_over':
            if offset:
                return disjoint_hash(x, y, seed)
            else:
                return disjoint(x, y, seed)
        elif technique == 'unique':
            if offset:
                return unique_hash(x, y, seed)
            else:
                return unique(x, y, seed)
        elif technique == '1_to_1':
            return one_one_ratio(x, y, seed)
        elif technique == 'random_undersampling':
            return random_undersampling(x, y, seed)
        else:
            self.app.log.info('No balancing technique applied.')
            return split_data(x, y, seed)

    def run(self, dataset: pd.DataFrame, technique: str = "", seed: int = 0, offset: bool = False,
            **kwargs) -> Union[pd.DataFrame, None]:
        """
            runs the plugin
        """

        self.set('train_data', Path(self.path, 'train.raw.txt'))
        self.set('val_data', Path(self.path, 'validation.raw.txt'))
        self.set('test_data', Path(self.path, 'test.raw.txt'))

        if technique and technique not in self.balance_techniques:
            self.app.log.warning(f"Technique must be one of the following: {self.balance_techniques}")
            return None

        self.app.log.info((f"Sampling with {technique}.\n" if technique else "") + f"Saving results to {self.path}")

        dataset_file = Path(self.get('context_paths_file'))
        datasets = self.get_datasets(dataset_file=dataset_file, delimiter=',' if offset else ' ')

        self.app.log.info(f"Dataset has {len(datasets.x)} samples.")
        split_dataset = self.__call__(x=datasets.x, y=datasets.y, offset=offset, technique=technique, seed=seed)

        self.app.log.info("Writing splits to files...")
        data_files = split_dataset.write(out_dir=Path(self.path), delimiter=(',' if offset else ' '),
                                         headers="label,hash,fpath,sline,scol,eline,ecol" if offset else None)

        jsonlines_files = []
        raw_files_path_str = self.get('raw_files_path')
        jsonlines_path = Path(self.path, 'jsonlines')
        self.set('jsonlines_path', jsonlines_path)

        if raw_files_path_str:
            raw_files_path = Path(raw_files_path_str)

            if not raw_files_path.exists():
                self.app.log.warning(f"{raw_files_path} not found")
                return None

            for file in data_files:
                self.app.log.info(f"======== Building JSONLines for {file} while removing comments... ========")
                output_filename = jsonlines_path / f"{file.stem}.jsonl"
                jsonlines_files.append(output_filename)
                funcs = to_dict(file, replace=(str(self.app.bind), str(raw_files_path)))
                to_json(raw_files_path, funcs, output_filename)

        if offset:
            for file in jsonlines_files:
                to_offset(file, file.parent / f"{file.stem}_offset_dict.json")

        return dataset

    def get_datasets(self, dataset_file: Path, delimiter: str) -> XYDataset:
        self.app.log.info("Reading dataset...")

        with dataset_file.open(mode="r") as dataset_file:
            reader = csv.reader(dataset_file, delimiter=delimiter)
            next(reader, None)  # skip the headers
            xy_dataset = XYDataset()

            for row in tqdm(reader):
                # x = x, y = context-paths
                xy_dataset.add(x_row=[row[1], delimiter.join(row[2:])], y_row=row[0])

        return xy_dataset


def load(app):
    app.handler.register(SamplingHandler)
