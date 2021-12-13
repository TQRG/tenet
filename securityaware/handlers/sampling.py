import csv
from pathlib import Path

from cement import Handler
from tqdm import tqdm

from securityaware.core.interfaces import HandlersInterface
from securityaware.core.sampling.balance import oversampling, disjoint_smote, disjoint_smote_hash, \
    random_undersampling, split_data, one_one_ratio, disjoint, disjoint_hash, unique, unique_hash
from securityaware.data.dataset import XYDataset, Dataset, SplitDataset


class SamplingHandler(HandlersInterface, Handler):
    class Meta:
        label = 'sampling'

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
