import csv
import re
import time
import jsonlines

import pandas as pd

from typing import Union, Tuple
from pathlib import Path
from tqdm import tqdm

from securityaware.core.sampling.offset import to_offset
from securityaware.core.sampling.balance import oversampling, disjoint_smote, disjoint_smote_hash, \
    random_undersampling, split_data, one_one_ratio, disjoint, disjoint_hash, unique, unique_hash
from securityaware.data.dataset import XYDataset, Dataset
from securityaware.data.runner import Runner, Task
from securityaware.handlers.plugin import PluginHandler
from securityaware.handlers.runner import ThreadPoolWorker

comment_regex = r"(([\"'])(?:\\[\s\S]|.)*?\2|\/(?![*\/])(?:\\.|\[(?:\\.|.)\]|.)*?\/)|\/\/.*?$|\/\*[\s\S]*?\*\/"
# TODO: fix the blacklist problem by removing comments from code in a different way
blacklist = ["vis.js", "highlight.js", "swagger-ui-bundle.js", "dhtmlx.js", "worker-lua.js", "worker-javascript.js",
             "jquery.inputmask.bundle.js", "jquery-ui.js", "highlight.pack.js", "docs/jquery.js", "c3.min.js",
             "customize-controls.min.js"]

# TODO: this should be dynamic
labels_dict = {
    "safe": 0,
    "unsafe": 1
}


class SamplingHandler(PluginHandler):
    """
        Sampling plugin
    """

    class Meta:
        label = "sampling"

    def __init__(self, **kw):
        super().__init__(**kw)
        self.token = None
        self.balance_techniques = ['over', 'disj_over', 'disj_smote', 'unique', 'undersampling', 'stratified', '1_to_1']

    def __call__(self, technique: str, seed: int, x: Dataset, y: Dataset, offset: bool = False) \
            -> Tuple[XYDataset, XYDataset, XYDataset]:
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
        # TODO: change these sets into something simpler
        train_data_path = Path(self.path, 'train.txt')
        val_data_path = Path(self.path, 'val.txt')
        test_data_path = Path(self.path, 'test.txt')

        self.set('train_data', train_data_path)
        self.set('val_data', val_data_path)
        self.set('test_data', test_data_path)

        jsonlines_path = Path(self.path, 'jsonlines')
        self.set('jsonlines_path', jsonlines_path)

        if offset:
            self.set('train_lines', Path(jsonlines_path, 'train.jsonl'))
            self.set('val_lines', Path(jsonlines_path, 'val.jsonl'))
            self.set('test_lines', Path(jsonlines_path, 'test.jsonl'))

            self.set('train_offset_path', Path(jsonlines_path, 'train_offset_dict.json'))
            self.set('val_offset_path', Path(jsonlines_path, 'val_offset_dict.json'))
            self.set('test_offset_path', Path(jsonlines_path, 'test_offset_dict.json'))

        if technique and technique not in self.balance_techniques:
            self.app.log.warning(f"Technique must be one of the following: {self.balance_techniques}")
            return None

        self.app.log.info((f"Sampling with {technique}.\n" if technique else "") + f"Saving results to {self.path}")

        dataset_file = Path(self.get('context_paths_file'))
        xy_dataset = self.get_datasets(dataset_file=dataset_file, delimiter=',' if offset else ' ', offset=offset)

        self.app.log.info(f"Dataset has {len(xy_dataset.x)} samples.")
        xy_train, xy_val, xy_test = self.__call__(x=xy_dataset.x, y=xy_dataset.y, offset=offset, technique=technique,
                                                  seed=seed)

        self.app.log.info("Writing splits to files...")
        self.app.log.info(f"Train: {len(xy_train)}, Val.: {len(xy_val)}, Test: {len(xy_test)}")

        headers = "label,hash,fpath,sline,scol,eline,ecol" if offset else None
        delimiter = (',' if offset else ' ')

        xy_train.write(train_data_path, delimiter, headers=headers)
        xy_val.write(val_data_path, delimiter, headers=headers)
        xy_test.write(test_data_path, delimiter, headers=headers)

        raw_files_path_str = self.get('raw_files_path')
        jsonlines_files = []

        if raw_files_path_str:
            jsonlines_path.mkdir(parents=True, exist_ok=True)
            raw_files_path = Path(raw_files_path_str)

            if not raw_files_path.exists():
                self.app.log.warning(f"{raw_files_path} not found")
                return None

            for file in [train_data_path, val_data_path, test_data_path]:
                self.app.log.info(f"======== Building JSONLines for {file} while removing comments... ========")
                output_filename = jsonlines_path / f"{file.stem}.jsonl"
                jsonlines_files.append(output_filename)

                # funcs = to_dict(file, replace=(str(self.app.bind), str(self.app.workdir)))
                funcs_df = pd.read_csv(file)
                funcs_df['fpath'] = funcs_df.fpath.apply(lambda x: x.replace(str(self.app.bind), str(self.app.workdir)))
                self.jsonify(raw_files_path, funcs_df, output_filename)

        if offset:
            for file in jsonlines_files:
                file_path = file.parent / f"{file.stem}_offset_dict.json"
                to_offset(file, file_path)

        return dataset

    def clean_functions(self, target_file: Path, filename: str, rows: pd.Series):
        labeled_code = []

        with target_file.open(encoding="latin-1") as code_file:
            self.app.log.info(f"Processing {filename}...")
            lines = code_file.readlines()
            for idx, row in rows.iterrows():
                self.app.log.info(f"\tProcessing row {idx}")
                code = lines[(row.sline - 1):row.eline]

                if (len(code) > 1):
                    code[0] = code[0][row.scol:]
                    code[-1] = code[-1][:row.ecol]

                else:
                    code[0] = code[0][row.scol:row.ecol]
                code = ''.join(code)

                if (not any(s in str(filename) for s in blacklist)) and (
                        not all(s in str(filename) for s in ["config", "mhjx_MhJx"])):
                    # TODO: FIX THIS SHORTCUT
                    if len(code) < 500:
                        code = re.sub(comment_regex, "\\1", code, 0, re.MULTILINE)

                labeled_code.append({"code": code, "label": labels_dict[row.label]})

        return labeled_code

    def clean_functions_task(self, task: Task):
        """
            Maps arguments to the call
        """
        return self.clean_functions(target_file=task['target_file'], filename=task['filename'], rows=task['rows'])

    def jsonify(self, files_path: Path, funcs_df: pd.DataFrame, output_filename: Path):
        runner_data = Runner()
        threads = self.app.get_config('local_threads')
        tasks = []

        for filename, rows in tqdm(funcs_df.groupby('fpath')):
            target_file = files_path / filename

            if not target_file.exists():
                print(f"{target_file} not found")
                continue

            task = Task()
            task['id'] = (rows.index[0], rows.index[-1])
            task['target_file'] = target_file
            task['filename'] = filename
            task['rows'] = rows

            tasks.append(task)

        worker = ThreadPoolWorker(runner_data, tasks=tasks, threads=threads, logger=self.app.log,
                                  func=self.clean_functions_task)
        worker.start()

        while worker.is_alive():
            time.sleep(1)

        labeled_functions = []

        for res in runner_data.finished:
            if 'result' in res and res['result']:
                labeled_functions.extend(res['result'])

        with jsonlines.open(output_filename, mode='a') as output_file:
            for el in labeled_functions:
                output_file.write(el)

    def get_datasets(self, dataset_file: Path, delimiter: str, offset: bool) -> XYDataset:
        self.app.log.info("Reading dataset...")

        with dataset_file.open(mode="r") as dataset_file:
            reader = csv.reader(dataset_file, delimiter=delimiter)
            next(reader, None)  # skip the headers
            xy_dataset = XYDataset()

            for row in tqdm(reader):
                # x = x, y = context-paths
                if offset:
                    xy_dataset.add(x_row=[row[1], delimiter.join(row[2:])], y_row=row[0])
                else:
                    xy_dataset.add(x_row=[row[1], delimiter.join(row[3:])], y_row=row[0])

        return xy_dataset


def load(app):
    app.handler.register(SamplingHandler)
