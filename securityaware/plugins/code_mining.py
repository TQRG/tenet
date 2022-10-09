import time

import numpy as np
import pandas as pd
import codeprep.api.text as cp

from scipy.sparse import save_npz
from pathlib import Path
from typing import Union

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm import tqdm

from securityaware.core.utils import clean_code
from securityaware.handlers.plugin import PluginHandler
from securityaware.data.runner import Task, Runner
from securityaware.handlers.runner import ThreadPoolWorker
from sklearn.model_selection import train_test_split
# TODO: this should be dynamic
labels_dict = {
    "safe": 0,
    "unsafe": 1
}


class CodeMining(PluginHandler):
    """
        Code feature extraction plugin
    """

    class Meta:
        label = "code_mining"

    def __init__(self, **kw):
        super().__init__(**kw)
        self.max_features: int = 1000
        self.vectorizer_type: str = 'tfidf'
        self.extension = 'js'

    def run(self, dataset: pd.DataFrame, max_features: int = 1000, vectorizer_type: str = 'tfidf', extension: str = 'js',
            **kwargs) -> Union[pd.DataFrame, None]:
        """
            runs the plugin
        """
        self.max_features = max_features
        self.vectorizer_type = vectorizer_type
        self.extension = extension
        dataset_name = self.output.stem
        train_vectorizer_model_path = Path(self.path, f"bow_{vectorizer_type}_{dataset_name}_train.model")
        test_vectorizer_model_path = Path(self.path, f"bow_{vectorizer_type}_{dataset_name}_test.model")
        train_vocabulary_path = Path(self.path, f"bow_{vectorizer_type}_{dataset_name}_train_vocab.txt")
        test_vocabulary_path = Path(self.path, f"bow_{vectorizer_type}_{dataset_name}_test_vocab.txt")
        train_sparse_matrix_path = Path(self.path, f"{vectorizer_type}_{dataset_name}_train_features_sparse.npz")
        test_sparse_matrix_path = Path(self.path, f"{vectorizer_type}_{dataset_name}_test_features_sparse.npz")
        train_labels_path = Path(self.path, f"{dataset_name}_train_labels.csv")
        test_labels_path = Path(self.path, f"{dataset_name}_test_labels.csv")
        train_data_path = Path(self.path, f"{dataset_name}_train.csv")
        test_data_path = Path(self.path, f"{dataset_name}_test.csv")

        # TODO: fix this
        self.set('train_tokenizer_model_path', train_vectorizer_model_path)
        self.set('test_tokenizer_model_path', test_vectorizer_model_path)
        self.set('train_sparse_matrix_path', train_sparse_matrix_path)
        self.set('test_sparse_matrix_path', test_sparse_matrix_path)
        self.set('train_labels_path', train_labels_path)
        self.set('test_labels_path', test_labels_path)
        self.set('train_data_path', train_data_path)
        self.set('test_data_path', test_data_path)

        raw_files_path_str = self.get('raw_files_path')

        if not raw_files_path_str:
            self.app.log.error(f"Raw files path not found")
            return None

        raw_files_path = Path(raw_files_path_str)

        functions = self.get_functions(raw_files_path, dataset)
        dataset.drop(columns='label', inplace=True)
        dataset_functions = dataset.merge(functions, how='inner', left_on="func_id", right_on='func_id')
        return dataset_functions

        train_data, test_data = train_test_split(dataset_functions, train_size=0.8, random_state=42)

        # save test train split data
        train_data.to_csv(str(train_data_path))
        test_data.to_csv(str(test_data_path))

        # select only code as train/test data
        x_train = train_data.code.apply(lambda x: np.str_(x))
        x_test = test_data.code.apply(lambda x: np.str_(x))

        # save labels
        train_data.label.to_csv(str(train_labels_path))
        test_data.label.to_csv(str(test_labels_path))

        #if model:
        #    features = apply_model(vectorizer, sentences, model_path=model)
        #else:
        #    features = train_model(vectorizer, sentences, n_feats, project)

        train_feature_model = self.train_bag_of_words(x_train, train_vectorizer_model_path)
        test_feature_model = self.train_bag_of_words(x_test, test_vectorizer_model_path)

        # Print the vocab of the 'BoW' model.
        self.app.log.info(f"Writing train vocabulary of the model to {train_vocabulary_path}")

        with train_vocabulary_path.open(mode='w') as vp:
            vp.write('\n'.join(train_feature_model.vocabulary_))

        self.app.log.info(f"Writing test vocabulary of the model to {test_vocabulary_path}")
        with test_vocabulary_path.open(mode='w') as vp:
            vp.write('\n'.join(test_feature_model.vocabulary_))

        self.app.log.info("Inferring...")
        train_features = train_feature_model.transform(x_train)
        test_features = test_feature_model.transform(x_test)

        self.app.log.info(train_features.shape)
        self.app.log.info(test_features.shape)
        self.app.log.info(train_features)
        self.app.log.info(test_features)

        save_npz(str(train_sparse_matrix_path), train_features)
        save_npz(str(test_sparse_matrix_path), test_features)

        return dataset_functions

    def get_vectorizer(self, vectorize=False):

        def tokenizer(code):
            # Tokenize, remove comments and blank lines
            tokens = cp.nosplit(code, no_spaces=True, no_com=True, extension=self.extension)
            # remove placeholders
            return list(filter(lambda x: x not in ['<comment>'], tokens))

        # Build a code vectorizer
        if self.vectorizer_type == 'cv':
            vectorizer = CountVectorizer(stop_words=None, ngram_range=(1, 1), max_features=self.max_features,
                                         lowercase=False, tokenizer=tokenizer, vocabulary=None)
        else:
            vectorizer = TfidfVectorizer(stop_words=None, ngram_range=(1, 1), use_idf=False,
                                         max_features=self.max_features, norm=None, smooth_idf=False,
                                         lowercase=False, tokenizer=tokenizer, vocabulary=None)

        if vectorize:
            return vectorizer

        return vectorizer.build_analyzer()

    # Train and save a BoW feature model
    def train_bag_of_words(self, sentences, model_path: Path):
        vectorizer = self.get_vectorizer(vectorize=True)
        vectorizer.fit(sentences)
        # pickle.dump(vectorizer, model_path.open(mode="wb"))
        return vectorizer

    def extract_functions(self, target_file: Path, rows: pd.Series):
        functions = []

        with target_file.open(encoding="latin-1") as code_file:
            self.app.log.info(f"Processing {target_file}...")
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
                code = clean_code(code)

                functions.append({"code": code, "label": labels_dict[row.label], 'func_id': row['func_id']})

        return functions

    def extract_functions_task(self, task: Task):
        """
            Maps arguments to the call
        """
        return self.extract_functions(target_file=task['target_file'], rows=task['rows'])

    def get_functions(self, files_path: Path, funcs_df: pd.DataFrame):
        runner_data = Runner()
        threads = self.app.get_config('local_threads')
        tasks = []

        for (project, filename), rows in tqdm(funcs_df.groupby(['project', 'fpath'])):
            target_file = files_path / project/ filename

            if not target_file.exists():
                self.app.log.warning(f"{target_file} not found")
                continue

            task = Task()
            task['id'] = (rows.index[0], rows.index[-1])
            task['target_file'] = target_file
            task['rows'] = rows

            tasks.append(task)

        worker = ThreadPoolWorker(runner_data, tasks=tasks, threads=threads, logger=self.app.log,
                                  func=self.extract_functions_task)
        worker.start()

        while worker.is_alive():
            time.sleep(1)

        functions = []

        for res in runner_data.finished:
            if 'result' in res and res['result']:
                functions.extend(res['result'])

        return pd.DataFrame(functions)


def load(app):
    app.handler.register(CodeMining)
