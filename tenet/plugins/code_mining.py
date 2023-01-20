import numpy as np
import pandas as pd

from pathlib import Path
from typing import Union
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm import tqdm

from tenet.handlers.plugin import PluginHandler


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

    def set_sources(self):
        if 'vectorizer_type' in self.node.kwargs:
            self.vectorizer_type = self.node.kwargs['vectorizer_type']

        if isinstance(self.sinks['train_data_path'], list):
            train_sparse_matrix_path = []
            test_sparse_matrix_path = []
            train_labels_path = []
            test_labels_path = []

            for i, _ in enumerate(self.sinks['train_data_path']):
                train_sparse_matrix_path.append(
                    self.path / f"{self.vectorizer_type}_{self.output.stem}_train_features_sparse_{i}.npz")
                test_sparse_matrix_path.append(
                    self.path / f"{self.vectorizer_type}_{self.output.stem}_test_features_sparse_{i}.npz")
                train_labels_path.append(self.path / f"{self.output.stem}_train_labels_{i}.csv")
                test_labels_path.append(self.path / f"{self.output.stem}_test_labels_{i}.csv")
        else:
            train_sparse_matrix_path = [
                self.path / f"{self.vectorizer_type}_{self.output.stem}_train_features_sparse.npz"]
            test_sparse_matrix_path = [
                self.path / f"{self.vectorizer_type}_{self.output.stem}_test_features_sparse.npz"]
            train_labels_path = [self.path / f"{self.output.stem}_train_labels.csv"]
            test_labels_path = [self.path / f"{self.output.stem}_test_labels.csv"]

        self.set('train_sparse_matrix_path', train_sparse_matrix_path)
        self.set('test_sparse_matrix_path', test_sparse_matrix_path)
        self.set('train_labels_path', train_labels_path)
        self.set('test_labels_path', test_labels_path)

    def get_sinks(self):
        self.get('train_data_path')
        self.get('test_data_path')

    def run(self, dataset: pd.DataFrame, max_features: int = 1000, vectorizer_type: str = 'tfidf',
            remove_comments: bool = False, **kwargs) -> Union[pd.DataFrame, None]:
        """
            runs the plugin
        """
        self.max_features = max_features
        self.vectorizer_type = vectorizer_type

        if not isinstance(self.sinks['train_data_path'], list):
            self.sinks['train_data_path'] = [self.sinks['train_data_path']]
            self.sinks['test_data_path'] = [self.sinks['test_data_path']]

        paths = zip(self.sinks['train_data_path'], self.sources['train_labels_path'],
                    self.sources['train_sparse_matrix_path'], self.sinks['test_data_path'],
                    self.sources['test_labels_path'], self.sources['test_sparse_matrix_path'])

        for i, (train_data_path, train_labels_path, train_sparse_matrix_path, test_data_path, test_labels_path, test_sparse_matrix_path) in enumerate(paths):
            train_vectorizer_model_path = Path(self.path, f"bow_{vectorizer_type}_{self.output.stem}_train_{i}.model")
            test_vectorizer_model_path = Path(self.path, f"bow_{vectorizer_type}_{self.output.stem}_test_{i}.model")
            train_vocabulary_path = Path(self.path, f"bow_{vectorizer_type}_{self.output.stem}_train_vocab_{i}.txt")
            test_vocabulary_path = Path(self.path, f"bow_{vectorizer_type}_{self.output.stem}_test_vocab_{i}.txt")

            # get test/train data
            train_data = pd.read_csv(str(train_data_path))
            test_data = pd.read_csv(str(test_data_path))

            if remove_comments:
                self.app.log.info(f"Removing comments from code...")
                if 'fpath' in train_data:
                    train_data.rename(columns={'fpath': 'file_path'}, inplace=True)
                train_data = self.clean_data(train_data)
                del self.multi_task_handler
                if 'fpath' in test_data:
                    test_data.rename(columns={'fpath': 'file_path'}, inplace=True)
                test_data = self.clean_data(test_data)

            x_train = train_data.input.to_numpy()
            x_test = test_data.input.to_numpy()

            # save labels
            train_data.label.to_csv(str(train_labels_path))
            test_data.label.to_csv(str(test_labels_path))

            # if model:
            #    features = apply_model(vectorizer, sentences, model_path=model)
            # else:
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

            self.app.log.info(f"Train features shape: {train_features.shape}")
            self.app.log.info(f"Test features shape: {test_features.shape}")

            save_npz(str(train_sparse_matrix_path), train_features)
            save_npz(str(test_sparse_matrix_path), test_features)

        return dataset

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        initial_size = len(data)
        for row in tqdm(data.to_dict(orient='records')):
            self.multi_task_handler.add(row=row)

        self.multi_task_handler(func=self.remove_comments)
        rows = self.multi_task_handler.results()
        new_data = pd.DataFrame(rows)
        lost_samples = initial_size - len(new_data)

        if lost_samples > 0:
            self.app.log.warning(f"Dropped {lost_samples} samples in the training set with empty input.")

        return new_data

    def remove_comments(self, row: dict):
        clean_code = self.code_parser_handler.filter_comments(code=np.str_(row['input']),
                                                              extension=Path(row['file_path']).suffix.split('.')[-1])
        if len(clean_code.strip()) == 0:
            self.app.log.warning(f"Empty string for file {row['file_path']}.")
            return None

        row['input'] = clean_code

        return row

    def get_vectorizer(self, vectorize=False):
        # Build a code vectorizer
        if self.vectorizer_type == 'cv':
            vectorizer = CountVectorizer(stop_words=None, ngram_range=(1, 1), max_features=self.max_features,
                                         lowercase=False, tokenizer=self.code_parser_handler.tokenize, vocabulary=None)
        else:
            vectorizer = TfidfVectorizer(stop_words=None, ngram_range=(1, 1), use_idf=False,
                                         max_features=self.max_features, norm=None, smooth_idf=False,
                                         lowercase=False, tokenizer=self.code_parser_handler.tokenize, vocabulary=None)

        if vectorize:
            return vectorizer

        return vectorizer.build_analyzer()

    # Train and save a BoW feature model
    def train_bag_of_words(self, sentences, model_path: Path):
        vectorizer = self.get_vectorizer(vectorize=True)
        vectorizer.fit(sentences)
        # pickle.dump(vectorizer, model_path.open(mode="wb"))
        return vectorizer


def load(app):
    app.handler.register(CodeMining)
