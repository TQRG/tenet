import numpy as np
import pandas as pd

from pathlib import Path
from typing import Union
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from securityaware.handlers.plugin import PluginHandler


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
        self.extension: str = 'js'

    def run(self, dataset: pd.DataFrame, max_features: int = 1000, vectorizer_type: str = 'tfidf',
            extension: str = 'js', **kwargs) -> Union[pd.DataFrame, None]:
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

        # TODO: fix this
        #self.set('train_tokenizer_model_path', train_vectorizer_model_path)
        #self.set('test_tokenizer_model_path', test_vectorizer_model_path)
        self.set('train_sparse_matrix_path', train_sparse_matrix_path)
        self.set('test_sparse_matrix_path', test_sparse_matrix_path)
        self.set('train_labels_path', train_labels_path)
        self.set('test_labels_path', test_labels_path)

        train_data_path = self.get('train_data_path')
        test_data_path = self.get('test_data_path')

        if not train_data_path:
            self.app.log.error(f"Train data path not instantiated")
            return None

        if not test_data_path:
            self.app.log.error(f"Test data path not instantiated")
            return None

        # save test train split data
        train_data = pd.read_csv(str(train_data_path))
        test_data = pd.read_csv(str(test_data_path))

        # select only code as train/test data
        x_train = train_data.input.apply(lambda x: np.str_(x))
        x_test = test_data.input.apply(lambda x: np.str_(x))

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

        return dataset

    def get_vectorizer(self, vectorize=False):

        def tokenizer(code):
            # Tokenize, remove comments and blank lines
            import codeprep.api.text as cp_text
            tokens = cp_text.nosplit(code, no_spaces=True, no_com=True, extension=self.extension)
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


def load(app):
    app.handler.register(CodeMining)
