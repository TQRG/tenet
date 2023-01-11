import pandas as pd

from typing import Union
from pathlib import Path

from tenet.core.astminer.preprocess import save_dictionaries
from tenet.core.preprocess.context_paths import get_histograms, ContextPathsTruncator
from tenet.handlers.plugin import PluginHandler


class PreprocessHandler(PluginHandler):
    """
        Preprocess plugin
    """

    class Meta:
        label = "preprocess"

    def __init__(self, **kw):
        super().__init__(**kw)

    def run(self, dataset: pd.DataFrame, max_contexts: int = 200, word_vocab_size: int = 1301136,
            path_vocab_size: int = 911417, target_vocab_size: int = 261245, **kwargs) -> Union[pd.DataFrame, None]:
        """
            Parameters:
                max_contexts (int): number of max contexts to keep
                word_vocab_size (int): Max number of origin word in to keep in the vocabulary
                path_vocab_size (int): Max number of paths to keep in the vocabulary
                target_vocab_size (int): Max number of target words to keep in the vocabulary
        """

        dataset_name = self.output.stem
        train_output_path = f'{self.path}/{dataset_name}.train.c2v'
        self.set('train_output_path', train_output_path)
        val_output_path = f'{self.path}/{dataset_name}.val.c2v'
        self.set('val_output_path', val_output_path)
        test_output_path = f'{self.path}/{dataset_name}.test.c2v'
        self.set('test_output_path', test_output_path)
        dict_file_path = Path(self.path, f'{dataset_name}.dict.c2v')
        self.set('dict_file_path', dict_file_path)

        train_data_path = self.get('train_data_path')
        val_data_path = self.get('val_data_path')
        test_data_path = self.get('test_data_path')

        if not train_data_path:
            self.app.log.warning(f"Train data file not instantiated.")
            return None

        if not Path(train_data_path).exists():
            self.app.log.warning(f"Train data file not found.")
            return None

        if not val_data_path:
            self.app.log.warning(f"Validation data file not instantiated.")
            return None

        if not Path(val_data_path).exists():
            self.app.log.warning(f"Validation data file not found.")
            return None

        if not test_data_path:
            self.app.log.warning(f"Test data file not instantiated.")
            return None

        if not Path(test_data_path).exists():
            self.app.log.warning(f"Test data file not found.")
            return None

        train_data = pd.read_csv(str(train_data_path))
        test_data = pd.read_csv(str(test_data_path))
        val_data = pd.read_csv(str(val_data_path))

        self.app.log.info("Creating histogram from training data")
        word_vocab, path_vocab = get_histograms(train_data['context_paths'].to_list())
        word_to_count: dict = word_vocab(max_size=word_vocab_size)
        path_to_count: dict = path_vocab(max_size=path_vocab_size)
        target_to_count: dict = train_data['label'].value_counts().to_dict()
        cp_truncator = ContextPathsTruncator(max_vectors=max_contexts, word_to_count=word_to_count,
                                             path_to_count=path_to_count)
        self.app.log.info("Truncating train data")
        train_context_paths, train_data_info = cp_truncator(context_paths=train_data['context_paths'].to_list(),
                                                            labels=train_data['label'].to_list(), path=train_output_path)
        self.app.log.info(f'Train data info: {train_data_info}')

        with Path(train_output_path).open(mode='w') as f:
            for cp in train_context_paths:
                f.write(f"{cp}\n")

        save_dictionaries(dict_file_path=dict_file_path, word_to_count=word_to_count, path_to_count=path_to_count,
                          target_to_count=target_to_count, num_training_examples=train_data_info.total)

        self.app.log.info("Truncating test data")
        test_context_paths, test_data_info = cp_truncator(context_paths=test_data['context_paths'].to_list(),
                                                          labels=test_data['label'].to_list(), path=test_output_path)
        self.app.log.info(f'Test data info: {test_data_info}')

        with Path(test_output_path).open(mode='w') as f:
            for cp in test_context_paths:
                f.write(f"{cp}\n")

        self.app.log.info("Truncating validation data")
        val_context_paths, val_data_info = cp_truncator(context_paths=val_data['context_paths'].to_list(),
                                                        labels=val_data['label'].to_list(), path=val_output_path)
        self.app.log.info(f'Val data info: {val_data_info}')

        return dataset


def load(app):
    app.handler.register(PreprocessHandler)