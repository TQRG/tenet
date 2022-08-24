import pandas as pd

from typing import Union
from pathlib import Path

from securityaware.core.astminer.common import common
from securityaware.core.astminer.preprocess import process_file, save_dictionaries
from securityaware.handlers.plugin import PluginHandler


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
        word_histogram_path = self.get('word_histogram_path')
        path_histogram_file = self.get('path_histogram_file')
        target_histogram_path = self.get('target_histogram_path')

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

        if not word_histogram_path:
            self.app.log.warning(f"Word histogram data file not instantiated.")
            return None

        if not Path(word_histogram_path).exists():
            self.app.log.warning(f"Word histogram data file not found.")
            return None

        if not path_histogram_file:
            self.app.log.warning(f"Path histogram data file not instantiated.")
            return None

        if not Path(path_histogram_file).exists():
            self.app.log.warning(f"Path histogram data file not found.")
            return None

        if not target_histogram_path:
            self.app.log.warning(f"Target histogram data file not instantiated.")
            return None

        if not Path(target_histogram_path).exists():
            self.app.log.warning(f"Target histogram data file not found.")
            return None

        word_to_count, path_to_count, target_to_count = common.get_counts(word_histogram=word_histogram_path,
                                                                          word_vocab_size=word_vocab_size,
                                                                          path_histogram=path_histogram_file,
                                                                          path_vocab_size=path_vocab_size,
                                                                          target_histogram=target_histogram_path,
                                                                          target_vocab_size=target_vocab_size)

        num_training_examples = process_file(file_path=train_data_path, output_path=train_output_path,
                                             word_to_count=word_to_count, path_to_count=path_to_count,
                                             max_contexts=max_contexts)

        process_file(file_path=test_data_path, output_path=test_output_path, word_to_count=word_to_count,
                     path_to_count=path_to_count, max_contexts=max_contexts)

        process_file(file_path=val_data_path, output_path=val_output_path, word_to_count=word_to_count,
                     path_to_count=path_to_count, max_contexts=max_contexts)

        save_dictionaries(dict_file_path=dict_file_path, word_to_count=word_to_count, path_to_count=path_to_count,
                          target_to_count=target_to_count, num_training_examples=num_training_examples)

        return dataset


def load(app):
    app.handler.register(PreprocessHandler)
