import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


class HistogramVocab:
    def __init__(self):
        self.words: list = []

    def add(self, words: Tuple[str, list]):
        if isinstance(words, str):
            self.words.append(words)
        else:
            self.words.extend(words)

    def __call__(self, min_count: int = 0, max_size: int = None) -> dict:
        # Take min_count to be one plus the count of the max_size'th word
        counter = Counter(self.words)

        if max_size is not None and len(counter) > max_size:
            min_word, min_count = counter.most_common()[max_size - 1]

        if min_count > 0:
            return {word: count for word, count in counter.items() if count >= min_count}

        return dict(counter)


class Vector:
    def __init__(self, value):
        self.value = value
        self._parts = None

    @property
    def parts(self):
        if not self._parts:
            self._parts = self.value.split(',')
        return self._parts

    def __str__(self):
        return self.value


class ContextPath:
    def __init__(self, label: str, vectors: list):
        self.label = label
        self.vectors = [Vector(v) for v in vectors]
        self.padding = ""

    def pad_to(self, max_vectors: int):
        self.padding = " " * (max_vectors - len(self))

    def __len__(self):
        return len(self.vectors)

    def __str__(self):
        string_vectors = [str(v) for v in self.vectors]
        return self.label + ' ' + " ".join(string_vectors) + self.padding


def get_histograms(context_paths: list) -> Tuple[HistogramVocab, HistogramVocab]:
    """
        Computes the histogram of all source/target words and the histogram of all the path hashes.

        :return: tuple with words count dictionary and hashes count dictionary
    """
    word_histogram_vocab = HistogramVocab()
    path_histogram_vocab = HistogramVocab()

    for cp in context_paths:
        cp_split = cp.split()
        first_vector_split = cp_split[0].split(',')

        word_histogram_vocab.add(first_vector_split)
        path_histogram_vocab.add(first_vector_split[1])

    return word_histogram_vocab, path_histogram_vocab


@dataclass
class ContextPathsInfo:
    max_unfiltered = 0
    sum_total = 0
    sum_sampled = 0
    total = 0
    empty = 0

    @property
    def average(self):
        if self.total > 0:
            return float(self.sum_total) / self.total
        return 0

    @property
    def average_sampled(self):
        if self.total > 0:
            return float(self.sum_sampled) / self.total
        return 0

    def __str__(self):
        output = f'\tAverage total contexts: {self.average}\n'
        output += f'\tAverage final (after sampling) contexts: {self.average_sampled}\n'
        output += f'\tTotal examples: {self.total}\n'
        output += f'\tEmpty examples: {self.empty}\n'
        output += f'\tMax number of contexts per word: {self.max_unfiltered}\n'

        return output


class ContextPathsTruncator:
    """
        Truncates methods with too many contexts, and pads methods with fewer paths with spaces.
    """

    def __init__(self, word_to_count: dict, path_to_count: dict, max_vectors: int):
        self.max_vectors = max_vectors
        self.path_to_count = path_to_count
        self.word_to_count = word_to_count

    def has_word(self, word: str):
        return word in self.word_to_count

    def has_path(self, path: str):
        return path in self.path_to_count

    def is_full_found(self, vector_parts: list) -> bool:
        return self.has_word(vector_parts[0]) and self.has_path(vector_parts[1]) and self.has_word(vector_parts[2])

    def if_partial_found(self, vector_parts: list) -> bool:
        if len(vector_parts) != 3:
            print(vector_parts)

        return self.has_word(vector_parts[0]) or self.has_path(vector_parts[1]) or self.has_word(vector_parts[2])

    def __call__(self, context_paths: list, labels: list, path: str = None) -> Tuple[list, ContextPathsInfo]:
        new_context_paths = []
        cp_info = ContextPathsInfo()

        for line, label in zip(context_paths, labels):
            context_path = ContextPath(label=label, vectors=line.rstrip('\n').split(' '))

            if len(context_path) > cp_info.max_unfiltered:
                cp_info.max_unfiltered = len(context_path)

            cp_info.sum_total += len(context_path)

            if len(context_path) > self.max_vectors:
                context_path = self.cut(context_path)

            if len(context_path) == 0:
                cp_info.empty += 1
                continue

            cp_info.sum_sampled += len(context_path)
            context_path.pad_to(self.max_vectors)
            new_context_paths.append(str(context_path))
            cp_info.total += 1

        if path is not None:
            with Path(path).open(mode='w') as f:
                for cp in new_context_paths:
                    f.write(f"{cp}\n")

        return new_context_paths, cp_info

    def cut(self, context_path: ContextPath) -> ContextPath:
        full_found_vectors, partial_found_vectors = self.check_vectors(context_path)

        if len(full_found_vectors) > self.max_vectors:
            new_vectors = random.sample(full_found_vectors, self.max_vectors)
        elif len(full_found_vectors) <= self.max_vectors and \
                len(full_found_vectors) + len(partial_found_vectors) > self.max_vectors:
            new_vectors = full_found_vectors + random.sample(partial_found_vectors,
                                                             self.max_vectors - len(full_found_vectors))
        else:
            new_vectors = full_found_vectors + partial_found_vectors

        return ContextPath(label=context_path.label, vectors=new_vectors)

    def check_vectors(self, context_path) -> Tuple[list, list]:
        full_found_contexts = []
        partial_found_contexts = []

        for v in context_path.vectors:
            if self.is_full_found(v.parts):
                full_found_contexts.append(str(v))
            if self.if_partial_found(v.parts) and not self.is_full_found(v.parts):
                partial_found_contexts.append(str(v))

        return full_found_contexts, partial_found_contexts

