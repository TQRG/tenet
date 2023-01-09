import random
import pickle
from typing import Tuple

'''
This script preprocesses the data from MethodPaths. It truncates methods with too many contexts,
and pads methods with less paths with spaces.
'''


def save_dictionaries(dict_file_path, word_to_count, path_to_count, target_to_count, num_training_examples):
    with open(dict_file_path, 'wb') as file:
        pickle.dump(word_to_count, file)
        pickle.dump(path_to_count, file)
        pickle.dump(target_to_count, file)
        pickle.dump(num_training_examples, file)
        print('Dictionaries saved to: {}'.format(dict_file_path))


def process_file(file_path: str, output_path: str, word_to_count, path_to_count, max_contexts: int) -> Tuple[int, str]:
    sum_total = 0
    sum_sampled = 0
    total = 0
    empty = 0
    max_unfiltered = 0

    with open(output_path, 'w') as outfile:
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.rstrip('\n').split(' ')
                target_name = parts[0]
                contexts = parts[1:]

                if len(contexts) > max_unfiltered:
                    max_unfiltered = len(contexts)
                sum_total += len(contexts)

                if len(contexts) > max_contexts:
                    context_parts = [c.split(',') for c in contexts]
                    full_found_contexts = [c for i, c in enumerate(contexts)
                                           if context_full_found(context_parts[i], word_to_count, path_to_count)]
                    partial_found_contexts = [c for i, c in enumerate(contexts)
                                              if context_partial_found(context_parts[i], word_to_count, path_to_count)
                                              and not context_full_found(context_parts[i], word_to_count,
                                                                         path_to_count)]
                    if len(full_found_contexts) > max_contexts:
                        contexts = random.sample(full_found_contexts, max_contexts)
                    elif len(full_found_contexts) <= max_contexts \
                            and len(full_found_contexts) + len(partial_found_contexts) > max_contexts:
                        contexts = full_found_contexts + \
                                   random.sample(partial_found_contexts, max_contexts - len(full_found_contexts))
                    else:
                        contexts = full_found_contexts + partial_found_contexts

                if len(contexts) == 0:
                    empty += 1
                    continue

                sum_sampled += len(contexts)

                csv_padding = " " * (max_contexts - len(contexts))
                outfile.write(target_name + ' ' + " ".join(contexts) + csv_padding + '\n')
                total += 1

    print(f'File: {file_path}')
    print('Average total contexts: ' + str(float(sum_total) / total))
    print('Average final (after sampling) contexts: ' + str(float(sum_sampled) / total))
    print('Total examples: ' + str(total))
    print('Empty examples: ' + str(empty))
    print('Max number of contexts per word: ' + str(max_unfiltered))
    return total, output_path


def context_full_found(context_parts, word_to_count, path_to_count):
    return context_parts[0] in word_to_count \
           and context_parts[1] in path_to_count and context_parts[2] in word_to_count


def context_partial_found(context_parts, word_to_count, path_to_count):
    if len(context_parts) != 3:
        print(context_parts)
    return context_parts[0] in word_to_count or context_parts[1] in path_to_count or context_parts[2] in word_to_count

