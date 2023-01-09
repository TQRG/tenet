import itertools
import os
from tqdm import tqdm
import csv
import argparse
import numpy as np
from collections import Counter
import math
import time


def load_data(file_path):
    """
    :param file_path: dataset path
    :return: labels, count of frequency, L2 norm, raw X data
    """
    X_dataset = []
    Y_dataset = []
    count = []
    norm = []

    """
    loading data
    data format: [n * 2]
        safe a b c ... f
        unsafe d e f ... g
        ... ...
        safe x y z ... h
    """
    with open(file_path) as dataset_file:
        reader = csv.reader(dataset_file, delimiter=' ')
        for row in tqdm(reader):
            if row[1:] not in X_dataset:
                # labels
                Y_dataset.append(row[0])
                # context-paths
                X_dataset.append(row[1:])
                # count frequency of each context path in a bag of context paths
                count0 = Counter(row[1:])
                count.append(count0)
                # L2 norm, for calculating cosine similarity
                norm.append(math.sqrt(sum(count0[e] ** 2 for e in list(count0))))

    """
    Change: directly use numpy since it is much more efficient when doing massive calculation 
    Saved counts and L2 norm for each bag of context paths to avoid repeated calculation, 
    however the memory consuming may be a problem if the dataset is extremely large
    """
    return np.array(Y_dataset), np.array(count, dtype=object), np.array(norm), np.array(X_dataset, dtype=object)


def get_unsafe_data(labels, counts, norms, context_paths):
    """
    :param labels: labels for each bag of context paths, [n]
    :param counts: frequency count of each context path, [n]
    :param norms: L2 norm of each bag of context paths, [n]
    :param context_paths: raw X data, [n]
    :return: unsafe data of X data, frequency count, and norm.
    """
    index_unsafe = np.where(labels == 'unsafe')
    counts_unsafe = counts[index_unsafe]
    print(index_unsafe)
    norms_unsafe = norms[index_unsafe]
    context_unsafe = context_paths[index_unsafe]
    return context_unsafe, counts_unsafe, norms_unsafe


def cosine_similarity(counts, norms, i, j):
    """
    :param counts: frequency count of each context path, [n]
    :param norms: L2 norm of each bag of context paths, [n]
    :param i: the ith data
    :param j: the jth data
    :return: cosine similarity between the ith and the jth data
    Since the length of each bag of context paths is different, and a string in one data may appear multiple times.
    I use cosine similarity to estimate the distance between two data.
    """
    unique_union = list(counts[i] | counts[j])
    dot_product = sum(counts[i][e] * counts[j][e] for e in unique_union)
    return dot_product / (norms[i] * norms[j])


def find_nearest(counts, norms, k_nearest):
    """
    :param counts: frequency count of each context path, [n]
    :param norms: L2 norm of each bag of context paths, [n]
    :param k_nearest: the number of neighbors we'd like to find
    :return: index list of the unsafe data, [n * k_nearest]
             index list of its neighbors, [n * k_nearest]
             similarity between the data and its neighbors, [n * k_nearest]
             num_neighbors: number of near neighbors of each unsafe data, [n]
    """
    num = norms.shape[0]
    similarity_temp = np.zeros([num, num], dtype='f')
    index0 = []
    index1 = []
    similarity = []

    for i in tqdm(range(num), leave=True):
        for j in range(i + 1, num):
            '''
            similarity_temp is a symmetric matrix, 
            the element at (i, j) represents the similarity between ith and jth dataframe
            similarities are saved so the later dataframe do not calculate it again with the former one.
            '''
            distance = cosine_similarity(counts, norms, i, j)
            similarity_temp[i, j] = distance
            similarity_temp[j, i] = distance

        # find the k nearest neighbors
        ind = np.argpartition(similarity_temp[i, :], -k_nearest)[-k_nearest:]
        index0.append([i] * k_nearest)
        index1.append(ind)
        similarity.append(similarity_temp[i, ind])

    num_neighbors = np.count_nonzero(similarity_temp > 0, axis=1)

    return index0, index1, similarity, num_neighbors


def write_similarity_to_file(index0, index1, similarity, num_neighbors, context_paths, file_name, k_nearest):
    """
    :param index0: index list of the unsafe data, [n * k_nearest]
    :param index1: index list of its neighbors, [n * k_nearest]
    :param similarity: similarity between the data and its neighbors, [n * k_nearest]
    :param num_neighbors: number of near neighbors of each unsafe data, [n]
    :param context_paths: X data, [n]
    :param file_name: output file name
    :param k_nearest: the number of neighbors
    """
    index0 = np.array(index0).flatten()
    index1 = np.array(index1).flatten()
    similarity = np.array(similarity).flatten()
    num_neighbors = np.repeat(num_neighbors, k_nearest)
    frame0 = context_paths[index0]
    frame1 = context_paths[index1]

    # write to file
    filename = 'dataset/%s_%d_nearest.txt' % (file_name, k_nearest)
    with open(filename, 'a+') as out:
        out.seek(0)
        out.truncate()
        out.write("total_number_of_neighbors top_%d_similarity unsafe_data top_%d_neighbors" % (k_nearest, k_nearest) + '\n')
        for i in range(index0.shape[0]):
            str = "%d\t%.4f\t%s\t%s" % (
                num_neighbors[i], similarity[i], ' '.join(x for x in frame0[i]), ' '.join(x for x in frame1[i]))
            out.write(str + '\n')
    print("Done!\nOutput file: dataset/%s_%d_nearest.txt" % (file_name, k_nearest))


def interpolate(data, neighbor):
    """
    :param data: the count of vulnerable context paths in a data, type: 'dictionary'
    :param neighbor: its neighbor, type: 'dictionary'
    :return: the interpolation between vulnerable data and its neighbor, type: 'list of context paths'
    """
    unique_union = list(data | neighbor)
    new_data = []
    for e in unique_union:
        frequency = round((data[e] + neighbor[e]) / 2.0 + np.random.uniform(-0.2, 0.2))
        new_data.append([e] * frequency)
    flatten = list(itertools.chain.from_iterable(new_data))
    return flatten


def over_sample(index1, counts_unsafe, context_unsafe, num_increase, similarity):
    """
    :param index1: index list of neighbors, [n * k_nearest]
    :param counts_unsafe: list of counts of unsafe data, [n]
    :param context_unsafe: all unsafe data, [n]
    :param num_increase: number of unsafe data to be increased
    :param similarity: similarity between unsafe data, [n * k_nearest]
    :return: all the new unsafe data (synthetic)
    """
    num_unsafe = counts_unsafe.shape[0]
    new_unsafe_data = []
    total = 0  # total number of new unsafe data without duplicates
    temp = 0

    progress = tqdm(total=num_increase)
    while total < num_increase:
        index = temp % num_unsafe
        if np.sum(similarity[index]) > 0:
            data = counts_unsafe[index]

            # get neighbors with similarity > 0
            neighbor_index_mask = np.argwhere(similarity[index] > 0).flatten()
            neighbor_index = np.random.choice(neighbor_index_mask)
            neighbor = counts_unsafe[index1[index][neighbor_index]]
            new_data = interpolate(data, neighbor)

            # check duplicates
            if new_data not in context_unsafe.tolist() and new_data not in new_unsafe_data:
                new_unsafe_data.append(new_data)
                total += 1
                progress.update(1)

        temp += 1
        if temp > 10 * num_increase:
            print("Not enough data!\n")
            break
    progress.close()
    return np.array(new_unsafe_data, dtype=object)


def sub_sample(context_safe, num_sub_safe):
    """
    :param context_safe: list of counts of safe data, [n]
    :param num_sub_safe: number of unsafe data after subsampling
    :return: all safe data after sub-sampling
    """
    return np.random.choice(context_safe, num_sub_safe, replace=False)


def smote(labels, context_paths, context_unsafe, index1, counts_unsafe, similarity):
    """
    :param labels: labels of raw data, [n]
    :param context_paths: all X data, [n]
    :param context_unsafe: all unsafe data, [n']
    :param index1: list of index of neighbors for each unsafe data, [n' * k_nearest]
    :param counts_unsafe: list of counts of unsafe data, [n']
    :param similarity: similarity between unsafe data, [n * k_nearest]
    :return: all balanced data, [n]
    """
    num = len(labels)
    num_unsafe = len(counts_unsafe)
    num_safe = num - num_unsafe
    print(f"labels (total): {num}, safe: {num_safe}, unsafe: {num_unsafe}")

    num_increase_unsafe = max(0, num // 2 - num_unsafe)
    num_sub_safe = max(0, num // 2)
    print(f"num_increase_unsafe: {num_increase_unsafe}, num_sub_safe: {num_sub_safe}")


    context_safe = context_paths[labels == 'safe']
    # over-sample and sub-sample data
    # print(context_safe, num_sub_safe)
    context_safe_flatten = np.concatenate(context_safe)

    sub_safe_data = sub_sample(context_safe_flatten, num_sub_safe)
    print(type(sub_safe_data))
    new_unsafe_data = over_sample(index1, counts_unsafe, context_unsafe, num_increase_unsafe, similarity)
    new_unsafe_data = np.array([[' '.join(c)] for c in new_unsafe_data])
    print(type(new_unsafe_data))

    unsafe_data = np.append(context_unsafe, new_unsafe_data)
    # print('UNSAFE', unsafe_data)
    # print('SAFE', sub_safe_data)
    smote_data = np.append(sub_safe_data, unsafe_data)
    print(smote_data)
    smote_label = ['safe'] * (num // 2) + ['unsafe'] * (num // 2)

    print("unsafe data: %d -> %d\nsafe data: %d -> %d\nentire data: %d -> %d" %
          (num_unsafe, len(unsafe_data), num_safe, len(sub_safe_data), len(context_paths), len(smote_data)))

    return smote_data, smote_label


def write_data_to_file(smote_data, smote_label, output):
    """
    :param smote_data: all balanced data
    :param smote_label: all balanced label
    :param output: output file name
    """
    # write to file
    file = 'dataset/%s' % output
    with open(file, 'a+') as out:
        out.seek(0)
        out.truncate()
        for i in range(len(smote_label)):
            str = "%s %s" % (smote_label[i], ' '.join(x for x in smote_data[i]))
            out.write(str + '\n')
    print("Done!\nOutput file: dataset/%s" % output)


def check_duplicates(data):
    """
    :param data: data in nd-array
    """
    if len(np.unique(data)) != len(data):
        print("Duplicates")
        print(len(np.unique(data)), len(data))
    else:
        print("No duplicates")


def main(file_path, k_nearest, output, details):
    """
    :param file_path: dataset file path
    :param k_nearest: the number of nearest neighbors
    :param output: output file name with extension
    :param details: if set, output similarity and neighbors
    """
    # parse file name and path
    base_name = os.path.basename(file_path)
    file_name = base_name.split('.')[0]
    print("=" * 20 + " loading data " + "=" * 20)
    start = time.time()

    # load data from file
    labels, counts, norms, context_paths = load_data(file_path)
    print("time: %.2fs" % (time.time() - start))
    print("number of data: %d" % (labels.shape[0]))

    # check duplicates
    print("\n" + "=" * 20 + " check duplicates in loaded data " + "=" * 20)
    check_duplicates(context_paths)

    # get unsafe data
    print("\n" + "=" * 20 + " calculating k-nearest neighbors " + "=" * 20)
    start = time.time()
    context_unsafe, counts_unsafe, norms_unsafe = get_unsafe_data(labels, counts, norms, context_paths)

    # compute similarity and find the neighbors
    index0, index1, similarity, num_neighbors = find_nearest(counts_unsafe, norms_unsafe, k_nearest)
    if details:
        print("time: %.2fs\n" % (time.time() - start))
        print("=" * 20 + " writing details " + "=" * 20)
        write_similarity_to_file(index0, index1, similarity, num_neighbors, context_unsafe, file_name, k_nearest)

    # smote algorithm
    print("\n" + "=" * 20 + " applying SMOTE algorithm " + "=" * 20)
    smote_data, smote_label = smote(labels, context_paths, context_unsafe, index1, counts_unsafe, similarity)
    print("\n" + "=" * 20 + " writing balanced data " + "=" * 20)
    write_data_to_file(smote_data, smote_label, output)

    # check duplicates
    print("\n" + "=" * 20 + " check duplicates in balanced data " + "=" * 20)
    check_duplicates(smote_data)


# if __name__ == "__main__":
#     # CLI
#     parser = argparse.ArgumentParser(description="Balance dataset")
#     parser.add_argument("-f", "--fpath", required=True, help="dataset file path")
#     parser.add_argument("-k", "--neighbors", type=int, default=5, help="number of neighbors")
#     parser.add_argument("-o", "--output", required=True, help="output file name with extension")
#     parser.add_argument("--details", action='store_true', help="output the similarity and neighbors in a file")
#     args = parser.parse_args()
#     file_path = args.fpath
#     k_nearest = args.neighbors
#     output = args.output

#     # main function
#     main(file_path, k_nearest, output, args.details)