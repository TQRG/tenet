import sklearn
import imblearn
import math
import pandas as pd
import numpy as np

from termcolor import colored
from collections import Counter

from securityaware.core.sampling import smote as sm
from securityaware.data.dataset import XYDataset, SplitDataset, Dataset

default_label = "safe"
other_label = "unsafe"


def restore_dataset(hash_dataset, dataset_dict):
    result = []
    for sample in hash_dataset:
        result.append(dataset_dict[int(sample[0])])

    return result


def restore_datasets(X_train, X_val, X_test, X_dataset):
    dataset_dict = {}
    for sample in X_dataset:
        dataset_dict[sample[1]] = sample[0]

    return restore_dataset(X_train, dataset_dict), restore_dataset(X_val, dataset_dict), restore_dataset(X_test,
                                                                                                         dataset_dict)


def calculate_labels_dist(dataset, dataset_type):
    n_default_label = 0
    n_other_label = 0
    for label in dataset:
        if label == default_label:
            n_default_label += 1
        else:
            n_other_label += 1

    label_distribution = {
        default_label: n_default_label,
        other_label: n_other_label
    }

    return label_distribution


def get_ratio(Y_train):
    n_default_label = 0
    n_other_label = 0
    for label in Y_train:
        if label == default_label:
            n_default_label += 1
        else:
            n_other_label += 1

    print("Train set has", n_default_label, "samples with default label.")
    print("Train set has", n_other_label, "samples with other label.")

    if n_other_label > 0 and (n_default_label > n_other_label):
        ratio = max(math.ceil(n_default_label // n_other_label * 2 / 3), 1)
    else:
        ratio = 10

    label_distribution = {
        default_label: n_default_label,
        other_label: n_other_label * ratio
    }

    print("We will multiply the minority class by", ratio)
    return ratio, label_distribution


def one_one_ratio(X_dataset: Dataset, Y_dataset: Dataset, seed) -> SplitDataset:
    context_paths = np.array(X_dataset.rows)
    labels = np.array(Y_dataset.rows)
    df = pd.DataFrame({'label': labels, 'context_path': list(context_paths)}, columns=['label', 'context_path'])

    for idx, row in df.iterrows():
        df.at[idx, 'context_path_string'] = row['context_path'][0]

    is_dup = df.duplicated(subset=['label', 'context_path_string'], keep='first')
    no_dup = df[~is_dup]
    dup = df[is_dup]

    print(f"Dataset has {len(no_dup)} unique functions and {len(dup)} duplicated functions.")
    labels = list(no_dup['label'].values)
    context_paths = [group.tolist() for group in no_dup['context_path'].values]

    n_default_label = 0
    n_other_label = 0
    for label in labels:
        if label == default_label:
            n_default_label += 1
        else:
            n_other_label += 1

    print("Resampled train set has", len(labels), "samples.")
    print("Resampled train set has", n_default_label, "samples with default label.")
    print("Resampled train set has", n_other_label, "samples with other label.")

    label_distribution = {
        default_label: n_other_label,
        other_label: n_other_label
    }

    ros = imblearn.under_sampling.RandomUnderSampler(sampling_strategy=label_distribution, random_state=seed)
    X_dataset, Y_dataset = ros.fit_resample(context_paths, labels)

    n_default_label = 0
    n_other_label = 0
    for label in Y_dataset:
        if label == default_label:
            n_default_label += 1
        else:
            n_other_label += 1

    print("Resampled train set has", len(Y_dataset), "samples.")
    print("Resampled train set has", n_default_label, "samples with default label.")
    print("Resampled train set has", n_other_label, "samples with other label.")

    print("Splitting dataset into train and test...")
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X_dataset.rows, Y_dataset.rows,
                                                                                test_size=0.1, random_state=seed)
    print("Splitting train set into train and validation...")
    X_train, X_val, Y_train, Y_val = sklearn.model_selection.train_test_split(X_train, Y_train, test_size=1 / 9,
                                                                              random_state=seed)  # (1/9) x 0.9 = 0.1

    print("Train set has", len(X_train), "samples.")
    print("Validation set has", len(X_val), "samples.")
    print("Test set has", len(X_test), "samples.")

    calculate_labels_dist(Y_train, 'train')
    calculate_labels_dist(Y_val, 'val')
    calculate_labels_dist(Y_test, 'test')

    return SplitDataset(train=XYDataset(Dataset(X_train), Dataset(Y_train)),
                        val=XYDataset(Dataset(X_val), Dataset(Y_val)),
                        test=XYDataset(Dataset(X_test), Dataset(Y_test)))


def unique_hash(X_dataset: Dataset, Y_dataset: Dataset, seed) -> SplitDataset:
    hashes = [sample[0] for sample in X_dataset.rows]
    code_locations = [sample[1] for sample in X_dataset.rows]
    #    context_paths = np.array(X_dataset)
    #    labels = np.array(Y_dataset)

    df = pd.DataFrame({'label': Y_dataset.rows, 'code_location': code_locations, 'hash': hashes},
                      columns=['label', 'code_location', 'hash'])

    no_unsafe_samples = len(df[df['label'] == 'unsafe'])
    print(f"Number of samples labeled as {other_label}: {no_unsafe_samples}")

    for idx, row in df.iterrows():
        df.at[idx, 'hash_string'] = row['hash'][0]

    is_dup = df.duplicated(subset=['label', 'hash_string'], keep='first')
    no_dup = df[~is_dup]
    dup = df[is_dup]

    print(f"Dataset has {len(no_dup)} unique functions and {len(dup)} duplicated functions.")
    labels = list(no_dup['label'].values)
    code_locations = [group for group in no_dup['code_location'].values]

    print("Splitting dataset into train and test...")
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(code_locations, labels, test_size=0.1,
                                                                                random_state=seed)
    print("Splitting train set into train and validation...")
    X_train, X_val, Y_train, Y_val = sklearn.model_selection.train_test_split(X_train, Y_train, test_size=1 / 9,
                                                                              random_state=seed)  # (1/9) x 0.9 = 0.1

    calculate_labels_dist(Y_train, 'train')
    calculate_labels_dist(Y_val, 'val')
    calculate_labels_dist(Y_test, 'test')

    return SplitDataset(train=XYDataset(Dataset(X_train), Dataset(Y_train)),
                        val=XYDataset(Dataset(X_val), Dataset(Y_val)),
                        test=XYDataset(Dataset(X_test), Dataset(Y_test)))


def unique(X_dataset: Dataset, Y_dataset: Dataset, seed) -> SplitDataset:
    context_paths = np.array(X_dataset.rows)
    labels = np.array(Y_dataset.rows)
    df = pd.DataFrame({'label': labels, 'context_path': list(context_paths)}, columns=['label', 'context_path'])

    no_unsafe_samples = len(df[df['label'] == 'unsafe'])
    print(f"Number of samples labeled as {other_label}: {no_unsafe_samples}")

    for idx, row in df.iterrows():
        df.at[idx, 'context_path_string'] = row['context_path'][0]

    is_dup = df.duplicated(subset=['label', 'context_path_string'], keep='first')
    no_dup = df[~is_dup]
    dup = df[is_dup]

    print(f"Dataset has {len(no_dup)} unique functions and {len(dup)} duplicated functions.")
    labels = list(no_dup['label'].values)
    context_paths = [group.tolist() for group in no_dup['context_path'].values]

    print("Splitting dataset into train and test...")
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(context_paths, labels, test_size=0.1,
                                                                                random_state=seed)
    print("Splitting train set into train and validation...")
    X_train, X_val, Y_train, Y_val = sklearn.model_selection.train_test_split(X_train, Y_train, test_size=1 / 9,
                                                                              random_state=seed)  # (1/9) x 0.9 = 0.1

    calculate_labels_dist(Y_train, 'train')
    calculate_labels_dist(Y_val, 'val')
    calculate_labels_dist(Y_test, 'test')

    return SplitDataset(train=XYDataset(Dataset(X_train), Dataset(Y_train)),
                        val=XYDataset(Dataset(X_val), Dataset(Y_val)),
                        test=XYDataset(Dataset(X_test), Dataset(Y_test)))


def disjoint(X_dataset: Dataset, Y_dataset: Dataset, seed) -> SplitDataset:
    context_paths = np.array(X_dataset.rows)
    labels = np.array(Y_dataset.rows)
    df = pd.DataFrame({'label': labels, 'context_path': list(context_paths)}, columns=['label', 'context_path'])

    no_unsafe_samples = len(df[df['label'] == 'unsafe'])
    print(f"Number of samples labeled as {other_label}: {no_unsafe_samples}")

    for idx, row in df.iterrows():
        df.at[idx, 'context_path_string'] = row['context_path'][0]

    is_dup = df.duplicated(subset=['label', 'context_path_string'], keep='first')
    no_dup = df[~is_dup]
    dup = df[is_dup]

    print(f"Dataset has {len(no_dup)} unique functions and {len(dup)} duplicated functions.")
    labels = list(no_dup['label'].values)
    context_paths = [group.tolist() for group in no_dup['context_path'].values]

    print("Splitting dataset into train and test...")
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(context_paths, labels, test_size=0.2,
                                                                                random_state=seed)
    print("Splitting train set into train and validation...")
    X_train, X_val, Y_train, Y_val = sklearn.model_selection.train_test_split(X_train, Y_train, test_size=2 / 8,
                                                                              random_state=seed)  # (1/9) x 0.9 = 0.1

    print("Train set has", len(X_train), "samples.")
    print("Validation set has", len(X_val), "samples.")
    print("Test set has", len(X_test), "samples.")

    X_test_samples_str = [group[0] for group in X_test]
    X_val_samples_str = [group[0] for group in X_val]

    # dups that are not in X_val and X_test dataset
    train_dups = dup[
        (~dup['context_path_string'].isin(X_test_samples_str)) & (~dup['context_path_string'].isin(X_val_samples_str))]
    print(f"Adding {len(train_dups)} in the training set...")

    # add train_dups to training dataset
    X_train += [list(group) for group in train_dups['context_path'].values]
    Y_train += list(train_dups['label'].values)

    print(f"Train dataset has now {len(X_train)} samples and {len(Y_train)} labels.")

    ratio, label_dist = get_ratio(Y_train)

    ros = imblearn.over_sampling.RandomOverSampler(sampling_strategy=label_dist, random_state=seed)

    X_train, Y_train = ros.fit_resample(X_train, Y_train)

    n_default_label = 0
    n_other_label = 0
    for label in Y_train:
        if label == default_label:
            n_default_label += 1
        else:
            n_other_label += 1

    print("Resampled train set has", len(X_train), "samples.")
    print("Resampled train set has", n_default_label, "samples with default label.")
    print("Resampled train set has", n_other_label, "samples with other label.")

    calculate_labels_dist(Y_train, 'train')
    calculate_labels_dist(Y_val, 'val')
    calculate_labels_dist(Y_test, 'test')

    return SplitDataset(train=XYDataset(Dataset(X_train), Dataset(Y_train)),
                        val=XYDataset(Dataset(X_val), Dataset(Y_val)),
                        test=XYDataset(Dataset(X_test), Dataset(Y_test)))


def disjoint_hash(X_dataset: Dataset, Y_dataset: Dataset, seed) -> SplitDataset:
    ids = [[sample[1]] for sample in X_dataset.rows]
    hashes = [[sample[2]] for sample in X_dataset.rows]
    df = pd.DataFrame({'id': ids, 'label': Y_dataset.rows, 'hash': hashes}, columns=['id', 'label', 'hash'])

    no_unsafe_samples = len(df[df['label'] == other_label])
    print(f"Number of samples labeled as {other_label}: {no_unsafe_samples}")

    for idx, row in df.iterrows():
        df.at[idx, 'hash_string'] = row['hash'][0]
        df.at[idx, 'id_number'] = row['id'][0]

    df = df.astype({'id_number': int})

    is_dup = df.duplicated(subset=['label', 'hash_string'], keep='first')
    no_dup = df[~is_dup]
    dup = df[is_dup]

    print(f"Dataset has {len(no_dup)} unique functions and {len(dup)} duplicated functions.")
    labels = list(no_dup['label'].values)
    ids_and_hashes = [[this_id, this_hash] for this_id, this_hash in
                      zip(no_dup['id_number'].values, no_dup['hash_string'].values)]

    print("Splitting dataset into train and test...")
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(ids_and_hashes, labels, test_size=0.2,
                                                                                random_state=seed)
    print("Splitting train set into train and validation...")
    X_train, X_val, Y_train, Y_val = sklearn.model_selection.train_test_split(X_train, Y_train, test_size=2 / 8,
                                                                              random_state=seed)  # (1/9) x 0.9 = 0.1

    print("Train set has", len(X_train), "samples.")
    print("Validation set has", len(X_val), "samples.")
    print("Test set has", len(X_test), "samples.")

    X_test_samples_str = [group[1] for group in X_test]
    X_val_samples_str = [group[1] for group in X_val]

    # dups that are not in X_val and X_test dataset
    train_dups = dup[(~dup['hash_string'].isin(X_test_samples_str)) & (~dup['hash_string'].isin(X_val_samples_str))]
    print(f"Adding {len(train_dups)} in the training set...")

    # add train_dups to training dataset
    X_train += [[this_id, this_hash] for this_id, this_hash in
                zip(train_dups['id_number'].values, train_dups['hash_string'].values)]
    Y_train += list(train_dups['label'].values)

    print(f"Train dataset has now {len(X_train)} samples and {len(Y_train)} labels.")

    calculate_labels_dist(Y_train, 'train')
    calculate_labels_dist(Y_val, 'val')
    calculate_labels_dist(Y_test, 'test')

    ratio, label_dist = get_ratio(Y_train)

    ros = imblearn.over_sampling.RandomOverSampler(sampling_strategy=label_dist, random_state=seed)

    X_train, Y_train = ros.fit_resample(X_train, Y_train)

    n_default_label = 0
    n_other_label = 0
    for label in Y_train:
        if label == default_label:
            n_default_label += 1
        else:
            n_other_label += 1

    print("Resampled train set has", len(X_train), "samples.")
    print("Resampled train set has", n_default_label, "samples with default label.")
    print("Resampled train set has", n_other_label, "samples with other label.")

    calculate_labels_dist(Y_train, 'train')
    calculate_labels_dist(Y_val, 'val')
    calculate_labels_dist(Y_test, 'test')

    X_train, X_val, X_test = restore_datasets(X_train, X_val, X_test, X_dataset.rows)

    return SplitDataset(train=XYDataset(Dataset(X_train), Dataset(Y_train)),
                        val=XYDataset(Dataset(X_val), Dataset(Y_val)),
                        test=XYDataset(Dataset(X_test), Dataset(Y_test)))


def disjoint_smote(X_dataset: Dataset, Y_dataset: Dataset, seed) -> SplitDataset:
    context_paths = np.array(X_dataset.rows)
    labels = np.array(Y_dataset.rows)
    df = pd.DataFrame({'label': labels, 'context_path': list(context_paths)})

    print(
        f"Samples dist: [{other_label}: {len(df[df['label'] == 'unsafe'])}, {default_label}: {len(df[df['label'] == 'safe'])}]")

    for idx, row in df.iterrows():
        df.at[idx, 'context_path_string'] = row['context_path'][0]

    is_dup = df.duplicated(subset=['label', 'context_path_string'], keep='first')
    no_dup = df[~is_dup]
    dup = df[is_dup]

    print(
        f"Samples dist: [{other_label}: {len(no_dup[no_dup['label'] == 'unsafe'])}, {default_label}: {len(no_dup[no_dup['label'] == 'safe'])}]")

    print(f"Dataset has {len(no_dup)} unique functions and {len(dup)} duplicated functions.")
    labels = list(no_dup['label'].values)
    context_paths = [group.tolist() for group in no_dup['context_path'].values]

    print("Splitting dataset into train and test...")
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(context_paths, labels, test_size=0.2,
                                                                                random_state=seed)
    print("Splitting train set into train and validation...")
    X_train, X_val, Y_train, Y_val = sklearn.model_selection.train_test_split(X_train, Y_train, test_size=2 / 8,
                                                                              random_state=seed)  # (1/9) x 0.9 = 0.1

    print(f"Train set has {len(X_train)} context-paths, {len(Y_train)} labels.")
    print(f"Validation set has {len(X_val)} context-paths, {len(Y_val)} labels.")
    print(f"Testing set has {len(X_test)} context-paths, {len(Y_test)} labels.")

    count = []
    norm = []

    for row in X_train:
        # count frequency of each context path in a bag of context paths
        count0 = Counter(row[0].split(' '))
        count.append(count0)
        # L2 norm, for calculating cosine similarity
        norm.append(math.sqrt(sum(count0[e] ** 2 for e in list(count0))))

    counts = np.array(count, dtype=object)
    norms = np.array(norm)
    train_labels = np.array(Y_train)
    train_context_paths = np.array(X_train, dtype=object)

    context_unsafe, counts_unsafe, norms_unsafe = sm.get_unsafe_data(train_labels, counts, norms, train_context_paths)
    # print(counts_unsafe)
    index0, index1, similarity, num_neighbors = sm.find_nearest(counts_unsafe, norms_unsafe, 5)
    X_train, Y_train = sm.smote(train_labels, train_context_paths, context_unsafe, index1, counts_unsafe, similarity)
    # print(X_train)
    # print(Y_train)
    X_train = [[c] for c in X_train]
    # print(X_train)
    print(X_test)
    print(type(Y_test))
    print(type(Y_train))
    # print(smote_data)

    return SplitDataset(train=XYDataset(Dataset(X_train), Dataset(Y_train)),
                        val=XYDataset(Dataset(X_val), Dataset(Y_val)),
                        test=XYDataset(Dataset(X_test), Dataset(Y_test)))


def disjoint_smote_hash(X_dataset: Dataset, Y_dataset: Dataset, seed) -> SplitDataset:
    feature_vectors = [sample[0] for sample in X_dataset.rows]
    df = pd.DataFrame({'label': Y_dataset.rows, 'feature_vector': feature_vectors}, columns=['label', 'feature_vector'])
    print(
        f"Samples dist: [{other_label}: {len(df[df['label'] == other_label])}, {default_label}: {len(df[df['label'] == default_label])}]")

    for idx, row in df.iterrows():
        df.at[idx, "feature_vector_string"] = str(row["feature_vector"])

    is_dup = df.duplicated(subset=['label', 'feature_vector_string'], keep='first')
    no_dup = df[~is_dup]
    dup = df[is_dup]

    print(
        f"Samples dist: [{other_label}: {len(no_dup[no_dup['label'] == other_label])}, {default_label}: {len(no_dup[no_dup['label'] == default_label])}]")

    print(f"Dataset has {len(no_dup)} unique functions and {len(dup)} duplicated functions.")
    labels = list(no_dup['label'].values)
    feature_vectors = [group for group in no_dup['feature_vector'].values]

    print("Splitting dataset into train and test...")
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(feature_vectors, labels, test_size=0.2,
                                                                                random_state=seed)
    print("Splitting train set into train and validation...")
    X_train, X_val, Y_train, Y_val = sklearn.model_selection.train_test_split(X_train, Y_train, test_size=2 / 8,
                                                                              random_state=seed)  # (1/9) x 0.9 = 0.1

    print(f"Train set has {len(X_train)} samples, {len(Y_train)} labels.")
    print(f"Validation set has {len(X_val)} samples, {len(Y_val)} labels.")
    print(f"Testing set has {len(X_test)} samples, {len(Y_test)} labels.")

    print(f"Train dataset has now {len(X_train)} samples and {len(Y_train)} labels.")

    calculate_labels_dist(Y_train, 'train')
    calculate_labels_dist(Y_val, 'val')
    calculate_labels_dist(Y_test, 'test')

    n_default_label = 0
    n_other_label = 0
    for label in Y_train:
        if label == default_label:
            n_default_label += 1
        else:
            n_other_label += 1

    num = len(Y_train)

    label_dist = {
        default_label: max(0, num // 2),
        other_label: n_other_label
    }

    rus = imblearn.under_sampling.RandomUnderSampler(sampling_strategy=label_dist, random_state=seed)

    X_train, Y_train = rus.fit_resample(X_train, Y_train)

    n_default_label = 0
    n_other_label = 0
    for label in Y_train:
        if label == default_label:
            n_default_label += 1
        else:
            n_other_label += 1

    label_dist = {
        default_label: n_default_label,
        other_label: max(n_other_label, num // 2)
    }

    smote = imblearn.over_sampling.SMOTE(sampling_strategy=label_dist, random_state=seed)

    X_train, Y_train = smote.fit_resample(X_train, Y_train)

    n_default_label = 0
    n_other_label = 0
    for label in Y_train:
        if label == default_label:
            n_default_label += 1
        else:
            n_other_label += 1

    print("Resampled train set has", len(X_train), "samples.")
    print("Resampled train set has", n_default_label, "samples with default label.")
    print("Resampled train set has", n_other_label, "samples with other label.")

    return SplitDataset(train=XYDataset(Dataset(X_train), Dataset(Y_train)),
                        val=XYDataset(Dataset(X_val), Dataset(Y_val)),
                        test=XYDataset(Dataset(X_test), Dataset(Y_test)))


def split_data(X_dataset: Dataset, Y_dataset: Dataset, seed) -> SplitDataset:
    print("Splitting dataset into train and test...")
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X_dataset.rows, Y_dataset.rows,
                                                                                test_size=0.1, random_state=seed)
    print("Splitting train set into train and validation...")
    X_train, X_val, Y_train, Y_val = sklearn.model_selection.train_test_split(X_train, Y_train, test_size=1 / 9,
                                                                              random_state=seed)  # (1/9) x 0.9 = 0.1

    calculate_labels_dist(Y_train, 'train')
    calculate_labels_dist(Y_val, 'val')
    calculate_labels_dist(Y_test, 'test')

    return SplitDataset(train=XYDataset(Dataset(X_train), Dataset(Y_train)),
                        val=XYDataset(Dataset(X_val), Dataset(Y_val)),
                        test=XYDataset(Dataset(X_test), Dataset(Y_test)))


def oversampling(X_dataset: Dataset, Y_dataset: Dataset, seed) -> SplitDataset:
    print("Splitting dataset into train and test...")
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X_dataset.rows, Y_dataset.rows,
                                                                                test_size=0.1, random_state=seed)
    print("Splitting train set into train and validation...")
    X_train, X_val, Y_train, Y_val = sklearn.model_selection.train_test_split(X_train, Y_train, test_size=1 / 9,
                                                                              random_state=seed)  # (1/9) x 0.9 = 0.1

    print("Train set has", len(X_train), "samples.")
    print("Validation set has", len(X_val), "samples.")
    print("Test set has", len(X_test), "samples.")

    print("Calculating label ratio in train set...")

    n_default_label = 0
    n_other_label = 0
    for label in Y_train:
        if label == default_label:
            n_default_label += 1
        else:
            n_other_label += 1

    print("Train set has", n_default_label, "samples with default label.")
    print("Train set has", n_other_label, "samples with other label.")

    if n_other_label > 0 and (n_default_label > n_other_label):
        ratio = max(math.ceil(n_default_label // n_other_label * 2 / 3), 1)
    else:
        ratio = 10

    print("We will multiply the minority class by", ratio)

    label_distribution = {
        default_label: n_default_label,
        other_label: n_other_label * ratio
    }
    print("Final label distribution in train set will be", label_distribution)

    print("Resampling train set...")
    ros = imblearn.over_sampling.RandomOverSampler(sampling_strategy=label_distribution, random_state=seed)

    X_train, Y_train = ros.fit_resample(X_train, Y_train)

    n_default_label = 0
    n_other_label = 0
    for label in Y_train:
        if label == default_label:
            n_default_label += 1
        else:
            n_other_label += 1

    calculate_labels_dist(Y_train, 'train')
    calculate_labels_dist(Y_val, 'val')
    calculate_labels_dist(Y_test, 'test')

    print("Resampled train set has", len(X_train), "samples.")
    print("Resampled train set has", n_default_label, "samples with default label.")
    print("Resampled train set has", n_other_label, "samples with other label.")

    return SplitDataset(train=XYDataset(Dataset(X_train), Dataset(Y_train)),
                        val=XYDataset(Dataset(X_val), Dataset(Y_val)),
                        test=XYDataset(Dataset(X_test), Dataset(Y_test)))


def random_undersampling(X_dataset: Dataset, Y_dataset: Dataset, seed) -> SplitDataset:
    context_paths = np.array(X_dataset.rows)
    labels = np.array(Y_dataset.rows)
    df = pd.DataFrame({'label': labels, 'context_path': list(context_paths)}, columns=['label', 'context_path'])

    unsafe_samples = len(df[df['label'] == 'unsafe'])
    print(f"Number of samples labeled as {other_label}: {unsafe_samples}")
    safe_samples = len(df[df['label'] == 'safe'])
    print(f"Number of samples labeled as {default_label}: {safe_samples}")

    for idx, row in df.iterrows():
        df.at[idx, 'context_path_string'] = row['context_path'][0]

    is_dup = df.duplicated(subset=['label', 'context_path_string'], keep='first')
    no_dup = df[~is_dup]
    dup = df[is_dup]

    print(f"Dataset has {len(no_dup)} unique functions and {len(dup)} duplicated functions.")
    labels = list(no_dup['label'].values)
    context_paths = [group.tolist() for group in no_dup['context_path'].values]

    print("Splitting dataset into train and test...")
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(context_paths, labels, test_size=0.1,
                                                                                random_state=seed)
    print("Splitting train set into train and validation...")
    X_train, X_val, Y_train, Y_val = sklearn.model_selection.train_test_split(X_train, Y_train, test_size=1 / 9,
                                                                              random_state=seed)  # (1/9) x 0.9 = 0.1

    print(f"Train set has {len(X_train)} samples. | {calculate_labels_dist(Y_train, 'train')}")
    print(f"Validation set has {len(X_val)} samples. | {calculate_labels_dist(Y_val, 'val')}")
    print(f"Test set has {len(X_test)} samples. | {calculate_labels_dist(Y_test, 'test')}")

    X_test_samples_str = [group[0] for group in X_test]
    X_val_samples_str = [group[0] for group in X_val]

    # dups that are not in X_val and X_test dataset
    train_dups = dup[
        (~dup['context_path_string'].isin(X_test_samples_str)) & (~dup['context_path_string'].isin(X_val_samples_str))]
    print(colored(f"Adding {len(train_dups)} in the training set...", 'green'))

    # add train_dups to training dataset
    X_train += [list(group) for group in train_dups['context_path'].values]
    Y_train += list(train_dups['label'].values)

    print(f"Train set has {len(X_train)} samples. | {calculate_labels_dist(Y_train, 'train')}")
    print(f"Validation set has {len(X_val)} samples. | {calculate_labels_dist(Y_val, 'val')}")
    print(f"Test set has {len(X_test)} samples. | {calculate_labels_dist(Y_test, 'test')}")

    train_dist = calculate_labels_dist(Y_train, 'train')
    print(train_dist)
    label_distribution = {
        default_label: train_dist['unsafe'] * 3,
        other_label: train_dist['unsafe']
    }

    ros = imblearn.under_sampling.RandomUnderSampler(sampling_strategy=label_distribution, random_state=seed)
    X_train, Y_train = ros.fit_resample(X_train, Y_train)
    print(colored(f"Undersampling train dataset...", 'green'))

    print(f"Train set has {len(X_train)} samples. | {calculate_labels_dist(Y_train, 'train')}")

    return SplitDataset(train=XYDataset(Dataset(X_train), Dataset(Y_train)),
                        val=XYDataset(Dataset(X_val), Dataset(Y_val)),
                        test=XYDataset(Dataset(X_test), Dataset(Y_test)))
