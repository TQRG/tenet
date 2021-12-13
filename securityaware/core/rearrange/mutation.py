import shutil
import subprocess
import csv
import time
import os
from pathlib import Path
from typing import List, TextIO, Tuple

import pandas as pd

from securityaware.core.rearrange.convert_bound import transform_inline_diff
from tqdm import tqdm

from securityaware.data.diff import InlineDiff

transform_files = ["transforms/randomizeFunctionName_bound1.js",  # 1
                   "transforms/randomizeFunctionName_bound2.js",  # 2
                   "transforms/randomizeFunctionName_bound3.js",  # 3
                   "transforms/randomizeVariableNames_bound.js",  # 4
                   "transforms/randomizeFunctionName.js",  # 5
                   "transforms/randomizeVariableNames.js",  # 6
                   "transforms/shuffleParameters.js",  # 7
                   "transforms/introduceParameter.js"]


def clone_files(out_file: TextIO, dataset: pd.DataFrame, output_path: Path, source_path: Path):
    out_file.seek(0)
    out_file.truncate()
    # write headers
    out_file.write("project,fpath,sline,scol,eline,ecol,label,func_id,n_mut" + '\n')

    print()
    print("=" * 20 + " Copying original data files " + "=" * 20)
    unique_files = set()

    for i, row in tqdm(dataset.iterrows()):
        # get the file from the label file
        file_name = "%s/%s" % (row.project, row.file_path)
        folder = file_name.split('/')[:-1]
        folder = output_path / '/'.join(folder)

        if not os.path.isfile(source_path / file_name):
            print("No such file: %s" % (source_path / file_name))
            continue

        if file_name not in unique_files:
            subprocess.run(["mkdir", '-p', folder])
            subprocess.run(["cp", source_path / file_name, folder])
        unique_files.add(file_name)


def write_files(out_file: TextIO, dataset: list):
    print()
    print("=" * 20 + " Writing original label file " + "=" * 20)
    for sample in tqdm(dataset):
        old_boundary = "%s,%s,%s,%s" % (sample[2], sample[3], sample[4], sample[5])
        old_fn_id = "%s_%s_%s" % (sample[0], sample[1], old_boundary.replace(",", "_"))
        old_sample = "%s,%s,%s,%s,%s,%d" % (sample[0], sample[1], old_boundary, "unsafe", old_fn_id, 0)
        out_file.write(old_sample + '\n')


def transform(transform_ids: List[int], output_path: Path, new_file: str, old_boundary: str):
    timeouts = 0
    for t in transform_ids:
        tf = transform_files[t - 1]
        print("\tPerforming " + tf)
        try:
            subprocess.call(
                ["jscodeshift", '-s', '-t', tf, output_path / new_file, "--loc=" + old_boundary],
                timeout=10)
        except subprocess.TimeoutExpired:
            print("timed out")
            timeouts += 1
            continue

    return timeouts


def write_boundary(input_file: str, label_file: TextIO, inline_diff: InlineDiff, sample: list, index: int):
    # get new boundary and update the new label file
    result = transform_inline_diff(code_path=input_file, inline_diff=inline_diff)
    new_boundary = '\n'.join(result)[:-1]

    if new_boundary and 'No' not in new_boundary:
        new_fn_id = "%s_%d_%s_%s_MUT" % (sample[0], index, sample[1], new_boundary.replace(",", "_"))
        new_sample = "%s_%d,%s,%s,%s,%s,%d" % (sample[0], index, sample[1], new_boundary, "mutated_unsafe",
                                               new_fn_id, index + 1)
        label_file.write(new_sample + '\n')
    else:
        print("No boundary matches")


def read_dataset(csv_file: str):
    """
        Get files that appears in the label file
    """

    with open(csv_file) as dataset_file:
        reader = csv.reader(dataset_file, delimiter=',')
        next(reader, None)  # skip the headers

        return [row for row in tqdm(reader)]


def find_resume(new_label_file: Path, dataset: pd.DataFrame) -> Tuple[int, list]:
    """
        Determine if resume by checking label file
    """
    flag = 0
    repos = []
    if new_label_file.is_file():
        flag = 1

        dataset_new = read_dataset(str(new_label_file))
        repos = dataset['project']

        if len(dataset_new) < len(dataset):
            flag = 0
        del dataset_new

    return flag, repos


def mutate(dataset: pd.DataFrame, original_label_file: Path, output_path: Path, source_path: Path, mutate_num: int,
           transforms: List[int]):
    new_label_file = output_path / original_label_file.name
    resume_flag, all_repos = find_resume(new_label_file, dataset)

    # if not resume, make new folder and copy the label file
    if not resume_flag:
        if output_path.exists():
            shutil.rmtree(str(output_path))
        output_path.mkdir(parents=True)
        shutil.copy(original_label_file, output_path)
    else:
        print("=" * 20 + " Auto-resume, looking for resuming point " + "=" * 20)

    with new_label_file.open(mode='a') as label_file:
        times = [0]
        num_samples = len(dataset)
        start = time.time()

        if not resume_flag:
            clone_files(label_file, dataset, output_path=output_path, source_path=source_path)
            # subprocess.run(["rsync", "-ahr", "--info=progress2", "--no-inc-recursive", source_path, output_path])
            write_files(label_file, dataset)

        # n rounds for the entire dataset
        for i in range(mutate_num):
            print()
            print("=" * 20 + " Round %d/%d " % (i + 1, mutate_num) + "=" * 20)
            unique_file_list = set()

            # traverse the dataset
            j = 0
            while j < num_samples:
                sample = dataset[j]
                inline_diff = InlineDiff(sample[2], sample[3], sample[4], sample[5])
                # if resume, find the resuming point
                if resume_flag and "%s_%d" % (sample[0], i) in all_repos:
                    print("Skip %d/%d sample" % (j + 1, num_samples))
                    j += 1
                    continue
                elif resume_flag:
                    print("Resume from %d/%d sample" % (j, num_samples))
                    j -= 1
                    resume_flag = 0
                    continue

                # get the file from the label file
                file = "%s/%s" % (sample[1], sample[2])

                if not os.path.isfile(source_path / file):
                    print("No such file: %s" % (source_path / file))
                    j += 1
                    continue

                # copy the file to new folder and perform transforms
                new_file = "%s_%d/%s" % (sample[0], i, sample[1])
                old_boundary = "%s,%s,%s,%s" % (sample[2], sample[3], sample[4], sample[5])

                # if it's first time see the file, copy file
                if file not in unique_file_list or max(transforms) <= 4:
                    print("Executing on %d/%d sample" % (j + 1, num_samples))
                    new_folder = output_path / '/'.join(new_file.split('/')[:-1])

                    # if it's first time see the file, copy file
                    if file not in unique_file_list:
                        print("\tCopying file " + new_file)
                        new_folder.mkdir(parents=True, exist_ok=True)
                        shutil.copy(source_path / file, new_folder)

                    # perform transforms on files
                    j += transform(transforms, output_path=output_path, new_file=new_file,
                                   old_boundary=old_boundary)
                    unique_file_list.add(file)

                print("\tGetting new boundaries on %d/%d file: %s" % (j + 1, num_samples, new_file))
                write_boundary(output_file=str(output_path / new_file), label_file=label_file, index=i, sample=sample,
                               old_boundary=old_boundary)
                j += 1

            times.append(time.time() - start)
            print(
                "\nTime consumed for the round %d/%d: %.2fmin\n" % (i + 1, mutate_num, (times[i + 1] - times[i]) / 60))
        print("\nTime consumed for the whole mutation: %.2fmin\n" % ((time.time() - start) / 60))
