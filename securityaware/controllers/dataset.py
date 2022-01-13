import pandas as pd

from pathlib import Path
from cement import Controller, ex

from securityaware.core.astminer.common import common
from securityaware.core.astminer.preprocess import process_data, save_dictionaries
from securityaware.core.sampling.jsonify import to_json, to_dict
from securityaware.core.sampling.offset import to_offset
from securityaware.data.output import CommandData
from securityaware.core.rearrange.mutation import mutate, read_dataset

transform_choices = ['randomizeVariableNames', 'randomizeFunctionName', 'shuffleParameters', 'introduceParameter',
                     'outputFnBoundary']

balance_techniques = ['over', 'disj_over', 'disj_smote', 'unique', 'undersampling', 'stratified', '1_to_1']


class Dataset(Controller):
    class Meta:
        label = 'dataset'
        stacked_on = 'base'
        stacked_type = 'nested'

    @ex(
        help='Gets the project commits in the dataset',
        arguments=[(['-o', '--output'], {'help': 'The output directory for the projects.', 'action': 'store',
                                         'required': True}),
                   (['-d', '--dataset'], {'help': 'Path to the input dataset.', 'action': 'store', 'required': True})
                   ]
    )
    def download(self):
        out_dir = Path(self.app.pargs.output)

        if not out_dir.exists():
            out_dir.mkdir(parents=True, exist_ok=True)

        dataset_path = Path(self.app.pargs.dataset)
        dataset_handler = self.app.handler.get('handlers', 'dataset', setup=True)
        results = dataset_handler(out_dir, dataset=dataset_path)

        with (out_dir / (dataset_path.stem + '.raw.csv')).open(mode="w") as out:
            out.write("a_proj,b_proj,start,a_path,b_path,a_file,b_file,label\n")

            for res in results.finished:
                if 'result' in res and res['result']:
                    for entry in res['result']:
                        out.write(f"{entry}\n")

    @ex(
        help='Label and transform the raw commit files into inline diffs.',
        arguments=[
            (['-o', '--output'], {'help': 'The output directory for the inline diffs.', 'action': 'store',
                                  'required': True}),
            (['-d', '--dataset'], {'help': 'Path to the input dataset.', 'action': 'store', 'required': True}),
            (['-ml', '--multi_label'], {'help': 'Flag that enables multi-labeling.', 'action': 'store_true'}),
            (['-fsl', '--file_size_limit'], {'help': 'Filters files with a size greater than the specified.',
                                             'type': int, 'required': False})
        ]
    )
    def label(self):
        """
            inline: /etc/securityaware/xss_advisories_inline.csv
        """
        out_dir = Path(self.app.pargs.output)

        if not out_dir.exists():
            out_dir.mkdir(parents=True, exist_ok=True)

        dataset_path = Path(self.app.pargs.dataset)
        label_handler = self.app.handler.get('handlers', 'label', setup=True)

        if self.app.pargs.multi_label:
            label_handler.multi_label = self.app.pargs.multi_label

        if self.app.pargs.file_size_limit:
            label_handler.file_size_limit = self.app.pargs.file_size_limit

        results = label_handler(out_dir, dataset=dataset_path)

        # Save similarity ratio as a csv file in the same output folder
        with (out_dir / f"{dataset_path.stem}.labels.csv").open(mode="w") as out, \
                (out_dir / f"{dataset_path.stem}.ratios.csv").open(mode="w") as ratio_file:
            out.write("project,fpath,sline,scol,eline,ecol,label\n")
            ratio_file.write("project_a,fpath_a,project_b,fpath_b,sim_ratio\n")

            for res in results.finished:
                if 'result' in res and res['result']:
                    if not res['result'][0]:
                        continue
                    ratio_file.write(f"{res['result'][1]}\n")
                    for inline_diff in res['result'][0]:
                        out.write(f"{inline_diff}\n")

    @ex(
        help='Rearrange/refactor JS functions.',
        arguments=[(['-tf', '--transform_file'], {'help': 'Transform file', 'required': True}),
                   (['-d', '--dataset'], {'help': 'Path to the inline dataset', 'required': True}),
                   (['-o', '--out_dir'], {'help': 'Path to output the resulting dataset', 'required': True}),
                   (['-cd', '--code_dir'], {'help': 'Path to the directory with the code files', 'required': True})],
    )
    def rearrange(self):
        """
           rearrange: /etc/securityaware/xss_advisories_rearrange.csv
        """
        out_dir = Path(self.app.pargs.out_dir)

        if not out_dir.exists():
            out_dir.mkdir(parents=True, exist_ok=True)

        rearrange_handler = self.app.handler.get('handlers', 'rearrange', setup=True)
        results = rearrange_handler(csv_in=Path(self.app.pargs.dataset),
                                    output_fn=self.app.pargs.transform_file,
                                    code_dir=Path(self.app.pargs.code_dir))
        fn_bounds = []

        for res in results.finished:
            if 'result' in res and res['result']:
                for fn_bound in res['result']:
                    fn_bounds.append(fn_bound.to_list())

        df = pd.DataFrame(fn_bounds, columns=['project', 'fpath', 'sline', 'scol', 'eline', 'ecol', 'label'])

        # Remove duplicates
        df = df.drop_duplicates(ignore_index=True)
        df = df.reset_index().rename(columns={'index': 'func_id'})
        df["n_mut"] = [0] * df.shape[0]
        out_dataset = out_dir / (Path(self.app.pargs.dataset).stem + '.func.csv')
        df.to_csv(out_dataset, index=False)

    @ex(
        help='Transform JS functions.',
        arguments=[(["-l", "--label_file"], {'help': "label file", 'required': True}),
                   (["-s", "--source"], {'help': "source file path", 'required': True}),
                   (["-o", "--output"], {'help': "output file path", 'required': True}),
                   (["-t", "--transforms"], {'help': "type of transforms", 'nargs': "+", 'type': int,
                                             'required': True}),
                   (["-n", "--mute_num"], {'help': "number of mutations", 'type': int, 'required': True})]
    )
    def mutation(self):
        dataframe = pd.read_csv(self.app.pargs.label_file)
        dataframe.rename(columns={"fpath": "file_path"}, inplace=True)
        mutate(dataset=dataframe, original_label_file=Path(self.app.pargs.label_file),
               mutate_num=self.app.pargs.mute_num,
               output_path=Path(self.app.pargs.output), source_path=Path(self.app.pargs.source),
               transforms=self.app.pargs.transforms)

    @ex(
        help='Sampling dataset into train/test/val.',
        arguments=[(["-i", "--input-dir"], {'help': 'Path to the input directory with extracted code.',
                                            'required': True}),
                   (["-o", "--output-dir"], {'help': 'Dir with dataset file where splits will be written',
                                             'required': True}),
                   (['-t', '--technique'], {'help': 'Type of balancing technique', 'required': False,
                                            'choices': balance_techniques}),
                   (['-s', '--seed'], {'help': 'Seed for random state.', 'required': False, 'default': 42,
                                       'type': int}),
                   (['-r', '--raw'], {'help': 'Path to the raw files in the dataset to build the JSONLines',
                                      'type': str, 'required': False}),
                   (['--offset'], {'help': '', 'required': False, 'action': 'store_true'})
                   ]
    )
    def sample(self):
        self.app.log.info((f"Sampling with {self.app.pargs.technique}.\n" if self.app.pargs.technique else "") +
                          f"Saving results to {self.app.pargs.output_dir}")

        sampling_handler = self.app.handler.get('handlers', 'sampling', setup=True)
        dataset_file = Path(self.app.pargs.input_dir,
                            'dataset.raw.txt' if self.app.pargs.offset else 'path_contexts.csv')
        datasets = sampling_handler.get_datasets(dataset_file=dataset_file,
                                                 delimiter=',' if self.app.pargs.offset else ' ')

        self.app.log.info(f"Dataset has {len(datasets.x)} samples.")
        split_dataset = sampling_handler(x=datasets.x, y=datasets.y, offset=self.app.pargs.offset,
                                         technique=self.app.pargs.technique, seed=self.app.pargs.seed)

        self.app.log.info("Writing splits to files...")
        data_files = split_dataset.write(out_dir=Path(self.app.pargs.output_dir),
                                         delimiter=',' if self.app.pargs.offset else ' ',
                                         headers="label,hash,fpath,sline,scol,eline,ecol" if self.app.pargs.offset else None)

        jsonlines_files = []

        if self.app.pargs.raw:
            raw_files = Path(self.app.pargs.raw)

            if not raw_files.exists():
                self.app.log.warning(f"{raw_files} not found")
                exit(1)

            for file in data_files:
                self.app.log.info(f"======== Building JSONLines for {file} while removing comments... ========")
                output_filename = Path(self.app.pargs.output_dir, f"{file.stem}.jsonl")
                jsonlines_files.append(output_filename)
                funcs = to_dict(file, replace=('/tmp/files', self.app.pargs.raw))
                to_json(raw_files, funcs, output_filename)

        if self.app.pargs.offset:
            for file in jsonlines_files:
                to_offset(file, file.parent / f"{file.stem}_offset_dict.json")

    @ex(
        help="Extracts path-contexts from dataset.",
        arguments=[(["-i", "--input-dir"], {'help': 'Path to the input directory with raw code.', 'required': True}),
                   (['-d', '--dataset'], {'help': 'Dataset name.', 'required': True}),
                   (["-o", "--output-dir"], {'help': 'Path to the output directory.', 'required': True}),
                   (['-b', '--balanced'], {'help': 'Flag to balance datasets', 'action': 'store_true'}),
                   (['-e', '--extractor'], {'help': 'astminer or jsextractor', 'default': 'astminer',
                                            'required': False}),
                   (['-t', '--technique'], {'help': 'Type of balancing technique', 'required': False,
                                            'choices': balance_techniques}),
                   (['-n', '--node_max'], {'help': 'NODE_MAX_MEM', 'type': int, 'default': 8192}),
                   (["-m", "--model"],
                    {'help': 'Name of the neural network model (codebert/code2vec).', 'required': True}),
                   ]
    )
    def extract(self):
        labels_filename = f"labels-{self.app.pargs.dataset}.csv"
        MUTATIONS = 0

        if self.app.pargs.balanced:
            dataset_name = f"{self.app.pargs.dataset}-balanced-{self.app.pargs.technique}"
        else:
            dataset_name = f"{self.app.pargs.dataset}-unbalanced"

        output_dir = Path(self.app.pargs.output_dir, "data_" + self.app.pargs.extractor, dataset_name)
        container_handler = self.app.handler.get('handlers', 'container', setup=True)
        astminer_container = container_handler.get("astminer")
        container_handler.start("astminer")

        cmd_data = container_handler(container_id=astminer_container.id, cmd_str=f"mkdir -p {output_dir}")

        if not cmd_data.error:

            cmd_data = container_handler(container_id=astminer_container.id,
                                         cmd_str=f"export NODE_OPTIONS=\"--max-old-space-size={self.app.pargs.node_max}\"")
            if not cmd_data.error:
                cmd_data = container_handler(container_id=astminer_container.id,
                                             cmd_str=f"java -jar -Xms4g -Xmx4g build/shadow/astminer.jar {self.app.pargs.model} {self.app.pargs.input_dir} {output_dir} {labels_filename} {MUTATIONS}")

                if not cmd_data.error:
                    # TODO: do something with the dataset when done
                    self.app.log.info("Done")

    @ex(
        help="",
        arguments=[
            (["-tdf", "--train_data_file"], {'help': 'The train dataset file.', 'required': True}),
            (["-thf", "--target_hist_file"], {'help': 'Target histogram output file.', 'required': True}),
            (["-ohf", "--origin_hist_file"], {'help': 'Origin histogram output file.', 'required': True}),
            (["-phf", "--path_hist_file"], {'help': 'Path histogram output file.', 'required': True}),
        ]
    )
    def histogram(self):
        # TRAIN_DATA_FILE=${OUTPUT_DIR}/train.raw.txt
        # TARGET_HISTOGRAM_FILE=${OUTPUT_DIR}/${DATASET_NAME}.histo.tgt.c2v
        # ORIGIN_HISTOGRAM_FILE=${OUTPUT_DIR}/${DATASET_NAME}.histo.ori.c2v
        # PATH_HISTOGRAM_FILE=${OUTPUT_DIR}/${DATASET_NAME}.histo.path.c2v
        command_handler = self.app.handler.get('handlers', 'command', setup=True)

        # histogram of the labels
        cmd_args = f"cat {self.app.pargs.train_data_file} | cut -d' ' -f1"
        cmd_args += " | awk '{n[$0]++} END {for (i in n) print i,n[i]}' > " + str(self.app.pargs.target_hist_file)
        command_handler(cmd_data=CommandData(args=cmd_args))

        # histogram of all source/target words
        cmd_args = f"cat {self.app.pargs.train_data_file} | cut -d' ' -f2 - | tr ' ' '\\n' | cut -d ',' -f1-3 | tr ',' '\\n'"
        cmd_args += " | awk '{n[$0]++} END {for (i in n) print i,n[i]}' > " + str(self.app.pargs.origin_hist_file)
        command_handler(cmd_data=CommandData(args=cmd_args))

        # histogram of all the path hashes
        cmd_args = f"cat {self.app.pargs.train_data_file} | cut -d ' ' -f2 - | tr ' ' '\\n' | cut -d ',' -f2"
        cmd_args += " | awk '{n[$0]++} END {for (i in n) print i,n[i]}' > " + str(self.app.pargs.path_hist_file)
        command_handler(cmd_data=CommandData(args=cmd_args))

    @ex(
        help="Extracts path-contexts from dataset.",
        arguments=[(["-trd", "--train_data"], {'dest': "train_data_path", 'help': "path to training data file",
                                               'required': True}),
                   (["-ted", "--test_data"], {'dest': "test_data_path", 'help': "path to test data file",
                                              'required': True}),
                   (["-vd", "--val_data"], {'dest': "val_data_path", 'help': "path to validation data file",
                                            'required': True}),
                   (["-mc", "--max_contexts"], {'dest': "max_contexts", 'default': 200, 'type': int,
                                                'help': "number of max contexts to keep", 'required': False}),
                   (["-wvs", "--word_vocab_size"], {'dest': "word_vocab_size", 'default': 1301136, 'required': False,
                                                    'help': "Max number of origin word in to keep in the vocabulary",
                                                    'type': int}),
                   (["-pvs", "--path_vocab_size"], {'dest': "path_vocab_size", 'default': 911417, 'required': False,
                                                    'help': "Max number of paths to keep in the vocabulary",
                                                    'type': int}),
                   (["-tvs", "--target_vocab_size"], {'dest': "target_vocab_size", 'default': 261245, 'required': False,
                                                      'help': "Max number of target words to keep in the vocabulary",
                                                      'type': int}),
                   (["-wh", "--word_histogram"], {'dest': "word_histogram", 'required': True,
                                                  'help': "word histogram file", 'metavar': "FILE"}),
                   (["-ph", "--path_histogram"], {'dest': "path_histogram", 'required': True,
                                                  'help': "path_histogram file", 'metavar': "FILE"}),
                   (["-th", "--target_histogram"], {'dest': "target_histogram", 'required': True,
                                                    'help': "target histogram file", 'metavar': "FILE"}),
                   (["-o", "--output_name"], {'dest': "output_name",
                                              'help': "output name - the base name for the created dataset",
                                              'metavar': "FILE", 'required': True, 'default': 'data'})
                   ]
    )
    def preprocess(self):
        # securityaware dataset preprocess -trd /tmp
        word_to_count, path_to_count, target_to_count = common.get_counts(word_histogram=self.app.pargs.word_histogram,
                                                                          word_vocab_size=self.app.pargs.word_vocab_size,
                                                                          path_histogram=self.app.pargs.path_histogram,
                                                                          path_vocab_size=self.app.pargs.path_vocab_size,
                                                                          target_histogram=self.app.pargs.target_histogram,
                                                                          target_vocab_size=self.app.pargs.target_vocab_size)

        num_training_examples = process_data(test_data_path=self.app.pargs.test_data_path,
                                             val_data_path=self.app.pargs.val_data_path,
                                             train_data_path=self.app.pargs.train_data_path,
                                             word_to_count=word_to_count, path_to_count=path_to_count,
                                             output_name=self.app.pargs.output_name,
                                             max_contexts=self.app.pargs.max_contexts)

        save_dictionaries(dataset_name=self.app.pargs.output_name, word_to_count=word_to_count,
                          path_to_count=path_to_count, target_to_count=target_to_count,
                          num_training_examples=num_training_examples)
