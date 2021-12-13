from pathlib import Path

from cement import Controller, ex

nn_models = ['code2vec', 'code_bert']


def count_labels(path: Path, kind: str):
    with path.open(mode='r') as df:
        safe = 0
        unsafe = 0
        for line in df.readlines():
            if line.startswith('safe'):
                safe += 1
            else:
                unsafe += 1

    print(f'=== {kind} dataset ===\nsafe:{safe}\nunsafe:{unsafe}')


class Model(Controller):
    class Meta:
        label = 'model'
        stacked_on = 'base'
        stacked_type = 'nested'

        arguments = [
            (['-m', '--model'], {'help': "Name of the neural network model", 'choices': nn_models, 'required': True})
        ]

    @ex(
        help="",
        arguments=[
            (['-dd', '--data_dir'], {'help': "The path to the directory with the train, val, and test data sets",
                                     'type': str, 'required': True}),
            (['-d', '--dataset'], {'help': "the name of the dataset, as was preprocessed using preprocess.sh",
                                   'type': str, 'required': True}),
            (['-vd', '--val_data'], {'help': 'the validation set, since this is the set that will be evaluated '
                                             'after each training iteration.'}),
            (['-n', '--name'], {'help': 'the name of the new model, used as the saved file name.', 'type': str,
                                'required': True}),
            (['-md', '--model_dir'], {'help': "The path to the directory to output the model",
                                     'type': str, 'required': True}),
        ]
    )
    def train(self):
        data_dir = Path(self.app.pargs.data_dir)
        data = data_dir / self.app.pargs.dataset

        #count_labels(data_dir / "train.raw.txt", 'train')
        #count_labels(data_dir / "validation.raw.txt", 'validation')
        #count_labels(data_dir / "test.raw.txt", 'test')

        model_dir = Path(self.app.pargs.model_dir)

        if model_dir.exists():
            self.app.log.warning(f"Path {model_dir} exists and will be used. "
                                 f"Some of the files might be overwritten")

        container_handler = self.app.handler.get('handlers', 'container', setup=True)
        model_container = container_handler.get(self.app.pargs.model)
        container_handler.start(self.app.pargs.model)

        cmd_data = container_handler(container_id=model_container.id, cmd_str=f"mkdir -p {model_dir}")

        if not cmd_data.error:
            train_cmd = f"python3 {self.app.pargs.model}.py --data {data} --test {data_dir / self.app.pargs.val_data} " \
                        f"--save {model_dir}/saved_model"
            cmd_data = container_handler(container_id=model_container.id, cmd_str=train_cmd, cmd_cwd='/code2vec_vp')

            if not cmd_data.error:
                container_handler.write(container=model_container, data=cmd_data.output, path=model_dir,
                                        name="logs.txt")
