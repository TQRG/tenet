import pandas as pd
import re

from typing import Union
from pathlib import Path

from securityaware.core.exc import Skip
from securityaware.utils.misc import count_labels
from securityaware.data.schema import ContainerCommand
from securityaware.handlers.plugin import PluginHandler


class Code2vecHandler(PluginHandler):
    """
        Code2vec plugin
    """

    class Meta:
        label = "code2vec"

    def __init__(self, **kw):
        super().__init__(**kw)

    def run(self, dataset: pd.DataFrame, max_contexts: int = 200, emb_size: int = 128, train: bool = True,
            image_name: str = "code2vec", **kwargs) -> Union[pd.DataFrame, None]:
        """
            runs the plugin
        """

        model_dir = Path(self.app.workdir, Path(self.path).name)

        if self.get('save_path'):
            save_path = self.get('save_path')
        else:
            save_path = f"{model_dir}/saved_model"

        self.set('save_path', save_path)

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

        count_labels(Path(train_data_path), 'train')
        count_labels(Path(val_data_path), 'validation')
        count_labels(Path(test_data_path), 'test')

        val_data_path = val_data_path.replace(str(self.app.workdir), str(self.app.bind))

        # TODO: find better way to skip training when model exists
        if Path(save_path + '_iter19.index').exists() and train:
            raise Skip(f"Model path {save_path} exists. Skipping")

        container_name = f"{self.app.workdir.name}_code2vec"
        container_handler = self.app.handler.get('handlers', 'container', setup=True)
        code2vec_container = container_handler[container_name]

        if code2vec_container:
            _id = code2vec_container.id
        else:
            _id = container_handler.create(image_name, container_name)
            code2vec_container = container_handler[container_name]

        container_handler.start(_id)
        # TODO: find better way of performing this bind
        model_dir = Path(self.app.bind, Path(self.path).name)
        save_path = save_path.replace(str(self.app.workdir), str(self.app.bind))
        container_handler.mkdir(_id, str(model_dir))

        # TODO: fix this
        #container_handler.load(self.edge, dataset_name='output', ext='')
        container_handler.path = ''

        dataset_name = Path(val_data_path).stem.split('.')[0]
        data_dir = Path(Path(val_data_path).parent, dataset_name)
        step = 'train' if train else 'test'
        w2v_file = Path(str(self.path).replace(str(self.app.workdir), str(self.app.bind)), f'{step}_embeddings.kv')
        #default = f"python3 code2vec.py --max-contexts {max_contexts} --emb-size {emb_size}"
        default = f"python3 code2vec.py --max-contexts {max_contexts}"

        if train:
            cmd = ContainerCommand(
                org=f"{default} --data {data_dir} --test {val_data_path} --save {save_path}")
        else:
            cmd = ContainerCommand(org=f"{default} --load {save_path} --test {data_dir}.test.c2v")

        log_file = Path(self.path, f'{step}_output.txt')
        error_file = Path(self.path, f'{step}_errors.txt')

        outcome, cmd_data = container_handler.run_cmds(code2vec_container.id, [cmd])

        if cmd_data[0].error:
            with error_file.open(mode='w') as f:
                f.write(cmd_data[0].error)

        if cmd_data[0].output:
            self.parse_results(cmd_data[0].output, train)

            with log_file.open(mode='w') as f:
                f.write(cmd_data[0].output)

        return None

    def parse_results(self, output: str, train: bool):
        reg_exp = '\s+Precision: (?P<precision>\d+\.*\d*), Sensitivity\/Recall: (?P<recall>\d+\.*\d*), ' + \
                  'Accuracy: (?P<acc>\d+\.*\d*), Error Rate: (?P<err>\d+\.*\d*), F1: (?P<f1>\d+\.*\d*), ' + \
                  '\#TPs=(?P<tp>\d+) \(shoud be \d*\), #TNs=(?P<tn>\d+), #FPs=(?P<fp>\d+), #FNs=(?P<fn>\d+),' + \
                  ' TNR=(?P<tnr>\d+\.*\d*), FPR=(?P<fpr>\d+\.*\d*)'
        results = []
        step = 'train' if train else 'test'

        for line in output.splitlines():
            match = re.search(reg_exp, line)

            if match:
                results.append(match.groupdict())

        df = pd.DataFrame(results)
        df.to_csv(f"{self.path}/{step}_results.csv")


def load(app):
    app.handler.register(Code2vecHandler)
