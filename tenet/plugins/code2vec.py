import pandas as pd
import re

from typing import Union
from pathlib import Path

from tenet.utils.misc import count_labels
from tenet.data.schema import ContainerCommand
from tenet.handlers.plugin import PluginHandler


def parse_results(output: str):
    reg_exp = '\s+Precision: (?P<precision>\d+\.*\d*), Sensitivity\/Recall: (?P<recall>\d+\.*\d*), ' + \
              'Accuracy: (?P<acc>\d+\.*\d*), Error Rate: (?P<err>\d+\.*\d*), F1: (?P<f1>\d+\.*\d*), ' + \
              'MCC: (?P<mcc>\d+\.*\d*), \#TPs=(?P<tp>\d+) \(shoud be \d*\), #TNs=(?P<tn>\d+), #FPs=(?P<fp>\d+), ' + \
              '#FNs=(?P<fn>\d+), TNR=(?P<tnr>\d+\.*\d*), FPR=(?P<fpr>\d+\.*\d*)'
    results = []

    for line in output.splitlines():
        match = re.search(reg_exp, line)

        if match:
            results.append(match.groupdict())

    return results


class Code2vecHandler(PluginHandler):
    """
        Code2vec plugin
    """

    class Meta:
        label = "code2vec"

    def __init__(self, **kw):
        super().__init__(**kw)

    def run(self, dataset: pd.DataFrame, max_contexts: int = 200, emb_size: int = 128, training: bool = True,
            evaluate: bool = True, image_name: str = "code2vec", **kwargs) -> Union[pd.DataFrame, None]:
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
        container = self.container_handler.run(image_name=image_name)
        # TODO: find better way of performing this bind
        save_path = save_path.replace(str(self.app.workdir), str(self.app.bind))
        dataset_name = Path(val_data_path).stem.split('.')[0]
        data_dir = Path(Path(val_data_path).parent, dataset_name)
        # step = 'train' if train else 'test'
        # w2v_file = Path(str(self.path).replace(str(self.app.workdir), str(self.app.bind)), f'{step}_embeddings.kv')
        default = f"python3 /code2vec_vp/code2vec.py --max-contexts {max_contexts}"
        cmds = []

        if training:
            cmds.append(ContainerCommand(org=f"{default} --data {data_dir} --test {val_data_path} --save {save_path}",
                                         parse_fn=parse_results, tag='train'))
        if evaluate:
            # TODO: fix path
            cmds.append(ContainerCommand(org=f"{default} --load {save_path} --test {data_dir}.test.c2v",
                                         parse_fn=parse_results, tag='test'))

        outcome, cmd_data = self.container_handler.run_cmds(container.id, cmds)

        for cd in cmd_data:
            df = pd.DataFrame(cd.parsed_output)
            df.to_csv(f"{self.path}/{cd.tag}_results.csv")

        self.container_handler.stop(container)

        return dataset


def load(app):
    app.handler.register(Code2vecHandler)
