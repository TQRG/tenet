import pandas as pd
import re

from typing import Union
from pathlib import Path

from securityaware.core.exc import Skip
from securityaware.utils.misc import count_labels
from securityaware.data.schema import ContainerCommand
from securityaware.handlers.plugin import PluginHandler


class CodeBERTHandler(PluginHandler):
    """
        CodeBERT plugin
    """

    class Meta:
        label = "codebert"

    def __init__(self, **kw):
        super().__init__(**kw)

    def run(self, dataset: pd.DataFrame, train: bool = True, gpus: int = 0, max_epochs: int = 20,
            image_name: str = 'codebert', **kwargs) -> Union[pd.DataFrame, None]:
        """
            runs the plugin
        """

        model_dir = Path(self.app.workdir, Path(self.path).name)

#        if self.get('save_path'):
#            save_path = self.get('save_path')
#        else:
#            save_path = f"{model_dir}/saved_model"

#        self.set('save_path', save_path)

        train_lines_path = self.get('train_lines_path')
        val_lines_path = self.get('val_lines_path')
        test_lines_path = self.get('test_lines_path')

        train_offset_file = self.get('train_offset_file')
        val_offset_file = self.get('val_offset_file')
        test_offset_file = self.get('test_offset_file')

        if not train_lines_path:
            self.app.log.warning(f"Train lines file not instantiated.")
            return None

        if not Path(train_lines_path).exists():
            self.app.log.warning(f"Train lines file not found.")
            return None

        if not val_lines_path:
            self.app.log.warning(f"Validation lines file not instantiated.")
            return None

        if not Path(val_lines_path).exists():
            self.app.log.warning(f"Validation lines file not found.")
            return None

        if not test_lines_path:
            self.app.log.warning(f"Test lines file not instantiated.")
            return None

        if not Path(test_lines_path).exists():
            self.app.log.warning(f"Test lines file not found.")
            return None

        if not train_offset_file:
            self.app.log.warning(f"Train offset file not instantiated.")
            return None

        if not Path(train_offset_file).exists():
            self.app.log.warning(f"Train offset file not found.")
            return None

        if not val_offset_file:
            self.app.log.warning(f"Validation offset file not instantiated.")
            return None

        if not Path(val_offset_file).exists():
            self.app.log.warning(f"Validation offset file not found.")
            return None

        if not test_offset_file:
            self.app.log.warning(f"Test offset file not instantiated.")
            return None

        if not Path(test_offset_file).exists():
            self.app.log.warning(f"Test offset file not found.")
            return None

        train_lines_path = str(train_lines_path).replace(str(self.app.workdir), str(self.app.bind))
        val_lines_path = str(val_lines_path).replace(str(self.app.workdir), str(self.app.bind))
        test_lines_path = str(test_lines_path).replace(str(self.app.workdir), str(self.app.bind))

        train_offset_file = str(train_offset_file).replace(str(self.app.workdir), str(self.app.bind))
        val_offset_file = str(val_offset_file).replace(str(self.app.workdir), str(self.app.bind))
        test_offset_file = str(test_offset_file).replace(str(self.app.workdir), str(self.app.bind))

        # TODO: find better way to skip training when model exists
#        if Path(save_path + '_iter19.index').exists() and train:
#            raise Skip(f"Model path {save_path} exists. Skipping")

        container_name = f"{self.app.workdir.name}_codebert"
        container_handler = self.app.handler.get('handlers', 'container', setup=True)
        codebert_container = container_handler[container_name]

        if codebert_container:
            _id = codebert_container.id
        else:
            _id = container_handler.create(image_name, container_name)
            codebert_container = container_handler[container_name]

        container_handler.start(_id)
        # TODO: find better way of performing this bind
        model_dir = Path(self.app.bind, Path(self.path).name)
        #save_path = save_path.replace(str(self.app.workdir), str(self.app.bind))
        container_handler.mkdir(_id, str(model_dir))

        # TODO: fix this
        #container_handler.load(self.edge, dataset_name='output', ext='')
        container_handler.path = ''
        step = 'train' if train else 'test'

        default = f"python3 codebert.py --task {step} --gpus {gpus} --max_epochs {max_epochs}"
        default = f"{default} --train_dataset_file {train_lines_path} --validation_dataset_file {val_lines_path}"
        default = f"{default} --test_dataset_file {test_lines_path} --train_offset_file {train_offset_file}"
        default = f"{default} --validation_offset_file {val_offset_file} --test_offset_file {test_offset_file}"

        if train:
            cmd = ContainerCommand(org=f"{default}")
        else:
            cmd = ContainerCommand(org=f"{default}")

        log_file = Path(self.path, f'{step}_output.txt')
        error_file = Path(self.path, f'{step}_errors.txt')

        outcome, cmd_data = container_handler.run_cmds(codebert_container.id, [cmd])

        if cmd_data[0].error:
            with error_file.open(mode='w') as f:
                f.write(cmd_data[0].error)

        if cmd_data[0].output:
            # self.parse_results(cmd_data[0].output, train)

            with log_file.open(mode='w') as f:
                f.write(cmd_data[0].output)

        return None

    def parse_results(self, output: str, train: bool):
        results = []
        step = 'train' if train else 'test'

        # TODO: Parse output into results
        for line in output.splitlines():
            print(line)
        #    match = re.search(reg_exp, line)

        #    if match:
        #        results.append(match.groupdict())

        df = pd.DataFrame(results)
        df.to_csv(f"{self.path}/{step}_results.csv")


def load(app):
    app.handler.register(CodeBERTHandler)
