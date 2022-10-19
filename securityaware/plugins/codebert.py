import pandas as pd

from typing import Union
from pathlib import Path

from securityaware.data.schema import ContainerCommand
from securityaware.handlers.plugin import PluginHandler


def parse_results(output: str):
    results = []

    # TODO: Parse output into results
    for line in output.splitlines():
        print(line)
    #    match = re.search(reg_exp, line)

    #    if match:
    #        results.append(match.groupdict())
    return results


class CodeBERTHandler(PluginHandler):
    """
        CodeBERT plugin
    """

    class Meta:
        label = "codebert"

    def __init__(self, **kw):
        super().__init__(**kw)

    def run(self, dataset: pd.DataFrame, train: bool = True, evaluate: bool = True, gpus: int = 0, max_epochs: int = 20,
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

        container = self.container_handler.run(image_name=image_name, node_name=self.node.name)
        #save_path = save_path.replace(str(self.app.workdir), str(self.app.bind))

        #container_handler.load(self.edge, dataset_name='output', ext='')
        step = 'train' if train else 'test'

        default = f"python3 /code_bert/codebert.py --task {step} --gpus {gpus} --max_epochs {max_epochs}"
        default = f"{default} --train_dataset_file {train_lines_path} --validation_dataset_file {val_lines_path}"
        default = f"{default} --test_dataset_file {test_lines_path} --train_offset_file {train_offset_file}"
        default = f"{default} --validation_offset_file {val_offset_file} --test_offset_file {test_offset_file}"

        cmds = []

        if train:
            cmds.append(ContainerCommand(org=f"{default}", parse_fn=parse_results, tag='train'))
        if evaluate:
            cmds.append(ContainerCommand(org=f"{default}", parse_fn=parse_results, tag='test'))

        outcome, cmd_data = self.container_handler.run_cmds(container.id, cmds)

        for cd in cmd_data:
            df = pd.DataFrame(cd.parsed_output)
            df.to_csv(f"{self.path}/{cd.tag}_results.csv")

        self.container_handler.stop(container)

        return dataset


def load(app):
    app.handler.register(CodeBERTHandler)
