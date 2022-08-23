import pandas as pd

from typing import Union
from pathlib import Path

from securityaware.data.output import CommandData
from securityaware.handlers.plugin import PluginHandler


class HistogramHandler(PluginHandler):
    """
        Histogram plugin
    """

    class Meta:
        label = "histogram"

    def __init__(self, **kw):
        super().__init__(**kw)

    (["-tdf", "--train_data_file"], {'help': 'The train dataset file.', 'required': True}),
    (["-thf", "--target_hist_file"], {'help': 'Target histogram output file.', 'required': True}),
    (["-ohf", "--origin_hist_file"], {'help': 'Origin histogram output file.', 'required': True}),
    (["-phf", "--path_hist_file"], {'help': 'Path histogram output file.', 'required': True}),

    def run(self, dataset: pd.DataFrame, **kwargs) -> Union[pd.DataFrame, None]:
        """
            runs the plugin
        """

        if not self.get('train_data_file'):
            self.app.log.warning(f"Train data file not instantiated.")
            return None

        if not Path(self.get('train_data_file')).exists():
            self.app.log.warning(f"Train data file not found.")
            return None

        target_hist_file = Path(self.path, 'target.hist.txt')
        origin_hist_file = Path(self.path, 'origin.hist.txt')
        path_hist_file = Path(self.path, 'path.hist.txt')

        self.set('target_hist_file', target_hist_file)
        self.set('origin_hist_file', origin_hist_file)
        self.set('path_hist_file', path_hist_file)

        command_handler = self.app.handler.get('handlers', 'command', setup=True)

        # histogram of the labels
        cmd_args = f"cat {self.get('train_data_file')} | cut -d' ' -f1"
        cmd_args += " | awk '{n[$0]++} END {for (i in n) print i,n[i]}' > " + str(target_hist_file)
        command_handler(cmd_data=CommandData(args=cmd_args))

        # histogram of all source/target words
        cmd_args = f"cat {self.get('train_data_file')} | cut -d' ' -f2 - | tr ' ' '\\n' | cut -d ',' -f1-3 | tr ',' '\\n'"
        cmd_args += " | awk '{n[$0]++} END {for (i in n) print i,n[i]}' > " + str(origin_hist_file)
        command_handler(cmd_data=CommandData(args=cmd_args))

        # histogram of all the path hashes
        cmd_args = f"cat {self.get('train_data_file')} | cut -d ' ' -f2 - | tr ' ' '\\n' | cut -d ',' -f2"
        cmd_args += " | awk '{n[$0]++} END {for (i in n) print i,n[i]}' > " + str(path_hist_file)
        command_handler(cmd_data=CommandData(args=cmd_args))

        return dataset


def load(app):
    app.handler.register(HistogramHandler)
