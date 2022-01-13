import json
from pathlib import Path

import pandas as pd

from typing import Union

from securityaware.handlers.plugin import PluginHandler
from sklearn.metrics import classification_report


class CompareHandler(PluginHandler):
    """
        Separate plugin
    """
    class Meta:
        label = "compare"

    def run(self, dataset: pd.DataFrame, labels=None, **kwargs) -> Union[pd.DataFrame, None]:
        """
            runs the plugin
        """

        if labels is None:
            labels = ['safe', 'unsafe']
        self.set('labels', labels)

        diff_dataset = self.get('diff_dataset')
        diff_funcs_path = self.get('diff_funcs')
        static_funcs_path = self.get('static_funcs')

        headers = ['label', 'hash', 'fpath', 'sline', 'scol', 'eline', 'ecol']
        diff_dataset = pd.read_csv(str(diff_funcs_path), names=headers)
        diff_dataset.drop_duplicates(subset=['hash'], inplace=True, keep='last')
        codeql_dataset = pd.read_csv(str(static_funcs_path), names=headers)
        codeql_dataset.drop_duplicates(subset=['hash'], inplace=True, keep='last')

        print(
            f"Diff: {len(diff_dataset)}; Safe: {len(diff_dataset[diff_dataset['label'] == 'safe'])}; Unsafe: {len(diff_dataset[diff_dataset['label'] == 'unsafe'])};")
        print(
            f"CodeQl: {len(codeql_dataset)}; Safe: {len(codeql_dataset[codeql_dataset['label'] == 'safe'])}; Unsafe: {len(codeql_dataset[codeql_dataset['label'] == 'unsafe'])};")

        merged = pd.merge(diff_dataset, codeql_dataset, on=['hash', 'hash'])
        print(f"Merged {len(merged)}")

        orig = merged.label_x.apply(lambda x: 0 if x == "safe" else 1).values
        pred = merged.label_y.apply(lambda x: 0 if x == "safe" else 1).values
        self.set('orig', orig)
        self.set('pred', pred)
        # precision, recall, th = precision_recall_curve(orig, pred)
        report = classification_report(orig, pred, target_names=labels)
        print(report)

        with Path(self.path, 'report.json').open(mode="w") as rf:
            json.dump(report, rf)

        return merged


def load(app):
    app.handler.register(CompareHandler)
