from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Union

from tenet.handlers.plugin import PluginHandler
from pylab import rcParams
from sklearn.metrics import confusion_matrix

sns.set(style='whitegrid', palette='muted', font_scale=3.5)
rcParams['figure.figsize'] = 14, 8


class PlotMatrixHandler(PluginHandler):
    """
        Separate plugin
    """
    class Meta:
        label = "plot_matrix"

    def run(self, dataset: pd.DataFrame, **kwargs) -> Union[pd.DataFrame, None]:
        """
            runs the plugin
        """
        labels = self.get('labels')
        conf_matrix = confusion_matrix(self.get('orig'), self.get('pred'))
        plt.figure(figsize=(12, 12))
        akws = {"ha": 'center', "va": 'center'}
        sns.set(font_scale=3.0)
        sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white',})
        sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, annot_kws=akws, fmt="d", cmap="flare", color='white')
        #tick_label.set_fontsize("30")
        plt.title("Confusion matrix")
        plt.ylabel('True')
        plt.xlabel('Detected')
        plt.savefig(str(Path(self.path, "confusion_matrix.png")))
        plt.clf()
        tn, fp, fn, tp = conf_matrix.ravel()
        print(f"TN: {tn} | FP: {fp} | FN: {fn} | TP: {tp}")
        print(f"Sensitivity: {tp / (fn+tp)}")
        print(f"Specificity: {tn / (tn+fp)}")

        return None


def load(app):
    app.handler.register(PlotMatrixHandler)
