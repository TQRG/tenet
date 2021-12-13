import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_curve

sns.set(style='whitegrid', palette='muted', font_scale=1.5)
rcParams['figure.figsize'] = 14, 8
LABELS = ["Diff", "Static"]


class Plotter:
    def __init__(self, orig, pred, show=True):
        """Plotting class

        # Arguments
            show (bool): optional boolean to show the plots or save
        """
        self.orig = orig
        self.pred = pred
        self.precision, self.recall, self.th = precision_recall_curve(orig, pred)
        self.show = show

    def plt_show(self, plt_type):
        # Shows plot or saves it in the plots folder
        if not self.show:
            plt.savefig(f'./compare/{plt_type}.png')
            plt.clf()
        else:
            plt.show()

    def plot_roc(self):
        # Plots and calculates the ROC
        fpr, tpr, thresholds = roc_curve(self.orig, self.pred)
        roc_auc = auc(fpr, tpr)

        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, label='AUC = %0.4f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([-0.001, 1])
        plt.ylim([0, 1.001])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        self.plt_show("roc")

    def plot_precision_recall(self):
        plt.plot(self.recall, self.precision, 'b', label='Precision-Recall curve')
        plt.title('Recall vs Precision')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        self.plt_show("precision_recall")

    def plot_prediction(self):
        groups = self.error_df.groupby('true_class')
        fig, ax = plt.subplots()

        for name, group in groups:
            ax.plot(group.index, group.reconstruction_error, marker='o', ms=3.5, linestyle='',
                    label="Unsafe" if name == 1 else "Safe")

        plt.title("Reconstruction error for different classes")
        plt.ylabel("Reconstruction error")
        plt.xlabel("Data point index")
        self.plt_show("prediction")

    def plot_confusion_matrix(self):
        conf_matrix = confusion_matrix(self.orig, self.pred)
        plt.figure(figsize=(12, 12))
        akws = {"ha": 'center', "va": 'center'}
        sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, annot_kws=akws, fmt="d")
        plt.title("Confusion matrix")
        plt.ylabel('True')
        plt.xlabel('Detected')
        self.plt_show("confusion_matrix")
        tn, fp, fn, tp = conf_matrix.ravel()
        print(tn, fp, fn, tp)
        print(f"Sensitivity: {tp / (tn+tp)}")
        print(f"Specificity: {tn / (tn+fp)}")
