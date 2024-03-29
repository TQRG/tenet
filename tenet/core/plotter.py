import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go

from pathlib import Path
from typing import Tuple, Any


class Plotter:
    def __init__(self, path: Path, style: str = 'darkgrid', context: Any = 'poster', font_scale: float = 0.9,
                 seaborn_presets: dict = {"grid.linewidth": 0.8}, fig_size: Tuple[int, int] = (12, 8)):
        self.path = path
        sns.set_style(style)
        sns.set_context(context, font_scale=font_scale, rc=seaborn_presets)
        plt.rcParams["figure.figsize"] = fig_size
        self.linestyles = ['-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted']

    def histogram_pairs(self, df: pd.DataFrame, column: str, bins: int = 20, x_label: str = None,
                        y_label: str = "Occurrences", title: str = None, filter_outliers: bool = False):

        labels = list(df.label.value_counts().keys())

        if not x_label:
            x_label = column

        if not title:
            title = f"Histogram of {','.join(labels)} {x_label}"

        if filter_outliers:
            df = df[df[column].between(df[column].quantile(.1), df[column].quantile(.9))]  # without outliers
            print(f"#samples: {len(df)}")

        series = [df[df['label'] == l][column] for l in labels]
        colors = list(mcolors.TABLEAU_COLORS.keys())[:len(labels)]
        linestyles = self.linestyles[:len(labels)]
        plt.hist([s.to_list() for s in series], bins, alpha=0.5, label=labels, color=colors)

        for s, l, c, ls in zip(series, labels, reversed(colors), linestyles):
            plt.axvline(x=s.mean(), color=c, alpha=0.5, linestyle=ls, label=f"{l} mean")

        plt.legend(loc='upper right')
        plt.title(title)
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.savefig(str(self.path / f"{x_label.lower().replace(' ', '_')}_histogram.png"))
        plt.clf()

    def histogram_columns(self, df: pd.DataFrame, columns: list, x_label: str, y_label: str, labels: list,
                          bins: int = 20, title: str = None, filter_outliers: bool = False, mean: bool = False):

        if filter_outliers:
            # without outliers
            rows = set()

            for c in columns:
                no_outliers = df[c].between(df[c].quantile(.1), df[c].quantile(.9))
                idxs = no_outliers[no_outliers].index.to_list()

                if rows is None:
                    rows = idxs
                else:
                    rows = rows.intersection(set(idxs))

            df = df.isin(list(rows))
            print(f"#samples: {len(df)}")

        if not title:
            title = f"Histogram of {','.join(labels)} {x_label}"

        series = [df[c] for c in columns]
        colors = list(mcolors.TABLEAU_COLORS.keys())[:len(labels)]
        linestyles = self.linestyles[:len(labels)]
        plt.hist([s.to_list() for s in series], bins, alpha=0.5, label=labels, color=colors)

        if mean:
            for s, l, c, ls in zip(series, labels, reversed(colors), linestyles):
                plt.axvline(x=s.mean(), color=c, alpha=0.5, linestyle=ls, label=f"{l} mean")

        plt.legend(loc='upper right')
        plt.title(title)
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.savefig(str(self.path / f"{x_label.lower().replace(' ', '_')}_histogram.png"))
        plt.clf()

    def bar_labels(self, df: pd.DataFrame, column: str, x_label: str, y_label: str, title: str = None,
                   bar_value_label: bool = True, rotate_labels: bool = True, top_n: int = None):
        if not title:
            title = f"Bar plot of {x_label}"

        if top_n:
            labels, counts = zip(*df[column].value_counts().head(top_n).items())
        else:
            labels, counts = zip(*df[column].value_counts().items())

        colors = list(mcolors.TABLEAU_COLORS.keys())[:len(counts)]
        fig, ax = plt.subplots(1, 1)
        ax.bar(labels, counts, color=colors)

        if rotate_labels:
            # aligned rotation on x-axis
            ax.set_xticklabels(labels, rotation=40, ha='right', rotation_mode='anchor')
        else:
            plt.xticks(labels)

        if bar_value_label:
            # add values on each bar
            for x_tick, height, color in zip(ax.get_xticks(), counts, colors):
                ax.text(x_tick - .25, height + 5, str(height), color=color)

        plt.legend(loc='upper right')
        plt.title(title)
        plt.tight_layout()
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.savefig(str(self.path / f"{x_label.lower().replace(' ', '_')}_bar.png"))
        plt.clf()

    def sankey(self, sources: list, targets: list, values: list, node_labels: list, link_labels: list, title: str,
               node_colors: list = None, link_colors: list = None, color_map_name: str = 'hsv', opacity: float = 1):

        if not node_colors:
            node_colors = []
            color_map = plt.cm.get_cmap(color_map_name)

            for i in range(len(node_labels)):
                rgb = color_map((i + 1) / len(node_labels))
                node_colors.append(f"rgba({rgb[0] * 255},{rgb[1] * 255},{rgb[2] * 255}, {opacity})")

        if not link_colors:
            link_colors = [node_colors[i] for i in sources]

        fig = go.Figure(data=[go.Sankey(
            valueformat=".0f",
            valuesuffix=" entries",
            # Define nodes
            node=dict(
                pad=15,
                thickness=15,
                line=dict(color="black", width=0.5),
                label=node_labels,
                color=node_colors
            ),
            # Add links
            link=dict(
                source=sources,
                target=targets,
                value=values,
                label=link_labels,
                color=link_colors
            ))])
        fig.layout.width = 1920
        fig.layout.height = 1080

        fig.update_layout(title_text=title, font_size=16)
        fig.write_image(str(self.path / f"{title.lower().replace(' ', '_')}_sankey.png"))
