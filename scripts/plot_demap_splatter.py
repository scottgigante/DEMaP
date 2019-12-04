import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt

metric_titles = {
    "bcv": "Biological Variation",
    "dropout": "Dropout",
    "n_cells": "Subsampled cells",
    "n_genes": "Subsampled genes",
}
metric_xlabels = {
    "bcv": "Coefficient of Variation",
    "dropout": "Dropout probability",
    "n_cells": "# of cells retained",
    "n_genes": "# of genes retained",
}
dataset_ylabels = {"groups": "Splatter Groups", "paths": "Splatter Paths"}
method_colors = {"PHATE": "k", "MDS_on_DM": "tab:grey", "TSNE_on_DM": "tab:orange"}


def map_color(method):
    try:
        return method_colors[method]
    except KeyError:
        return None


def load_all(dataset="paths", metric="bcv"):
    df = pd.concat(
        [
            pd.read_csv(filename, index_col=0)
            for filename in glob.glob("../results/{}_{}_*.csv".format(dataset, metric))
        ],
        sort=True,
    )
    df = df.set_index(np.arange(df.shape[0]))
    return df


def summarize_one(dataset="paths", metric="bcv"):
    df = load_all(dataset, metric)
    del df["seed"]
    df.columns = ["value" if c == metric else c for c in df.columns]
    df = (
        df.groupby(["method", "value"])
        .agg({"demap": [np.mean, np.std]})
        .sort_values(["value", ("demap", "mean")], ascending=False)
    )
    score = df[("demap", "mean")].values
    std = df[("demap", "std")].values
    method, value = list(zip(*df.index.values.tolist()))
    method = np.array(method)
    value = np.array(value)
    return pd.DataFrame(
        dict(
            method=method,
            value=value,
            mean=score,
            std=std,
            metric=metric,
            dataset=dataset,
        )
    )


def summarize_all():
    result = pd.concat(
        [
            summarize_one("paths", "dropout"),
            summarize_one("groups", "dropout"),
            summarize_one("paths", "bcv"),
            summarize_one("groups", "bcv"),
            summarize_one("paths", "n_cells"),
            summarize_one("groups", "n_cells"),
            summarize_one("paths", "n_genes"),
            summarize_one("groups", "n_genes"),
        ],
        sort=True,
    )
    return result


def plot_results(result, methods, figsize=(16, 8)):
    plt.rc("font", size=18)
    fig, axes = plt.subplots(2, 4, sharey="all", sharex="col", figsize=figsize)
    ax_iter = iter(axes.flatten().tolist())

    for j, metric in enumerate(["dropout", "bcv", "n_cells", "n_genes"]):
        axes[1, j].set_xlabel(metric_xlabels[metric], fontsize="large")
        axes[0, j].set_title(metric_titles[metric])

    for dataset in ["paths", "groups"]:
        for metric in ["dropout", "bcv", "n_cells", "n_genes"]:
            ax = next(ax_iter)
            lines = []
            for method in methods:
                plot_idx = (
                    (result["method"] == method)
                    & (result["dataset"] == dataset)
                    & (result["metric"] == metric)
                )
                x = result.loc[plot_idx]["value"]
                if metric == "n_cells":
                    x = (x.values * 3000).astype(int)
                lines.append(
                    ax.plot(
                        x,
                        result.loc[plot_idx]["mean"],
                        label=method,
                        linewidth=3 if method == "PHATE" else 1,
                        c=map_color(method),
                        linestyle="--" if "_on_" in method else "-",
                    )[0]
                )
            ax.set_xticks(x)
            ax.set_yticks(np.linspace(0, 0.8, 5))
            ax.set_ylim((ax.get_ylim()[0], 0.8))

    for i, dataset in enumerate(["paths", "groups"]):
        axes[i, 0].set_ylabel(dataset_ylabels[dataset], fontsize="x-large")
        twinax = axes[i, -1].twinx()
        # twinax.set_ylabel("Spearman Correlation",
        #                   fontsize='large', rotation=270, labelpad=20)
        twinax.set_yticks(axes[i, 0].get_yticks())
        twinax.set_ylim(axes[i, 0].get_ylim())

    return fig, axes, lines


def plot_all():
    result = summarize_all()
    methods = (
        result.groupby("method")
        .agg({"mean": np.mean})
        .sort_values("mean", ascending=False)
        .index.values
    )

    fig, axes, lines = plot_results(result, methods)
    fig.tight_layout()
    fig.savefig("../output/plot.svg")
    fig.savefig("../output/plot.png")

    plt.rc("font", size=14)
    legend_fig, legend_ax = plt.subplots()
    legend_ax.legend(lines, [m.replace("_", " ") for m in methods])
    legend_ax.xaxis.set_visible(False)
    legend_ax.yaxis.set_visible(False)
    legend_fig.tight_layout()
    legend_fig.savefig("../output/legend.svg")
    legend_fig.savefig("../output/legend.png")

    plt.rc("font", size=11)
    fig.set_figwidth(18)
    axes[0, -1].legend(
        lines, [m.replace("_", " ") for m in methods], bbox_to_anchor=[1.2, 1.02]
    )
    fig.tight_layout()
    fig.savefig("../output/performance.svg")
    fig.savefig("../output/performance.png")


if __name__ == "__main__":
    plot_all()
