import pandas as pd
import numpy as np
import glob


def tofloat(val):
    return float(val.split(" ")[0])


def unique_inorder(array):
    uniq, index = np.unique(array, return_index=True)
    return uniq[index.argsort()]


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
        .sort_values(("demap", "mean"), ascending=False)
    )
    score = np.array(
        [
            "{:.2f} +- {:.2f}".format(m, s)
            for m, s in zip(
                np.round(df[("demap", "mean")], 2), np.round(df[("demap", "std")], 2)
            )
        ]
    )
    method, value = list(zip(*df.index.values.tolist()))
    method = np.array(method)
    value = np.array(value)
    if metric == "bcv":
        metric = "BCV"
    result = []
    for v in unique_inorder(value):
        idx = value == v
        val_result = pd.DataFrame(
            [score[idx]],
            columns=method[idx],
            index=["{}, {} = {}".format(dataset, metric, v)],
        )
        result.append(val_result)
    return pd.concat(result, sort=True)


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

    means = np.vectorize(tofloat)(result.values)
    result = result.iloc[:, np.argsort(means.mean(0))[::-1]]
    return result


def to_latex(result, filename):
    means = np.vectorize(tofloat)(result.values)
    for i in range(result.shape[0]):
        for j in np.argwhere(means[i] == np.max(means[i])).flatten():
            result.iloc[i, j] = "mathbf{" + result.iloc[i, j] + "}"
        for j in range(result.shape[1]):
            result.iloc[i, j] = "$" + result.iloc[i, j] + "$"
    result.columns = [c.replace("_", " ") for c in result.columns]
    result.index = [i.replace("bcv", "BCV") for i in result.index]
    with open(filename, "w") as handle:
        handle.write(
            result.to_latex()
            .replace("\\$", "$")
            .replace("\\{", "{")
            .replace("\\}", "}")
            .replace("mathbf", "\\mathbf")
            .replace("+-", "\\pm")
            .replace("l" * (result.shape[1] + 1), "l" + "r" * result.shape[1])
        )


if __name__ == "__main__":
    result = summarize_all()
    print(result)
    to_latex(result, "../output/result.tex")
