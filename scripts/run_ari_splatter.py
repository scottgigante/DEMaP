import pandas as pd
import numpy as np
import sys
import tasklogger

import demap
from joblib import Parallel, delayed
from functools import partial

N_JOBS = 12


def measure_method(data_noised, method, labels, data_name, subsample_idx=None):
    if subsample_idx is not None:
        data_noised = data_noised[subsample_idx]
    tasklogger.log_start(method.__name__, logger="demap")
    embedding = method(data_noised)
    tasklogger.log_complete(method.__name__, logger="demap")
    ari_score = demap.ari.ARI(labels, embedding, subsample_idx=subsample_idx)
    df = pd.DataFrame(
        {"dataset": data_name, "method": method.__name__, "ARI": ari_score}, index=[""]
    )
    return df


def measure_all_methods(load_fn, n_cells=None, n_jobs=1, load_params=None, seed=None):
    if load_params is None:
        load_params = {}
    if "n_cells" in load_params:
        n_cells = load_params["n_cells"]
        del load_params["n_cells"]
    tasklogger.log_info(
        "Generating noisy data with {}...".format(load_params), logger="demap"
    )
    data_noised, labels = load_fn(return_groups=True, **load_params)
    data_name = load_fn.__name__
    if n_cells is not None:
        subsample_idx = np.random.choice(data_noised.shape[0], n_cells, replace=False)
    else:
        subsample_idx = None
    measure = partial(
        measure_method,
        labels=labels,
        data_noised=data_noised,
        data_name=data_name,
        subsample_idx=subsample_idx,
    )
    if n_jobs == 1:
        results = [measure(method=method) for method in demap.embed.all_methods]
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(measure)(method=method) for method in demap.embed.parallel_methods
        )
        results = results + [
            measure(method=method) for method in demap.embed.non_parallel_methods
        ]
    df = pd.concat(results)
    df = df.sort_values("ARI", ascending=False)
    for key, value in load_params.items():
        df[key] = value
    if n_cells is not None:
        df["n_cells"] = n_cells
    print(df)
    return df


def measure_splat_range(
    load_fn, var_name, var_range, n_jobs=1, seed=None, **load_kwargs
):
    results = pd.concat(
        [
            measure_all_methods(
                load_fn,
                load_params=dict(**{var_name: var_value}, seed=seed, **load_kwargs),
                n_jobs=n_jobs,
                seed=seed,
            )
            for var_value in var_range
        ]
    )
    results.to_csv(
        "../results/ari_s{}_{}_{}_{}_{}.csv".format(
            load_fn.__name__, var_name, var_range.min(), var_range.max(), seed
        )
    )
    results_agg = (
        results.groupby("method")
        .agg({"ARI": [np.mean, np.std]})
        .sort_values(("ARI", "mean"), ascending=False)
    )
    print(results_agg)
    return results_agg


def run_test(seed, n_jobs=1):
    metrics = {
        "dropout": np.array([0] + [0.5] + [0.95]),
        "bcv": np.array([0] + [0.25] + [0.5]),
        "n_cells": np.array([150] + [1500] + [2850]),
        "n_genes": np.array([2000] + [10000] + [17000]),
    }
    metric = list(metrics.keys())[seed % len(metrics)]
    seed = seed // len(metrics)
    datasets = [demap.splatter.paths, demap.splatter.groups]
    dataset = datasets[seed % len(datasets)]
    seed = seed // len(datasets)
    measure_splat_range(dataset, metric, metrics[metric], n_jobs=n_jobs, seed=seed)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_test(int(sys.argv[1]), n_jobs=N_JOBS)
    else:
        for seed in range(400):
            run_test(seed, n_jobs=N_JOBS)
