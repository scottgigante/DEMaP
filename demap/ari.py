import numpy as np
import sklearn.cluster
import sklearn.metrics


def ARI(labels, embedding, subsample_idx=None, n_rep=10):
    if subsample_idx is not None:
        labels = labels[subsample_idx]
    n_clusters = len(np.unique(labels))
    results = []
    for _ in range(n_rep):
        clusters = sklearn.cluster.KMeans(n_clusters).fit_predict(embedding)
        results.append(sklearn.metrics.adjusted_rand_score(labels, clusters))
    return np.mean(results)
