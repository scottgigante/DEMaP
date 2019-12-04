from sklearn import manifold


def MDS(data, n_jobs=-1, **kwargs):
    return manifold.MDS(n_components=2, n_jobs=n_jobs,
                        **kwargs).fit_transform(data)
