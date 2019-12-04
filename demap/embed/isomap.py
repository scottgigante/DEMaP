from sklearn import manifold


def Isomap(data, n_jobs=-1, **kwargs):
    return manifold.Isomap(n_components=2, n_jobs=n_jobs, **kwargs).fit_transform(data)
