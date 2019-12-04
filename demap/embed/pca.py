from sklearn import decomposition


def PCA(data, n_components=2, **kwargs):
    return decomposition.PCA(n_components=n_components).fit_transform(data)
