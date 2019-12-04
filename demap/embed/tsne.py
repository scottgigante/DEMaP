from sklearn import manifold


def TSNE(data, perplexity=30, **kwargs):
    return manifold.TSNE(n_components=2, perplexity=perplexity, **kwargs).fit_transform(
        data
    )
