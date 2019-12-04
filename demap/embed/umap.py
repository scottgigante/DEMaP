import scprep

umap = scprep._lazyload.AliasModule("umap", [])

@scprep.utils._with_pkg(pkg="umap")
def UMAP(data, **kwargs):
    return umap.UMAP(**kwargs).fit_transform(data)
