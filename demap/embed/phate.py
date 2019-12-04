import phate


def PHATE(data, verbose=False, n_jobs=-1, **kwargs):
    return phate.PHATE(verbose=verbose, n_jobs=n_jobs, **kwargs).fit_transform(data)
