from sklearn import preprocessing

from scprep.run.r_function import RFunction

_Monocle2 = RFunction(
    setup="library(monocle)",
    args="data",
    body="""
    data <- t(data)
    colnames(data) <- 1:ncol(data)
    rownames(data) <- 1:nrow(data)
    fd <- new("AnnotatedDataFrame", data = as.data.frame(rownames(data)))
    data <- newCellDataSet(data,phenoData=NULL,featureData = fd,
                         expressionFamily=uninormal())
    varLabels(data@featureData) <- 'gene_short_name'
    data <- estimateSizeFactors(data)
    data_reduced <- suppressMessages(suppressWarnings(reduceDimension(data,
                                   max_components=2,
                                   reduction_method='DDRTree',
                                   norm_method="none", scaling=FALSE)))

    # 2D embedding
    cell_embedding = reducedDimS(data_reduced)
    t(cell_embedding)""")


def Monocle2(data):
    """Short function descriptor

    Long function descriptor

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data from both samples

    Returns
    -------
    embedding : data type
        Description

    Examples
    --------
    >>> import scprep
    >>> data = scprep.io.load_csv("my_data.csv")
    >>> data_ln = scprep.normalize.library_size_normalize(data)
    >>> data_log = scprep.transform.log(data)
    >>> results = scprep.run.Monocle(data_log)
    """
    data = preprocessing.scale(data)
    return _Monocle2(data)
