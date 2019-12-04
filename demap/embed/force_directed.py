import graphtools
import numpy as np
import scprep

networkx = scprep._lazyload.AliasModule("networkx", ["drawing"])


@scprep.utils._with_pkg("networkx")
def Force_Directed_Layout(data, knn=5, decay=40):
    # symmetric affinity matrix
    K = graphtools.Graph(data, n_jobs=-1, knn=knn, decay=decay).kernel
    graph = networkx.from_numpy_matrix(K.toarray())
    embedding = networkx.drawing.layout.spring_layout(graph)
    return np.vstack([embedding[i] for i in range(K.shape[0])])
