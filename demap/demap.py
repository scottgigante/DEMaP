import graphtools
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr


def DEMaP(data, embedding, knn=30, subsample_idx=None):
    geodesic_dist = geodesic_distance(data, knn=knn)
    if subsample_idx is not None:
        geodesic_dist = geodesic_dist[subsample_idx, :][:, subsample_idx]
    geodesic_dist = squareform(geodesic_dist)
    embedded_dist = pdist(embedding)
    return spearmanr(geodesic_dist, embedded_dist).correlation


def geodesic_distance(data, knn=30, distance='data'):
    G = graphtools.Graph(data, knn=knn, decay=None)
    return G.shortest_path(distance=distance)
