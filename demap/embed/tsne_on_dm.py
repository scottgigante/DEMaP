from .tsne import TSNE
from .dm import DM


def TSNE_on_DM(data, perplexity=30, **kwargs):
    return TSNE(DM(data, **kwargs), perplexity=perplexity)
