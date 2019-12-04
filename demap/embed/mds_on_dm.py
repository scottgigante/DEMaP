from .mds import MDS
from .dm import DM


def MDS_on_DM(data, **kwargs):
    return MDS(DM(data, **kwargs))
