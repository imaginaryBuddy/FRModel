from dataclasses import dataclass
import matplotlib.pyplot as plt
from seaborn import FacetGrid

from sklearn.cluster import KMeans


@dataclass
class KMeans2D:

    model: KMeans
    plot: FacetGrid = None



