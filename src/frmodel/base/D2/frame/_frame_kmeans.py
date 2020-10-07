from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize as sk_normalize

from frmodel.base.D2.kmeans2D import KMeans2D


class _Frame2DKmeans(ABC):

    @abstractmethod
    def data_flatten(self): ...

    def kmeans(self, model: KMeans) -> KMeans2D:
        """ Creates a KMeans Object from current data

        :param model: KMeans Model
        :returns: KMeans2D Instance
        """
        return KMeans2D(model, self.data_flatten())

