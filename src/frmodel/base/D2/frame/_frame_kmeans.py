from abc import ABC, abstractmethod
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from frmodel.base.D2.kmeans2D import KMeans2D


class _Frame2DKmeans(ABC):

    @abstractmethod
    def data_flatten(self): ...

    def kmeans(self,
               model: KMeans,
               fit_indexes,
               sample_weight=None,
               scaler=normalize) -> KMeans2D:
        """ Creates a KMeans Object from current data

        :param model: KMeans Model
        :param fit_indexes: The indexes to .fit()
        :param sample_weight: The sample weight for each record, if any. Can be None.
        :param scaler: The scaler to use, must be a callable(np.ndarray)
        :returns: KMeans2D Instance

        """
        frame_xy_trans = scaler(self.data_flatten()[:, fit_indexes])
        fit = model.fit(frame_xy_trans,
                        sample_weight=frame_xy_trans[:, sample_weight] if np.all(sample_weight) else None)

        return KMeans2D(fit, self.data_flatten())

