from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from frmodel.base import CONSTS
from frmodel.base.D2 import Frame2D


class KMeans2D:

    def __init__(self,
                 frame: Frame2D,
                 model: KMeans,
                 fit_indexes,
                 sample_weight=None,
                 scaler=None):
        """ Creates a KMeans Object from current data

        :param model: KMeans Model
        :param fit_indexes: The indexes to .fit()
        :param sample_weight: The sample weight for each record, if any. Can be None.
        :param scaler: The scaler to use, must be a callable(np.ndarray)
        :returns: KMeans2D Instance

        """
        data = frame.data_flatten_xy()[..., fit_indexes]
        if scaler:
            data = scaler(data)
        fit = model.fit(data,
                        sample_weight=np.abs(data[:, sample_weight]) if np.all(sample_weight) else None)
        self.model = fit
        self.frame = frame

    def as_frame(self) -> Frame2D:
        """ Converts current model into Frame2D based on labels. Places label at the end of channel dimension

        :return: Frame2D
        """

        return self.frame.append(self.model.labels_.reshape(self.frame.shape[0:2]), CONSTS.CHN.KMEANS.LABEL)

