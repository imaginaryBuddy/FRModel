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
                 frame_1dmask: np.ndarray = None,
                 scaler=None):
        """ Creates a KMeans Object from current data

        :param model: KMeans Model
        :param fit_indexes: The indexes to .fit()
        :param scaler: The scaler to use, must be a callable(np.ndarray)
        :param frame_1dmask: The 2D mask to exclude certain points. Must be in 2 Dimensions
        :returns: KMeans2D Instance

        """
        data = frame.data_flatten_xy()[..., fit_indexes]

        if frame_1dmask is not None:
            assert frame_1dmask.shape == data.shape[:-1],\
                f"The Frame 2D Mask must match the size of first 2 Dimensions of frame exactly." \
                f"(Mask Shape: {frame_1dmask.shape}, Frame 2D Shape: {data.shape[:-1]})"
            data = data[frame_1dmask, ...]
        if scaler:
            data = scaler(data)
        fit = model.fit(data)
        self.model = fit
        self.frame = frame
        self.frame_1dmask = frame_1dmask

    def frame_masked(self, with_xy: bool = True):
        return self.frame.get_chns(self_=True, chns=[Frame2D.CHN.XY])\
                   .data_flatten_xy()[self.frame_1dmask, :]

    def as_frame(self) -> Frame2D:
        """ Converts current model into Frame2D based on labels. Places label at the end of channel dimension

        :return: Frame2D
        """

        return self.frame.append(self.model.labels_.reshape(self.frame.shape[0:2]), CONSTS.CHN.KMEANS.LABEL)

