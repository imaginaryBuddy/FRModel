from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sklearn.neighbors import KDTree

# noinspection PyProtectedMember
from frmodel.base.D2.frame._frame_channel import _Frame2DChannel
# noinspection PyProtectedMember
from frmodel.base.D2.frame._frame_image import _Frame2DImage
# noinspection PyProtectedMember
from frmodel.base.D2.frame._frame_loader import _Frame2DLoader
# noinspection PyProtectedMember
from frmodel.base.D2.frame._frame_partition import _Frame2DPartition
# noinspection PyProtectedMember
from frmodel.base.D2.frame._frame_plot import _Frame2DPlot
# noinspection PyProtectedMember
from frmodel.base.D2.frame._frame_scaling import _Frame2DScaling
# noinspection PyProtectedMember
from frmodel.base.D2.frame._frame_scoring import _Frame2DScoring
from frmodel.base.consts import CONSTS

CHANNEL = CONSTS.CHANNEL

MAX_RGB = 255

@dataclass
class Frame2D(_Frame2DLoader,
              _Frame2DPartition,
              _Frame2DChannel,
              _Frame2DScaling,
              _Frame2DImage,
              _Frame2DPlot,
              _Frame2DScoring):
    """ A Frame is an alias to an Image.

    The underlying representation is a 2D array, each cell is a array of channels
    """

    data: np.ndarray

    def data_kdtree(self, leaf_size=40, metric='minkowski', **kwargs) -> KDTree:
        """ Constructs a KDTree with current data.

        Uses sklearn.neighbours.KDTree API."""
        return KDTree(self.data_flatten_xy(),
                      leaf_size=leaf_size,
                      metric=metric,
                      **kwargs)

    def data_flatten_xy(self) -> np.ndarray:
        """ Flattens the data on XY only. """
        return self.data.reshape([-1, self.shape[-1]])

    def data_chn(self, idx: int or List[int]) -> np.ndarray:
        """ Gets channels as pure np.ndarray data

        :param idx: Can be a single int index or multiple in a List"""
        if isinstance(idx, int): idx = [idx]
        return self.data[..., idx]

    def data_rgb(self) -> np.ndarray:
        return self.data[..., [CHANNEL.RED, CHANNEL.GREEN, CHANNEL.BLUE]]

    def append(self, ar: np.ndarray):
        shape_ar = ar.shape
        shape_self = self.shape

        assert shape_ar[0] == shape_self[0], f"Mismatch Axis 0, Target: {shape_ar[0]}, Self: {shape_self[0]}"
        assert shape_ar[1] == shape_self[1], f"Mismatch Axis 0, Target: {shape_ar[1]}, Self: {shape_self[1]}"

        if ar.ndim == 2:
            ar = ar[..., np.newaxis]
            shape_ar = ar.shape  # Update shape if ndim is 2

        buffer = np.zeros((*shape_self[0:2], shape_ar[-1] + shape_self[-1]), self.dtype)
        buffer[..., :self.shape[-1]] = self.data
        buffer[..., self.shape[-1]:] = ar

        return Frame2D(buffer)

    def size(self) -> np.ndarray:
        """ Returns the number of pixels """
        return self.data.size

    @property
    def shape(self) -> Tuple:
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    def height(self) -> int:
        return self.shape[0]

    def width(self) -> int:
        return self.shape[1]

    # def data_cat_labels(self):
    #     """ Categorizes unique values in the frame """
    #     data = self.data.astype(np.uint)
    #     flat = data[..., 0] + data[..., 1] * 255 + data[..., 2] * 255 * 255
    #     return (rankdata(flat.flatten(), method='dense') - 1).reshape(self.data.shape[:-1])
