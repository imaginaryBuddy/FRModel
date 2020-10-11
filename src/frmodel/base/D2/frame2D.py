from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PIL import Image
from sklearn.neighbors import KDTree

# noinspection PyProtectedMember
from frmodel.base.D2.frame._frame_channel import _Frame2DChannel
# noinspection PyProtectedMember
from frmodel.base.D2.frame._frame_kmeans import _Frame2DKmeans
# noinspection PyProtectedMember
from frmodel.base.D2.frame._frame_loader import _Frame2DLoader
# noinspection PyProtectedMember
from frmodel.base.D2.frame._frame_partition import _Frame2DPartition
# noinspection PyProtectedMember
from frmodel.base.D2.frame._frame_scaling import _Frame2DScaling
from frmodel.base.consts import CONSTS

CHANNEL = CONSTS.CHANNEL

MAX_RGB = 255

@dataclass
class Frame2D(_Frame2DLoader,
              _Frame2DKmeans,
              _Frame2DPartition,
              _Frame2DChannel,
              _Frame2DScaling):
    """ A Frame is an alias to an Image.

    The underlying representation is a 2D array, each cell is a array of channels
    """

    data: np.ndarray

    def save(self, file_path: str, **kwargs) -> None:
        """ Saves the current Frame file """
        Image.fromarray(self.data_rgb().astype(np.uint8)).save(file_path, **kwargs)

    def data_kdtree(self, leaf_size=40, metric='minkowski', **kwargs) -> KDTree:
        """ Constructs a KDTree with current data.

        Uses sklearn.neighbours.KDTree API."""
        return KDTree(self.data_flatten(),
                      leaf_size=leaf_size,
                      metric=metric,
                      **kwargs)

    def data_flatten(self) -> np.ndarray:
        return self.data.reshape([-1, self.shape[-1]])

    def data_chn(self, idx: int or List[int]) -> np.ndarray:
        """ Gets channels as pure np.ndarray data

        :param idx: Can be a single int index or multiple in a List"""
        if isinstance(idx, int): idx = [idx]
        return self.data[..., idx]

    def data_rgb(self) -> np.ndarray:
        return self.data[..., [CHANNEL.RED, CHANNEL.GREEN, CHANNEL.BLUE]]

    def size(self) -> np.ndarray:
        """ Returns the number of pixels """
        return self.data.size

    @property
    def shape(self) -> Tuple:
        return self.data.shape

    def height(self) -> int:
        return self.shape[0]

    def width(self) -> int:
        return self.shape[1]