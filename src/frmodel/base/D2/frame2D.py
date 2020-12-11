from __future__ import annotations

from copy import deepcopy
from typing import List, Tuple, Iterable

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

MAX_RGB = 255

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

    _data: np.ndarray
    _labels: dict

    def __init__(self, data: np.ndarray, labels: str or dict or List[str]):
        self._data = data
        labels = [labels] if isinstance(labels, str) else labels

        assert data.ndim == 3, f"Number of dimensions for initialization must be 3. (Given: {data.ndim})"
        assert data.shape[-1] == len(labels),\
            f"Number of labels ({len(labels)}) must be same as number of Channels ({data.shape[-1]})."

        if isinstance(labels, Iterable) and not isinstance(labels, dict):
            # Converts list to enumerated dict
            labels = {k: e for e, k in enumerate(labels)}

        self._labels = labels

    @staticmethod
    def create(data:np.ndarray, labels: dict or List[str]) -> Frame2D:
        """ This is an init function that allows you to receive the class upon initiation.

        This function will not modify the caller as it's static. """
        return Frame2D(data, labels)

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def labels(self) -> dict:
        return self._labels

    def data_kdtree(self, leaf_size=40, metric='minkowski', **kwargs) -> KDTree:
        """ Constructs a KDTree with current data.

        Uses sklearn.neighbours.KDTree API."""
        return KDTree(self.data_flatten_xy(),
                      leaf_size=leaf_size,
                      metric=metric,
                      **kwargs)

    def data_flatten_xy(self) -> np.ndarray:
        """ Flattens the data on XY only.

        This means that the first 2 dimensions will be merged together.
        """
        return self.data.reshape([-1, self.shape[-1]])

    def data_rgb_flatten(self) -> np.ndarray:
        """ Flattens the RGB data by merging all RGB channels

        The algorithm used is
        R + G * 256 + B * 256 * 256.

        This is used to flatten the dimension while keeping all distinct values distinct.

        Note the new dtype is uint32.
        """

        rgb = self.data_rgb().astype(dtype=np.uint32)
        return rgb[..., 0] + rgb[..., 1] * 256 + rgb[..., 2] * (256 ** 2)

    def _labels_to_ix(self, labels: str or List[str]):
        """ Converts keys to indexes for splicing """

        try:
            return self._labels[labels] if isinstance(labels, str) else [self._labels[label] for label in labels]
        except KeyError:
            raise KeyError(f"Labels {[label for label in labels if label not in self._labels]} not found in the Frame.")

    def data_chn(self, labels: str or List[str]) -> np.ndarray:
        """ Gets channels as pure np.ndarray data

        :param labels: Can be a single str or multiple in a List"""
        return self.data[..., self._labels_to_ix(labels)]

    def data_rgb(self) -> np.ndarray:
        return self.data_chn(CONSTS.CHN.RGB)

    def append(self, ar: np.ndarray, labels: str or Tuple[str]) -> Frame2D:
        """ Appends another channel onto the Frame2D.

        It is compulsory to include channel labels when appending.
        They can be arbitrary, however, the labels are used to easily extract the channel later.

        It is recommended to use the consts provided in consts.py

        :param ar: The array to append to self array. Must be of the same dimensions for the first 2 axes
        :param labels: A list of string labels. Must be the same length as the number of channels to append
        :return: Returns a new Frame2D.
        """
        ar_shape = ar.shape
        self_shape = self.shape

        labels = [labels] if isinstance(labels, str) else labels

        if ar.ndim == 2:
            ar = ar[..., np.newaxis]
            ar_shape = ar.shape  # Update shape if ndim is 2

        assert ar_shape[0] == self_shape[0], f"Mismatch Axis 0, Target: {ar_shape[0]}, Self: {self_shape[0]}"
        assert ar_shape[1] == self_shape[1], f"Mismatch Axis 1, Target: {ar_shape[1]}, Self: {self_shape[1]}"
        assert len(labels) == ar_shape[-1], f"Mismatch Label Length, Target: {ar_shape[-1]}, Labels: {len(labels)}"

        buffer = np.zeros((*self_shape[0:2], ar_shape[-1] + self_shape[-1]), self.dtype)
        buffer[..., :self.shape[-1]] = self.data
        buffer[..., self.shape[-1]:] = ar

        return Frame2D(buffer, [*self._labels, *labels])

    def size(self) -> int:
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
