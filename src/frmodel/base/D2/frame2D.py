from __future__ import annotations

from typing import List, Tuple, Iterable, Union, Any

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
              _Frame2DScoring,
              np.ndarray):
    """ A Frame is an alias to an Image.

    The underlying representation is a 2D array, each cell is a array of channels
    """

    _labels: dict

    def __new__(cls, ar, labels: str or dict or List[str], *args, **kwargs) -> 'Frame2D':
        obj = np.asarray(ar).view(cls)

        labels = [labels] if isinstance(labels, str) else list(Frame2D._util_flatten(labels))

        assert obj.shape[-1] == len(labels), \
            f"Number of labels ({len(labels)}) must be same as number of Channels ({obj.shape[-1]})."

        if isinstance(labels, Iterable) and not isinstance(labels, dict):
            # Converts list to enumerated dict
            labels = {k: e for e, k in enumerate(labels)}

        obj._labels = labels
        return obj

    def __array_finalize__(self, obj):
        """ Check array after viewing """

        if obj is None: return
        self._labels = getattr(obj, 'labels', None)
        if obj.ndim == 2:
            obj[:] = obj[..., np.newaxis]

        assert obj.ndim == 3, f"Number of dimensions for initialization must be 2 or 3. (Given: {obj.ndim})"

    class CHN(CONSTS.CHN):
        """ This inherits the CONSTS CHN, nothing needs to be added here.

        This makes it really easy to access the CONSTS without having to import it in.

        e.g. f.CHN.XY
        """
        pass

    @staticmethod
    def create(data:np.ndarray, labels: dict or List[str]) -> Frame2D:
        """ This is an init function that allows you to receive the class upon initiation.

        This function will not modify the caller as it's static.

        :param data: A Frame2D acceptable array, see Frame2D wiki for details
        :param labels: Labels for the 2D array, it must be the same length as the data Channel Dimension
        """
        return Frame2D(data, labels)

    @property
    def labels(self) -> dict:
        """ The dictionary of labels.

        :returns: Dictionary, {'NAME'(str): ix(int)} """
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

        :returns: np.ndarray, Shape = (Height * Width, Channels)
        """
        return self.reshape([-1, self.shape[-1]])

    def data_rgb_flatten(self) -> np.ndarray:
        """ Flattens the RGB data by merging all RGB channels

        The algorithm used is
        R + G * 256 + B * 256 * 256.

        This is used to flatten the dimension while keeping all distinct values distinct.

        Note the new dtype is uint32.

        :returns: np.ndarray, Shape = (Height * Width)
        """

        rgb = self.data_rgb().astype(dtype=np.uint32)
        return rgb[..., 0] + rgb[..., 1] * 256 + rgb[..., 2] * (256 ** 2)

    def _labels_to_ix(self, labels: str or List[str]) -> List[int]:
        """ Converts keys to indexes for splicing

        :returns: np.ndarray, [A, B, C, ...]
        """

        if isinstance(labels, str): labels = [labels]
        else: labels = list(self._util_flatten(labels))

        try:
            return [self._labels[label] for label in labels]
        except KeyError:
            raise KeyError(f"Labels {[label for label in labels if label not in self._labels]} not found in the Frame.")

    def data_chn(self, labels: Union[List, CONSTS.CHN, Any]) -> Frame2D:
        """ Gets channels as another Frame2D

        This will still return (Height, Width, 1) even on 1 channel, use squeeze to explicitly remove last dim

        :param labels: Can be a single str or multiple in a List
        :returns: Frame2D, Shape=(Height, Width, Channels)
        """
        return self.create(self[..., self._labels_to_ix(labels)], labels)

    def data_rgb(self) -> Frame2D:
        """ Gets RGB as another Frame2D

        :returns: Frame2D, Shape=(Height, Width, RGB Channels)
        """
        return self.data_chn(CONSTS.CHN.RGB)

    def append(self, ar: np.ndarray, labels: str or Tuple[str]) -> Frame2D:
        """ Appends another channel onto the Frame2D.

        It is compulsory to include channel labels when appending.
        They can be arbitrary, however, the labels are used to easily extract the channel later.

        It is recommended to use the consts provided in consts.py

        :param ar: The array to append to self array. Must be of the same dimensions for the first 2 axes
        :param labels: A list of string labels. Must be the same length as the number of channels to append
        :return: Returns a new Frame2D, Shape = (Height, Width, Channel A + B)
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
        buffer[..., :self.shape[-1]] = self
        buffer[..., self.shape[-1]:] = ar
        self.__getitem__()

        return Frame2D(buffer, [*self._labels, *labels])

    def __getitem__(self, *args, **kwargs) -> 'Frame2D':
        # assert len(args[0]) < 3, "Cannot slice Channel index using numpy slicing."
        arg = list(args[0])
        if arg[0] == Ellipsis:
            arg = [slice(None, None, None), slice(None, None, None), arg[-1]]

        if len(arg) < 3:
            # For this, we pass to numpy to handle XY slicing
            return super(Frame2D, self).__getitem__(*args, **kwargs)
        elif len(arg) == 3:
            # Handle Channel Indexing here
            if isinstance(arg[2], (Iterable, str)):
                # Means, we get a list of indexes to index
                return super(Frame2D, self).__getitem__((*arg[:-1], self._labels_to_ix(arg[2])))
            else:
                raise KeyError(f"Cannot slice the Channel Index, use Indexes. {arg} provided")
        else:
            raise KeyError(f"Too many indexes. {len(arg)} provided.")

    def height(self) -> int:
        return self.shape[0]

    def width(self) -> int:
        return self.shape[1]

    @staticmethod
    def _get_chn_size(chns: Iterable[CONSTS.CHN]) -> int:
        """ This gets the channel size

        That is, if you have [(R, G, B), X, H], this algorithm can flatten it can return 5.
        """
        return len(list(Frame2D._util_flatten(chns)))

    @staticmethod
    def _util_flatten(iterable):
        """ Flattens potentially nested iterables """
        """ Reference: https://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists """
        for el in iterable:
            if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
                yield from Frame2D._util_flatten(el)
            else:
                yield el
