from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

import numpy as np
from sklearn.preprocessing import minmax_scale as sk_minmax_scale, minmax_scale
from sklearn.preprocessing import normalize as sk_normalize

if TYPE_CHECKING:
    from frmodel.base.D2.frame2D import Frame2D

class _Frame2DScaling(ABC):

    def normalize(self: 'Frame2D', **kwargs) -> 'Frame2D':
        return self.scale(sk_normalize, **kwargs)

    def minmax_scale(self: 'Frame2D', **kwargs) -> 'Frame2D':
        return self.scale(sk_minmax_scale, **kwargs)

    def scale(self: 'Frame2D', scaler, **scaler_kwargs) -> 'Frame2D':
        shape = self.data.shape
        return self.create(data=scaler(self.data_flatten_xy(), **scaler_kwargs).reshape(shape),
                           labels=self.labels)

    def scale_values(self: 'Frame2D',
                     from_min=None, from_max=None,
                     to_min=0, to_max=2**8-1) -> 'Frame2D':
        """ Linearly Scales data to another range. Modifies itself.

        :param from_min: The minimum, if none, data minimum will be used
        :param from_max: The maximum, if none, data maximum will be used
        :param to_min: The minimum to scale to (default 0)
        :param to_max: The maximum to scale to (default 255)
        :return: self, the data will be modified internally regardless
        """
        from_min = self.data.min() if from_min is None else from_min
        from_max = self.data.max() if from_max is None else from_max
        return self.create(data=(self.data - from_min) / (from_max - from_min) * (to_max - to_min) + to_min,
                           labels=self.labels)

    def scale_values_independent(self: 'Frame2D',
                                 from_min=None, from_max=None,
                                 to_min=0, to_max=2**8-1) -> 'Frame2D':
        """ Min Max Scales each channel to another range

        :param from_min: The minimum of each channel, if none, data minimum will be used
        :param from_max: The maximum of each channel, if none, data maximum will be used
        :param to_min: The minimum to scale to (default 0)
        :param to_max: The maximum to scale to (default 255)
        :return: self, the data will be modified internally regardless
        """
        if from_min is None or from_max is None:
            flat = self.data.reshape([-1, self.shape[-1]])
            # noinspection PyTypeChecker
            from_min = np.min(flat, axis=0)
            # noinspection PyTypeChecker
            from_max = np.max(flat, axis=0)
        else:
            from_min = np.asarray(from_min)
            from_max = np.asarray(from_max)

        return self.create(data=(self.data - from_min) / (from_max - from_min) * (to_max - to_min) + to_min,
                           labels=self.labels)

    def scale_values_on_band(self: 'Frame2D', to_min: int, to_max: int):
        return self.create(data=minmax_scale(self.data.reshape([-1, self.shape[-1]]), axis=-1)
                           .reshape(self.shape) * (to_max - to_min),
                           labels=self.labels)
