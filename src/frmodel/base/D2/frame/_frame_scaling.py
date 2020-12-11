from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from sklearn.preprocessing import minmax_scale as sk_minmax_scale
from sklearn.preprocessing import normalize as sk_normalize

if TYPE_CHECKING:
    from frmodel.base.D2.frame2D import Frame2D

MAX_RGB = 255

class _Frame2DScaling(ABC):

    data: np.ndarray

    @abstractmethod
    def data_flatten_xy(self, *args, **kwargs): ...

    @abstractmethod
    def width(self): ...

    @abstractmethod
    def height(self): ...

    def normalize(self, **kwargs) -> _Frame2DScaling:
        return self.scale(sk_normalize, **kwargs)

    def minmax_scale(self, **kwargs) -> _Frame2DScaling:
        return self.scale(sk_minmax_scale, **kwargs)

    def scale(self, scaler, **scaler_kwargs):
        shape = self.data.shape
        self: 'Frame2D'
        return self.create(data=scaler(self.data_flatten_xy(), **scaler_kwargs).reshape(shape),
                           labels=self.labels)
