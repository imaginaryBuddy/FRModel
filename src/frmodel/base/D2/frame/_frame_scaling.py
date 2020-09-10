from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sklearn.preprocessing import minmax_scale as sk_minmax_scale
from sklearn.preprocessing import normalize as sk_normalize

from frmodel.base.consts import CONSTS

CHANNEL = CONSTS.CHANNEL
MAX_RGB = 255

class _Frame2DScaling(ABC):

    data: np.ndarray
    
    @abstractmethod
    def data_rgb(self): ...
    
    @abstractmethod
    def data_chn(self, *args, **kwargs): ...

    @abstractmethod
    def width(self): ...

    @abstractmethod
    def height(self): ...
    
    @abstractmethod
    def slide_xy(self, *args, **kwargs): ...

    # noinspection PyArgumentList
    @classmethod
    def init(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    def normalize(self) -> _Frame2DScaling:
        shape = self.data.shape
        return self.init(sk_normalize(self.data.reshape([-1, shape[-1]]), axis=0).reshape(shape))

    def minmax_scale(self) -> _Frame2DScaling:
        shape = self.data.shape
        return self.init(sk_minmax_scale(self.data.reshape([-1, shape[-1]]), axis=0).reshape(shape))
