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

    def normalize(self, **kwargs) -> _Frame2DScaling:
        return self.scale(sk_normalize, **kwargs)
    
    def minmax_scale(self, **kwargs) -> _Frame2DScaling:
        return self.scale(sk_minmax_scale, **kwargs)

    def scale(self, scaler, **scaler_kwargs):
        shape = self.data.shape
        return self.init(scaler(self.data.reshape([-1, shape[-1]]), **scaler_kwargs, axis=0).reshape(shape))
