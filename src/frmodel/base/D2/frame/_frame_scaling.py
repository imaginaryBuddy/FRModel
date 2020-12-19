from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

from sklearn.preprocessing import minmax_scale as sk_minmax_scale
from sklearn.preprocessing import normalize as sk_normalize

if TYPE_CHECKING:
    from frmodel.base.D2.frame2D import Frame2D

class _Frame2DScaling(ABC):

    def normalize(self: 'Frame2D', **kwargs) -> _Frame2DScaling:
        return self.scale(sk_normalize, **kwargs)

    def minmax_scale(self: 'Frame2D', **kwargs) -> _Frame2DScaling:
        return self.scale(sk_minmax_scale, **kwargs)

    def scale(self: 'Frame2D', scaler, **scaler_kwargs):
        shape = self.data.shape
        return self.create(data=scaler(self.data_flatten_xy(), **scaler_kwargs).reshape(shape),
                           labels=self.labels)
