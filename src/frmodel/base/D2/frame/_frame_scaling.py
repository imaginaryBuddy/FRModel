from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

from sklearn.preprocessing import minmax_scale as sk_minmax_scale
from sklearn.preprocessing import normalize as sk_normalize
import numpy as np

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

    def scale_discrete(self: 'Frame2D', minimum=0, maximum=255, dtype=np.uint8) -> 'Frame2D':
        return self.create(
            sk_minmax_scale(self.data.flatten(),
                            feature_range=(minimum, maximum),
                            copy=False).astype(dtype).reshape(self.shape),
            labels=self.labels)
