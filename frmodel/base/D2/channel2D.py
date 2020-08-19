from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from frmodel.base.D2.glcm2D import GLCM2D
from frmodel.base.consts import CONSTS


@dataclass
class Channel2D:
    """ A Channel is a slice of the frame, a layer.

    The channel still holds the X, Y data.

    """

    data: np.ndarray

    def save(self, file_path: str, **kwargs) -> None:
        """ Saves the current Frame file"""
        Image.fromarray(self.data.view(np.uint8)).save(file_path, **kwargs)

    def glcm(self,
             by: int = 1,
             axis: CONSTS.AXIS = CONSTS.AXIS.X) -> GLCM2D:
        """ Gray-level co-occurrence matrix. """

        w, h = self.data.shape

        if axis == CONSTS.AXIS.X:
            # Int 32 to be used to support 255 * 255 (worst case scenario)
            b = self.data[:, 0:w - by].ravel().astype(np.int32)
            c = self.data[:, by:w].ravel().astype(np.int32)
        elif axis == CONSTS.AXIS.Y:
            # Int 32 to be used to support 255 * 255 (worst case scenario)
            b = self.data[0:h - by, :].ravel().astype(np.int32)
            c = self.data[by:w, :].ravel().astype(np.int32)
        else:
            raise NotImplementedError(f"Invalid Axis {axis}.")

        return GLCM2D(b, c)

