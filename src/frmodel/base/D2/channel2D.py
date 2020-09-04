from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
import warnings

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
             by_x: int = 1,
             by_y: int = 1) -> GLCM2D:
        """ Gray-level co-occurrence matrix.

        :param by_x: The number of cells to shift on the x-axis
        :param by_y: The number of cells to shift on the y-axis
        """
        warnings.warn("glcm() is deprecated for 0.0.3, use Frame2D get_glcm to generate RGB GLCM.")

        # Int 32 to be used to support 255 * 255 (worst case scenario)
        b = self.data[:-by_y, :-by_x].astype(np.int32)
        c = self.data[by_y:, by_x:].astype(np.int32)

        return GLCM2D(b, c)

