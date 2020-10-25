from __future__ import annotations

from abc import ABC
from typing import Tuple

import numpy as np
from PIL import Image
from skimage.transform import rescale


class _Frame2DImage(ABC):
    """ This class handles the transformations like an image editor would have. """

    data: np.ndarray

    # noinspection PyArgumentList
    @classmethod
    def init(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    def crop(self,
             top:int = 0,
             right:int = 0,
             bottom:int = 0,
             left:int = 0) -> _Frame2DImage:
        """ Crops the frame by specifying how many rows/columns to remove from each side."""
        return self.init(self.data[top:-bottom or None, left:-right or None, ...])

    def save(self, file_path: str, rgb_indexes: Tuple = (0, 1, 2), **kwargs) -> None:
        """ Saves the current Frame file """
        Image.fromarray(self.data[..., rgb_indexes].astype(np.uint8)).save(file_path, **kwargs)

    def _rescale(self,
                scale: float,
                rgb_indexes: Tuple = (0, 1, 2),
                dtype=np.uint8,
                anti_aliasing=False) -> _Frame2DImage:
        """ Rescales the image. NOTE THAT THIS WILL RETURN A RGB FRAME ONLY

        Private because it causes weird behavior

        :param scale: The scaling factor. 0.5 for zoom 2x, 2 for 0.5x
        :param rgb_indexes: The indexes of the RGB Channels.
        :param dtype: The resulting dtype of the Frame2D
        :param anti_aliasing: Whether to have anti-aliasing or not
        :return: RGB Frame2D
        """

        return self.init(
            (np.round
                (rescale
                     (self.data[..., rgb_indexes].astype(dtype),
                      scale=scale,
                      anti_aliasing=anti_aliasing,
                      multichannel=True
                      ) * 256)).astype(dtype))
