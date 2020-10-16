from __future__ import annotations
from abc import ABC, abstractmethod
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

    def rescale(self, scale: float, rgb_indexes: Tuple = (0, 1, 2), anti_aliasing=False) -> _Frame2DImage:
        """ Rescales the image.

        Note that this will only the Frame with the RGB channels only
        """
        return self.init(rescale(self.data[..., rgb_indexes], scale=scale, anti_aliasing=anti_aliasing))

    def crop(self,
             top:int = 0,
             right:int = 0,
             bottom:int = 0,
             left:int = 0) -> _Frame2DImage:
        """ Crops the frame """
        return self.init(self.data[top:-bottom, left:-right, ...])

    def save(self, file_path: str, **kwargs) -> None:
        """ Saves the current Frame file """
        Image.fromarray(self.data_rgb().astype(np.uint8)).save(file_path, **kwargs)