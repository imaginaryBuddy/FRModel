from __future__ import annotations
from math import ceil
from PIL import Image
import numpy as np
from abc import ABC


class _Frame2DLoader(ABC):

    # noinspection PyArgumentList
    @classmethod
    def init(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    @classmethod
    def from_image(cls, file_path: str):
        """ Creates an instance using the file path. """
        img = Image.open(file_path)
        ar = np.asarray(img)
        return cls.init(ar)

    @classmethod
    def from_rgbxy_(cls, ar: np.ndarray, xy_pos=(3,4), width=None, height=None) -> _Frame2DLoader:
        """ Rebuilds the frame with XY values. XY should be of integer values, otherwise, will be casted.

        Note that RGB channels MUST be on index 0, 1, 2 else some functions may break. However, can be ignored.

        The frame will be rebuild and all data will be retained, including XY.

        :param ar: The array to rebuild
        :param xy_pos: The positions of X and Y.
        :param height: Height of expected image, if None, Max will be used
        :param width: Width of expected image, if None, Max will be used
        """
        max_y = height if height else np.max(ar[:,xy_pos[1]]) + 1
        max_x = width if width else np.max(ar[:,xy_pos[0]]) + 1

        fill = np.zeros(( ceil(max_y), ceil(max_x), ar.shape[-1]), dtype=ar.dtype)

        # Vectorized X, Y <- RGBXY... Assignment
        fill[ar[:, xy_pos[1]].astype(int),
             ar[:, xy_pos[0]].astype(int)] = ar[:]

        return cls.init(fill)
