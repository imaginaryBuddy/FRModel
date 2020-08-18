from __future__ import annotations

from typing import List
from dataclasses import dataclass
import numpy as np
from PIL import Image
from FRModel.base.D2.channel2D import Channel2D
from FRModel.base.consts import CONSTS

D_TYPE: np.dtype = \
        np.dtype([(CONSTS.CHANNEL.RED,   'u1'),   # 0 - 255
                  (CONSTS.CHANNEL.GREEN, 'u1'),   # 0 - 255
                  (CONSTS.CHANNEL.BLUE,  'u1')])  # 0 - 255

@dataclass
class Frame2D:
    """ A Frame is an alias to an Frame, however, it holds more than just the XYRGB Channels.

    Note that due to the nature of np arrays, it cannot be any irregular shape.
    The structure is very simple, it's a 4D object always. That is, the X, Y, Z, ?.

    Because X, Y, Z are directional, we can place it logically it in an array index.
    However, RGB(and other spectral ranges) are not directional, and are dependent on the XYZ.

    Hence they are placed in a structured array.

    """

    data: np.ndarray

    class SplitMethod:
        DROP = 0
        CROP = 1

        # Require to support padding? Not implemented yet.
    def split_xy(self,
                 by: int,
                 method: SplitMethod = SplitMethod.DROP):
        """ Short hand for splitting by both axes.

        Splits by X first, then Y.

        E.g.::

            | 1 | 6  | 11 | 16 |
            | 2 | 7  | 12 | 17 |
            | 3 | 8  | 13 | 18 |
            | 4 | 9  | 14 | 19 |
            | 5 | 10 | 15 | 20 |

            [[1,2,3,4,5],[6,7,8,...], ...]

        """
        return [f.split(by, axis=CONSTS.AXIS.Y, method=method)
                for f in self.split(by, axis=CONSTS.AXIS.X, method=method)]

    def split(self,
              by: int,
              axis: CONSTS.AXIS = CONSTS.AXIS.X,
              method: SplitMethod = SplitMethod.DROP) -> List[Frame2D]:
        """ Splits the current Frame into windows of specified size.

        E.g.::

            frame.split(
                by = 50,
                axis = CONSTS.AXIS.X,
                method = Frame2D.SplitMethod.DROP
            )

        Will slice the images vertically, drops the last slice if it's not perfectly divisible.
        """

        # Pre-process by as modified by_
        # np.split_array splits it by the number of slices generated,
        # we need to transform this into the slice locations
        if    axis == CONSTS.AXIS.X: by_ = np.arange(by, self.width(), by)
        elif  axis == CONSTS.AXIS.Y: by_ = np.arange(by, self.height(), by)
        else: raise TypeError(f"Axis {axis} is not recognised. Use CONSTS.AXIS class.")

        spl = np.array_split(self.data, by_, axis=axis)
        if method == Frame2D.SplitMethod.CROP:
            # By default, it'll just "crop" the edges
            return [Frame2D(s) for s in spl]
        elif method == Frame2D.SplitMethod.DROP:
            # We will use a conditional to drop any images that is "cropped"
            return [Frame2D(s) for s in spl if s.shape[axis] == by]

    @staticmethod
    def from_image(file_path: str) -> Frame2D:
        """ Creates an instance using the file path. """
        img = Image.open(file_path)
        ar = np.asarray(img)
        return Frame2D(ar.view(dtype=D_TYPE))

    def save(self, file_path: str, **kwargs) -> None:
        """ Saves the current Frame file"""
        Image.fromarray(
            self.data            # Grab Data
                .ravel()         # Flatten
                .view(np.uint8)  # Unwrap structured array
                                 # Reshape as original shape, -1 as last index, for dynamic layer count.
                .reshape([*self.shape()[0:2], -1]))\
            .save(file_path, **kwargs)

    def size(self):
        """ Returns the number of pixels """
        return self.data.size

    def shape(self):
        return self.data.shape

    def height(self):
        return self.data.shape[0]

    def width(self):
        return self.data.shape[1]

    def channel(self, channel: CONSTS.CHANNEL) -> Channel2D:
        """ Gets the red channel of the Frame """
        return Channel2D(self.data[channel]
                             .reshape(self.shape()[0:2]))