from __future__ import annotations

from dataclasses import dataclass
from typing import List
from typing import List, Tuple

import numpy as np
from skimage.color import rgb2hsv
from PIL import Image

from frmodel.base.D2.channel2D import Channel2D
from frmodel.base.consts import CONSTS

D_TYPE: np.dtype = \
        np.dtype([(CONSTS.CHANNEL.RED,   'u1'),   # 0 - 255
                  (CONSTS.CHANNEL.GREEN, 'u1'),   # 0 - 255
                  (CONSTS.CHANNEL.BLUE,  'u1')])  # 0 - 255

@dataclass
class Frame2D:
    """ A Frame is an alias to an Frame, however, it holds more than just the XYRGB Channels.

    The underlying representation is a 2D array, each cells holds a structured array, consisting of RGB Channels.

    It's easily extendable for more channels, but for now we only have the RGB.
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

        This operation is much faster but cannot have overlapping windows

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

    def slide_xy(self, by, stride=1):
        """ Short hand for sliding by both axes.

        Slides by X first, then Y.

        E.g.::

            | 1 | 6  | 11 | 16 |
            | 2 | 7  | 12 | 17 |
            | 3 | 8  | 13 | 18 |
            | 4 | 9  | 14 | 19 |
            | 5 | 10 | 15 | 20 |

            [[1,2,3,4,5],[6,7,8,...], ...]

        """
        return [f.slide(by=by, stride=stride, axis=CONSTS.AXIS.Y)
                for f in self.slide(by=by, stride=stride, axis=CONSTS.AXIS.X)]

    def slide(self, by, stride, axis=CONSTS.AXIS.X):
        """ Slides a window along an axis and grabs that window as a new Frame2D

        This operation is slower due to looping but allows for overlapping windows

        If the window meets the edge, it will be dropped.

        :param by: The size of the window to slide
        :param stride: The stride of the window on specified axis
        :param axis: Axis to slide on
        :return:
        """

        if axis == CONSTS.AXIS.X:
            return [Frame2D(self.data[:, i: i + by])
                    for i in range(0, self.width() - by + 1, stride)]
        elif axis == CONSTS.AXIS.Y:
            return [Frame2D(self.data[i: i + by, :])
                    for i in range(0, self.height() - by + 1, stride)]

    @staticmethod
    def from_image(file_path: str) -> Frame2D:
        """ Creates an instance using the file path. """
        img = Image.open(file_path)
        ar = np.asarray(img).view(dtype=D_TYPE).squeeze()
        return Frame2D(ar)

    def to_hsv(self) -> np.ndarray:
        """ Converts to a HSV Array """
        return rgb2hsv(self.data_unstruct())

    def to_ex_g(self, modified=False) -> np.ndarray:
        """ Calculates the excessive green index

        Original: 2g - 1r - 1b
        Modified: 1.262G - 0.884r - 0.331b
        """

    def to_ex_gr(self) -> np.ndarray:
        """ Calculates the excessive green minus excess red index

        """

    def data_unstruct(self) -> np.ndarray:
        """ Returns the data as a regular numpy array """
        shape = (*self.shape[0:2], -1)
        return self.data.view(dtype=np.uint8).reshape(shape)

    def save(self, file_path: str, **kwargs) -> None:
        """ Saves the current Frame file"""
        Image.fromarray(
            self.data            # Grab Data
                .ravel()
                .view(np.uint8)  # Unwrap structured array
                                 # Reshape as original shape, -1 as last index, for dynamic layer count.
                .reshape([*self.shape[0:2], -1]))\
            .save(file_path, **kwargs)

    def size(self) -> np.ndarray:
        """ Returns the number of pixels """
        return self.data.size

    @property
    def shape(self) -> Tuple:
        return self.data.shape

    def height(self):
        return self.data.shape[0]

    def width(self):
        return self.data.shape[1]

    def channel(self, channel: CONSTS.CHANNEL) -> Channel2D:
        """ Gets the red channel of the Frame """
        return Channel2D(self.data[channel]
                             .reshape(self.shape[0:2]))
