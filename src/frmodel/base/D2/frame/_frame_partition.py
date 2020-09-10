from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from frmodel.base.consts import CONSTS


class _Frame2DPartition(ABC):

    data: np.ndarray

    @abstractmethod
    def width(self): ...

    @abstractmethod
    def height(self): ...

    # noinspection PyArgumentList
    @classmethod
    def init(cls, *args, **kwargs):
        return cls(*args, **kwargs)

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
        return [f.split(by, axis=CONSTS.AXIS.X, method=method)
                for f in self.split(by, axis=CONSTS.AXIS.Y, method=method)]

    def split(self,
              by: int,
              axis: CONSTS.AXIS = CONSTS.AXIS.Y,
              method: SplitMethod = SplitMethod.DROP) -> List[_Frame2DPartition]:
        """ Splits the current Frame into windows of specified size.

        This operation is much faster but cannot have overlapping windows

        E.g.::

            frame.split(
                by = 50,
                axis = CONSTS.AXIS.Y,
                method = Frame2D.SplitMethod.DROP
            )

        Will slice the images vertically, drops the last slice if it's not perfectly divisible.
        """

        # Pre-process by as modified by_
        # np.split_array splits it by the number of slices generated,
        # we need to transform this into the slice locations
        if axis == CONSTS.AXIS.Y:
            by_ = np.arange(by, self.width(), by)
        elif axis == CONSTS.AXIS.X:
            by_ = np.arange(by, self.height(), by)
        else:
            raise TypeError(f"Axis {axis} is not recognised. Use CONSTS.AXIS class.")

        spl = np.array_split(self.data, by_, axis=axis)
        if method == _Frame2DPartition.SplitMethod.CROP:
            # By default, it'll just "crop" the edges
            return [self.init(s) for s in spl]
        elif method == _Frame2DPartition.SplitMethod.DROP:
            # We will use a conditional to drop any images that is "cropped"
            return [self.init(s) for s in spl if s.shape[axis] == by]

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
        return [f.slide(by=by, stride=stride, axis=CONSTS.AXIS.X)
                for f in self.slide(by=by, stride=stride, axis=CONSTS.AXIS.Y)]

    def slide(self, by, stride, axis=CONSTS.AXIS.Y):
        """ Slides a window along an axis and grabs that window as a new Frame2D

        This operation is slower due to looping but allows for overlapping windows

        If the window meets the edge, it will be dropped.

        :param by: The size of the window to slide
        :param stride: The stride of the window on specified axis
        :param axis: Axis to slide on
        """

        if axis == CONSTS.AXIS.Y:
            return [self.init(self.data[:, i: i + by])
                    for i in range(0, self.width() - by + 1, stride)]
        elif axis == CONSTS.AXIS.X:
            return [self.init(self.data[i: i + by, :])
                    for i in range(0, self.height() - by + 1, stride)]
