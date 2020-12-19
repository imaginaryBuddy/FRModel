from __future__ import annotations

from abc import ABC
from typing import List, TYPE_CHECKING

import numpy as np

from frmodel.base.consts import CONSTS

if TYPE_CHECKING:
    from frmodel.base.D2.frame2D import Frame2D


class _Frame2DPartition(ABC):

    class SplitMethod:
        DROP = 0
        CROP = 1

        # Require to support padding? Not implemented yet.

    def split(self: 'Frame2D',
              by: int,
              axis_cut: CONSTS.AXIS = CONSTS.AXIS.Y,
              method: SplitMethod = SplitMethod.DROP) -> List[_Frame2DPartition]:
        """ Splits the current Frame into windows of specified size.

        This operation is much faster but cannot have overlapping windows

        :param by: Size of each piece
        :param axis_cut: Axis to cut
        :param method: Method of splitting
        :return: List of Frame2D

        """

        # Pre-process by as modified by_
        # np.split_array splits it by the number of slices generated,
        # we need to transform this into the slice locations

        if axis_cut == CONSTS.AXIS.Y:
            by_ = np.arange(by, self.width(), by)
        elif axis_cut == CONSTS.AXIS.X:
            by_ = np.arange(by, self.height(), by)
        else:
            raise TypeError(f"Axis {axis_cut} is not recognised. Use CONSTS.AXIS class.")

        spl = np.array_split(self.data, by_, axis=axis_cut)
        if method == _Frame2DPartition.SplitMethod.CROP:
            # By default, it'll just "crop" the edges
            return [self.create(data=s, labels=self.labels) for s in spl]
        elif method == _Frame2DPartition.SplitMethod.DROP:
            # We will use a conditional to drop any images that is "cropped"
            return [self.create(data=s, labels=self.labels) for s in spl if s.shape[axis_cut] == by]

    def split_xy(self: 'Frame2D',
                 by: int,
                 method: SplitMethod = SplitMethod.DROP) -> List[List[_Frame2DPartition]]:
        """ Short hand for splitting by both axes.

        Splits by X first, then Y.

        E.g.::

            | 1 | 6  | 11 | 16 |
            | 2 | 7  | 12 | 17 |
            | 3 | 8  | 13 | 18 |
            | 4 | 9  | 14 | 19 |
            | 5 | 10 | 15 | 20 |

            [[1,2,3,4,5],[6,7,8,...], ...]


        :param by: Size of each piece
        :param method: Method of splitting
        :return: List of List of Frame2D
        """
        return [f.split(by, axis_cut=CONSTS.AXIS.X, method=method)
                for f in self.split(by, axis_cut=CONSTS.AXIS.Y, method=method)]

    def slide_xy(self, by, stride=1) -> List[List[_Frame2DPartition]]:
        """ Short hand for sliding by both axes.

        Slides by X first, then Y.

        E.g.::

            | 1 | 6  | 11 | 16 |
            | 2 | 7  | 12 | 17 |
            | 3 | 8  | 13 | 18 |
            | 4 | 9  | 14 | 19 |
            | 5 | 10 | 15 | 20 |

            [[1,2,3,4,5],[6,7,8,...], ...]


        :param by: Size of each piece
        :param stride: Stride when sliding
        :return: List of List of Frame2D

        """
        return [f.slide(by=by, stride=stride, axis_cut=CONSTS.AXIS.X)
                for f in self.slide(by=by, stride=stride, axis_cut=CONSTS.AXIS.Y)]

    def slide(self, by, stride, axis_cut: CONSTS.AXIS = CONSTS.AXIS.Y) -> List[_Frame2DPartition]:
        """ Slides a window along an axis and grabs that window as a new Frame2D

        This operation is slower due to looping but allows for overlapping windows

        If the window meets the edge, it will be dropped.

        :param by: Size of each piece
        :param stride: Stride when sliding
        :param axis_cut: Axis to cut
        :return: List of Frame2D
        """

        self: 'Frame2D'
        if axis_cut == CONSTS.AXIS.Y:
            return [self.create(data=self.data[:, i: i + by],
                                labels=self.labels)
                    for i in range(0, self.width() - by + 1, stride)]
        elif axis_cut == CONSTS.AXIS.X:
            return [self.create(data=self.data[i: i + by, :],
                                labels=self.labels)
                    for i in range(0, self.height() - by + 1, stride)]
