from __future__ import annotations

from abc import ABC
from typing import List, TYPE_CHECKING

import numpy as np
from skimage.util import view_as_windows, view_as_blocks

from frmodel.base.consts import CONSTS

if TYPE_CHECKING:
    from frmodel.base.D2.frame2D import Frame2D

from warnings import warn


class _Frame2DPartition(ABC):

    def view_windows(self: 'Frame2D',
                     height: int,
                     width: int,
                     height_step: int,
                     width_step: int) -> np.ndarray:
        """ Partitions the image into windows using striding

        Note that the parameters must cleanly divide the image else it'll be cropped
        
        E.g. 200 x 200 cannot be 100 Step 3 x 100 Step 7

        The dims are [RowBlock, ColBlock, Row, Col, Channels]

        :param height: Height of expected window
        :param width: Width of expected window
        :param height_step: Step of height window
        :param width_step: Step of width window
        """

        view: np.ndarray =\
            view_as_windows(self.data,
                            (height, width, self.shape[-1]),
                            (height_step, width_step, 1)).squeeze(2)
        # [HW, WW, H, W, C]
        return view

    def view_windows_as_frames(self: 'Frame2D',
                               height: int,
                               width: int,
                               height_step: int,
                               width_step: int) -> List[List['Frame2D']]:
        """ Partitions the image into windows using striding

        Note that the parameters must cleanly divide the image else it'll be cropped

        E.g. 200 x 200 cannot be 100 Step 3 x 100 Step 7

        :param height: Height of expected window
        :param width: Width of expected window
        :param height_step: Step of height window
        :param width_step: Step of width window
        """
        view = self.view_windows(height, width, height_step, width_step)
        return [[self.create(view[h, w], self.labels)
                 for h in range(view.shape[0])] for w in range(view.shape[1])]

    def view_blocks(self: 'Frame2D',
                    height: int,
                    width: int) -> np.ndarray:
        """ Partitions the image into blocks using striding

        Note that the height and width must be a divisor.

        The dims are [RowBlock, ColBlock, Row, Col, Channels]

        :param height: Height of expected block, must be int divisor of original height
        :param width: Width of expected block, must be int divisor of original width
        """
        view: np.ndarray =\
            view_as_blocks(self.data,
                           (height, width, self.shape[-1])).squeeze(2)
        # [HW, WW, H, W, C]
        return view

    def view_blocks_as_frames(self: 'Frame2D',
                              height: int,
                              width: int) -> List[List['Frame2D']]:
        """ Partitions the image into blocks using striding

        Note that the height and width must be a divisor.

        :param height: Height of expected block, must be int divisor of original height
        :param width: Width of expected block, must be int divisor of original width
        """
        view = self.view_blocks(height, width)

        return [[self.create(view[h, w], self.labels)
                 for h in range(view.shape[0])] for w in range(view.shape[1])]

    # -------- DEPRECATED FUNCTIONS --------
    class SplitMethod:
        DROP = 0
        CROP = 1

    def split(self: 'Frame2D',
              by: int,
              axis_cut: CONSTS.AXIS = CONSTS.AXIS.Y,
              method: SplitMethod = SplitMethod.DROP) -> List['Frame2D']:
        """ Splits the current Frame into windows of specified size.

        This operation is much faster but cannot have overlapping windows

        :param by: Size of each piece
        :param axis_cut: Axis to cut
        :param method: Method of splitting
        :return: List of Frame2D

        """

        warn(f"split is deprecated, use view_blocks instead", DeprecationWarning, stacklevel=2)

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
                 method: SplitMethod = SplitMethod.DROP) -> List[List['Frame2D']]:
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

        warn(f"split_xy is deprecated, use view_blocks instead", DeprecationWarning, stacklevel=2)

        return [f.split(by, axis_cut=CONSTS.AXIS.X, method=method)
                for f in self.split(by, axis_cut=CONSTS.AXIS.Y, method=method)]

    def slide_xy(self, by, stride=1) -> List[List['Frame2D']]:
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

        warn(f"slide_xy is deprecated, use view_windows instead", DeprecationWarning, stacklevel=2)

        return [f.slide(by=by, stride=stride, axis_cut=CONSTS.AXIS.X)
                for f in self.slide(by=by, stride=stride, axis_cut=CONSTS.AXIS.Y)]

    def slide(self: 'Frame2D', by, stride, axis_cut: CONSTS.AXIS = CONSTS.AXIS.Y) -> List['Frame2D']:
        """ Slides a window along an axis and grabs that window as a new Frame2D

        This operation is slower due to looping but allows for overlapping windows

        If the window meets the edge, it will be dropped.

        :param by: Size of each piece
        :param stride: Stride when sliding
        :param axis_cut: Axis to cut
        :return: List of Frame2D
        """

        warn(f"slide is deprecated, use view_windows instead", DeprecationWarning, stacklevel=2)

        if axis_cut == CONSTS.AXIS.Y:
            return [self.create(data=self.data[:, i: i + by],
                                labels=self.labels)
                    for i in range(0, self.width() - by + 1, stride)]
        elif axis_cut == CONSTS.AXIS.X:
            return [self.create(data=self.data[i: i + by, :],
                                labels=self.labels)
                    for i in range(0, self.height() - by + 1, stride)]
