import unittest

from frmodel.base.D2.frame2D import Frame2D
from frmodel.base.consts import CONSTS
from tests.base.D2Fixture.test_fixture import TestD2Fixture


class TestFrame(TestD2Fixture):

    def test_view_windows(self):
        height = self.frame.height()
        width = self.frame.width()
        step = 100
        windows = self.frame.view_windows(height // 2, width // 2, step, step)
        self.assertEqual(((height - (height // 2)) // step + 1,  # +1: Fencepost error
                          (width - (width // 2)) // step + 1,    # +1: Fencepost error
                           height // 2, width // 2, self.frame.shape[-1]),
                         windows.shape)

    def test_view_blocks(self):
        height = self.frame.height()
        width = self.frame.width()
        blocks = 5
        windows = self.frame.view_blocks(height // blocks, width // blocks)
        self.assertEqual((blocks, blocks,
                          height // blocks, width // blocks, self.frame.shape[-1]),
                         windows.shape)

    def test_split_xy(self):
        """ Deprecated Function """
        # Split by both X and Y
        frames = self.frame.split_xy(by=self.window, method=Frame2D.SplitMethod.DROP)

        self.assertEqual((self.window, self.window, self.channels), frames[0][0].shape)

    def test_split(self):
        """ Deprecated Function """
        # Split by X Axis only, horizontal slices
        frames = self.frame.split(by=self.window, method=Frame2D.SplitMethod.DROP, axis_cut=CONSTS.AXIS.X)

        # Hence the number of rows (index 0) must be equal to window size
        self.assertEqual(self.window, frames[0].shape[0])

    def test_slide_xy(self):
        """ Deprecated Function """
        # Slide by both X and Y
        frames = self.frame.slide_xy(by=self.window, stride=self.window // 2)

        self.assertEqual((self.window, self.window, self.channels), frames[0][0].shape)

    def test_slide(self):
        """ Deprecated Function """
        # Slide by X Axis only, horizontal slices
        frames = self.frame.slide(by=self.window, stride=self.window // 2, axis_cut=CONSTS.AXIS.X)

        # Hence the number of rows (index 0) must be equal to window size
        self.assertEqual(self.window, frames[0].shape[0])


if __name__ == '__main__':
    unittest.main()
