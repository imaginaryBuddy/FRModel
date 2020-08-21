import unittest

import numpy as np

from frmodel.base.D2.frame2D import Frame2D
from frmodel.base.D2.channel2D import Channel2D
from frmodel.base.consts import CONSTS
from rsc.samples.frames import chestnut_0


class FrameTest(unittest.TestCase):

    def __init__(self):
        super().__init__()
        self.frame = chestnut_0(0)

    def test_split_xy(self):
        # Split by both X and Y
        frames = self.frame.split_xy(by=100, method=Frame2D.SplitMethod.DROP)

        self.assertEqual(frames[0][0].shape, (100, 100))

    def test_split(self):
        # Split by Y Axis only, horizontal slices
        frames = self.frame.split(by=100, method=Frame2D.SplitMethod.DROP, axis=CONSTS.AXIS.Y)

        self.assertEqual(frames[0].shape[0], 100)

    def test_flatten(self):
        self.assertEqual(self.frame.data_flatten().shape, (*self.frame.shape, 3))

    def test_channel(self):
        # Grab the red channel
        red = self.frame.channel(CONSTS.CHANNEL.RED)

        self.assertIsInstance(red, Channel2D)


if __name__ == '__main__':
    unittest.main()
