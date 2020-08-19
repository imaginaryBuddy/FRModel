import unittest

from rsc.rsc_paths import *
from rsc.samples.frames import chestnut_0
from FRModel.base.consts import CONSTS
from FRModel.base.D2.frame2D import Frame2D
import numpy as np


class FrameTest(unittest.TestCase):

    def test(self):

        # Grab Sample
        frame = chestnut_0(0)

        # Split by both X and Y
        frames = frame.split_xy(by=100, method=Frame2D.SplitMethod.DROP)

        # Grab the red channel
        red_channel = frames[0][0].channel(CONSTS.CHANNEL.RED)

        # Get underlying data, numpy ndarray
        ar = red_channel.data

        # Assert above is true
        self.assertIsInstance(ar, np.ndarray)

if __name__ == '__main__':
    unittest.main()
