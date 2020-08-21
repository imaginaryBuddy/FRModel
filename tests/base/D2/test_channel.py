import unittest

from frmodel.base.consts import CONSTS
from frmodel.base.D2.glcm2D import GLCM2D
from frmodel.base.D2.channel2D import Channel2D
from tests.base.D2.test_d2 import TestD2


class ChannelTest(TestD2):

    def test(self):
        # Grab the green channel from a frame
        green = self.frame_window.channel(CONSTS.CHANNEL.GREEN)

        self.assertIsInstance(green, Channel2D)

    def test_glcm(self):
        self.assertIsInstance(self.frame_window.channel(CONSTS.CHANNEL.RED).glcm(), GLCM2D)

if __name__ == '__main__':
    unittest.main()
