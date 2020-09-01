import unittest

from frmodel.base.consts import CONSTS
from tests.base.D2.test_d2 import TestD2


class GLCMTest(TestD2):

    def test(self):
        GLCM_SHIFT = 1

        # Grab red channel from first 100x100
        frame_red = self.frame_window.channel(CONSTS.CHANNEL.RED)

        # Create the pseudo-matrix with .glcm. Shift by Y
        glcm = frame_red.glcm(by=GLCM_SHIFT, axis=CONSTS.AXIS.Y)

        # Verify Window Size
        self.assertEqual((self.window - GLCM_SHIFT, self.window), glcm.data0.shape)
        self.assertEqual((self.window - GLCM_SHIFT, self.window), glcm.data1.shape)

        # Extract statistics of each window
        glcm.contrast(True)
        glcm.correlation(True)
        glcm.entropy(True)


if __name__ == '__main__':
    unittest.main()
