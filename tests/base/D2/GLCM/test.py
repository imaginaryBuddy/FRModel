import unittest

from frmodel.base.consts import CONSTS
from rsc.samples.frames import chestnut_0


class GLCMTest(unittest.TestCase):

    def test(self):

        WINDOW_SIZE = 100
        # Load in sample
        frame = chestnut_0(0)

        # Create 2-deep nested list of 100x100 frames
        frames = frame.split_xy(WINDOW_SIZE)

        # Grab red channel from first 100x100
        frame_red = frames[0][0].channel(CONSTS.CHANNEL.RED)

        GLCM_SHIFT = 1
        # Create the pseudo-matrix with .glcm. Shift by Y
        glcm = frame_red.glcm(by=GLCM_SHIFT, axis=CONSTS.AXIS.Y)

        # Verify Window Size
        self.assertEqual((WINDOW_SIZE - GLCM_SHIFT, WINDOW_SIZE), glcm.data0.shape)
        self.assertEqual((WINDOW_SIZE - GLCM_SHIFT, WINDOW_SIZE), glcm.data1.shape)

        # Extract statistics of each window
        glcm.contrast()
        glcm.correlation()
        glcm.entropy()


if __name__ == '__main__':
    unittest.main()
