import unittest

from FRModel.base.consts import CONSTS
from rsc.samples.frames import chestnut_0


class GLCMTest(unittest.TestCase):

    def test(self):

        # Load in sample
        frame = chestnut_0(0)

        # Create 2-deep nested list of 100x100 frames
        frames = frame.split_xy(100)

        # Loop through the list
        for xframes in frames:
            for frame in xframes:
                # Grab red channel
                frame_red = frame.channel(CONSTS.CHANNEL.RED)

                # Create the pseudo-matrix with .glcm
                glcm = frame_red.glcm(by=1, axis=CONSTS.AXIS.Y)

                # Extract statistics of each window
                glcm.contrast()
                glcm.correlation()
                glcm.entropy()

if __name__ == '__main__':
    unittest.main()
