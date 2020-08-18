import unittest

from rsc.rsc_paths import *
from FRModel.base.consts import CONSTS
from FRModel.base.D2.frame2D import Frame2D

class GLCMTest(unittest.TestCase):
    def test(self):

        frame = Frame2D.from_image(SAMPLE_CHESTNUT_0S_IMG)
        frames = frame.split_xy(100)

        for xframes in frames:
            for frame in xframes:
                frame_red = frame.channel(CONSTS.CHANNEL.RED)
                glcm = frame_red.glcm(by=1, axis=CONSTS.AXIS.Y)
                glcm.contrast()
                glcm.correlation()
                glcm.entropy()

if __name__ == '__main__':
    unittest.main()
