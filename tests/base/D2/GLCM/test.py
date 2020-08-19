import unittest

from rsc.samples.frames import chestnut_0
from FRModel.base.consts import CONSTS
from FRModel.base.D2.frame2D import Frame2D


class GLCMTest(unittest.TestCase):

    def test(self):

        frame = chestnut_0(0)
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
