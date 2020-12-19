import unittest


from frmodel.base.D2.frame2D import Frame2D
from frmodel.base.consts import CONSTS
from tests.base.D2.test_d2 import TestD2
import numpy as np

class GLCMTest(TestD2):

    def test_glcm(self):
        ar = np.asarray([[5, 8, 9, 5],
                         [0, 0, 1, 7],
                         [6, 9, 2, 4],
                         [5, 2, 4, 2]]).transpose()[...,np.newaxis]

        f = Frame2D(ar.astype(np.uint8), CONSTS.CHN.RED)
        fc = f.get_chns(self_=False,
                        glcm=Frame2D.GLCM(radius=1,
                                     contrast=[CONSTS.CHN.RED],
                                     correlation=[CONSTS.CHN.RED],
                                     entropy=[CONSTS.CHN.RED])
                        )

        """ The reason why I made calling this so verbose is to make it easy for development. """

        self.assertAlmostEqual(fc.data_chn(fc.CHN.GLCM.CON(fc.CHN.RED)).data.squeeze(), 213)
        self.assertAlmostEqual(fc.data_chn(fc.CHN.GLCM.COR(fc.CHN.RED)).data.squeeze(), -0.12209306360906494,
                               places=4)
        self.assertAlmostEqual(fc.data_chn(fc.CHN.GLCM.ENT(fc.CHN.RED)).data.squeeze(), 1)


if __name__ == '__main__':
    unittest.main()
