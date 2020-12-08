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
                         [5, 2, 4, 2]])

        ar = np.tile(ar, [3, 1, 1]).swapaxes(0,1).swapaxes(1,2)

        f = Frame2D(ar.astype(np.uint8))
        fc = f.get_chns(glcm_con=True, glcm_cor=True, glcm_ent=True, glcm_radius=1)
        ar2 = fc.data.squeeze()

        self.assertAlmostEqual(ar2[0], ar2[1])
        self.assertAlmostEqual(ar2[1], ar2[2])
        self.assertAlmostEqual(ar2[2], 213)

        self.assertAlmostEqual(ar2[3], ar2[4])
        self.assertAlmostEqual(ar2[4], ar2[5])
        self.assertAlmostEqual(ar2[5], -0.12209306360906494, places=4)

        self.assertAlmostEqual(ar2[6], ar2[7])
        self.assertAlmostEqual(ar2[7], ar2[8])
        self.assertAlmostEqual(ar2[8], 1)


if __name__ == '__main__':
    unittest.main()
