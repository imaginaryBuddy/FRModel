import unittest

from sklearn.cluster import KMeans
from sklearn.preprocessing import minmax_scale

from frmodel.base import CONSTS
from frmodel.base.D2 import Frame2D
from frmodel.base.D2.kmeans2D import KMeans2D
from tests.base.D2.test_d2 import TestD2


class ScoreTest(TestD2):

    def test_box(self):
        C = CONSTS.CHN

        f = Frame2D.from_image(self._RSC + "/imgs/basic/box.png")
        # frame_xy = f.get_chns(xy=True, hsv=True, mex_g=True, ex_gr=True, ndi=True)
        frame_xy = f.get_chns(self_=False,
                              chns=[C.XY, C.HSV, C.MEX_G, C.EX_GR, C.NDI])

        km = KMeans2D(frame_xy,
                      KMeans(n_clusters=3, verbose=False),
                      fit_to=[C.MEX_G, C.EX_GR, C.NDI],
                      scaler=minmax_scale)
        kmf = km.as_frame()
        score = kmf.score(f)
        self.assertAlmostEqual(score['Custom'], 1)
        self.assertAlmostEqual(score['Homogeneity'], 1)
        self.assertAlmostEqual(score['Completeness'], 1)
        self.assertAlmostEqual(score['V Measure'], 1)


if __name__ == '__main__':
    unittest.main()
