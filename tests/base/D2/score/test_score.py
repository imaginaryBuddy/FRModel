import unittest

from sklearn.cluster import KMeans
from sklearn.preprocessing import minmax_scale

from frmodel.base import CONSTS
from frmodel.base.D2 import Frame2D
from frmodel.base.D2.kmeans2D import KMeans2D
from tests.base.D2Fixture.test_fixture import TestD2Fixture


class TestScore(TestD2Fixture):

    def test_box(self):
        C = CONSTS.CHN

        frame_xy = self.frame_box.get_chns(self_=False,
                              chns=[C.XY, C.HSV, C.MEX_G, C.EX_GR, C.NDI])

        km = KMeans2D(frame_xy,
                      KMeans(n_clusters=3, verbose=False),
                      fit_to=[C.MEX_G, C.EX_GR, C.NDI],
                      scaler=minmax_scale)
        kmf = km.as_frame()
        score = kmf.score(self.frame_box)
        self.assertAlmostEqual(score['Custom'], 1)
        self.assertAlmostEqual(score['Homogeneity'], 1)
        self.assertAlmostEqual(score['Completeness'], 1)
        self.assertAlmostEqual(score['V Measure'], 1)


if __name__ == '__main__':
    unittest.main()
