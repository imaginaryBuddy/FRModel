import unittest

from sklearn.cluster import KMeans
from sklearn.preprocessing import minmax_scale

from frmodel.base.D2 import Frame2D
from frmodel.base.D2.draw2D import Draw2D
from tests.base.D2.test_d2 import TestD2

import numpy as np


class ScoreTest(TestD2):

    def test_box(self):
        f = Frame2D.from_image(self._RSC + "/imgs/basic/box.png")
        frame_xy = f.get_chns(xy=True, hsv=True, mex_g=True, ex_gr=True, ndi=True)

        km = frame_xy.kmeans(KMeans(n_clusters=3, verbose=False),
                             fit_indexes=[2, 3, 4, 5, 6, 7],
                             scaler=minmax_scale)

        self.assertAlmostEqual(km.score(self._RSC + "/imgs/basic/box.png")[1], 1)

if __name__ == '__main__':
    unittest.main()
