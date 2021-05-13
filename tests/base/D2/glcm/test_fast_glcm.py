import unittest

import numpy as np

from tests.base.D2Fixture.test_fixture import TestD2Fixture


class TestGLCM(TestD2Fixture):

    def test_glcm(self):
        import numpy as np
        from frmodel.base.D2.frame._cy_fast_glcm import cy_fast_glcm
        from frmodel.base.D2.frame2D import Frame2D
        from frmodel.base.consts import CONSTS
        """ Custom validated and calculated GLCM Test. See Journal for example. """
        f = Frame2D.from_image("sample.jpg")
        HEIGHT_WIDTH = (5,5)

        BINS = 4

        # Maximum value achievable after binning
        # BIN_WIDTH = 256 // BINS

        windows = (f.view_windows(*HEIGHT_WIDTH,1,1) //
                   (CONSTS.BOUNDS.MAX_RGB // BINS + 1)).astype(np.int)

        windows_a, windows_b = windows[:-1, :-1], windows[1:, 1:]
        windows = windows_a + windows_b * BINS

        BIN_COMBINED_MAXIMUM = BINS ** 2

        result = np.zeros((BIN_COMBINED_MAXIMUM,
                           f.height(), f.width(), windows.shape[-1]), dtype=np.uint8)

        cy_fast_glcm(windows.astype(np.uint16), result, True)

        resh = result.reshape([BINS, BINS, f.height(), f.width(), f.shape[-1]])





if __name__ == '__main__':
    unittest.main()
