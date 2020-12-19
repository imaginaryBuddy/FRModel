import unittest

from tests.base.D2.test_d2 import TestD2

class DrawTest(TestD2):

    def test_plot(self):
        fc = self.frame.get_chns(self_=False, chns=[self.frame.CHN.XY, self.frame.CHN.HSV])

        fpl = fc.plot()
        ROWS = 3
        COLS = 2
        PLT_SCALE = 1.1
        fpl.subplot_shape = (ROWS, COLS)
        fpl.image(scale=PLT_SCALE)


if __name__ == '__main__':
    unittest.main()
