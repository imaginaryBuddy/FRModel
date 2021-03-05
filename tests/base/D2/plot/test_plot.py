import unittest

from tests.base.D2Fixture.test_fixture import TestD2Fixture

class TestPlot(TestD2Fixture):

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
