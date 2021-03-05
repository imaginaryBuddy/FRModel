import unittest

from frmodel.base.D2.draw2D import Draw2D
from tests.base.D2Fixture.test_fixture import TestD2Fixture

import numpy as np


class DrawTest(TestD2Fixture):

    def test_draw_multiple(self):
        draw = Draw2D.load_frame(self.frame)
        x = np.linspace(0, 1500, 200)
        y = np.random.randint(100, 300, 200)
        draw.mark_multiple(x,
                           y,
                           outline=(255, 0, 0),
                           labels=[f"{i:.0f}, {j:.0f}" for i, j in zip(x, y)])

    def test_draw_single(self):
        draw = Draw2D.load_frame(self.frame)
        draw.mark_single(100,
                         200,
                         outline=(255, 0, 0),
                         label="100, 200")

if __name__ == '__main__':
    unittest.main()
