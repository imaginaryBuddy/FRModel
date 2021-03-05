import unittest

from frmodel.base.D2.draw2D import Draw2D
from tests.base.D2Fixture.test_fixture import TestD2Fixture

import numpy as np


class TestDraw(TestD2Fixture):

    def test_draw_multiple(self):
        draw = Draw2D.load_frame(self.frame)
        x = np.linspace(0, self.frame.width(), 200)
        y = np.random.randint(0, self.frame.height(), 200)
        draw.mark_multiple(x,
                           y,
                           outline=(255, 0, 0),
                           labels=[f"{i:.0f}, {j:.0f}" for i, j in zip(x, y)])
        draw.save("./DrawMultipleRandom.png")

    def test_draw_single(self):
        draw = Draw2D.load_frame(self.frame)
        draw.mark_single(self.frame.width() // 2,
                         self.frame.height() // 2,
                         outline=(255, 0, 0),
                         label="Mid Point")
        draw.save("./DrawSingle.png")

if __name__ == '__main__':
    unittest.main()
