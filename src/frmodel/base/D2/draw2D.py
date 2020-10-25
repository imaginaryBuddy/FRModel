from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Any

from PIL import Image, ImageDraw

from frmodel.base.D2 import Frame2D

DRAW_MODE = "RGB"

@dataclass
class Draw2D:

    canvas: Image.Image
    canvas_draw: ImageDraw.ImageDraw

    @staticmethod
    def new(height, width, default_color=(255,255,255)):
        canvas = Image.new(DRAW_MODE, [height, width], color=default_color)
        return Draw2D(canvas, ImageDraw.Draw(canvas, mode=DRAW_MODE))

    def save(self, file_path):
        self.canvas.save(file_path)

    @staticmethod
    def load_image(file_path):
        canvas = Image.open(file_path, mode=DRAW_MODE)
        return Draw2D(canvas, ImageDraw.Draw(canvas, mode=DRAW_MODE))

    def mark_single(self, x: int, y: int, label: Any = None,
                    radius: int = 2, outline: Tuple = (255, 255, 255),
                    fill: Tuple = None):
        """ Marks a single point

        :param x: x Position
        :param y: y Position
        :param label: Label of point, no label is NOne
        :param radius: Radius of marker
        :param outline: Outline color of marker, also text color
        :param fill: Fill color of marker, no fill is None.
        """
        self.canvas_draw.ellipse([x - radius, y - radius,
                                  x + radius, y + radius],
                                 outline=outline, fill=fill)
        if label: self.canvas_draw.text([x + radius, y + radius], label, fill=outline)

    def mark_multiple(self, xs, ys, labels=None,
                      radius: int = 2,
                      outline: Tuple = (255,255,255),
                      fill: Tuple = None):
        """ Marks on multiple points

        :param xs: Any iterable of X values
        :param ys: Any iterable of Y values
        :param labels: Any iterable of Labels, no label if None
        :param radius: Radius of marker
        :param outline: Outline color of marker, also text color
        :param fill: Fill color of marker, no fill is None.
        """
        assert len(xs) == len(ys), "xy Lengths must be the same"
        if not labels: labels = [None] * len(xs)
        else: assert len(xs) == len(labels), "label Lengths must be the same as xy."

        for x, y, l in zip(xs, ys, labels):
            self.mark_single(x, y, l, radius, outline, fill)

    @staticmethod
    def load_frame(frame: Frame2D) -> Draw2D:
        canvas = Image.fromarray(frame.data_rgb(), mode=DRAW_MODE)
        return Draw2D(canvas, ImageDraw.Draw(canvas, DRAW_MODE))

    def draw(self):
        return ImageDraw.Draw(self.canvas)
