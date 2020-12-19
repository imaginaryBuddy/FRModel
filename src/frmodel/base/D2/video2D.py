from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2

from frmodel.base import CONSTS
from frmodel.base.D2.frame2D import Frame2D


@dataclass
class Video2D:
    """ This class holds the data as OpenCV2.VideoCapture.

    Extract the actual Capture with .vid property
    """

    vid: cv2.VideoCapture

    @staticmethod
    def from_video(file_path: str) -> Video2D:
        """ Creates an instance using the file path. """
        return Video2D(cv2.VideoCapture(file_path))

    def to_frames(self,
                  offsets_msec: List[int] or int,
                  failure_default: None = None
                  ) -> List[Frame2D]:
        """ Extracts images from the video.

        Returns frmodel.Image

        :param offsets_msec: The timestamps to extract images from.
        :param failure_default: Value to default to on failure on offset read.
        :returns: List of Images
        """

        # Correct it as a List if int
        if isinstance(offsets_msec, int): offsets_msec = [offsets_msec]

        img_list = []
        for offset in offsets_msec:
            # Move to offset and attempt reading.
            self.vid.set(cv2.CAP_PROP_POS_MSEC, offset)
            success, image = self.vid.read()

            if success:
                # CV2 uses BGR, we swap the channels here
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                img_list.append(Frame2D(image, CONSTS.CHN.RGB))
            else:
                img_list.append(failure_default)

        return img_list

    """ Deprecated
    OpenCV has a lot of issues detecting duration. This returns a negative value in a test, not reliable.
    def duration(self) -> int:
        self.vid.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
        return self.vid.get(cv2.CAP_PROP_POS_MSEC)
    """
