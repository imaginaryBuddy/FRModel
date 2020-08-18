""" Implementation for extract_run
"""

from FRModel.base.D2.video2D import Video2D
import os
import numpy as np
from pathlib import Path

def extract(file_path,
            start:int = 0,
            end:int = 120000,
            interval:int = 10000,
            return_on_exist: bool = True):
    """ Extracts images from video

    If return_on_exist, we return upon noticing the directory exists.

    To fully regenerate all files, delete the entire directory.

    :param file_path: Path to file
    :param start: Start of extraction in ms
    :param end: End of extraction in ms
    :param interval: Gap in ms
    :param return_on_exist: Lazy evaluation on if the extraction is done
    """
    # Create a directory if doesn't exist
    dir_name = Path(file_path).stem
    try:
        os.mkdir(dir_name)
    except FileExistsError:
        # This just means the dir exists.
        if return_on_exist: return

    # Load video and extract images
    v = Video2D.from_video(file_path)
    for e, img in enumerate(v.to_images(np.arange(start, end, interval))):
        img.save(f"{dir_name}/frame{e * interval}ms.jpg")