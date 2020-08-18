import numpy as np
from dataclasses import dataclass


@dataclass
class GLCM2D:
    """ This is the Gray-Level Co-occurrence Matrix.

    Created using Channel2D.glcm

    This is useful in calculating the difference between 2 ndarrays.
    """

    data0: np.ndarray
    data1: np.ndarray

    def contrast(self) -> np.ndarray:
        """ SUM((x-y)^2) """
        return np.sum((self.data0 - self.data1) ** 2)

    def correlation(self) -> np.ndarray:
        """ SUM{[(x-y)-(mean_x-mean_y)]/(std_x * std_y)} """
        mean_x = np.mean(self.data0)
        mean_y = np.mean(self.data1)
        mean = mean_x - mean_y
        std_x = np.std(self.data0)
        std_y = np.std(self.data1)
        std = std_x * std_y
        return np.sum(((self.data0 * self.data1) - mean) / std)

    def entropy(self) -> np.ndarray:
        """ SUM(Occurrence^2)

        This creates all pairs then groups them together,
        """
        # We create a new array with double data0's shape to fit data1
        data2 = np.zeros([*self.data0.shape, 2])
        data2[..., 0] = self.data0
        data2[..., 1] = self.data1

        # We get the occurrences of each pair
        # Can we optimize this?
        _, counts = np.unique(data2, axis=0, return_counts=True)
        return np.sum(counts ** 2)
