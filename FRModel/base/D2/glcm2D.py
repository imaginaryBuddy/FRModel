import numpy as np
from dataclasses import dataclass


@dataclass
class GLCM2D:

    data0: np.ndarray
    data1: np.ndarray

    def contrast(self):
        return np.sum((self.data0 - self.data1) ** 2)

    def correlation(self):
        mean_x = np.mean(self.data0)
        mean_y = np.mean(self.data1)
        mean = mean_x - mean_y
        std_x = np.std(self.data0)
        std_y = np.std(self.data1)
        std = std_x * std_y
        return np.sum(((self.data0 * self.data1) - mean) / std)

    def entropy(self):
        # We create a new array with double data0's shape to fit data1
        data2 = np.zeros([*self.data0.shape, 2])
        data2[..., 0] = self.data0
        data2[..., 1] = self.data1

        # We get the occurrences of each pair
        _, counts = np.unique(data2, axis=0, return_counts=True)
        return np.sum(counts ** 2)