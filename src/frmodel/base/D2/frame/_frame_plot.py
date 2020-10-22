from abc import abstractmethod
from dataclasses import dataclass
from math import ceil
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import seaborn as sns
from sklearn.preprocessing import minmax_scale, normalize


@dataclass
class Frame2DPlot:

    data: np.ndarray

    def _create_grid(self, scale):
        channels = self.data.shape[-1]
        rows = int(channels ** 0.5)
        cols = ceil(channels / rows)

        gs = GridSpec(rows, cols, wspace=0)
        fig: plt.Figure = plt.gcf()
        fig.set_figheight(self.data.shape[0] / 60 * rows * scale)
        fig.set_figwidth(self.data.shape[1] / 60 * cols * scale)

        for i in range(channels):
            ax = plt.subplot(gs[i])
            ax.set_title("Index " + str(i), loc='left')
            ax.axis('off')
            ax: plt.Axes
            yield ax, self.data[..., i]

    def image(self, scale=1, colormap='magma'):
        for ax, d in self._create_grid(scale):
            ax.imshow(d, cmap=colormap)

    def hist(self, scale=1, bins=50):
        for ax, d in self._create_grid(scale):
            ax.hist(d.flatten(), bins=bins)

    def kde(self, scale=1, smoothing=0.5):
        for ax, d in self._create_grid(scale):
            sns.kdeplot(d.flatten(), ax=ax, bw_adjust=smoothing)


class _Frame2DPlot:
    data: np.ndarray

    def plot(self, ixs: Iterable or slice or None = None) -> Frame2DPlot:
        """ Gets a plot object. Note that you need to call a plot function to plot """
        if ixs:
            return Frame2DPlot(self.data[..., ixs])
        else:
            return Frame2DPlot(self.data)
