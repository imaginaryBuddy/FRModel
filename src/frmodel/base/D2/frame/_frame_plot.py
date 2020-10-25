from dataclasses import dataclass
from math import ceil
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import seaborn as sns

import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from frmodel.base.D2 import Frame2D

pio.renderers.default = "browser"

@dataclass
class Frame2DPlot:

    f: 'Frame2D'

    def _create_grid(self, scale):
        channels = self.f.data.shape[-1]
        rows = int(channels ** 0.5)
        cols = ceil(channels / rows)

        gs = GridSpec(rows, cols, wspace=0)
        fig: plt.Figure = plt.gcf()
        fig.set_figheight(self.f.data.shape[0] / 60 * rows * scale)
        fig.set_figwidth(self.f.data.shape[1] / 60 * cols * scale)

        for i in range(channels):
            ax = plt.subplot(gs[i])
            if channels != 1:
                ax.set_title("Index " + str(i), loc='left')
            ax.axis('off')
            ax.legend_ = None
            ax: plt.Axes
            yield ax, self.f.data[..., i]

    def set_browser_plotting(self):
        pio.renderers.default = "browser"

    def image(self, scale=1, colormap='magma'):
        for ax, d in self._create_grid(scale):
            ax.imshow(d, cmap=colormap)
        return plt.gcf()

    def hist(self, scale=1, bins=50):
        for ax, d in self._create_grid(scale):
            ax.hist(d.flatten(), bins=bins)
        return plt.gcf()

    def kde(self, scale=1, smoothing=0.5):
        for ax, d in self._create_grid(scale):
            sns.kdeplot(d.flatten(), ax=ax, bw_adjust=smoothing)
        return plt.gcf()

    def image3d(self, ix, sample_size=50000):
        d = self.f.get_chns(self_=True,xy=True).data_flatten_xy()
        d = d[np.random.choice(d.shape[0], replace=False, size=sample_size)]

        data = [
            go.Scatter3d(
                x=d[..., -2],
                y=d[..., -1],
                z=d[..., ix],
                mode='markers',

                marker=dict(size=np.ones(d.shape[0]) * 7,
                            line=dict(width=0),
                            color=d[..., ix],
                            colorscale=px.colors.sequential.Viridis),
            )
        ]

        layout = go.Layout(
            scene=dict(xaxis={'title': 'x'},
                       yaxis={'title': 'y'},
                       zaxis={'title': 'z'},
                       aspectratio=dict(x=1, y=1, z=0.2)),
            margin={'l': 60, 'b': 40, 't': 10, 'r': 10},
            legend={'x': 0, 'y': 1},
            hovermode='closest'
        )

        fig = go.Figure(data=data, layout=layout)
        fig.show()

class _Frame2DPlot:
    data: np.ndarray

    # noinspection PyArgumentList
    @classmethod
    def init(cls, *args, **kwargs) -> 'Frame2D':
        return cls(*args, **kwargs)

    def plot(self, ixs: Iterable or slice or None = None) -> Frame2DPlot:
        """ Gets a plot object. Note that you need to call a plot function to plot """
        if ixs:
            return Frame2DPlot(self.init(self.data[..., ixs]))
        else:
            return Frame2DPlot(self.init(self.data))
