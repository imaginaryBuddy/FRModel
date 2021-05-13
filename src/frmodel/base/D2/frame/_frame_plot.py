from dataclasses import dataclass
from math import ceil
from typing import List
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
import seaborn as sns
from matplotlib.gridspec import GridSpec

from frmodel.base import CONSTS

if TYPE_CHECKING:
    from frmodel.base.D2 import Frame2D

@dataclass
class Frame2DPlot:

    f: 'Frame2D'
    subplot_shape: tuple = None
    titles: list = None

    def _create_grid(self,
                     scale: float = 1.0):
        """ Facilitates in create a subplot grid for plotting functions.

        :param scale: Scale of the plot
        :returns: An Axes reference generator
        """
        channels = self.f.data.shape[-1]
        if self.subplot_shape is None:
            rows = ceil(channels ** 0.5)
            cols = ceil(channels / rows)
        else:
            rows = self.subplot_shape[0]
            cols = self.subplot_shape[1]

        gs = GridSpec(rows, cols, wspace=0)
        fig: plt.Figure = plt.gcf()
        fig.set_figheight(int(self.f.data.shape[0] / 60 * rows * scale))
        fig.set_figwidth(int(self.f.data.shape[1] / 60 * cols * scale))

        titles = self.titles if self.titles else self.f.labels

        assert len(titles) == channels, "Title Length must be same as number of Channels"

        for i, t in enumerate(titles):
            ax = plt.subplot(gs[i])
            if channels != 1:
                ax.set_title(t, loc='left')
            ax.axis('off')
            ax.legend_ = None
            ax: plt.Axes
            yield ax, self.f.data[..., i]

    @staticmethod
    def set_browser_plotting():
        """ Makes Plotly render on browser by changing the flag. """
        pio.renderers.default = "browser"

    def image(self,
              scale: float = 1,
              colormap: str = 'magma'):
        """ For each index, create a separate subplot imshow.

        :param scale: Scale of the subplots
        :param colormap: The cmap of imshow. See plt.imshow for available cmaps.
        :returns: A plt.Figure
        """
        for ax, d in self._create_grid(scale):
            ax.imshow(d, cmap=colormap, origin='upper')
        return plt.gcf()

    def hist(self, scale=1, bins=50):
        """ For each index, create a separate subplot hist.

        :param scale: Scale of the subplots
        :param bins: Number of bins to pass into hist
        :returns: A plt.Figure
        """
        for ax, d in self._create_grid(scale):
            ax.hist(d.flatten(), bins=bins)
        return plt.gcf()

    def kde(self, scale=1, smoothing=0.5):
        """ For each index, create a separate subplot hist.

        Note: smoothing may not work on some versions of seaborn.

        :param scale: Scale of the subplots
        :param smoothing: The amount of smoothing to apply to the KDE
        :returns: A plt.Figure
        """
        for ax, d in self._create_grid(scale):
            sns.kdeplot(d.flatten(), ax=ax, bw_adjust=smoothing)
        return plt.gcf()

    def surface3d(self,
                  chn: CONSTS.CHN,
                  nan_value: float = 0):
        """ Plots a surface 3d plot on Plotly

        :param chn: The channel to plot as height
        :param nan_value: The value to replace NaNs"""

        g = self.f.data_chn(chn).data
        g[np.isnan(g)] = nan_value

        return go.Figure(data=[
            go.Surface(z=g[..., 0]),
        ])

    def scatter3d(self,
                  chn: CONSTS.CHN,
                  colored: bool = False,
                  z_scale:int = 1,
                  point_size:float = 7,
                  sample_size:int or None = 10000,
                  colorscale=px.colors.sequential.Viridis
                  ):
        """ Plot a single index with respect to its X and Y.

        :param chn: A single channel.
        :param colored: Whether to have the point cloud coloured with RGB channels. Only works if RGB exists.
        :param z_scale: The scale of the Z Axis. If lower than 1, it'll look flatter, vice versa.
        :param point_size: The size of the markers.
        :param sample_size: The amount of points to sample. If None, use all points
        :param colorscale: The color scale to use when plotting.
        :returns: A plt.Figure
        """
        if colored:
            d = self.f.get_chns(self_=False, chns=[self.f.CHN.XY, chn, self.f.CHN.RGB]).data_flatten_xy()
        else:
            d = self.f.get_chns(self_=False, chns=[self.f.CHN.XY, chn]).data_flatten_xy()

        if sample_size:
            d = d[np.random.choice(d.shape[0], replace=False, size=sample_size)]

        # Remove NaN Cases
        d = d[~np.any(np.isnan(d), axis=1), ...]

        data = [
            go.Scatter3d(
                x=d[..., 0],
                y=d[..., 1],
                z=d[..., 2],
                mode='markers',

                marker=dict(size=np.ones(d.shape[0]) * point_size,
                            line=dict(width=0),
                            color=[f'rgb({int(r[3])},{int(r[4])},{int(r[5])})' for r in d] if colored else d[..., 2],
                            colorscale=colorscale),
            )
        ]

        layout = go.Layout(
            scene=dict(xaxis={'title': 'x'},
                       yaxis={'title': 'y'},
                       zaxis={'title': 'z'},
                       aspectratio=dict(x=1, y=1, z=z_scale)),
            margin={'l': 60, 'b': 40, 't': 10, 'r': 10},
            legend={'x': 0, 'y': 1},
            hovermode='closest'
        )

        fig = go.Figure(data=data, layout=layout)
        return fig

class _Frame2DPlot:
    data: np.ndarray

    def plot(self: 'Frame2D', labels: str or List[str] = None) -> Frame2DPlot:
        """ Gets a plot object. Note that you need to call a plot function to plot.

        :param labels: The labels to plot with.
        """

        return Frame2DPlot(self.create(data=self.data_chn(labels).data, labels=labels) if labels else self)
