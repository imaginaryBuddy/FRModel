from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize as sk_normalize
import matplotlib.pyplot as plt
from typing import Tuple


class _Frame2DKmeans(ABC):

    @abstractmethod
    def data_flatten(self): ...

    def kmeans(self,
               clusters,
               fit_indexes,
               sample_weight=None,
               verbose=False,
               scaler=sk_normalize,
               plot_figure=False,
               xy_indexes=(3 ,4),
               scatter_size=0.2
               ) -> KMeans or Tuple[KMeans, plt.Figure]:
        """ Fits a KMeans on current frame, on certain axes

        Example::

            frame_idxs = frame.get_idxs(xy=True,hsv=True,veg=True)
            frame_idxs.kmeans(clusters=5, fit_indexes=[2,3,4,5],
                              plot_figure=True,xy_indexes=(0,1),scatter_size=1)

        This will firstly get the coordinate, hsv and veg indexes, placing it in a df

        :param clusters: The number of clusters
        :param fit_indexes: The indexes to .fit()
        :param sample_weight: The sample weight for each record, if any. Can be None.
        :param verbose: Whether to print out the KMeans' verbose log
        :param scaler: The scaler to use, must be a callable(np.ndarray)
        :param plot_figure: Whether to plot a figure or not, using plt.gcf()
        :param xy_indexes: The indexes of XY. Must be present for plotting.
        :param scatter_size: Size of the marker on the scatter plot
        :return:
        """
        flat = self.data_flatten()
        frame_xy_trans = scaler(flat[:, fit_indexes])
        km = KMeans(n_clusters=clusters, verbose=verbose) \
            .fit(frame_xy_trans,
                 sample_weight=frame_xy_trans[:, sample_weight] if np.all(sample_weight) else None)
        if plot_figure:
            df = pd.DataFrame(np.append(flat, km.labels_[... ,np.newaxis] ,axis=-1))
            df.columns = [f"c{e}" for e, _ in enumerate(df.columns)]
            fg = sns.lmplot(data=df,
                            x=f'c{xy_indexes[0]}',
                            y=f'c{xy_indexes[1]}',
                            hue=df.columns[-1],
                            fit_reg=False,
                            legend=True,
                            legend_out=True,
                            scatter_kws={"s": scatter_size})
            fg.ax.set_aspect('equal')
            fg.ax.invert_yaxis()
            return km, fg
        return km
