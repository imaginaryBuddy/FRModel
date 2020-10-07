from __future__ import annotations

from dataclasses import dataclass
from seaborn import FacetGrid

from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import normalize


@dataclass
class KMeans2D:

    model: KMeans
    data: np.ndarray

    def fit(self,
            fit_indexes,
            sample_weight=None,
            scaler=normalize) -> KMeans:
        """ Fits a KMeans on current frame, on certain axes

        Example::

            frame_idxs = frame.get_idxs(xy=True,hsv=True,veg=True)
            frame_idxs.kmeans(clusters=5, fit_indexes=[2,3,4,5],
                              plot_figure=True,xy_indexes=(0,1),scatter_size=1)

        This will firstly get the coordinate, hsv and veg indexes, placing it in a df

        :param fit_indexes: The indexes to .fit()
        :param sample_weight: The sample weight for each record, if any. Can be None.
        :param scaler: The scaler to use, must be a callable(np.ndarray)
        :return:
        """
        frame_xy_trans = scaler(self.data[:, fit_indexes])
        km_est = self.model.fit(frame_xy_trans,
                                sample_weight=frame_xy_trans[:, sample_weight] if np.all(sample_weight) else None)
        return km_est

    def plot(self,
             km_est: KMeans,
             xy_indexes=(3 ,4),
             scatter_size=0.2) -> FacetGrid:
        df = pd.DataFrame(np.append(self.data, km_est.labels_[..., np.newaxis], axis=-1))
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

        return fg


