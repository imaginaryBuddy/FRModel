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

    def plot(self,
             xy_indexes=(3 ,4),
             scatter_size=0.2) -> FacetGrid:
        """ Generates a plot with fitted KMeans

        Implicitly set 1:1 ratio plotting
        Implicitly inverts y-axis

        :param xy_indexes: The indexes of X & Y for plotting
        :param scatter_size: Size of marker
        :return: A FacetGrid
        """
        df = pd.DataFrame(np.append(self.data, self.model.labels_[..., np.newaxis], axis=-1))
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

    def score(self,
              score_file_path: str):
        """ Scores the current Kmeans fit

        :rtype: object
        """

