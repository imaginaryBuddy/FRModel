from __future__ import annotations

from dataclasses import dataclass

from PIL import Image
from scipy.stats import rankdata
from seaborn import FacetGrid

from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import seaborn as sns


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
        """ Scores the current Kmeans model with a scoring image

        :param score_file_path: The file path to the scoring image.
        :return: Count Array, Score out of 1.
        """

        # Convert to array
        im: Image.Image = Image.open(score_file_path).convert('LA')
        ar = np.asarray(im)[..., 0]

        # Convert Image Grayscale (0-255) to quantized rank
        # This will only work if all Grayscale values are unique.
        ar = rankdata(ar, method='dense') - 1

        # Count each unique pair occurrence and return count.
        # Because return_count returns separately, we vstack it
        # Then we transpose the data for iterrows() op
        ar = \
            np.vstack(
                np.unique(axis=1, return_counts=True,
                          ar=np.vstack([self.model.labels_, ar]))).transpose()

        # This sorts by the last column (Counts)
        ar: np.ndarray = ar[ar[:, -1].argsort()[::-1]]

        # There's no simple way to get the maximum unique of 2 dimensions I believe
        # We'll loop through the cells using a naive approach
        # This approach is naive because if we were to permutate all possible
        # combinations, we'll end up with a really large list.
        # This is not ideal if we want to scale this up for more trees
        # However, it's not a hard limitation.

        # We have the following array structure
        # PREDICT ACTUAL COUNT
        # The catch is that predict and actual cannot appear more than once.

        visited_pred = []
        visited_act = []
        counts = []
        for r in ar:
            if r[0] in visited_pred or r[1] in visited_act:
                continue
            else:
                visited_pred.append(int(r[0]))
                visited_act.append(int(r[1]))
                counts.append(r)

        ar = np.asarray(counts)
        return ar, np.sum(ar[:, -1]) / (im.size[0] * im.size[1])


