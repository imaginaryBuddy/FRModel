from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from PIL import Image
from scipy.stats import rankdata
from seaborn import FacetGrid

from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import homogeneity_completeness_v_measure

if TYPE_CHECKING:
    from frmodel.base.D2 import Frame2D


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

    def score(self, score_frame: Frame2D, exclude_0=True, glcm_radius=None):
        """ Scores the current frame kmeans with a scoring image

        :param score_frame: The score as Frame2D
        :param exclude_0: Excludes the first label from calculation, that is the lowest grayscale value
        :param glcm_radius: The radius of GLCM used if applicable. This will crop the Frame2D automatically to fit.
        :return:
        """
        # Convert grayscale to labels
        true = self._frame_as_label(score_frame)

        if glcm_radius is not None:
            true = true[glcm_radius+1:-glcm_radius, glcm_radius+1:-glcm_radius]

        # Convert Image Grayscale (0-255) to quantized rank
        # This will only work if all Grayscale values are unique.
        true = rankdata(true.flatten(), method='dense') - 1
        pred = self.model.labels_

        if exclude_0:
            mask = true == 0
            true = np.ma.masked_array(true, mask=mask)
            pred = pred[~true.mask]
            true = true[~true.mask]

        score = self.score_map(true, pred), *homogeneity_completeness_v_measure(true, pred)
        return {"Custom": score[0],
                "Homogeneity": score[1],
                "Completeness": score[2],
                "V Measure": score[3]}

    @staticmethod
    def _frame_as_label(frame: Frame2D):
        return (rankdata(frame.data[..., 0].flatten(), method='dense') - 1).reshape(frame.shape[0:2])

    @staticmethod
    def score_map(true_labels: np.ndarray,
                  pred_labels: np.ndarray):
        """ Scores the current Kmeans model with a scoring image

        This is a custom algorithm.

        This attempts to find the best label to prediction mapping and returns that maximum
        score.

        :param pred_labels: Predicted Labels
        :param true_labels: Actual Labels
        :return: Count Array, Score out of 1.
        """

        # Count each unique pair occurrence and return count.
        # Because return_count returns separately, we vstack it
        # Then we transpose the data for iterrows() op
        ar = \
            np.vstack(
                np.unique(axis=1, return_counts=True,
                          ar=np.vstack([true_labels, pred_labels]))).transpose()

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
        return np.sum(ar[:, -1]) / pred_labels.size



