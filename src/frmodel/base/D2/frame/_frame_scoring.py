from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from warnings import warn

import numpy as np
from PIL import Image
from sklearn.metrics import homogeneity_completeness_v_measure
from sklearn.preprocessing import LabelEncoder

if TYPE_CHECKING:
    from frmodel.base.D2.frame2D import Frame2D


class _Frame2DScoring(ABC):

    data: np.ndarray

    # noinspection PyArgumentList
    @classmethod
    def init(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    @abstractmethod
    def crop(self,
             top:int = 0,
             right:int = 0,
             bottom:int = 0,
             left:int = 0) -> _Frame2DScoring:
        ...

    @staticmethod
    def labelize(ar:np.ndarray) -> np.ndarray:
        """ Labelizes the np.ndarray.

        This is used to make distinct values into categorical like value.

        E.g. [[0, 10, 20], [10, 20, 34]]
        labelizes to
        [[0, 1, 2], [1, 2, 3]].

        Note that if your array is >1D, all points of data will be treated equally due
        to flattening and the shape is preserved.

        That is, [[A, B], [B, C]] will labelize as
        [[0, 1], [1, 2]] instead of [0, 1].

        :param ar: The array to labelize
        :returns: An np.ndarray of the same shape
        """

        if ar.ndim >= 3: warn(f"Your data with {ar.ndim} dimensions will be flattened entirely,"
                              f"labelling all data points as equal! Recommend to use only ndim 2")

        return LabelEncoder().fit_transform(np.round(ar).astype(int).flatten()).reshape(ar.shape)


    @staticmethod
    def scorer(x: 'Frame2D',
               y: 'Frame2D',
               x_label_ix: int = 0):
        """ A custom scoring algorithm.

        Note that these parameters are non-reversible,
        that is scoring x against y is not the same as y against x.

        y_frame, should be loaded in as RGB as it'll go through an RGB
        distinct flatten, where every RGB value will be labelled distinctly.

        :param x: Actual Labels
        :param y: Predict Frame
        :param x_label_ix: The label index of x
        :returns:
        """

        assert x.shape == y.shape, "x's Shape must match y's Shape. If x used GLCM, you need to crop_glcm on y to fix" \
                                   "its shape"

        x = x.data_rgb_flatten()
        y = y.data_rgb_flatten()

        data = np.stack([_Frame2DScoring.labelize(x.flatten()),
                         _Frame2DScoring.labelize(y.flatten())])

        pass




    def score(self, score_frame: 'Frame2D', label_ix: int = -1, glcm_radius=None):
        """ Scores the current frame kmeans with a scoring image

        :param label_ix: The label index to score against score_frame
        :param score_frame: The score as Frame2D
        :param glcm_radius: The radius of GLCM used if applicable. This will crop the Frame2D automatically to fit.
        :return: A Dictionary of various scoring algorithm results
        """
        # Convert grayscale to labels
        if glcm_radius is not None: score_frame = score_frame.crop_glcm(glcm_radius)
        true = self.labelize(score_frame.data[...,0]).flatten()
        pred = self.data[..., label_ix].flatten()

        score = self.score_custom(true, pred), *homogeneity_completeness_v_measure(true, pred)
        return {"Custom":       score[0],
                "Homogeneity":  score[1],
                "Completeness": score[2],
                "V Measure":    score[3]}

    def _pair_frames(self, score_frame: 'Frame2D'):
        """"""

        ...


    @staticmethod
    def score_custom(true_labels: np.ndarray,
                     pred_labels: np.ndarray):
        """ Scores the current Kmeans model with a scoring image

        This is a custom algorithm.

        This attempts to find the best label to prediction mapping and returns that maximum
        score.

        :param pred_labels: Predicted Labels
        :param true_labels: Actual Labels
        :return: Score out of 1.
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
