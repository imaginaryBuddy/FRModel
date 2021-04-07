from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING
from warnings import warn

import numpy as np
from sklearn.metrics import homogeneity_completeness_v_measure
from sklearn.preprocessing import LabelEncoder

if TYPE_CHECKING:
    from frmodel.base.D2.frame2D import Frame2D

class _Frame2DScoring(ABC):

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
    def scorer_pair(predict: np.ndarray,
                    actual: 'Frame2D' or np.ndarray):
        """ A custom scoring algorithm. Called by score

        Note that these parameters are non-reversible,
        that is scoring x against y is not the same as y against x.

        y_frame, should be loaded in as RGB as it'll go through an RGB
        distinct flatten, where every RGB value will be labelled distinctly.

        :param predict: Actual Labels. Must be a 1D label array
        :param actual: Predict Frame. Can be either Frame with RGB or 1D label array
        :returns: A Dictionary {score: float, score_pairs: np.ndarray of counts, labels: np.ndarray of labels}
        """
        predict:np.ndarray = predict.flatten()

        if not isinstance(actual, np.ndarray):
            actual: np.ndarray = _Frame2DScoring.labelize(actual.data_rgb_flatten()).flatten()

        assert predict.shape[0] == actual.shape[0],\
            "x's Size must match y's Size. If x used GLCM, you need to crop_glcm on y to fix its size"

        pairs = np.vstack([predict, actual]).transpose()
        unqpairs, counts = np.unique(pairs, axis=0, return_counts=True)

        unqpairs = np.hstack([unqpairs, counts[..., np.newaxis]])
        unqpairs = unqpairs[unqpairs[:,-1].argsort()[::-1]]

        # ------ Score algo

        visited_pred = []
        visited_act = []
        counts = []

        for r in unqpairs:
            if r[0] in visited_pred or r[1] in visited_act:
                continue
            else:
                visited_pred.append(r[0])
                visited_act.append(r[1])
                counts.append(r)

        counts_ar = np.asarray(counts)
        labels = np.zeros((pairs.shape[0], pairs.shape[1] + 1))
        labels[..., :-1] = pairs

        for i in range(counts_ar.shape[0]):
            labels[np.where((pairs == counts_ar[i, :-1]).all(axis=1)), -1] = 1

        return dict(score=np.sum(counts_ar[:, -1]) / actual.size,
                    score_pairs=counts_ar,
                    labels=labels)

    def score(self: 'Frame2D', score_frame: 'Frame2D', label_ix: int = -1, glcm_radius=None):
        """ Scores the current frame kmeans with a scoring image

        :param label_ix: The label index to score against score_frame
        :param score_frame: The score as Frame2D
        :param glcm_radius: The radius of GLCM used if applicable. This will crop the Frame2D automatically to fit.
        :return: A Dictionary of various scoring algorithm results,
            {'Custom', 'Homogeneity', 'Completeness', 'V Measure'}
        """
        # Convert grayscale to labels
        if glcm_radius is not None: score_frame = score_frame.crop_glcm(glcm_radius)
        true = self.labelize(score_frame.data[...,0]).flatten()
        pred = self.data[..., label_ix].flatten()

        score = self.scorer_pair(true, pred)['score'],\
                *homogeneity_completeness_v_measure(true, pred)
        return {"Custom":       score[0],
                "Homogeneity":  score[1],
                "Completeness": score[2],
                "V Measure":    score[3]}
