from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.stats
from matplotlib import pyplot as plt
from scipy.stats import moment
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import scale, minmax_scale

from frmodel.base.D2 import Frame2D

@dataclass
class Classification:
    m_label: np.ndarray
    d_label: np.ndarray
    m_fake_label: np.ndarray
    d_fake_label: np.ndarray
    m_scaled: np.ma.MaskedArray
    d_scaled: np.ma.MaskedArray
    m_part: tuple
    d_part: tuple
    m_fake_part: tuple
    d_fake_part: tuple

    MOMENTS = 4
    PARTS = 5
    TRAIN_PARTS = 3
    TEST_PARTS = PARTS - TRAIN_PARTS
    FEATURES = 30
    EXPECTED = [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12, 12, 13, 14]

    def __init__(self):
        print("Loading Frame")
        self.m = Frame2D.load("rsc/imgs/spec/chestnut_10May_90deg43m85pct255deg/map/result_band_scale.npz").data
        self.d = Frame2D.load("rsc/imgs/spec/chestnut_18Dec/result_band_scale.npz").data
        print("Getting Labels")
        self.m_label = self.get_labels("rsc/imgs/spec/chestnut_10May_90deg43m85pct255deg/Labelling.csv")
        self.d_label = self.get_labels("rsc/imgs/spec/chestnut_18Dec/Labelling.csv")
        print("Creating Fake Labels")
        self.m_fake_label = self.get_fake_labels(self.m_label)
        self.d_fake_label = self.get_fake_labels(self.d_label)
        self.repartition()
        self.repartition_fake()

    def repartition(self):
        print("Partitioning And Creating Moments")
        self.m_part = self.partition_and_moment(self.m, self.m_label,
                                                self.FEATURES, self.MOMENTS, self.PARTS, self.TEST_PARTS)
        self.d_part = self.partition_and_moment(self.d, self.d_label,
                                                self.FEATURES, self.MOMENTS, self.PARTS,self.TEST_PARTS)
    def repartition_fake(self):
        print("Fake Partitioning And Creating Moments")
        self.m_fake_part = self.partition_and_moment(self.m, self.m_label,
                                                     self.FEATURES, self.MOMENTS, self.PARTS, self.TEST_PARTS)
        self.d_fake_part = self.partition_and_moment(self.d, self.d_label,
                                                     self.FEATURES, self.MOMENTS, self.PARTS, self.TEST_PARTS)

    def m_m(self, estimators, max_depth):
        train = self.m_part[0]
        test = self.m_part[1]
        return self.evaluate(train, test, estimators, max_depth, self.EXPECTED)
    def d_d(self, estimators, max_depth):
        train = self.d_part[0]
        test = self.d_part[1]
        return self.evaluate(train, test, estimators, max_depth, self.EXPECTED)
    def m_d(self, estimators, max_depth):
        train = np.vstack(self.m_part)
        test = np.vstack(self.d_part)
        return self.evaluate(train, test, estimators, max_depth, self.EXPECTED)
    def d_m(self, estimators, max_depth):
        train = np.vstack(self.d_part)
        test = np.vstack(self.m_part)
        return self.evaluate(train, test, estimators, max_depth, self.EXPECTED)
    def md_md(self, estimators, max_depth):
        train = np.vstack([self.m_part[0], self.d_part[0]])
        test = np.vstack([self.m_part[1], self.d_part[1]])
        return self.evaluate(train, test, estimators, max_depth, self.EXPECTED)
    def mc_mc(self, estimators, max_depth):
        train = self.m_fake_part[0]
        test = self.m_fake_part[1]
        return self.evaluate(train, test, estimators, max_depth, self.EXPECTED)
    def dc_dc(self, estimators, max_depth):
        train = self.d_fake_part[0]
        test = self.d_fake_part[1]
        return self.evaluate(train, test, estimators, max_depth, self.EXPECTED)
    def mdc_mdc(self, estimators, max_depth):
        train = np.vstack([self.m_fake_part[0], self.d_fake_part[0]])
        test = np.vstack([self.m_fake_part[1], self.d_fake_part[1]])
        return self.evaluate(train, test, estimators, max_depth, self.EXPECTED)

    @staticmethod
    def calculate_glcm(path="rsc/imgs/spec/chestnut_18Dec/", size_scale=0.5, scale_on_bands=False) -> Frame2D:
        f = Frame2D.from_image_spec(
            path + "result_Red.tif",
            path + "result_Green.tif",
            path + "result_Blue.tif",
            path + "result_NIR.tif",
            path + "result_RedEdge.tif",
            scale=size_scale
        )
        return f.get_chns(self_=True, glcm=f.GLCM(channels=f.CHN.RGBRENIR, scale_on_bands=scale_on_bands))

    @staticmethod
    def get_labels(path="rsc/imgs/spec/chestnut_18Dec/Labelling.csv") -> np.ndarray:
        labels = pd.read_csv(path,header=None)
        return (labels.iloc[:, 1:].to_numpy() * 0.5).astype(int)

    @staticmethod
    def get_fake_labels(original_labels,
                        X_BOUNDS: tuple = (500, 2000),
                        Y_BOUNDS: tuple = (500, 2500)) -> np.ndarray:
        y_sizes = original_labels[:, 1] - original_labels[:, 0]
        x_sizes = original_labels[:, 3] - original_labels[:, 2]

        fake_labels = np.zeros_like(original_labels)
        for i in range(len(y_sizes)):
            y_size = y_sizes[i]
            x_size = x_sizes[i]
            y_0 = np.random.randint(Y_BOUNDS[0], Y_BOUNDS[1] - y_size)
            x_0 = np.random.randint(X_BOUNDS[0], X_BOUNDS[1] - x_size)
            fake_labels[i] = [y_0, y_0 + y_size, x_0, x_0 + x_size]

        return fake_labels

    @staticmethod
    def rescale_glcm(f: Frame2D):
        f = scale(f.data_flatten_xy()).reshape(f.shape)
        return np.ma.masked_invalid(f)

    @staticmethod
    def partition_and_moment(ar: np.ndarray, labels: np.ndarray,
                             FEATURES, MOMENTS, PARTS, TEST_PARTS):
        TRAIN_PARTS = PARTS - TEST_PARTS
        train = np.zeros([len(labels), TRAIN_PARTS, FEATURES * MOMENTS])
        test = np.zeros([len(labels), TEST_PARTS, FEATURES * MOMENTS])

        for e, i in enumerate(labels):
            window = ar[i[0]:i[1], i[2]:i[3]]

            window = window.reshape([-1, window.shape[-1]])
            window = scale(window)
            np.random.shuffle(window)
            parts = np.array_split(window, PARTS)

            for j in range(TRAIN_PARTS):
                train[e, j, FEATURES * 0:FEATURES * 1] = np.nanmean(parts[j], axis=0)
                train[e, j, FEATURES * 1:FEATURES * 2] = np.nanvar(parts[j], axis=0)
                train[e, j, FEATURES * 2:FEATURES * 3] = scipy.stats.skew(parts[j], nan_policy='omit')
                train[e, j, FEATURES * 3:FEATURES * 4] = scipy.stats.kurtosis(parts[j], nan_policy='omit')
            for j in range(TEST_PARTS):
                test[e, j, FEATURES * 0:FEATURES * 1] = np.nanmean(parts[j], axis=0)
                test[e, j, FEATURES * 1:FEATURES * 2] = np.nanvar(parts[j], axis=0)
                test[e, j, FEATURES * 2:FEATURES * 3] = scipy.stats.skew(parts[j], nan_policy='omit')
                test[e, j, FEATURES * 3:FEATURES * 4] = scipy.stats.kurtosis(parts[j], nan_policy='omit')

        # Note to swapaxes to align it correctly
        train = train.swapaxes(0, 1)
        test = test.swapaxes(0, 1)
        train = train.reshape([-1, train.shape[-1]])
        test = test.reshape([-1, test.shape[-1]])

        return scale(train), scale(test)

    @staticmethod
    def evaluate(train, test, estimators, max_depth, EXPECTED, random_state=0):
        clf = RandomForestClassifier(n_estimators=estimators, max_depth=max_depth, random_state=random_state)
        clf.fit(train, EXPECTED * (train.shape[0] // len(EXPECTED)))
        pred = clf.predict(test)

        pred = np.asarray(pred)
        actual = np.asarray(EXPECTED * (test.shape[0] // len(EXPECTED)))
        accuracy = 1 - (np.count_nonzero(pred - actual) / len(pred))
        return accuracy, pred, actual

    @staticmethod
    def plot_windows(labels, ar):
        plt.figure(figsize=(20, 10))
        for j in range(18):
            plt.subplot(3, 6, j + 1)
            i = labels[j]
            OFFSET = 100
            g = ar[i[0] - OFFSET:i[1] + OFFSET, i[2] - OFFSET:i[3] + OFFSET][..., :3]
            plt.imshow(minmax_scale(g.flatten()).reshape(g.shape))
            plt.axis('off')
            plt.subplots_adjust(0, 0, 1, 1, 0, 0)
        plt.tight_layout()
        plt.show()
