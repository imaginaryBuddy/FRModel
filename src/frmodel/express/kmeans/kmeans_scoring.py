from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_completeness_v_measure
from sklearn.preprocessing import scale
import pandas as pd
import seaborn as sns
import numpy as np

from frmodel.base.D2 import Frame2D
from frmodel.base.D2.kmeans2D import KMeans2D

def kmeans_scoring_005(map_path: str,
                       test_path: str,
                       grouping: str = "PREDICT",
                       color: str = "ACTUAL",
                       img_scale: float = 0.5,
                       clusters_mnl: int = 3,
                       clusters_mnf: int = 5,

                       verbose: bool = True):
    """ Runs the KMeans model developed at 12/12/2020 """

    """ We load the Frames here and run the KMeans Directly on it.
    
    Note that the KMeans is being run on the RGB Channels only, we may change this later
    """
    predict = Frame2D.from_image(map_path, scale=img_scale)
    actual = Frame2D.from_image(test_path, scale=img_scale)

    # Predict using KMeans
    predict_km_mnl = KMeans2D(predict,
                              KMeans(clusters_mnl, verbose=verbose), [0, 1, 2],
                              scaler=scale)

    # Score the prediction
    score_mnl = predict.scorer(predict_km_mnl.model.labels_, actual)['labels']\
                   .reshape([*predict.shape[0:-1], -1])  # Reshape label prediction to original shape

    # Convert Score to Frame2D
    # Grab XY to assist in sns.lmplot
    score_frame = Frame2D(score_mnl,
                          labels=["PREDICT", "ACTUAL", "SELECTED"]).get_chns(self_=True, xy=True)

    # Flatten on XY, retaining the XY columns
    score_mnl = score_frame.data_flatten_xy()

    # Create DataFrame for lmplot
    score_mnl_df = pd.DataFrame(score_mnl, columns=('PREDICT', 'ACTUAL', 'SELECTED', 'X', 'Y'))

    # Call lmplot
    fig_mnl = sns.lmplot('X', 'Y',
                         data=score_mnl_df,
                         fit_reg=False,  # Don't render regression
                         col=grouping,  # Group By Predict
                         hue=color,
                         col_wrap=3,  # Wrap around on 3 column plots
                         scatter_kws={'s': 1},
                         legend=True,
                         legend_out=True)  # Scatter Size

    centers_mnl = np.mean(predict_km_mnl.model.cluster_centers_, axis=1)
    ix_mnl = centers_mnl.argmin()

    predict_km_mnf =\
        KMeans2D(predict, KMeans(clusters_mnf, verbose=True), [0, 1, 2],
                 frame_1dmask=predict_km_mnl.model.labels_ != ix_mnl,
                 scaler=scale)

    frame_mnf = predict_km_mnf.frame_masked()

    score_mnf = Frame2D.scorer(predict_km_mnf.model.labels_,
                               actual.data_flatten_xy()[predict_km_mnf.frame_1dmask, 0])

    score_mnf = np.hstack([score_mnf['labels'], frame_mnf[:, -2:]])

    # Create DataFrame for lmplot
    score_mnf_df = pd.DataFrame(score_mnf,
                                columns=('PREDICT', 'ACTUAL', 'SELECTED', 'X', 'Y'))

    # Call lmplot
    fig_mnf = sns.lmplot('X', 'Y', data=score_mnf_df,
                     fit_reg=False,  # Don't render regression
                     col='PREDICT',  # Group By Predict
                     hue='ACTUAL',
                     col_wrap=3,  # Wrap around on 3 column plots
                     scatter_kws={'s': 1},
                     legend=False)  # Scatter Size

    return dict(fig_mnl=fig_mnl,
                fig_mnf=fig_mnf)


