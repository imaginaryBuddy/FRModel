from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_completeness_v_measure
from sklearn.preprocessing import scale
import pandas as pd
import seaborn as sns
import numpy as np

from frmodel.base.D2 import Frame2D
from frmodel.base.D2.kmeans2D import KMeans2D

def kmeans_scoring_12122020(test_path: str,
                            score_path: str,
                            channels: dict = None,
                            grouping: str = "PREDICT",
                            color: str = "ACTUAL",
                            img_scale: float = 0.5,
                            clusters_mnl: int = 3,
                            clusters_mnf: int = 5,
                            scatter_size: float = 1.0,
                            verbose: bool = True):

    """ Runs the KMeans model developed at 12/12/2020

    channels can be passed as a dictionary, whereby it's similar to how you would call get_chns

    E.g. for f.get_chns(xy=True, hsv=True, glcm_con=True),
    you'd pass channels=dict(xy=True, hsv=True, glcm_con=True)

    grouping and color only accept these following values:
    'PREDICT', 'ACTUAL', 'SELECTED'

    :param test_path: Path to the test image
    :param score_path: Path to the scoring image
    :param channels: The channels to get, See Description on how to pass argument.
    :param grouping: The categorical grouping of the plots. i.e. how to create subplots.
        See Description on allowable values
    :param color: The categorical color/hue.
        See Description on allowable values
    :param img_scale: The scaling of the test/score loaded in
    :param clusters_mnl: Clusters to use for Meaningless Clustering
    :param clusters_mnf: Clusters to use for Meaningful Clustering
    :param scatter_size: Scatter size of plot
    :param verbose: Whether to output into console the details
    :return:
    """

    """ MEANINGLESS CLASSIFICATION
    
    Here, we start off with the meaningless classification.
    Some acronym pre/suffixes:
    MNL: Meaningless, MNF: Meaningful
    
    In this part, we're concerned about removing the meaningless labels so that the clustering
    is more meaningful in a way.    
    """

    assert grouping in ('PREDICT', 'ACTUAL', 'SELECTED', None), "Invalid grouping, see description on allowable values."
    assert color in ('PREDICT', 'ACTUAL', 'SELECTED', None), "Invalid grouping, see description on allowable values."

    # We load the Frames here and run the KMeans Directly on it.
    # Note that the KMeans is being run on the RGB Channels only, we may change this later

    predict = Frame2D.from_image(test_path, scale=img_scale)
    predict_rgb = predict.data_rgb()
    actual = Frame2D.from_image(score_path, scale=img_scale)

    if channels:
        predict = predict.get_chns(**channels)

        # If there are any GLCM channels, we have to crop it.
        if any([k in ("glcm_con", "glcm_cor", "glcm_ent") for k in channels.keys()]):
            if 'glcm_radius' not in channels.keys():
                raise Exception("glcm_radius must be explicitly specified on glcm features")
            predict_rgb = predict_rgb.convolute(channels['glcm_radius'], method='average')
            actual = actual.crop_glcm(glcm_radius=channels['glcm_radius'])

    fit_indexes = list(range(predict.shape[-1]))

    # Predict using KMeans
    predict_km_mnl = KMeans2D(predict_rgb,
                              KMeans(clusters_mnl, verbose=verbose),
                              fit_to=list(range(3)),
                              scaler=scale)

    # Score the prediction
    # The labels are in 1D, we reshape it to recreate the channels
    score_mnl = Frame2D.scorer_pair(predict_km_mnl.model.labels_, actual)['labels']\
                   .reshape([-1, 3])  # Reshape label prediction to PRED, ACT, COUNT

    # We retrieve the xy using predict or actual, then stack it onto the score
    score_mnl_xy = predict.get_xy()[0].reshape([-1, 2])
    score_mnl = np.hstack([score_mnl, score_mnl_xy])

    # Create DataFrame for lmplot
    score_mnl_df = pd.DataFrame(score_mnl, columns=('PREDICT', 'ACTUAL', 'SELECTED', 'X', 'Y'))

    # Call lmplot
    fig_mnl = sns.lmplot('X', 'Y',
                         data=score_mnl_df,
                         fit_reg=False,  # Don't render regression
                         col=grouping,  # Group By Predict
                         hue=color,
                         col_wrap=3,  # Wrap around on 3 column plots
                         scatter_kws={'s': scatter_size},
                         legend=True,
                         aspect=predict.width() / predict.height(),
                         legend_out=True)  # Scatter Size

    """ MEANINGLESS CLASSIFICATION DETERMINANT
    
    This is the algorithm to determine the cluster that is the least meaningful.
    
    If there's too many clusters, this wouldn't work well as depicted in the paper.
    
    This will only rid off the least meaningful one, hence it'll fail on >1 MNL cluster
    """

    # Contains the meaningless cluster number as labelled by KMeans
    ix_mnl: int = np.mean(predict_km_mnl.model.cluster_centers_, axis=1).argmin()

    # Contains the mask [XY] where you can mask against np.ndarrays
    # noinspection PyTypeChecker
    mask_mnl: np.ndarray = predict_km_mnl.model.labels_ != ix_mnl

    """ MEANINGFUL CLASSIFICATION
    
    For this part, we remove the MNL Cluster and perform another KMeans on it.
    """

    predict_km_mnf =\
        KMeans2D(predict,
                 KMeans(clusters_mnf, verbose=True),
                 fit_to=fit_indexes,
                 frame_1dmask=mask_mnl,
                 scaler=scale)

    # Contains the Label in 1D
    score_mnf = Frame2D.scorer_pair(predict_km_mnf.model.labels_,
                                    actual.data_flatten_xy()[predict_km_mnf.frame_1dmask, 0])['labels']

    # We retrieve the xy again, but we need to mask it since we removed the MNL cluster
    score_mnf_xy = predict.get_xy()[0].reshape([-1, 2])[mask_mnl, :]
    score_mnf = np.hstack([score_mnf, score_mnf_xy])

    # Create DataFrame for lmplot
    score_mnf_df = pd.DataFrame(score_mnf,
                                columns=('PREDICT', 'ACTUAL', 'SELECTED', 'X', 'Y'))

    # Call lmplot
    fig_mnf = sns.lmplot('X', 'Y', data=score_mnf_df,
                         fit_reg=False,  # Don't render regression
                         col='PREDICT',  # Group By Predict
                         hue='ACTUAL',
                         col_wrap=3,  # Wrap around on 3 column plots
                         scatter_kws={'s': scatter_size},
                         aspect=predict.width() / predict.height(),
                         legend=False)  # Scatter Size

    # Return both Figures, Score and the detected MNL Cluster
    return dict(fig_mnl=fig_mnl,
                fig_mnf=fig_mnf,
                score_mnl=homogeneity_completeness_v_measure(score_mnl_df.ACTUAL,
                                                             score_mnl_df.PREDICT),
                score_mnf=homogeneity_completeness_v_measure(score_mnf_df.ACTUAL,
                                                             score_mnf_df.PREDICT),
                ix_mnl=ix_mnl)


