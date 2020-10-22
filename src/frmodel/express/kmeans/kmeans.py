from itertools import combinations
from sklearn.cluster import KMeans
from sklearn.preprocessing import minmax_scale
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.gui import tqdm

from frmodel.base.D2 import Frame2D
import os

from frmodel.base.D2.kmeans2D import KMeans2D


def kmeans_matrix(test_path: str,
                  score_path: str,
                  scale: float,
                  output_dir: str,
                  ixs_per_kmeans: int,
                  verbose=True,
                  clusters=6,
                  scatter_size=0.5,
                  imgs_dir="imgs",
                  scaler=minmax_scale,
                  glcm_radius=5):
    """ Runs the KMeans Matrix generation

    :param test_path: Path to the test file
    :param score_path: Path to the score file
    :param scale: How much to scale the Frame before running the loop
    :param output_dir: The directory to output results in
    :param ixs_per_kmeans: The number of indexes to consider.
    :param verbose: Whether to output information in console
    :param clusters: The number of KMeans clusters
    :param scatter_size: Size of the scatter used in reconstructing image
    :param imgs_dir: The subdir folder name of images
    :param scaler: The scaler to use to normalize data
    :param glcm_radius: Radius of GLCM
    """
    f = Frame2D.from_image(test_path, scale=scale)
    score = Frame2D.from_image(score_path, scale=scale)
    frame = f.get_all_chns(glcm_verbose=verbose, glcm_radius=glcm_radius)

    try: os.makedirs(output_dir + "/" + imgs_dir)
    except: pass

    with open(f"{output_dir}/results.csv", "w+") as file:

        file.write(f"a,b,Custom,Homogeneity,Completeness,V Measure\n")
        for ixs in tqdm(list(combinations(22, ixs_per_kmeans))):
            print("Processing ", ixs)
            km = frame.kmeans(
                KMeans(n_clusters=clusters, verbose=verbose),
                       fit_indexes=ixs,
                       scaler=scaler)

            sns.set_palette(sns.color_palette("magma"), n_colors=clusters)
            p = km.plot(scatter_size=scatter_size)
            plt.gcf().set_size_inches(f.width() / 96 * 2, f.height() / 96 * 2)
            p.savefig(f"{output_dir}/{imgs_dir}/" +
                      "_".join([str(i) for i in ixs]) +
                      ".png")
            plt.cla()

            file.write(",".join(ixs) + ",")
            file.write(",".join([str(s) for s in
                                 km.score(score,
                                          glcm_radius=glcm_radius).values()]) + '\n')
            file.flush()


def kmeans(f: Frame2D,
           clusters: int,
           verbose: bool,
           fit_indexes: list,
           scaler=minmax_scale,
           fig_name:str or None = "out.png"):
    km = KMeans2D(f,
                  model=KMeans(n_clusters=clusters,
                               verbose=verbose),
                  fit_indexes=fit_indexes,
                  scaler=scaler)

    sns.set_palette(sns.color_palette("magma"), n_colors=clusters)
    km.plot()

    if fig_name:
        plt.gcf().set_size_inches(f.width() / 96 * 2,
                                  f.height() / 96 * 2)
        plt.gcf().savefig(fig_name)
        plt.cla()

    return km


def kmeans_score(f: Frame2D,
                 score: Frame2D or str,
                 glcm_radius: int,
                 clusters: int,
                 verbose: bool,
                 fit_indexes: list,
                 scaler=minmax_scale,
                 fig_name:str or None = "out.png",
                 scatter_size=0.5):
    km = kmeans(f, clusters, verbose, fit_indexes, scaler,
                fig_name, scatter_size)
    if isinstance(score, str):
        score = Frame2D.from_image(score)
    print(km.score(score, glcm_radius=glcm_radius))