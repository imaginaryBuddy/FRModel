""" This is old code that is meant for archiving purposes, do not use. """

import os
from itertools import combinations
from string import ascii_lowercase

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objs as go
import seaborn as sns
import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import minmax_scale

from frmodel.base.D2 import Frame2D
from frmodel.base.D2.kmeans2D import KMeans2D


def kmeans_matrix(test_path: str,
                  score_path: str,
                  scale: float,
                  output_dir: str,
                  ixs_per_kmeans: int,
                  verbose=True,
                  clusters=6,
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
    :param imgs_dir: The subdir folder name of images
    :param scaler: The scaler to use to normalize data
    :param glcm_radius: Radius of GLCM
    """
    f = Frame2D.from_image(test_path, scale=scale)
    score = Frame2D.from_image(score_path, scale=scale)
    frame = f.get_all_chns(glcm=Frame2D.GLCM(verbose=verbose, radius=radius))

    try: os.makedirs(output_dir + "/" + imgs_dir)
    except OSError: pass

    with open(f"{output_dir}/results.csv", "w+") as file:

        file.write(",".join([a for a in ascii_lowercase[0:ixs_per_kmeans]]) +
                   f",Custom,Homogeneity,Completeness,V Measure\n")
        combos = list(combinations(range(22), ixs_per_kmeans))
        for ixs in tqdm.tqdm(combos):

            km = KMeans2D(frame, KMeans(n_clusters=clusters, verbose=verbose),
                          fit_to=ixs,
                          scaler=scaler)

            sns.set_palette(sns.color_palette("magma"), n_colors=clusters)
            km.plot()
            plt.savefig(f"{output_dir}/{imgs_dir}/" +
                              "_".join([str(i) for i in ixs]) +
                              ".png")
            plt.cla()

            file.write(",".join([str(i) for i in ixs]) + ",")
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
                  fit_to=fit_indexes,
                  scaler=scaler)

    sns.set_palette(sns.color_palette("magma"), n_colors=clusters)

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
                 fig_name:str or None = "out.png"):
    km = kmeans(f, clusters, verbose, fit_indexes, scaler,
                fig_name)
    if isinstance(score, str):
        score = Frame2D.from_image(score)
    print(km.score(score, glcm_radius=glcm_radius))

def kmeans_matrix_2dscore_heatmap(result_path: str,
                                  i1='a', i2='b',
                                  output_path: str = "out.png"):
    df = pd.read_csv(result_path, ",")

    df_v  = df.pivot(i1, i2, "V Measure")
    df_cu = df.pivot(i1, i2, "Custom")
    df_h  = df.pivot(i1, i2, "Homogeneity")
    df_co = df.pivot(i1, i2, "Completeness")

    fig, ax = plt.subplots(2, 2)
    sns.heatmap(df_cu.round(2) * 100, annot=True, ax=ax[0][0])
    sns.heatmap(df_h.round(2) * 100, annot=True, ax=ax[0][1])
    sns.heatmap(df_co.round(2) * 100, annot=True, ax=ax[1][0])
    sns.heatmap(df_v.round(2) * 100, annot=True, ax=ax[1][1])

    ax[0][0].set_title("Custom")
    ax[0][1].set_title("Homogeneity")
    ax[1][0].set_title("Completeness")
    ax[1][1].set_title("V Measure")

    fig: plt.Figure
    fig.set_figheight(15)
    fig.set_figwidth(15)

    fig.savefig(output_path)


def kmeans_matrix_3dscore_heatmap(result_path: str,
                                  i1='a', i2='b', i3='c',
                                  score_method='V Measure',
                                  output_path: str = "out.html"):
    df = pd.read_csv(result_path, ",")

    data = [
        go.Scatter3d(
            x=df[i1],
            y=df[i2],
            z=df[i3],
            mode='markers',
            marker=dict(line=dict(width=0),
                        color=df[score_method]),
        )
    ]

    layout = go.Layout(
        scene=dict(xaxis={'title': i1},
                   yaxis={'title': i2},
                   zaxis={'title': i3}),
        hovermode='closest'
    )

    fig = go.Figure(data=data, layout=layout)
    fig.write_html(output_path)
