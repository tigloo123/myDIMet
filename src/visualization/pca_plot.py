#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
from typing import List, Union
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.patches import Ellipse
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
import matplotlib.figure as figure

logger = logging.getLogger(__name__)

cs = ConfigStore.instance()


def variance_expl_plot(var_explained_df) -> figure.Figure:
    """
    returns the bar-plot with the percentage of explained variances by PC
    """
    fig = plt.figure()
    sns.barplot(x='PC', y='Explained Variance %',
                data=var_explained_df, color="cadetblue")
    plt.title("Percent variability explained by the principal components")
    plt.ylabel("Explained variances (%)")
    return fig


def eigsorted(cov: np.array) -> tuple:
    """
    used for calculating ellipses
    many thanks to :
    https://rayblick.gitbooks.io/my-python-scrapbook/content/
    analysis/plotting/scatterplot_ellipse.html

    use make_ellipse OR eigsorted, but not both
    """
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]


def pca_scatter_plot( pc_df, var_explained_df, col1, col2,
                       pointlabels, *args) -> figure.Figure:
    """
    returns the scatterplot
       *args :  options for advanced PCA plotting:
             args[0] is column name for ellipses
    """

    col_ellipses = None
    try:
        col_ellipses = args[0]
    except IndexError:
        pass  # ellipses args not set, not drawn by default
    except Exception as e:
        print("unknown error! ", e)

    # scatterplot
    fig, ax = plt.subplots()
    sns.scatterplot(x="PC1", y="PC2",
                    ax=ax,
                    data=pc_df,
                    hue=col1,
                    style=col2,
                    legend=True,
                    s=80, zorder=3)
    ax.axhline(0, ls="--", color="gray", zorder=1)
    ax.axvline(0, ls="--", color="gray", zorder=1)
    if pointlabels != "":
        for i, row in pc_df.iterrows():
            ax.text(pc_df.at[i, 'PC1'] + 0.2, pc_df.at[i, 'PC2'],
                    pc_df.at[i, pointlabels],
                    size='x-small')
    # end if
    row_xlab = var_explained_df.iloc[0, :]
    row_ylab = var_explained_df.iloc[1, :]

    plt.xlabel(
        f"{row_xlab['PC']} {round(row_xlab['Explained Variance %'], 2)} %")
    plt.ylabel(
        f"{row_ylab['PC']} {round(row_ylab['Explained Variance %'], 2)} %")
    plt.title("")
    # ellipses
    if col_ellipses is not None:
        myellipsesnames = pc_df[col_ellipses].unique()
        for lab in myellipsesnames:
            xdata = pc_df.loc[pc_df[col_ellipses] == lab, 'PC1']
            ydata = pc_df.loc[pc_df[col_ellipses] == lab, 'PC2']
            # get values to build the ellipse
            cov = np.cov(xdata, ydata)
            vals, vecs = eigsorted(cov)
            theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            # print(vals, vecs, theta)
            w, h = 2 * 2 * np.sqrt(vals)
            # create the ellipse
            ell = Ellipse(xy=(np.mean(xdata), np.mean(ydata)),
                          width=w, height=h,
                          angle=theta,
                          edgecolor="lightgray",
                          linestyle='-', facecolor='none')
            ax.add_artist(ell)
    # end  if
    return fig


def pca_scatter_2_pdf(figure_pc: figure.Figure,
                      name_elements: List[str], out_plot_dir) -> None:
    name_plot = f"{'--'.join(name_elements)}_pc.pdf"
    figure_pc.savefig(os.path.join(out_plot_dir, name_plot))
    return 0


def save_pca_plot(name_plot, pc_df, var_explained_df, col1, col2,
                  pointlabels,
                  out_plot_dir, *args) -> None: # ellipses_col: Union[str, None]
    """
       *args :  options for advanced PCA plotting:
             args[0] is column name for ellipses
    """

    col_ellipses = None
    try:
        col_ellipses = args[0]
    except IndexError:
        pass  # ellipses args not set, not drawn by default
    except Exception as e:
        print("unknown error! ", e)


    # scatterplot
    fig, ax = plt.subplots()
    sns.scatterplot(x="PC1", y="PC2",
                    ax=ax,
                    data=pc_df,
                    hue=col1,
                    style=col2,
                    legend=True,
                    s=80, zorder=3)
    ax.axhline(0, ls="--", color="gray", zorder=1)
    ax.axvline(0, ls="--", color="gray", zorder=1)

    yesnolabel = "no"
    if pointlabels != "":
        yesnolabel = "yes"
        for i, row in pc_df.iterrows():
            ax.text(pc_df.at[i, 'PC1'] + 0.2, pc_df.at[i, 'PC2'],
                    pc_df.at[i, pointlabels],
                    size='x-small')
    # end if

    row_xlab = var_explained_df.iloc[0, :]
    row_ylab = var_explained_df.iloc[1, :]

    plt.xlabel(
        f"{row_xlab['PC']} {round(row_xlab['Explained Variance %'], 2)} %")
    plt.ylabel(
        f"{row_ylab['PC']} {round(row_ylab['Explained Variance %'], 2)} %")
    plt.title("")

    # ellipses
    if col_ellipses is not None:
        myellipsesnames = pc_df[col_ellipses].unique()
        for lab in myellipsesnames:
            xdata = pc_df.loc[pc_df[col_ellipses] == lab, 'PC1']
            ydata = pc_df.loc[pc_df[col_ellipses] == lab, 'PC2']

            # get values to build the ellipse
            cov = np.cov(xdata, ydata)
            # ell = make_ellipse((np.mean(xdata), np.mean(ydata)),cov,
            #                  level= 0.95, color=None)
            vals, vecs = eigsorted(cov)
            theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            # print(vals, vecs, theta)
            w, h = 2 * 2 * np.sqrt(vals)

            # create the ellipse
            ell = Ellipse(xy=(np.mean(xdata), np.mean(ydata)),
                          width=w, height=h,
                          angle=theta,
                          edgecolor="lightgray",
                          linestyle='-', facecolor='none')

            # ell.set_facecolor() # reference the colour for each factor
            ax.add_artist(ell)
    # end  if ellipses

    plt.savefig(
        os.path.join(
            out_plot_dir,
            f'pca_{name_plot}_label{yesnolabel}.pdf'
        ),
        format="pdf")

    plt.close()


def demo_pca_iris(out_plot_dir) -> None:
    from sklearn.decomposition import PCA
    iris = sns.load_dataset("iris")
    sns.relplot(data=iris, x="sepal_width", y="petal_width",
                hue="species")
    iris = iris.assign(name_to_plot=[str(i) for i in iris.index])
    fakemeta = iris[['name_to_plot', "species"]]
    iris = iris.drop(columns=['name_to_plot', "species"])
    fakedf = iris.T  # variables rows, samples columns
    fakedf = fakedf.div(fakedf.std(axis=1, ddof=0), axis=0)

    fakedf.columns = [str(i) for i in iris.index]

    X = np.transpose(np.array(fakedf))
    pca = PCA(n_components=4)
    pc = pca.fit_transform(X)
    pc_df = pd.DataFrame(data=pc,
                         columns=['PC' + str(i) for i in range(1, 4 + 1)])
    pc_df = pc_df.assign(name_to_plot=fakedf.columns)
    pc_df = pd.merge(pc_df, fakemeta, on='name_to_plot')

    var_explained_df = pd.DataFrame({
        'Explained Variance %': pca.explained_variance_ratio_ * 100,
        'PC': ['PC' + str(i) for i in range(1, 4 + 1)]})

    save_pca_plot("Iris", pc_df, var_explained_df,
                   "species", "species", "", out_plot_dir, "species")


def run_pca_plot(pca_results_dict: dict,  cfg: DictConfig,
                 out_plot_dir: str) -> None:
    for tup in pca_results_dict.keys():
        pc_df =  pca_results_dict[tup]['pc']
        var_explained_df = pca_results_dict[tup]['var']
        figure_var = variance_expl_plot(var_explained_df)
        name_plot_var = f"{'--'.join(tup)}_var.pdf"
        figure_var.savefig(os.path.join(out_plot_dir, name_plot_var))

        options_labels = {'label-y': "name_to_plot",
                          'label-n': ""}  # when empty string, no dot labels
        if cfg.analysis.method.draw_ellipses is not None:
            for ellipses_column in cfg.analysis.method.draw_ellipses:
                for choice in options_labels.keys():

                    label_column = options_labels[choice]

                    # name_plot_pc = f"{'--'.join(tup)}"  # TODO  bad
                    # name_plot_pc += ("---" + choice)  # TODO bad
                    # save_pca_plot(name_plot_pc, pc_df,
                    #               var_explained_df, "condition",
                    #               "condition", label_column,
                    #               out_plot_dir, ellipses_column)

                    name_elements = list(tup) + [choice]
                    scatter_fig = pca_scatter_plot( pc_df,
                                  var_explained_df, "condition",
                                  "condition", label_column,
                                   ellipses_column)
                    pca_scatter_2_pdf(scatter_fig, name_elements, out_plot_dir)
                    plt.close()


        else:
            for choice in options_labels.keys():
                label_column = options_labels[choice]

                # name_plot_pc = f"{'--'.join(tup)}" # TODO  bad
                # name_plot_pc += ("---" + choice)# TODO bad
                # save_pca_plot(name_plot_pc, pc_df,
                #               var_explained_df, "condition",
                #               "condition", label_column,
                #               out_plot_dir)

                name_elements = list(tup) + [choice]
                scatter_fig = pca_scatter_plot(pc_df,
                                               var_explained_df, "condition",
                                               "condition", label_column,
                                               )
                pca_scatter_2_pdf(scatter_fig, name_elements, out_plot_dir)
                plt.close()

    # end for
    if cfg.analysis.method.run_iris_demo:
        demo_pca_iris(out_plot_dir)

# end

# calculate Ellipses , many thanks to :
# https://rayblick.gitbooks.io/my-python-scrapbook/content/analysis/plotting/scatterplot_ellipse.html

# ex4:
# https://www.programcreek.com/python/example/61396/matplotlib.patches.Ellipse


