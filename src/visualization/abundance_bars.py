#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Johanna Galvis, Florian Specque, Macha Nikolski
"""
import logging
import os
from typing import List
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from data import Dataset
from helpers import dynamic_xposition_ylabeltext

logger = logging.getLogger(__name__)

cs = ConfigStore.instance()

def pile_up_abundance(df: pd.DataFrame, metada_sel: pd.DataFrame) -> pd.DataFrame:
    dfcompartment = df.set_index("ID").T.reset_index()
    dafull = pd.DataFrame(columns=["timepoint", "condition", "metabolite", "abundance"])

    for metabolite in dfcompartment.columns[1:]:
        subdf = dfcompartment[['index', metabolite]].rename(columns={'index': 'name_to_plot'})
        subdf = subdf.merge(metada_sel, on='name_to_plot')
        subdf['metabolite'] = metabolite
        subdf.rename(columns={metabolite: 'abundance'}, inplace=True)
        dafull = pd.concat([dafull, subdf[['timepoint', 'condition', 'metabolite', 'abundance']]], ignore_index=True)

    return dafull


def plot_abundance_bars(
    piled_sel_df: pd.DataFrame,
    selected_metabolites: List[str],
    CO: str,
    SMX: str,
    axisx_var: str,
    hue_var: str,
    plotwidth: float,
    output_directory: str,
    axisx_labeltilt: int,
    wspace_subfigs: float,
    cfg: DictConfig,
) -> None:
    plt.rcParams.update({"font.size": 21})
    YLABE = "Abundance"
    fig, axs = plt.subplots(1, len(selected_metabolites), sharey=False, figsize=(plotwidth, 5.5))

    for i in range(len(selected_metabolites)):
        herep = piled_sel_df.loc[piled_sel_df["metabolite"] == selected_metabolites[i], :]
        herep = herep.reset_index()
        sns.barplot(
            ax=axs[i],
            x=axisx_var,
            y="abundance",
            hue=str(hue_var),
            data=herep,
            palette=cfg.palette,
            alpha=1,
            edgecolor="black",
            errcolor="black",
            errwidth=1.7,
            # errorbar='sd',
            capsize=0.12,
        )
        try:
            sns.stripplot(
                ax=axs[i],
                x=axisx_var,
                y="abundance",
                hue=str(hue_var),
                data=herep,
                palette=cfg.palette,
                dodge=True,
                edgecolor="black",
                linewidth=1.5,
                alpha=1,
            )
        except Exception as e:
            # When Nan it throws error, avoid it
            print(e)
            pass

        axs[i].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        if cfg.x_text != "":
            the_x_text = cfg.x_text
            try:
                xticks_text_l = the_x_text.split(",")
                axs[i].set_xticklabels(xticks_text_l)
            except Exception as e:
                print(e, "The argument x_text is incorrectly set, see help")

        axs[i].set(title=" " + selected_metabolites[i] + "\n")
        axs[i].set(ylabel="")
        axs[i].set(xlabel="")
        sns.despine(ax=axs[i])
        axs[i].tick_params(axis="x", labelrotation=axisx_labeltilt)
        axs[i].set_ylim(bottom=0)  # set minimal val display : y axis : 0

    thehandles, thelabels = axs[-1].get_legend_handles_labels()
    for i in range(len(selected_metabolites)):
        axs[i].legend_.remove()

    plt.subplots_adjust(left=0.2, top=0.76, bottom=0.2, wspace=wspace_subfigs, hspace=1)

    # plt.tight_layout(pad = 0.01, w_pad = -2, h_pad=0.1)

    fig.text(x=dynamic_xposition_ylabeltext(plotwidth), y=0.5, s=YLABE, va="center", rotation="vertical", size=26)
    output_path = os.path.join(output_directory, f"bars_{CO}_{SMX}.pdf")
    plt.savefig(output_path, bbox_inches="tight", format="pdf")
    plt.close()
    plt.figure()
    plt.legend(handles=thehandles, labels=thelabels, loc="upper right")
    plt.axis("off")
    plt.savefig(os.path.join(output_directory, "legend.pdf"), format="pdf")
    logger.info(f"Saved abundance plot in {output_path}")


def run_plot_abundance_bars(dataset: Dataset, out_plot_dir, cfg: DictConfig) -> None:
    metadata_df = dataset.metadata_df

    ##############################
    timepoints = cfg.analysis.timepoints  # locate where it is used
    metabolites = (
        cfg.analysis.metabolites
    )  # will define which metabolites are plotted in the abundance plot
    conditions = cfg.analysis.dataset.conditions  # <= locate where it is used

    axisx_labeltilt = cfg.analysis.method.axisx_labeltilt
    axisx_var = cfg.analysis.method.axisx
    hue_var = cfg.analysis.method.barcolor

    width_each_subfig = cfg.analysis.method.width_each_subfig
    wspace_subfigs = cfg.analysis.method.wspace_subfigs
    ##############################

    compartments = set(metadata_df["short_comp"])
    for compartment in compartments:
        metadata_compartment_df: pd.DataFrame = metadata_df.loc[metadata_df["short_comp"] == compartment, :]
        compartment_df = dataset.compartmentalized_dfs["abundances"][compartment]
        # metadata and abundances time of interest
        metadata_slice = metadata_compartment_df.loc[metadata_compartment_df["timepoint"].isin(timepoints), :]
        values_slice = compartment_df[["ID"] + list(metadata_slice["name_to_plot"])]

        # total piled-up data:
        piled_sel = pile_up_abundance(values_slice, metadata_slice)
        piled_sel["condition"] = pd.Categorical(piled_sel["condition"], conditions)
        piled_sel["timepoint"] = pd.Categorical(piled_sel["timepoint"], timepoints)

        plotwidth = width_each_subfig * len(metabolites[compartment])

        plot_abundance_bars(
            piled_sel,
            metabolites[compartment],
            compartment,
            "total_abundance",
            axisx_var,
            hue_var,
            plotwidth,
            out_plot_dir,
            axisx_labeltilt,
            wspace_subfigs,
            cfg,
        )
