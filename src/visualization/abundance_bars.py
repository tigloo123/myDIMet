#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Credits: Johanna Galvis, Macha Nikolski

import logging
import os
from typing import List, Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from data import Dataset
from helpers import dynamic_xposition_ylabeltext

logger = logging.getLogger(__name__)

cs = ConfigStore.instance()


def pile_up_abundance(abu_sel, metada_sel):
    dfcompartment = abu_sel.T
    metabolites = dfcompartment.columns
    dfcompartment['name_to_plot'] = dfcompartment.index
    dfcompartment = pd.merge(dfcompartment, metada_sel, on='name_to_plot')
    dafull = pd.DataFrame(columns=["timepoint", "condition",
                                   "metabolite", "abundance"])
    for z in range(len(metabolites)):
        subdf = dfcompartment.loc[:,
                [metabolites[z], "timepoint", "condition"]]
        subdf["metabolite"] = metabolites[z]
        subdf["abundance"] = subdf[metabolites[z]]
        subdf = subdf.drop(columns=[metabolites[z]])
        dafull = pd.concat(
            [dafull, subdf], ignore_index=True
        )
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
        analysis_confidic: Any) -> int:
    selected_metabs = selected_metabolites
    sns.set_style({"font.family": "sans-serif",
                   "font.sans-serif": "Liberation Sans"})
    plt.rcParams.update({"font.size": 21})
    YLABE = "Abundance"
    fig, axs = plt.subplots(1, len(selected_metabs),
                            sharey=False, figsize=(plotwidth, 5.5))

    for il in range(len(selected_metabs)):
        herep = piled_sel_df.loc[
                piled_sel_df["metabolite"] == selected_metabs[il], :]
        herep = herep.reset_index()
        sns.barplot(
            ax=axs[il],
            x=axisx_var,
            y="abundance",
            hue=str(hue_var),
            data=herep,
            palette=analysis_confidic.palette,
            alpha=1,
            edgecolor="black",
            errcolor="black",
            errwidth=1.7,
            # errorbar='sd',
            capsize=0.12
        )
        try:
            sns.stripplot(
                ax=axs[il],
                x=axisx_var,
                y="abundance",
                hue=str(hue_var),
                data=herep,
                palette=analysis_confidic.palette,
                dodge=True,
                edgecolor="black",
                linewidth=1.5,
                alpha=1
            )
        except Exception as e:
            # When Nan it throws error, avoid it
            print(e)
            pass

        axs[il].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        if analysis_confidic.x_text != "":
            the_x_text = analysis_confidic.x_text
            try:
                xticks_text_l = the_x_text.split(",")
                axs[il].set_xticklabels(xticks_text_l)
            except Exception as e:
                print(e, "The argument x_text is incorrectly set, see help")

        axs[il].set(title=" " + selected_metabs[il] + "\n")
        axs[il].set(ylabel="")
        axs[il].set(xlabel="")
        sns.despine(ax=axs[il])
        axs[il].tick_params(axis="x", labelrotation=axisx_labeltilt)
        axs[il].set_ylim(bottom=0)  # set minimal val display : y axis : 0

    thehandles, thelabels = axs[-1].get_legend_handles_labels()
    for il in range(len(selected_metabs)):
        axs[il].legend_.remove()

    plt.subplots_adjust(left=0.2, top=0.76, bottom=0.2,
                        wspace=wspace_subfigs, hspace=1)

    # plt.tight_layout(pad = 0.01, w_pad = -2, h_pad=0.1)

    fig.text(x=dynamic_xposition_ylabeltext(plotwidth),
             y=0.5, s=YLABE,
             va="center", rotation="vertical", size=26)
    # fig.suptitle(f"{CO} ({SMX} abundance)".upper())
    output_path = os.path.join(output_directory, f"bars_{CO}_{SMX}.pdf")
    plt.savefig(output_path, bbox_inches="tight", format="pdf")
    plt.close()
    plt.figure()
    plt.legend(handles=thehandles, labels=thelabels, loc='upper right')
    plt.axis("off")
    plt.savefig(os.path.join(output_directory, "legend.pdf"), format="pdf")
    return 0


def run_steps_abund_bars(
        table_prefix,
        dataset: Dataset,
        out_plot_dir,
        cfg: DictConfig) -> None:
    metadata_df = dataset.metadata_df

    # This has to migrate somewhere else than the top level configuration
    ##############################
    time_sel = cfg.analysis.dataset.time_sel  # locate where it is used
    selectedmetsD = cfg.analysis.dataset.metabolites_to_plot  # locate where it is used
    condilevels = cfg.analysis.dataset.conditions  # <= locate where it is used

    axisx_labeltilt = cfg.analysis.method.axisx_labeltilt
    axisx_var = cfg.analysis.method.axisx
    hue_var = cfg.analysis.method.barcolor

    width_each_subfig = cfg.analysis.method.width_each_subfig
    wspace_subfigs = cfg.analysis.method.wspace_subfigs
    ##############################

    # data_path = analysis_confidic["data_path"]
    # suffix = analysis_confidic['suffix']

    compartments = metadata_df['short_comp'].unique().tolist()
    # dynamically open the file based on prefix, compartment and suffix:
    for c in compartments:
        metadata_compartment_df: pd.DataFrame = metadata_df.loc[metadata_df['short_comp'] == c, :]
        # the_folder = f'{data_path}/processed/'
        # fn = f'{the_folder}{table_prefix}--{c}--{suffix}.tsv'
        # abundance_df = pd.read_csv(fn, sep='\t', header=0, index_col=0)
        compartment_df = dataset.compartmentalized_dfs['abundances_file_name'][c]
        # metadata and abundances time of interest
        metada_sel = metadata_compartment_df.loc[metadata_compartment_df["timepoint"].isin(time_sel), :]
        abu_sel = compartment_df[metada_sel['name_to_plot']]

        # total piled-up data:
        piled_sel = pile_up_abundance(abu_sel, metada_sel)
        piled_sel["condition"] = pd.Categorical(
            piled_sel["condition"], condilevels)
        piled_sel["timepoint"] = pd.Categorical(
            piled_sel["timepoint"], time_sel)

        plotwidth = width_each_subfig * len(selectedmetsD[c])

        plot_abundance_bars(piled_sel, selectedmetsD[c], c,
                            "total abundance",
                            axisx_var, hue_var, plotwidth,
                            out_plot_dir, axisx_labeltilt,
                            wspace_subfigs, cfg)
