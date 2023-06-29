#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Johanna Galvis, Florian Specque, Macha Nikolski
"""
import logging
import os
from typing import List, Any
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from data import Dataset
from helpers import dynamic_xposition_ylabeltext

logger = logging.getLogger(__name__)

cs = ConfigStore.instance()


def isotopol_prop_2piled_df(compartment_df, metada_co):
    """
    melt the compartment_df, several steps required:
    - transpose compratment_df
    - combine compratment_df and metadata
    - melt
    - multiply isotopologues by 100 (are given as proportions, must be %
        for plotting)
    - set
    example output:
        timenum    condition    isotop_full_name    Isotopologue Contribution (%)
        0           control      L-Phenylalanine     0.01
    """
    combined_isos_metadata_df = compartment_df.T

    combined_isos_metadata_df['name_to_plot'] = combined_isos_metadata_df.index
    combined_isos_metadata_df = pd.merge(
        combined_isos_metadata_df, metada_co, on='name_to_plot')

    combined_isos_metadata_df = combined_isos_metadata_df.drop(
        columns=['short_comp', 'original_name', 'name_to_plot', 'timepoint'])
    piled_df = pd.melt(combined_isos_metadata_df,
                       id_vars=['timenum', 'condition'],
                       var_name="isotop_full_name",
                       value_name="Isotopologue Contribution (%)")

    piled_df['timenum'] = piled_df['timenum'].astype(str)
    piled_df['Isotopologue Contribution (%)'] = \
        piled_df['Isotopologue Contribution (%)'] * 100
    return piled_df


def massage_isotopologues(piled_df) -> pd.DataFrame:
    """
    returns dataframe splitting metabolite and m+x into two separate columns
    and also correcting weird values
    """
    tmp_df = piled_df['isotop_full_name'].str.split("_m+",
                                                    expand=True, regex=False)
    tmp_df.rename(columns={0: 'name', 1: 'm+x'}, inplace=True)
    piled_df["metabolite"] = tmp_df["name"]
    piled_df["m+x"] = tmp_df["m+x"]
    piled_df["m+x"] = "m+" + tmp_df["m+x"].astype(str)

    # dealing with weird values: bigger than 100 and less than 0 :
    piled_df.loc[
        piled_df["Isotopologue Contribution (%)"] > 100,
        "Isotopologue Contribution (%)"
    ] = 100

    piled_df.loc[
        piled_df["Isotopologue Contribution (%)"] < 0,
        "Isotopologue Contribution (%)"
    ] = 0

    return piled_df


def preparemeansreplicates(piled_df, metaboli_selected) -> dict:
    """
    returns a dictionary of dataframes, keys are metabolites
    """
    dfcopy = piled_df.copy()
    # instead groupby isotop_full_name, using m+x and metabolite works better
    dfcopy = dfcopy.groupby(
        ["condition", "metabolite", "m+x", "timenum"])\
        .mean("Isotopologue Contribution %")  # df.mean skips nan by default
    dfcopy = dfcopy.reset_index()

    dfs_dict = dict()
    for i in metaboli_selected:
        tmp = dfcopy.loc[dfcopy["metabolite"] == i, ].reset_index(drop=True)
        # set m+x as numeric to avoid any bad reordering of stacked m+x
        tmp["m+x"] = tmp["m+x"].str.split("m+", regex=False).str[1]
        tmp["m+x"] = tmp["m+x"].astype(int)

        dfs_dict[i] = tmp
    return dfs_dict


def addcombinedconditime(dfs_dict: dict, combined_tc_levels: List[str]) -> dict:
    """
    add column 'time_and_condition' to each  metabolite dataframe in Dictio
    """
    for metab in dfs_dict.keys():
        dfs_dict[metab]["time_and_condition"] = \
            dfs_dict[metab]["timenum"] + " : " + dfs_dict[metab]["condition"]

        dfs_dict[metab]["time_and_condition"] = pd.Categorical(
            dfs_dict[metab]["time_and_condition"],
            combined_tc_levels)
    return dfs_dict


def addcategoricaltime(dfs_dict, levelstime_str) -> dict:
    """
    existing column 'timenum' as categorical, with specific order.
    Each metabolite dataframe in Dictio is processed.
    """
    for metab in dfs_dict.keys():
        dfs_dict[metab]["timenum"] = pd.Categorical(
            dfs_dict[metab]["timenum"],
            levelstime_str)
    return dfs_dict


def give_colors_carbons(nb_of_carbons):
    """
    currently 30 carbons (30 colors) supported
    """
    color_d = dict()
    # color_d[0] = "lightgray"  # m+0
    color_d[0] = "#410257"
    # set colors m+1 to m+8 from Spectral palette,
    # with custom spaced selected colors (validated)
    spectralPal = sns.color_palette("Spectral", 30)
    color_d[1] = spectralPal[29]
    color_d[2] = spectralPal[26]
    color_d[3] = spectralPal[21]
    color_d[4] = spectralPal[17]
    color_d[5] = spectralPal[10]
    color_d[6] = spectralPal[6]
    color_d[7] = spectralPal[3]
    color_d[8] = spectralPal[0]
    # rest of the colors from tab20b palette
    added_pal = sns.color_palette("tab20b", 20)
    i = 9
    j = 19
    while i <= nb_of_carbons:
        color_d[i] = added_pal[j]
        j = j - 1
        i += 1

    return color_d


def complex_stacked_plot(
        metaboli_selected: List[str], dfs_dict: dict,
        outfilename: str, cfg: DictConfig,
        figu_width: float, xlab_text: bool,
        wspace_stacks: float, numbers_size: int,
        x_to_plot: str, x_ticks_text_tilt: int) -> None:
    """
    Using the isotopologues proportions, generates stacked barplots
    per metabolite, all arranged in a single pdf file.
    A legend is produced separately also as pdf file.
    """
    palsautoD = give_colors_carbons(cfg.analysis.method.max_nb_carbons_possible)

    sns.set_style({"font.family": "sans-serif",
                   "font.sans-serif": "Liberation Sans"})
    f, axs = plt.subplots(1, len(metaboli_selected), sharey=cfg.analysis.method.sharey,
                          figsize=(figu_width, cfg.analysis.method.plots_height))
    plt.rcParams.update({"font.size": 20})

    for z in range(len(metaboli_selected)):
        axs[z].set_title(metaboli_selected[z])
        sns.histplot(
            ax=axs[z],
            data=dfs_dict[metaboli_selected[z]],
            x=x_to_plot,
            # Use the value variable here to turn histogram counts into
            # weighted values.
            weights="Isotopologue Contribution (%)",
            hue="m+x",
            multiple="stack",
            palette=palsautoD,
            # Add  borders to the bars.
            edgecolor="black",
            # Shrink the bars a bit, so they don't touch.
            shrink=0.85,
            alpha=1,
            legend=False,
        )
        #
        for xtick in axs[z].get_xticklabels():
            if xtick.get_text().endswith("xemptyspace"):
                xtick.set_color("white")
            else:
                xtick.set_color("black")

        axs[z].tick_params(axis="x",
                           labelrotation=x_ticks_text_tilt,
                           labelsize=cfg.analysis.method.x_ticks_text_size)

        axs[z].tick_params(axis="y", length=3,
                           labelsize=19)
        axs[z].set_ylim([0, 100])

        # Inner text
        # if numbers_size is zero, set alpha equally, to make it invisible
        if numbers_size <= 0:
            numbers_alpha = 0
        else:
            numbers_alpha = 1
        # for defining the color of inner text in bar when M0
        rgba_eq_hex_410257 = \
            (0.2549019607843137, 0.00784313725490196, 0.3411764705882353, 1.0)

        for bar in axs[z].patches:
            # assign stacked bars inner text color
            inner_text_color = "black"
            here_rgba = bar.get_facecolor()
            if here_rgba == rgba_eq_hex_410257:
                inner_text_color = "white"
            thebarvalue = round(bar.get_height(), 1)
            if thebarvalue >= 100:
                thebarvalue = 100  # no decimals if 100
            if round(bar.get_height(), 1) >= 4:
                axs[z].text(
                    # Put the text in the middle of each bar. get_x returns t
                    # he start, so we add half the width to get to the middle.
                    bar.get_x() + bar.get_width() / 2,
                    # Vertically, add the height of the bar to the start of
                    # the bar, along with the offset.
                    (bar.get_height() / 2) + (bar.get_y()) + 2,  #
                    # This is actual value we'll show.
                    thebarvalue,
                    # Center the labels and style them a bit.
                    ha="center",
                    color=inner_text_color,
                    size=numbers_size,
                    alpha=numbers_alpha
                )  # end axs[z].text
            else:
                continue
            # end if round
        # end for bar

        axs[z].set_ylabel("", size=20)
        axs[z].xaxis.set_tick_params(length=0)  # no need of x ticks
        axs[z].set_xlabel("", size=13)
    # end for z

    [ax.invert_yaxis() for ax in axs]  # invert y, step 1

    for ax in axs:
        ylabels = list(ax.get_yticks())
        ax.yaxis.set_major_locator(mticker.FixedLocator(ylabels))
        ax.set_yticklabels([100 - int(i) for i in ylabels])  # invert y, step2

    # when panel without x labels
    if not xlab_text:
        for ax in axs:
            ax.get_xaxis().set_ticks([])

    f.subplots_adjust(hspace=0.5, wspace=wspace_stacks, top=0.85,
                      bottom=0.26, left=0.15, right=0.99)

    f.text(dynamic_xposition_ylabeltext(figu_width),
           0.57, "Isotopologue Contribution (%)\n", va="center",
           rotation="vertical", size=20)
    f.savefig(outfilename,
              bbox_inches="tight", format="pdf")
    plt.close()
    logger.info(f"Saved isotopologue stacked barplots in {outfilename}")
    return 0  


def add_xemptyspace_tolabs(conditions, time_levels_list):
    """
    adds an 'empty space' between each timepoint in the metabolite plot,
    conditions are kept next to each other in comparative aspect.
    Note : If willing to _not_ having conditions in comparative aspect
      but each condition in separate pdf instead,
      see .separated_plots_by_condition attribute.
    """
    conditions.extend(["xemptyspace"])  # to add space among time categories
    combined_tc_levels = list()
    tmp = conditions.copy()
    conditions = list()  # warranty uniqueness
    for i in tmp:
        if i not in conditions:
            conditions.append(i)
    for x in time_levels_list:
        for y in conditions:
            if y == "xemptyspace":
                combined_tc_levels.append(str(x)+"xemptyspace")
            else:
                combined_tc_levels.append(f'{x} : {y}')
    return conditions, combined_tc_levels


def time_plus_condi_labs(conditions, time_levels_list):
    combined_tc_levels = list()
    for x in time_levels_list:
        for y in conditions:
            combined_tc_levels.append(f'{x} : {y}')
    return combined_tc_levels


def givelabelstopalsD(palsautoD):
    tmp = dict()
    for k in palsautoD.keys():
        tmp["m+"+str(k)] = palsautoD[k]
    return tmp


def create_legend(cfg, out_plot_dir) -> None:
    plt.figure(figsize=(4, cfg.analysis.method.max_nb_carbons_possible * 0.6))
    palsautoD = give_colors_carbons(cfg.analysis.method.max_nb_carbons_possible)
    palsautoD_labeled = givelabelstopalsD(palsautoD)
    myhandless = []
    for c in palsautoD_labeled.keys():
        paobj = mpatches.Patch(facecolor=palsautoD_labeled[c],
                               label=c, edgecolor="black")
        myhandless.append(paobj)
    plt.legend(handles=myhandless, labelspacing=0.01)
    plt.axis("off")
    plt.savefig(
        os.path.join(
            out_plot_dir, "legend_isotopologues_stackedbars.pdf"
        ),
        format="pdf")


def run_isotopologue_proportions_plot(dataset: Dataset,
                                      out_plot_dir, cfg: DictConfig) -> None:
    metadata_df = dataset.metadata_df
    timepoints = cfg.analysis.timepoints
    metabolites = (
        cfg.analysis.metabolites
    )  # will define which metabolites are plotted
    conditions = cfg.analysis.dataset.conditions

    width_each_stack = cfg.analysis.width_each_stack
    wspace_stacks = cfg.analysis.wspace_stacks
    numbers_size = cfg.analysis.inner_numbers_size
    x_ticks_text_tilt_fixed: int = 90  # usr def tilt bad result, let fixed
    time_levels_list: List[str] = [
        str(i) for i in sorted(metadata_df['timenum'].unique())]

    compartments = list(metadata_df['short_comp'].unique())

    for co in compartments:
        metadata_compartment_df: pd.DataFrame = \
            metadata_df.loc[metadata_df["short_comp"] == co, :]
        compartment_df = dataset.compartmentalized_dfs["isotopologue_proportions"][co]

        # metadata, isotopologues and time of interest
        time_metadata_df = metadata_compartment_df.loc[
                     metadata_compartment_df["timepoint"].isin(timepoints), :]
        time_compartment_df = compartment_df[time_metadata_df["name_to_plot"]]
        # note that pandas automatically transform any 99.9% in decimal 0.999
        piled_df = isotopol_prop_2piled_df(time_compartment_df, 
                                           time_metadata_df)
        # values now are in %
        piled_df = massage_isotopologues(piled_df)
        metaboli_selected = metabolites[co]
        dfs_dict = preparemeansreplicates(piled_df, metaboli_selected)

        # figure width: adapt to nb of metabolites
        figu_width = width_each_stack * len(metaboli_selected)

        if cfg.analysis.method.separated_plots_by_condition:
            for condition in conditions:
                output_path = os.path.join(
                    out_plot_dir, f"isotopologues_stack_{condition}--{co}.pdf"
                )
                cond_time_metadata = time_metadata_df.loc[
                    time_metadata_df['condition'] == condition, :]
                condition_df = time_compartment_df[
                    cond_time_metadata['name_to_plot']]
                piled_df = isotopol_prop_2piled_df(condition_df,
                                                   cond_time_metadata)
                piled_df = massage_isotopologues(piled_df)
                dfs_dict = preparemeansreplicates(piled_df, metaboli_selected)
                dfs_dict = addcategoricaltime(dfs_dict, time_levels_list)
                complex_stacked_plot(
                    metaboli_selected, dfs_dict, output_path, cfg,
                    figu_width, xlab_text=True, wspace_stacks=wspace_stacks,
                    numbers_size=numbers_size, x_to_plot="timenum",
                    x_ticks_text_tilt=x_ticks_text_tilt_fixed
                )
                plt.close()
        else:
            if cfg.analysis.method.appearance_separated_time:
                conditions, combined_tc_levels = add_xemptyspace_tolabs(
                                               conditions, time_levels_list)
            else:
                combined_tc_levels = time_plus_condi_labs(conditions,
                                                          time_levels_list)
            # end if

            dfs_dict = addcombinedconditime(dfs_dict, combined_tc_levels)

            output_path = os.path.join(
                out_plot_dir, f"isotopologues_stack--{co}.pdf"
            )
            complex_stacked_plot(
                metaboli_selected, dfs_dict, output_path, cfg,
                figu_width,
                xlab_text=True,
                wspace_stacks=wspace_stacks,
                numbers_size=numbers_size, x_to_plot="time_and_condition",
                x_ticks_text_tilt=x_ticks_text_tilt_fixed
            )

            # drop x text: facilitate manual changes in graph external tool
            output_path_NoXlab = os.path.join(
                out_plot_dir, f"isotopologues_stack--{co}_noxlab.pdf"
            )
            complex_stacked_plot(
                metaboli_selected, dfs_dict,  output_path_NoXlab, cfg,
                figu_width,
                xlab_text=False,
                wspace_stacks=wspace_stacks,
                numbers_size=numbers_size, x_to_plot="time_and_condition",
                x_ticks_text_tilt=x_ticks_text_tilt_fixed
            )
        # end if

        create_legend(cfg, out_plot_dir)  # legend alone

    return 0


