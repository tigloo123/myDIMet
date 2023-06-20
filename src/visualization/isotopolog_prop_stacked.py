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


def isotopol_prop_2piled_df(df_co, metada_co, levelshours_str):
    """
    Imitates de behaviour of a 'melt', but this function is more secure
    example:
      pd.merge output, where 1st column has the values we want:

        L-Phenylalanine_m+2    timenum   condition ...  sample descr...
        0.01                           0          Control        xxxxxx

      is transformed into:
          timenum    condition    isotopolgFull        Isotopologue
          0           control      L-Phenylala...    0.01

    """
    dfcompartment = df_co.T

    metabolites = dfcompartment.columns
    dfcompartment['name_to_plot'] = dfcompartment.index
    dfcompartment = pd.merge(dfcompartment, metada_co, on='name_to_plot')
    # empty dataframe to fill
    piled_df = pd.DataFrame(
        columns=[
            "timenum",
            "condition",
            "isotopolgFull",
            "Isotopologue Contribution (%)",
        ]
    )
    piled_df["timenum"] = pd.Categorical(piled_df["timenum"], levelshours_str)
    # iteratively pile up
    for z in range(len(metabolites)):
        subdf = dfcompartment.loc[:, [metabolites[z], "timenum", "condition"]]
        subdf['timenum'] = subdf['timenum'].astype(str)
        # 1st colname as cell value, reps
        subdf["isotopolgFull"] = metabolites[z]
        subdf["Isotopologue Contribution (%)"] = subdf[metabolites[z]] * 100
        # 1st col no longer needed
        subdf = subdf.drop(columns=[metabolites[z]])
        piled_df = pd.concat([piled_df, subdf])
        del subdf

    return piled_df


def massageisotopologues(piled_df):
    """
    returns dataframe splitting metabolite and m+x into two separate columns
    and also correcting weird values
    """
    xu = {"name": [], "m+x": []}
    for ch in piled_df["isotopolgFull"]:
        elems = ch.split("_m+")
        xu["name"].append(elems[0])
        xu["m+x"].append("m+" + elems[1])
    piled_df["metabolite"] = xu["name"]
    piled_df["m+x"] = xu["m+x"]
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


def preparemeansreplicates(piled_df, selectedmets):
    """
    returns a dictionary of dataframes, keys are metabolites
    """
    dfcopy = piled_df.copy()
    dfcopy = dfcopy.groupby(
        ["condition", "metabolite", "m+x", "timenum"])\
        .mean("Isotopologue Contribution %")
    dfcopy = dfcopy.reset_index()

    dfs_Dico = dict()
    for i in selectedmets:
        tmp = dfcopy.loc[dfcopy["metabolite"] == i, ].reset_index()
        # set m+x as numeric to avoid any bad reordering of stacked m+x
        tmp["m+x"] = tmp["m+x"].str.split("m+", regex=False).str[1]
        tmp["m+x"] = tmp["m+x"].astype(int)

        dfs_Dico[i] = tmp
    return dfs_Dico


def addcombinedconditime(dfs_Dico, combined_tc_levels):
    """
    add column 'timeANDcondi' to each  metabolite dataframe in Dico
    """
    for metab in dfs_Dico.keys():
        dfs_Dico[metab]["timeANDcondi"] = \
            dfs_Dico[metab]["timenum"] + " : " + dfs_Dico[metab]["condition"]

        dfs_Dico[metab]["timeANDcondi"] = pd.Categorical(
            dfs_Dico[metab]["timeANDcondi"],
            combined_tc_levels)
    return dfs_Dico


def addcategoricaltime(dfs_Dico, levelstime_str):
    """
    existing column 'timenum' as categorical, with specific order.
    Each metabolite dataframe in Dico is processed.
    """
    for metab in dfs_Dico.keys():
        dfs_Dico[metab]["timenum"] = pd.Categorical(
            dfs_Dico[metab]["timenum"],
            levelstime_str)
    return dfs_Dico


def give_colors_carbons(nb_of_carbons):
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


def complexstacked(selectedmets, dfs_Dico, outfilename, cfg,
                   figu_width, xlabyesno,  wspace_stacks, numbers_size,
                   x_to_plot, x_ticks_text_tilt):
    """plot highly custom, recommended that selectedmets <= 6 subplots"""
    palsautoD = give_colors_carbons(cfg.analysis.method.max_nb_carbons_possible)

    sns.set_style({"font.family": "sans-serif",
                   "font.sans-serif": "Liberation Sans"})
    f, axs = plt.subplots(1, len(selectedmets), sharey=cfg.analysis.method.sharey,
                          figsize=(figu_width, cfg.analysis.method.plots_height))
    plt.rcParams.update({"font.size": 20})

    for z in range(len(selectedmets)):
        axs[z].set_title(selectedmets[z])
        sns.histplot(
            ax=axs[z],
            data=dfs_Dico[selectedmets[z]],
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
        ylabels = ax.get_yticks().tolist()
        ax.yaxis.set_major_locator(mticker.FixedLocator(ylabels))
        ax.set_yticklabels([100 - int(i) for i in ylabels])  # invert y, step2

    # when panel without x labels
    if xlabyesno == "no":
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


def add_xemptyspace_tolabs(conditions, levelshours_str):
    conditions.extend(["xemptyspace"])  # to add space among time categories
    combined_tc_levels = list()
    tmp = conditions.copy()
    conditions = list()  # warranty uniqueness
    for i in tmp:
        if i not in conditions:
            conditions.append(i)
    for x in levelshours_str:
        for y in conditions:
            if y == "xemptyspace":
                combined_tc_levels.append(str(x)+"xemptyspace")
            else:
                combined_tc_levels.append(f'{x} : {y}')
    return conditions, combined_tc_levels


def time_plus_condi_labs(conditions, levelshours_str):
    combined_tc_levels = list()
    for x in levelshours_str:
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


def run_isotopol_prop_stacked(dataset: Dataset,
                              out_plot_dir, cfg: DictConfig) -> None:
    metadata_df = dataset.metadata_df # change name to metadata_df
    timepoints = cfg.analysis.timepoints
    metabolites = (
        cfg.analysis.metabolites
    )  # will define which metabolites are plotted
    conditions = cfg.analysis.dataset.conditions  # <= locate where it is used

    width_each_stack = cfg.analysis.width_each_stack
    wspace_stacks = cfg.analysis.wspace_stacks
    numbers_size = cfg.analysis.inner_numbers_size
    x_ticks_text_tilt_fixed = 90   # changing tilt gives bad result, let fixed

    levelshours_str = [str(i) for i in sorted(metadata_df['timenum'].unique())]

    compartments = metadata_df['short_comp'].unique().tolist()

    for co in compartments:
        metadata_compartment_df: pd.DataFrame = \
            metadata_df.loc[metadata_df["short_comp"] == co, :]
        compartment_df = \
            dataset.compartmentalized_dfs["isotopologue_prop_file_name"][co]
        # metadata, isotopologues and time of interest
        metada_sel = metadata_compartment_df.loc[
                     metadata_compartment_df["timepoint"].isin(timepoints), :]
        iso_sel = compartment_df[metada_sel["name_to_plot"]]
        # note that pandas automatically transform any 99.9% in decimal 0.999
        piled_df = isotopol_prop_2piled_df(iso_sel, metada_sel, levelshours_str)

        piled_df = massageisotopologues(piled_df)

        selectedmets = metabolites[co] # this must be compartment specific !

        dfs_Dico = preparemeansreplicates(piled_df, selectedmets)

        # adapt width to nb of metabolites
        figu_width = width_each_stack * len(selectedmets)

        if cfg.analysis.method.separated_plots_by_condition:
            for condition in conditions:
                output_path = os.path.join(
                    out_plot_dir, f"isotopologues_stack_{condition}--{co}.pdf"
                )
                metada_this_condi = metada_sel.loc[
                    metada_sel['condition'] == condition, :]
                df_this_condi = iso_sel[metada_this_condi['name_to_plot']]
                piled_df = isotopol_prop_2piled_df(df_this_condi,
                                                 metada_this_condi,
                                                 levelshours_str)
                piled_df = massageisotopologues(piled_df)
                dfs_Dico = preparemeansreplicates(piled_df,  selectedmets)
                dfs_Dico = addcategoricaltime(dfs_Dico, levelshours_str)
                complexstacked(
                    selectedmets, dfs_Dico, output_path, cfg,
                    figu_width, xlabyesno="yes", wspace_stacks=wspace_stacks,
                    numbers_size=numbers_size, x_to_plot="timenum",
                    x_ticks_text_tilt=x_ticks_text_tilt_fixed
                )
                plt.close()
        else:
            if cfg.analysis.method.appearance_separated_time:
                conditions, combined_tc_levels = add_xemptyspace_tolabs(
                                               conditions, levelshours_str)
            else:
                combined_tc_levels = time_plus_condi_labs(conditions,
                                                          levelshours_str)

            dfs_Dico = addcombinedconditime(dfs_Dico, combined_tc_levels)

            output_path = os.path.join(
                out_plot_dir, f"isotopologues_stack--{co}.pdf"
            )
            complexstacked(
                selectedmets, dfs_Dico, output_path, cfg,
                figu_width,
                xlabyesno="yes",
                wspace_stacks=wspace_stacks,
                numbers_size=numbers_size, x_to_plot="timeANDcondi",
                x_ticks_text_tilt=x_ticks_text_tilt_fixed
            )

            # new :
            output_path_NoXlab = os.path.join(
                out_plot_dir, f"sotopologues_stack--{co}_noxlab.pdf"
            )
            complexstacked(
                selectedmets, dfs_Dico,  output_path_NoXlab, cfg,
                figu_width,
                xlabyesno="no",
                wspace_stacks=wspace_stacks,
                numbers_size=numbers_size, x_to_plot="timeANDcondi",
                x_ticks_text_tilt=x_ticks_text_tilt_fixed
            )
        # legend alone
        create_legend(cfg, out_plot_dir)

    return 0


