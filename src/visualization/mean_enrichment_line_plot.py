import logging
import os
from typing import List, Any
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from data import Dataset


logger = logging.getLogger(__name__)

cs = ConfigStore.instance()


def melt_data_metadata_2df(compartment_df: pd.DataFrame,
                           metadata_co_df: pd.DataFrame) -> pd.DataFrame:
    """
    - merges quantitative data and metadata
    - melts the merged df to obtain df:
         "timenum", "condition", "metabolite" "Fractional Contribution (%)"
    - quantitative values are multiplied by 100, as expressed in %
    """
    compartment_df = compartment_df.T
    compartment_df['name_to_plot'] = compartment_df.index
    compartment_df = pd.merge(compartment_df, metadata_co_df,
                              on='name_to_plot')
    compartment_df = compartment_df.drop(columns=['name_to_plot',
                                                  'timepoint',
                                                  'short_comp',
                                                  'original_name'])
    melted_df = pd.melt(compartment_df,
                        id_vars=['timenum', 'condition'],
                        var_name="metabolite",
                        value_name="Fractional Contribution (%)")
    melted_df["Fractional Contribution (%)"] = \
        melted_df["Fractional Contribution (%)"] * 100
    return melted_df


def nested_dict__2_list(nested_dict) -> List[str]:
    result = []
    for metabolites_list in nested_dict.values():
        for metabolite in metabolites_list:
            result.append(metabolite)
    return result


def metabolite_df__mean_and_sd(one_metabolite_df: pd.DataFrame) -> pd.DataFrame:
    """
    input: dataframe by one metabolite:
      "timenum", "condition", "metabolite" "Fractional Contribution (%)"
    returns :
        dataframe by one metabolite
                condition  timenum    mean    sd    metabolite
    108        Control     0  0.000893  0.002611
    111        Control     1  0.236453  0.023246
    ...
    123        Control    24  0.854101  0.055241
    126  L-Cycloserine     0  0.010083  0.003465
    129  L-Cycloserine     1  0.259570  0.008602
    ...
    141  L-Cycloserine    24  0.815613  0.050756
    """
    assert len(one_metabolite_df["metabolite"].unique()) == 1
    df = one_metabolite_df.copy()
    df = df.drop_duplicates()
    # by default both std and mean in pandas ignore NaN
    mean_df = df.groupby(["condition", "timenum", "metabolite"])[
        "Fractional Contribution (%)"].mean().reset_index(name="mean")
    # std by pandas : ddof=0 to have same result as with numpy std
    std_df = df.groupby(["condition", "timenum", "metabolite"])[
        "Fractional Contribution (%)"].std(ddof=0).reset_index(name="sd")

    one_metabolite_result = mean_df.merge(std_df, how='inner',
                              on=["condition", "timenum", "metabolite"])
    return one_metabolite_result


def add_mean_and_sd__df(metabolites_selected_df: pd.DataFrame) -> pd.DataFrame:
    tmp_list: List[pd.DataFrame] = list()
    for metabolite_i in set(metabolites_selected_df["metabolite"]):
        one_metabolite_df = metabolites_selected_df[
            metabolites_selected_df["metabolite"] == metabolite_i]
        tmp_df = metabolite_df__mean_and_sd(one_metabolite_df)
        tmp_list.append(tmp_df)
    result_df = pd.concat(tmp_list, axis=0)
    return result_df


def empty_row_in_grid(axs, nb_columns: int) -> np.ndarray:
    # avoid legend (top) overlap with plots
    for grid_column in range(0, nb_columns):  # do empty row
        axs[0, grid_column].set_axis_off()
    return axs


def save_line_plot_2pdf(melted_compartment_df: pd.DataFrame,
                        metabolites_compartment_dict: dict,
                        cfg: DictConfig, palette_option: dict,
                        outfile: str) -> None:
    """
    constructs and saves the compartmentalized plot to pdf
    """
    time_ticks = melted_compartment_df['timenum'].unique()
    metabolites_selected_df = \
        melted_compartment_df.loc[melted_compartment_df["metabolite"].isin(
             nested_dict__2_list(metabolites_compartment_dict)), :].copy()

    result_df = add_mean_and_sd__df(metabolites_selected_df)

    sns.set_style(
        {"font.family": "sans-serif", "font.sans-serif": "Liberation Sans"}
    )
    plt.rcParams.update({"font.size": 22})
    fig_total_w = \
        cfg.analysis.method.width_subplot * len(metabolites_compartment_dict)
    fig, axs = plt.subplots(
        2,  # 2 rows: avoid legend (top) overlap with plots
        len(metabolites_compartment_dict),
        sharey=False,
        figsize=(fig_total_w, cfg.analysis.method.height_plot_pdf)
    )
    axs = empty_row_in_grid(axs, len(metabolites_compartment_dict))

    for z in range(len(metabolites_compartment_dict)):
        sns.lineplot(
            ax=axs[1, z],  # starts in 1 (after the empty row index 0)
            x="timenum",
            y="Fractional Contribution (%)",
            hue=cfg.analysis.method.color_lines_by,
            style="condition",
            err_style=None,
            alpha=cfg.analysis.method.alpha,
            linewidth=4.5,
            palette=palette_option[cfg.analysis.method.color_lines_by],
            zorder=1,
            data=metabolites_selected_df.loc[
                metabolites_selected_df["metabolite"].isin(
                    metabolites_compartment_dict[z]
                    )
                ],
            legend=True,
        )
        axs[1, z].set_xticks([int(i) for i in time_ticks])
        res_local_metabolite = result_df.loc[
            result_df["metabolite"].isin(metabolites_compartment_dict[z])]
        axs[1, z].scatter(
            res_local_metabolite["timenum"], 
            res_local_metabolite["mean"], s=22,
            facecolors="none", edgecolors="black"
        )
        axs[1, z].errorbar(
            res_local_metabolite["timenum"],
            res_local_metabolite["mean"],
            yerr=res_local_metabolite["sd"],
            fmt="none",
            capsize=3,
            ecolor="black",
            zorder=2
        )
        axs[1, z].set(ylabel=None),
        axs[1, z].set(xlabel=cfg.analysis.method.xaxis_title)
        axs[1, z].set(title=metabolites_compartment_dict[z][0])
        axs[1, z].legend(loc="upper center", bbox_to_anchor=(0.5, 2),
                         frameon=False)

    plt.subplots_adjust(bottom=0.1, right=0.8, hspace=0.1, wspace=0.4)

    fig.text(
        0.06,
        0.3, "Fractional Contribution (%)", va="center",
        rotation="vertical"
    )
    fig.savefig(outfile, format="pdf")
    logger.info(f"Saved mean enrichment line plots in {outfile}")
    return 0


def give_colors_by_metabolite(cfg: DictConfig,
                              metabolites_numbered_dict) -> dict:
    handycolors = ["rosybrown", "lightcoral", "brown", "firebrick",
                   "tomato", "coral", "sienna", "darkorange",   "peru",
                   "darkgoldenrod", "gold", "darkkhaki", "olive",
                   "yellowgreen", "limegreen", "green", "lightseagreen",
                   "mediumturquoise", "darkcyan", "teal", "cadetblue",
                   "slategrey", "steelblue", "navy", "darkslateblue",
                   "blueviolet",
                   "darkochid", "purple", "mediumvioletred", "crimson"]

    colors_dict = dict()

    if cfg.analysis.method.palette_metabolite == "auto_multi_color":
        tmp = set()
        for co in metabolites_numbered_dict.keys():
            for k in metabolites_numbered_dict[co].keys():
                tmp.update(set(metabolites_numbered_dict[co][k]))
        metabolites = sorted(list(tmp))
        if len(metabolites) <= 12:
            palettecols = sns.color_palette("Paired", 12)
            for i in range(len(metabolites)):
                colors_dict[metabolites[i]] = palettecols[i]
        else:
            for i in range(len(metabolites)):
                colors_dict[metabolites[i]] = handycolors[i]
    else:  # argument_color_metabolites is a csv file
        try:
            file_containing_colors = cfg.analysis.method.palette_metabolite
            df = pd.read_csv(file_containing_colors, header=0)
            for i, row in df.iterrows():
                metabolite = df.iloc[i, 0]  # first column metabolite
                color = df.iloc[i, 1]  # second column is color
                colors_dict[metabolite] = color
        except Exception as e:
            logger.info(e, f"\n could not assign color, wrong csv file: \
                   {cfg.analysis.method.palette_metabolite}")
            colors_dict = None

    return colors_dict


def give_colors_by_option(cfg: DictConfig, metabolites_numbered_dict) -> dict:
    """
    if option color_lines_by = metabolite, returns a dictionary of colors;
    otherwise color_lines_by = condition, returns a string (palette name)
    """
    assert cfg.analysis.method.color_lines_by in ["metabolite", "condition"]
    if cfg.analysis.method.color_lines_by == "metabolite":
        colors_dict: dict = give_colors_by_metabolite(cfg,
                                                metabolites_numbered_dict)
        palette_option = {
            "metabolite": colors_dict
        }
    else:
        try:
            palette_condition: str = cfg.analysis.method.palette_condition
        except ValueError:
            palette_condition: str = "paired"
        palette_option = {
            "condition": palette_condition
        }
    return palette_option


def line_plot_by_compartment(dataset: Dataset,
                             conditions_leveled: List[str],
                             out_plot_dir: str,
                             metabolites_numbered_dict,
                             cfg: DictConfig) -> None:
    """ calls function to construct and save plot """
    metadata_df = dataset.metadata_df
    compartments = list(metadata_df['short_comp'].unique())

    palette_option = give_colors_by_option(cfg, metabolites_numbered_dict)

    for co in compartments:
        metadata_co_df = metadata_df.loc[metadata_df['short_comp'] == co, :]
        compartment_df = dataset.compartmentalized_dfs["mean_enrichment"][co]

        melted_co_df = melt_data_metadata_2df(compartment_df, metadata_co_df)
        melted_co_df["condition"] = pd.Categorical(
            melted_co_df["condition"], conditions_leveled)
        metabolites_compartment_dict = metabolites_numbered_dict[co]
      
        # https://stackoverflow.com/questions/53137983/define-custom-seaborn-color-palette
        out_file = os.path.join(out_plot_dir,
                                f"mean_enrichment_plot--{co}.pdf")
        save_line_plot_2pdf(melted_co_df, metabolites_compartment_dict,
                            cfg, palette_option, out_file )
    return 0


def run_mean_enrichment_line_plot(dataset: Dataset,
                                  out_plot_dir: str,
                                  cfg: DictConfig) -> None:
    metabolites: dict = (
        cfg.analysis.metabolites
    )  # will define which metabolites are plotted
    conditions_leveled = cfg.analysis.dataset.conditions

    if cfg.analysis.method.plot_grouped_by_dict is not None:
        metabolites_numbered_dict = cfg.analysis.method.plot_grouped_by_dict
    else:
        try:
            tmp = dict()
            for compartment in metabolites.keys():
                tmp[compartment] = dict()
                for i, m in enumerate(metabolites[compartment]):
                    tmp[compartment][i] = [m]
            metabolites_numbered_dict = tmp
        except KeyError:
            logger.info("run_mean_enrichment_line_plot: \
            No metabolites for plotting in your config file")
    # end try
    line_plot_by_compartment(dataset,
                             conditions_leveled,
                             out_plot_dir,
                             metabolites_numbered_dict,
                             cfg)
