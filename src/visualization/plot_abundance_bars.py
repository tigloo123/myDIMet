#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Credits: Johanna Galvis, Macha Nikolski

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import helpers

def bars_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--aconfig', type=str,
                        help="configuration file in absolute path")

    parser.add_argument('--dconfig', type=str,
                        help="data configuration file in absolute path")

    parser.add_argument('--palette',  default="pastel",
                        help="qualitative or categorical palette name as in \
                        Seaborn or Matplotlib libraries (Python)")

    parser.add_argument(
        '--x_text', type=str, default="",
        help='abbreviations for x axis ticks text. \
        First run default to get aware of exact ticks text. Then write them \
        separated by commas, example: "Ctl,Reac-a,Reac-b,..." ')

    return parser


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


def printabundbarswithdots(piled_sel, selectedmets, CO, SMX,
                           axisx_var, hue_var, plotwidth,
                           odirbars, axisx_labeltilt, wspace_subfigs, args):
    selected_metabs = selectedmets
    sns.set_style({"font.family": "sans-serif",
                   "font.sans-serif": "Liberation Sans"})
    plt.rcParams.update({"font.size": 21})
    YLABE = "Abundance"
    fig, axs = plt.subplots(1, len(selected_metabs),
                            sharey=False, figsize=(plotwidth, 5.5))

    for il in range(len(selected_metabs)):
        herep = piled_sel.loc[
                piled_sel["metabolite"] == selected_metabs[il], :]
        herep = herep.reset_index()
        sns.barplot(
            ax=axs[il],
            x=axisx_var,
            y="abundance",
            hue=str(hue_var),
            data=herep,
            palette=args.palette,
            alpha=1,
            edgecolor="black",
            errcolor="black",
            errwidth=1.7,
            #errorbar='sd',
            capsize=0.12
        )
        try:
            sns.stripplot(
                ax=axs[il],
                x=axisx_var,
                y="abundance",
                hue=str(hue_var),
                data=herep,
                palette=args.palette,
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

        if args.x_text != "":
            the_x_text = args.x_text
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

    def dynamic_xposition_ylabeltext(plotwidth) -> float:
        position_float = (plotwidth * 0.00145)
        if position_float < 0.01:
            position_float = 0.01
        return position_float

    fig.text(x=dynamic_xposition_ylabeltext(plotwidth),
             y=0.5, s=YLABE,
             va="center", rotation="vertical", size=26)
    # fig.suptitle(f"{CO} ({SMX} abundance)".upper())
    plt.savefig(f"{odirbars}bars_{CO}_{SMX}.pdf",
                bbox_inches="tight", format="pdf")
    plt.close()
    plt.figure()
    plt.legend(handles=thehandles, labels=thelabels, loc='upper right')
    plt.axis("off")
    plt.savefig(f"{odirbars}legend.pdf", format="pdf")
    return 0


def run_steps_abund_bars(table_prefix,  metadatadf,
                         out_plot_dir, analysis_confidic, args) -> None:
    time_sel = analysis_confidic["time_sel"]  # locate where it is used
    selectedmetsD = analysis_confidic["metabolites_to_plot"]  # locate where it is used
    condilevels = analysis_confidic["conditions"]  # <= locate where it is used

    axisx_labeltilt = int(analysis_confidic["axisx_labeltilt"])
    axisx_var = analysis_confidic["axisx"]
    hue_var = analysis_confidic["barcolor"]

    width_each_subfig = float(analysis_confidic["width_each_subfig"])
    wspace_subfigs = float(analysis_confidic["wspace_subfigs"])

    out_path = analysis_confidic["output_path"]
    suffix = analysis_confidic['suffix']
    compartments = metadatadf['short_comp'].unique().tolist()
    # dynamically open the file based on prefix, compartment and suffix:
    for co in compartments:
        metada_co = metadatadf.loc[metadatadf['short_comp'] == co, :]
        the_folder = f'{out_path}results/prepared_tables/'
        fn = f'{the_folder}{table_prefix}--{co}--{suffix}.tsv'
        abutab = pd.read_csv(fn, sep='\t', header=0, index_col=0)

        # metadata and abundances time of interest
        metada_sel = metada_co.loc[metada_co["timepoint"].isin(time_sel), :]
        abu_sel = abutab[metada_sel['name_to_plot']]

        # total piled-up data:
        piled_sel = pile_up_abundance(abu_sel, metada_sel)
        piled_sel["condition"] = pd.Categorical(
            piled_sel["condition"], condilevels)
        piled_sel["timepoint"] = pd.Categorical(
            piled_sel["timepoint"], time_sel)

        plotwidth = width_each_subfig * len(selectedmetsD[co])

        printabundbarswithdots(piled_sel, selectedmetsD[co], co,
                               "total abundance",
                               axisx_var, hue_var, plotwidth,
                               out_plot_dir, axisx_labeltilt,
                               wspace_subfigs, args)

# plot_abundances_bars.py --aconfig /Users/macha/Projects/myDIMet/config/config.yaml
# --dconfig /Users/macha/Projects/myDIMet/data/example_diff/raw/data_config_example.yml
if __name__ == "__main__":
    parser = bars_args()
    args = parser.parse_args()
    analysis_confidic = helpers.open_config_file(args.aconfig)
    data_confidic = helpers.open_config_file(args.dconfig)
#    helpers.auto_check_validity_configuration_file(confidic) --> was for files
#    confidic = helpers.remove_extensions_names_measures(confidic) --> I do not see why

    out_path = os.path.expanduser(analysis_confidic['output_path'])
    meta_file = "/Users/macha/Projects/myDIMet/data/example_diff/raw/metadata.csv"
    metadatadf = helpers.open_metadata(meta_file)

    abund_tab_prefix = data_confidic['abundance_file_name']
    out_plot_dir = out_path
    helpers.detect_and_create_dir(out_plot_dir)
    run_steps_abund_bars(abund_tab_prefix, metadatadf,
                         out_plot_dir, analysis_confidic, args)
