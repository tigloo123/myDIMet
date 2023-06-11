#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Johanna Galvis, Florian Specque, Macha Nikolski
"""

import os
import yaml
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import locale
import re

import constants

def df_to_dict_by_compartment(df: pd.DataFrame, metadata: pd.DataFrame) -> dict:
    '''
    splits df into a dictionary of dataframes, each for one compartment
    '''
    output_dict = dict()
    for compartment in metadata['short_comp'].unique():
        metada_co = metadata.loc[metadata['short_comp'] == compartment, :]
        df_co = df.loc[:, metada_co['original_name']]
        output_dict[compartment] = df_co
    return output_dict

def open_config_file(config_file):
    """
        Opens and loads a YAML configuration file.
        Returns: Loaded configuration dictionary.
        """
    try:
        with open(config_file, "r") as f:
            config_dic = yaml.load(f, Loader=yaml.Loader)
    except yaml.YAMLError as yam_err:
        print(yam_err)
        config_dic = None
    except Exception as e:
        print(e)
        config_dic = None

    if config_dic is None:
        raise ValueError("\nimpossible to read the configuration file")

    return config_dic


def check_dict_has_keys(d: dict, expected_keys: list) -> np.array:
    has_key = []
    for k in expected_keys:
        has_key.append(k in d.keys())
    return np.array(has_key)


def check_dict_has_known_values(d: dict, possible_values: list) -> np.array:
    known_val = []
    for v in d.values():
        known_val.append(v in possible_values)
    return np.array(known_val)


def auto_check_validity_configuration_file(confidic) -> None:
    expected_keys = constants.data_files_keys#  + ['conditions', 'suffix']
    has_key = check_dict_has_keys(confidic, expected_keys)
    missing_keys = np.array(expected_keys)[~has_key].tolist()
    assert all(has_key), f"{missing_keys} : missing in configuration file! "


def verify_good_extensions_measures(confidic) -> None:
    """
    All DIMet modules use measures names without extension,
    if user put them by mistake, verify the format is ok.
    See also 'remove_extensions_names_measures()'
    """
    list_config_tabs = [confidic['abundance_file_name'],
                        confidic['meanE_or_fracContrib_file_name'],
                        confidic['isotopologue_prop_file_name'],
                        confidic['isotopologue_abs_file_name']]

    list_config_tabs = [i for i in list_config_tabs if i is not None]
    for lc in list_config_tabs:
        if lc.endswith(".txt") or lc.endswith(".TXT"):
            raise ValueError("Error : your files must be .csv, not .txt/TXT")
        elif lc.endswith(".xlsx"):
            raise ValueError("Error : your files must be .csv",
                  "Moreover : .xlsx files are not admitted !")


def remove_extensions_names_measures(confidic) -> dict:
    """
    All DIMet modules use measures names without extension,
    if user put them by mistake, this function internally removes them.
    Call it in all modules before using config keys
    """
    keys_names = constants.data_files_keys
    for k in keys_names:
        tmp = confidic[k]
        if tmp is not None:
            tmp = re.sub('.csv|.tsv|.CSV|.TSV', '', tmp)
            confidic[k] = tmp
    return confidic


def detect_and_create_dir(namenesteddir):
    if not os.path.exists(namenesteddir):
        os.makedirs(namenesteddir)


def open_metadata(file_path):
    try:
        metadata = pd.read_csv(file_path, sep='\t')
        return metadata
    except Exception as e:
        print(e)
        print('problem with opening metadata file')
        metadata = None
    if metadata is None:
        raise ValueError("\nproblem opening configuration file")


def verify_metadata_sample_not_duplicated(metadata_df : pd.DataFrame) -> None:
    def yield_repeated_elems(mylist):
        occur_dic = dict(map(lambda x: (x, list(mylist).count(x)),
                             mylist))  # credits: w3resource.com
        repeated_elems = list()
        for k in occur_dic.keys():
            if occur_dic[k] > 1:
                repeated_elems.append(k)
        return repeated_elems

    sample_duplicated = yield_repeated_elems(list(metadata_df['name_to_plot']))
    if len(sample_duplicated) > 0:
        txt_errors = f"-> duplicated sample names: {sample_duplicated}\n"
        raise ValueError(
            f"Error, found these conflicts in your metadata:\n{txt_errors}")


def isotopologues_meaning_df(isotopologues_full_list):
    xu = {"metabolite": [], "m+x": [], "isotopologue_name": []}
    for ch in isotopologues_full_list:
        elems = ch.split("_m+")
        xu["metabolite"].append(elems[0])
        xu["m+x"].append("m+{}".format(elems[-1].split("-")[-1]))
        xu["isotopologue_name"].append(ch)
    df = pd.DataFrame.from_dict(xu)
    return df


def prepare4contrast(idf, ametadata, grouping: list, contrast: list):
    """
    grouping,  example :  ['condition', 'timepoint' ]
          if (for a sample)  condition = "treatment" and  timepoint = "t12h",
          then newcol = "treatment_t12h"
    contrast : example : ["treatment_t12h", "control_t12h" ]
    """
    cc = ametadata.copy()
    if len(grouping) > 1:
        cc = cc.assign(newcol=['' for i in range(cc.shape[0])])
        for i, row in cc.iterrows():
            elems = row[grouping].tolist()
            cc.at[i, "newcol"] = "_".join(elems)
    else:
        cc = cc.assign(newcol=cc[grouping])
    metas = cc.loc[cc["newcol"].isin(contrast), :]
    newdf = idf[metas['name_to_plot']]
    return newdf, metas


def splitrowbynewcol(row, metas):
    """
    Returns : miniD
    example : Control_T24h : [0.0, 0.0, 0.0] , Treated_T24h : [0.0, 0.0, 0.0]
    """
    newcoluniq = list(set(metas["newcol"]))
    miniD = dict()
    for t in newcoluniq:
        # print(metas.loc[metas["newcol"] == t,:])
        koo = metas.loc[metas["newcol"] == t, :]
        selsams = koo['name_to_plot']
        miniD[t] = row[selsams].tolist()
    return miniD


def a12(lst1, lst2, rev=True):
    """
    Non-parametric hypothesis testing using Vargha and Delaney's A12 statistic:
    how often is x in lst1 greater than y in lst2?
    == > it gives a size effect, good to highlight potentially real effects <==
    """
    # credits : Macha Nikolski
    more = same = 0.0
    for x in lst1:
        for y in lst2:
            if x == y:
                same += 1
            elif rev and x > y:
                more += 1
            elif not rev and x < y:
                more += 1
    return (more + 0.5 * same) / (len(lst1) * len(lst2))


def compute_reduction(df, ddof):
    """
    modified, original from ProteomiX
    johaGL 2023:
    - if all row is zeroes, set same protein_values
    - if nanstd(array, ddof) equals 0, set same protein_values
    (example: nanstd([0.1,nan,0.1,0.1,0.1,nan])
    """
    res = df.copy()
    for protein in df.index.values:
        # get array with abundances values
        protein_values = np.array(
            df.iloc[protein].map(
                lambda x: locale.atof(x) if type(x) == str else x))
        # return array with each value divided by standard deviation, row-wise
        if (np.nanstd(protein_values, ddof=ddof) == 0) or (
                sum(protein_values) == 0):
            reduced_abundances = protein_values
        else:
            reduced_abundances = protein_values / np.nanstd(protein_values,
                                                            ddof=ddof)

        # replace values in result df
        res.loc[protein] = reduced_abundances
    return res


def give_reduced_df(df, ddof):
    rownames = df.index
    df.index = range(len(rownames))
    # index must be numeric because compute reduction accepted
    df_red = compute_reduction(df, ddof)  # reduce
    df_red.index = rownames
    return df_red


def compute_cv(reduced_abund):
    reduced_abu_np = reduced_abund.to_numpy().astype('float64')
    if np.nanmean(reduced_abu_np) != 0:
        return np.nanstd(reduced_abu_np) / np.nanmean(reduced_abu_np)
    elif np.nanmean(reduced_abu_np) == 0 and np.nanstd(reduced_abu_np) == 0:
        return 0
    else:
        return np.nan


def give_coefvar_new(df_red, red_meta, newcol: str):
    print("give cv")

    groups_ = red_meta[newcol].unique()
    tmpdico = dict()
    for group in groups_:
        samplesthisgroup = red_meta.loc[
            red_meta[newcol] == group, 'name_to_plot']
        subdf = df_red[samplesthisgroup]
        subdf = subdf.assign(CV=subdf.apply(compute_cv, axis=1))
        tmpdico[f"CV_{group}"] = subdf.CV.tolist()

    dfout = pd.DataFrame.from_dict(tmpdico)
    dfout.index = df_red.index
    return dfout


def compute_gmean_nonan(anarray):
    anarray = np.array(anarray, dtype=float)
    anarray = anarray[~np.isnan(anarray)]
    if sum(anarray) == 0:  # replicates all zero
        outval = 0
    else:
        outval = stats.gmean(anarray)
    return outval


def give_geommeans_new(df_red, metad4c, newcol: str, c_interest, c_control):
    """
    output: df, str, str
    """

    sams_interest = metad4c.loc[metad4c[newcol] == c_interest, 'name_to_plot']
    sams_control = metad4c.loc[metad4c[newcol] == c_control, 'name_to_plot']
    dfout = df_red.copy()
    geomcol_interest = "geommean_" + c_interest
    geomcol_control = "geommean_" + c_control
    dfout[geomcol_interest] = [np.nan for i in range(dfout.shape[0])]
    dfout[geomcol_control] = [np.nan for i in range(dfout.shape[0])]

    for i, row in df_red.iterrows():
        # i : the metabolite name in current row
        vec_interest = np.array(row[sams_interest])  # [ sams_interest]
        vec_control = np.array(row[sams_control])

        val_interest = compute_gmean_nonan(vec_interest)
        val_control = compute_gmean_nonan(vec_control)

        dfout.loc[i, geomcol_interest] = val_interest
        dfout.loc[i, geomcol_control] = val_control

    return dfout, geomcol_interest, geomcol_control


def give_ratios_df(df1, geomInterest, geomControl):
    """ratio of geometric means is Fold Change: FC
       Note : if zero replacement by min, which is
       defined by default, will never enter in 'if contr == 0'
    """
    df = df1.copy()
    df = df.assign(FC=[np.nan for i in range(df.shape[0])])
    for i, row in df1.iterrows():
        intere = row[geomInterest]
        contr = row[geomControl]
        if contr == 0:
            df.loc[i, "FC"] = intere / 1e-10
        else:
            df.loc[i, "FC"] = intere / contr

    return df


def countnan_samples(df, metad4c):
    """ only works if two classes or levels """
    vecout = []
    grs = metad4c['newcol'].unique()
    gr1 = metad4c.loc[metad4c['newcol'] == grs[0], 'name_to_plot']
    gr2 = metad4c.loc[metad4c['newcol'] == grs[1], 'name_to_plot']

    for i, row in df.iterrows():
        vec1 = row[gr1].tolist()
        vec2 = row[gr2].tolist()
        val1 = np.sum(np.isnan(vec1))
        val2 = np.sum(np.isnan(vec2))
        vecout.append(tuple([str(val1) + '/' + str(len(vec1)),
                             str(val2) + '/' + str(len(vec2))]))

    df['count_nan_samples'] = vecout
    return df


# from here, functions for isotopologue preview

def add_metabolite_column(df):
    theindex = df.index
    themetabolites = [i.split("_m+")[0] for i in theindex]
    df = df.assign(metabolite=themetabolites)

    return df


def add_isotopologue_type_column(df):
    theindex = df.index
    preisotopologue_type = [i.split("_m+")[1] for i in theindex]
    theisotopologue_type = [int(i) for i in preisotopologue_type]
    df = df.assign(isotopologue_type=theisotopologue_type)

    return df


def save_heatmap_sums_isos(thesums, figuretitle, outputfigure) -> None:
    fig, ax = plt.subplots(figsize=(9, 10))
    sns.heatmap(thesums,
                annot=True, fmt=".1f", cmap="crest",
                square=True,
                annot_kws={
                    'fontsize': 6
                },
                ax=ax)
    plt.xticks(rotation=90)
    plt.title(figuretitle)
    plt.savefig(outputfigure)
    plt.close()


def givelevels(melted):
    another = melted.copy()
    another = another.groupby('metabolite').min()
    another = another.sort_values(by='value', ascending=False)
    levelsmetabolites = another.index
    tmp = melted['metabolite']
    melted['metabolite'] = pd.Categorical(tmp, categories=levelsmetabolites)

    return melted


def table_minimalbymet(melted, fileout) -> None:
    another = melted.copy()
    another = another.groupby('metabolite').min()
    another = another.sort_values(by='value', ascending=False)
    another.to_csv(fileout, sep='\t', header=True)


def save_rawisos_plot(dfmelt, figuretitle, outputfigure) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    sns.stripplot(ax=ax, data=dfmelt, x="value", y="metabolite", jitter=False,
                  hue="isotopologue_type", size=4, palette="tab20")
    plt.axvline(x=0,
                ymin=0,
                ymax=1,
                linestyle="--", color="gray")
    plt.axvline(x=1,
                ymin=0,
                ymax=1,
                linestyle="--", color="gray")
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.title(figuretitle)
    plt.xlabel("fraction")
    plt.savefig(outputfigure)
    plt.close()


def dynamic_xposition_ylabeltext(plotwidth) -> float:
    position_float = (plotwidth * 0.00145)
    if position_float < 0.01:
        position_float = 0.01
    return position_float