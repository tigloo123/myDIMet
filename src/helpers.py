#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Johanna Galvis, Florian Specque, Macha Nikolski
"""

import os
from typing import Dict, List
import scipy
from collections.abc import Iterable

import constants
import numpy as np
import pandas as pd
from scipy import stats
import locale
import re
from functools import reduce

from constants import assert_literal, overlap_methods_types


def row_wise_nanstd_reduction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs a row-wise reduction of the DataFrame by dividing each value by its row's standard deviation,
    considering the presence of NaN values.
    """
    std_values = df.apply(lambda row: np.nanstd(row), axis=1)
    std_values[std_values == 0] = 1  # Replace zero standard deviations with 1 to avoid division by zero
    result = df.div(std_values, axis=0)
    return result


def concatenate_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, df3: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenate df2 and df2 to df1 ; fill the missing values with np.nan
    """
    assert set(df2.columns).issubset(set(df1.columns))
    assert set(df3.columns).issubset(set(df1.columns))
    df2 = df2.reindex(columns=df1.columns, fill_value=np.nan)
    df3 = df3.reindex(columns=df1.columns, fill_value=np.nan)
    result = pd.concat([df1, df2, df3], ignore_index=True)
    return result


def split_rows_by_threshold(df: pd.DataFrame, column_name: str, threshold: float) -> (pd.DataFrame, pd.DataFrame):
    """
    Splits the dataframe into rows having column_name value >= threshold and the rest
    Returns two dataframes
    """
    try:
        good_df = df.loc[df[column_name] >= threshold, :]
        undesired_rows = set(df.index) - set(good_df.index)
        bad_df = df.loc[list(undesired_rows)]
    except Exception as e:
        print(e)
        print("Error in split_rows_by_threshold", " check qualityDistanceOverSpan parameter in the analysis YAML file")

    return good_df, bad_df


def calculate_gmean(df: pd.DataFrame, groups: List[List[str]]) -> pd.DataFrame:
    """
    Calculates the geometric mean for each row in the specified column groups and adds the corresponding values
    as new columns to the DataFrame. Additionally, adds a column with the ratio of the two geometric means.
    Takes care of the potential division by zero error by replacing 0 by 1e-10 in division

    groups: A list with two sublists containing column names from df.

    Returns:
        The modified DataFrame with additional columns for the calculated geometric means
        and the ratio of means (FC - Fold Change).
    """
    for i, group in enumerate(groups):
        gmean_col = f"gmean_{i + 1}"
        df[gmean_col] = df[group].apply(lambda x: stats.gmean(x.dropna()), axis=1)

    ratio_col = "FC"
    mask = df[f"gmean_{2}"] == 0
    df[ratio_col] = df[f"gmean_{1}"] / np.where(mask, 1e-10, df[f"gmean_{2}"])

    return df


def first_column_for_column_values(df: pd.DataFrame, columns: List, values: List) -> List:
    """
    Given a dataframe df and selection columns, selects rows where values are equal to the those
     in the "values" List (provided in pairwise fashion).
    Returns: list of values of the first column for selected rows.
    """

    if not all(len(sublist) == len(columns) for sublist in values):
        raise ValueError("Number of values in each sublist must be the same as number of columns")

    # Create a mask for each column-value pair
    first_column_values_list = []
    for vals in values:
        masks = []
        for column, value in zip(columns, vals):
            masks.append(df[column] == value)
        # Combine the masks using logical AND
        mask = reduce(lambda x, y: x & y, masks)
        first_column_values_list.append(list(df[mask].iloc[:, 0]))

    return first_column_values_list


def zero_repl_arg(zero_repl_arg: str) -> None:  # TODO: this has to be cleaned up
    """
    zero_repl_arg is a string representing the argument for replacing zero values (e.g. "min/2").
    The result is a dictionary of replacement arguments.
    """
    zero_repl_arg = zero_repl_arg.lower()
    err_msg = "replace_zero_with argument is not correctly formatted"
    if zero_repl_arg.startswith("min"):
        if zero_repl_arg == "min":
            n = int(1)  # n is the denominator, default is 1
        else:
            try:
                n = float(str(zero_repl_arg.split("/")[1]))
            except Exception as e:
                print(e)
                raise ValueError(err_msg)

        def foo(x, n):
            return min(x) / n

    else:
        try:
            n = float(str(zero_repl_arg))
        except Exception as e:
            print(e)
            raise ValueError(err_msg)

        def foo(x, n):
            return n

    return {"repZero": foo, "n": n}


def arg_repl_zero2value(argum_zero_rep: str, df: pd.DataFrame) -> float:
    """
    Applies the repZero function to the DataFrame df where the values are greater than 0.
    This filters df to include only values greater than 0 and applies the repZero function element-wise.
    WARNING: only authorised values are (min | min/n | VALUE) (default: min)
    """
    d = zero_repl_arg(argum_zero_rep)
    repZero = d["repZero"]  # argum_zero_rep['repZero']
    n = d["n"]  # argum_zero_rep['n']
    replacement = repZero(df[df > 0].apply(repZero, n=1), n=n)
    return replacement


def overlap_symmetric(x: np.array, y: np.array) -> int:
    a = [np.nanmin(x), np.nanmax(x)]
    b = [np.nanmin(y), np.nanmax(y)]

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    overlap = np.nanmax([a[0], b[0]]) - np.nanmin([a[1], b[1]])
    return overlap


def overlap_asymmetric(x: np.array, y: np.array) -> int:
    # x is the reference group
    overlap = np.nanmin(y) - np.nanmax(x)
    return overlap


def compute_distance_between_intervals(group1: np.array, group2: np.array, overlap_method: str) -> pd.DataFrame:
    """
    computes the distance between intervals provided in group1 and group2
    """
    assert_literal(overlap_method, overlap_methods_types, "overlap method : ")

    if overlap_method == "symmetric":
        return overlap_symmetric(group1, group2)
    else:
        return overlap_asymmetric(group1, group2)


def df_to_dict_by_compartment(df: pd.DataFrame, metadata: pd.DataFrame) -> dict:
    """
    splits df into a dictionary of dataframes, each for one compartment
    """
    output_dict = dict()
    for compartment in metadata["short_comp"].unique():
        metada_co = metadata.loc[metadata["short_comp"] == compartment, :]
        df_co = df.loc[:, metada_co["original_name"]]
        output_dict[compartment] = df_co
    return output_dict


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
    expected_keys = constants.data_files_keys  # + ['conditions', 'suffix']
    has_key = check_dict_has_keys(confidic, expected_keys)
    missing_keys = np.array(expected_keys)[~has_key].tolist()
    assert all(has_key), f"{missing_keys} : missing in configuration file! "


def verify_good_extensions_measures(confidic) -> None:
    """
    All DIMet modules use measures names without extension,
    if user put them by mistake, verify the format is ok.
    See also 'remove_extensions_names_measures()'
    """
    list_config_tabs = [
        confidic["abundances_file_name"],
        confidic["meanE_or_fracContrib_file_name"],
        confidic["isotopologue_prop_file_name"],
        confidic["isotopologue_abs_file_name"],
    ]

    list_config_tabs = [i for i in list_config_tabs if i is not None]
    for lc in list_config_tabs:
        if lc.endswith(".txt") or lc.endswith(".TXT"):
            raise ValueError("Error : your files must be .csv, not .txt/TXT")
        elif lc.endswith(".xlsx"):
            raise ValueError("Error : your files must be .csv", "Moreover : .xlsx files are not admitted !")


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
            tmp = re.sub(".csv|.tsv|.CSV|.TSV", "", tmp)
            confidic[k] = tmp
    return confidic


def detect_and_create_dir(namenesteddir):  # TODO : replace by os.makedirs(file_name, exist_ok = True)
    if not os.path.exists(namenesteddir):
        os.makedirs(namenesteddir)


def verify_metadata_sample_not_duplicated(metadata_df: pd.DataFrame) -> None:
    def yield_repeated_elems(mylist):
        occur_dic = dict(map(lambda x: (x, list(mylist).count(x)), mylist))  # credits: w3resource.com
        repeated_elems = list()
        for k in occur_dic.keys():
            if occur_dic[k] > 1:
                repeated_elems.append(k)
        return repeated_elems

    sample_duplicated = yield_repeated_elems(list(metadata_df["name_to_plot"]))
    if len(sample_duplicated) > 0:
        txt_errors = f"-> duplicated sample names: {sample_duplicated}\n"
        raise ValueError(f"Error, found these conflicts in your metadata:\n{txt_errors}")


def isotopologues_meaning_df(isotopologues_full_list):
    xu = {"metabolite": [], "m+x": [], "isotopologue_name": []}
    for ch in isotopologues_full_list:
        elems = ch.split("_m+")
        xu["metabolite"].append(elems[0])
        xu["m+x"].append("m+{}".format(elems[-1].split("-")[-1]))
        xu["isotopologue_name"].append(ch)
    df = pd.DataFrame.from_dict(xu)
    return df


import pandas as pd


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
        protein_values = np.array(df.iloc[protein].map(lambda x: locale.atof(x) if type(x) == str else x))
        # return array with each value divided by standard deviation, row-wise
        if (np.nanstd(protein_values, ddof=ddof) == 0) or (sum(protein_values) == 0):
            reduced_abundances = protein_values
        else:
            reduced_abundances = protein_values / np.nanstd(protein_values, ddof=ddof)

        # replace values in result df
        res.loc[protein] = reduced_abundances
    return res


def compute_cv(reduced_abund):
    reduced_abu_np = reduced_abund.to_numpy().astype("float64")
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
        samplesthisgroup = red_meta.loc[red_meta[newcol] == group, "name_to_plot"]
        subdf = df_red[samplesthisgroup]
        subdf = subdf.assign(CV=subdf.apply(compute_cv, axis=1))
        tmpdico[f"CV_{group}"] = subdf.CV.tolist()

    dfout = pd.DataFrame.from_dict(tmpdico)
    dfout.index = df_red.index
    return dfout


def compute_gmean_nonan(anarray: np.array) -> float:
    arr_nonzero = np.where(anarray == 0, np.finfo(float).eps, anarray)
    return stats.gmean(arr_nonzero[~np.isnan(arr_nonzero)])


def countnan_samples(df: pd.DataFrame, groups: List) -> pd.DataFrame:
    """
    Calculates the count of NaN values in each row of the DataFrame and for each
    group within the specified columns, and adds two new columns to the DataFrame with the counts.

    Only works if groups contains two sublists of column names
    """
    assert len(groups) == 2
    df["count_nan_samples_group1"] = df[groups[0]].isnull().sum(axis=1)
    df["count_nan_samples_group2"] = df[groups[1]].isnull().sum(axis=1)

    return df


def dynamic_xposition_ylabeltext(plotwidth) -> float:
    position_float = plotwidth * 0.00145
    if position_float < 0.01:
        position_float = 0.01
    return position_float


def flatten(xs):
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x


def compute_ranksums_allH0(vInterest: np.array, vBaseline: np.array):
    """
    The Wilcoxon rank-sum test tests the null hypothesis that two sets of
     measurements are drawn from the same distribution.
    ‘two-sided’: one of the distributions (underlying x or y) is
        stochastically greater than the other.
    ‘less’: the distribution underlying x is stochastically less
        than the distribution underlying y.
    ‘greater’: the distribution underlying x is stochastically
        greater than the distribution underlying y.
    """
    vInterest = vInterest[~np.isnan(vInterest)]
    vBaseline = vBaseline[~np.isnan(vBaseline)]
    sta, p = scipy.stats.ranksums(vInterest, vBaseline, alternative="less")
    sta2, p2 = scipy.stats.ranksums(vInterest, vBaseline, alternative="greater")
    sta3, p3 = scipy.stats.ranksums(vInterest, vBaseline, alternative="two-sided")

    # best (smaller pvalue) among all tailed tests
    pretups = [(sta, p), (sta2, p2), (sta3, p3)]
    tups = []
    for t in pretups:  # make list of tuples with no-nan pvalues
        if not np.isnan(t[1]):
            tups.append(t)

    if len(tups) == 0:  # if all pvalues are nan assign two sided result
        tups = [(sta3, p3)]

    stap_tup = min(tups, key=lambda x: x[1])  # nan already excluded
    stat_result = stap_tup[0]
    pval_result = stap_tup[1]

    return stat_result, pval_result


def compute_wilcoxon_allH0(vInterest: np.array, vBaseline: np.array):
    #  Wilcoxon signed-rank test
    vInterest = vInterest[~np.isnan(vInterest)]
    vBaseline = vBaseline[~np.isnan(vBaseline)]
    sta, p = scipy.stats.wilcoxon(vInterest, vBaseline, alternative="less")
    sta2, p2 = scipy.stats.wilcoxon(vInterest, vBaseline, alternative="greater")
    sta3, p3 = scipy.stats.wilcoxon(vInterest, vBaseline, alternative="two-sided")

    # best (smaller pvalue) among all tailed tests
    pretups = [(sta, p), (sta2, p2), (sta3, p3)]
    tups = []
    for t in pretups:  # make list of tuples with no-nan pvalues
        if not np.isnan(t[1]):
            tups.append(t)

    if len(tups) == 0:  # if all pvalues are nan assign two sided result
        tups = [(sta3, p3)]

    stap_tup = min(tups, key=lambda x: x[1])  # nan already excluded
    stat_result = stap_tup[0]
    pval_result = stap_tup[1]

    return stat_result, pval_result


def compute_brunnermunzel_allH0(vInterest: np.array, vBaseline: np.array):
    vInterest = vInterest[~np.isnan(vInterest)]
    vBaseline = vBaseline[~np.isnan(vBaseline)]
    sta, p = scipy.stats.brunnermunzel(vInterest, vBaseline, alternative="less")
    sta2, p2 = scipy.stats.brunnermunzel(vInterest, vBaseline, alternative="greater")
    sta3, p3 = scipy.stats.brunnermunzel(vInterest, vBaseline, alternative="two-sided")

    # best (smaller pvalue) among all tailed tests
    pretups = [(sta, p), (sta2, p2), (sta3, p3)]
    tups = []
    for t in pretups:  # make list of tuples with no-nan pvalues
        if not np.isnan(t[1]):
            tups.append(t)

    if len(tups) == 0:  # if all pvalues are nan assign two sided result
        tups = [(sta3, p3)]

    stap_tup = min(tups, key=lambda x: x[1])  # nan already excluded
    stat_result = stap_tup[0]
    pval_result = stap_tup[1]

    return stat_result, pval_result


def absolute_geommean_diff(b_values: np.array, a_values: np.array):
    m_b = compute_gmean_nonan(b_values)
    m_a = compute_gmean_nonan(a_values)
    diff_absolute = abs(m_b - m_a)
    return diff_absolute
