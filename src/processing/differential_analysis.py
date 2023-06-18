#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Johanna Galvis, Florian Specque, Macha Nikolski
"""

import os
from typing import List
import logging
import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.stats.multitest as ssm
from functools import reduce
import operator
from omegaconf import DictConfig

from constants import (
    availtest_methods,
    correction_methods,
    availtest_methods_type,
    assert_literal,
    data_files_keys_type,
)
from processing import fit_statistical_distribution

import helpers
from data import Dataset

logger = logging.getLogger(__name__)


def compute_span_incomparison(df: pd.DataFrame, groups: List) -> pd.DataFrame:
    """
    For each row in the input DataFrame, computes the difference between the maximum and minimum values
    from columns in two sublists provided in the 'groups' list.
    Returns:
         DataFrame with an additional 'span_allsamples' column containing the computed differences.
    """
    for i in df.index.values:
        all_values = list(df.loc[i, groups[0]]) + list(df.loc[i, groups[1]])

        interval = max(all_values) - min(all_values)
        df.loc[i, "span_allsamples"] = interval

    return df


def calc_ratios(df4c: pd.DataFrame, groups: List) -> pd.DataFrame:
    df4c = helpers.calculate_gmean(df4c, groups)

    return df4c


def divide_groups(df4c, metad4c, selected_contrast):
    """split into two df"""
    sc = selected_contrast
    sam0 = metad4c.loc[metad4c["newcol"] == sc[0], "name_to_plot"]  # interest
    sam1 = metad4c.loc[metad4c["newcol"] == sc[1], "name_to_plot"]  # control
    group_interest = df4c[sam0]
    group_control = df4c[sam1]
    group_interest.index = range(group_interest.shape[0])
    group_control.index = range(group_control.shape[0])

    return group_interest, group_control


def distance_or_overlap(df: pd.DataFrame, groups: List) -> pd.DataFrame:
    """
    For each row in the input DataFrame, computes the distance or overlap between intervals
    provided as sublists of column names in the 'groups' list.
    Returns:
        DataFrame with an additional 'distance' column containing computed distances.
    """
    for i in df.index.values:
        group1 = df.loc[i, groups[0]].values
        group2 = df.loc[i, groups[1]].values
        overlap_method = "symmetric"  # Modify as needed, can be "symmetric" or "asymmetric"
        df.at[i, "distance"] = helpers.compute_distance_between_intervals(group1, group2, overlap_method)

    return df


def select_rows_with_sufficient_non_nan_values(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
    Identifies rows in the input DataFrame that have enough replicates.
    Separates the DataFrame into two parts based on the presence or absence of enough replicates,
    returning them as separate DataFrames.
    """
    try:
        bad_df = df[(df["count_nan_samples_group1"] > 0) | (df["count_nan_samples_group2"] > 0)]
        good_df = df[(df["count_nan_samples_group1"] == 0) & (df["count_nan_samples_group2"] == 0)]
        # removed the side effect
        # good_df = good_df.drop(columns=['count_nan_samples_group1', 'count_nan_samples_group2'])
        # bad_df = bad_df.drop(columns=['count_nan_samples_group1', 'count_nan_samples_group2'])
    except Exception as e:
        print(e)
        print("Error in separate_before_stats (enough replicates ?)")

    return good_df, bad_df


def compute_mann_whitney_allH0(vInterest, vBaseline):
    """
    Calculate Mann–Whitney U test (a.k.a Wilcoxon rank-sum test,
    or Wilcoxon–Mann–Whitney test, or Mann–Whitney–Wilcoxon (MWW/MWU), )
    DO NOT : use_continuity False AND method "auto" at the same time.
    because "auto" will set continuity depending on ties and sample size.
    If ties in the data  and method "exact" (i.e use_continuity False)
    pvalues cannot be calculated, check scipy doc
    """
    usta, p = scipy.stats.mannwhitneyu(
        vInterest,
        vBaseline,
        # method = "auto",
        use_continuity=False,
        alternative="less",
    )

    usta2, p2 = scipy.stats.mannwhitneyu(
        vInterest,
        vBaseline,
        # method = "auto",
        use_continuity=False,
        alternative="greater",
    )

    usta3, p3 = scipy.stats.mannwhitneyu(
        vInterest,
        vBaseline,
        # method = "auto",
        use_continuity=False,
        alternative="two-sided",
    )

    # best (smaller pvalue) among all tailed tests
    pretups = [(usta, p), (usta2, p2), (usta3, p3)]
    tups = []
    for t in pretups:  # make list of tuples with no-nan pvalues
        if not np.isnan(t[1]):
            tups.append(t)

    if len(tups) == 0:  # if all pvalues are nan assign two sided result
        tups = [(usta3, p3)]

    stap_tup = min(tups, key=lambda x: x[1])  # nan already excluded
    stat_result = stap_tup[0]
    pval_result = stap_tup[1]

    return stat_result, pval_result


# TODO: removed "stat" as it seems not to be used anywhere but creates conflits with disfit
def run_statistical_test(df: pd.DataFrame, comparison: List, test: str) -> pd.DataFrame:
    """
    This is a switch function for computing statistics for a pairwise differential analysis
    The comparison is a list with 2 sublists that contain column names
    """
    metabolites = df.index.values
    stare, pval = [], []
    for i in df.index:
        a1 = np.array(df.loc[i, comparison[0]], dtype=float)
        a2 = np.array(df.loc[i, comparison[1]], dtype=float)
        vInterest = a1[~np.isnan(a1)]
        vBaseline = a2[~np.isnan(a2)]

        if (len(vInterest) < 2) | (len(vBaseline) < 2):
            return pd.DataFrame(
                data={
                    "metabolite": metabolites,
                    "stat": [float("nan")] * len(metabolites),
                    "pvalue": [float("nan")] * len(metabolites),
                }
            )

        if test == "MW":
            stat_result, pval_result = compute_mann_whitney_allH0(vInterest, vBaseline)

        elif test == "Tt":
            stat_result, pval_result = scipy.stats.ttest_ind(vInterest, vBaseline, alternative="two-sided")

        elif test == "KW":
            stat_result, pval_result = scipy.stats.kruskal(vInterest, vBaseline)

        elif test == "ranksum":
            stat_result, pval_result = helpers.compute_ranksums_allH0(vInterest, vBaseline)

        elif test == "Wcox":
            # signed-rank test: one sample (independence),
            # or two paired or related samples
            stat_result, pval_result = helpers.compute_wilcoxon_allH0(vInterest, vBaseline)

        elif test == "BrMu":
            stat_result, pval_result = helpers.compute_brunnermunzel_allH0(vInterest, vBaseline)

        elif test == "prm-scipy":
            # test statistic is absolute geommean differences,
            # so "greater" satisfy
            prm_res = scipy.stats.permutation_test(
                (vInterest, vBaseline),
                statistic=helpers.absolute_geommean_diff,
                permutation_type="independent",
                vectorized=False,
                n_resamples=9999,
                batch=None,
                alternative="greater",
            )
            stat_result, pval_result = prm_res.statistic, prm_res.pvalue

        #        stare.append(stat_result)
        pval.append(pval_result)

    #    assert (len(metabolites) == len(stare))
    assert len(metabolites) == len(pval)
    return pd.DataFrame(
        data={
            "metabolite": metabolites,
            #                              "stat": stare,
            "pvalue": pval,
        }
    )


def auto_detect_tailway(good_df, best_distribution, args_param):
    min_pval_ = list()
    for tail_way in ["two-sided", "right-tailed"]:
        tmp = compute_p_value(good_df, tail_way, best_distribution, args_param)

        min_pval_.append(tuple([tail_way, tmp["pvalue"].min()]))

    return min(min_pval_, key=lambda x: x[1])[0]


# TODO: inside of run_distribution there is a hidden visualization,
#  removed it to avoid having to pass the file name deep in the call stack
#  instead just logging the distribution name and parameters
#  If plotting is needed, it has to be done in the 'visualization' folder
def run_distribution_fitting(df: pd.DataFrame):
    df = fit_statistical_distribution.compute_z_score(df, "FC")
    best_distribution, args_param = fit_statistical_distribution.find_best_distribution(df)
    # out_histogram_distribution=out_histo_file)
    autoset_tailway = auto_detect_tailway(df, best_distribution, args_param)
    logger.info(f"auto, best pvalues calculated : {autoset_tailway}")
    df = compute_p_value(df, autoset_tailway, best_distribution, args_param)

    return df


def compute_p_value(df: pd.DataFrame, test: str, best_dist, args_param) -> pd.DataFrame:
    if test == "right-tailed":
        df["pvalue"] = 1 - best_dist.cdf(df["zscore"], **args_param)
    elif test == "two-sided":
        df["pvalue"] = 2 * (1 - best_dist.cdf(abs(df["zscore"]), **args_param))
    else:
        print("WARNING: two-tailed or not")  # TODO: clarify the warning message
    return df


def compute_padj_version2(df, correction_alpha, correction_method):  # TODO: there is no version1, change name?
    tmp = df.copy()
    # inspired from R documentation in p.adjust :
    tmp["pvalue"] = tmp[["pvalue"]].fillna(1)

    (sgs, corrP, _, _) = ssm.multipletests(tmp["pvalue"], alpha=float(correction_alpha), method=correction_method)
    df = df.assign(padj=corrP)
    truepadj = []
    for v, w in zip(df["pvalue"], df["padj"]):
        if np.isnan(v):
            truepadj.append(v)
        else:
            truepadj.append(w)
    df = df.assign(padj=truepadj)

    return df


def filter_diff_results(ratiosdf, padj_cutoff, log2FC_abs_cutoff):
    ratiosdf["abslfc"] = ratiosdf["log2FC"].abs()
    ratiosdf = ratiosdf.loc[(ratiosdf["padj"] <= padj_cutoff) & (ratiosdf["abslfc"] >= log2FC_abs_cutoff), :]
    ratiosdf = ratiosdf.sort_values(["padj", "pvalue", "distance/span"], ascending=[True, True, False])
    ratiosdf = ratiosdf.drop(columns=["abslfc"])

    return ratiosdf


# TODO : the intention is not clear, why do this?
def reorder_columns_diff_end(df: pd.DataFrame) -> pd.DataFrame:
    standard_cols = [
        "count_nan_samples_group1",
        "count_nan_samples_group2",
        "distance",
        "span_allsamples",
        "distance/span",
        #        'stat',
        "pvalue",
        "padj",
        "log2FC",
        "FC",
        "compartment",
    ]

    desired_order = [
        "log2FC",
        #        'stat',
        "pvalue",
        "padj",
        "distance/span",
        "FC",
        "count_nan_samples_group1",
        "count_nan_samples_group2",
        "distance",
        "span_allsamples",
        "compartment",
    ]

    standard_df = df[standard_cols]
    df = df.drop(columns=standard_cols)
    # reorder the standard part
    standard_df = standard_df[desired_order]
    # re-join them, indexes are the metabolites
    df = pd.merge(standard_df, df, left_index=True, right_index=True, how="left")
    return df


def pairwise_comparison(
    df: pd.DataFrame, dataset: Dataset, cfg: DictConfig, comparison: List[str], test: availtest_methods_type
) -> pd.DataFrame:
    """
    Runs a pairwise comparison according to the comparison list in the analysis yaml file
    """
    conditions_list = helpers.first_column_for_column_values(
        df=dataset.metadata_df, columns=cfg.analysis.method.grouping, values=comparison
    )
    # flatten the list of lists and select the subset of column names present in the sub dataframe
    columns = [i for i in reduce(operator.concat, conditions_list) if i in df.columns]
    this_comparison = [list(filter(lambda x: x in columns, sublist)) for sublist in conditions_list]
    df4c = df[columns].copy()
    df4c = df4c[(df4c.T != 0).any()]  # delete rows being zero everywhere
    df4c = df4c.dropna(axis=0, how="all")
    df4c = helpers.row_wise_nanstd_reduction(df4c)
    df4c = helpers.countnan_samples(df4c, this_comparison)
    df4c = distance_or_overlap(df4c, this_comparison)
    df4c = compute_span_incomparison(df4c, this_comparison)
    df4c["distance/span"] = df4c.distance.div(df4c.span_allsamples)
    df4c = helpers.calculate_gmean(df4c, this_comparison)
    df_good, df_bad = select_rows_with_sufficient_non_nan_values(df4c)

    if test == "disfit":
        df_good = run_distribution_fitting(df_good)
    else:
        result_test_df = run_statistical_test(df_good, this_comparison, test)
        assert result_test_df.shape[0] == df_good.shape[0]
        result_test_df.set_index("metabolite", inplace=True)
        df_good = pd.merge(df_good, result_test_df, left_index=True, right_index=True)

    df_good["log2FC"] = np.log2(df_good["FC"])

    df_good, df_no_padj = helpers.split_rows_by_threshold(
        df_good, "distance/span", cfg.analysis.method.qualityDistanceOverSpan
    )
    df_good = compute_padj_version2(df_good, 0.05, cfg.analysis.method.correction_method)

    # re-integrate the "bad" sub-dataframes to the full dataframe
    result = helpers.concatenate_dataframes(df_good, df_bad, df_no_padj)
    return result


def differential_comparison(
    file_name: data_files_keys_type, dataset: Dataset, cfg: DictConfig, test: availtest_methods_type, out_table_dir: str
) -> None:
    """
    Differential comparison is performed on compartemnatalized versions of data files
    Attention: we replace zero values using the provided method
    Writes the table with computed statistics in the relevant output directory
    """
    assert_literal(test, availtest_methods_type, "Available test")
    assert_literal(file_name, data_files_keys_type, "file name")

    impute_value = cfg.analysis.method.impute_values[file_name]
    for compartment, compartmentalized_df in dataset.compartmentalized_dfs[file_name].items():
        val_instead_zero = helpers.arg_repl_zero2value(impute_value, compartmentalized_df)
        df = compartmentalized_df.replace(to_replace=0, value=val_instead_zero)

        for comparison in cfg.analysis.comparisons:
            if cfg.analysis.method.comparison_mode == "pairwise":
                result = pairwise_comparison(df, dataset, cfg, comparison, test)
                result["compartment"] = compartment
                result = reorder_columns_diff_end(result)
                result = result.sort_values(["padj", "distance/span"], ascending=[True, False])
                comp = "-".join(map(lambda x: "-".join(x), comparison))
                base_file_name = f"{dataset.get_file_for_label(file_name)}--{compartment}-{comp}-{test}"
                result.to_csv(
                    os.path.join(out_table_dir, f"{base_file_name}.tsv"),
                    index_label="metabolite",
                    header=True,
                    sep="\t",
                )
                # filtered by thresholds :
                filtered_df = filter_diff_results(
                    result, cfg.analysis.thresholds.padj, cfg.analysis.thresholds.absolute_log2FC
                )
                output_file_name = os.path.join(out_table_dir, f"{base_file_name}_filter.tsv")
                filtered_df.to_csv(output_file_name, index_label="metabolite", header=True, sep="\t")
                logger.info(f"Saved the result in {output_file_name}")

def multi_sample_compairson(
        file_name: data_files_keys_type, dataset: Dataset, cfg: DictConfig, out_table_dir: str
) -> None:
    '''
    Multi-sample comparison using Kruskal-Wallis non-parametric  method
    for comparing k independent samples.
    '''
    assert_literal(file_name, data_files_keys_type, "file name")

    impute_value = cfg.analysis.method.impute_values[file_name]
    for compartment, compartmentalized_df in dataset.compartmentalized_dfs[file_name].items():
        val_instead_zero = helpers.arg_repl_zero2value(impute_value, compartmentalized_df)
        df = compartmentalized_df.replace(to_replace=0, value=val_instead_zero)

    return