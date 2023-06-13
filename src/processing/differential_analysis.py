#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Johanna Galvis, Florian Specque, Macha Nikolski
"""

import argparse
import os
from typing import List

import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.stats.multitest as ssm
from functools import reduce
import operator
from omegaconf import DictConfig

from constants import availtest_methods, correction_methods, availtest_methods_type, assert_literal, \
    data_files_keys_type
from processing import fit_statistical_distribution

import helpers
from data import Dataset


def diff_args():
    parser = argparse.ArgumentParser(
        prog="python -m DIMet.src.differential_analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('config', type=str,
                        help="configuration file in absolute path")

    # Before reduction and gmean, way to replace zeros:
    parser.add_argument("--abundance_replace_zero_with",
                        default="min", type=helpers.zero_repl_arg,
                        help="chose a way to replace zero-value abundances \
(min | min/n | VALUE)")

    parser.add_argument("--meanEnrichOrFracContrib_replace_zero_with",
                        metavar="MEorFC_replace_zero_with",
                        default="min", type=helpers.zero_repl_arg,
                        help="chose a way to replace zero-value enrichments  \
                              (min | min/n | VALUE)")

    parser.add_argument("--isotopologueProp_replace_zero_with",
                        default="min", type=helpers.zero_repl_arg,
                        help="chose a way to replace zero-value proportions of \
                              isotopologues  (min | min/n | VALUE)")

    parser.add_argument("--isotopologueAbs_replace_zero_with",
                        default="min", type=helpers.zero_repl_arg,
                        help="chose a way to replace absolute zero-values of \
                              isotopologues (min | min/n | VALUE)")

    parser.add_argument(
        "--multitest_correction", default='fdr_bh', choices=multest_methods,
        help="see : https://www.statsmodels.org/dev/generated/\
              statsmodels.stats.multitest.multipletests.html")

    parser.add_argument(
        "--qualityDistanceOverSpan", default=-0.5, type=float,
        help="By metabolite, for samples (x, y) the distance is calculated,\
        and span is max(x U y) - min(x U y).  A 'distance/span' inferior\
        to this value excludes the metabolite from testing (pvalue=NaN).")

    parser.add_argument('--multiclass_analysis', choices=('KW', 'none'),
                        type=str, default='none',
                        help='chose a test for more than 2 classes')

    parser.add_argument('--time_course', choices=availtest_methods,
                        type=str, default='none',
                        help='chose a test for comparison of time-points')

    # by default include all the types of measurements:

    parser.add_argument('--abundances',
                        action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument('--meanEnrich_or_fracContrib',
                        action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument('--isotopologues',
                        action=argparse.BooleanOptionalAction, default=True)

    return parser


def verify_thresholds_defined(confidic) -> bool:
    if 'thresholds' not in confidic.keys():
        raise KeyError("thresholds --> NOT defined by user. Abort")

    expected_keys_sub = ['padj',
                         'absolute_log2FC']
    has_key = helpers.check_dict_has_keys(confidic['thresholds'], expected_keys_sub)
    missing_keys = np.array(expected_keys_sub)[~has_key].tolist()
    assert all(has_key), f"thresholds -> {missing_keys} : \
           missing in configuration file!"
    return True


def check_validity_configfile_diff2group(confidic: dict,
                                         metadatadf: pd.DataFrame) -> bool:
    expected_keys = ['grouping',
                     'comparisons',
                     'statistical_test']

    has_key = helpers.check_dict_has_keys(confidic, expected_keys)
    missing_keys = np.array(expected_keys)[~has_key].tolist()
    assert all(has_key), f"{missing_keys} : missing in configuration file! "
    for k in expected_keys:
        if k == 'grouping':
            if type(confidic[k]) is str:
                confidic[k] = [confidic[k]]
            for group in confidic[k]:
                if group not in metadatadf.columns:
                    raise ValueError(f'{group} grouping key not found '
                                     f'in metadata.')
        elif k == 'comparisons':
            if not all([len(ali) == 2 for ali in confidic[k]]):
                raise ValueError('comparisons should not involve more '
                                 'than 2 conditions')
        elif k == 'statistical_test':
            expected_keys_sub = ['abundances',
                                 'meanE_or_fracContrib',
                                 'isotopologue_abs',
                                 'isotopologue_prop']
            has_key = helpers.check_dict_has_keys(confidic[k], expected_keys_sub)
            missing_keys = np.array(expected_keys_sub)[~has_key].tolist()
            assert all(has_key), f"statistical_test -> {missing_keys} : \
                   missing in configuration file!"
            known_test = helpers.check_dict_has_known_values(
                confidic[k], availtest_methods + (None,))
            unknown_tests = np.array(list(confidic[k].values()))[~known_test]
            assert all(known_test), f'statistical_test -> {unknown_tests} : \
                   not valid tests!, available: {availtest_methods}'
        else:
            verify_thresholds_defined(confidic)

    return True


def flag_has_replicates(ratiosdf: pd.DataFrame):
    bool_results = list()
    tuples_list = ratiosdf['count_nan_samples'].tolist()
    for tup in tuples_list:
        group_x = tup[0].split("/")
        group_y = tup[1].split("/")
        x_usable = int(group_x[1]) - int(group_x[0])
        y_usable = int(group_y[1]) - int(group_y[0])
        if x_usable <= 1 or y_usable <= 1:
            # if any side only has one replicate
            bool_results.append(0)  # 0 for not usable
        else:
            bool_results.append(1)  # 1 for usable
    return bool_results


def compute_span_incomparison(df: pd.DataFrame, groups: List) -> pd.DataFrame:
    """
    For each row in the input DataFrame, computes the difference between the maximum and minimum values
    from columns in two sublists provided in the 'groups' list.
    Returns:
         DataFrame with an additional 'span_allsamples' column containing the computed differences.
    """
    for i in df.index.values:
        all_values = list(df.loc[i, groups[0]] )+ list(df.loc[i, groups[1]])

        interval = max(all_values) - min(all_values)
        df.loc[i, 'span_allsamples'] = interval

    return df

def calc_reduction(df, metad4c):
    def renaming_original_col_sams(df):
        newcols = ["input_" + i for i in df.columns]
        df.columns = newcols

        return df

    ddof = 0  # for compute reduction
    df4c = df[metad4c['name_to_plot']]

    df4c = helpers.give_reduced_df(df4c, ddof)

    df_orig_vals = renaming_original_col_sams(df[metad4c['name_to_plot']])

    df4c = pd.merge(df_orig_vals, df4c, left_index=True, right_index=True)

    return df4c


def calc_ratios(df4c: pd.DataFrame, groups: List) -> pd.DataFrame:
    df4c = helpers.calculate_gmean(df4c, groups)
    # df4c, col_g_interest, col_g_control = helpers.give_geommeans_new(
    #     df4c, metad4c, 'newcol', c_interest, c_control)
    # df4c = helpers.give_ratios_df(df4c, col_g_interest, col_g_control)

    return df4c


def divide_groups(df4c, metad4c, selected_contrast):
    """split into two df"""
    sc = selected_contrast
    sam0 = metad4c.loc[metad4c['newcol'] == sc[0], 'name_to_plot']  # interest
    sam1 = metad4c.loc[metad4c['newcol'] == sc[1], 'name_to_plot']  # control
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
        df.at[i, 'distance'] = helpers.compute_distance_between_intervals(group1, group2, overlap_method)

    return df


def select_rows_with_sufficient_non_nan_values(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
    Identifies rows in the input DataFrame that have enough replicates.
    Separates the DataFrame into two parts based on the presence or absence of enough replicates,
    returning them as separate DataFrames.
    """
    try:
        bad_df = df[(df['count_nan_samples_group1'] > 0) | (df['count_nan_samples_group2'] > 0)]
        good_df = df[(df['count_nan_samples_group1'] == 0) & (df['count_nan_samples_group2'] == 0)]
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
    usta, p = scipy.stats.mannwhitneyu(vInterest, vBaseline,
                                       # method = "auto",
                                       use_continuity=False,
                                       alternative="less")

    usta2, p2 = scipy.stats.mannwhitneyu(vInterest, vBaseline,
                                         # method = "auto",
                                         use_continuity=False,
                                         alternative="greater")

    usta3, p3 = scipy.stats.mannwhitneyu(vInterest, vBaseline,
                                         # method = "auto",
                                         use_continuity=False,
                                         alternative="two-sided")

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
    sta, p = scipy.stats.ranksums(vInterest, vBaseline,
                                  alternative="less")
    sta2, p2 = scipy.stats.ranksums(vInterest, vBaseline,
                                    alternative="greater")
    sta3, p3 = scipy.stats.ranksums(vInterest, vBaseline,
                                    alternative="two-sided")

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
    sta, p = scipy.stats.wilcoxon(vInterest, vBaseline,
                                  alternative="less")
    sta2, p2 = scipy.stats.wilcoxon(vInterest, vBaseline,
                                    alternative="greater")
    sta3, p3 = scipy.stats.wilcoxon(vInterest, vBaseline,
                                    alternative="two-sided")

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
    sta, p = scipy.stats.brunnermunzel(vInterest, vBaseline,
                                       alternative="less")
    sta2, p2 = scipy.stats.brunnermunzel(vInterest, vBaseline,
                                         alternative="greater")
    sta3, p3 = scipy.stats.brunnermunzel(vInterest, vBaseline,
                                         alternative="two-sided")

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


def statistic_absolute_geommean_diff(b_values: np.array, a_values: np.array):
    m_b = helpers.compute_gmean_nonan(b_values)
    m_a = helpers.compute_gmean_nonan(a_values)
    # denom = m_a + m_b
    # diff_normalized = abs((m_b - m_a) / denom)
    diff_absolute = abs(m_b - m_a)
    return diff_absolute


def run_statistical_test(df: pd.DataFrame, dataset: Dataset, cfg: DictConfig,
                         comparison: List, test: str):  # redu_df, metas, contrast, whichtest):
    """
    This is a switch function for running a pairwise differential analysis statistical test
    """
    metabolites = df.index.values
    stare, pval = [], []
    for i in df.index:
        a1 = np.array(df.loc[i, comparison[0]], dtype=float)
        a2 = np.array(df.loc[i, comparison[1]], dtype=float)
        vInterest = a1[~np.isnan(a1)]
        vBaseline = a2[~np.isnan(a2)]

        if (len(vInterest) < 2) | (len(vBaseline) < 2):
            return pd.DataFrame(data={"metabolite": metabolites,
                                      "stat": [float('nan')] * len(metabolites),
                                      "pvalue": [float('nan')] * len(metabolites)})

        if test == "MW":
            stat_result, pval_result = compute_mann_whitney_allH0(
                vInterest, vBaseline)

        elif test == "Tt":
            stat_result, pval_result = scipy.stats.ttest_ind(
                vInterest, vBaseline, alternative="two-sided")

        elif test == "KW":
            stat_result, pval_result = scipy.stats.kruskal(
                vInterest, vBaseline)

        elif test == "ranksum":
            stat_result, pval_result = compute_ranksums_allH0(
                vInterest, vBaseline)

        elif test == "Wcox":
            # signed-rank test: one sample (independence),
            # or two paired or related samples
            stat_result, pval_result = compute_wilcoxon_allH0(
                vInterest, vBaseline)

        elif test == "BrMu":
            stat_result, pval_result = compute_brunnermunzel_allH0(
                vInterest, vBaseline)

        elif test == "prm-scipy":
            # test statistic is absolute geommean differences,
            # so "greater" satisfy
            prm_res = scipy.stats.permutation_test(
                (vInterest, vBaseline),
                statistic=statistic_absolute_geommean_diff,
                permutation_type='independent',
                vectorized=False,
                n_resamples=9999,
                batch=None,
                alternative='greater')
            stat_result, pval_result = prm_res.statistic, prm_res.pvalue

        stare.append(stat_result)
        pval.append(pval_result)

    assert (len(metabolites) == len(stare))
    assert (len(metabolites) == len(pval))
    return pd.DataFrame(data={"metabolite": metabolites,
                              "stat": stare,
                              "pvalue": pval})


def steps_fitting_method(ratiosdf, out_histo_file):
    def auto_detect_tailway(good_df, best_distribution, args_param):
        min_pval_ = list()
        for tail_way in ["two-sided", "right-tailed"]:
            tmp = fit_statistical_distribution.compute_p_value(good_df, tail_way,
                                                               best_distribution, args_param)

            min_pval_.append(tuple([tail_way, tmp["pvalue"].min()]))

        return min(min_pval_, key=lambda x: x[1])[0]

    ratiosdf = fit_statistical_distribution.compute_z_score(ratiosdf)
    best_distribution, args_param = fit_statistical_distribution.find_best_distribution(
        ratiosdf,
        out_histogram_distribution=out_histo_file)
    autoset_tailway = auto_detect_tailway(ratiosdf,
                                          best_distribution, args_param)
    print("auto, best pvalues calculated :", autoset_tailway)
    ratiosdf = compute_p_value(ratiosdf, autoset_tailway,
                               best_distribution, args_param)

    return ratiosdf


def compute_padj_version2(df, correction_alpha, correction_method):
    tmp = df.copy()
    # inspired from R documentation in p.adjust :
    tmp["pvalue"] = tmp[["pvalue"]].fillna(1)

    (sgs, corrP, _, _) = ssm.multipletests(tmp["pvalue"],
                                           alpha=float(correction_alpha),
                                           method=correction_method)
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
    ratiosdf['abslfc'] = ratiosdf['log2FC'].abs()
    ratiosdf = ratiosdf.loc[(ratiosdf['padj'] <= padj_cutoff) &
                            (ratiosdf['abslfc'] >= log2FC_abs_cutoff), :]
    ratiosdf = ratiosdf.sort_values(['padj', 'pvalue', 'distance/span'],
                                    ascending=[True, True, False])
    ratiosdf = ratiosdf.drop(columns=['abslfc'])

    return ratiosdf


def reorder_columns_diff_end(df: pd.DataFrame) -> pd.DataFrame:
    standard_cols = [
        'count_nan_samples_group1',
        'count_nan_samples_group2',
        'distance',
        'span_allsamples',
        'distance/span',
        'stat',
        'pvalue',
        'padj',
        'log2FC',
        'FC',
        'compartment']

    desired_order = [
        'log2FC',
        'stat',
        'pvalue',
        'padj',
        'distance/span',
        'FC',
        'count_nan_samples_group1',
        'count_nan_samples_group2',
        'distance',
        'span_allsamples',
        'compartment']

    standard_df = df[standard_cols]
    df = df.drop(columns=standard_cols)
    # reorder the standard part
    standard_df = standard_df[desired_order]
    # re-join them, indexes are the metabolites
    df = pd.merge(standard_df, df, left_index=True,
                  right_index=True, how='left')
    return df


def pairwise_comparison(df: pd.DataFrame, dataset: Dataset, cfg: DictConfig,
                        comparison: List[str], test: availtest_methods_type) -> pd.DataFrame:
    '''
    Runs a pairwise comparison according to the comparison list in the analysis yaml file
    '''
    conditions_list = helpers.select_rows_by_fixed_values(df=dataset.metadata_df,
                                                          columns=cfg.analysis.method.grouping,
                                                          values=comparison)
    # flatten the list of lists and select the subset of column names present in the sub dataframe
    columns = [i for i in reduce(operator.concat, conditions_list) if i in df.columns]
    this_comparison = [list(filter(lambda x: x in columns, sublist)) for sublist in conditions_list]
    df4c = df[columns]
    df4c = df4c[(df4c.T != 0).any()]  # delete rows being zero everywhere
    df4c = df4c.dropna(axis=0, how='all')
    df4c = helpers.countnan_samples(df4c, this_comparison)
    df4c = distance_or_overlap(df4c, this_comparison)
    df4c = compute_span_incomparison(df4c, this_comparison)
    df4c['distance/span'] = df4c.distance.div(df4c.span_allsamples)
    df4c = helpers.calculate_gmean(df4c, this_comparison)
    df_good, df_bad = select_rows_with_sufficient_non_nan_values(df4c)

    if test == "disfit":
        df_good = steps_fitting_method(df_good, dataset, cfg)
    else:
        result_test_df = run_statistical_test(df_good, dataset, cfg, this_comparison, test)
        assert (result_test_df.shape[0] == df_good.shape[0])
        result_test_df.set_index("metabolite", inplace=True)
        df_good = pd.merge(df_good, result_test_df,
                           left_index=True, right_index=True)

    df_good["log2FC"] = np.log2(df_good['FC'])

    df_good, df_no_padj = helpers.split_rows_by_threshold(df_good, 'distance/span',
                                                          cfg.analysis.method.qualityDistanceOverSpan)
    df_good = compute_padj_version2(df_good, 0.05,
                                    cfg.analysis.method.correction_method)

    # re-integrate the "bad" sub-dataframes to the full dataframe
    result = helpers.concatenate_dataframes(df_good, df_bad, df_no_padj)
    return result


def run_multiclass(measurements: pd.DataFrame, metadatadf: pd.DataFrame,
                   out_file_elements: dict, confidic,
                   method_multiclass, args) -> None:
    out_dir = out_file_elements['odir']
    prefix = out_file_elements['prefix']
    co = out_file_elements['co']
    suffix = out_file_elements['suffix']

    helpers.detect_and_create_dir(f"{out_dir}/extended/")
    helpers.detect_and_create_dir(f"{out_dir}/filtered/")

    def create_dictio_arrays(row, metadata):
        all_classes = metadata['condition'].unique().tolist()
        dico_out = dict()
        for group in all_classes:
            samples = metadata.loc[
                metadata['condition'] == group, 'name_to_plot']
            dico_out[group] = row[samples].to_numpy()
        return dico_out

    def compute_multiclass_KW(df, metadata):
        for i, row in df.iterrows():
            the_dictionary_of_arrays = create_dictio_arrays(row, metadata)
            # using kruskal on three or more groups
            stat_result, pval_result = scipy.stats.kruskal(
                *the_dictionary_of_arrays.values(), axis=0,
                nan_policy='omit')
            df.at[i, 'statistic'] = stat_result
            df.at[i, 'pvalue'] = pval_result
        return df

    # separate by timepoint, when several
    timepoints = metadatadf['timepoint'].unique().tolist()
    for tpoint in timepoints:
        metada_tp = metadatadf.loc[metadatadf['timepoint'] == tpoint, :]
        measures_tp = measurements[metada_tp['name_to_plot']]
        measures_tp = measures_tp[(measures_tp.T != 0).any()]  # being zero
        measures_tp = measures_tp.dropna(axis=0, how='all')
        measures_tp = calc_reduction(measures_tp, metada_tp)

        if method_multiclass == "KW":
            multi_result = compute_multiclass_KW(measures_tp, metada_tp)

        multi_result = compute_padj_version2(
            multi_result, 0.05, args.multitest_correction)

        # reorder the columns:
        input_cols = [i for i in multi_result.columns if i.startswith("input")]
        resu_cols = ['statistic', 'pvalue', 'padj']
        remaining_cols = [i for i in multi_result.columns if
                          (i not in resu_cols) & (not i.startswith("input_"))]
        new_order = resu_cols
        new_order.extend(remaining_cols)
        new_order.extend(input_cols)
        multi_result = multi_result[new_order]
        preotsv = f'{prefix}--{co}--{tpoint}-{method_multiclass}'
        otsv = f"{preotsv}-multiclass-{suffix}.tsv"
        multi_result.to_csv(
            f"{out_dir}/extended/{otsv}",
            index_label="metabolite", header=True, sep='\t')
        # filtered by thresholds :
        padj_cutoff = confidic['thresholds']['padj']
        filtered_df = multi_result.loc[multi_result['padj'] <= padj_cutoff]
        preotv = f'{prefix}--{co}--{tpoint}-{method_multiclass}'
        otv = f'{preotv}-multiclass-{suffix}.tsv'
        filtered_df.to_csv(f"{out_dir}/filtered/{otv}",
                           index_label="metabolite",
                           header=True, sep='\t')


def run_time_course(measurements: pd.DataFrame,
                    metadatadf: pd.DataFrame, out_file_elements: dict,
                    confidic: dict, whichtest: str, args) -> None:
    out_dir = out_file_elements['odir']
    prefix = out_file_elements['prefix']
    co = out_file_elements['co']
    suffix = out_file_elements['suffix']

    helpers.detect_and_create_dir(f"{out_dir}/extended/")
    helpers.detect_and_create_dir(f"{out_dir}/filtered/")

    for condition in metadatadf['condition'].unique().tolist():
        metada_cd = metadatadf.loc[metadatadf['condition'] == condition, :]

        auto_set_comparisons_l = list()
        ordered_timenum = metada_cd['timenum'].unique()  # already is numpy
        ordered_timenum = np.sort(np.array(ordered_timenum))

        for h in range(len(ordered_timenum) - 1):
            contrast = [str(ordered_timenum[h + 1]), str(ordered_timenum[h])]
            auto_set_comparisons_l.append(contrast)

        for contrast in auto_set_comparisons_l:
            strcontrast = '_'.join(contrast)
            metada_cd = metada_cd.assign(
                timenum=metada_cd["timenum"].astype('str'))
            df4c, metad4c = helpers.prepare4contrast(measurements, metada_cd,
                                                     ['timenum'], contrast)

            df4c = df4c[(df4c.T != 0).any()]  # delete rows being zero every
            df4c = df4c.dropna(axis=0, how='all')
            # sort them by 'newcol' the column created by prepare4contrast
            metad4c = metad4c.sort_values("newcol")
            df4c = calc_reduction(df4c, metad4c)
            # adds nan_count_samples column :
            df4c = helpers.countnan_samples(df4c, metad4c)
            df4c = distance_or_overlap(df4c, metad4c, contrast)
            df4c = compute_span_incomparison(df4c, metad4c, contrast)
            df4c['distance/span'] = df4c.distance.div(df4c.span_allsamples)
            ratiosdf = calc_ratios(df4c, metad4c, contrast)
            ratiosdf, df_bad = select_rows_with_sufficient_non_nan_values(ratiosdf)

            result_test_df = run_statistical_test(ratiosdf, metad4c,
                                                  contrast, whichtest)
            result_test_df.set_index("metabolite", inplace=True)
            ratiosdf = pd.merge(ratiosdf, result_test_df,
                                left_index=True, right_index=True)

            ratiosdf["log2FC"] = np.log2(ratiosdf['FC'])

            ratiosdf, df_no_padj = separate_before_compute_padj(
                ratiosdf,
                args.qualityDistanceOverSpan)
            ratiosdf = compute_padj_version2(ratiosdf, 0.05,
                                             args.multitest_correction)

            # re-integrate the "bad" sub-dataframes to the full dataframe
            df_bad = complete_columns_for_bad(df_bad, ratiosdf)
            df_no_padj = complete_columns_for_bad(df_no_padj, ratiosdf)
            ratiosdf = pd.concat([ratiosdf, df_no_padj])
            if df_bad.shape[0] >= 1:
                ratiosdf = pd.concat([ratiosdf, df_bad])

            ratiosdf["compartment"] = co
            ratiosdf = reorder_columns_diff_end(ratiosdf)
            ratiosdf = ratiosdf.sort_values(['padj', 'distance/span'],
                                            ascending=[True, False])
            preotsv = f'{prefix}--{co}--{condition}-{strcontrast}'
            otsv = f"{preotsv}-{whichtest}-{suffix}.tsv"
            ratiosdf.to_csv(
                f"{out_dir}/extended/{otsv}",
                index_label="metabolite", header=True, sep='\t')
            # filtered by thresholds :
            filtered_df = filter_diff_results(
                ratiosdf,
                confidic['thresholds']['padj'],
                confidic['thresholds']['absolute_log2FC'])

            preotv = f'{prefix}--{co}--{condition}-{strcontrast}'
            otv = f'{preotv}-{whichtest}-{suffix}_filter.tsv'

            filtered_df.to_csv(f"{out_dir}/filtered/{otv}",
                               index_label="metabolite",
                               header=True, sep='\t')


def multiclass_andor_timecourse_andor_diff2groups(
        measurements, meta_co, out_file_elems,
        confidic, modes_specs, mode, args):
    method_multiclass = args.multiclass_analysis
    if method_multiclass and method_multiclass.lower() != "none":
        out_file_elems['odir'] = confidic['out_path'] + \
                                 "results/multiclass_analysis/" + \
                                 out_file_elems['modedir'] + '/'

        helpers.detect_and_create_dir(f"{out_file_elems['odir']}/extended/")
        helpers.detect_and_create_dir(f"{out_file_elems['odir']}/filtered/")

        run_multiclass(measurements, meta_co, out_file_elems,
                       confidic, method_multiclass, args)

    method_time_course = args.time_course
    if method_time_course and method_time_course.lower() != "none":
        out_file_elems['odir'] = confidic['out_path'] + \
                                 "results/timecourse_analysis/" + \
                                 out_file_elems['modedir'] + '/'

        helpers.detect_and_create_dir(f"{out_file_elems['odir']}/extended/")
        helpers.detect_and_create_dir(f"{out_file_elems['odir']}/filtered/")

        run_time_course(measurements, meta_co, out_file_elems,
                        confidic, method_time_course, args)

    if 'grouping' in confidic.keys():
        out_file_elems['odir'] = confidic['out_path'] + \
                                 "results/differential_analysis/" + \
                                 out_file_elems['modedir'] + '/'

        helpers.detect_and_create_dir(f"{out_file_elems['odir']}/extended/")
        helpers.detect_and_create_dir(f"{out_file_elems['odir']}/filtered/")

        whichtest = confidic['statistical_test'][modes_specs[mode]['test_key']]

        run_differential_steps(measurements, meta_co, out_file_elems,
                               confidic, whichtest, args)


def differential_comparison(file_name: data_files_keys_type, dataset: Dataset, cfg: DictConfig,
                            test: availtest_methods_type, out_table_dir: str) -> None:
    '''
    Differential comparison is performed on compartemnatalized versions of data files
    Moreover, we replace zero values using the provided method
    '''
    assert_literal(test, availtest_methods_type, 'Available test')
    assert_literal(file_name, data_files_keys_type, 'file name')

    impute_value = cfg.analysis.method.impute_values[file_name]
    for compartment, compartmentalized_df in dataset.compartmentalized_dfs[file_name].items():
        val_instead_zero = helpers.arg_repl_zero2value(impute_value, compartmentalized_df)
        df = compartmentalized_df.replace(to_replace=0, value=val_instead_zero)

        for comparison in cfg.analysis.comparisons:
            if cfg.analysis.method.comparison_mode == 'pairwise':
                result = pairwise_comparison(df, dataset, cfg, comparison, test)
                result["compartment"] = compartment
                result = reorder_columns_diff_end(result)
                result = result.sort_values(['padj', 'distance/span'],
                                            ascending=[True, False])
                strcontrast = "-".join(map(lambda x: "-".join(x), comparison))
                base_file_name = f"{dataset.get_file_for_label(file_name)}--{compartment}--{cfg.analysis.dataset.suffix}-{strcontrast}_{test}"
                result.to_csv(
                    os.path.join(out_table_dir, f"{base_file_name}.tsv"),
                    index_label="metabolite", header=True, sep='\t')
                # filtered by thresholds :
                filtered_df = filter_diff_results(
                    result,
                    cfg.analysis.thresholds.padj,
                    cfg.analysis.thresholds.absolute_log2FC)
                filtered_df.to_csv(os.path.join(out_table_dir, f"{base_file_name}_filter.tsv"),
                                   index_label="metabolite",
                                   header=True, sep='\t')


def pass_confidic_timecourse_multiclass_to_arg(confidic, args):
    if (args.time_course != 'none') and ('time_course' in confidic.keys()):
        raise ValueError("not allowed to set time_course twice: in .yml file"
                         "and arg")
    if (args.multiclass_analysis != 'none') and \
            ('multiclass_analysis' in confidic.keys()):
        raise ValueError("not allowed to set multiclass_analysis "
                         "twice: .yml file and arg")
    if args.time_course == 'none':
        try:
            args.time_course = confidic['time_course']
        except KeyError:
            pass
    if args.multiclass_analysis == 'none':
        try:
            args.multiclass_analysis = confidic['multiclass_analysis']
        except KeyError:
            pass
    return args


def check_at_least_one_method_demanded(confidic, args) -> None:
    diff_bool, time_bool, multiclass_bool = False, False, False
    try:
        confidic['grouping']
        intends_2groups_analysis = True
    except KeyError:
        intends_2groups_analysis = False
        pass

    if verify_thresholds_defined(confidic):
        if intends_2groups_analysis:
            diff_bool = check_validity_configfile_diff2group(
                confidic, metadatadf)

        if args.multiclass_analysis != 'none':
            multiclass_bool = True

        if args.time_course != 'none':
            time_bool = True

    at_least_one_method_user = sum([diff_bool, multiclass_bool, time_bool])
    assert at_least_one_method_user > 0, "No methods chosen, abort"
