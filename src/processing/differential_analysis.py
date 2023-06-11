#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Johanna Galvis, Florian Specque, Macha Nikolski
"""

import argparse
import os

import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.stats.multitest as ssm
from botocore.config import Config

from constants import availtest_methods, correction_methods, availtest_methods_type, assert_literal, \
    data_files_keys_type
import fit_statistical_distribution

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


def compute_span_incomparison(df: pd.DataFrame, metadata: pd.DataFrame,
                              contrast: list):
    expected_samples = metadata.loc[metadata['newcol'].isin(contrast),
                                    'name_to_plot']
    selcols_df = df[expected_samples].copy()
    for i in df.index.values:
        values_this_comparison = selcols_df.loc[i, :].to_numpy()
        span = values_this_comparison.max() - values_this_comparison.min()
        df.loc[i, 'span_allsamples'] = span

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


def calc_ratios(df4c, metad4c, selected_contrast):
    c_interest = selected_contrast[0]  # columns names interest
    c_control = selected_contrast[1]  # columns names control

    df4c, col_g_interest, col_g_control = helpers.give_geommeans_new(
        df4c, metad4c, 'newcol', c_interest, c_control)
    df4c = helpers.give_ratios_df(df4c, col_g_interest, col_g_control)

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


def distance_or_overlap(df4c, metad4c, selected_contrast):
    """ calculate distance between groups (synonym: overlap) """
    groupinterest, groupcontrol = divide_groups(df4c, metad4c,
                                                selected_contrast)
    rownames = df4c.index
    tmp_df = df4c.copy()
    tmp_df.index = range(len(rownames))
    tmp_df = helpers.compute_distance_between_intervals(tmp_df, groupcontrol, groupinterest, "symmetric")
    tmp_df.columns = [*tmp_df.columns[:-1], "distance"]
    tmp_df.index = rownames
    df4c = tmp_df.copy()

    return df4c


def separate_before_stats(ratiosdf):
    ratiosdf['has_replicates'] = flag_has_replicates(ratiosdf)
    try:
        # has_replicates: 1 true, is usable
        good_df = ratiosdf.loc[ratiosdf['has_replicates'] == 1, :]
        undesired_mets = set(ratiosdf.index) - set(good_df.index)
        bad_df = ratiosdf.loc[list(undesired_mets)]
        good_df = good_df.drop(columns=['has_replicates'])
        bad_df = bad_df.drop(columns=['has_replicates'])
    except Exception as e:
        print(e)
        print("Error in separate_before_stats (enough replicates ?)")

    return good_df, bad_df


def separate_before_compute_padj(ratiosdf, quality_dist_span):
    try:
        quality_dist_span = float(quality_dist_span)
        good_df = ratiosdf.loc[
                  ratiosdf['distance/span'] >= quality_dist_span, :]

        undesired_mets = set(ratiosdf.index) - set(good_df.index)
        bad_df = ratiosdf.loc[list(undesired_mets)]
    except Exception as e:
        print(e)
        print("Error in separate_before_comp_padj",
              " check qualityDistanceOverSpan arg")

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


def run_statistical_test(redu_df, metas, contrast, whichtest):
    """
    Parameters
    ----------
    redu_df : pandas, reduced dataframe
    metas : pandas, 2nd element output of prepare4contrast.
    contrast : a list
    Returns
    -------
    DIFFRESULT : pandas
        THE PAIR-WISE DIFFERENTIAL ANALYSIS RESULTS.
    """
    mets = []
    stare = []
    pval = []
    metaboliteshere = redu_df.index
    for i in metaboliteshere:
        mets.append(i)
        row = redu_df.loc[i, :]  # row is a series, colnames pass to index

        columnsInterest = metas.loc[metas["newcol"] == contrast[0],
                                    'name_to_plot']
        columnsBaseline = metas.loc[metas["newcol"] == contrast[1],
                                    'name_to_plot']

        vInterest = np.array(row[columnsInterest], dtype=float)
        vBaseline = np.array(row[columnsBaseline], dtype=float)

        vInterest = vInterest[~np.isnan(vInterest)]  # exclude nan elements
        vBaseline = vBaseline[~np.isnan(vBaseline)]  # exclude nan elements

        if (len(vInterest) >= 2) and (len(vBaseline) >= 2):

            if whichtest == "MW":
                stat_result, pval_result = compute_mann_whitney_allH0(
                    vInterest, vBaseline)

            elif whichtest == "Tt":
                stat_result, pval_result = scipy.stats.ttest_ind(
                    vInterest, vBaseline, alternative="two-sided")

            elif whichtest == "KW":
                stat_result, pval_result = scipy.stats.kruskal(
                    vInterest, vBaseline)

            elif whichtest == "ranksum":
                stat_result, pval_result = compute_ranksums_allH0(
                    vInterest, vBaseline)

            elif whichtest == "Wcox":
                # signed-rank test: one sample (independence),
                # or two paired or related samples
                stat_result, pval_result = compute_wilcoxon_allH0(
                    vInterest, vBaseline)

            elif whichtest == "BrMu":
                stat_result, pval_result = compute_brunnermunzel_allH0(
                    vInterest, vBaseline)

            elif whichtest == "prm-scipy":
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

        else:
            stare.append(np.nan)
            pval.append(np.nan)

        # end if
    # end for
    prediffr = pd.DataFrame(data={"metabolite": mets,
                                  "stat": stare,
                                  "pvalue": pval})
    return prediffr


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


def complete_columns_for_bad(bad_df, ratiosdf):
    columns_missing = set(ratiosdf.columns) - set(bad_df.columns)
    for col in list(columns_missing):
        bad_df[col] = np.nan
    return bad_df


def filter_diff_results(ratiosdf, padj_cutoff, log2FC_abs_cutoff):
    ratiosdf['abslfc'] = ratiosdf['log2FC'].abs()
    ratiosdf = ratiosdf.loc[(ratiosdf['padj'] <= padj_cutoff) &
                            (ratiosdf['abslfc'] >= log2FC_abs_cutoff), :]
    ratiosdf = ratiosdf.sort_values(['padj', 'pvalue', 'distance/span'],
                                    ascending=[True, True, False])
    ratiosdf = ratiosdf.drop(columns=['abslfc'])

    return ratiosdf


def reorder_columns_diff_end(df):
    standard_cols = [
        'count_nan_samples',
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
        'count_nan_samples',
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



def pairwise_comparison(df: pd.DataFrame, cfg: Config) -> None:
    '''
    Runs a pairwise comparison according to grouping and comparisons in the analysis yaml file
    '''
# def run_differential_steps(measurements: pd.DataFrame,
#                            metadatadf: pd.DataFrame, out_file_elements: dict,
#                            confidic: dict, whichtest: str, args) -> None:
#     out_dir = out_file_elements['odir']
#     prefix = out_file_elements['prefix']
#     co = out_file_elements['co']
#     suffix = out_file_elements['suffix']

    for contrast in cfg.analysis.method.comparisons:
        strcontrast = '_'.join(contrast)
        df4c, metad4c = helpers.prepare4contrast(measurements, metadatadf,
                                            confidic['grouping'], contrast)
        df4c = df4c[(df4c.T != 0).any()]  # delete rows being zero everywhere
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
        ratiosdf, df_bad = separate_before_stats(ratiosdf)

        if whichtest == "disfit":
            opdf = f"{prefix}--{co}--{suffix}-{strcontrast}_fitdist_plot.pdf"
            out_histo_file = f"{out_dir}/extended/{opdf}"
            ratiosdf = steps_fitting_method(ratiosdf, out_histo_file)

        else:
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
        otsv = f"{prefix}--{co}--{suffix}-{strcontrast}-{whichtest}.tsv"
        ratiosdf.to_csv(
            f"{out_dir}/extended/{otsv}",
            index_label="metabolite", header=True, sep='\t')
        # filtered by thresholds :
        filtered_df = filter_diff_results(
            ratiosdf,
            confidic['thresholds']['padj'],
            confidic['thresholds']['absolute_log2FC'])
        otv = f'{prefix}--{co}--{suffix}-{strcontrast}-{whichtest}_filter.tsv'
        filtered_df.to_csv(f"{out_dir}/filtered/{otv}",
                           index_label="metabolite",
                           header=True, sep='\t')


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
            multi_result = compute_multiclass_KW(measures_tp,  metada_tp)

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

        for h in range(len(ordered_timenum)-1):
            contrast = [str(ordered_timenum[h+1]), str(ordered_timenum[h])]
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
            ratiosdf, df_bad = separate_before_stats(ratiosdf)

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
                                 out_file_elems['modedir']+'/'

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


#def perform_tests(mode: str, clean_tables_path, table_prefix,
                  #metadatadf, confidic, args) -> None:
def differential_comparison_test(file_name: data_files_keys_type, test: availtest_methods_type,
                                 dataset: Dataset, cfg: Config) -> None:
    '''
    Differential comparison is performed on compartemnatalized versions of data files
    Moreover, we replace zero values using the provided method
    '''
    assert_literal(test, availtest_methods_type)
    assert_literal(file_name, data_files_keys_type)

    impute_value = cfg.analysis.method.impute_value[file_name]
    for compartment, compartemantalized_df in dataset.compartmentalized_dfs[file_name].items():
        val_instead_zero = helpers.arg_repl_zero2value(impute_value, compartemantalized_df)
        df = compartemantalized_df.replace(to_replace=0, value=val_instead_zero)

        # out_file_elems = {'modedir': modes_specs[mode]['dir'],
        #                   'prefix': table_prefix,
        #                   'co': c,
        #                   'suffix': suffix}
        #
        # multiclass_andor_timecourse_andor_diff2groups(
        #     measurements, meta_co, out_file_elems,
        #     confidic, modes_specs, mode, args)

        if cfg.analysis.method.comparisons.comparison_mode == 'pairwise':
            pairwise_comparison(df, cfg)

def pass_confidic_timecourse_multiclass_to_arg(confidic, args):
    if (args.time_course != 'none') and ('time_course' in confidic.keys()):
        raise ValueError("not allowed to set time_course twice: in .yml file"
                         "and arg")
    if (args.multiclass_analysis != 'none') and\
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

#
# if __name__ == "__main__":
#     print("\n  -*- searching for   \
#            Differentially Abundant-or-Marked Metabolites (DAM) -*-\n")
#     parser = diff_args()
#     args = parser.parse_args()
#
#     configfile = os.path.expanduser(args.config)
#     confidic = helpers.open_config_file(configfile)
#     helpers.auto_check_validity_configuration_file(confidic)
#     confidic = helpers.remove_extensions_names_measures(confidic)
#
#     args = pass_confidic_timecourse_multiclass_to_arg(confidic, args)
#
#     out_path = os.path.expanduser(confidic['out_path'])
#     confidic['out_path'] = out_path
#     meta_path = os.path.expanduser(confidic['metadata_path'])
#     clean_tables_path = out_path + "results/prepared_tables/"
#
#     metadatadf = helpers.open_metadata(meta_path)
#
#     check_at_least_one_method_demanded(confidic, args)
#
#     # 1- abund
#     if args.abundances:
#         print("processing abundances")
#         mode = "abund"
#         abund_tab_prefix = confidic['name_abundance']
#         perform_tests(mode, clean_tables_path, abund_tab_prefix,
#                       metadatadf, confidic, args)
#
#     # 2- ME or FC
#     if args.meanEnrich_or_fracContrib:
#         print("processing mean enrichment or fractional contributions")
#         mode = "mefc"
#         fraccon_tab_prefix = confidic['name_meanE_or_fracContrib']
#         perform_tests(mode, clean_tables_path, fraccon_tab_prefix,
#                       metadatadf, confidic, args)
#
#     # 3- isotopologues
#     if args.isotopologues:
#         mode = "isoabsol"
#         isos_abs_tab_prefix = confidic['name_isotopologue_abs']
#         isos_prop_tab_prefix = confidic['name_isotopologue_prop']
#         if (isos_abs_tab_prefix is not np.nan) and \
#                 (isos_abs_tab_prefix != "None") and \
#                 (isos_abs_tab_prefix is not None):
#             print("processing absolute isotopologues")
#             perform_tests(mode, clean_tables_path, isos_abs_tab_prefix,
#                           metadatadf, confidic, args)
#         else:
#             print("processing isotopologues (values given as proportions)")
#             mode = "isoprop"
#             perform_tests(mode, clean_tables_path, isos_prop_tab_prefix,
#                           metadatadf, confidic, args)
#
#     print("end")
# # #END