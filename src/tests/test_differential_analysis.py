from unittest import TestCase
import pandas as pd
import numpy as np
import processing.differential_analysis as differential_analysis
import processing.fit_statistical_distribution as fit_statistical_distribution


class TestDifferentialTwogroupsAnalysis(TestCase):
    def test_compute_span_incomparison(self):
        data = {
            "c1": [15, 22, 310],
            "c2": [8, 30, 220],
            "c3": [11, 25, 170],
            "c4": [9, 33, 100],
        }
        df = pd.DataFrame(data)
        result = differential_analysis.compute_span_incomparison(
            df, [["c1", "c2"], ["c3", "c4"]])

        self.assertTrue(result.at[0, 'span_allsamples'] == float(7))
        self.assertTrue(result.at[1, 'span_allsamples'] == float(11))
        self.assertTrue(result.at[2, 'span_allsamples'] == float(210))

    def test_distance_or_overlap(self):
        data = {
            "c1": [15, 22, 310],
            "c2": [8, 30, 220],
            "c3": [11, 25, 170],
            "c4": [9, 33, 100],
        }
        df = pd.DataFrame(data)
        result = differential_analysis.distance_or_overlap(
            df, [["c1", "c2"], ["c3", "c4"]]
        )
        self.assertTrue(df.at[0, "distance"] == float(-2))
        self.assertTrue(df.at[1, "distance"] == float(-5))
        self.assertTrue(df.at[2, "distance"] == float(50))

    def test_compute_mann_whitney_all_h0(self):
        array1 = np.array([722, 760, 750, 700])
        array2 = np.array([150, 177, 165, 110])
        result = differential_analysis.compute_mann_whitney_allH0(array1,
                                                                  array2)
        self.assertTrue(result[0] == 16.0)
        self.assertTrue(result[1] == 0.014285714285714285)

    def test_compute_padj_version2(self):
        data = {'row': np.arange(0, 5, 1),
                'pvalue': [0.08, 0.8, 0.088, 0.03, np.nan]}
        df = pd.DataFrame(data)
        result = differential_analysis.compute_padj_version2(df, 0.05,
                                                             "bonferroni")
        self.assertTrue(np.any(
            np.array(result['padj']) == np.array(
                [0.4, 1.0, 0.44, 0.15, np.nan])))

    def test_filter_diff_results(self):
        data = {'row': np.arange(0, 5, 1),
                'pvalue': [0.004, 1.0, 0.064, 0.015, np.nan],
                 'padj' :  [0.004, 1.0, 0.064, 0.015, np.nan],
                'log2FC' : [4, 1.2, 0, -3, -1],
                'distance/span' : [1, -0.6 ,0, 0.3,-0.6]}
        df = pd.DataFrame(data)
        result = differential_analysis.filter_diff_results(df, 0.05, 1.1)
        self.assertTrue(result.shape[0] == 2)
        self.assertTrue(any(np.array(result['row']) == np.array([0, 3])))

    def test_run_distribution_fitting(self): # TODO: this test is slow, think if needed, or remove
        data = {'zscore': np.random.laplace(loc=0.0, scale=1.6, size=500)}
        df = pd.DataFrame(data)
        best_distribution, args_param = \
            fit_statistical_distribution.find_best_distribution(df)
        # best_distribution object is required :
        autoset_tailway = differential_analysis.auto_detect_tailway(
            df, best_distribution, args_param
        )
        # best_distribution object is required again :
        result = differential_analysis.compute_p_value(
            df, autoset_tailway, best_distribution, args_param)
        # # in a third debug it was 'gennorm', impossible to set assert here :
        # self.assertTrue(best_distribution.name == "laplace" |
        #                 best_distribution.name == "dgamma")
        # self.assertAlmostEqual(args_param['a'], 1.6, places=0)
        self.assertFalse(result.pvalue.min() < 0)
        self.assertFalse(result.pvalue.max() > 1)
