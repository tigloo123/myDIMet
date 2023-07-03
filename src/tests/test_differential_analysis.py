from unittest import TestCase
import pandas as pd
import numpy as np
import processing.differential_analysis as differential_analysis
import processing.fit_statistical_distribution as fit_statistical_distribution


class TestDifferentialAnalysis(TestCase):
    def test_time_course_auto_list_comparisons(self):
        metadata = pd.DataFrame({
            'condition': ['cond1', 'cond1', 'cond1', 'cond1',
                          'cond2', 'cond2', 'cond2', 'cond2'],
            'timenum': [1, 2.7, 3, 1, 2.7, 3, 4, 4],
            'timepoint': ['1h', '2.7h', '3h', '1h', '2.7h', '3h', '4h', '4h']
             })
        result = differential_analysis.time_course_auto_list_comparisons(
            metadata
        )
        self.assertListEqual(result[0], [['cond2', '4h'], ['cond2', '3h']])
        self.assertListEqual(result[1], [['cond1', '3h'], ['cond1', '2.7h']])
        self.assertListEqual(result[2], [['cond2', '3h'], ['cond2', '2.7h']])
        self.assertListEqual(result[3], [['cond1', '2.7h'], ['cond1', '1h']])








