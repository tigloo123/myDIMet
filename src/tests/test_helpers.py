#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Johanna Galvis, Florian Specque, Macha Nikolski
"""

from unittest import TestCase

import pandas as pd
import numpy as np
import helpers


class TestHelpers(TestCase):
    def test_df_to_dict_bycomp(self):
        metadata_dict = {"name_to_plot": ["Ctrl_cell_T0-1", "Ctrl_cell_T0-2"],
                         "condition": ["Control", "Control"],
                         "timepoint": ["T0", "T1", ],
                         "timenum": [0, 1],
                         "short_comp": ["cell", "medium"],
                         "original_name": ["MCF001089_TD01", "MCF001089_TD02"]}
        abundances_dict = {"metabolite_or_isotopologue": ["Fructose_1, 6 - bisphosphate", "Fumaric_acid"],
                           "MCF001089_TD01": [81.467, 1765.862],
                           "MCF001089_TD02": [31.663, 2350.101],
                           "MCF001089_TD07": [488.622, 565.610]
                           }

        metadata_df = pd.DataFrame(metadata_dict)
        abundances_df = pd.DataFrame(abundances_dict)
        d = helpers.df_to_dict_by_compartment(df=abundances_df, metadata=metadata_df)
        self.assertEqual(list(d.keys()), ['cell', 'medium'])
        self.assertEqual(d['cell'].shape, (2, 1))
        self.assertAlmostEqual(d['cell'].at[0, "MCF001089_TD01"], 81.47, 2)

    def test_select_rows_by_fixed_values(self):
        data = {'Name': ['Alice', 'Bob', 'Charlie', 'Dave', 'Francoise'],
                'Age': [25, 30, 35, 40, 67],
                'City': ['London', 'New York', 'Paris', 'London', 'Paris'],
                'Country': ['UK', 'USA', 'France', 'UK', 'France']}
        df = pd.DataFrame(data)

        # Specify the columns and values as lists
        columns_to_match = ['City', 'Country']
        values_to_match = [['London', 'UK'], ['Paris', 'France']]

        # Select rows based on fixed values
        selected_rows = helpers.select_rows_by_fixed_values(df, columns_to_match, values_to_match)

        self.assertEqual(selected_rows, [['Alice', 'Dave'], ['Charlie', 'Francoise']])

    def test_split_rows_by_threshold(self):
        data = {'Name': ['Alice', 'Bob', 'Charlie', 'Dave', 'Francoise'],
                'Age': [25, 30, 35, 40, 67],
                'City': ['London', 'New York', 'Paris', 'London', 'Paris'],
                'Country': ['UK', 'USA', 'France', 'UK', 'France']}
        df = pd.DataFrame(data)
        df1, df2 = helpers.split_rows_by_threshold(df, "Age", 35)
        self.assertEqual(list(df1["Name"].values),
                         ['Charlie', 'Dave', 'Francoise'])  # assert might break due to ordering
        self.assertEqual(list(df2["Name"].values), ['Alice', 'Bob'])

    def test_concatenate_dataframes(self):
        df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
        df2 = pd.DataFrame({'A': [10, 11, 12], 'B': [13, 14, 15]})
        df3 = pd.DataFrame({'B': [16, 17, 18], 'C': [19, 20, 21]})

        result = helpers.concatenate_dataframes(df1, df2, df3)
        result = result.fillna(-1)

        self.assertTrue(all(result["C"] == [7.0, 8.0, 9.0, -1.0, -1.0, -1.0, 19.0, 20.0, 21.0]))
        self.assertTrue(all(result["A"] == [1.0, 2.0, 3.0, 10.0, 11.0, 12.0, -1.0, -1., 0 - 1.0]))

    def test_row_wise_nanstd_reduction(self):
        data = {
            'A': [1, 0, 3, 4],
            'B': [5, 0, 6, 0],
            'C': [7, 0, 8, 9],
            'D': [10, 0, 11, 12],
        }
        df = pd.DataFrame(data)

        # Apply row-wise nanstd reduction
        result = helpers.row_wise_nanstd_reduction(df)
        self.assertTrue(np.allclose(np.array(result["B"]), np.array([1.529438, 0.0, 2.057983, 0.0])))
        self.assertTrue(np.allclose(np.array(result.iloc[1]), np.array([0.0, 0.0, 0.0, 0.0])))

    def test_compute_gmean_nonan(self):
        arr1 = np.array([1, 2, np.nan, 4, 5])
        arr2 = np.array([0, 0, 0, 0])
        gmean1 = helpers.compute_gmean_nonan(arr1)
        gmean2 = helpers.compute_gmean_nonan(arr2)
        print(gmean1, gmean2)
        self.assertAlmostEqual(gmean1, 2.514, 2)
        self.assertAlmostEqual(gmean2, np.finfo(float).eps)