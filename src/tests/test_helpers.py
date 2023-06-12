#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Johanna Galvis, Florian Specque, Macha Nikolski
"""

from unittest import TestCase
import pandas as pd

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

        self.assertEqual(selected_rows, [['Alice', 'Dave'],['Charlie', 'Francoise']])

