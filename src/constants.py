#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Johanna Galvis, Florian Specque, Macha Nikolski
"""
from typing import Literal, get_args


def assert_literal(value:str,lit_type):
    assert value in get_args(lit_type)

data_files_keys = ['abundances_file_name', 'meanE_or_fracContrib_file_name',
                   'isotopologue_prop_file_name', 'isotopologue_abs_file_name']

data_files_keys_type = Literal['abundances_file_name', 'meanE_or_fracContrib_file_name',
                                'isotopologue_prop_file_name', 'isotopologue_abs_file_name']


availtest_methods = ['MW', 'KW', 'ranksum', 'Wcox', 'Tt', 'BrMu',
                     'prm-scipy', 'disfit', 'none']

availtest_methods_type = Literal['MW', 'KW', 'ranksum', 'Wcox', 'Tt', 'BrMu',
                                'prm-scipy', 'disfit', 'none']

correction_methods = ['bonferroni', 'sidak', 'holm-sidak', 'holm',
                      'simes-hochberg', 'hommel', 'fdr_bh', 'fdr_by',
                      'fdr_tsbh', 'fdr_tsbky']

correction_methods_type = Literal['bonferroni', 'sidak', 'holm-sidak', 'holm',
                        'simes-hochberg', 'hommel', 'fdr_bh', 'fdr_by',
                        'fdr_tsbh', 'fdr_tsbky']

comparison_modes = ["paiwise", "multigroup"]

comparison_modes_types = Literal["paiwise", "multigroup"]

overlap_methods = ["symmetric", "asymmetric"]

overlap_methods_types = Literal["symmetric", "asymmetric"]