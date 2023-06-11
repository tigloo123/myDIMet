#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Johanna Galvis, Florian Specque, Macha Nikolski
"""

from unittest import TestCase
from pathlib import Path
import helpers

class TestHelpers(TestCase):
    def test_open_config_file(self):
        config_file = Path(__file__, "/data/example_diff/raw/data_config_diff.yaml").resolve().parents[2]

