#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Johanna Galvis, Florian Specque, Macha Nikolski
"""
from typing import Dict

import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os
import argparse
import pandas as pd
from omegaconf import DictConfig

import helpers
from data import Dataset
import logging

logger = logging.getLogger(__name__)

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())

def prep_args():
    show_defaults = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(prog="python -m DIMet.src.prepare",
                                     formatter_class=show_defaults)

    parser.add_argument('config', type=str,
                        help="Configuration file in absolute path")

    return parser

def tabs_2_frames_dict(cfg, dataset : Dataset) -> dict:
    frames_dict = dict()
    df_list = [(cfg.analysis.dataset.abundances_file_name , dataset.abundance_df),
               (cfg.analysis.dataset.meanE_or_fracContrib_file_name, dataset.meanE_or_fracContrib_df),
               (cfg.analysis.dataset.isotopologue_prop_file_name, dataset.isotopologue_prop_df),
               (cfg.analysis.dataset.isotopologue_abs_file_name , dataset.isotopologue_abs_df)]

    for file_name, df in df_list:
        if file_name == None or df is None : continue
        tmp = df.copy().set_index(df.columns[0]) # strange that here dfs were loaded differently
        badcols = [i for i in list(tmp.columns) if i.startswith("Unnamed")]
        tmp = tmp.loc[:, ~tmp.columns.isin(badcols)]
        tmp.columns = tmp.columns.str.replace(" ", "_")
        tmp.index = tmp.index.str.replace(" ", "_")
        tmp = tmp.replace(" ", "_", regex=False)
        tmp = tmp.dropna(axis=0, how="all")
        frames_dict[file_name] = tmp
    return frames_dict

def set_samples_names(frames_dic, metadata):
    compartments = metadata['short_comp'].unique().tolist()

    for tab in frames_dic.keys():
        for co in compartments:
            metada_co = metadata.loc[metadata['short_comp'] == co, :]
            df = frames_dic[tab][co]
            df = df.T
            df.reset_index(inplace=True)
            df.rename(columns={df.columns[0]: "original_name"},
                      inplace=True)
            careful_samples_order = pd.merge(df.iloc[:, 0],
                                             metada_co[['name_to_plot',
                                                        'original_name']],
                                             how="left", on="original_name")
            df = df.assign(name_to_plot=careful_samples_order['name_to_plot'])
            df = df.set_index('name_to_plot')
            df = df.drop(columns=['original_name'])
            frames_dic[tab][co] = df.T

    return frames_dic


def drop_all_nan_metabolites_on_comp_frames(frames_dic, metadata):
    """ metabolites must be in rows """
    compartments = metadata['short_comp'].unique().tolist()
    for tab in frames_dic.keys():
        for co in compartments:
            tmp = frames_dic[tab][co]
            tmp = tmp.dropna(how="all", axis=0)
            frames_dic[tab][co] = tmp
    return frames_dic


def split_datafiles_by_compartment(cfg : DictConfig, dataset : Dataset, out_data_path : str) -> Dict:
    helpers.verify_metadata_sample_not_duplicated(dataset.metadata_df)

    frames_dict = tabs_2_frames_dict(cfg, dataset)

    tabs_isotopologues = [s for s in frames_dict.keys() if
                          "isotopol" in s.lower()]
    assert len(
        tabs_isotopologues) >= 1, "\nError, bad or no isotopologues input"

    for k in frames_dict.keys():
        tmp_co_dic = helpers.df_to_dict_by_compartment(frames_dict[k], dataset.metadata_df)  # split by compartment
        frames_dict[k] = tmp_co_dic

    frames_dict = drop_all_nan_metabolites_on_comp_frames(frames_dict, dataset.metadata_df)
    frames_dict = set_samples_names(frames_dict, dataset.metadata_df)

    return frames_dict

def save_datafiles_split_by_compartment(cfg: DictConfig, dataset: Dataset, out_data_path: str) -> None:
    file_name_to_df_dict = split_datafiles_by_compartment(cfg, dataset, out_data_path)

    suffix_str = cfg.suffix
    for file_name in file_name_to_df_dict.keys():
        for compartment in file_name_to_df_dict[file_name].keys():
            tmp = file_name_to_df_dict[file_name][compartment]
            tmp.index.name = "metabolite_or_isotopologue"
            tmp = tmp.reset_index() # again index manipulation
            tmp = tmp.drop_duplicates()
            output_file_name = f"{file_name}--{compartment}--{suffix_str}.tsv"
            tmp.to_csv(os.path.join(out_data_path, output_file_name),
                sep='\t', header=True, index=False)
            logger.info(f"Saved the {compartment} compartment version of {file_name} in {out_data_path}")

def split_datasets(cfg: DictConfig, dataset: Dataset) -> None:
    out_data_path = os.path.join(os.getcwd(), cfg.data_path, "processed")
    os.makedirs(out_data_path, exist_ok=True)
    save_datafiles_split_by_compartment(cfg, dataset=dataset, out_data_path=out_data_path)
