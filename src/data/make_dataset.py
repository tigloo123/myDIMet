#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Johanna Galvis, Florian Specque, Macha Nikolski
"""

import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os
import argparse
import pandas as pd
import helpers

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

def df_to__dic_bycomp(df: pd.DataFrame, metadata: pd.DataFrame) -> dict:
    # splits df into dictionary of dataframes, each for one compartment:
    out_dic = dict()
    for co in metadata['short_comp'].unique():
        metada_co = metadata.loc[metadata['short_comp'] == co, :]
        df_co = df.loc[:, metada_co['original_name']]
        out_dic[co] = df_co
    return out_dic


def tabs_2_frames_dic(confidic, data_dir) -> dict:
    frames_dic = dict()
    list_config_tabs = [confidic['abundance_file_name'],
                        confidic['meanE_or_fracContrib_file_name'],
                        confidic['isotopologue_prop_file_name'],
                        confidic['isotopologue_abs_file_name']]
    list_config_tabs = [i for i in list_config_tabs if i is not None]

    for fi_name in list_config_tabs:
        try:
            measu_file = f'{data_dir}{fi_name}.csv'
            tmp = pd.read_csv(measu_file, sep='\t', header=0, index_col=0)
        except FileNotFoundError:
            measu_file = f'{data_dir}{fi_name}.tsv'
            tmp = pd.read_csv(measu_file, sep='\t', header=0, index_col=0)
        except FileNotFoundError:
            measu_file = f'{data_dir}{fi_name}.TSV'
            tmp = pd.read_csv(measu_file, sep='\t', header=0, index_col=0)
        except FileNotFoundError:
            measu_file = f'{data_dir}{fi_name}.CSV'
            tmp = pd.read_csv(measu_file, sep='\t', header=0, index_col=0)
        except Exception as e:
            print("Error in tabs_2_frames_dic : ", e)

        badcols = [i for i in list(tmp.columns) if i.startswith("Unnamed")]
        tmp = tmp.loc[:, ~tmp.columns.isin(badcols)]
        tmp.columns = tmp.columns.str.replace(" ", "_")
        tmp.index = tmp.index.str.replace(" ", "_")
        tmp = tmp.replace(" ", "_", regex=False)
        tmp = tmp.dropna(axis=0, how="all")
        frames_dic[fi_name] = tmp
    return frames_dic


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


def do_prep(args, confidic, meta_path):
    metadata = helpers.open_metadata(meta_path)
    helpers.verify_metadata_sample_not_duplicated(metadata)
    elems_path_meta = meta_path.split("/")[:-1]
    data_dir = "/".join(elems_path_meta) + "/"

    frames_dic = tabs_2_frames_dic(confidic, data_dir)

    tabs_isotopologues = [s for s in frames_dic.keys() if
                          "isotopol" in s.lower()]
    assert len(
        tabs_isotopologues) >= 1, "\nError, bad or no isotopologues input"

    for k in frames_dic.keys():
        tmp_co_dic = df_to__dic_bycomp(frames_dic[k],
                                       metadata)  # split by compartment
        frames_dic[k] = tmp_co_dic

    frames_dic = drop_all_nan_metabolites_on_comp_frames(frames_dic, metadata)
    frames_dic = set_samples_names(frames_dic, metadata)

    return frames_dic


def perform_prep(args, confidic, meta_path, out_path) -> None:
    helpers.detect_and_create_dir(out_path) # useless : hardcoded dir name

    frames_dic = do_prep(args, confidic, meta_path)

    suffix_str = confidic['suffix']
    for k in frames_dic.keys():
        for compartment in frames_dic[k].keys():
            tmp = frames_dic[k][compartment]
            tmp.index.name = "metabolite_or_isotopologue"
            tmp = tmp.reset_index()
            tmp = tmp.drop_duplicates()
            tmp.to_csv(
                f"{out_path}{k}--{compartment}--{suffix_str}.tsv",
                sep='\t', header=True, index=False)


# Runs data processing scripts to turn raw data from (../raw) into
#  cleaned data ready to be analyzed (saved in ../processed).
# make_dataset.py --datadir /Users/macha/Projects/myDIMet/data/example_diff/ --config raw/data_config_example_diff.yml
# make_dataset.py --datadir /Users/hayssam/temp/myDIMet/data/example_diff/ --config raw/data_config_example_diff.yml

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str,
                        help="Data directory")
    parser.add_argument('--config', type=str,
                        help="config file under the data directory")
    args = parser.parse_args()
    confidic = helpers.open_config_file(args.datadir + args.config)
    helpers.auto_check_validity_configuration_file(confidic)
    helpers.verify_good_extensions_measures(confidic)
    confidic = helpers.remove_extensions_names_measures(confidic)
    meta_file = os.path.expanduser(confidic['metadata_file_name'])
    out_path = args.datadir + "/processed/"
    perform_prep(args, confidic, args.datadir + "/raw/" + meta_file, out_path)
