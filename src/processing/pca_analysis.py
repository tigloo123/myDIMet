#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import List, Tuple, Union
import logging
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from omegaconf import DictConfig

from constants import (
    availtest_methods_type,
    assert_literal,
    data_files_keys_type,
)
import helpers
from data import Dataset

logger = logging.getLogger(__name__)


def clean_reduce_datadf_4pca(df: pd.DataFrame,
                             metadata_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares quantitative dataframe for pca
    """
    df = df[metadata_df['name_to_plot']]
    df = df.loc[~(df == 0).all(axis=1)]  # drop 'zero all' rows
    df = df.fillna(df[df > 0].min().min())
    df = df.replace(0, df[df > 0].min().min())
    df_red = helpers.row_wise_nanstd_reduction(df)  # reduce rows
    if np.isinf(np.array(df_red)).any():  # avoid Inf error
        return df
    else:
        return df_red


def compute_pca(mymat, metadata_df) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Uses sklearn PCA.
    Returns the two computed metrics:
     - Computed dimensions coeficients (pc_df)
     - Explained Variance in percentages (var_explained_df)
    """
    dims = min(metadata_df.shape[0],
               mymat.shape[0])  # min(nb_samples, nb_features)

    X = np.transpose(np.array(mymat))
    pca = PCA(n_components=dims)
    pc = pca.fit_transform(X)
    pc_df = pd.DataFrame(data=pc,
                         columns=['PC' + str(i) for i in range(1, dims + 1)])
    pc_df = pc_df.assign(name_to_plot=mymat.columns)
    pc_df = pd.merge(pc_df, metadata_df, on='name_to_plot')

    var_explained_df = pd.DataFrame({
        'Explained Variance %': pca.explained_variance_ratio_ * 100,
        'PC': ['PC' + str(i) for i in range(1, dims + 1)]})

    return pc_df, var_explained_df


def pca_on_split_dataset(pca_tables_dict, compartment_df, metadata_co_df,
                         chosen_column: str, description: List[str]):
    """
    Using compartment specific dataframes,
    splits by selected criteria (condition or timepoint)
    and computes PCA on each subset.
    The results are added to the dictionary of results.
    """
    assert len(metadata_co_df['short_comp'].unique()) == 1
    my_criteria: list = metadata_co_df[chosen_column].unique().tolist()
    for criterion in my_criteria:
        metadata_tmp = metadata_co_df.loc[
            metadata_co_df[chosen_column] == criterion, :]
        data_tmp = compartment_df[metadata_tmp['name_to_plot']]

        mat = clean_reduce_datadf_4pca(data_tmp, metadata_tmp)
        pc_df, var_explained_df = compute_pca(mat, metadata_tmp)

        tmp_unique = tuple([description[0], criterion, description[1]])
        pca_tables_dict[tmp_unique] = {
              'pc': pc_df,
              'var': var_explained_df
        }

    return pca_tables_dict


def send_to_tables(pca_results_compartment_dict, out_table_dir) -> None:
    """ Save each result to .csv files """
    for tup in pca_results_compartment_dict.keys():
        out_table = "--".join(list(tup))
        for df in pca_results_compartment_dict[tup].keys():
            pca_results_compartment_dict[tup][df].to_csv(
                os.path.join(out_table_dir, f"pca_{out_table}.csv"),
                sep='\t', index=False)
    logger.info(f"Saved pca tables in {out_table_dir}")


def pca_analysis(file_name: data_files_keys_type,
                 dataset: Dataset, cfg: DictConfig,
                 out_table_dir: str, mode: str) -> Union[None, dict]:
    """
    Generates all PCA results, both global (default) and with splited data.
    If mode is:
     - save_tables, the PCA tables are saved to .csv;
     - return_results_dict, returns the results object (dict)
    """
    assert_literal(file_name, data_files_keys_type, "file name")

    metadata_df = dataset.metadata_df

    for compartment, compartmentalized_df in dataset.compartmentalized_dfs[
        file_name].items():

        df = compartmentalized_df
        metadata_co_df = metadata_df[metadata_df['short_comp'] == compartment]

        mat = clean_reduce_datadf_4pca(df, metadata_co_df)
        pc_df, var_explained_df = compute_pca(mat, metadata_co_df)

        pca_results_dict = {tuple([file_name, compartment]): {
            'pc': pc_df,
            'var': var_explained_df
        }}

        if cfg.analysis.method.pca_split_further is not None:
            for column in cfg.analysis.method.pca_split_further:
                pca_results_dict = pca_on_split_dataset(
                    pca_results_dict, df, metadata_co_df,
                    column,  description=[file_name, compartment])

        if mode == "save_tables":
            send_to_tables(pca_results_dict, out_table_dir)

        elif mode == "return_results_dict":
            return pca_results_dict

