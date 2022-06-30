import contextlib
from typing import Callable
import numpy as np

import pandas as pd

with contextlib.redirect_stdout(None):
    # Suppress the "Could not import regions" warning from XMatch.
    # - it is a `print()`` call, so I have to redirect stdout,
    # running the risk of missing some other warning
    from astroquery.xmatch import XMatch

from astropy import units as u
from astropy.table import Table

from common import to_csv
import tic_meta


def _xmatch_vsx_meta_of_tics(df_tics_ra_dec, max_distance=120 * u.arcsec):
    """Return VSX records of the given list of TICs by coordinate match using Vizier Crossmatch."""

    src_tab = Table.from_pandas(df_tics_ra_dec[["ID", "ra", "dec"]])
    src_tab.rename_column("ID", "TIC_ID")
    src_tab.rename_column("ra", "TIC_RA")
    src_tab.rename_column("dec", "TIC_DEC")

    return XMatch.query(cat1=src_tab, cat2="vizier:B/vsx/vsx", max_distance=max_distance, colRA1="TIC_RA", colDec1="TIC_DEC")


def xmatch_and_save_vsx_meta_of_all_by_tics(dry_run=False, dry_run_size=1000):
    out_path = "cache/vsx_tics_xmatch.csv"

    df_tics = tic_meta.load_tic_meta_table_from_file()
    if dry_run and dry_run_size is not None:
        df_tics = df_tics[:dry_run_size]

    res = _xmatch_vsx_meta_of_tics(df_tics)

    # sort the result by TIC_ID + angDist to make subsequent processing easier
    res.sort(["TIC_ID", "angDist"])

    if not dry_run:
        to_csv(res, out_path, mode="w")

    return res


def _load_vsx_match_table_from_file(csv_path="cache/vsx_tics_xmatch.csv"):
    df = pd.read_csv(
        csv_path,
        keep_default_na=False,  # to keep empty string in n_max column (VSX band) as empty string
    )
    return df


def _do_calc_matches_with_tics(df: pd.DataFrame, df_tics: pd.DataFrame, match_result_columns: dict, match_func: Callable):
    """For each row in `df`, find the best match in TIC meta data `df_tics` using the given `match_func`.

    `df` is typically raw crossmatch by co-ordinate data of TICs against some catalog.
    `match_func`: for each pair, calculates the `Match_Score` and stores it in `match_result_columns`, and optionally any
    additional data

    Return: `df` with `match_result_columns` added to it.
    """
    # TODO: move to common.py once it's stable

    # optimization: make lookup by tic_id fast
    df_tics = df_tics.set_index("ID", drop=False)  # we still want df_tics["ID"] work after using it as an index

    df = df.reset_index(drop=True)  # ensure a 0-based index is used in iteration
    for i, row_s in df.iterrows():
        tic_id = row_s["TIC_ID"]
        # Note: a KeyError would be raised if tic_id is unexpected not found in df_tics
        # in practice it shouldn't happen to our dataset
        row_t = df_tics.loc[tic_id]
        match_func(row_s, row_t, i, match_result_columns)

    for colname in match_result_columns.keys():
        df[colname] = match_result_columns[colname]

    if "angDist" in df.columns:
        sort_colnames, ascending = ["TIC_ID", "Match_Score", "angDist"], [True, False, True]
    else:
        sort_colnames, ascending = ["TIC_ID", "Match_Score"], [True, False]
    df.sort_values(sort_colnames, ascending=ascending, inplace=True, ignore_index=True)

    # For each TIC, select the one with the best score (it's sorted above)
    df = df.groupby("TIC_ID").head(1).reset_index(drop=True)

    return df


def _calc_matches_for_all(df: pd.DataFrame, df_tics: pd.DataFrame):
    df_len = len(df)
    match_result_columns = {
        "Match_Score": np.zeros(df_len, dtype=int),
    }

    def match_func(row_xmatch, row_tics, i, match_result_columns):
        tic_id = row_xmatch["TIC_ID"]
        # TODO: do actual calc
        cols = match_result_columns  # to abbreviate it
        cols["Match_Score"][i] = tic_id % 10

    return _do_calc_matches_with_tics(df, df_tics, match_result_columns, match_func)


def find_and_save_vsx_best_xmatch_meta(dry_run=False, dry_run_size=1000, min_score_to_include=0):
    out_path_accepted = "../data/vsx_meta.csv"
    out_path_rejected = "../data/vsx_meta_rejected.csv"  # those with low match score

    df_vsx = _load_vsx_match_table_from_file()
    df_tics = tic_meta.load_tic_meta_table_from_file()

    if dry_run and dry_run_size is not None:
        # the running time is drive by the vsx table, so we limit it
        df_vsx = df_vsx[:dry_run_size]

    df = _calc_matches_for_all(df_vsx, df_tics)

    df_accepted = df[df["Match_Score"] >= min_score_to_include].reset_index(drop=True)
    df_rejected = df[df["Match_Score"] < min_score_to_include].reset_index(drop=True)

    if not dry_run:
        to_csv(df_accepted, out_path_accepted, mode="w")
        to_csv(df_rejected, out_path_rejected, mode="w")

    return df_accepted, df_rejected


def load_vsx_meta_table_from_file(csv_path="../data/vsx_meta.csv"):
    df = pd.read_csv(
        csv_path,
        keep_default_na=False,  # to keep empty string in VSX band as empty string
    )
    return df


def _load_vsx_passband_map_from_file(csv_path="../data/auxillary/vsx_passband_map.csv", set_vsx_band_as_index=True):
    df_passband_vsx_2_tic = pd.read_csv(
        csv_path,
        keep_default_na=False,  # to keep empty string in VSX band as empty string
    )
    # it is a domain table, VSX_band is typically the lookup key
    if set_vsx_band_as_index:
        df_passband_vsx_2_tic.set_index("VSX_band", drop=False, append=False, inplace=True)
    return df_passband_vsx_2_tic


if __name__ == "__main__":
    # Get crossmatch result from Vizier / VSX
    xmatch_and_save_vsx_meta_of_all_by_tics()
    # Process the result to find the best matches.
    find_and_save_vsx_best_xmatch_meta(min_score_to_include=1)
