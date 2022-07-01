# Generic codes for handling Crossmatching TICs with a Vizier Catalog

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


def _xmatch_vizier_of_tics(df_tics_ra_dec, catalog, max_distance):
    """Return the Vizier Catalog records of the given list of TICs by coordinate match using Vizier Crossmatch."""

    src_tab = Table.from_pandas(df_tics_ra_dec[["ID", "ra", "dec"]])
    src_tab.rename_column("ID", "TIC_ID")
    src_tab.rename_column("ra", "TIC_RA")
    src_tab.rename_column("dec", "TIC_DEC")

    catalog = f"vizier:{catalog}"
    return XMatch.query(cat1=src_tab, cat2=catalog, max_distance=max_distance, colRA1="TIC_RA", colDec1="TIC_DEC")


def xmatch_and_save_vizier_meta_of_all_by_tics(
    catalog: str, out_path: str, max_distance=120 * u.arcsec, dry_run=False, dry_run_size=1000
):

    df_tics = tic_meta.load_tic_meta_table_from_file()
    if dry_run and dry_run_size is not None:
        df_tics = df_tics[:dry_run_size]

    res = _xmatch_vizier_of_tics(df_tics, catalog=catalog, max_distance=max_distance)

    # sort the result by TIC_ID + angDist to make subsequent processing easier
    res.sort(["TIC_ID", "angDist"])

    if not dry_run:
        to_csv(res, out_path, mode="w")

    return res


def do_calc_matches_with_tics(df: pd.DataFrame, df_tics: pd.DataFrame, match_result_columns: dict, match_func: Callable):
    """For each row in `df`, find the best match in TIC meta data `df_tics` using the given `match_func`.

    `df` is typically raw crossmatch by co-ordinate data of TICs against some catalog.
    `match_func`: for each pair, calculates the `Match_Score` and stores it in `match_result_columns`, and optionally any
    additional data

    Return: `df` with `match_result_columns` added to it.
    """

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
