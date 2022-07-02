# Generic codes for handling Crossmatching TICs with a Vizier Catalog

from abc import ABC, abstractmethod
import contextlib
from types import SimpleNamespace
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

from common import to_csv, has_value
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


class AbstractMatchResult(SimpleNamespace, ABC):
    def _flag_to_score(self, val):
        if val is None:
            return 0
        elif val:
            return 1
        else:
            return -1

    @abstractmethod
    def get_flags(self):
        pass

    @abstractmethod
    def get_weights(self):
        pass

    def to_flag_str(self, flag_name):
        # flag_name should be an attribute of MatchResult
        # so we do not give it a default, and raise error instead.
        flag_val = getattr(self, flag_name)
        return _3val_flag_to_str(flag_val)

    def score(self):
        flags = self.get_flags()
        weights = self.get_weights()
        scores = [self._flag_to_score(f) * w for f, w in zip(flags, weights)]
        return np.sum(scores)


def _diff(val1, val2, in_percent=False, label=""):
    if has_value(val1) and has_value(val2):
        diff = np.abs(val1 - val2)
        if not in_percent:
            return diff
        else:
            if val1 == 0:
                print(f"WARN in calculating the difference percentage of {label} , division by zero happens. returning nan")
                return np.nan
            else:
                return 100.0 * diff / np.abs(val1)
    else:
        return None


def calc_pm_matches(tic_meta_row, xmatch_pmra, xmatch_pmdec, max_pmra_diff_pct=25, max_pmdec_diff_pct=25):
    tic_label = f"TIC {tic_meta_row['ID']}"
    pmra_diff_pct = _diff(tic_meta_row["pmRA"], xmatch_pmra, in_percent=True, label=f"{tic_label} pmRA")
    pmdec_diff_pct = _diff(tic_meta_row["pmDEC"], xmatch_pmdec, in_percent=True, label=f"{tic_label} pmDEC")

    pm_match = None
    if pmra_diff_pct is not None and pmdec_diff_pct is not None:
        if pmra_diff_pct < max_pmra_diff_pct and pmdec_diff_pct < max_pmdec_diff_pct:
            pm_match = True
        else:
            pm_match = False
    return pm_match, pmra_diff_pct, pmdec_diff_pct


def calc_scalar_matches(tic_meta_row, tic_scalar_colname, xmatch_scalar_val, max_diff_pct):
    tic_label = f"TIC {tic_meta_row['ID']}"
    val_diff_pct = _diff(
        tic_meta_row[tic_scalar_colname], xmatch_scalar_val, in_percent=True, label=f"{tic_label} {tic_scalar_colname}"
    )
    val_match = None
    if val_diff_pct is not None:
        val_match = val_diff_pct < max_diff_pct
    return val_match, val_diff_pct


def _3val_flag_to_str(val):
    if val is None:
        return "-"
    elif val:
        return "T"
    else:
        return "F"


def find_and_save_best_xmatch_meta(
    df_xmatch, out_path_accepted, out_path_rejected, match_func, dry_run, dry_run_size, min_score_to_include
):
    df_tics = tic_meta.load_tic_meta_table_from_file()

    if dry_run and dry_run_size is not None:
        # the running time is drive by the xmatch table, so we limit it when asked
        df_xmatch = df_xmatch[:dry_run_size]

    df = match_func(df_xmatch, df_tics)

    if min_score_to_include is not None:
        df_accepted = df[df["Match_Score"] >= min_score_to_include].reset_index(drop=True)
        df_rejected = df[df["Match_Score"] < min_score_to_include].reset_index(drop=True)
    else:
        df_accepted = df
        df_rejected = df[np.full(len(df), False)]

    if not dry_run:
        to_csv(df_accepted, out_path_accepted, mode="w")
        to_csv(df_rejected, out_path_rejected, mode="w")

    return df_accepted, df_rejected
