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
        keep_default_na=True,
    )
    # VSX band (in `n_max` and `n_min` columns) could be empty string
    # we revert panda's behavior and convert them from N/A back to empty string
    df.loc[pd.isna(df["n_max"]), ["n_max"]] = ""
    df.loc[pd.isna(df["n_min"]), ["n_min"]] = ""
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


def _calc_matches(row_vsx, row_tic, vsx_to_tic_band_map, tic_band_preference_map):
    max_mag_diff = 1.0

    mag_vsx, band_vsx = row_vsx["max"], row_vsx["n_max"]

    if pd.isna(mag_vsx):
        # case no information on magnitude, return a neutral match score 0
        return 0, np.nan, ""

    band_tic_of_vsx = vsx_to_tic_band_map.get(band_vsx)
    if band_tic_of_vsx is None:
        print(f"WARN VSX band {band_vsx} is not mapped. Use V. VSX name: {row_vsx['Name']}")
        band_tic_of_vsx = "V"
    ordered_bands = tic_band_preference_map[band_tic_of_vsx]

    mag_tic, band_tic = None, None
    for band in ordered_bands:
        mag_tic_of_band = row_tic[f"{band}mag"]
        if not pd.isna(mag_tic_of_band):
            mag_tic, band_tic = mag_tic_of_band, band
            break

    mag_diff = np.abs(mag_vsx - mag_tic)
    match_score = 1 if mag_diff <= max_mag_diff else -1

    return match_score, mag_diff, band_tic


def _calc_matches_for_all(df: pd.DataFrame, df_tics: pd.DataFrame):
    df_len = len(df)
    match_result_columns = {
        "Match_Score": np.zeros(df_len, dtype=int),
        "Match_Mag_Band": np.full(df_len, "", dtype="O"),
        "Match_Mag_Diff": np.zeros(df_len, dtype=float),
    }

    vsx_to_tic_band_map = _load_vsx_passband_map_from_file()
    vsx_to_tic_band_map = vsx_to_tic_band_map["TIC_band"]  # the VSX_band is set to be the index by default
    vsx_to_tic_band_map = vsx_to_tic_band_map.to_dict()  # TODO: move to helper

    tic_band_preference_map = _create_tic_passband_preference_table()

    def match_func(row_xmatch, row_tics, i, match_result_columns):
        match_score, mag_diff, band_tic = _calc_matches(row_xmatch, row_tics, vsx_to_tic_band_map, tic_band_preference_map)
        cols = match_result_columns  # to abbreviate it
        cols["Match_Score"][i] = match_score
        cols["Match_Mag_Band"][i] = band_tic
        cols["Match_Mag_Diff"][i] = mag_diff

    df = _do_calc_matches_with_tics(df, df_tics, match_result_columns, match_func)

    # Format the result table
    df.drop(["_RAJ2000", "_DEJ2000"], axis=1, inplace=True)  # redundant columns
    df.drop(["TIC_RA", "TIC_DEC"], axis=1, inplace=True)  # not useful. They are already in tic_meta.csv

    # move angDist to before the matching columns at the end.
    col_match_score = df.pop("angDist")
    idx_match_score = df.columns.get_loc("Match_Score")
    df.insert(idx_match_score, "angDist", col_match_score)

    return df


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


def _load_tic_passband_meta_from_file(csv_path="../data/auxillary/tic_passband_meta.csv", set_tic_band_as_index=True):
    df = pd.read_csv(csv_path)
    if set_tic_band_as_index:
        df.set_index("TIC_band", drop=False, append=False, inplace=True)
    return df


def _create_tic_passband_preference_table():
    df_tic_bands = _load_tic_passband_meta_from_file()

    # For each band, we compute the difference between it and other bands,
    # so as to create an ordered list of preference.
    # E.g., caller has a VSX entry with magnitude in sloan g band
    # `df["g"]` below will have an ordered list of TIC bands that is closest to sloan g

    df_diff = pd.DataFrame()
    for band in df_tic_bands["TIC_band"]:
        band_wavelength = df_tic_bands.loc[band]["Wavelength"]
        df_diff[band] = np.abs(df_tic_bands["Wavelength"] - band_wavelength, dtype=float)

    # an override: for `V`` band, `B` will be chosen over `GAIA` based on wavelength difference
    # (106 vs 122)
    # however, because `GAIA` is very broadband, it is a better approximation
    diff_v_and_b = df_diff["V"].loc["B"]
    # the override to ensure GAIA will be ahead of B
    # I make it a floating point number as an cue that the diff has been overridden.
    df_diff["V"].loc["GAIA"] = diff_v_and_b - 0.1

    df = pd.DataFrame()
    for band in df_tic_bands["TIC_band"]:
        band_pref = df_diff[band].sort_values()
        df[band] = band_pref.index.to_numpy()  # the ordered list of band
        df[f"{band}_diff"] = band_pref.to_numpy()  # the wavelength difference for reference

    return df


if __name__ == "__main__":
    # Get crossmatch result from Vizier / VSX
    xmatch_and_save_vsx_meta_of_all_by_tics()
    # Process the result to find the best matches.
    find_and_save_vsx_best_xmatch_meta(min_score_to_include=0)
