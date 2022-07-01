import warnings

import numpy as np

import pandas as pd

from astropy.utils.exceptions import AstropyWarning

from common import insert, move, to_csv, AbstractTypeMapAccessor
import tic_meta
import vsx_meta
import xmatch_util


def xmatch_and_save_asas_sn_meta_of_all_by_tics(dry_run=False, dry_run_size=1000):
    out_path = "cache/asas_sn_tics_xmatch.csv"
    with warnings.catch_warnings():
        # Ignore astropy table's warning above Gaia IDs.
        # They get converted to string and saved in csv
        # we will properly handle it in the reader
        warnings.filterwarnings(
            "ignore", category=AstropyWarning, message=".*OverflowError converting to IntType in column Gaia.*"
        )
        return xmatch_util.xmatch_and_save_vizier_meta_of_all_by_tics(
            "II/366/catalog",
            out_path=out_path,
            dry_run=dry_run,
            dry_run_size=dry_run_size,
        )


def _load_asas_sn_xmatch_table_from_file(csv_path="cache/asas_sn_tics_xmatch.csv"):
    df = pd.read_csv(
        csv_path,
        dtype={
            # force integer IDs to be nullable (or they'd be converted to float)
            "Gaia": "Int64",
            "APASS": "Int64",
        },
        keep_default_na=True,
    )
    return df


def _calc_matches(row_asas_sn, row_tic, tic_band_preference_map):
    max_mag_diff = 1.0

    # ASAS-SN records always have Vmag in practice.
    # So we use it for magnitude matching
    mag_asas_sn = row_asas_sn["Vmag"]
    band_asas_sn = "V"

    if pd.isna(mag_asas_sn):
        # case no information on magnitude, return a neutral match score 0
        return 0, np.nan, ""

    ordered_bands = tic_band_preference_map[band_asas_sn]

    mag_tic, band_tic = None, None
    for band in ordered_bands:
        mag_tic_of_band = row_tic[f"{band}mag"]
        if not pd.isna(mag_tic_of_band):
            mag_tic, band_tic = mag_tic_of_band, band
            break

    mag_diff = np.abs(mag_asas_sn - mag_tic)
    match_score = 1 if mag_diff <= max_mag_diff else -1

    return match_score, mag_diff, band_tic


def _calc_matches_for_all(df: pd.DataFrame, df_tics: pd.DataFrame):
    df_len = len(df)
    match_result_columns = {
        "Match_Score": np.zeros(df_len, dtype=int),
        "Match_Mag_Band": np.full(df_len, "", dtype="O"),
        "Match_Mag_Diff": np.zeros(df_len, dtype=float),
    }

    tic_band_preference_map = vsx_meta._create_tic_passband_preference_table()

    def match_func(row_xmatch, row_tics, i, match_result_columns):
        match_score, mag_diff, band_tic = _calc_matches(row_xmatch, row_tics, tic_band_preference_map)
        cols = match_result_columns  # to abbreviate it
        cols["Match_Score"][i] = match_score
        cols["Match_Mag_Band"][i] = band_tic
        cols["Match_Mag_Diff"][i] = mag_diff

    df = xmatch_util.do_calc_matches_with_tics(df, df_tics, match_result_columns, match_func)

    # Format the result table
    df.drop(["_RAJ2000", "_DEJ2000"], axis=1, inplace=True)  # redundant columns
    df.drop(["TIC_RA", "TIC_DEC"], axis=1, inplace=True)  # not useful. They are already in tic_meta.csv

    # move angDist to before the matching columns at the end.
    move(df, colname="angDist", before_colname="Match_Score")

    return df


def find_and_save_asas_sn_best_xmatch_meta(dry_run=False, dry_run_size=1000, min_score_to_include=0):
    out_path_accepted = "../data/asas_sn_meta.csv"
    out_path_rejected = "../data/asas_sn_meta_rejected.csv"  # those with low match score

    df_asas_sn = _load_asas_sn_xmatch_table_from_file()
    df_tics = tic_meta.load_tic_meta_table_from_file()

    if dry_run and dry_run_size is not None:
        # the running time is drive by the xmatch table, so we limit it
        df_asas_sn = df_asas_sn[:dry_run_size]

    df = _calc_matches_for_all(df_asas_sn, df_tics)

    df_accepted = df[df["Match_Score"] >= min_score_to_include].reset_index(drop=True)
    df_rejected = df[df["Match_Score"] < min_score_to_include].reset_index(drop=True)

    if not dry_run:
        to_csv(df_accepted, out_path_accepted, mode="w")
        to_csv(df_rejected, out_path_rejected, mode="w")

    return df_accepted, df_rejected


def load_asas_sn_meta_table_from_file(csv_path="../data/asas_sn_meta.csv"):
    # the final ASAS-SN meta table is so similar to the interim crossmatch table
    # that the logic can be reused
    return _load_asas_sn_xmatch_table_from_file(csv_path)


def map_and_save_asas_sn_is_eb_of_all():
    # TODO:
    pass


if __name__ == "__main__":
    # Get crossmatch result from Vizier / ASAS-SN
    xmatch_and_save_asas_sn_meta_of_all_by_tics()
    # Process the result to find the best matches.
    find_and_save_asas_sn_best_xmatch_meta(min_score_to_include=0)
    # For each ASAS-SN record, map `Is_EB`
    map_and_save_asas_sn_is_eb_of_all()
