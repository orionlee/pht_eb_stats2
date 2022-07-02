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


class MatchResult(xmatch_util.AbstractMatchResult):
    def __init__(self, mag, mag_band, mag_diff, pm, pmra_diff_pct, pmdec_diff_pct, dist, dist_diff_pct):
        self.mag = mag
        self.mag_band = mag_band
        self.mag_diff = mag_diff
        self.pm = pm
        self.pmra_diff_pct = pmra_diff_pct
        self.pmdec_diff_pct = pmdec_diff_pct
        self.dist = dist
        self.dist_diff_pct = dist_diff_pct

    def get_flags(self):
        return [self.mag, self.pm, self.dist]

    def get_weights(self):
        # Tweak the weight based on anecdotal observation that
        # the PM / Distance in ASAS-SN / Vizier seem to be more error prone
        # (the live version on https://asas-sn.osu.edu/variables/ is noticeably better)
        # Intent:
        # - if magnitude matches, we pretty much ignore PM / Distance
        # - if magnitude does not mach but the diff is big (> 2),
        #   we make it override PM / Distance match
        # - if magnitude does not mach but the diff is small (< 2),
        #   we let PM and Distance override it if both PM and distance match
        if self.mag:
            return [2, 1, 1]
        elif not self.mag and self.mag_diff > 2:
            return [2, 1, 1]
        else:
            return [1, 1, 1]


def _calc_matches(row_asas_sn, row_tic, tic_band_preference_map):
    max_mag_diff = 1.0
    max_dist_diff_pct = 25

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
    mag_match = True if mag_diff <= max_mag_diff else False

    pm_match, pmra_diff_pct, pmdec_diff_pct = xmatch_util.calc_pm_matches(row_tic, row_asas_sn["pmRA"], row_asas_sn["pmDE"])

    dist_match, dist_diff_pct = xmatch_util.calc_scalar_matches(
        row_tic, "d", row_asas_sn["Dist"], max_diff_pct=max_dist_diff_pct
    )

    return MatchResult(mag_match, band_tic, mag_diff, pm_match, pmra_diff_pct, pmdec_diff_pct, dist_match, dist_diff_pct)


def _calc_matches_for_all(df: pd.DataFrame, df_tics: pd.DataFrame):
    df_len = len(df)
    match_result_columns = {
        "Match_Score": np.zeros(df_len, dtype=int),
        "Match_Mag": np.full(df_len, "-"),
        "Match_PM": np.full(df_len, "-"),
        "Match_Dist": np.full(df_len, "-"),
        "Match_Mag_Band": np.full(df_len, "", dtype="O"),
        "Match_Mag_Diff": np.full(df_len, np.nan, dtype=float),
        "Match_PMRA_DiffPct": np.zeros(df_len, dtype=float),
        "Match_PMDEC_DiffPct": np.full(df_len, np.nan, dtype=float),
        "Match_Dist_DiffPct": np.full(df_len, np.nan, dtype=float),
    }

    tic_band_preference_map = vsx_meta._create_tic_passband_preference_table()

    def match_func(row_xmatch, row_tics, i, match_result_columns):
        match_result = _calc_matches(row_xmatch, row_tics, tic_band_preference_map)
        cols = match_result_columns  # to abbreviate it
        cols["Match_Score"][i] = match_result.score()
        cols["Match_Mag"][i] = match_result.to_flag_str("mag")
        cols["Match_PM"][i] = match_result.to_flag_str("pm")
        cols["Match_Dist"][i] = match_result.to_flag_str("dist")
        cols["Match_Mag_Band"][i] = match_result.mag_band
        cols["Match_Mag_Diff"][i] = match_result.mag_diff
        cols["Match_PMRA_DiffPct"][i] = match_result.pmra_diff_pct
        cols["Match_PMDEC_DiffPct"][i] = match_result.pmdec_diff_pct
        cols["Match_Dist_DiffPct"][i] = match_result.dist_diff_pct

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
    return xmatch_util.find_and_save_best_xmatch_meta(
        df_asas_sn,
        out_path_accepted,
        out_path_rejected,
        _calc_matches_for_all,
        dry_run=dry_run,
        dry_run_size=dry_run_size,
        min_score_to_include=min_score_to_include,
    )


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
