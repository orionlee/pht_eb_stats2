import warnings

import numpy as np

import pandas as pd

from astropy import units as u
from astropy.utils.exceptions import AstropyWarning

from common import insert, move, to_csv, AbstractTypeMapAccessor
import xmatch_util


def xmatch_and_save_gaia_dr3_meta_of_all_by_tics(dry_run=False, dry_run_size=1000):
    out_path = "cache/gaia_dr3_tics_xmatch.csv"
    with warnings.catch_warnings():
        # Ignore astropy table's warning above Gaia's ID (Source) and SolID
        # They get converted to string and saved in csv
        # we will properly handle it in the reader
        warnings.filterwarnings(
            "ignore", category=AstropyWarning, message=".*OverflowError converting to IntType in column SolID.*"
        )
        warnings.filterwarnings(
            "ignore", category=AstropyWarning, message=".*OverflowError converting to IntType in column Source.*"
        )
        # OPEN: we might need to increase timeout.
        # Empirically, the run for the ~12K of TICs took 11 minutes
        # it should have timed out (the default is 300 seconds), but it did not.
        # A workaround is to have caller set it with `XMatch.TIMEOUT` directly
        return xmatch_util.xmatch_and_save_vizier_meta_of_all_by_tics(
            "I/355/gaiadr3",
            out_path=out_path,
            # TIC's coordinate are mostly based on Gaia DR2, so they should match Gaia DR3 quite well
            # 15 arcsec seems to have sufficient coverage based on a 1000 sample test
            max_distance=15 * u.arcsec,
            dry_run=dry_run,
            dry_run_size=dry_run_size,
        )


def _load_gaia_dr3_xmatch_table_from_file(csv_path="cache/gaia_dr3_tics_xmatch.csv"):
    df = pd.read_csv(
        csv_path,
        dtype={
            # force large integer IDs to use non-nullable int64
            # use non-nullable because they are required ids in Gaia
            "Source": "int64",
            "SolID": "int64",
        },
        keep_default_na=True,
    )
    return df


class MatchResult(xmatch_util.AbstractMatchResult):
    def __init__(self, mag, mag_band, mag_diff, pm, pmra_diff_pct, pmdec_diff_pct, plx, plx_diff_pct):
        self.mag = mag
        self.mag_band = mag_band
        self.mag_diff = mag_diff
        self.pm = pm
        self.pmra_diff_pct = pmra_diff_pct
        self.pmdec_diff_pct = pmdec_diff_pct
        self.plx = plx
        self.plx_diff_pct = plx_diff_pct

    def get_flags(self):
        return [self.mag, self.pm, self.plx]

    def get_weights(self):
        # We trust the consistency of magnitude better than PM / parallax
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


def _calc_matches(row_gaia, row_tic, tic_band_preference_map):
    max_mag_diff = 1.0
    max_pmra_diff_pct =  25
    max_pmdec_diff_pct = 25
    max_plx_diff_pct = 25

    # GAIA records always have Gmag in practice.
    # So we use it for magnitude matching
    mag_gaia = row_gaia["Gmag"]
    band_gaia = "GAIA"  # the term used in TIC

    if pd.isna(mag_gaia):
        # occasionally there are Gaia entries with no GMag
        # It seems many of them are dim with only RPmag (> 17), with very little information.
        # I could have used RPmag for those cases, but they aren't likely to be actual matches anyway.
        print(f"WARN Gaia row Gaia DR3 {row_gaia['Source']} has no Gmag")

    ordered_bands = tic_band_preference_map[band_gaia]

    mag_tic, band_tic = None, None
    for band in ordered_bands:
        mag_tic_of_band = row_tic[f"{band}mag"]
        if not pd.isna(mag_tic_of_band):
            mag_tic, band_tic = mag_tic_of_band, band
            break

    mag_diff = np.abs(mag_gaia - mag_tic)
    mag_match = True if mag_diff <= max_mag_diff else False

    pm_match, pmra_diff_pct, pmdec_diff_pct = xmatch_util.calc_pm_matches(
        row_tic, row_gaia["pmRA"], row_gaia["pmDE"],
        max_pmra_diff_pct=max_pmra_diff_pct, max_pmdec_diff_pct=max_pmdec_diff_pct
        )

    plx_match, plx_diff_pct = xmatch_util.calc_scalar_matches(
        row_tic, "plx", row_gaia["Plx"], max_diff_pct=max_plx_diff_pct
    )

    return MatchResult(
        mag_match, band_tic, mag_diff,
        pm_match, pmra_diff_pct, pmdec_diff_pct,
        plx_match, plx_diff_pct,
        )


def _calc_matches_for_all(df: pd.DataFrame, df_tics: pd.DataFrame):
    import vsx_meta

    df_len = len(df)
    match_result_columns = {
        "Match_Score": np.zeros(df_len, dtype=int),
        "Match_Mag": np.full(df_len, "-"),
        "Match_PM": np.full(df_len, "-"),
        "Match_Plx": np.full(df_len, "-"),
        "Match_Mag_Band": np.full(df_len, "", dtype="O"),
        "Match_Mag_Diff": np.full(df_len, np.nan, dtype=float),
        "Match_PMRA_DiffPct": np.full(df_len, np.nan, dtype=float),
        "Match_PMDEC_DiffPct": np.full(df_len, np.nan, dtype=float),
        "Match_Plx_DiffPct": np.full(df_len, np.nan, dtype=float),
    }

    # TODO: move _create_tic_passband_preference_table() to tic_meta.py
    tic_band_preference_map = vsx_meta._create_tic_passband_preference_table()

    def match_func(row_xmatch, row_tics, i, match_result_columns):
        match_result = _calc_matches(row_xmatch, row_tics, tic_band_preference_map)
        cols = match_result_columns  # to abbreviate it
        cols["Match_Score"][i] = match_result.score()
        cols["Match_Mag"][i] = match_result.to_flag_str("mag")
        cols["Match_PM"][i] = match_result.to_flag_str("pm")
        cols["Match_Plx"][i] = match_result.to_flag_str("plx")
        cols["Match_Mag_Band"][i] = match_result.mag_band
        cols["Match_Mag_Diff"][i] = match_result.mag_diff
        cols["Match_PMRA_DiffPct"][i] = match_result.pmra_diff_pct
        cols["Match_PMDEC_DiffPct"][i] = match_result.pmdec_diff_pct
        cols["Match_Plx_DiffPct"][i] = match_result.plx_diff_pct

    df = xmatch_util.do_calc_matches_with_tics(df, df_tics, match_result_columns, match_func)

    # Format the result table
    df.drop(["TIC_RA", "TIC_DEC"], axis=1, inplace=True)  # not useful. They are already in tic_meta.csv

    # move angDist to before the matching columns at the end.
    move(df, colname="angDist", before_colname="Match_Score")

    return df


def find_and_save_gaia_dr3_best_xmatch_meta(dry_run=False, dry_run_size=1000, min_score_to_include=0):
    out_path_accepted = "../data/gaia_dr3_meta.csv"
    out_path_rejected = "../data/gaia_dr3_meta_rejected.csv"  # those with low match score
    df_gaia_dr3 = _load_gaia_dr3_xmatch_table_from_file()
    return xmatch_util.find_and_save_best_xmatch_meta(
        df_gaia_dr3,
        out_path_accepted,
        out_path_rejected,
        _calc_matches_for_all,
        dry_run=dry_run,
        dry_run_size=dry_run_size,
        min_score_to_include=min_score_to_include,
    )


def load_gaia_dr3_meta_table_from_file(csv_path="../data/gaia_dr3_meta.csv"):
    # the final ASAS-SN meta table is so similar to the interim crossmatch table
    # that the logic can be reused
    return _load_gaia_dr3_xmatch_table_from_file(csv_path)



if __name__ == "__main__":
    # Get crossmatch result from Vizier / Gaia DR3
    xmatch_and_save_gaia_dr3_meta_of_all_by_tics()
    # Process the result to find the best matches.
    find_and_save_gaia_dr3_best_xmatch_meta(min_score_to_include=0)
