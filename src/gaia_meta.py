import warnings

import numpy as np

import pandas as pd

from astropy import units as u
from astropy.utils.exceptions import AstropyWarning
from astroquery.vizier import Vizier

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
        #
        # Issue: there are some rare cases that multiple TICs matched to the same Gaia ID
        # in the dataset (Sectors 1 - 39), 12507 TICs got mapped to 12485 Gaia IDs
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


def load_gaia_dr3_meta_table_from_file(csv_path="../data/gaia_dr3_meta.csv", add_variable_meta=False):
    # the final ASAS-SN meta table is so similar to the interim crossmatch table
    # that the logic can be reused
    df = _load_gaia_dr3_xmatch_table_from_file(csv_path)
    if (add_variable_meta):
        df_var = load_gaia_dr3_var_meta_table_from_file()
        df = _join_gaia_meta_with_var_meta(df, df_var)
    return df


def _join_gaia_meta_with_var_meta(df_main, df_var):
    # left outer join the 2 tables by Source column
    # column-merge the tables by Gaia ID (Source)
    # The relationship is many_to_one
    # because multiple TICs mapped to single Gaia ID

    #  retain useful columns only, and ignore columns like RA, DEC, etc.
    df_var = df_var[["Source", "Classifier", "Class", "ClassSc"]]
    df = pd.merge(df_main, df_var, how="left", on="Source", validate="many_to_one")

    return df


#
# Gaia DR# Variable crossmatch
#

def _get_gaia_dr3_var_meta_of_gaia_ids(gaia_ids, **kwargs):
    # Vizier requires a list of TIC in string (the leading zero is not needed, however)
    gaia_ids = [str(t) for t in gaia_ids]

    GAIA_DR3_VAR_RESULT_CATALOG = "I/358/vclassre" # only the classification result, but not any specifics
    columns = ["*", "ClassSc", ]  # include score of the best class
    vizier = Vizier(catalog=GAIA_DR3_VAR_RESULT_CATALOG, columns=columns)
    vizier.ROW_LIMIT = -1  # it is not necessary in practice, but to be on the safe side.
    vizier.TIMEOUT = 60 * 30  # elapsed time for 1000 ids is about 11 min.
    result_list = vizier.query_constraints(Source=gaia_ids, **kwargs)

    if len(result_list) < 1:
        return None
    result = result_list[0]  # there is only 1 table in the catalog
    return result


def _save_gaia_dr3_var_meta(meta_table, csv_mode="w"):
    out_path = "../data/gaia_dr3_var_meta.csv"
    return to_csv(meta_table, out_path, mode=csv_mode)


def _get_and_save_gaia_dr3_var_meta_of_all(dry_run=False, dry_run_size=1000, chunk_size=1000):

    # get the Gaia DR3 Ids, use only those that are known to be variable
    df = load_gaia_dr3_meta_table_from_file()[["Source", "VarFlag"]]
    df = df[df["VarFlag"] == "VARIABLE"]
    ids = df["Source"].to_numpy()

    # Note: the approach is not really efficient
    # - use query_constraints() on Gaia DR3 variable to bulk fetch is quite slow
    # - 100 IDs took about 1.5 minute
    # - 1000 IDs took about 12 minutes
    # If we need to do it repeatedly, XMatch might be a better choice.

    if dry_run and dry_run_size is not None:
        ids = ids[:dry_run_size]

    num_chunks = int(np.ceil(len(ids) / chunk_size))

    num_fetched = 0
    for idx in range(0, num_chunks):
        ids_chunk = ids[idx * chunk_size:(idx + 1) * chunk_size]
        print(f"DEBUG  chunk {idx} ; num. ids to send to Vizier: {len(ids_chunk)}")

        res = _get_gaia_dr3_var_meta_of_gaia_ids(ids_chunk)
        num_fetched += len(res)

        if not dry_run:
            csv_mode = "w" if idx == 0 else "a"
            _save_gaia_dr3_var_meta(res, csv_mode=csv_mode)

    # ugly workaround:
    # the input Gaia IDs are not unique (upstream issue in Gaia Crossmatch).
    # - we dedupe here after getting the entire data set
    # - I could have deduped the input IDs, but it'd mess up the vizier result cache locally
    # - ultimately, the source of duplicates in TIC - Gaia crossmatch should be fixed instead.
    if not dry_run:
        df = load_gaia_dr3_var_meta_table_from_file()
        df.drop_duplicates(subset="Source", inplace=True, ignore_index=True)
        _save_gaia_dr3_var_meta(df, csv_mode="w")

    return num_fetched


def load_gaia_dr3_var_meta_table_from_file(csv_path="../data/gaia_dr3_var_meta.csv"):
    # its structure and quirks are the same as main Gaia DR3 (mainly on SolID and Source)
    return _load_gaia_dr3_xmatch_table_from_file(csv_path)


if __name__ == "__main__":
    # Get crossmatch result from Vizier / Gaia DR3
    xmatch_and_save_gaia_dr3_meta_of_all_by_tics()
    # Process the result to find the best matches.
    find_and_save_gaia_dr3_best_xmatch_meta(min_score_to_include=0)

    # Fetch Gaia DR3 Variable classification when applicable
    _get_and_save_gaia_dr3_var_meta_of_all()
