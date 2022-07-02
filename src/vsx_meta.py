import contextlib
import re
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

from common import insert, move, to_csv, AbstractTypeMapAccessor
import tic_meta
import xmatch_util


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


def _load_vsx_xmatch_table_from_file(csv_path="cache/vsx_tics_xmatch.csv"):
    df = pd.read_csv(
        csv_path,
        keep_default_na=True,
        # For VSX band (in `n_max` and `n_min` columns), empty string is a proper value
        # (visual magnitude). Use the converters to force panda treat
        # them as empty string rather than NA
        # (for all other columns, empty strings do mean NA)
        # cf. https://stackoverflow.com/a/70172587
        converters={"n_min": str, "n_max": str},
    )
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

    vsx_to_tic_band_map = _load_vsx_passband_map_from_file(as_dict=True)

    tic_band_preference_map = _create_tic_passband_preference_table()

    def match_func(row_xmatch, row_tics, i, match_result_columns):
        match_score, mag_diff, band_tic = _calc_matches(row_xmatch, row_tics, vsx_to_tic_band_map, tic_band_preference_map)
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


def find_and_save_vsx_best_xmatch_meta(dry_run=False, dry_run_size=1000, min_score_to_include=0):
    out_path_accepted = "../data/vsx_meta.csv"
    out_path_rejected = "../data/vsx_meta_rejected.csv"  # those with low match score
    df_vsx = _load_vsx_xmatch_table_from_file()
    return xmatch_util.find_and_save_best_xmatch_meta(
        df_vsx,
        out_path_accepted,
        out_path_rejected,
        _calc_matches_for_all,
        dry_run=dry_run,
        dry_run_size=dry_run_size,
        min_score_to_include=min_score_to_include,
    )


def load_vsx_meta_table_from_file(csv_path="../data/vsx_meta.csv"):
    # the final VSX meta table is so similar to the interim crossmatch table
    # that the logic can be reused
    return _load_vsx_xmatch_table_from_file(csv_path)


def _load_vsx_passband_map_from_file(
    csv_path="../data/auxillary/vsx_passband_map.csv", set_vsx_band_as_index=True, as_dict=False
):
    df_passband_vsx_2_tic = pd.read_csv(
        csv_path,
        keep_default_na=False,  # to keep empty string in VSX band as empty string
    )
    # it is a domain table, VSX_band is typically the lookup key
    if set_vsx_band_as_index or as_dict:
        df_passband_vsx_2_tic.set_index("VSX_band", drop=False, append=False, inplace=True)

    if as_dict:
        # the dict's key is VSX_band, set in the set_index() call above
        return df_passband_vsx_2_tic["TIC_band"].to_dict()
    else:
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


class VSXTypeMapAccessor(AbstractTypeMapAccessor):
    # `../data/auxillary/vsx_vartype_map.csv` is based on
    # https://www.aavso.org/vsx/index.php?view=about.vartypes
    # - `vsx_vartype_scrapper.js` is used to obtain the preliminary list
    # - further manual editing is done to add the Variable group and IS_EB column.
    def __init__(self, csv_path="../data/auxillary/vsx_vartype_map.csv"):
        super().__init__(csv_path, "VSX_Type")

    def _map_1_type(self, type):
        # VSX-specific processing, a type may end up :, indicating the classification is uncertain
        type = re.sub(":$", "", type)
        return super()._map_1_type(type)

    def _split_types_str(self, types_str):
        return re.split(r"[|+/]", types_str)


def map_and_save_vsx_is_eb_of_all(warn_types_not_mapped=False):
    out_path = "../data/vsx_is_eb.csv"
    typemap = VSXTypeMapAccessor()
    df = load_vsx_meta_table_from_file()

    map_res = [typemap.map(types).label for types in df["Type"]]

    # return a useful subset of columns, in addition to the EB map result
    # TIC_ID,OID,n_OID,Name,V,Type,l_max,max,u_max,n_max,f_min,l_min,min,u_min,n_min,l_Period,Period,u_Period,RAJ2000,DEJ2000,angDist,Match_Score,Match_Mag_Band,Match_Mag_Diff
    res = df[["OID", "n_OID", "V", "Name", "TIC_ID", "Type", "Period", "angDist", "Match_Score"]]
    insert(res, before_colname="Type", colname="Is_EB", value=map_res)

    to_csv(res, out_path, mode="w")
    not_mapped_types_seen = list(typemap.not_mapped_types_seen)
    if warn_types_not_mapped and len(not_mapped_types_seen) > 1:
        print(f"WARN: there are {len(not_mapped_types_seen)} number of TYPE value not mapped.")
        print(not_mapped_types_seen)
    return res, not_mapped_types_seen


def load_vsx_is_eb_table_from_file(csv_path="../data/vsx_is_eb.csv"):
    df = pd.read_csv(csv_path)
    return df


if __name__ == "__main__":
    # Get crossmatch result from Vizier / VSX
    xmatch_and_save_vsx_meta_of_all_by_tics()
    # Process the result to find the best matches.
    find_and_save_vsx_best_xmatch_meta(min_score_to_include=0)
    # For each VSX record, map `Is_EB`
    map_and_save_vsx_is_eb_of_all(warn_types_not_mapped=True)
