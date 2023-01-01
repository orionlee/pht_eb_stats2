import contextlib
import json

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astroquery.simbad import Simbad

with contextlib.redirect_stdout(None):
    # Suppress the "Could not import regions" warning from XMatch.
    # - it is a `print()`` call, so I have to redirect stdout,
    # running the risk of missing some other warning
    from astroquery.xmatch import XMatch

import numpy as np
import pandas as pd

from ratelimit import limits, sleep_and_retry

from common import bulk_process, fetch_json, has_value, insert, to_csv, load_tic_ids_from_file, AbstractTypeMapAccessor
import tic_meta
import xmatch_util


# throttle HTTP calls to MAST
# somewhat large result set (~1000 rows), so I set a conservative throttle to be extra safe
NUM_CALLS = 1
PERIOD_IN_SECONDS = 20


def _get_simbad(add_typed_id=False):
    simbad = Simbad()

    simbad.remove_votable_fields("coordinates")
    if add_typed_id:
        simbad.add_votable_fields("typed_id")
    fields_to_add = [
        "otypes",
        "v*",  # GCVS params, if any
        # fields for crossmatch purposes
        "ra(d;ICRS;J2000;2000)",
        "dec(d;ICRS;J2000;2000)",
        "plx",
        "pmra",
        "pmdec",
        "flux(B)",
        "flux(V)",
        "flux(R)",
        "flux(G)",
        "flux(J)",
        "ids",
    ]
    simbad.add_votable_fields(*fields_to_add)
    return simbad


def _format_result(res):
    # format output result
    res.rename_column("RA_d_ICRS_J2000_2000", "RA")
    res.rename_column("DEC_d_ICRS_J2000_2000", "DEC")
    # column "SCRIPT_NUMBER_ID": retained for now as it could be useful for troubleshooting
    # , as it is referenced by SIMBAD warnings/ errors.


@sleep_and_retry
@limits(calls=NUM_CALLS, period=PERIOD_IN_SECONDS)
def _get_simbad_meta_of_tics(tics):
    simbad = _get_simbad(add_typed_id=True)
    res = simbad.query_objects([f"TIC {id}" for id in tics])
    _format_result(res)
    res.rename_column("TYPED_ID", "TIC_ID")
    res["TIC_ID"] = [int(s.replace("TIC ", "")) for s in res["TIC_ID"]]
    return res


@sleep_and_retry
@limits(calls=NUM_CALLS, period=PERIOD_IN_SECONDS)
def _get_simbad_meta_of_ids(ids):
    simbad = _get_simbad(add_typed_id=True)
    res = simbad.query_objects(ids)
    _format_result(res)
    return res


def _get_simbad_meta_of_coordinates(
    ra,
    dec,
    coord_kwargs=dict(unit=u.deg, frame="icrs", equinox="J2000"),
    radius=2 * u.arcmin,
    max_rows_per_coord=5,
):
    coord = SkyCoord(ra=ra, dec=dec, **coord_kwargs)
    simbad = _get_simbad(add_typed_id=True)
    res = simbad.query_region(coord, radius=radius)
    _format_result(res)
    return res


def xmatch_and_save_all_unmatched_tics():
    """For TICs with no SIMBAD entry by TIC ID, crossmatch by coordinate to get the matching SIMBAD ids"""
    out_path = "cache/simbad_tics_xmatch.csv"

    df_simbad = load_simbad_meta_table_from_file(csv_path="cache/simbad_meta_by_ticid.csv")
    # list of TIC ids not matched with no SIMBAD match
    src_ticids = df_simbad[df_simbad["MAIN_ID"].isnull()][["TIC_ID"]]["TIC_ID"].to_numpy()

    # for the TICs, find their RA/DEC from TIC metadata
    df_tics = tic_meta.load_tic_meta_table_from_file()
    df_tics = df_tics[df_tics["ID"].isin(src_ticids)]
    src_tab = Table.from_pandas(df_tics[["ID", "ra", "dec"]])
    src_tab.rename_column("ID", "TIC_ID")
    src_tab.rename_column("ra", "TIC_RA")
    src_tab.rename_column("dec", "TIC_DEC")

    # we just care about the main_id returned, as we will use teh main_id to query the up-to-date metadata from SIMBAD later
    # I don't know how to tell XMatch to not include the other columns
    res = XMatch.query(cat1=src_tab, cat2="simbad", max_distance=180 * u.arcsec, colRA1="TIC_RA", colDec1="TIC_DEC")

    to_csv(res, out_path, mode="w")

    res


def _save_simbad_meta(meta_table, out_path, mode="a"):
    to_csv(meta_table, out_path, mode=mode)


def get_and_save_simbad_meta_of_all_by_tics(chunk_size=1000, start_chunk=0, end_chunk_inclusive=None):
    # TODO: refactor with logic in tic_meta
    ids = load_tic_ids_from_file()
    num_chunks = np.floor(len(ids) / chunk_size)
    # the actual trunk size could be slightly different, as array_split would split it to equal size chunk
    id_chunks = np.array_split(ids, num_chunks)
    max_chunk_id = len(id_chunks) - 1  # largest possible value

    if end_chunk_inclusive is None:
        end_chunk_inclusive = max_chunk_id

    if end_chunk_inclusive > max_chunk_id:
        print(
            f"WARN end_chunk_inclusive {end_chunk_inclusive} is larger than actual "
            f"num. of chunks. Set it to the largest {max_chunk_id}"
        )
        end_chunk_inclusive = max_chunk_id

    id_chunks = id_chunks[slice(start_chunk, end_chunk_inclusive + 1)]

    # Process the rest of the chunks (append to the existing csv)
    out_path = "cache/simbad_meta_by_ticid.csv"
    kwargs_list = [dict(tics=ids) for ids in id_chunks]

    bulk_process(
        _get_simbad_meta_of_tics,
        kwargs_list,
        process_result_func=lambda res, call_i, call_kwargs: _save_simbad_meta(res, out_path),
    )


def load_simbad_meta_table_from_file(csv_path="../data/simbad_meta.csv"):
    df = pd.read_csv(csv_path)
    return df


def _load_simbad_xmatch_table_from_file(csv_path="cache/simbad_tics_xmatch.csv", max_results_per_target=None):
    df = pd.read_csv(csv_path)

    if max_results_per_target is not None:
        # for each TIC, select n closest one.
        # based on: https://stackoverflow.com/a/41826756
        df = df.sort_values("angDist", ascending=True).groupby("TIC_ID").head(max_results_per_target)

    df = df.sort_values(
        ["TIC_ID", "angDist"],
        ascending=True,
    )
    df = df.reset_index(drop=True)

    return df


def get_and_save_simbad_meta_of_all_by_xmatch(max_results_per_target=5):
    """For TICs not found by tic id lookup, use crossmatch result"""
    out_path = "cache/simbad_meta_candidates_by_xmatch.csv"

    df_xmatch = _load_simbad_xmatch_table_from_file(
        csv_path="cache/simbad_tics_xmatch.csv", max_results_per_target=max_results_per_target
    )

    # for expedience, I make 1 call rather than splitting it into smaller chunks
    # empirically, I know the result set size (~5000) is small enough to be done in 1 call.
    res = _get_simbad_meta_of_ids(df_xmatch["main_id"])

    # we lookup by main_id , so TYPED_ID is just redundant, we replace it with TIC_ID
    # we cannot just copy df_xmatch["TIC_ID"] over because of edge cases that some lookups fail
    #  (probably because the simbad data used by xmatch is out-of-date)
    res.rename_column("TYPED_ID", "TIC_ID")
    res["TIC_ID"] = [-1 for s in res["TIC_ID"]]
    res["angDist"] = [-1.0 for s in res["TIC_ID"]]
    for row in res:
        main_id = row["MAIN_ID"]
        xmatch_rows = df_xmatch[df_xmatch["main_id"] == main_id].reset_index(drop=True)
        if len(xmatch_rows) > 0:
            row["TIC_ID"] = xmatch_rows["TIC_ID"][0]
            row["angDist"] = xmatch_rows["angDist"][0]
        else:
            print(f"WARN for SIMBAD entry {main_id}, cannot find TIC ID in crossmatch result unexpectedly.")

    _save_simbad_meta(res, out_path, mode="w")
    return


class MatchResult(xmatch_util.AbstractMatchResult):
    def __init__(
        self, mag, mag_band, mag_diff, pm, pmra_diff_pct, pmdec_diff_pct, plx, plx_diff_pct, aliases, num_aliases_matched
    ):
        self.mag = mag
        self.mag_band = mag_band
        self.mag_diff = mag_diff
        self.pm = pm
        self.pmra_diff_pct = pmra_diff_pct
        self.pmdec_diff_pct = pmdec_diff_pct
        self.plx = plx
        self.plx_diff_pct = plx_diff_pct
        self.aliases = aliases
        self.num_aliases_matched = num_aliases_matched

    def get_flags(self):
        return [self.mag, self.pm, self.plx, self.aliases]

    def get_weights(self):
        return [2, 1, 1, 1]


def _calc_matches(simbad_meta_row, tic_meta_row):

    max_mag_diff = 1.0
    max_pmra_diff_pct = 25
    max_pmdec_diff_pct = 25
    max_plx_diff_pct = 25

    tic_label = f"TIC {tic_meta_row['ID']}"

    bands_t = ["Vmag", "Tmag", "GAIAmag", "Bmag"]  # in TIC
    bands_s = ["FLUX_V", "FLUX_R", "FLUX_G", "FLUX_B"]  # in SIMBAD
    mag_match = None
    mag_match_band = None
    mag_diff = None
    for bt, bs in zip(bands_t, bands_s):
        mag_diff = xmatch_util.diff_between(tic_meta_row[bt], simbad_meta_row[bs], label=f"{tic_label} magnitude ({bt} {bs})")
        if mag_diff is not None:
            mag_match_band = bt
            mag_match = mag_diff < max_mag_diff
            break
        #  else no data in TIC and/or SIMBAD, try the next band

    pm_match, pmra_diff_pct, pmdec_diff_pct = xmatch_util.calc_pm_matches(
        tic_meta_row,
        simbad_meta_row["PMRA"],
        simbad_meta_row["PMDEC"],
        max_pmra_diff_pct=max_pmra_diff_pct,
        max_pmdec_diff_pct=max_pmdec_diff_pct,
    )

    plx_match, plx_diff_pct = xmatch_util.calc_scalar_matches(
        tic_meta_row, "plx", simbad_meta_row["PLX_VALUE"], max_diff_pct=max_plx_diff_pct
    )

    simbad_aliases = get_aliases(simbad_meta_row)
    tic_aliases = tic_meta.get_aliases(tic_meta_row)
    num_aliases_matched = len([1 for a in tic_aliases if a in simbad_aliases])
    aliases_match = num_aliases_matched > 0

    return MatchResult(
        mag_match,
        mag_match_band,
        mag_diff,
        pm_match,
        pmra_diff_pct,
        pmdec_diff_pct,
        plx_match,
        plx_diff_pct,
        aliases_match,
        num_aliases_matched,
    )


def _calc_matches_for_all(df, df_tics, match_method_label):
    # we basically filter the candidates list, `df`
    # by comparing the metadata against those from TIC Catalog, `df_tics`
    # all of the smart logic is encapsulated here

    df_len = len(df)
    match_result_columns = {
        "Match_Method": np.full(df_len, match_method_label),
        "Match_Score": np.zeros(df_len, dtype=int),
        "Match_Mag": np.full(df_len, "", dtype="O"),
        "Match_PM": np.full(df_len, "", dtype="O"),
        "Match_Plx": np.full(df_len, "", dtype="O"),
        "Match_Aliases": np.full(df_len, "", dtype="O"),
        "Match_Mag_Band": np.full(df_len, "", dtype="O"),
        "Match_Mag_Diff": np.zeros(df_len, dtype=float),
        "Match_PMRA_DiffPct": np.zeros(df_len, dtype=float),
        "Match_PMDEC_DiffPct": np.zeros(df_len, dtype=float),
        "Match_Plx_DiffPct": np.zeros(df_len, dtype=float),
        "Match_Aliases_NumMatch": np.zeros(df_len, dtype=int),
    }

    def match_func(row_xmatch, row_tics, i, match_result_columns):
        match_result = _calc_matches(row_xmatch, row_tics)
        cols = match_result_columns  # to abbreviate it
        cols["Match_Score"][i] = match_result.score()
        cols["Match_Mag"][i] = match_result.to_flag_str("mag")
        cols["Match_Mag_Band"][i] = match_result.mag_band
        cols["Match_Mag_Diff"][i] = match_result.mag_diff

        cols["Match_PM"][i] = match_result.to_flag_str("pm")
        cols["Match_PMRA_DiffPct"][i] = match_result.pmra_diff_pct
        cols["Match_PMDEC_DiffPct"][i] = match_result.pmdec_diff_pct

        cols["Match_Plx"][i] = match_result.to_flag_str("plx")
        cols["Match_Plx_DiffPct"][i] = match_result.plx_diff_pct

        cols["Match_Aliases"][i] = match_result.to_flag_str("aliases")
        cols["Match_Aliases_NumMatch"][i] = match_result.num_aliases_matched

    return xmatch_util.do_calc_matches_with_tics(df, df_tics, match_result_columns, match_func)


def find_and_save_simbad_best_xmatch_meta(dry_run=False, dry_run_size=1000, min_score_to_include=None):
    out_path_accepted = "cache/simbad_meta_by_xmatch.csv"
    out_path_rejected = "cache/simbad_meta_by_xmatch_rejected.csv"  # those with low match score

    df = load_simbad_meta_table_from_file("cache/simbad_meta_candidates_by_xmatch.csv")
    # filter out non-stellar candidates, they are not relevant for TIC matches
    df = df[df["OTYPES"].str.contains("[*]", na=False)].reset_index(drop=True)

    def _calc_matches_for_all_for_xmatch(df, df_tics):
        return _calc_matches_for_all(df, df_tics, match_method_label="co")  # shorthand for coordinate

    return xmatch_util.find_and_save_best_xmatch_meta(
        df,
        out_path_accepted,
        out_path_rejected,
        _calc_matches_for_all_for_xmatch,
        dry_run=dry_run,
        dry_run_size=dry_run_size,
        min_score_to_include=min_score_to_include,
    )


def combine_and_save_simbad_meta_by_tics_and_xmatch(min_score_to_include=0):
    out_path_accepted = "../data/simbad_meta.csv"
    out_path_rejected = "../data/simbad_meta_rejected.csv"  # those with low match score

    df_tic_meta = tic_meta.load_tic_meta_table_from_file()

    df_by_xmatch = load_simbad_meta_table_from_file("cache/simbad_meta_by_xmatch.csv")
    df_by_ticid = load_simbad_meta_table_from_file("cache/simbad_meta_by_ticid.csv")

    # for those found by TIC ID lookups,
    #
    # 1. we exclude those that will be replaced by xmatch
    #   (the by ticid lookup produces a row even for TICs that is not found)
    df_by_ticid = df_by_ticid[~df_by_ticid["TIC_ID"].isin(df_by_xmatch["TIC_ID"])]

    # 2. we add match scores (and angDist column) to make its schema the same as those from xmatch
    df_by_ticid["angDist"] = np.nan
    df_by_ticid = _calc_matches_for_all(df_by_ticid, df_tic_meta, match_method_label="tic")

    df = pd.concat([df_by_ticid, df_by_xmatch])
    df = df.sort_values("TIC_ID", ascending=True)

    # tied to the original astroquery. not really useful in the final output
    df = df.drop("SCRIPT_NUMBER_ID", axis=1)

    df_accepted = df[df["Match_Score"] >= min_score_to_include].reset_index(drop=True)
    df_rejected = df[df["Match_Score"] < min_score_to_include].reset_index(drop=True)

    to_csv(df_accepted, out_path_accepted, mode="w")
    to_csv(df_rejected, out_path_rejected, mode="w")

    return df_accepted, df_rejected


def get_aliases(simbad_meta_row):
    aliases_str = simbad_meta_row["IDS"]
    if has_value(aliases_str):
        # TODO: some of the names in TIC are not standardized,
        # (incompatible with the form used by SIMBAD, VSX, etc.)
        # e.g., TYC has leading zeros
        # they should be normalized
        return aliases_str.split("|")
    else:
        return []


def to_simbad_var_otypes_str(otypes_str):
    """Return a SIMBAD OTYPES without those unrelated to variability"""
    if pd.isna(otypes_str):
        return np.nan

    non_var_otypes = set(["*", "**", "PM*", "IR", "X", "UV"])
    otypes = set(otypes_str.split("|"))
    var_otypes = otypes - non_var_otypes

    if len(var_otypes) > 0:
        return "|".join(list(var_otypes))
    else:
        return np.nan


#
# Mapping SIMBAD type (OTYPE) to EB Classification
#
class SIMBADTypeMapAccessor(AbstractTypeMapAccessor):
    # Note:
    # `../data/auxillary/simbad_typemap.csv` is constructed by
    # - turning a list of otypes to a table based on SIMBAD definition
    #   , from `_to_typemap_df()`
    # - manually enter Is_EB value
    def __init__(self, csv_path="../data/auxillary/simbad_typemap.csv"):
        super().__init__(csv_path, "SIMBAD_Type")

    def _split_types_str(self, types_str):
        return types_str.split("|")


def map_and_save_simbad_is_eb_of_all(warn_types_not_mapped=False):
    out_path = "../data/simbad_is_eb.csv"
    typemap = SIMBADTypeMapAccessor()
    df = load_simbad_meta_table_from_file()

    map_res = [typemap.map(otypes).label for otypes in df["OTYPES"]]

    # return a useful subset of columns, in addition to the EB map result
    res = df[["MAIN_ID", "TIC_ID", "OTYPES", "V__vartyp", "angDist", "Match_Score"]]
    insert(res, before_colname="OTYPES", colname="Is_EB", value=map_res)

    to_csv(res, out_path, mode="w")
    not_mapped_types_seen = list(typemap.not_mapped_types_seen)
    if warn_types_not_mapped and len(not_mapped_types_seen) > 1:
        print(f"WARN: there are {len(not_mapped_types_seen)} number of OTYPE value not mapped.")
        print(not_mapped_types_seen)
    return res, not_mapped_types_seen


def _to_typemap_df(otypes, default_is_eb_value=""):
    # use case: map a list of otypes that is previously not in OTYPES - IsEB map
    otypes_map = SIMBADOTypesAccessor().otypes

    def get_description(otype):
        r = otypes_map.get(otype)
        if r is not None:
            return f"{r.get('description', '')} | {r.get('category', '')} | {r.get('subcategory', '')}"
        else:
            return ""

    is_eb = np.full_like(otypes, default_is_eb_value)
    description = [get_description(otype) for otype in otypes]
    notes = np.full_like(otypes, "")
    df = pd.DataFrame(
        {
            "SIMBAD_Type": otypes,
            "Is_EB": is_eb,
            "Description": description,
            "Notes": notes,
        }
    )
    return df


def load_simbad_is_eb_table_from_file(csv_path="../data/simbad_is_eb.csv"):
    df = pd.read_csv(csv_path)
    return df


class SIMBADOTypesAccessor:
    @classmethod
    def _get_otypes_from_remote(cls, url=None):
        if url is None:
            # local version of https://simbad.cds.unistra.fr/guide/otypes/json/otype_nodes.json
            with open("../data/auxillary/simbad_otype_nodes.json", mode="r") as f:
                return json.load(f)
        else:
            return fetch_json(url)

    def __init__(self, url=None):
        self.raw_list = SIMBADOTypesAccessor._get_otypes_from_remote(url)
        self.otypes = dict()

        for row in self.raw_list:
            key = row.get("id")
            if key is not None:
                self.otypes[key] = row
            key = row.get("candidate")
            if key is not None:
                # create an entry for the candidate variant
                label = row.get("label", "")
                label = f"{label}?"
                desc = row.get("description", "")
                desc = f"{desc} candidate"
                row = row.copy()
                row["label"] = label
                row["description"] = desc
                self.otypes[key] = row


if __name__ == "__main__":
    # For now, the calls that would actually query SIMBAD / Vizier are commented out.
    # The typical workflow is to tweak the mapping logic on the downloaded data.
    # No need to reissue queries in such cases.

    # 1. process those that can be found by TIC id lookups
    # get_and_save_simbad_meta_of_all_by_tics()

    # 2. process the rest by coordinate search
    # 2a. use crossmatch to get a list of potential simbad objects
    # xmatch_and_save_all_unmatched_tics()
    # 2b. Use the list from crossmatch to get and save the simbad entries
    # get_and_save_simbad_meta_of_all_by_xmatch(max_results_per_target=5)
    # 2c. for each applicable TIC, select the best candidate among the results
    #    from crossmatch
    find_and_save_simbad_best_xmatch_meta()

    # 3. Combine those from TIC id lookups and those from coordinate crossmatch
    #    - filter out those with low match scores
    combine_and_save_simbad_meta_by_tics_and_xmatch(min_score_to_include=0)

    # for each SIMBAD record, map it OTYPES to Is_EB
    # it depends on the mapping defined in `data/auxillary/simbad_typemap.csv`
    map_and_save_simbad_is_eb_of_all(warn_types_not_mapped=True)
