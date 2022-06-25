from enum import Enum, unique
import json
from types import SimpleNamespace

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astroquery.simbad import Simbad
from astroquery.xmatch import XMatch

from ratelimit import limits, sleep_and_retry

from common import *
import tic_meta

# throttle HTTP calls to MAST
# somewhat large result set (~1000 rows), so I set a conservative throttle to be extra safe
NUM_CALLS = 1
PERIOD_IN_SECONDS = 20

def _get_simbad(add_typed_id=False):
    simbad = Simbad()

    simbad.remove_votable_fields('coordinates')
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
    res["TIC_ID"] = [int(s.replace('TIC ', '')) for s in res["TIC_ID"]]
    return res

@sleep_and_retry
@limits(calls=NUM_CALLS, period=PERIOD_IN_SECONDS)
def _get_simbad_meta_of_ids(ids):
    simbad = _get_simbad(add_typed_id=True)
    res = simbad.query_objects(ids)
    _format_result(res)
    return res


def _get_simbad_meta_of_coordinates(ra, dec, coord_kwargs=dict(unit=u.deg, frame="icrs", equinox="J2000"), radius=2 * u.arcmin, max_rows_per_coord=5,):
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
    res = XMatch.query(cat1=src_tab, cat2="simbad", max_distance=180*u.arcsec, colRA1="TIC_RA", colDec1="TIC_DEC")

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
        print(f"WARN end_chunk_inclusive {end_chunk_inclusive} is larger than actual num. of chunks. Set it to the largest {max_chunk_id}")
        end_chunk_inclusive = max_chunk_id

    id_chunks = id_chunks[slice(start_chunk, end_chunk_inclusive + 1)]

    # Process the rest of the chunks (append to the existing csv)
    out_path = "cache/simbad_meta_by_ticid.csv"
    kwargs_list = [dict(tics=ids) for ids in id_chunks]

    bulk_process(_get_simbad_meta_of_tics, kwargs_list, process_result_func=lambda res, call_i, call_kwargs: _save_simbad_meta(res, out_path))


def load_simbad_meta_table_from_file(csv_path="../data/simbad_meta.csv"):
    df = pd.read_csv(csv_path)
    return df


def _load_simbad_xmatch_table_from_file(csv_path="cache/simbad_tics_xmatch.csv", max_results_per_target=None):
    df = pd.read_csv(csv_path)

    if max_results_per_target is not None:
        # for each TIC, select n closest one.
        # based on: https://stackoverflow.com/a/41826756
        df = df.sort_values("angDist", ascending=True).groupby("TIC_ID").head(max_results_per_target)

    df = df.sort_values(["TIC_ID", "angDist"], ascending=True, )
    df = df.reset_index(drop=True)

    return df


def get_and_save_simbad_meta_of_all_by_xmatch(max_results_per_target=5):
    """For TICs not found by tic id lookup, use crossmatch result"""
    out_path = "cache/simbad_meta_candidates_by_xmatch.csv"

    df_xmatch = _load_simbad_xmatch_table_from_file(csv_path="cache/simbad_tics_xmatch.csv", max_results_per_target=max_results_per_target)

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


def _3val_flag_to_str(val):
    if val is None:
        return '-'
    elif val:
        return 'T'
    else:
        return 'F'

class MatchResult(SimpleNamespace):
    def __init__(self, mag, mag_band, mag_diff, pm, pmra_diff_pct, pmdec_diff_pct, plx, plx_diff_pct, aliases, num_aliases_matched):
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

    def _flag_to_score(self, val):
        if val is None:
            return 0
        elif val:
            return 1
        else:
            return -1

    def score(self):
        flags = [self.mag, self.pm, self.plx, self.aliases]
        weight = [2, 1, 1, 1]
        scores = [self._flag_to_score(f) * w for f, w in zip(flags, weight)]
        return np.sum(scores)



def _calc_matches(simbad_meta_row, tic_meta_row):

    max_mag_diff = 1.0
    max_pmra_diff_pct = 25
    max_pmdec_diff_pct = 25
    max_plx_diff_pct = 25

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

    tic_label = f"TIC {tic_meta_row['ID']}"

    bands_t = ["Vmag", "Tmag", "GAIAmag", "Bmag"]  # in TIC
    bands_s = ["FLUX_V", "FLUX_R", "FLUX_G",  "FLUX_B"]  # in SIMBAD
    mag_match = None
    mag_match_band = None
    mag_diff = None
    for bt, bs in zip(bands_t, bands_s):
        mag_diff = _diff(tic_meta_row[bt], simbad_meta_row[bs], label=f"{tic_label} magnitude ({bt} {bs})")
        if mag_diff is not None:
            mag_match_band = bt
            mag_match = mag_diff < max_mag_diff
            break
        #  else no data in TIC and/or SIMBAD, try the next band

    pmra_diff_pct = _diff(tic_meta_row["pmRA"], simbad_meta_row["PMRA"], in_percent=True, label=f"{tic_label} pmRA")
    pmdec_diff_pct = _diff(tic_meta_row["pmDEC"], simbad_meta_row["PMDEC"], in_percent=True, label=f"{tic_label} pmDEC")

    pm_match = None
    if pmra_diff_pct is not None and pmdec_diff_pct is not None:
        if  pmra_diff_pct < max_pmra_diff_pct and pmdec_diff_pct < max_pmdec_diff_pct:
            pm_match = True
        else:
            pm_match = False

    plx_diff_pct = _diff(tic_meta_row["plx"], simbad_meta_row["PLX_VALUE"], in_percent=True, label=f"{tic_label} plx")

    plx_match = None
    if plx_diff_pct is not None:
        plx_match = plx_diff_pct < max_plx_diff_pct

    simbad_aliases = get_aliases(simbad_meta_row)
    tic_aliases = tic_meta.get_aliases(tic_meta_row)

    num_aliases_matched =  len([1 for a in tic_aliases if a in simbad_aliases])
    aliases_match = num_aliases_matched > 0


    return MatchResult(mag_match, mag_match_band, mag_diff, pm_match, pmra_diff_pct, pmdec_diff_pct, plx_match, plx_diff_pct, aliases_match, num_aliases_matched)


def _calc_matches_for_all(df, df_tics, match_method_label, min_score_to_include=None):
    # we basically filter the candidates list, `df`
    # by comparing the metadata against those from TIC Catalog, `df_tics`
    # all of the smart logic is encapsulated here

    df["Match_Method"] = match_method_label
    df["Match_Score"] = 0
    df["Match_Mag"] = ""
    df["Match_PM"] = ""
    df["Match_Plx"] = ""
    df["Match_Aliases"] = ""
    df["Match_Mag_Band"] = ""
    df["Match_Mag_Diff"] = 0.0
    df["Match_PMRA_DiffPct"] = 0.0
    df["Match_PMDEC_DiffPct"] = 0.0
    df["Match_Plx_DiffPct"] = 0.0
    df["Match_Aliases_NumMatch"] = 0

    # for each candidate in df, compute how it matches with the expected TIC
    # Technical note: update via .iterrows() is among the slowest methods
    # but given our match semantics is not trivial, I settle for using it.
    #
    # I also consider defer the DataFrame update after the iteration in a batch
    # (and hold the result in ndarray in the loop)
    # Empirical test shows that the iteration alone (along with fetch the match row in df_tics)
    # account for 50+% of the running time. So I decide not to pursue any more optimization for now.
    #
    # optimization: make lookup a row in df_tics by tic_id fast by using it as an index
    # it saves ~30+% of the running time for a ~10,000 rows dataset
    df_tics = df_tics.set_index("ID", drop=False)  # we still want df_tics["ID"] work after using it as an index
    for i_s, row_s in df.iterrows():
        tic_id = row_s["TIC_ID"]
        # Note: a KeyError would be raised if tic_id is unexpected not found in df_tics
        # in practice it shouldn't happen to our dataset.
        row_t = df_tics.loc[tic_id]
        match_result = _calc_matches(row_s, row_t)
        # print(f"DBG {tic_id} {match_result}")
        df.at[i_s, 'Match_Score'] = match_result.score()
        df.at[i_s, 'Match_Mag'] = _3val_flag_to_str(match_result.mag)
        df.at[i_s, 'Match_Mag_Band'] = match_result.mag_band
        df.at[i_s, 'Match_Mag_Diff'] = match_result.mag_diff

        df.at[i_s, 'Match_PM'] = _3val_flag_to_str(match_result.pm)
        df.at[i_s, 'Match_PMRA_DiffPct'] = match_result.pmra_diff_pct
        df.at[i_s, 'Match_PMDEC_DiffPct'] = match_result.pmdec_diff_pct

        df.at[i_s, 'Match_Plx'] = _3val_flag_to_str(match_result.plx)
        df.at[i_s, 'Match_Plx_DiffPct'] = match_result.plx_diff_pct

        df.at[i_s, 'Match_Aliases'] = _3val_flag_to_str(match_result.aliases)
        df.at[i_s, 'Match_Aliases_NumMatch'] = match_result.num_aliases_matched

    # Exclude those with low match scores, default to exclude negative scores
    if min_score_to_include is not None:
        df = df[df["Match_Score"] >= min_score_to_include].reset_index(drop=True)

    if "angDist" in df.columns:
        sort_colnames, ascending = ["TIC_ID", "Match_Score", "angDist"], [True, False, True]
    else:
        sort_colnames, ascending = ["TIC_ID", "Match_Score"], [True, False]

    df.sort_values(sort_colnames, ascending=ascending, inplace=True, ignore_index=True)

    # For each TIC, select the one with the best score (it's sorted above)
    df = df.groupby("TIC_ID").head(1).reset_index(drop=True)

    return df


def find_and_save_simbad_best_xmatch_meta(min_score_to_include=None):
    out_path = "cache/simbad_meta_by_xmatch.csv"

    df = load_simbad_meta_table_from_file("cache/simbad_meta_candidates_by_xmatch.csv")
    # filter out non-stellar candidates, they are not relevant for TIC matches
    df = df[df["OTYPES"].str.contains("[*]", na=False)].reset_index(drop=True)

    df_tics = tic_meta.load_tic_meta_table_from_file()

    df = _calc_matches_for_all(
        df, df_tics, match_method_label="co",  # shorthand for co-ordinate
        min_score_to_include=min_score_to_include)

    to_csv(df, out_path, mode="w")

    return df


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
    df_by_ticid = _calc_matches_for_all(
        df_by_ticid, df_tic_meta, match_method_label="tic",
        min_score_to_include=None)

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
        return aliases_str.split("|")
    else:
        return []


#
# Mapping SIMBAD type (OTYPE) to EB Classification
#

@unique
class MapResult(Enum):
    def __new__(cls, value, label):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.label = label
        return obj

    TRUE = (4, "T")
    FALSE = (3, "F")
    NOT_MAPPED = (2, "?")
    NA = (1, "-")

class SIMBADTypeMapAccessor:
    def __init__(self, csv_path="../data/simbad_typemap.csv"):
        def _to_map_result(is_eb):
            if pd.isna(is_eb):
                res =  MapResult.NA
            elif is_eb:
                res = MapResult.TRUE
            else:
                res = MapResult.FALSE
            return res

        self.df = pd.read_csv(csv_path)

        # convert the mapping needed from dataframe to a dictionary
        # to avoid the overhead of repeated dataframe access
        is_eb_col = [_to_map_result(is_eb) for is_eb in self.df["Is_EB"]]
        self._otypes_dict = dict(zip(self.df["SIMBAD_Type"], is_eb_col ))

        self.not_mapped_otypes_seen = set()


    def _map_1_otype(self, otype):
        res = self._otypes_dict.get(otype, MapResult.NOT_MAPPED)
        if res == MapResult.NOT_MAPPED:
            self.not_mapped_otypes_seen.add(otype)
        return res

    def map(self, otypes):
        otypes = otypes.split("|")
        res_list = [self._map_1_otype(ot).value for ot in otypes]
        best = np.max(res_list)
        return MapResult(best)


def map_and_save_simbad_otypes_of_all():
    out_path = "../data/simbad_is_eb.csv"
    typemap = SIMBADTypeMapAccessor()
    df = load_simbad_meta_table_from_file()

    map_res = [typemap.map(otypes).label for otypes in df["OTYPES"]]

    # return a useful subset of columns, in addition to the EB map result
    res = df[["MAIN_ID", "TIC_ID", "OTYPES", "V__vartyp", "angDist", "Match_Score"]]
    res.insert(2, "Is_EB_SIMBAD", map_res)

    to_csv(res, out_path, mode="w")
    return res, list(typemap.not_mapped_otypes_seen)


def _to_typemap_df(otypes, default_is_eb_value=""):
    # use case: map a list of otypes that is previously not in OTYPES - IsEB map
    otypes_map = SIMBADOTypesAccessor().otypes
    def get_description(otype):
        r = otypes_map.get(otype)
        if r is not None:
            return f"{r.get('description', '')} | {r.get('category', '')} | {r.get('subcategory', '')}"
        else:
            return ''

    is_eb = np.full_like(otypes, default_is_eb_value)
    description = [get_description(otype) for otype in otypes]
    notes = np.full_like(otypes, "")
    df = pd.DataFrame({
        "SIMBAD_Type": otypes,
        "Is_EB": is_eb,
        "Description": description,
        "Notes": notes,
        })
    return df


class SIMBADOTypesAccessor():

    @classmethod
    def _get_otypes_from_remote(cls, url=None):
        if url is None:
            # local version of https://simbad.cds.unistra.fr/guide/otypes/json/otype_nodes.json
            with open("../data/simbad_otype_nodes.json", mode="r") as f:
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


if __name__ =="__main__":
    # 1. process those that can be found by TIC id lookups
    # get_and_save_simbad_meta_of_all_by_tics()

    # 2. process the rest by coordinate search
    # 2a. use crossmatch to get a list of potential simbad objects
    # xmatch_and_save_all_unmatched_tics()
    # 2b. Use the list from crossmatch to get and save the simbad entries
    # get_and_save_simbad_meta_of_all_by_xmatch(max_results_per_target=5)
    # 2c. for each applicable TIC, select the best candidate among the results
    #    from crossmatch
    # find_and_save_simbad_best_xmatch_meta()

    # 3. Combine those from TIC id lookups and those from coordinate crossmatch
    #    - filter out those with low match scores
    combine_and_save_simbad_meta_by_tics_and_xmatch(min_score_to_include=0)

