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



def _calc_matches(tic_meta_row, simbad_meta_row):

    max_mag_diff = 0.5
    max_pmra_diff_pct = 25
    max_pmdec_diff_pct = 25
    max_plx_diff_pct = 25

    def _diff(val1, val2, in_percent=False):
        if has_value(val1) and has_value(val2):
            diff = np.abs(val1 - val2)
            if not in_percent:
                return diff
            else:
                return 100.0 * diff / val1
        else:
            return None

    bands_t = ["Vmag", "Tmag", "GAIAmag", "Bmag"]  # in TIC
    bands_s = ["FLUX_V", "FLUX_R", "FLUX_G",  "FLUX_B"]  # in SIMBAD
    mag_match = None
    mag_match_band = None
    mag_diff = None
    for bt, bs in zip(bands_t, bands_s):
        mag_diff = _diff(tic_meta_row[bt], simbad_meta_row[bs])
        if mag_diff is not None:
            mag_match_band = bt
            mag_match = mag_diff < max_mag_diff
            break
        #  else no data in TIC and/or SIMBAD, try the next band

    pmra_diff_pct = _diff(tic_meta_row["pmRA"], simbad_meta_row["PMRA"], in_percent=True)
    pmdec_diff_pct = _diff(tic_meta_row["pmDEC"], simbad_meta_row["PMDEC"], in_percent=True)

    pm_match = None
    if pmra_diff_pct is not None and pmdec_diff_pct is not None:
        if  pmra_diff_pct < max_pmra_diff_pct and pmdec_diff_pct < max_pmdec_diff_pct:
            pm_match = True
        else:
            pm_match = False

    plx_diff_pct = _diff(tic_meta_row["plx"], simbad_meta_row["PLX_VALUE"], in_percent=True)

    plx_match = None
    if plx_diff_pct is not None:
        plx_match = plx_diff_pct < max_plx_diff_pct

    simbad_aliases = get_aliases(simbad_meta_row)
    tic_aliases = tic_meta.get_aliases(tic_meta_row)

    num_aliases_matched =  len([1 for a in tic_aliases if a in simbad_aliases])
    aliases_match = num_aliases_matched > 0


    return MatchResult(mag_match, mag_match_band, mag_diff, pm_match, pmra_diff_pct, pmdec_diff_pct, plx_match, plx_diff_pct, aliases_match, num_aliases_matched)


def find_and_save_simbad_best_xmatch_meta(min_scores_to_include=0):
    out_path = "cache/simbad_meta_by_xmatch.csv"

    # we basically filter the candidates list by comparing the metadata against those from TIC Catalog
    # all of the smart logic is encapsulated here

    df = load_simbad_meta_table_from_file("cache/simbad_meta_candidates_by_xmatch.csv")
    # filter out non-stellar candidates, they are not relevant for TIC matches
    df = df[df["OTYPES"].str.contains("[*]", na=False)].reset_index(drop=True)

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
    df_tics = tic_meta.load_tic_meta_table_from_file()
    for i_s, row_s in df.iterrows():
        tic_id = row_s["TIC_ID"]
        df_t = df_tics[df_tics["ID"] == tic_id]
        if len(df_t) < 1:
            print(f"WARN TIC {tic_id} cannot be found in TIC metadata table")
            continue
        match_result = _calc_matches(df_t.iloc[0], row_s)
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
    if min_scores_to_include is not None:
        df = df[df["Match_Score"] >= min_scores_to_include].reset_index(drop=True)

    df.sort_values(["TIC_ID", "Match_Score", "angDist"], ascending=[True, False, True], inplace=True, ignore_index=True)

    # For each TIC, select the one with the best score
    df = df.groupby("TIC_ID").head(1).reset_index(drop=True)

    to_csv(df, out_path, mode="w")

    return df


def get_aliases(simbad_meta_row):
    aliases_str = simbad_meta_row["IDS"]
    if has_value(aliases_str):
        return aliases_str.split("|")
    else:
        return []


if __name__ =="__main__":
    # 1. process those that can be found by TIC id lookups
    # get_and_save_simbad_meta_of_all_by_tics()

    # 2. process the rest by coordinate search
    # 2a. use crossmatch to get a list of potential simbad objects
    # xmatch_and_save_all_unmatched_tics()
    # 2b. Use the list from crossmatch to get and save the simbad entries
    get_and_save_simbad_meta_of_all_by_xmatch(max_results_per_target=5)

