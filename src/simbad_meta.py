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


if __name__ =="__main__":
    # 1. process those that can be found by TIC id lookups
    # get_and_save_simbad_meta_of_all_by_tics()

    # 2. process the rest by coordinate search
    # 2a. use crossmatch to get a list of potential simbad objects
    # xmatch_and_save_all_unmatched_tics()
    # 2b. Use the list from crossmatch to get and save the simbad entries
    get_and_save_simbad_meta_of_all_by_xmatch(max_results_per_target=5)

