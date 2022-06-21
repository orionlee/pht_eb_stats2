from astroquery.simbad import Simbad

from ratelimit import limits, sleep_and_retry

from common import *

# throttle HTTP calls to MAST
# somewhat large result set (~1000 rows), so I set a conservative throttle to be extra safe
NUM_CALLS = 1
PERIOD_IN_SECONDS = 20

@sleep_and_retry
@limits(calls=NUM_CALLS, period=PERIOD_IN_SECONDS)
def _get_simbad_meta_of_tics(tics):
    simbad = Simbad()

    simbad.remove_votable_fields('coordinates')
    fields_to_add = [
        "typed_id",
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

    res = simbad.query_objects([f"TIC {id}" for id in tics])

    # changed output result
    res.rename_column("RA_d_ICRS_J2000_2000", "RA")
    res.rename_column("DEC_d_ICRS_J2000_2000", "DEC")
    res.rename_column("TYPED_ID", "TIC_ID")
    res["TIC_ID"] = [int(s.replace('TIC ', '')) for s in res["TIC_ID"]]
    # column "SCRIPT_NUMBER_ID": retained for now as it could be useful for troubleshooting
    # , as it is referenced by SIMBAD warnings/ errors.

    return res


def _save_simbad_meta(meta_table, out_path):
    to_csv(meta_table, out_path, mode="a")


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


if __name__ =="__main__":
    # process those that can be found by TIC id lookups
    get_and_save_simbad_meta_of_all_by_tics()
    # TODO: process the rest by coordinate search
