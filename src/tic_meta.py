import re

from astroquery.mast import Catalogs
import numpy as np
import pandas as pd

from ratelimit import limits, sleep_and_retry

from common import *

# throttle HTTP calls to MAST
# somewhat large result set (~1000 rows), so I set a conservative throttle to be extra safe
NUM_CALLS = 1
PERIOD_IN_SECONDS = 30


@sleep_and_retry
@limits(calls=NUM_CALLS, period=PERIOD_IN_SECONDS)
def _get_tic_meta_of_tics(tics, **kwargs):
    """Return the info of TICs in the TIC catalog

    For TIC table column description, see:
    https://outerspace.stsci.edu/display/TESS/TIC+v8.2+and+CTL+v8.xx+Data+Release+Notes
    """
    # from astropy.table import Table
    # return Table(dict(xid=tics)  # dummy result for testing
    return Catalogs.query_criteria(catalog="Tic", ID=tics, **kwargs)


def save_tic_meta(meta_table, call_i=None, call_kwargs=None, csv_mode="a", csv_header=False):
    out_path = "../data/tic_meta.csv"
    meta_table.to_pandas().to_csv(out_path, index=False, mode=csv_mode, header=csv_header)


def get_and_save_tic_meta_of_all(
    chunk_size=1000, start_chunk=0, end_chunk_inclusive=None, pause_time_between_chunk_seconds=10
):
    ids = load_tic_ids_from_file()
    num_chunks = np.floor(len(ids) / chunk_size)
    # the actual trunk size could be slightly different, as array_split would split it to equal size chunk
    id_chunks = np.array_split(ids, num_chunks)
    max_chunk_id = len(id_chunks) - 1  # largest possible value

    if end_chunk_inclusive is None:
        end_chunk_inclusive = max_chunk_id

    if end_chunk_inclusive > max_chunk_id:
        print(
            f"WARN end_chunk_inclusive {end_chunk_inclusive} is larger than actual num. of chunks. Set it to the largest {max_chunk_id}"
        )
        end_chunk_inclusive = max_chunk_id

    # chunk 0: create a new csv, add header
    if start_chunk == 0:
        kwargs_list = [dict(tics=id_chunks[0])]
        bulk_process(
            _get_tic_meta_of_tics,
            kwargs_list,
            process_result_func=lambda res, call_i, call_kwargs: save_tic_meta(res, csv_mode="w", csv_header=True),
        )

        # Process the rest of the chunks (append to the existing csv)
        kwargs_list = [dict(tics=ids) for ids in id_chunks[1:]]
        bulk_process(_get_tic_meta_of_tics, kwargs_list, process_result_func=save_tic_meta)


def load_tic_meta_table_from_file(csv_path="../data/tic_meta.csv"):
    # force pandas to treat the number-based IDs as int that allow N/A
    # (it'd default to float otherwise)
    df = pd.read_csv(csv_path, dtype={"HIP": "Int64", "GAIA": "Int64", "APASS": "Int64", "SDSS": "Int64"})
    return df


def get_aliases(tic_meta_row):
    aliases = [f"TIC {tic_meta_row['ID']}"]

    def normalize_tyc_id(id):
        # In TIC Catalog, the TYC ids have leading zeros, but in SIMBAD and others, they don't
        id = re.sub(r"-0+(\d+)", r"-\1", id)
        id = re.sub(r"^0+(\d+)", r"-\1", id)
        return id

    def add(colname, prefix=None, transform_func=None):
        if prefix is None:
            prefix = f"{colname} "

        id = tic_meta_row[colname]
        if has_value(id):
            if transform_func is not None:
                id = transform_func(id)
            aliases.append(f"{prefix}{id}")

    add("HIP")
    add("TYC", transform_func=normalize_tyc_id)
    add("UCAC", "UCAC4 ")
    add("TWOMASS", "2MASS J")
    add("ALLWISE", "WISEA ")
    add("GAIA", "Gaia DR2 ")
    add("SDSS", "SDSS DR9 ")
    add("APASS")
    add("KIC")
    return aliases


if __name__ == "__main__":
    get_and_save_tic_meta_of_all()
