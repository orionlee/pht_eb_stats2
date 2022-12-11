import re

from astroquery.vizier import Vizier
from astropy import units as u
import numpy as np
import pandas as pd

from ratelimit import limits, sleep_and_retry

from common import bulk_process, has_value, load_tic_ids_from_file


def _add_convenience_columns(result, tesseb_source_type):
    # add convenience columns to TESS EB results
    # for both primary and secondary eclipses,
    # - epoch in BTJD
    # - duration in hour

    result["Epochp"] = result["BJD0"]

    for m in ["pf", "2g"]:
        result[f"Durationp-{m}"] = result["Per"] * result[f"Wp-{m}"] * 24  # assuming "Per" is in unit day
        result[f"Durationp-{m}"].unit = u.hour

        result[f"Epochs-{m}"] = result["BJD0"] + result["Per"] * (result[f"Phis-{m}"] - result[f"Phip-{m}"])
        result[f"Epochs-{m}"].unit = result["BJD0"].unit

        result[f"Durations-{m}"] = result["Per"] * result[f"Ws-{m}"] * 24  # assuming "Per" is in unit day
        result[f"Durations-{m}"].unit = u.hour

    result["tesseb_source"] = tesseb_source_type


def _get_vizier_tesseb_meta_of_tics(tics, **kwargs):
    # Vizier requires a list of TIC in string (the leading zero is not needed, however)
    tics = [str(t) for t in tics]

    TESS_EB_CATALOG = "J/ApJS/258/16/tess-ebs"
    columns = ["*", "Sectors", "UpDate"]  # UpDate: the "date_modified" column
    result_list = Vizier(catalog=TESS_EB_CATALOG, columns=columns).query_constraints(TIC=tics, **kwargs)
    if len(result_list) < 1:
        return None
    result = result_list[0]  # there is only 1 table in the catalog

    #
    # output tweak
    #

    # somehow the column name for BJD0 is incorrect
    result.rename_column("_tab1_10", "BJD0")

    # convert the TIC in zero-padded string to integers, to be consistent with the tables outside TESSEB
    result["TIC"] = [int(t) for t in result["TIC"]]

    # duplicates from TIC catalog
    result.remove_columns(["RAJ2000", "DEJ2000", "pmRA", "pmDE", "Tmag"])
    # columns for Vizier UI to generate links to Live TESS EB / Simbad. Irrelevant here
    result.remove_columns(["TESSebs", "Simbad"])

    _add_convenience_columns(result, tesseb_source_type=TESS_EB_CATALOG)

    return result


def _do_save_tesseb_meta(out_path, meta_table, csv_mode, csv_header):
    meta_table.to_pandas().to_csv(out_path, index=False, mode=csv_mode, header=csv_header)


def _save_vizier_tesseb_meta(meta_table):
    out_path = "cache/tesseb_meta_from_vizier.csv"
    return _do_save_tesseb_meta(out_path, meta_table, csv_mode="w", csv_header=True)


def _get_and_save_vizier_tesseb_meta_of_all(dry_run=False, dry_run_size=1000):
    ids = load_tic_ids_from_file()

    if dry_run and dry_run_size is not None:
        ids = ids[:dry_run_size]

    res = _get_vizier_tesseb_meta_of_tics(ids)

    if not dry_run:
        _save_vizier_tesseb_meta(res)

    return res


def load_tesseb_meta_table_from_file(csv_path="../data/tesseb_meta.csv"):
    df = pd.read_csv(csv_path, dtype={})
    return df


if __name__ == "__main__":
    _get_and_save_vizier_tesseb_meta_of_all(dry_run=False)
    print("TODO: to complete")
