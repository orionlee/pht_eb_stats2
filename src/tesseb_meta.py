from pathlib import Path
import re

from astroquery.vizier import Vizier
from astropy import units as u
import numpy as np
import pandas as pd


# BEGIN for live TESS EB access
import requests
from bs4 import BeautifulSoup
from astropy.table import Table
# END for live TESS EB access

from memoization import cached

from ratelimit import limits, sleep_and_retry

from common import bulk_process, has_value, to_csv, load_tic_ids_from_file
import tic_pht_stats


def _add_convenience_columns(result):
    # add convenience columns to TESS EB results
    # for both primary and secondary eclipses,
    # - epoch in BTJD
    # - duration in hour

    result["Epochp"] = result["BJD0"]

    for m in ["pf", "2g"]:
        result[f"Durationp-{m}"] = result["Per"] * result[f"Wp-{m}"] * 24  # assuming "Per" is in unit day
        result[f"Epochs-{m}"] = result["BJD0"] + result["Per"] * (result[f"Phis-{m}"] - result[f"Phip-{m}"])
        result[f"Durations-{m}"] = result["Per"] * result[f"Ws-{m}"] * 24  # assuming "Per" is in unit day


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

    result["tesseb_source"] = TESS_EB_CATALOG

    return result


def _do_save_tesseb_meta(out_path, meta_table, csv_mode, fieldnames=None):
    return to_csv(meta_table, out_path, mode=csv_mode, fieldnames=fieldnames)


def _save_vizier_tesseb_meta(meta_table):
    out_path = "cache/tesseb_meta_from_vizier.csv"
    return _do_save_tesseb_meta(out_path, meta_table, csv_mode="w")


def _get_and_save_vizier_tesseb_meta_of_all(dry_run=False, dry_run_size=1000):
    ids = load_tic_ids_from_file()

    if dry_run and dry_run_size is not None:
        ids = ids[:dry_run_size]

    res = _get_vizier_tesseb_meta_of_tics(ids)

    if not dry_run:
        _save_vizier_tesseb_meta(res)

    return res


# throttle HTTP calls to Live TESS EB
NUM_LIVE_TESS_EB_CALLS = 1
TWO_SECONDS = 2
FIVE_SECONDS = 5

@sleep_and_retry
@limits(calls=NUM_LIVE_TESS_EB_CALLS, period=FIVE_SECONDS)
def _do_get_live_tesseb_meta_html_of_tic(tic):
    def get_live_tess_eb_url_of_tic(tic_padded):
        tic_padded = str(tic).zfill(10)  # the canonical TIC in TESS EB is zero-padded to 10 digits
        url = f"http://tessebs.villanova.edu/{tic_padded}"
        return url

    url = get_live_tess_eb_url_of_tic(tic)
    r = requests.get(url)

    # if a TIC is not in the DB, the site returns HTTP 500 with IndexError. The text:
    # f"<h1>IndexError at /{tic_padded}</h1>" (with some \s, \n in the spaces)
    # we do not treat this as an actual exception
    if r.status_code == 500 and "<h1>IndexError" in r.text:
        return r.text

    r.raise_for_status()

    return r.text


def _get_live_tesseb_meta_html_of_tic(tic, cache=True):
    local_path = Path(f"cache/tesseb/l{tic}.html")
    # case cache hit
    if cache and local_path.is_file():
        # print(f"cache hit. local_path: {local_path}")
        return local_path.read_text(encoding="utf-8")

    # case cache miss
    html = _do_get_live_tesseb_meta_html_of_tic(tic)

    if cache:
        local_path.write_text(html, encoding="utf-8")

    return html


def _get_live_tesseb_meta_of_tic(tic, also_return_soap=False):
    def none_result():
        if also_return_soap:
            return None, None
        else:
            return None

    html = _get_live_tesseb_meta_html_of_tic(tic)

    if "<h1>IndexError" in html:
        return none_result()

    # parse the HTML
    soup = BeautifulSoup(html, 'html.parser')

    def extract(table_idx, col_idx, col_header):
        """Helper to extract a cell from the HTML tables"""
        col_header_actual = soup.select_one(f"body > table:nth-of-type({table_idx}) th:nth-of-type({col_idx}) > div ").text
        # the text above contains cooltip too, In catalog?<span class="tooltiptext">Is this signal in the EB catalog?</span>
        # so the match by startswith
        if not col_header_actual.startswith(col_header):
            raise Exception(f"Extraction failed for {col_header}. The column is actually {col_header_actual} for {tic}")
        cell_el = soup.select_one(f"body > table:nth-of-type({table_idx}) tr:nth-of-type(1) > td:nth-of-type({col_idx})")
        if cell_el is not None:
            cell_val = cell_el.text.strip()
        else:
            # case the cell is not found at all, it happens for some TIC (table rows are empty)
            cell_val = None
        return cell_val

    def safe_float(val):
        if val is None or val == '':
            return np.ma.masked
        return float(val)

    in_catalog_val = extract(2, 2, "In catalog?")
    # cases that there is an entry, but is not really in the catalog
    # 1. in_catalog cell is False, e.g., 237280189
    # 2. the table has no row at all, e.g., 9054318
    if in_catalog_val is None or "False" == in_catalog_val:
        return none_result()
    if "True" != in_catalog_val:
        raise Exception(f"Unrecognized In Catalog value {in_catalog_val} in TESS DB for TIC {tic}")

    # case the tic is in catalog, scrape the rest of the page
    # the keys are made to be consistent with those from static TESS EB Vizier result

    result = {}
    result["TIC"] = tic  # use the int rather than the non 0-padded string, to be consistent with vizier code path

    result["m_TIC"] = int(extract(2, 1, "Signal"))
    result["BJD0"] = safe_float(extract(2, 3, "t0 [days]"))
    result["e_BJD0"] = safe_float(extract(2, 4, "σt0 [days]"))
    result["Per"] = safe_float(extract(2, 5, "P [days]"))
    result["e_Per"] = safe_float(extract(2, 6, "σP [days]"))
    result["Morph"] = safe_float(extract(2, 7, "Morphology"))

    result["Wp-pf"] = safe_float(extract(3, 1, "wp,pf"))
    result["Dp-pf"] = safe_float(extract(3, 3, "dp,pf"))
    result["Phip-pf"] = safe_float(extract(3, 5, "φp,pf"))
    result["Ws-pf"] = safe_float(extract(3, 2, "ws,pf"))
    result["Ds-pf"] = safe_float(extract(3, 4, "ds,pf"))
    result["Phis-pf"] = safe_float(extract(3, 6, "φs,pf"))

    result["Wp-2g"] = safe_float(extract(3, 6 + 1, "wp,2g"))
    result["Dp-2g"] = safe_float(extract(3, 6 + 3, "dp,2g"))
    result["Phip-2g"] = safe_float(extract(3, 6 + 5, "φp,2g"))
    result["Ws-2g"] = safe_float(extract(3, 6 + 2, "ws,2g"))
    result["Ds-2g"] = safe_float(extract(3, 6 + 4, "ds,2g"))
    result["Phis-2g"] = safe_float(extract(3, 6 + 6, "φs,2g"))

    result["Sectors"] = str(extract(1, 11, "Sectors"))
    result["UpDate"] = str(extract(2, 8, "Last modified"))
    # TODO: the datetime from Vizier is a string ISO format. Consider to convert it.
    # the datetime text in the cell cannot be parsed by datetime.strptime() easily, e.g., for
    # "Sept. 16, 2021, 5:37 p.m."
    # strptime() can only work in a slightly different format:
    # "Sep. 16, 2021, 05:37 pm"
    # res = datetime.strptime("Sep. 16, 2021, 05:37 pm", "%b. %d, %Y, %I:%M %p")

    result["tesseb_source"] = "live"

    # the table is on par with the the one from Vizier
    result = Table(rows=[result])

    if also_return_soap:
        return result, soup  # return parsed HTML for debug
    else:
        return result



@cached
def _get_tesseb_meta_header_names(dummy_arg=True):
    # The dummy_arg is used to fool memoization

    # The use case for this function is to get
    # headers for live TESS EB code path ( _get_and_save_live_tesseb_meta_of_tic() below)
    #
    # Live TESS EB path is run only after Vizier path.
    # so we assume Vizier CSV is there

    # we could add nrows=2 to pd.read_csv() call to further save time/memory
    # since the result is cached, I don't bother the complication for now.
    df = load_tesseb_meta_table_from_file(csv_path="cache/tesseb_meta_from_vizier.csv")
    return list(df.columns)


def _get_and_save_live_tesseb_meta_of_tic(tic, is_append=True):
    out_path = "cache/tesseb_meta_from_live.csv"

    res = _get_live_tesseb_meta_of_tic(tic)
    fieldnames = None

    if res is None:
        # if the tic is not found in live TESS EB,
        # still write a row for the given TIC
        #
        # include fieldnames for underlying csv writer,
        # for the edge case that csv file is empty
        # (i.e., no header yet)
        # fieldnames will be used to supply the header
        res = {"TIC": tic}
        fieldnames = _get_tesseb_meta_header_names()

    if is_append:
        csv_mode = "a"
    else:
        csv_mode = "w"

    _do_save_tesseb_meta(out_path, res, csv_mode, fieldnames=fieldnames)

    return res


def _get_remaining_tics_to_send_to_live_tesseb():
    def get_max_sector(sectors_str):
        sectors = sectors_str.split(",")
        sectors = [int(s) for s in sectors]
        return max(sectors)

    # the file has the all the TIC ids,
    # but sectors with PHT eb tagging, that will be used for filtering here
    df = tic_pht_stats.load_tic_pht_stats_table_from_file()
    df = df[["tic_id", "sectors"]]
    df["max_sector"] = [get_max_sector(s) for s in df["sectors"]]

    # remove those TIC max_sector <= 26 ,
    # because static TESS EB would have covered them anyway, if it is there
    # (static TESS EB covers sectors 1 - 26)
    df = df[~(df["max_sector"] <= 26)]

    # remove those already found in static TESS EB
    # note: this code path requires the csv file from static TESS EB has been created.
    df_from_vizier = load_tesseb_meta_table_from_file("cache/tesseb_meta_from_vizier.csv")
    df = df[~(df["tic_id"].isin(df_from_vizier["TIC"]))]

    # remove those already fetched from live TESS EB
    try:
        df_from_live = load_tesseb_meta_table_from_file("cache/tesseb_meta_from_live.csv")
        df = df[~(df["tic_id"].isin(df_from_live["TIC"]))]
    except FileNotFoundError:
        # no-op if none from live TESS EB has been fetched yet.
        pass

    df.reset_index(drop=True, inplace=True)

    return df


def _get_and_save_live_tesseb_meta_of_remaining(subset_slice=None):
    ids = _get_remaining_tics_to_send_to_live_tesseb()["tic_id"].to_numpy()
    print("Num. of TICs remaining:", len(ids))
    if subset_slice is not None:
        ids = ids[subset_slice]
        print("Process a subset of ", subset_slice, " , num. of tics:", len(ids))

    # Note: the logic here appends to the existing output csv file
    # if you want to start from scratch, you need to delete first delete the csv
    #
    kwargs_list = [dict(tic=id) for id in ids]
    return bulk_process(_get_and_save_live_tesseb_meta_of_tic, kwargs_list)


def combine_and_save_tesseb_meta_from_vizier_and_live():
    out_path = "../data/tesseb_meta.csv"

    df_from_vizier = load_tesseb_meta_table_from_file("cache/tesseb_meta_from_vizier.csv")
    df_from_live = load_tesseb_meta_table_from_file("cache/tesseb_meta_from_live.csv")
    df_from_live = df_from_live[~(pd.isna(df_from_live["BJD0"]))]  # remove rows that indicate no match in live TESS EB
    # OPEN: convert the updated time in live TESS EB from English-like string to ISO format that is used in Vizier.

    df = pd.concat([df_from_vizier, df_from_live])
    df = df.sort_values("TIC", ascending=True)
    df.reset_index(drop=True, inplace=True)

    to_csv(df, out_path, mode="w")

    return df


def load_tesseb_meta_table_from_file(csv_path="../data/tesseb_meta.csv", add_convenience_columns=False):
    def keep_empty_str(in_val):
        """Force pandas to treat empty string as is (rather than the default NaN) when reading csv."""
        if in_val == "":
            return ""
        else:
            return in_val

    df = pd.read_csv(
            csv_path,
            converters={"Sectors": keep_empty_str, "UpDate": keep_empty_str, "tesseb_source": keep_empty_str},
        )

    if add_convenience_columns:
        _add_convenience_columns(df)

    return df


if __name__ == "__main__":
    _get_and_save_vizier_tesseb_meta_of_all(dry_run=False)
    _get_and_save_live_tesseb_meta_of_remaining()
    combine_and_save_tesseb_meta_from_vizier_and_live()
