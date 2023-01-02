# A standalone script to batch prefetch products (LCs, TPFs, DVs) used in vetting

import asyncio
from datetime import datetime
import sys

from astropy import units as u
import numpy as np

import lightkurve as lk

import asyncio_compat
import lightkurve_ext as lke
import lightkurve_ext_tess as lket


# lightkurve config
lk_download_dir = "data"

if hasattr(lk.search, "sr_cache"):   # to support PR for persistent query result cache
    lk.search.sr_cache.cache_dir = lk_download_dir
    lk.search.sr_cache.expire_second = 7 * 86400


def info(msg):
    ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{ts_str} \t{msg}", flush=True)


async def prefetch_tics(tics, **kwargs):
    info(f"BEGIN prefetch_products() for {len(tics)}...")
    for tic in tics:
        await prefetch_one_tic(tic, **kwargs)
    info("END  prefetch_products()")
    return


def search_lightcurves_of_tic_with_priority(tic, author_priority=["SPOC", "QLP", "TESS-SPOC"], download_filter_func=None):
    sr_unfiltered = lk.search_lightcurve(f"TIC{tic}", mission="TESS")
    if len(sr_unfiltered) < 1:
        print(f"WARNING: no result found for TIC {tic}")
        return

    sr_unfiltered = sr_unfiltered[sr_unfiltered.target_name == str(tic)]  # in case we get some other nearby TICs

    # filter out HLSPs not supported by lightkurve yet
    sr = sr_unfiltered[sr_unfiltered.author != "DIAMANTE"]
    if len(sr) < len(sr_unfiltered):
        print("Note: there are products not supported by Lightkurve, which are excluded from download.")

    # for each sector, filter based on the given priority.
    # - note: by default, prefer QLP over TESS-SPOC because QLP is detrended, with multiple apertures within 1 file
    sr = lke.filter_by_priority(
        sr,
        author_priority=author_priority,
        exptime_priority=["short", "long", "fast"],
    )

    if download_filter_func is not None:
        sr = download_filter_func(sr)

    return sr, sr_unfiltered


def download_lightcurves(sr, download_dir):
    return sr.download_all(download_dir=download_dir)


async def prefetch_one_tic(tic, download_dir=lk_download_dir):
    info(f"Prefetching TIC {tic}...")

    # emulate "Enter TIC" cell logic

    def limit_sr_to_download(sr):
        return sr  # get all available

    # BEGIN emulate download_lightcurves_of_tic_with_priority(), but breaks it down
    # to make parallel download of LCs, TPFs, and DVs, possible
    sr, sr_unfiltered = search_lightcurves_of_tic_with_priority(
        tic,
        download_filter_func=limit_sr_to_download,
        author_priority=["SPOC", "TESS-SPOC", "QLP"],  # prefer TESS-SPOC over QLP
    )

    # sector to download TPF: use the first sector with 2 minute cadence data
    # Reason: if the TIC has TCEs, the epoch is generally from the first 2 minute cadence sector.
    sector = sr[sr.exptime == 120 * u.s].table[0]["sequence_number"]
    tpf_task = lke.create_download_tpf_task(
        f"TIC{tic}", sector=sector, exptime="short", author="SPOC", mission="TESS", download_dir=download_dir
    )

    tce_task = asyncio_compat.create_background_task(lket.get_tce_infos_of_tic, tic, download_dir=download_dir)

    # download LCs in the background has intermittent error
    #   ValueError: I/O operation on closed file.
    # apparently from some of the messages about lcf_coll, possibly the Warning line
    # "Warning: 30% (5871/19412) of the cadences will be ignored due to the quality mask..."
    # (OPEN: Consider to see if redirecting stdout/stderr could help)
    #
    # For now we run it in the foreground,
    # but after tpf and tce background tasks have been spawned.
    # lc_task = asyncio_compat.create_background_task(download_lightcurves, sr, download_dir=download_dir)
    # lcf_coll = await lc_task
    lcf_coll = download_lightcurves(sr, download_dir=download_dir)  # download in foreground

    tpf_coll, sr_tpf = await tpf_task
    tce_res = await tce_task

    result_msg = f"Downloaded - num. LCs: {len(lcf_coll)}, num. TPFs: {len(tpf_coll)}, num. TCEs: {len(tce_res)}"

    info(f"result: {result_msg}\n")
    return


#
# The main logic
#

def get_tics_from_file(tic_list_filename):
    return np.genfromtxt(tic_list_filename, dtype=int)


# Usage:   python prefetch_products.py <tic_list_filename>
# Example: python prefetch_products.py tics_prefetch_list.txt
if __name__ == "__main__":
    tic_list_filename = sys.argv[1]
    tics = get_tics_from_file(tic_list_filename)
    asyncio.run(
        prefetch_tics(tics)
    )
