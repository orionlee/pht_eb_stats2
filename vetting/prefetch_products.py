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


def info(msg):
    ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{ts_str} \t{msg}", flush=True)


async def prefetch_tics(tics, **kwargs):
    info(f"BEGIN prefetch_products() for {len(tics)}...")
    for tic in tics:
        await prefetch_one_tic(tic, **kwargs)
    info("END  prefetch_products()")
    return


async def prefetch_one_tic(tic, download_dir=lk_download_dir):
    info(f"Prefetching TIC {tic}...")

    # emulate "Enter TIC" cell logic

    def limit_sr_to_download(sr):
        return sr  # get all available

    # OPEN: redirect stdout and IPython.display.display to null
    # Problems:
    # - IPython display: don't know how to do it.
    # - stdout: using "contextlib.redirect_stdout(open(os.devnull, "w")):" would work

    lcf_coll, sr, sr_unfiltered = lke.download_lightcurves_of_tic_with_priority(
        tic,
        download_filter_func=limit_sr_to_download,
        download_dir=download_dir,
        author_priority=["SPOC", "TESS-SPOC", "QLP"],
    )

    # sector to download TPF: use the first sector with 2 minute cadence data
    # Reason: if the TIC has TCEs, the epoch is generally from the first 2 minute cadence sector.
    sector = sr[sr.exptime == 120 * u.s].table[0]["sequence_number"]
    tpf_task = lke.create_download_tpf_task(
        f"TIC{tic}", sector=sector, exptime="short", author="SPOC", mission="TESS", download_dir=download_dir
    )

    tce_task = asyncio_compat.create_background_task(lket.get_tce_infos_of_tic, tic, download_dir=download_dir)

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
