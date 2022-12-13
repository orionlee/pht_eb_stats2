import warnings

import numpy as np

import pandas as pd

from astropy import units as u
from astropy.utils.exceptions import AstropyWarning

from common import insert, move, to_csv, AbstractTypeMapAccessor
import xmatch_util


def xmatch_and_save_gaia_dr3_meta_of_all_by_tics(dry_run=False, dry_run_size=1000):
    out_path = "cache/gaia_dr3_tics_xmatch.csv"
    with warnings.catch_warnings():
        # Ignore astropy table's warning above Gaia's ID (Source) and SolID
        # They get converted to string and saved in csv
        # we will properly handle it in the reader
        warnings.filterwarnings(
            "ignore", category=AstropyWarning, message=".*OverflowError converting to IntType in column SolID.*"
        )
        warnings.filterwarnings(
            "ignore", category=AstropyWarning, message=".*OverflowError converting to IntType in column Source.*"
        )
        # OPEN: we might need to increase timeout.
        # Empirically, the run for the ~12K of TICs took 11 minutes
        # it should have timed out (the default is 300 seconds), but it did not.
        # A workaround is to have caller set it with `XMatch.TIMEOUT` directly
        return xmatch_util.xmatch_and_save_vizier_meta_of_all_by_tics(
            "I/355/gaiadr3",
            out_path=out_path,
            # TIC's coordinate are mostly based on Gaia DR2, so they should match Gaia DR3 quite well
            # 15 arcsec seems to have sufficient coverage based on a 1000 sample test
            max_distance=15 * u.arcsec,
            dry_run=dry_run,
            dry_run_size=dry_run_size,
        )


def _load_gaia_dr3_xmatch_table_from_file(csv_path="cache/gaia_dr3_tics_xmatch.csv"):
    df = pd.read_csv(
        csv_path,
        dtype={
            # force large integer IDs to use non-nullable int64
            # use non-nullable because they are required ids in Gaia
            "Source": "int64",
            "SolID": "int64",
        },
        keep_default_na=True,
    )
    return df
