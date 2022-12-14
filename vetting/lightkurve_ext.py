"""
Convenience helpers for `lightkurve` package.
"""

# so that TransitTimeSpec can be referenced in type annotation in the class itself
# see: https://stackoverflow.com/a/49872353
from __future__ import annotations


import os
import logging
import math
import json
import re
import warnings
from collections import OrderedDict
from types import SimpleNamespace

import astropy
from astropy.io import fits
from astropy import coordinates as coord
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.time import Time
import astropy.units as u

import numpy as np
from scipy.interpolate import UnivariateSpline

from IPython.display import display, HTML

import lightkurve as lk

import asyncio_compat

log = logging.getLogger(__name__)


def of_lcs(lc_coll, filter_func):
    """Filter a LightCurveCollection using the given filter_func.

    Example
    --------

    Only retain TESS SPOC 2 minute cadence lightcurves
    > of_lcs(lc_coll, lambda lc: lc.author == "SPOC")
    """
    return type(lc_coll)([lc for lc in lc_coll if filter_func(lc)])


def of_sector(lcf_coll, sectorNum):
    res_list = of_sectors(lcf_coll, sectorNum)
    return res_list[0] if len(res_list) > 0 else None


def of_sectors(*args):
    lk_coll_or_sr = args[0]
    if len(args) == 1:
        # when no sectors are specified, return entire collection
        # For convenience: when a notebooks is modified such that
        # a user sometimes use a subset of sectors , and sometimes everything
        # the user can can still use of_sectors() wrapper regardless
        return lk_coll_or_sr
    sector_nums = args[1:]

    if hasattr(lk_coll_or_sr, "sector"):
        return lk_coll_or_sr[np.in1d(lk_coll_or_sr.sector, sector_nums)]
    elif hasattr(lk_coll_or_sr, "table") and lk_coll_or_sr.table["sequence_number"] is not None:
        return lk.SearchResult(lk_coll_or_sr.table[np.in1d(lk_coll_or_sr.table["sequence_number"], sector_nums)])
    else:
        raise TypeError(f"Unsupported type of collection: {type(lk_coll_or_sr)}")


def of_sector_n_around(lk_coll_or_sr, sector_num, num_additions=8):
    def get_sector_for_lk_coll(lk_coll):
        return lk_coll.sector

    def get_sector_for_sr(sr):
        return sr.table["sequence_number"]

    sector_accessor_func = None
    if hasattr(lk_coll_or_sr, "sector"):
        sector_accessor_func = get_sector_for_lk_coll
    elif hasattr(lk_coll_or_sr, "table") and lk_coll_or_sr.table["sequence_number"] is not None:
        sector_accessor_func = get_sector_for_sr
    else:
        raise TypeError(f"Unsupported type of collection: {type(lk_coll_or_sr)}")

    subset_slice = _get_slice_for_of_sector_n_around(
        lk_coll_or_sr,
        sector_accessor_func,
        sector_num,
        num_additions=num_additions,
    )
    if subset_slice is not None:
        return lk_coll_or_sr[subset_slice]
    else:
        return type(lk_coll_or_sr)([])


def _get_slice_for_of_sector_n_around(coll, sector_accessor_func, sector_num, num_additions):
    if sector_num not in sector_accessor_func(coll):
        return None

    idx = np.where(sector_accessor_func(coll) == sector_num)[0][0]
    # if num_additions is odd number, we add one to older sector
    start = max(idx - math.ceil(num_additions / 2), 0)
    end = min(idx + math.floor(num_additions / 2) + 1, len(coll))

    # case the start:end slice does not fill up the requested num_additions,
    # try to fill it up
    cur_slice_size = end - start - 1
    if cur_slice_size < num_additions:
        num_more_needed = num_additions - cur_slice_size
        if start > 0:
            start = max(start - num_more_needed, 0)
        else:
            end = min(end + num_more_needed, len(coll))

    return slice(start, end)


def of_2min_cadences(lcf_coll):
    """Return LightCurveFiles of short, typically 2-minute cadence, only.
    Primary use case is to filter out 20-second files.
    """
    filtered = [lcf for lcf in lcf_coll if "short" == estimate_cadence_type(lcf)]
    return lk.LightCurveCollection(filtered)


def estimate_cadence(lc, unit=None, round_unit_result=True):
    """Estimate the cadence of a lightcurve by returning the median of a sample"""
    res = np.nanmedian(np.diff(lc.time[:100].value))
    if unit is not None:
        res = (res * u.day).to(unit)  # LATER: handle cases lc.time is not in days
        if round_unit_result:
            res = res.round()
    return res


def map_cadence_type(cadence_in_days):
    long_minimum = 9.9 / 60 / 24  # 10 minutes in days, with some margin of error
    short_minimum = 0.9 / 60 / 24  # 1 minute in days, with some margin of error
    if cadence_in_days is None:
        return None
    if cadence_in_days >= long_minimum:
        return "long"
    if cadence_in_days >= short_minimum:
        return "short"
    return "fast"


def estimate_cadence_type(lc):
    """Estimate the type of cadence to be one of long, short, or fast.
    The definition is the same as ``exptime`` in `lightkurve.search_lightcurve()`.
    """
    return map_cadence_type(estimate_cadence(lc))


def of_tic(lcf_coll, tic):
    """Return LightCurveFiles of the given TIC.

    Useful in case the default MAST result returned nearby targets.
    """
    filtered = [lcf for lcf in lcf_coll if lcf.meta.get("TICID", None) == tic]
    return lk.LightCurveCollection(filtered)


def select(lcf_coll_or_sr, filter_func):
    """Filter the given LightCurveCollection or SearchResult with the filter_func."""
    return type(lcf_coll_or_sr)([obj for obj in lcf_coll_or_sr if filter_func(obj)])


def exclude_range(lc, start, end):
    """Exclude the specified range of time from the given lightcurve."""
    tmask = (lc.time.value >= start) & (lc.time.value < end)
    return lc[~tmask]


def get_obs_date_range(lcf_coll):
    """Return the observation date span and the number of days with observation."""
    # the code assumes the time are in all in BTJD, or other consistent format in days
    if isinstance(lcf_coll, lk.LightCurve):
        lcf_coll = lk.LightCurveCollection([lcf_coll])

    # to support folded lightcurve
    time_colname = "time_original" if "time_original" in lcf_coll[0].colnames else "time"

    t_start = lcf_coll[0][time_colname].min().value
    t_end = lcf_coll[-1][time_colname].max().value

    obs_span = t_end - t_start
    obs_actual = len(set(np.concatenate([lc[time_colname].value.astype("int") for lc in lcf_coll])))

    return obs_span, obs_actual


def estimate_object_radius_in_r_jupiter(lc, depth):
    """Return a back of envelope estimate of a companion object's radius."""
    R_JUPITER_IN_R_SUN = 71492 / 695700

    r_star = lc.meta.get("RADIUS")  # assumed to be in R_sun
    if r_star is None or r_star < 0 or depth <= 0:
        return None  # cannot estimate
    r_obj = math.sqrt(r_star * r_star * depth)
    r_obj_in_r_jupiter = r_obj / R_JUPITER_IN_R_SUN
    return r_obj_in_r_jupiter


def download_lightcurves_of_tic_with_priority(
    tic, author_priority=["SPOC", "QLP", "TESS-SPOC"], download_filter_func=None, download_dir=None
):
    """For a given TIC, download lightcurves across all sectors.
    For each sector, download one based on pre-set priority.
    """

    sr_unfiltered = lk.search_lightcurve(f"TIC{tic}", mission="TESS")
    if len(sr_unfiltered) < 1:
        print(f"WARNING: no result found for TIC {tic}")
        return None, None, None

    sr_unfiltered = sr_unfiltered[sr_unfiltered.target_name == str(tic)]  # in case we get some other nearby TICs

    # filter out HLSPs not supported by lightkurve yet
    sr = sr_unfiltered[sr_unfiltered.author != "DIAMANTE"]
    if len(sr) < len(sr_unfiltered):
        print("Note: there are products not supported by Lightkurve, which are excluded from download.")

    # for each sector, filter based on the given priority.
    # - note: prefer QLP over TESS-SPOC because QLP is detrended, with multiple apertures within 1 file
    sr = filter_by_priority(
        sr,
        author_priority=author_priority,
        exptime_priority=["short", "long", "fast"],
    )
    num_filtered = len(sr_unfiltered) - len(sr)
    num_fast = len(sr_unfiltered[sr_unfiltered.exptime < 60 * u.second])
    if num_filtered > 0:
        msg = f"{num_filtered} rows filtered"
        if num_fast > 0:
            msg = msg + f" ; {num_fast} fast (20secs) products."
        print(msg)

    display(sr)

    # let caller to optionally further restrict a subset to be downloaded
    sr_to_download = sr
    if download_filter_func is not None:
        sr_to_download = download_filter_func(sr)
        if len(sr_to_download) < len(sr):
            display(
                HTML(
                    """<font style="background-color: yellow;">Note</font>:
SearchResult is further filtered - only a subset will be downloaded."""
                )
            )

    lcf_coll = sr_to_download.download_all(download_dir=download_dir)

    if lcf_coll is not None and len(lcf_coll) > 0:
        print(f"TIC {tic} \t#sectors: {len(lcf_coll)} ; {lcf_coll[0].meta['SECTOR']} - {lcf_coll[-1].meta['SECTOR']}")
        print(
            (
                f"   sector {lcf_coll[-1].meta['SECTOR']}: \t"
                f"camera = {lcf_coll[-1].meta['CAMERA']} ; ccd = {lcf_coll[-1].meta['CCD']}"
            )
        )
    else:
        print(f"TIC {tic}: no data")

    return lcf_coll, sr, sr_unfiltered


def download_lightcurve(
    target,
    mission=("Kepler", "K2", "TESS"),
    exptime="short",
    author="SPOC",
    download_dir=None,
    use_cache="yes",
    display_search_result=True,
):
    """
    Wraps `lightkurve.search_lightcurve()` and the
    subsequent `lightkurve.search.SearchResult.download_all()` calls,
    with the option of caching, so that for a given search,
    if the the result has been downloaded, the cache will be used.

    The parameters all propagate to the underlying `search_lightcurvefile()`
    and `download_all()` calls. The lone exception is `use_cache`.

    Parameters
    ----------
    use_cache : str, must be one of 'yes', or 'no'\n
        OPEN: an option of 'fallback': cache will be used when offline.\n
        OPEN: for now, actual download lightcurve cache will still be used if
        available irrespective of the settings.

    Returns
    -------
    collection : `~lightkurve.collections.Collection` object
        Returns a `~lightkurve.collections.LightCurveCollection`
        containing all lightcurve files that match the criteria
    """

    if use_cache == "no":
        return _search_and_cache(target, mission, exptime, author, download_dir, display_search_result)
    if use_cache == "yes":
        result_file_ids = _load_from_cache_if_any(target, mission, download_dir)
        if result_file_ids is not None:
            result_files = list(map(lambda e: f"{download_dir}/mastDownload/{e}", result_file_ids))
            return lk.collections.LightCurveCollection(list(map(lambda f: lk.read(f), result_files)))
        # else
        return _search_and_cache(target, mission, exptime, author, download_dir, display_search_result)
    # else
    raise ValueError("invalid value for argument use_cache")


# Private helpers for `download_lightcurvefiles`


def _search_and_cache(target, mission, exptime, author, download_dir, display_search_result):
    search_res = lk.search_lightcurve(target=target, mission=mission, exptime=exptime, author=author)
    if len(search_res) < 1:
        return None
    if display_search_result:
        _display_search_result(search_res)
    _cache_search_result_product_identifiers(search_res, download_dir, target, mission)
    return search_res.download_all(quality_bitmask="default", download_dir=download_dir)


def _display_search_result(search_res):
    from IPython.core.display import display

    tab = search_res.table
    # move useful columns to the front
    preferred_cols = ["proposal_id", "target_name", "sequence_number", "t_exptime"]
    colnames_reordered = preferred_cols + [c for c in tab.colnames if c not in preferred_cols]
    display(tab[colnames_reordered])


def _load_from_cache_if_any(target, mission, download_dir):
    key = _get_cache_key(target, mission)
    return _load_search_result_product_identifiers(download_dir, key)


def _cache_search_result_product_identifiers(search_res, download_dir, target, mission):
    key = _get_cache_key(target, mission)
    identifiers = _to_product_identifiers(search_res)
    _save_search_result_product_identifiers(identifiers, download_dir, key)
    return key


def _get_search_result_cache_dir(download_dir):
    # TODO: handle download_dir is None (defaults)
    cache_dir = f"{download_dir}/mastQueries"

    if os.path.isdir(cache_dir):
        return cache_dir

    # else it doesn't exist, make a new cache directory
    try:
        os.mkdir(cache_dir)
    # downloads locally if OS error occurs
    except OSError:
        log.warning(
            "Warning: unable to create {}. "
            "Cache MAST query results to the current "
            "working directory instead.".format(cache_dir)
        )
        cache_dir = "."
    return cache_dir


def _get_cache_key(target, mission):
    # TODO: handle cases the generated key is not a valid filename
    return f"{target}_{mission}_ids"


def _to_product_identifiers(search_res):
    """
    Returns
    -------
    A list of str, constructed from `(obs_collection, obs_id, productFilename)` tuples, that can
    identify cached lightcurve file,s if any.
    """
    return list(
        map(
            lambda e: e["obs_collection"] + "/" + e["obs_id"] + "/" + e["productFilename"],
            search_res.table,
        )
    )


def _save_search_result_product_identifiers(identifiers, download_dir, key):
    resolved_cache_dir = _get_search_result_cache_dir(download_dir)
    filepath = f"{resolved_cache_dir}/{key}.json"
    fp = open(filepath, "w+")
    json.dump(identifiers, fp)
    return filepath


def _load_search_result_product_identifiers(download_dir, key):
    resolved_cache_dir = _get_search_result_cache_dir(download_dir)
    filepath = f"{resolved_cache_dir}/{key}.json"
    try:
        fp = open(filepath, "r")
        return json.load(fp)
    except OSError as err:
        # errno == 2: file not found, typical case of cache miss
        # errno != 2: unexpected error, log a warning
        if err.errno != 2:
            log.warning("Unexpected OSError in retrieving cached search result: {}".format(err))
        return None


def filter_by_priority(
    sr,
    author_priority=["SPOC", "TESS-SPOC", "QLP"],
    exptime_priority=["short", "long", "fast"],
):
    author_sort_keys = {}
    for idx, author in enumerate(author_priority):
        author_sort_keys[author] = idx + 1

    exptime_sort_keys = {}
    for idx, exptime in enumerate(exptime_priority):
        exptime_sort_keys[exptime] = idx + 1

    def calc_filter_priority(row):
        # Overall priority key is in the form of <author_key><exptime_key>, e.g., 101
        # - "01" is the exptime_key
        # - the leading "1" is the author_key, given it is the primary one
        author_default = max(dict(author_sort_keys).values()) + 1
        author_key = author_sort_keys.get(row["author"], author_default) * 100

        # secondary priority
        exptime_default = max(dict(exptime_sort_keys).values()) + 1
        exptime_key = exptime_sort_keys.get(map_cadence_type(row["exptime"] / 60 / 60 / 24), exptime_default)
        return author_key + exptime_key

    sr.table["_filter_priority"] = [calc_filter_priority(r) for r in sr.table]

    # A temporary table that sorts the table by the priority
    sorted_t = sr.table.copy()
    sorted_t.sort(["mission", "_filter_priority"])

    # create an empty table for results, with the same set of columns
    res_t = sr.table[np.zeros(len(sr), dtype=bool)].copy()

    # for each mission (e.g., TESS Sector 01), select a row based on specified priority
    # - select the first row given the table has been sorted by priority
    uniq_missions = list(OrderedDict.fromkeys(sorted_t["mission"]))
    for m in uniq_missions:
        mission_t = sorted_t[sorted_t["mission"] == m]
        # OPEN: if for a given mission, the only row available is not listed in the priorities,
        # the logic still add a row to the result.
        # We might want it to be an option specified by the user.
        res_t.add_row(mission_t[0])

    return lk.SearchResult(table=res_t)


# Download TPF asynchronously


def search_and_download_tpf(*args, **kwargs):
    """Search and Download a TPFs.

    All parameters are passed on ``search_targetpixelfile()``,
    with the exception of ``download_dir`` and ``quality_bitmask``,
    which are passed to ``download_all()`
    """

    # extract download_all() parameters
    download_dir = kwargs.pop("download_dir", None)
    quality_bitmask = kwargs.pop("quality_bitmask", None)
    sr = lk.search_targetpixelfile(*args, **kwargs)  # pass the rest of the argument to search_targetpixelfile
    tpf_coll = sr.download_all(download_dir=download_dir, quality_bitmask=quality_bitmask)
    return tpf_coll, sr


def create_download_tpf_task(*args, **kwargs):
    return asyncio_compat.create_background_task(search_and_download_tpf, *args, **kwargs)


#
# Other misc. extensions
#
def get_bkg_lightcurve(lcf):
    """Returns the background flux, i.e., ``SAP_BKG`` in the file"""
    lc = lcf.copy()
    lc["flux"] = lc["sap_bkg"]
    lc["flux_err"] = lc["sap_bkg_err"]
    lc.label = lc.label + " BKG"
    return lc


def _do_create_quality_issues_mask(quality, flux, flags_included=0b0101001010111111):
    """Returns a boolean array which flags cadences with *issues*.

    The default `flags_included` is a TESS default, based on
    https://outerspace.stsci.edu/display/TESS/2.0+-+Data+Product+Overview#id-2.0DataProductOverview-Table:CadenceQualityFlags
    """
    if np.issubdtype(quality.dtype, np.integer):
        return np.logical_and(quality & flags_included, np.isfinite(flux))
    else:
        # quality column is not an integer, probably a non-standard product
        return np.zeros_like(quality, dtype=bool)


def create_quality_issues_mask(lc, flags_included=0b0101001010111111):
    """Returns a boolean array which flags cadences with *issues*.

    The default `flags_included` is a TESS default, based on
    https://outerspace.stsci.edu/display/TESS/2.0+-+Data+Product+Overview#id-2.0DataProductOverview-Table:CadenceQualityFlags
    """

    # use sap_flux when available (it may not be there in some HLSP)
    # we prefer sap_flux over pdcsap_flux as
    # pdcsap_flux is more likely to be NaN (due to exclusion by quality flags)
    if "sap_flux" in lc.colnames:
        flux = lc["sap_flux"]
    else:
        flux = lc.flux

    return _do_create_quality_issues_mask(lc.quality, flux)


def _get_n_truncate_fits_data(lc, before, after, return_columns, return_mask=False):
    with fits.open(lc.filename) as hdu:
        time = hdu[1].data["TIME"]
        mask = (time >= before) & (time < after)
        res = dict()
        for col in return_columns:
            res[col] = hdu[1].data[col][mask]
        if return_mask:
            return res, mask
        else:
            return res


def list_times_w_quality_issues(lc, include_excluded_cadences=False):
    if not include_excluded_cadences:
        mask = create_quality_issues_mask(lc)
        return lc.time[mask], lc.quality[mask]
    else:
        # case we want cadences that have been excluded in the lc object
        # use the underlying fits file

        flux_colname = lc.meta.get("FLUX_ORIGIN", "sap_flux")
        if flux_colname == "pdcsap_flux":
            flux_colname = "sap_flux"  # pdcsap_flux would not have values in excluded cadences, defeating the purpose

        # data is truncated if the given lc is truncated
        # TODO: cadences excluded before lc.time.min() or after lc.time.max() will still be missing.
        data = _get_n_truncate_fits_data(lc, lc.time.min().value, lc.time.max().value, ["time", "quality", flux_colname])
        mask = _do_create_quality_issues_mask(data["quality"], data[flux_colname])
        return data["time"][mask], data["quality"][mask]


def list_transit_times(t0, period, steps_or_num_transits=range(0, 10), return_string=False):
    """List the transit times based on the supplied transit parameters"""
    if isinstance(steps_or_num_transits, int):
        steps = range(0, steps_or_num_transits)
    else:
        steps = steps_or_num_transits
    times = [t0 + period * i for i in steps]
    if return_string:
        return ",".join(map(str, times))
    else:
        return times


def get_segment_times_idx(times, break_tolerance=5):
    """Segment the input array of times into segments due to data gaps. Return the indices of the segments.

    The minimal gap size is determined by `break_tolerance`.

    The logic is adapted from `LightCurve.flatten`
    """
    if hasattr(times, "value"):  # convert astropy Time to raw values if needed
        times = times.value
    dt = times[1:] - times[0:-1]
    with warnings.catch_warnings():  # Ignore warnings due to NaNs
        warnings.simplefilter("ignore", RuntimeWarning)
        cut = np.where(dt > break_tolerance * np.nanmedian(dt))[0] + 1
    low = np.append([0], cut)
    high = np.append(cut, len(times))
    return (low, high)


def get_segment_times(times, **kwargs):
    if hasattr(times, "value"):  # convert astropy Time to raw values if needed
        times = times.value
    low, high = get_segment_times_idx(times, **kwargs)
    # add a small 1e-10 to end so that the end time is exclusive (follow convention in range)
    return [(times[lo], times[hi - 1] + 1e-10) for lo, hi in zip(low, high)]


def get_transit_times_in_range(t0, period, start, end):
    t_start = t0 + math.ceil((start - t0) / period) * period
    num_t = math.ceil((end - t_start) / period)
    return [t_start + period * i for i in range(num_t)]


def get_transit_times_in_lc(lc, t0, period, return_string=False, **kwargs):
    """Get the transit times with observations of the given lightcurve, based on the supplied transit parameters.

    The method will exclude the times where there is no observation due to data gaps.
    """

    lc = lc.remove_nans()  # exclude cadences with no flux.
    # break up the times to exclude times in gap
    times_list = get_segment_times(lc.time, **kwargs)
    transit_times = []
    for start, end in times_list:
        transit_times.extend(get_transit_times_in_range(t0, period, start, end))
    if return_string:
        return ",".join(map(str, transit_times))
    else:
        return transit_times


class TransitTimeSpec(dict):
    def __init__(
        self,
        epoch: float = None,
        period: float = None,
        duration_hr: float = None,
        sector: int = None,
        steps_to_show: list = None,
        surround_time: float = None,
        label: str = None,
        defaults: TransitTimeSpec = None,
    ):
        # core parameters
        self["epoch"] = epoch
        self["period"] = period
        self["duration_hr"] = duration_hr

        # used for plotting
        self["sector"] = sector
        self["steps_to_show"] = steps_to_show
        self["surround_time"] = surround_time
        self["label"] = label

        if defaults is None:
            defaults = {}
        self._defaults = defaults  # put it as a custom attribute

    def __getitem__(self, key):
        res = super().get(key)
        if res is None:
            res = self._defaults.get(key)
        return res

    def get(self, key, default=None):
        res = self.__getitem__(key)
        if res is None:
            res = default
        return res


class TransitTimeSpecList(list):
    def __init__(self, *tt_spec_dict_list, defaults={}):
        self._defaults = TransitTimeSpec(**defaults)
        for tt_spec_dict in tt_spec_dict_list:
            self.append(TransitTimeSpec(**tt_spec_dict, defaults=self._defaults))

    def _spec_property_values(self, property_name):
        return np.array([tt[property_name] for tt in self])

    #
    # The following properties return the specific transit parameters
    # in an array. Together they can be used to create a mask
    # for the transits using ``LightCurve.create_transit_mask()``
    #

    @property
    def epoch(self):
        return self._spec_property_values("epoch")

    @property
    def period(self):
        return self._spec_property_values("period")

    @property
    def duration_hr(self):
        return self._spec_property_values("duration_hr")

    @property
    def duration(self):
        return self.duration_hr / 24

    @property
    def label(self):
        return self._spec_property_values("label")

    def to_table(self, columns=("label", "epoch", "duration_hr", "period")):
        """Convert the specs to an ``astropy.Table``"""
        data = [getattr(self, col) for col in columns]
        return Table(data, names=columns)


def stitch(lcf_coll, ignore_incompatible_column_warning=False, **kwargs):
    """Wrapper over native stitch(), and tweak the metadata so that it behaves like a typical single-sector lightcurve."""

    def update_meta_if_exists_in(lc_src, keys):
        for key in keys:
            val = lc_src.meta.get(key, None)
            if val is not None:
                lc_stitched.meta[key] = val

    def safe_del_meta(key):
        if lc_stitched.meta.get(key, None) is not None:
            del lc_stitched.meta[key]

    if ignore_incompatible_column_warning:
        with warnings.catch_warnings():
            # suppress useless warning. Use cases: stitching QLP lightcurves with SPOC lightcurves (sap_flux is incompatible)
            warnings.filterwarnings(
                "ignore",
                category=lk.LightkurveWarning,
                message="The following columns will be excluded from stitching because the column types are incompatible:.*",
            )
            lc_stitched = lcf_coll.stitch(**kwargs)
    else:
        lc_stitched = lcf_coll.stitch(**kwargs)

    # now update the metadata

    lc_stitched.meta["STITCHED"] = True

    # update observation start/stop dates
    update_meta_if_exists_in(lcf_coll[0], ("TSTART", "DATE-OBS"))
    update_meta_if_exists_in(lcf_coll[-1], ("TSTOP", "DATE-END"))

    # TODO: recalculate TELAPSE, LIVETIME, DEADC (which is LIVETIME / TELAPSE)

    safe_del_meta("FILENAME")  # don't associate it with a file anymore

    # record the sectors stitched and the associated metadata
    sector_list = [lc.meta.get("SECTOR") for lc in lcf_coll if lc.meta.get("SECTOR") is not None]
    meta_list = [lc.meta for lc in lcf_coll if lc.meta.get("SECTOR") is not None]
    if len(sector_list) > 0:
        lc_stitched.meta["SECTORS"] = sector_list
        meta_dict = dict()
        for sector, meta in zip(sector_list, meta_list):
            meta_dict[sector] = meta.copy()
        lc_stitched.meta["HEADERS_ORIGINAL"] = meta_dict

    return lc_stitched


def to_window_length_for_2min_cadence(length_day):
    """Helper for LightCurve.flatten().
    Return a `window_length` for the given number of days, assuming the data has 2-minute cadence."""
    return to_window_length_for_cadence(length_day * u.day, 2 * u.min)


def to_window_length_for_cadence(length, cadence):
    """Helper for LightCurve.flatten().
    Return a `window_length` for the given length and cadence.

    Parameters length and cadence should be ~~astropy.quantity.Quantity~~.
    If they are unitless number, they should be in the same unit.
    """
    res = math.floor(length / cadence)
    if res % 2 == 0:
        res += 1  # savgol_filter window length must be odd number
    return res


# detrend using spline
# Based on:  https://github.com/barentsen/kepler-athenaeum-tutorial/blob/master/how-to-find-a-planet-tutorial.ipynb
def flatten_with_spline_normalized(lc, return_trend=False, **kwargs):
    lc = lc.remove_nans()
    spline = UnivariateSpline(lc.time, lc.flux, **kwargs)
    trend = spline(lc.time)
    # detrended = lc.flux - trend
    detrended_relative = 100 * ((lc.flux / trend) - 1) + 100  # in percentage
    lc_flattened = lc.copy()
    lc_flattened.flux = detrended_relative
    lc_flattened.flux_unit = "percent"
    if not return_trend:
        return lc_flattened
    else:
        lc_trend = lc.copy()
        lc_trend.flux = trend
        return (lc_flattened, lc_trend)


def _lksl_statistics(ts):
    """Compute LKSL Statistics of the given (time-series) values.
    Useful to compare the noises in a folded lightcurve.

    Based on https://arxiv.org/pdf/1901.00009.pdf , equation 4.
    See https://www.aanda.org/articles/aa/pdf/2002/17/aa2208.pdf for more information.
    """
    ts = ts[~np.isnan(ts)]

    vector_length_sq_sum = np.square(np.diff(ts)).sum()
    if len(ts) > 2:
        # to fully utilize the data by including the vector length between the last and first measurement
        # section 2 of Clarke, 2002
        vector_length_sq_sum += np.square(ts[-1] - ts[0])

    diff_from_mean_sq_sum = np.square(ts - np.mean(ts)).sum()
    if diff_from_mean_sq_sum == 0:  # to avoid boundary case that'd cause division by zero
        diff_from_mean_sq_sum = 1e-10

    return (vector_length_sq_sum / diff_from_mean_sq_sum) * (len(ts) - 1) / (2 * len(ts))


def lksl_statistics(lc, column="flux"):
    return _lksl_statistics(lc[column].value)


#
# TODO: util to estimate transit depth
#


def estimate_snr(
    lc,
    signal_depth,
    signal_duration,
    num_signals,
    savgol_to_transit_window_ratio=4,
    cdpp_kwargs=None,
    return_diagnostics=False,
):
    """Estimate Signal-to-Noise Ratio (SNR) of the signals, e.g., transits.
    The estimate assumes:
    - there is no red noises (due to systematics, etc.)
    - noise level does not vary over time.

    References:
    - based on KSCI-19085-001: Planet Detection Metrics
      (section 3:  one-sigma depth function. the basic form quoted is used instead of one-sigma depth function)
      https://exoplanetarchive.ipac.caltech.edu/docs/KSCI-19085-001.pdf
    - Poster with more in-depth treatment on white noises, red noises, etc.
      https://mirasolinstitute.org/kaspar/publications/Window_Functions.pdf
    """

    if not isinstance(signal_duration, u.Quantity):
        signal_duration = signal_duration * u.hour
    if cdpp_kwargs is None:
        cdpp_kwargs = dict()

    cadence = estimate_cadence(lc, unit=u.min)
    transit_window = math.ceil(signal_duration / cadence)

    savgol_window = transit_window * savgol_to_transit_window_ratio
    if savgol_window % 2 == 0:
        savgol_window += 1

    if cdpp_kwargs.get("transit_duration") is None:
        cdpp_kwargs["transit_duration"] = transit_window
    if cdpp_kwargs.get("savgol_window") is None:
        cdpp_kwargs["savgol_window"] = savgol_window

    cdpp = lc.estimate_cdpp(**cdpp_kwargs)

    snr = np.sqrt(num_signals) * (signal_depth / cdpp).decompose().value

    if not return_diagnostics:
        return snr
    else:
        diagnostics = cdpp_kwargs.copy()
        diagnostics["cdpp"] = cdpp
        return snr, diagnostics


def estimate_b(depth, t_full, t_total):
    # equation 1.8 of https://www.astro.ex.ac.uk/people/alapini/Publications/PhD_chap1.pdf,
    # which quotes  Seager & Mall??n-Ornelas (2003)
    # limitations include: limb darkening is not taken into account. Circular orbit is assumed
    d, tF, tT = depth, t_full, t_total
    return (((1 - d**0.5) ** 2 - (tF / tT) ** 2 * (1 + d**0.5) ** 2) / (1 - (tF / tT) ** 2)) ** 0.5


def estimate_transit_duration_for_circular_orbit(period, rho, b):
    """Estimate the transit duration for circular orbit, T_circ.
    Usage: if the observed transit duration, T_obs,
    Case 1. T_obs < T_circ significantly: the planet candidate is potentially
    - Case 1a:
      - in a highly eccentric orbit, and
      - the transit occurs near periastron (nearest point to the host star)
        - the planet moves faster, thus the duration is shorter, figure 1 orange line
    - Case 1b:
      - actual impact parameter b is higher than the estimate
        - higher b implies the planet traverses across less of the host star's cross section
    - the uncertainty is manifestation of e-w-b degeneracy.

    Case 2. T_obs > T_circ significantly: the planet candidate is almost definitely
    - in a highly eccentric orbit, and
    - the transit occurs near apastron (farthest point to the host star)
      - the planet moves slower, thus the duration is longer, figure 1 green line

    cf. The TESS???Keck Survey. VI. Two Eccentric Sub-Neptunes Orbiting HIP-97166, MacDougall et. al
    https://ui.adsabs.harvard.edu/abs/2021AJ....162..265M/abstract

    - implementing the equations in section 2.2
    - see figure 1 for the concept, and section 2.2 for interpreting the result
    """
    # use default units if the input is not quantity
    if not isinstance(period, u.Quantity):
        period = period * u.day

    if not isinstance(rho, u.Quantity):
        rho = rho * u.gram / u.cm**3

    # implement eq. 1, 2, and 3
    return (
        period ** (1 / 3)
        * rho ** (-1 / 3)
        * (1 - b**2) ** (1 / 2)
        *
        # constants that are not explicitly stated in eq 3, but can be derived from eq 1 and 2
        np.pi ** (-2 / 3)
        * 3 ** (1 / 3)
        * astropy.constants.G ** (-1 / 3)
    ).to(u.hour)


def _calc_median_flux_around(lc_f: lk.FoldedLightCurve, epoch_phase, flux_window_in_min):
    flux_window = flux_window_in_min / 60 / 24  # in days
    lc_trunc = lc_f.truncate(epoch_phase - flux_window / 2, epoch_phase + flux_window / 2).remove_nans()
    flux_median = np.median(lc_trunc.flux)
    flux_median_sample_size = len(lc_trunc)

    return flux_median, flux_median_sample_size


def calc_flux_at_minimum(lc_f: lk.FoldedLightCurve, flux_window_in_min=10):
    """Return the flux at minimum by calculating the median of the flux at minimum, assumed to be at phase 0"""
    return _calc_median_flux_around(lc_f, 0, flux_window_in_min=flux_window_in_min)


def calc_peak_to_peak(lc_f: lk.FoldedLightCurve, flux_window_in_min=10):
    """Return the flux at minimum by calculating the median of the flux at minimum"""

    argmax_func, argmin_func = "argmax", "argmin"
    if lc_f.flux.unit == u.mag:
        # if the unit is magnitude, reverse max/min
        argmax_func, argmin_func = "argmin", "argmax"

    time_max = lc_f.time[getattr(lc_f.flux, argmax_func)()]
    flux_max, flux_max_sample_size = _calc_median_flux_around(lc_f, time_max.value, flux_window_in_min=flux_window_in_min)

    time_min = lc_f.time[getattr(lc_f.flux, argmin_func)()]
    flux_min, flux_min_sample_size = _calc_median_flux_around(lc_f, time_min.value, flux_window_in_min=flux_window_in_min)

    peak_to_peak = np.abs(flux_max - flux_min)

    return SimpleNamespace(
        peak_to_peak=peak_to_peak,
        time_max=time_max,
        flux_max=flux_max,
        flux_max_sample_size=flux_max_sample_size,
        time_min=time_min,
        flux_min=flux_min,
        flux_min_sample_size=flux_min_sample_size,
    )


def select_flux(lc, flux_cols):
    """Return a Lightcurve object with the named column as the flux column.

    flux_cols: either a column name (string), or a list of prioritized column names
    such that the first one that the lightcurve contains will be used.
    """

    def _to_lc_with_1flux(lc, flux_1col):
        flux_1col = flux_1col.lower()
        if "flux" == flux_1col:
            return lc
        elif flux_1col in lc.colnames:
            return lc.select_flux(flux_1col)
        else:
            return None

    if isinstance(flux_cols, str):
        flux_cols = [flux_cols]

    for flux_1col in flux_cols:
        res = _to_lc_with_1flux(lc, flux_1col)
        if res is not None:
            return res
    raise ValueError(f"'column {flux_cols}' not found")


def normalized_flux_val_to_mag(flux_val, base_mag):
    return base_mag + 2.5 * np.log10(1 / flux_val)


def to_flux_in_mag_by_normalization(lc, base_mag_header_name="TESSMAG"):
    """Convert the a lightcurve's flux to magnitude via a normalized lightcurve with a known average / base magnitude."""
    if lc.flux.unit is u.mag:
        return lc

    lc = lc.copy()

    base_mag = lc.meta.get(base_mag_header_name)
    if base_mag is None:
        raise ValueError(f"The given lightcurve does not have base magnitude in {base_mag_header_name} header ")

    lc_norm = lc.normalize()
    flux_mag = (base_mag + 2.5 * np.log10(1 / lc_norm.flux)) * u.mag
    flux_err_mag = (1.086 * lc_norm.flux_err / lc_norm.flux) * u.mag
    lc.flux = flux_mag
    lc.flux_err = flux_err_mag
    lc.meta["NORMALIZED"] = False
    return lc


def ratio_to_mag(val_in_ratio):
    """Convert normalized transit depth to magnitude."""
    return 2.5 * np.log10(1 / (1 - val_in_ratio))


def to_hjd_utc(t_obj: Time, sky_coord: SkyCoord) -> Time:
    # Based on astropy documentation
    # https://docs.astropy.org/en/stable/time/#barycentric-and-heliocentric-light-travel-time-corrections

    t_jd_tdb = t_obj.copy("jd").tdb

    # 1. convert the given time in JD TDB to local time UTC
    greenwich = coord.EarthLocation.of_site("greenwich")
    ltt_bary = t_jd_tdb.light_travel_time(sky_coord, location=greenwich, kind="barycentric")
    t_local_jd_utc = (t_jd_tdb - ltt_bary).utc

    # 2. convert local time UTC to HJD UTC
    ltt_helio = t_local_jd_utc.light_travel_time(sky_coord, location=greenwich, kind="heliocentric")
    t_hjd_utc = t_local_jd_utc + ltt_helio

    return t_hjd_utc


HAS_BOTTLENECK = False
try:
    import bottleneck

    HAS_BOTTLENECK = True
except:
    HAS_BOTTLENECK = False


def parse_aggregate_func(cenfunc):
    # based on Astropy SigmaClip
    # https://github.com/astropy/astropy/blob/326435449ad8d859f1abf36800c3fb88d49c27ea/astropy/stats/sigma_clipping.py#L263

    # cenfunc should really be aggfunc, but I keep the name from SigmaClip
    if cenfunc is None:
        cenfunc = "mean"

    if isinstance(cenfunc, str):
        if cenfunc == "median":
            if HAS_BOTTLENECK:
                cenfunc = bottleneck.nanmedian  # SigmaClip has more robust version
            else:
                cenfunc = np.nanmedian  # pragma: no cover

        elif cenfunc == "mean":
            if HAS_BOTTLENECK:
                cenfunc = bottleneck.nanmean  # SigmaClip has more robust version
            else:
                cenfunc = np.nanmean  # pragma: no cover

        else:
            raise ValueError(f"{cenfunc} is an invalid cenfunc.")

    return cenfunc


def bin_flux(lc, columns=["flux", "flux_err"], **kwargs):
    """Helper to bin() more efficiently."""
    # Note: the biggest slowdown comes from astropy regression
    # that this impl cannot address:
    # https://github.com/astropy/astropy/issues/13058

    # construct a lc_subset that only has a subset of columns,
    # to minimize the number of columns that need to be binned
    # see: https://github.com/lightkurve/lightkurve/issues/1191

    # lc_subset = lc['time', 'flux', 'flux_err'] does not work
    # due to https://github.com/lightkurve/lightkurve/issues/1194
    lc_subset = type(lc)(time=lc.time.copy())
    lc_subset.meta.update(lc.meta)
    for c in columns:
        if c in lc.colnames:
            lc_subset[c] = lc[c]
        else:
            warnings.warn(f"bin_flux(): column {c} cannot be found in lightcurve. It is ignored.")

    aggregate_func = parse_aggregate_func(kwargs.get("aggregate_func"))
    kwargs["aggregate_func"] = aggregate_func

    return lc_subset.bin(**kwargs)


def abbrev_sector_list(lcc_or_sectors):
    """Abbreviate a list of sectors, e.g., `1,2,3, 9` becomes `1-3, 9`."""

    # OPEN 1: consider to handle SearchResult as well, in addition to
    #         array like numbers, LightCurveCollection and TargetPixelFileCollection
    # OPEN 2: consider to handle Kepler quarter / K2 campaign.

    sectors = lcc_or_sectors
    if isinstance(sectors, lk.collections.Collection):  # LC / TPF collection
        sectors = [lc.meta.get("SECTOR") for lc in lcc_or_sectors]

    sectors = sectors.copy()
    sectors.sort()

    if len(sectors) < 1:
        return ""

    def a_range_to_str(range):
        if range[0] == range[1]:
            return str(range[0])
        else:
            return f"{range[0]}-{range[1]}"

    ranges = []
    cur_range = [sectors[0], sectors[0]]
    for s in sectors[1:]:
        if s == cur_range[1] + 1:
            cur_range = [cur_range[0], s]
        else:
            ranges.append(cur_range)
            cur_range = [s, s]
    ranges.append(cur_range)

    return ", ".join([a_range_to_str(r) for r in ranges])


#
# TargetPixelFile helpers
#


def truncate(tpf: lk.targetpixelfile.TargetPixelFile, before: float, after: float) -> lk.targetpixelfile.TargetPixelFile:
    return tpf[(tpf.time.value >= before) & (tpf.time.value <= after)]


def to_lightcurve_with_custom_aperture(tpf, aperture_mask, background_mask):
    """Create a lightcurve from the given TargetPixelFile, with suppplied aperture and background"""
    n_aperture_pixels = aperture_mask.sum()
    aperture_lc = tpf.to_lightcurve(aperture_mask=aperture_mask)  # aperture + background

    n_background_pixels = background_mask.sum()
    background_lc_per_pixel = tpf.to_lightcurve(aperture_mask=background_mask) / n_background_pixels
    background_lc = background_lc_per_pixel * n_aperture_pixels

    corrected_lc = aperture_lc - background_lc
    # OPEN: consider filling in lc metadata, label, etc.

    return corrected_lc, aperture_lc, background_lc


#
# Astropy extension
#

# Specify / display time delta in hours
# useful for specifying / displaying planet transits
class TimeDeltaHour(astropy.time.TimeDeltaNumeric):
    """Time delta in hours (3600 SI seconds)"""

    import erfa

    name = "hour"
    unit = 3600.0 / erfa.DAYSEC  # for quantity input


#
# Others
#


def coordinate_like_id_to_coordinate(id, style="decimal"):
    """Given an identifier that describes the target's coordinate, return the coordInate in string or `SkyCoord` object.
    The type of IDs supported are in the form of '<prefix-of-catalog> J<ra><dec>'.
    Catalogs that adopts such form of ids include PTF1, ASASSN-V, WISE, WISEA, 1SWASP, etc.

    """
    # e.g.,
    # PTF1 J2219+3135 (the shortened form often found in papers)
    # PTF1 J221910.09+313523.1  (the actual id, with higher precision)
    result = re.findall(r"^.{2,}\s*J(\d{2})(\d{2})([0-9.]*)([+-])(\d{2})(\d{2})([0-9.]*)\s*$", id)
    if len(result) < 1:
        return None
    [ra_hh, ra_mm, ra_ss, dec_sign, dec_deg, dec_min, dec_ss] = result[0]
    coord = SkyCoord(f"{ra_hh} {ra_mm} {ra_ss} {dec_sign} {dec_deg} {dec_min} {dec_ss}", unit=(u.hourangle, u.deg))
    if style is None:
        return coord
    else:
        return coord.to_string(style=style)


from astropy.coordinates import SkyCoord, Angle
from astroquery.vizier import Vizier


def search_nearby(
    ra,
    dec,
    equinox="J2000.0",
    catalog_name="I/355/gaiadr3",
    radius_arcsec=60,
    magnitude_limit_column=None,
    magnitude_lower_limit=None,
    magnitude_upper_limit=None,
    pmra=None,
    pmdec=None,
    pm_range_fraction=None,
    pmra_limit_column="pmRA",
    pmdec_limit_column="pmDE",
):
    """Stars around the given coordinate from Gaia DR2/EDR3, etc."""

    c1 = SkyCoord(ra, dec, equinox=equinox, frame="icrs", unit="deg")
    Vizier.ROW_LIMIT = -1
    columns = ["*"]
    if catalog_name == "I/350/gaiaedr3":
        columns = ["*", "epsi", "sepsi"]  # add astrometric excess noise to the output (see if a star wobbles)
    if catalog_name == "I/355/gaiadr3":
        columns = ["*", "epsi", "sepsi", "VarFlag", "Dup", "GRVSmag"]  # also add variability and Duplicate flag

    with warnings.catch_warnings():
        # suppress useless warning.  https://github.com/astropy/astroquery/issues/2352
        warnings.filterwarnings(
            "ignore", category=astropy.units.UnitsWarning, message="Unit 'e' not supported by the VOUnit standard"
        )
        result = Vizier(columns=columns).query_region(
            c1,
            catalog=[catalog_name],
            radius=Angle(radius_arcsec, "arcsec"),
        )
    if len(result) < 1:  # handle no search result case
        return None
    result = result[catalog_name]

    if magnitude_lower_limit is not None:
        result = result[(magnitude_lower_limit <= result[magnitude_limit_column]) | result[magnitude_limit_column].mask]

    if magnitude_upper_limit is not None:
        result = result[(result[magnitude_limit_column] <= magnitude_upper_limit) | result[magnitude_limit_column].mask]

    if pm_range_fraction is not None:
        if pmra is not None:
            pmra_range = np.abs(pmra) * pm_range_fraction
            pmra_lower, pmra_upper = pmra - pmra_range, pmra + pmra_range
            result = result[(pmra_lower <= result[pmra_limit_column]) | result[pmra_limit_column].mask]
            result = result[(result[pmra_limit_column] <= pmra_upper) | result[pmra_limit_column].mask]
        if pmdec is not None:
            pmdec_range = np.abs(pmdec) * pm_range_fraction
            pmdec_lower, pmdec_upper = pmdec - pmdec_range, pmdec + pmdec_range
            result = result[(pmdec_lower <= result[pmdec_limit_column]) | result[pmdec_limit_column].mask]
            result = result[(result[pmdec_limit_column] <= pmdec_upper) | result[pmdec_limit_column].mask]

    # Calculated separation is approximate, as proper motion is not accounted for
    r_coords = SkyCoord(result["RAJ2000"], result["DEJ2000"], unit=(u.hourangle, u.deg), frame="icrs")
    sep = r_coords.separation(c1).to(u.arcsec)
    result["separation"] = sep

    result.sort("separation")

    # tweak default format to make magnitudes and separation more succinct
    for col in ["separation", "RPmag", "Gmag", "BPmag", "BP-RP", "GRVSmag"]:
        if col in result.colnames:
            result[col].info.format = ".3f"

    return result
