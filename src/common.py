from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
import csv
from enum import Enum, unique
import json
import os
import time

from astropy.table import Table
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm


def fetch_json(url):
    # the header is needed for Zooniverse subject metadata API call
    headers = {"accept": "application/vnd.api+json; version=1", "content-type": "application/json"}
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    return r.json()


def bulk_process(
    process_fn,
    process_kwargs_list,
    return_result=False,
    process_result_func=None,
    tqdm_kwargs=None,
    num_retries=2,
    retry_wait_time_seconds=30,
):
    if tqdm_kwargs is None:
        tqdm_kwargs = dict()

    num_to_process = len(process_kwargs_list)
    res = []
    for i in tqdm(range(num_to_process), **tqdm_kwargs):
        for try_num in range(1, num_retries + 1):
            try:
                i_res = process_fn(**process_kwargs_list[i])
                if return_result:
                    res.append(i_res)
                else:
                    res.append(True)  # indicate the process is a success, in case we support skipping those that lead to error
                if process_result_func is not None:
                    process_result_func(i_res, i, process_kwargs_list[i])
                break
            except BaseException as err:
                print(f"Error in processing {i}th call. Arguments: {process_kwargs_list[i]}. Error: {type(err)} {err}")
                if try_num < num_retries:
                    print(f"Retry after {retry_wait_time_seconds} seconds for try #{try_num + 1}")
                    time.sleep(retry_wait_time_seconds)
                else:
                    raise err

    return res


def load_subject_ids_from_file(filepath="../data/pht_subj_ids.csv"):
    # use numpy array to make the processing downstream easier,
    # e.g., accessing last element by [-1] rather than the more cumbersome .iloc[-1]
    ids = pd.read_csv(filepath)["subject_id"].to_numpy()
    return ids


def load_tic_ids_from_file(filepath="../data/pht_tic_ids.csv"):
    # use numpy array to make the processing downstream easier,
    # e.g., accessing last element by [-1] rather than the more cumbersome .iloc[-1]
    ids = pd.read_csv(filepath)["tic_id"].to_numpy()
    return ids


# from https://stackoverflow.com/a/57915246
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def json_np_dump(obj, fp, **kwargs):
    """JSON dump that supports numpy data types"""
    kwargs["cls"] = NpEncoder
    return json.dump(obj, fp, **kwargs)


def _df_to_csv(df, out_path, mode="a"):
    if (not mode.startswith("w")) and (os.path.exists(out_path)) and (os.path.getsize(out_path) > 0):
        header = False
    else:
        header = True
    return df.to_csv(out_path, index=False, mode=mode, header=header)


def to_csv(data, out_path, mode="a", fieldnames=None):
    if isinstance(data, Table):
        data = data.to_pandas()

    if isinstance(data, pd.DataFrame):
        return _df_to_csv(data, out_path, mode=mode)

    # parameters processing
    if fieldnames is None:
        if isinstance(data, Mapping):
            fieldnames = data.keys()
        elif isinstance(data, Sequence):
            fieldnames = data[0].keys()
        else:
            raise TypeError(f"Unsupported type for `data`: {type(dict)}")

    def write_header_if_needed():
        if (not mode.startswith("w")) and (os.path.exists(out_path)) and (os.path.getsize(out_path) > 0):
            return False  # the file has content. no need to write header
        header = ",".join(fieldnames)
        header = header + "\n"
        with open(out_path, mode, encoding="utf-8") as f:
            f.write(header)

    def to_csv_of_dict(a_dict):
        with open(out_path, mode, encoding="utf-8") as f:
            csv_writer = csv.DictWriter(f, fieldnames, dialect="unix")
            csv_writer.writerow(a_dict)

    # Main logic
    write_header_if_needed()
    if isinstance(data, Mapping):
        to_csv_of_dict(data)
    elif isinstance(data, Sequence):
        [to_csv_of_dict(a_dict) for a_dict in data]
    else:
        raise TypeError(f"Unsupported type for `data`: {type(dict)}")


#
# Pandas Utilities
#


def has_value(val):
    return not pd.isna(val)


def prefix_columns(df: pd.DataFrame, prefix: str, **kwargs):
    column_map = {col: f"{prefix}_{col}" for col in df.columns}
    return df.rename(columns=column_map, **kwargs)


def insert(df: pd.DataFrame, before_colname: str, colname: str, value):
    """Insert the `(colname, value)` as a new column before `loc_colname`."""
    loc = df.columns.get_loc(before_colname)
    return df.insert(loc, colname, value)


def move(df: pd.DataFrame, colname: str, before_colname: str):
    """Move the column `move_colname` before the column `loc_colname`."""
    col_to_move = df.pop(colname)
    loc = df.columns.get_loc(before_colname)
    return df.insert(loc, colname, col_to_move)


def as_nullable_int(df: pd.DataFrame, columns: Sequence):
    if isinstance(columns, str):
        columns = [columns]

    for c in columns:
        df[c] = df[c].astype("Int64")

    return df


def left_outer_join_by_column_merge(df_main, df_aux, join_col_main, join_col_aux, prefix_aux, add_is_in_col=True):
    """Left outer join of 2 dataframes by the way of column merge.
       It is akin to horizontal stacking, except the right (auxillary) dataframe
       may not have records for a row in the left (main) dataframe; i.e.,
       the relationship of the 2 dataframes must be 1: (0 or 1).

       Typical usage is to join the main PHT EB Catalog with some auxillary data (from other catalogs) by TIC.
       """
    # column-merge the tables, typically by TIC
    df_main.set_index(join_col_main, drop=False, inplace=True)

    # drop the join column from aux, as it will be a duplicate in the result
    df_aux.set_index(join_col_aux, drop=True, inplace=True)
    if add_is_in_col:  # a convenience column to indicate if an entry has record in df_aux
        df_aux["Is_In"] = 'T'
    prefix_columns(df_aux, prefix_aux, inplace=True)

    df_main = pd.concat([df_main, df_aux], join="outer", axis=1)
    if add_is_in_col:
        df_main[f"{prefix_aux}_Is_In"] = df_main[f"{prefix_aux}_Is_In"].fillna("F")

    return df_main


#
# Helpers to map a catalog's type(s) to Is_EB
#


@unique
class MapResult(Enum):
    """Used to represent the possible values of a `Is_EB` column"""

    def __new__(cls, value, label):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.label = label
        return obj

    TRUE = (4, "T")
    FALSE = (3, "F")
    NOT_MAPPED = (2, "?")
    NA = (1, "-")


class AbstractTypeMapAccessor(ABC):
    def __init__(self, map_csv_path: str, type_colname: str, is_eb_colname: str = "Is_EB"):
        """
        Parameters
        ----------
        map_csv_path :
            the path of the csv file that provides catalog type to `Is_EB` mapping
        type_colname :
            the name of the column in the csv for catalog type
        is_eb_colname :
            the name of the column in the csv for `Is_EB`. Defaulted to `Is_EB`
        """

        def _to_map_result(is_eb):
            if pd.isna(is_eb):
                res = MapResult.NA
            elif is_eb:
                res = MapResult.TRUE
            else:
                res = MapResult.FALSE
            return res

        # Is_EB column: Nullable boolean,
        # N/A would mean the classification has no bearing on Is_EB
        self.df = pd.read_csv(map_csv_path, dtype={is_eb_colname: "boolean"})

        # convert the mapping needed from dataframe to a dictionary
        # to avoid the overhead of repeated dataframe access
        col_is_eb = [_to_map_result(is_eb) for is_eb in self.df[is_eb_colname]]
        self._types_dict = dict(zip(self.df[type_colname], col_is_eb))

        self.not_mapped_types_seen = set()

    def _map_1_type(self, type: str) -> MapResult:
        res = self._types_dict.get(type, MapResult.NOT_MAPPED)
        if res == MapResult.NOT_MAPPED:
            self.not_mapped_types_seen.add(type)
        return res

    @abstractmethod
    def _split_types_str(self, types_str: str) -> Sequence:
        """Split a catalog type list string into a list of type.

        Subclass must implement the method.
        """
        pass

    def map(self, types_str: str) -> MapResult:
        """Given a catalog list type string, return the best Is_EB value."""
        if pd.isna(types_str):
            return MapResult.NA
        types_str = self._split_types_str(types_str)
        res_list = [self._map_1_type(t).value for t in types_str]
        best = np.max(res_list)
        return MapResult(best)
