import json

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


def bulk_process(process_fn, process_kwargs_list, return_result=False, process_result_func=None, tqdm_kwargs=None):
    if tqdm_kwargs is None:
        tqdm_kwargs = dict()

    num_to_process = len(process_kwargs_list)
    res = []
    for i in tqdm(range(num_to_process), **tqdm_kwargs):
        try:
            i_res = process_fn(**process_kwargs_list[i])
            if (return_result):
                res.append(i_res)
            else:
                res.append(True) # indicate the process is a success, in case we support skipping those that lead to error
            if process_result_func is not None:
                process_result_func(i_res, i, process_kwargs_list[i])
        except Exception as err:
            print(f"Error in processing {i}th call. Arguments: {process_kwargs_list[i]}")
            raise err

    return res

def load_subject_ids_from_file(filepath="../data/pht_subj_ids.csv"):
    # use numpy array to make the processing downstream easier,
    # e.g., accessing last element by [-1] rather than the more cumbersome .iloc[-1]
    ids = pd.read_csv(filepath)["subject_id"].to_numpy()
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
