import requests
from tqdm import tqdm

def fetch_json(url):
    # the header is needed for Zooniverse subject metadata API call
    headers = {"accept": "application/vnd.api+json; version=1", "content-type": "application/json"}
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    return r.json()


def bulk_process(process_fn, process_kwargs_list, process_result_func=None, tqdm_kwargs=None):
    if tqdm_kwargs is None:
        tqdm_kwargs = dict()

    num_to_process = len(process_kwargs_list)
    res = []
    for i in tqdm(range(num_to_process), **tqdm_kwargs):
        try:
            i_res = process_fn(**process_kwargs_list[i])
            res.append(i_res)
            if process_result_func is not None:
                process_result_func(i_res, i, process_kwargs_list[i])
        except Exception as err:
            print(f"Error in processing {i}th call. Arguments: {process_kwargs_list[i]}")
            raise err

    return res
