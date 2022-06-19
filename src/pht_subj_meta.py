import csv
from pathlib import Path
import re

from ratelimit import limits, sleep_and_retry

from common import *

# throttle HTTP calls to Zooniverse
NUM_CALLS = 5
TEN_SECONDS = 10

@sleep_and_retry
@limits(calls=NUM_CALLS, period=TEN_SECONDS)
def _get_subject_meta_of_id(id, json=False):
    url = f"https://www.zooniverse.org/api/subjects?http_cache=true&id={id}"
    res = fetch_json(url)
    if json:
        return res
    subject = res["subjects"][0]
    img_url = subject["locations"][0].get("image/png", "")
    img_id = img_url
    match_res = re.match("https://panoptes-uploads.zooniverse.org\/subject_location\/(.+)[.]png", img_url)
    if match_res is not None:
        img_id = match_res[1]
    res = dict(
        subject_id=subject["id"],
        tic_id=subject["metadata"].get("!TIC ID", -1),  # -1 for simulated ones
        sector=subject["metadata"]["Sector"],
        img_id=img_id,
    )
    return res


def get_subject_meta_of_ids(ids, subject_result_func=None):
    kwargs_list = [dict(id=id) for id in ids]
    return bulk_process(_get_subject_meta_of_id, kwargs_list, process_result_func=subject_result_func)


def save_meta_of_subject(subject_meta, call_i, call_kwargs):
    out_path = Path("../data/pht_subj_meta.csv")
    fieldnames = ['subject_id', 'tic_id', 'sector', 'img_id']
    with open(out_path, "a") as f:
        csv_writer = csv.DictWriter(f, fieldnames, dialect="unix")
        csv_writer.writerow(subject_meta)

#
# Top level driver
#
if __name__ =="__main__":
    ids = load_subject_ids_from_file()
    # ids = ids[1000:]
    print(f"Meta for {len(ids)} subjects: {ids[0]} ... {ids[-1]}")
    get_subject_meta_of_ids(ids, subject_result_func=save_meta_of_subject)


