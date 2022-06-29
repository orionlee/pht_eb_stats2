from pathlib import Path

import numpy as np
from ratelimit import limits, sleep_and_retry

from common import fetch_json, bulk_process

# throttle HTTP calls to Zooniverse
NUM_CALLS = 5
TEN_SECONDS = 10


@sleep_and_retry
@limits(calls=NUM_CALLS, period=TEN_SECONDS)
def _get_subject_ids_of_tag_page(tag, page, end_subject_id_exclusive=None, json=False):
    url = (
        f"https://talk.zooniverse.org/tags/popular?http_cache=true&taggable_type=Subject"
        f"&section=project-7929&name={tag}&page={page}"
    )
    res = fetch_json(url)
    if json:
        return res

    res = [e["taggable_id"] for e in res["popular"]]

    # use case: filter out subjects beyond the intended sector range in the last page
    if end_subject_id_exclusive is not None:
        res_all = res
        res = []
        for id in res_all:
            if id < end_subject_id_exclusive:
                res.append(id)
            else:
                print(f"[DEBUG] subject {id} at page {page} is excluded")
    return res


def get_subject_ids_of_tag(tag, page_start, page_end_inclusive, end_subject_id_exclusive=None, page_result_func=None):
    kwargs_list = [
        dict(tag=tag, page=i, end_subject_id_exclusive=end_subject_id_exclusive)
        for i in range(page_start, page_end_inclusive + 1)
    ]
    return bulk_process(_get_subject_ids_of_tag_page, kwargs_list, process_result_func=page_result_func)


def get_subject_ids_of_tag_old(tag, page_start, page_end_inclusive, page_result_func=None):
    res = []
    for page in range(page_start, page_end_inclusive + 1):
        page_res = _get_subject_ids_of_tag_page(tag, page)
        if page_result_func is not None:
            page_result_func(page, page_res)
        res = res + page_res
    return res


def save_subject_ids_of_page(subject_ids, call_i, call_kwargs):
    out_path = Path("../data/pht_subj_ids.csv")

    with open(out_path, "a") as f:
        np.savetxt(f, subject_ids, fmt="%s")

    return out_path


#
# Top level driver
#
if __name__ == "__main__":
    # params for sectors #eclipsingbinary in sectors 1 to 39
    # Subject 68601250 is the first sector 40 on page 3230
    # get_subject_ids_of_tag(
    #     "eclipsingbinary", 1, 3230, end_subject_id_exclusive=68601250, page_result_func=save_subject_ids_of_page
    # )

    # I decided to crawl all pages instead of trying to limit to sectors 1 to 39
    # because the subject ids are not strictly increasing.
    # Filtering will be done afterwards
    page_start, page_end_inclusive, end_subject_id_exclusive = 1, 3602, None
    # page_start, page_end_inclusive, end_subject_id_exclusive = 1001, 3602, None

    print(f"EB subject ids of page [{page_start}, {page_end_inclusive}] ; end_subject_id_exclusive={end_subject_id_exclusive}")
    get_subject_ids_of_tag(
        "eclipsingbinary",
        page_start,
        page_end_inclusive,
        end_subject_id_exclusive=end_subject_id_exclusive,
        page_result_func=save_subject_ids_of_page,
    )
