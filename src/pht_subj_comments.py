from pathlib import Path

from ratelimit import limits, sleep_and_retry

from common import *

# throttle HTTP calls to Zooniverse
NUM_CALLS = 5
TEN_SECONDS = 10

@sleep_and_retry
@limits(calls=NUM_CALLS, period=TEN_SECONDS)
def _get_subject_comment_of_id_n_page(id, page):
    url = f"https://talk.zooniverse.org/comments?http_cache=true&section=project-7929&focus_type=Subject&sort=-created_at&focus_id={id}&page={page}"
    return fetch_json(url)

def _get_subject_comments_of_id(id):
    # fetch all pages and combine them to 1 JSON object

    res = _get_subject_comment_of_id_n_page(id, 1)
    res['meta']['subject_id'] = id  # add it to the result for ease of identification
    num_pages = res['meta']['comments']['page_count']
    for page in range(2, num_pages + 1):
        page_res = _get_subject_comment_of_id_n_page(id, page)
        res['comments'] = res['comments'] + page_res['comments']

    return res


def get_subject_comments_of_ids(ids, subject_result_func=None):
    kwargs_list = [dict(id=id) for id in ids]
    return bulk_process(_get_subject_comments_of_id, kwargs_list, process_result_func=subject_result_func)


def save_comments_of_subject(subject_comments, call_i, call_kwargs):
    id = subject_comments['meta']['subject_id']
    out_path = Path(f"cache/comments/c{id}.json")  # the c prefix hints it is a comment
    with open(out_path, "w") as f:
        json_np_dump(subject_comments, f)


def _get_subject_comments_of_id_from_cache(subject_id):
    with open(f"cache/comments/c{subject_id}.json", "r") as f:
        return json.load(f)


#
# Top level driver
#
if __name__ =="__main__":
    ids = load_subject_ids_from_file()
    # ids = ids[10:100]
    print(f"Comments for {len(ids)} subjects: {ids[0]} ... {ids[-1]}")
    get_subject_comments_of_ids(ids, subject_result_func=save_comments_of_subject)

