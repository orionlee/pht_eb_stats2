import re

from common import *
from pht_subj_comments import _get_subject_comments_of_id_from_cache

def _abbrev_bool(bool_val):
    if bool_val:
        return 1
    else:
        return 0

def _to_summary_of_comment(comment, subject_id):
    c = comment  # shortcut
    comment_id = c["id"]
    # note: c["focus_id"] is usually subject_id practice,
    # but there might be cases that it is not (e.g., a comment posted on Chat about a subject)
    # so I ask the caller to pass the subject_id instead.
    discussion_id = c["discussion_id"]  #  it's probably not useful. Grab it for just in case.
    tag_dict = c["tagging"]
    body = c["body"]
    is_deleted = c["is_deleted"]
    user_id = c["user_id"]
    updated_at = c["updated_at"]

    # Process some of them to make csv export easier
    is_deleted = _abbrev_bool(is_deleted)
    body = re.sub(r"[\n\r\t]", " | ", body)  # replace characters problematic in a csv
    tags_str = ",".join(tag_dict.values())

    return dict(
        comment_id=comment_id,
        subject_id=subject_id,
        discussion_id=discussion_id,
        is_deleted=is_deleted,
        tags=tags_str,
        user_id=user_id,
        updated_at=updated_at,
        body=body,
    )


def to_comment_summaries_of_subject(id):
    comments = _get_subject_comments_of_id_from_cache(id)
    return [_to_summary_of_comment(c, id) for c in comments["comments"]]


def to_comment_summaries_of_subject_ids(ids, subject_result_func=None):
    kwargs_list = [dict(id=id) for id in ids]
    return bulk_process(to_comment_summaries_of_subject, kwargs_list, process_result_func=save_comment_summaries)


def save_comment_summaries(comment_summaries, call_i, call_kwargs):
    out_path = "cache/comments_summary_per_comment.csv"
    to_csv(comment_summaries, out_path, mode="a")


def get_comment_summaries_table(table_csv_path="cache/comments_summary_per_comment.csv"):
    df = pd.read_csv(
        table_csv_path,
        keep_default_na=False,  # so that empty string in tags won't be read as NaN
        parse_dates=["updated_at",],
        dtype=dict(is_deleted=bool),
        )
    return df


if __name__ =="__main__":
    ids = load_subject_ids_from_file()
    print(f"Comment summaries for {len(ids)} subjects: {ids[0]} ... {ids[-1]}")
    to_comment_summaries_of_subject_ids(ids, subject_result_func=save_comment_summaries)
