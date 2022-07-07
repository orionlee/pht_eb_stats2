import numpy as np

import pandas as pd

from common import to_csv

from pht_subj_comments_per_comment import load_comment_summaries_table_from_file


def _load_tag_map():
    """Return a dict fo tag -> tag group"""
    df_map = pd.read_csv("../data/auxillary/pht_tag_map.csv")
    return df_map.set_index("tag", drop=True).to_dict()["group"]


def _to_summary_of_subject(df_subj):
    # Note: old, slower implementation for the aggregation
    # keep the codes here in case the use case evolves such that I'll need something like it later.
    res = dict(
        subject_id=df_subj.iloc[0]["subject_id"],
        eb_score=-1,
        # for `num_votes_eb`, we count number of distinct user_ids, so that if
        # an user has tagged on multiple comments in a subject, only 1 vote is counted
        num_votes_eb=df_subj[df_subj["has_tag_like_eb"] == 1]["user_id"].nunique(),
        num_votes_transit=df_subj[df_subj["has_tag_like_transit"] == 1]["user_id"].nunique(),
        num_users=df_subj["user_id"].nunique(),
        num_comments=len(df_subj),
        updated_at=df_subj["updated_at"].max(),
    )
    res["eb_score"] = res["num_votes_eb"] - res["num_votes_transit"]
    return pd.Series(res)


def _add_tag_groups(df_comments):
    # TODO: we handle cases such as marked as #rr-lyrae, #contamination, #NEB
    # TODO: handle the #EB in message (not recognized as tag by Zooniverse systems, but the intent is there)
    tag_map = _load_tag_map()
    # optimization: set the value as user_id to, rather than True/False
    # to make count users in aggregation downstream faster
    # Use Nullable Integer array to hold result:
    # - user_id if eb-like tags is in a comment, `None` otherwise.
    col_has_tag_like_eb = pd.array(np.full(len(df_comments), np.nan), "Int64")
    col_has_tag_like_transit = pd.array(np.full(len(df_comments), np.nan), "Int64")
    for i, (tags_str, user_id) in enumerate(zip(df_comments["tags"], df_comments["user_id"])):
        tags = tags_str.split(",")
        for tag in tags:
            if tag_map.get(tag) == "eb":
                col_has_tag_like_eb[i] = user_id
            elif tag_map.get(tag) == "transit":
                col_has_tag_like_transit[i] = user_id
    df_comments["has_tag_like_eb"] = col_has_tag_like_eb
    df_comments["has_tag_like_transit"] = col_has_tag_like_transit
    return df_comments


def save_and_summarize_of_all_subjects(also_return_df_comments=False, dry_run=False, dry_run_data_size=1000):
    out_path = "../data/pht_subj_comments_summary.csv"
    df_comments = load_comment_summaries_table_from_file(include_is_deleted=False)
    if dry_run and dry_run_data_size is not None:
        df_comments = df_comments[:dry_run_data_size]
    df_comments = _add_tag_groups(df_comments)
    # df_summary = df_comments.groupby("subject_id").apply(_to_summary_of_subject)
    # optimization: instead of using `apply` (that let me work on per-subject sub data frame)
    # use `agg` that work on per-column
    # I make it work by having user_id in has_tag_like_eb, has_tag_like_transit columns
    # running time reduced by ~75% for ~90,000 row dataset
    df_summary = df_comments.groupby("subject_id").agg(
        subject_id=("subject_id", "first"),
        # for `num_votes_eb`, we count number of distinct user_ids, so that if
        # an user has tagged on multiple comments in a subject, only 1 vote is counted
        num_votes_eb=("has_tag_like_eb", pd.Series.nunique),
        num_votes_transit=("has_tag_like_transit", pd.Series.nunique),
        num_users=("user_id", pd.Series.nunique),
        num_comments=("comment_id", "count"),
        updated_at=("updated_at", "max"),
    )
    col_eb_score = df_summary["num_votes_eb"] - df_summary["num_votes_transit"]
    df_summary.insert(1, "eb_score", col_eb_score)

    if not dry_run:
        to_csv(df_summary, out_path, mode="w")

    if also_return_df_comments:
        return df_summary, df_comments
    else:
        return df_summary


def load_pht_subj_comment_summary_table_from_file():
    csv_path = "../data/pht_subj_comments_summary.csv"
    df = pd.read_csv(csv_path)
    return df


if __name__ == "__main__":
    save_and_summarize_of_all_subjects(also_return_df_comments=False)
