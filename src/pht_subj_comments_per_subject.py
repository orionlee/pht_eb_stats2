import numpy as np

from common import *

from pht_subj_comments_per_comment import load_comment_summaries_table_from_file

def _load_tag_map():
    """Return a dict fo tag -> tag group"""
    df_map = pd.read_csv("../data/pht_tag_map.csv")
    return df_map.set_index("tag", drop=True).to_dict()["group"]

def _to_summary_of_subject(df_subj):
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
    col_has_tag_like_eb = np.zeros(len(df_comments), dtype=int)
    col_has_tag_like_transit = np.zeros(len(df_comments), dtype=int)
    for i, tags_str in enumerate(df_comments["tags"]):
        tags = tags_str.split(",")
        for tag in tags:
            if tag_map.get(tag) == "eb":
                col_has_tag_like_eb[i] = 1
            elif tag_map.get(tag) == "transit":
                col_has_tag_like_transit[i] = 1
    df_comments["has_tag_like_eb"] = col_has_tag_like_eb
    df_comments["has_tag_like_transit"] = col_has_tag_like_transit
    return df_comments


def save_and_summarize_of_all_subjects(also_return_df_comments=False):
    out_path = "../data/pht_subj_comments_summary.csv"
    df_comments = load_comment_summaries_table_from_file(include_is_deleted=False)
    df_comments = _add_tag_groups(df_comments)
    df_summary = df_comments.groupby("subject_id").apply(_to_summary_of_subject)

    to_csv(df_summary, out_path, mode="w")
    if also_return_df_comments:
        return df_summary, df_comments
    else:
        return df_summary


if __name__ =="__main__":
    save_and_summarize_of_all_subjects(also_return_df_comments=False)

