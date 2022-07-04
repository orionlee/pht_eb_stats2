from typing import Sequence

import numpy as np
import pandas as pd

from common import as_nullable_int, insert
import pht_subj_comments_per_comment
import pht_subj_comments_per_subject


def _has_common_elements(vals1: set, vals2: Sequence, as_int=False):
    # https://stackoverflow.com/a/17735466
    res = not vals1.isdisjoint(vals2)
    if not as_int:
        return res
    else:
        return 1 if res else 0


def _create_subject_user_stats(dry_run=False, dry_run_size=1000):
    """Summarize the tagging at per Subject-user level.
    i.e., if an users makes multiple comments on a subject, summarize them as one entry
    """
    df = pht_subj_comments_per_comment.load_comment_summaries_table_from_file()
    if dry_run and dry_run_size is not None:
        df = df[:dry_run_size]

    # summarize them such Subject-User level
    stats = (
        df.groupby(["subject_id", "user_id"])
        .agg(
            {
                "comment_id": "count",
                "tags": ",".join,
            }
        )
        .reset_index(drop=False)
    )
    stats.rename(columns={"comment_id": "num_comments"}, inplace=True)

    stats["tags_set"] = [set(tags_str.split(",")) for tags_str in stats["tags"]]

    tag_map = pht_subj_comments_per_subject._load_tag_map()
    eb_tags = {tag for tag, group in tag_map.items() if group == "eb"}
    transit_tags = {tag for tag, group in tag_map.items() if group == "transit"}

    stats["has_eb_tags"] = [_has_common_elements(tags, eb_tags, as_int=True) for tags in stats["tags_set"]]
    stats["has_transit_tags"] = [_has_common_elements(tags, transit_tags, as_int=True) for tags in stats["tags_set"]]

    insert(stats, before_colname="has_eb_tags", colname="eb_score", value=stats["has_eb_tags"] - stats["has_transit_tags"])

    stats.drop("tags_set", axis=1, inplace=True)  # remove the intermediate columns

    # remove users who  has not made any relevant comments
    stats = stats[(stats["has_eb_tags"] == 1) | (stats["has_transit_tags"] == 1)].reset_index(drop=True)

    return stats


def create_user_eb_stats(df_subject_user_stats: pd.DataFrame = None, also_return_styler=False):
    if df_subject_user_stats is None:
        df_subject_user_stats = _create_subject_user_stats()

    # abbreviate it
    df = df_subject_user_stats
    num_subjects = df["subject_id"].nunique()
    stats = df[df["eb_score"] > 0].groupby("user_id")[["subject_id"]].count()
    stats.rename(columns={"subject_id": "num_subjects"}, inplace=True)
    stats.sort_values("num_subjects", ascending=False, inplace=True)
    stats["subject %"] = stats["num_subjects"] / num_subjects
    stats.reset_index(drop=False, inplace=True)
    insert(stats, before_colname="num_subjects", colname="user_rank", value=[i + 1 for i in range(len(stats))])

    if not also_return_styler:
        return stats
    else:
        styler = stats.style.format({"subject %": "{:.1%}"})
        return stats, styler


def _add_user_rank_to_subject_user_stats(df_subject_user_stats: pd.DataFrame, df_user_stats: pd.DataFrame):
    df_subject_user_stats.set_index("user_id", drop=False, inplace=True)
    df_user_stats = df_user_stats[["user_id", "user_rank"]]
    df_user_stats.set_index("user_id", drop=True, inplace=True)
    df = pd.concat([df_subject_user_stats, df_user_stats], join="outer", axis=1)
    as_nullable_int(df, ["user_rank"])

    return df


def _rank_group_func_default(rank):
    if pd.isna(rank):
        return pd.NA
    elif rank == 1:
        return "001"
    elif rank == 2:
        return "002"
    elif rank <= 10:
        return "003-10"
    elif rank <= 100:
        return "010-100"
    else:
        return "100+"


def pivot_by_rank_group(df_subject_user_stats: pd.DataFrame = None, df_user_stats: pd.DataFrame = None, rank_group_func=None):
    if df_subject_user_stats is None:
        df_subject_user_stats = _create_subject_user_stats
    if df_user_stats is None:
        df_user_stats = create_user_eb_stats(df_subject_user_stats)
    if rank_group_func is None:
        rank_group_func = _rank_group_func_default

    df = _add_user_rank_to_subject_user_stats(df_subject_user_stats, df_user_stats)

    # only those who have tagged eclipsing binary
    df = df[df["eb_score"] > 0].reset_index(drop=True)

    df["user_rank_group"] = [rank_group_func(val) for val in df["user_rank"]]

    report = df.pivot_table(index="user_rank_group", values="subject_id", aggfunc="nunique")
    report.rename(columns={"subject_id": "num_subjects"}, inplace=True)

    total_num_subjects = df["subject_id"].nunique()
    report["subject %"] = report["num_subjects"] / total_num_subjects

    return report


def calc_top_users_contributions(
    df_subject_user_stats: pd.DataFrame = None, df_user_stats: pd.DataFrame = None, top_n=10
):
    if df_subject_user_stats is None:
        df_subject_user_stats = _create_subject_user_stats
    if df_user_stats is None:
        df_user_stats = create_user_eb_stats(df_subject_user_stats)

    df = _add_user_rank_to_subject_user_stats(df_subject_user_stats, df_user_stats)

    total_num_subjects = df["subject_id"].nunique()

    # only those
    # - who have tagged eclipsing binary
    # - subject_ids only
    df = df[df["eb_score"] > 0].reset_index(drop=True)[["user_rank", "subject_id"]]

    ranks = np.arange(1, top_n + 1)
    cum_num_subjects = np.zeros_like(ranks, dtype=int)
    for i, cur_rank in enumerate(ranks):
        cum_num_subjects[i] = df[df["user_rank"] <= cur_rank]["subject_id"].nunique()

    cum_num_subjects_pct = cum_num_subjects / total_num_subjects

    stats = pd.DataFrame(
        {
            "rank": ranks,
            "cum_num_subjects": cum_num_subjects,
            "cum_num_subjects %": cum_num_subjects_pct,
        }
    )

    return stats

