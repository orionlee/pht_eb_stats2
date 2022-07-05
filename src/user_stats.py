from types import SimpleNamespace
from typing import Sequence, Union

import numpy as np
import pandas as pd

from common import as_nullable_int, insert, to_csv
import pht_subj_meta
import pht_subj_comments_per_comment
import pht_subj_comments_per_subject

import catalog_stats

Str_Or_Str_Sequence = Union[str, Sequence[str]]


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


def _create_user_eb_stats(df_subject_user_stats: pd.DataFrame = None, also_return_styler=False):
    """Create a table of ordered list of user - num of subjects tagged"""
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
    elif rank <= 5:
        return "003-5"
    elif rank <= 20:
        return "006-20"
    elif rank <= 50:
        return "021-50"
    elif rank <= 100:
        return "051-100"
    else:
        return "101+"


def pivot_by_rank_group(df_subject_user_stats: pd.DataFrame = None, df_user_stats: pd.DataFrame = None, rank_group_func=None):
    if df_subject_user_stats is None:
        df_subject_user_stats = _create_subject_user_stats()
    if df_user_stats is None:
        df_user_stats = _create_user_eb_stats(df_subject_user_stats)
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


def calc_n_save_top_users_cum_contributions(
    df_subject_user_stats: pd.DataFrame = None,
    df_user_stats: pd.DataFrame = None,
    top_n=100,
    dry_run=False,
):
    """Create a table of cumulative contributions by top users."""
    out_path = "../data/users_top_cum_contributions.csv"
    if df_subject_user_stats is None:
        df_subject_user_stats = _create_subject_user_stats()
    if df_user_stats is None:
        df_user_stats = _create_user_eb_stats(df_subject_user_stats)

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

    if not dry_run:
        to_csv(stats, out_path, mode="w")

    return stats


def load_top_users_cum_contributions_from_file(csv_path="../data/users_top_cum_contributions.csv"):
    df = pd.read_csv(csv_path)
    return df


RANK_GROUP_SPECS = [
    # min, max, group_name_body
    (1, 1, "001"),
    (2, 2, "002"),
    (3, 5, "003-5"),
    (6, 20, "006-20"),
    (21, 50, "021-50"),
    (51, 100, "051-100"),
    (101, np.inf, "101+"),
]

RANK_GROUP_NAMES = [spec[2] for spec in RANK_GROUP_SPECS]


def calc_subject_user_rank_groups(df_subj_users: pd.DataFrame = None, rank_groups_specs=RANK_GROUP_SPECS):
    """For each subject, indicate if the following user_rank_group has tagged it as eclipsing binary.

    The groups are: 001, 002, 003-5, 006-20, 021-50, 051-100, 100+
    """

    if df_subj_users is None:
        df_subj_users = _create_subject_user_stats()

    # OPEN: should I filter by has_eb_tags == 1 (just tagged some eb-like tags), or
    # by eb_score > 0 (which would filter those that also tagged subjects with transit tags)?
    df_subj_users = df_subj_users[df_subj_users["has_eb_tags"] == 1].reset_index(drop=True)
    df_user_ranks = _create_user_eb_stats(df_subj_users)

    rank_subj_ids_df_list = []
    for a_rank_group_specs in rank_groups_specs:
        # rank_min, rank_max, rank_group_name = 3, 5, "By_User_Ranks_003-5"
        rank_min, rank_max, rank_group_name = a_rank_group_specs
        rank_group_name = f"By_User_Ranks_{rank_group_name}"

        rank_user_ids = df_user_ranks[(rank_min <= df_user_ranks["user_rank"]) & (df_user_ranks["user_rank"] <= rank_max)][
            "user_id"
        ]
        rank_subj_ids = df_subj_users[df_subj_users["user_id"].isin(rank_user_ids)]["subject_id"].unique()

        rank_subj_ids_series = pd.DataFrame(
            pd.Series(data=np.full_like(rank_subj_ids, True), index=rank_subj_ids, name=rank_group_name, dtype="boolean")
        )
        rank_subj_ids_df_list.append(rank_subj_ids_series)

    # prepare to join with users of a particular rank groups
    df = pd.DataFrame({"subject_id": df_subj_users["subject_id"].unique()})
    df = df.set_index("subject_id", drop=False)
    df = df.join(rank_subj_ids_df_list, how="left")
    df.reset_index(drop=True, inplace=True)  # drop the no-longer useful subject_id index
    return df


def calc_n_save_tic_user_rank_groups(df_subj_users: pd.DataFrame = None, dry_run=False):
    """For each TIC, indicate the user rank group(s) that have tagged it as an eclipsing binary.

    It is an aggregation of the per-subject stats in `calc_subject_user_rank_groups()`
    to per-TIC stats.
    """
    out_path = "../data/tic_eb_rank_groups.csv"

    df_subj_rank_groups = calc_subject_user_rank_groups(df_subj_users)

    df_tic_subj = pht_subj_meta.load_subject_meta_table_from_file()[["tic_id", "subject_id"]]

    df = df_tic_subj.set_index("subject_id").join(df_subj_rank_groups.set_index("subject_id"), how="left")
    df.sort_values("tic_id", inplace=True)

    # for each TIC, do a logical OR on the By_User_Ranks column
    # e.g., if any of the subject of a TIC is in User_Ranks_003-5,
    # the TIC is True for By_User_Ranks_003-5
    res = df.groupby("tic_id", as_index=False).any()

    if not dry_run:
        to_csv(res, out_path, mode="w")

    return res


def load_tic_user_rank_groups_table_from_file(csv_path="../data/tic_eb_rank_groups.csv"):
    df = pd.read_csv(csv_path)
    return df


def filter_pht_eb_catalog_by_user_ranks(df_catalog: pd.DataFrame, rank_groups: Str_Or_Str_Sequence = "003-005"):
    if isinstance(rank_groups, str):
        rank_groups = [rank_groups]

    df_filter = load_tic_user_rank_groups_table_from_file()
    for rank_group in rank_groups:
        df_filter = df_filter[df_filter[f"By_User_Ranks_{rank_group}"]]

    return df_catalog[df_catalog["tic_id"].isin(df_filter["tic_id"])].reset_index(drop=True)


def calc_proxy_accuracy_stats_by_rank_group(df_catalog, max_eb_score=7, rank_group_names=RANK_GROUP_NAMES):
    selector_proxy_accuracy_max = (f"0{max_eb_score}+", ("count", "tic_id", "T/(T+F)"))
    selector_proxy_accuracy_all = ("Totals", ("count", "tic_id", "T/(T+F)"))

    all_report, all_styler = catalog_stats.pivot_by_eb_score_group(
        df_catalog, group_min=1, group_max=max_eb_score, recalc_group=True, also_return_styler=True
    )
    all_styler = all_styler.set_caption("All Users")

    rank_group_reports, rank_group_stylers = [], []
    rank_group_proxy_accuracy_max, rank_group_proxy_accuracy_all = [], []

    for rank_groups in rank_group_names:
        df_of_ranks = filter_pht_eb_catalog_by_user_ranks(df_catalog, rank_groups=rank_groups)
        report, styler = catalog_stats.pivot_by_eb_score_group(df_of_ranks, also_return_styler=True)
        styler = styler.set_caption(f"Users in ranks {rank_groups}")
        rank_group_reports.append(report)
        rank_group_stylers.append(styler)
        rank_group_proxy_accuracy_max.append(report.loc[selector_proxy_accuracy_max])
        rank_group_proxy_accuracy_all.append(report.loc[selector_proxy_accuracy_all])

    df_rank_group_proxy_accuracy = pd.DataFrame(
        {
            "rank_group": rank_group_names,
            f"proxy_accuracy, eb_score >= {max_eb_score}": rank_group_proxy_accuracy_max,
            "proxy_accuracy, all": rank_group_proxy_accuracy_all,
        }
    )

    return SimpleNamespace(
        df_rank_group_proxy_accuracy=df_rank_group_proxy_accuracy,
        all_report=all_report,
        all_styler=all_styler,
        rank_group_reports=rank_group_reports,
        rank_group_stylers=rank_group_stylers,
    )


if __name__ == "__main__":
    calc_n_save_top_users_cum_contributions()
    calc_n_save_tic_user_rank_groups()
