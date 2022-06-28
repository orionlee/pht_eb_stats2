import pandas as pd

import pht_subj_meta
import pht_subj_comments_per_subject


from common import to_csv


def calc_and_save_pht_stats(dry_run=False, dry_run_size=1000, also_return_df_subjects=False):
    out_path = "../data/tic_pht_stats.csv"

    # include simulation because they are also present in df_comment_stats
    # the 2 data frames can then be easily column-merged together.
    df_meta = pht_subj_meta.load_subject_meta_table_from_file(include_simulation=False)
    df_comment_stats = pht_subj_comments_per_subject.load_pht_subj_comment_summary_table_from_file()

    if dry_run and dry_run_size is not None:
        df_meta = df_meta[:dry_run_size]
        df_comment_stats = df_comment_stats[:dry_run_size]

    # column-merge the 2 tables by subject_id
    # - use inner join to eliminate the simulation subjects, already excluded in df_meta, but not df_comment_stats.
    # - remove the duplicate subject_id column
    df_meta.set_index("subject_id", drop=False, inplace=True)
    df_comment_stats.set_index("subject_id", drop=True, inplace=True)
    df_subjects = pd.concat([df_meta, df_comment_stats], join="inner", axis=1)

    # a sanity check
    if not (df_meta["subject_id"] == df_subjects["subject_id"]).all():
        print("WARN Some subjects from PHT subject meta table (except simulation) are unexpectedly absent in the merge results")

    # For each TIC, get the subject with the best eb_score (with most recent sector as the tiebreaker)
    df_subjects = df_subjects.sort_values(["tic_id", "eb_score", "sector"], ascending=[True, False, False])
    df_tics_main = df_subjects.groupby("tic_id").first()

    df_tics_extras = df_subjects.groupby("tic_id").agg(
        tic_id=("tic_id", "first"),  # the df_tics_main does not have the tic_id column, so we get it here
        num_subjects=("subject_id", "count"),
        # include list of sectors / subject_ids for convenience
        sectors=("sector", lambda x: ",".join(list(x.astype(str)))),
        subject_ids=("subject_id", lambda x: ",".join(list(x.astype(str)))),
    )

    df_tics = pd.concat([df_tics_main, df_tics_extras], axis=1)

    # rename/rearrange the columns to make the result easier for inspection
    df_tics = df_tics.rename(columns={
        "subject_id": "best_subject_id",
        "sector": "best_subject_sector",
        "img_id": "best_subject_img_id",
        })

    # tic_id column is from df_tics_extras table
    # move it to the first
    idx_tic_id = df_tics.columns.get_loc("tic_id")
    # move tic_id to the first
    df_tics.insert(0, "tic_id", df_tics.pop("tic_id"))
    # move best_subject_img_id (it's not too useful)
    df_tics.insert(idx_tic_id, "best_subject_img_id", df_tics.pop("best_subject_img_id"))

    if not dry_run:
        to_csv(df_tics, out_path, mode="w")

    if also_return_df_subjects:
        return df_tics, df_subjects
    else:
        return df_tics


def load_tic_pht_stats_table_from_file(csv_path="../data/tic_pht_stats.csv"):
    df = pd.read_csv(csv_path)
    return df


if __name__ == "main":
    calc_and_save_pht_stats()
