import argparse

import numpy as np
import pandas as pd

from tqdm import tqdm

from common import insert, to_csv, prefix_columns
import pht_subj_comments_per_subject
import simbad_meta
import tic_pht_stats
import vsx_meta


def calc_is_eb_combined(*val_list_list):
    """Combine the Is_EB results from multiple catalogs to produce an aggregate one.

    The logic is similar to the "or" logic in Kleene three-value logic
    (N/A is "-", i.e., no data for a catalog).
    The one difference is that `False | NA` is considered `False` in our case, while it is
    `NA` in three-value logic.

    E.g., for a TIC, if VSX Is_EB is "F", and SIMBAD Is_EB is "-" (N/A, no data), we
    consider the final result is False. The false response trumps no data.
    """
    num_cols = len(val_list_list)

    # if input is pandas series, convert it to numpy array
    # to make the 0-based indexing work
    val_list_list = [v for v in val_list_list]  # convert the tuple to list to make assignment work
    for j in range(num_cols):
        if isinstance(val_list_list[j], pd.Series):
            val_list_list[j] = val_list_list[j].to_numpy()

    num_rows = len(val_list_list[0])
    res = np.full(num_rows, "-")
    for i in range(num_rows):
        row_vals = [val_list_list[j][i] for j in range(num_cols)]
        if "T" in row_vals:
            cur_is_eb = "T"
        elif "F" in row_vals:
            cur_is_eb = "F"
        else:  # result is "-" (no-data) only if all catalogs report so
            cur_is_eb = "-"
        res[i] = cur_is_eb

    return res


def test_calc_is_eb_combined():
    # TODO: setup unit tests
    val1 = pd.Series(["T", "T", "T", "F", "F", "-"])
    val2 = pd.Series(["T", "F", "-", "F", "-", "-"])
    val3 = calc_is_eb_combined(val1, val2)
    assert (val3 == np.array(["T", "T", "T", "F", "F", "-"])).all()


test_calc_is_eb_combined()


def combine_and_save_pht_eb_candidate_catalog(dry_run=False, dry_run_size=1000):
    # The resulting catalog would require additional filtering to exclude
    # those that are likely to be false positives
    out_path = "../data/catalog_pht_eb_candidates.csv"

    df_pht = tic_pht_stats.load_tic_pht_stats_table_from_file()
    df_simbad = simbad_meta.load_simbad_is_eb_table_from_file()
    df_vsx = vsx_meta.load_vsx_is_eb_table_from_file()

    if dry_run and dry_run_size is not None:
        df_pht = df_pht[:dry_run_size]
        df_simbad = df_simbad[:dry_run_size]
        df_vsx = df_vsx[:dry_run_size]

    # column-merge the tables by tic_id
    df_pht.set_index("tic_id", drop=False, inplace=True)
    df_simbad.set_index("TIC_ID", drop=True, inplace=True)  # drop TIC_ID column, as it will be a duplicate in the result
    prefix_columns(df_simbad, "SIMBAD", inplace=True)
    df_vsx = vsx_meta.load_vsx_is_eb_table_from_file()
    df_vsx.set_index("TIC_ID", drop=True, inplace=True)  # drop TIC_ID column, as it will be a duplicate in the result
    prefix_columns(df_vsx, "VSX", inplace=True)
    df = pd.concat([df_pht, df_simbad, df_vsx], join="outer", axis=1)

    # Misc. type fixing after concat()
    df["VSX_OID"] = df["VSX_OID"].astype("Int64")  # some cells is NA, so convert it to Nullable integer
    df["VSX_V"] = df["VSX_V"].astype("Int64")  # some cells is NA, so convert it to Nullable integer

    # after the table join, some Is_EB values would be NA,
    # we backfill it with the preferred "-", the preferred value that indicates no data.
    df["SIMBAD_Is_EB"] = df["SIMBAD_Is_EB"].fillna("-")
    df["VSX_Is_EB"] = df["VSX_Is_EB"].fillna("-")

    # Note: this will be updated when we combine additional catalog
    col_is_eb_catalog = calc_is_eb_combined(df["SIMBAD_Is_EB"], df["VSX_Is_EB"])

    insert(df, before_colname="eb_score", colname="is_eb_catalog", value=col_is_eb_catalog)

    if not dry_run:
        to_csv(df, out_path, mode="w")

    return df


def load_pht_eb_candidate_catalog_from_file(csv_path="../data/catalog_pht_eb_candidates.csv"):
    df = pd.read_csv(csv_path)
    return df


def reprocess_all_mapping_and_save_pht_eb_candidate_catalog():
    with tqdm(total=7) as pbar:

        pbar.set_description("Reprocessing all mapping to produce the catalog")

        pbar.write("PHT per-subject summary")
        # , affected by
        # - tag mapping table: `../data/pht_tag_map.csv`
        # - the eb vote count logic
        pht_subj_comments_per_subject.save_and_summarize_of_all_subjects()
        pbar.update(1)

        pbar.write("Per-TIC PHT stats")
        tic_pht_stats.calc_and_save_pht_stats()
        pbar.update(1)

        pbar.write("SIMBAD metadata")
        # , affected by
        # - crossmatch logic (primarily on Match_score calculation)
        simbad_meta.combine_and_save_simbad_meta_by_tics_and_xmatch()
        pbar.update(1)

        pbar.write("SIMBAD Is_EB table")
        # , affected by
        # - OTYPE mapping table: `../data/simbad_typemap.csv`
        simbad_meta.map_and_save_simbad_otypes_of_all()
        pbar.update(1)

        pbar.write("VSX metadata")
        # , affected by
        # - crossmatch logic (primarily on Match_score calculation)
        vsx_meta.find_and_save_vsx_best_xmatch_meta(min_score_to_include=0)
        pbar.update(1)

        pbar.write("VSX Is_EB table")
        # , affected by
        # - VSX Type mapping table: `../data/auxillary/vsx_vartype_map.csv`
        vsx_meta.map_and_save_vsx_is_eb_of_all()
        pbar.update(1)

        pbar.write("Overall catalog")
        # Merge to produce overall catalog
        combine_and_save_pht_eb_candidate_catalog()
        pbar.update(1)
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Produce PHT EB Candidate catalog table.")
    parser.add_argument(
        "--remap",
        action="store_true",
        help="Apply various mapping logic to produce the catalog, assuming the required data has been fetched.",
    )
    parser.add_argument(
        "--combine", action="store_true", help="Combine the underling tables to the catalog, no mapping logic is reapplied"
    )
    args = parser.parse_args()
    # print(args)

    if args.remap:
        reprocess_all_mapping_and_save_pht_eb_candidate_catalog()
    elif args.combine:
        combine_and_save_pht_eb_candidate_catalog()
    else:
        print("At least one argument must be specified.")
        parser.print_help()
        parser.exit()
