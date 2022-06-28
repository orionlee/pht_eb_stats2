import argparse
import pandas as pd

import tqdm

from common import *
import pht_subj_comments_per_subject
import simbad_meta
import tic_pht_stats


def combine_and_save_pht_eb_candidate_catalog(dry_run=False, dry_run_size=1000):
    # The resulting catalog would require additional filtering to exclude
    # those that are likely to be false positives
    out_path = "../data/catalog_pht_eb_candidates.csv"

    df_pht = tic_pht_stats.load_tic_pht_stats_table_from_file()
    df_simbad = simbad_meta.load_simbad_is_eb_table_from_file()

    if dry_run and dry_run_size is not None:
        df_pht = df_pht[:dry_run_size]
        df_simbad = df_simbad[:dry_run_size]

    # column-merge the 2 tables by tic_id
    df_pht.set_index("tic_id", drop=False, inplace=True)
    df_simbad.set_index("TIC_ID", drop=True, inplace=True)  # drop TIC_ID column, as it will be a duplicate in the result
    df = pd.concat([df_pht, df_simbad], join="inner", axis=1)

    # Note: this will be updated when we combine additional catalog
    col_is_eb_catalog = df["Is_EB_SIMBAD"].copy()

    idx_tic_id = df.columns.get_loc("eb_score")
    df.insert(idx_tic_id, "is_eb_catalog", col_is_eb_catalog)

    if not dry_run:
        to_csv(df, out_path, mode="w")

    return df


def load_pht_eb_candidate_catalog_from_file(csv_path="../data/catalog_pht_eb_candidates.csv"):
    df = pd.read_csv(csv_path)
    return df


def reprocess_all_mapping_and_save_pht_eb_candidate_catalog():
    with tqdm(total=5) as pbar:

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

        pbar.write("Overall catalog")
        # Merge to produce overall catalog
        combine_and_save_pht_eb_candidate_catalog()
        pbar.update(1)
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Produce PHT EB Candidate catalog table.')
    parser.add_argument(
        "--remap",
        action="store_true",
        help="Apply various mapping logic to produce the catalog, assuming the required data has been fetched."
        )
    parser.add_argument(
        "--combine",
        action="store_true",
        help="Combine the underling tables to the catalog, no mapping logic is reapplied"
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
