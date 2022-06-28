import pandas as pd

from common import *
from tic_pht_stats import load_tic_pht_stats_table_from_file
from simbad_meta import load_simbad_is_eb_table_from_file


def combine_and_save_pht_eb_candidate_catalog(dry_run=False, dry_run_size=1000):
    # The resulting catalog would require additional filtering to exclude
    # those that are likely to be false positives
    out_path = "../data/catalog_pht_eb_candidates.csv"

    df_pht = load_tic_pht_stats_table_from_file()
    df_simbad = load_simbad_is_eb_table_from_file()

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


if __name__ == "__main__":
    combine_and_save_pht_eb_candidate_catalog()