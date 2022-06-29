import pandas as pd
import pht_subj_meta as pht_sm


def save_tic_ids(**kwargs):
    out_path = "../data/pht_tic_ids.csv"

    df = pht_sm.load_subject_meta_table_from_file(include_simulation=False, **kwargs)
    res = df["tic_id"].unique()
    res.sort()

    df_out = pd.DataFrame(dict(tic_id=res))
    df_out.to_csv(out_path, index=False)  # ignore the row name (just the row number)
    return df_out
