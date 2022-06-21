import pandas as pd
import pht_subj_meta as pht_sm


def save_tic_ids(**kwargs):
    out_path = "../data/pht_tic_ids.csv"

    df = pht_sm.get_subject_meta_table(**kwargs)
    res = df["tic_id"].unique()
    res.sort()
    res = res[res > 0]  # filter those without TIC IDs (-1 in the csv)
    res

    df_out = pd.DataFrame(dict(tic_id=res))
    df_out.to_csv(out_path, index=False)  # ignore the row name (just the row number)
    # with open(out_path, "w") as f:
    #     # csv_writer = csv.writer(f, dialect="unix")
    #     # csv_writer.writerows(res)
    return df_out

