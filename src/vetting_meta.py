#
# Access individual vetting results
#

import pandas as pd


def _to_disposition_group(disposition):
    if disposition in ["Candidate", "Candidate:"]:
        return "Candidate"
    elif disposition in ["FP"]:
        return "FP"
    elif disposition in ["Candidate?", "Unsure"]:
        return "Unsure"
    else:
        return disposition  # should not happen


def load_vetting_statuses_table(
    csv_path="../data/vetting_statuses.csv",
    add_disposition_group=False,
    add_tesseb_url_col=False,
    add_vetting_doc_url_col=False,
    ):
    df = pd.read_csv(
        csv_path,
        keep_default_na=False,  # to treat empty cell as empty string
        )

    if add_disposition_group:
        df["Disposition_Group"] = [_to_disposition_group(d) for d in df["Disposition"]]

    if add_tesseb_url_col:
        df["TESSEB_URL"] = df["TIC"]  # a placeholder for UI to create links to TESSEB

    if add_vetting_doc_url_col:
        df["Vetting_URL"] = df["TIC"] # a placeholder for UI to create links to TESSEB

    return df
