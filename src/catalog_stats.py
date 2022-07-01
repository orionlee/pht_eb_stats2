from common import insert
from catalog import to_score_group


# A commonly used column subset in the catalog
COLUMNS_COMMON = [
    "tic_id",
    "best_subject_id",
    "is_eb_catalog",
    "eb_score",
    "SIMBAD_MAIN_ID",
    "SIMBAD_OTYPES",
    "SIMBAD_Is_EB",
    "VSX_Name",
    "VSX_Type",
    "VSX_Is_EB",
    "VSX_Period",
]


def pivot_by_eb_score_group(df_catalog, row="eb_score_group", group_max=7, columns="is_eb_catalog", also_return_styler=False):
    # dynamically compute a score group if needed
    if row not in df_catalog.columns and row.endswith("_group"):
        score_colname = row.replace("_group", "")
        col_score_group = [to_score_group(score, group_max) for score in df_catalog[score_colname]]
        insert(df_catalog, before_colname=score_colname, colname=row, value=col_score_group)

    report = df_catalog.pivot_table(
        index=[row],
        columns=columns,
        values=["tic_id"],
        aggfunc=["count"],
        margins=True,
        margins_name="Totals",
    )
    # print("DBG pivot table columns:", report.columns)

    # sort descending by eb_score_group (in the index)
    report["sort_key"] = report.index == "Totals"  # temporary sort key to make Totals be the last
    report.sort_values(["sort_key", row], ascending=[True, False], inplace=True)
    report.drop("sort_key", axis=1, inplace=True)

    col_key_t = ("count", "tic_id", "T")
    col_key_f = ("count", "tic_id", "F")
    col_key_na = ("count", "tic_id", "-")
    col_key_totals = ("count", "tic_id", "Totals")

    column_order = [col_key_t, col_key_f, col_key_na, col_key_totals]
    report = report.reindex(columns=column_order)

    # Add column of T / (T+F), a proxy of tagging accuracy
    col_totals_t_f = report[[col_key_t, col_key_f]].sum(axis=1)  # .sum() treats N/A as 0
    col_t_over_t_f = report[col_key_t] / col_totals_t_f
    col_key_t_over_t_f = ("count", "tic_id", "T/(T+F)")
    report[col_key_t_over_t_f] = col_t_over_t_f

    report[col_key_t] = report[col_key_t].astype("Int64")
    report[col_key_f] = report[col_key_f].astype("Int64")
    report[col_key_na] = report[col_key_na].astype("Int64")
    report[col_key_totals] = report[col_key_totals].astype("Int64")

    if not also_return_styler:
        return report
    else:
        # tweak output formatting.
        styler = report.style.format(
            {
                col_key_t_over_t_f: "{:.2%}",  # use percentage
            }
        )
        return report, styler
