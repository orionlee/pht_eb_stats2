from types import SimpleNamespace
from typing import Callable

import pandas as pd

from common import as_nullable_int, insert
import catalog
import pht_subj_meta
from catalog import to_score_group


def summary(min_eb_score) -> dict:
    df_catalog = catalog.load_pht_eb_candidate_catalog_from_file()
    num_tics = len(df_catalog)
    num_tics_with_high_certainty = len(df_catalog[df_catalog["eb_score"] >= min_eb_score])

    df_subj = pht_subj_meta.load_subject_meta_table_from_file(include_simulation=False)
    num_subjects = len(df_subj)
    sectors = df_subj["sector"].unique()
    sectors.sort()
    num_sectors = len(sectors)
    start_sector, end_sector = sectors[0], sectors[-1]

    return {
        "Num. of PHT Subjects": num_subjects,
        "Num. of EBs / EB Candidates (TICs)": num_tics,
        "Num. of EBs / EB Candidates with high certainty": num_tics_with_high_certainty,
        "Num. of sectors": num_sectors,
        "First Sector": start_sector,
        "Last Sector": end_sector,
    }


def add_group(df: pd.DataFrame, column: str, group_func: Callable, recalc_if_exists=True):
    """Add a column by grouping the values of the specified column."""
    colname_group = f"{column}_group"
    if not recalc_if_exists and colname_group in df.columns:
        # the group column is already there, don't do it.
        return df
    col_group = [group_func(val) for val in df[column]]
    if colname_group in df.columns:
        df.pop(colname_group)
    return insert(df, before_colname=column, colname=colname_group, value=col_group)


def add_eb_score_group(df: pd.DataFrame, group_min=0, group_max=7, column="eb_score", recalc_if_exists=True):
    # Usage: group the values in eb_score column in the catalog.
    # It can also be used to group columns with similar semantics, e.g., num_eb_votes
    def group_func(val):
        return to_score_group(val, max_cap=group_max, min_cap=group_min)

    return add_group(df, column, group_func, recalc_if_exists=recalc_if_exists)


def pivot_by_eb_score_group(
    df_catalog: pd.DataFrame,
    index="eb_score_group",
    group_min=0,
    group_max=7,
    recalc_group=False,
    calc_totals_pct_col=False,
    columns="is_eb_catalog",
    also_return_styler=False,
):
    # dynamically compute a score group if needed or explicitly requested
    if index.endswith("_group") and (index not in df_catalog.columns or recalc_group):
        score_colname = index.replace("_group", "")
        add_eb_score_group(df_catalog, column=score_colname, group_min=group_min, group_max=group_max, recalc_if_exists=True)

    report = df_catalog.pivot_table(
        index=[index],
        columns=columns,
        values=["tic_id"],
        aggfunc=["count"],
        margins=True,
        margins_name="Totals",
    )
    # print("DBG pivot table columns:", report.columns)

    # sort descending by eb_score_group (in the index)
    report["sort_key"] = report.index == "Totals"  # temporary sort key to make Totals be the last
    report.sort_values(["sort_key", index], ascending=[True, False], inplace=True)
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

    as_nullable_int(report, [col_key_t, col_key_f, col_key_na, col_key_totals])

    col_key_totals = ("count", "tic_id", "Totals")
    col_key_totals_pct = ("count", "tic_id", "Totals %")

    if calc_totals_pct_col:
        num_subjects = report.loc["Totals", col_key_totals]
        report[col_key_totals] / num_subjects
        insert(
            report, before_colname=col_key_t_over_t_f, colname=col_key_totals_pct, value=report[col_key_totals] / num_subjects
        )

    if not also_return_styler:
        return report
    else:
        # tweak output formatting.
        styler = report.style.format(
            {
                # use percentage
                col_key_t_over_t_f: "{:.1%}",
                col_key_totals_pct: "{:.1%}",
            }
        )
        # make caption visually better linked with the pivot table
        style_spec = [
            dict(
                selector="caption",
                props=[
                    ("caption-side", "top"),
                    ("font-weight", "bold"),
                    ("color", "#333"),
                    ("font-size", "110%"),
                    # remove padding-bottom, as the pivot header has plenty of whitespaces already
                    ("padding-bottom", "0"),
                    ("border-bottom", "1px dotted gray"),
                ],
            )
        ]
        styler = styler.set_table_styles(style_spec)

        return report, styler


def estimate_num_ebs_not_in_catalog(df: pd.DataFrame, min_eb_score):
    report, styler = pivot_by_eb_score_group(
        df, group_max=min_eb_score, group_min=min_eb_score - 1, recalc_group=True, also_return_styler=True
    )

    selector_proxy_accuracy = (f"0{min_eb_score}+", ("count", "tic_id", "T/(T+F)"))
    proxy_accuracy = report.loc[selector_proxy_accuracy[0], selector_proxy_accuracy[1]]
    styler = styler.applymap(lambda x: "background: rgba(255, 255, 0, 0.8)", subset=selector_proxy_accuracy)

    selector_num_not_classified = (f"0{min_eb_score}+", ("count", "tic_id", "-"))
    num_not_classified = report.loc[selector_num_not_classified[0], selector_num_not_classified[1]]
    styler = styler.applymap(lambda x: "background: rgba(255, 255, 0, 0.8)", subset=selector_num_not_classified)

    num_eb_not_in_catalog = int(num_not_classified * proxy_accuracy)

    results = SimpleNamespace(
        num_eb_not_in_catalog=num_eb_not_in_catalog,
        num_not_classified=num_not_classified,
        proxy_accuracy=proxy_accuracy,
    )
    return results, report, styler
