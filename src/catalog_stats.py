from types import SimpleNamespace
import pandas as pd

from common import as_nullable_int, insert
from catalog import to_score_group


def pivot_by_eb_score_group(
    df_catalog: pd.DataFrame,
    index="eb_score_group",
    group_min=0,
    group_max=7,
    recalc_group=False,
    columns="is_eb_catalog",
    also_return_styler=False,
):
    # dynamically compute a score group if needed or explicitly requested
    if index.endswith("_group") and (index not in df_catalog.columns or recalc_group):
        score_colname = index.replace("_group", "")
        col_score_group = [to_score_group(score, max_cap=group_max, min_cap=group_min) for score in df_catalog[score_colname]]
        if index in df_catalog.columns:
            df_catalog.pop(index)
        insert(df_catalog, before_colname=score_colname, colname=index, value=col_score_group)

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
