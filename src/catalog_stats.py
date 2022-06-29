def _to_score_group(val, max_cap):
    if val >= max_cap:
        res = f"0{max_cap}+"
    elif val <= 0:
        res = "00-"
    else:
        res = f"{val:02d}"
    return res


def pivot_by_eb_score_group(df_catalog, also_return_styler=False):

    df = df_catalog.copy(deep=False)

    # group eb_score to a smaller set, so that the extreme ones (negatives, larger eb score)
    # are grouped together
    df["eb_score_group"] = [_to_score_group(score, 7) for score in df["eb_score"]]

    report = df.pivot_table(
        index=["eb_score_group"],
        columns=["is_eb_catalog"],
        values=["tic_id"],
        aggfunc=["count"],
        margins=True,
        margins_name="Totals"
        )
    # print("DBG pivot table columns:", report.columns)

    # sort descending by eb_score_group (in the index)
    report["sort_key"] = report.index == "Totals"  # temporary sort key to make Totals be the last
    report.sort_values(["sort_key", "eb_score_group"], ascending=[True, False], inplace=True)
    report.drop("sort_key", axis=1, inplace=True)

    col_key_t = ('count', 'tic_id', 'T')
    col_key_f = ('count', 'tic_id', 'F')
    col_key_na = ('count', 'tic_id', '-')
    col_key_totals = ('count', 'tic_id', 'Totals')

    column_order = [ col_key_t, col_key_f, col_key_na, col_key_totals]
    report = report.reindex(columns=column_order)

    # Add column of T / (T+F)
    col_totals_t_f = report[[col_key_t, col_key_f]].sum(axis=1)  # .sum() treats N/A as 0
    col_t_over_t_f = report[col_key_t] / col_totals_t_f
    col_key_t_over_t_f = ("count", "tic_id", "T/(T+F)")
    report[col_key_t_over_t_f] = col_t_over_t_f

    if not also_return_styler:
        return report
    else:
        # tweak output formatting.
        styler = report.style.format({
            col_key_t_over_t_f: "{:.2%}",  # use percentage
        })
        return report, styler

