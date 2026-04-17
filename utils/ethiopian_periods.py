from datetime import date, timedelta

import pandas as pd


ETHIOPIAN_MONTH_NAMES = {
    1: "Meskerem",
    2: "Tikimt",
    3: "Hidar",
    4: "Tahsas",
    5: "Tir",
    6: "Yekatit",
    7: "Megabit",
    8: "Miazia",
    9: "Ginbot",
    10: "Sene",
    11: "Hamle",
    12: "Nehase",
    13: "Pagume",
}


def _is_valid_ethiopian_yearmonth(yearmonth):
    try:
        value = int(yearmonth)
    except (TypeError, ValueError):
        return False

    month = value % 100
    return 1 <= month <= 13


def is_ethiopian_leap_year(year):
    return int(year) % 4 == 3


def get_ethiopian_new_year_start(ec_year):
    ec_year = int(ec_year)
    gc_year = ec_year + 7
    start_day = 12 if (ec_year - 1) % 4 == 3 else 11
    return date(gc_year, 9, start_day)


def get_ethiopian_month_range(yearmonth):
    yearmonth = int(yearmonth)
    ec_year = yearmonth // 100
    ec_month = yearmonth % 100

    if not 1 <= ec_month <= 13:
        raise ValueError(f"Invalid Ethiopian month in period: {yearmonth}")

    gc_start = get_ethiopian_new_year_start(ec_year) + timedelta(days=(ec_month - 1) * 30)
    if ec_month <= 12:
        gc_end = gc_start + timedelta(days=29)
    else:
        gc_end = gc_start + timedelta(days=5 if is_ethiopian_leap_year(ec_year) else 4)

    return pd.Timestamp(gc_start), pd.Timestamp(gc_end)


def build_ethiopian_period_table(yearmonths):
    rows = []
    unique_values = sorted(
        {int(v) for v in pd.to_numeric(list(yearmonths), errors="coerce") if not pd.isna(v)}
    )

    for yearmonth in unique_values:
        if not _is_valid_ethiopian_yearmonth(yearmonth):
            continue

        ec_year = yearmonth // 100
        ec_month = yearmonth % 100
        gc_start, gc_end = get_ethiopian_month_range(yearmonth)
        rows.append(
            {
                "yearmonth": yearmonth,
                "year": ec_year,
                "ec_year": ec_year,
                "ec_month": ec_month,
                "ec_month_name": ETHIOPIAN_MONTH_NAMES.get(ec_month, str(ec_month)),
                "ec_label": f"{ETHIOPIAN_MONTH_NAMES.get(ec_month, str(ec_month))} {ec_year}",
                "gc_start": gc_start.normalize(),
                "gc_end": gc_end.normalize(),
            }
        )

    return pd.DataFrame(rows)


def add_ethiopian_period_metadata(df, yearmonth_col="yearmonth"):
    if df is None or df.empty or yearmonth_col not in df.columns:
        return df

    period_df = build_ethiopian_period_table(df[yearmonth_col].dropna().unique().tolist())
    if period_df.empty:
        return df

    return df.merge(period_df, how="left", left_on=yearmonth_col, right_on="yearmonth")


def filter_periods_by_overlap(period_df, start_date=None, end_date=None):
    if period_df is None or period_df.empty:
        return pd.DataFrame(columns=["yearmonth", "year", "ec_year", "ec_month", "ec_month_name", "ec_label", "gc_start", "gc_end"])

    working = period_df.copy()
    if start_date is not None:
        start_ts = pd.Timestamp(start_date).normalize()
        working = working[working["gc_end"] >= start_ts]
    if end_date is not None:
        end_ts = pd.Timestamp(end_date).normalize()
        working = working[working["gc_start"] <= end_ts]

    return working.sort_values(["gc_start", "yearmonth"]).reset_index(drop=True)


def map_gregorian_dates_to_ethiopian_yearmonths(dates, available_yearmonths):
    period_df = build_ethiopian_period_table(available_yearmonths)
    if period_df.empty:
        return pd.Series(pd.array([pd.NA] * len(dates), dtype="Int64"), index=pd.Index(range(len(dates))))

    date_series = pd.Series(dates).copy()
    mapped = pd.Series(pd.array([pd.NA] * len(date_series), dtype="Int64"), index=date_series.index)
    normalized_dates = pd.to_datetime(date_series, errors="coerce").dt.normalize()

    for row in period_df.itertuples(index=False):
        mask = normalized_dates.between(row.gc_start, row.gc_end, inclusive="both")
        mapped.loc[mask] = int(row.yearmonth)

    return mapped


def format_gregorian_range_label(start_date, end_date):
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    if start_ts.year == end_ts.year:
        if start_ts.month == end_ts.month:
            return f"{start_ts:%b} {start_ts.day}-{end_ts.day}, {start_ts.year}"
        return f"{start_ts:%b} {start_ts.day}-{end_ts:%b} {end_ts.day}, {start_ts.year}"

    return f"{start_ts:%b} {start_ts.day}, {start_ts.year}-{end_ts:%b} {end_ts.day}, {end_ts.year}"


def build_period_definitions_from_denominator(den_long, period_label, start_date=None, end_date=None):
    if den_long is None or den_long.empty or "yearmonth" not in den_long.columns:
        return []

    period_df = build_ethiopian_period_table(den_long["yearmonth"].dropna().unique().tolist())
    period_df = filter_periods_by_overlap(period_df, start_date=start_date, end_date=end_date)
    if period_df.empty:
        return []

    if period_label == "Yearly":
        period_defs = []
        for ec_year, group in period_df.groupby("ec_year", sort=True):
            yms = group["yearmonth"].astype(int).tolist()
            gc_start = group["gc_start"].min()
            gc_end = group["gc_end"].max()
            period_defs.append(
                {
                    "period_display": format_gregorian_range_label(gc_start, gc_end),
                    "period_sort": gc_start,
                    "yearmonths": yms,
                    "year": int(ec_year),
                }
            )
        return sorted(period_defs, key=lambda item: item["period_sort"])

    period_defs = []
    for row in period_df.itertuples(index=False):
        period_defs.append(
            {
                "period_display": format_gregorian_range_label(row.gc_start, row.gc_end),
                "period_sort": row.gc_start,
                "yearmonths": [int(row.yearmonth)],
                "year": int(row.ec_year),
                "gc_start": row.gc_start,
                "gc_end": row.gc_end,
            }
        )
    return period_defs
