import datetime as dt
import pandas as pd
import streamlit as st


def get_date_range(
    df: pd.DataFrame,
    quick_range: str,
    date_col_priority=["enrollmentDate", "event_date"],
):
    """
    Returns start and end dates based on the quick_range.
    Automatically chooses the first available date column from date_col_priority.
    """
    today = dt.date.today()
    # Find which date column exists
    date_col = next((col for col in date_col_priority if col in df.columns), None)

    # Quick ranges
    if quick_range == "Today":
        return today, today
    elif quick_range == "This Week":
        start = today - dt.timedelta(days=today.weekday())
        return start, start + dt.timedelta(days=6)
    elif quick_range == "Last Week":
        end = today - dt.timedelta(days=today.weekday() + 1)
        return end - dt.timedelta(days=6), end
    elif quick_range == "This Month":
        start = today.replace(day=1)
        next_month = (today.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
        return start, next_month - dt.timedelta(days=1)
    elif quick_range == "Last Month":
        first_this_month = today.replace(day=1)
        last_month_end = first_this_month - dt.timedelta(days=1)
        return last_month_end.replace(day=1), last_month_end
    elif quick_range == "This Year":
        return today.replace(month=1, day=1), today
    elif quick_range == "Last Year":
        return (
            today.replace(year=today.year - 1, month=1, day=1),
            today.replace(year=today.year - 1, month=12, day=31),
        )
    else:  # Custom
        if date_col and not df.empty:
            start_default = df[date_col].min().date()
            end_default = df[date_col].max().date()
        else:
            start_default = today.replace(month=1, day=1)
            end_default = today

        start_date = st.date_input("Start Date", value=start_default)
        end_date = st.date_input("End Date", value=end_default)
        return start_date, end_date


def get_available_aggregations(start_date, end_date):
    """
    Returns available aggregation levels based on date range duration.
    """
    delta = (end_date - start_date).days

    if delta < 7:
        return ["Daily"]
    elif delta < 30:
        return ["Daily", "Weekly"]
    elif delta < 90:
        return ["Daily", "Weekly", "Monthly"]
    elif delta < 365:
        return ["Daily", "Weekly", "Monthly", "Quarterly"]
    else:
        return ["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"]


def format_weekly_label(period_start):
    """
    Format weekly period as human-readable range (e.g., "2025 Jan 1-7")
    """
    period_end = period_start + dt.timedelta(days=6)

    # Same month
    if period_start.month == period_end.month:
        return f"{period_start.year} {period_start.strftime('%b')} {period_start.day}-{period_end.day}"
    # Different months
    else:
        return f"{period_start.year} {period_start.strftime('%b')} {period_start.day}-{period_end.strftime('%b')} {period_end.day}"


def format_quarterly_label(quarter_str):
    """
    Format quarterly period as human-readable (e.g., "2025 Q1 (Jan-Mar)")
    """
    year, quarter = quarter_str.split("Q")
    quarter = int(quarter)

    month_ranges = {1: "Jan-Mar", 2: "Apr-Jun", 3: "Jul-Sep", 4: "Oct-Dec"}

    return f"{year} Q{quarter} ({month_ranges[quarter]})"


def assign_period(df: pd.DataFrame, date_col: str, period_label: str):
    """
    Adds a 'period' column to the DataFrame based on the selected aggregation.
    Returns human-readable period labels.
    """
    if df.empty or date_col not in df.columns:
        return df

    if period_label == "Daily":
        df["period"] = df[date_col].dt.date.astype(str)

    elif period_label == "Weekly":
        # Get ISO week but format as readable range
        df["period"] = df[date_col].apply(
            lambda x: format_weekly_label(x - dt.timedelta(days=x.weekday()))
        )

    elif period_label == "Monthly":
        df["period"] = df[date_col].dt.strftime("%Y %b")

    elif period_label == "Quarterly":
        # First get quarter code, then format
        quarter_codes = df[date_col].dt.to_period("Q").astype(str)
        df["period"] = quarter_codes.apply(format_quarterly_label)

    else:  # Annual
        df["period"] = df[date_col].dt.to_period("Y").astype(str)

    return df
