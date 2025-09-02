import datetime as dt
import pandas as pd
import streamlit as st

def get_date_range(df: pd.DataFrame, quick_range: str, date_col_priority=["enrollmentDate", "event_date"]):
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
        return (today.replace(year=today.year - 1, month=1, day=1),
                today.replace(year=today.year - 1, month=12, day=31))
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


def assign_period(df: pd.DataFrame, date_col: str, period_label: str):
    """
    Adds a 'period' column to the DataFrame based on the selected aggregation.
    """
    if df.empty or date_col not in df.columns:
        return df

    if period_label == "Daily":
        df["period"] = df[date_col].dt.date
    elif period_label == "Monthly":
        df["period"] = df[date_col].dt.to_period("M").astype(str)
    elif period_label == "Quarterly":
        df["period"] = df[date_col].dt.to_period("Q").astype(str)
    else:  # Annual
        df["period"] = df[date_col].dt.to_period("Y").astype(str)

    return df
