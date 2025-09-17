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

    else:  # Custom range selection
        if date_col and not df.empty:
            start_default = df[date_col].min().date()
            end_default = df[date_col].max().date()
        else:
            start_default = today.replace(month=1, day=1)
            end_default = today

        start_date = st.date_input("Start Date", value=start_default)
        end_date = st.date_input("End Date", value=end_default)

        if end_date < start_date:
            st.error("⚠️ End date cannot be earlier than start date")

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
    Format weekly period as human-readable with week number and range (e.g., "W1 (Jan 1-7, 2025)")
    """
    period_end = period_start + dt.timedelta(days=6)

    # Calculate week number within the year
    week_number = period_start.isocalendar()[1]

    # Format the date range more clearly
    start_str = period_start.strftime("%b %d")
    end_str = period_end.strftime("%b %d, %Y")

    # Handle same year case
    if period_start.year == period_end.year:
        return f"Week {week_number} ({start_str} - {end_str})"
    else:
        # Different year (rare but possible for year-end weeks)
        start_str_full = period_start.strftime("%b %d, %Y")
        return f"Week {week_number} ({start_str_full} - {end_str})"


def format_quarterly_label(quarter_str):
    """
    Format quarterly period as human-readable with quarter number (e.g., "Q1 (Jan-Mar 2025)")
    """
    year, quarter = quarter_str.split("Q")
    quarter = int(quarter)

    month_ranges = {1: "Jan-Mar", 2: "Apr-Jun", 3: "Jul-Sep", 4: "Oct-Dec"}
    return f"Q{quarter} ({month_ranges[quarter]} {year})"


def assign_period(df: pd.DataFrame, date_col: str, period_label: str):
    """
    Adds a 'period' column to the DataFrame based on the selected aggregation.
    Returns human-readable period labels and adds a numeric sorting column.
    """
    if df.empty or date_col not in df.columns:
        st.warning("No valid date column found for period assignment")
        return df

    if period_label == "Daily":
        df["period"] = df[date_col].dt.strftime("%Y-%m-%d")  # Sortable format
        df["period_display"] = df[date_col].dt.strftime("%b %d, %Y")  # Display format
        df["period_sort"] = df[date_col].dt.normalize()

    elif period_label == "Weekly":
        df["week_start"] = df[date_col] - pd.to_timedelta(
            df[date_col].dt.weekday, unit="D"
        )
        df["period"] = df["week_start"].dt.strftime("%Y-%m-%d")  # Sortable format
        df["period_display"] = df["week_start"].apply(
            format_weekly_label
        )  # Use the improved formatting
        df["period_sort"] = df["week_start"]

    elif period_label == "Monthly":
        df["period"] = df[date_col].dt.strftime("%Y-%m")  # Sortable format
        df["period_display"] = df[date_col].dt.strftime("%b %Y")  # Display: Sep 2025
        df["period_sort"] = df[date_col].dt.to_period("M").dt.start_time

    elif period_label == "Quarterly":
        df["period"] = df[date_col].dt.to_period("Q").astype(str)  # Sortable: 2025Q1
        df["period_display"] = df["period"].apply(
            format_quarterly_label
        )  # Use the improved formatting
        df["period_sort"] = df[date_col].dt.to_period("Q").dt.start_time

    else:  # Yearly
        df["period"] = df[date_col].dt.strftime("%Y")  # Sortable: 2025
        df["period_display"] = df["period"]  # Display: 2025
        df["period_sort"] = df[date_col].dt.to_period("Y").dt.start_time

    # Sort the dataframe by period_sort to ensure chronological order
    df = df.sort_values("period_sort")

    return df
