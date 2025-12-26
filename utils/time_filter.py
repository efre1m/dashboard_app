import datetime as dt
import pandas as pd
import streamlit as st


def get_date_range(df, period_option):
    """
    Calculate date range based on period option.
    FIXED: Now returns correct full month/year ranges
    """
    import datetime
    import logging

    today = datetime.date.today()

    logging.info(
        f"📅 get_date_range called with period_option: '{period_option}', today: {today}"
    )

    if period_option == "Today":
        start_date = today
        end_date = today
    elif period_option == "This Week":
        # Week starts on Monday (0=Monday, 6=Sunday)
        start_date = today - datetime.timedelta(days=today.weekday())
        end_date = start_date + datetime.timedelta(days=6)
    elif period_option == "Last Week":
        start_date = today - datetime.timedelta(days=today.weekday() + 7)
        end_date = start_date + datetime.timedelta(days=6)
    elif period_option == "This Month":
        start_date = datetime.date(today.year, today.month, 1)
        # Last day of current month - CORRECT VERSION
        if today.month == 12:
            end_date = datetime.date(today.year, 12, 31)
        else:
            end_date = datetime.date(
                today.year, today.month + 1, 1
            ) - datetime.timedelta(days=1)
    elif period_option == "Last Month":
        # First day of last month
        if today.month == 1:
            start_date = datetime.date(today.year - 1, 12, 1)
            end_date = datetime.date(today.year - 1, 12, 31)
        else:
            start_date = datetime.date(today.year, today.month - 1, 1)
            # Last day of last month
            end_date = datetime.date(today.year, today.month, 1) - datetime.timedelta(
                days=1
            )
    elif period_option == "This Year":
        start_date = datetime.date(today.year, 1, 1)
        end_date = datetime.date(today.year, 12, 31)
    elif period_option == "Last Year":
        start_date = datetime.date(today.year - 1, 1, 1)
        end_date = datetime.date(today.year - 1, 12, 31)
    elif period_option == "All Time":
        # Get min/max from data
        from utils.dash_co import _get_patient_date_range

        min_date, max_date = _get_patient_date_range(df)
        return min_date, max_date
    else:
        # Default to current year
        start_date = datetime.date(today.year, 1, 1)
        end_date = datetime.date(today.year, 12, 31)

    logging.info(f"📅 get_date_range result: {start_date} to {end_date}")
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
    Format weekly period as human-readable with week number and range
    e.g., "W1 (Jan 1-7, 2025)"
    """
    period_end = period_start + dt.timedelta(days=6)

    # Calculate week number within the year
    week_number = period_start.isocalendar()[1]

    # Format: W1 (01-07 Jan, 2025)
    if period_start.year == period_end.year:
        if period_start.month == period_end.month:
            # Same month
            date_range = f"{period_start.day:02d}-{period_end.day:02d} {period_start.strftime('%b')}, {period_start.year}"
        else:
            # Crosses month boundary
            date_range = f"{period_start.day:02d} {period_start.strftime('%b')} - {period_end.day:02d} {period_end.strftime('%b')}, {period_start.year}"
    else:
        # Crosses year boundary (rare)
        date_range = (
            f"{period_start.strftime('%d %b, %Y')} - {period_end.strftime('%d %b, %Y')}"
        )

    return f"Week {week_number} ({date_range})"


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
    FIXED VERSION: Properly assign periods for ALL period types
    """
    if df.empty or date_col not in df.columns:
        st.warning("No valid date column found for period assignment")
        return df

    # Make sure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Remove rows with invalid dates
    df = df[df[date_col].notna()]

    if df.empty:
        return df

    print(f"\n📅 assign_period: Creating '{period_label}' periods")
    print(f"   Date range: {df[date_col].min()} to {df[date_col].max()}")
    print(f"   Total rows: {len(df)}")

    # RESET any existing period columns
    for col in ["period", "period_display", "period_sort"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    if period_label == "Daily":
        df["period"] = df[date_col].dt.strftime("%Y-%m-%d")  # Sortable: 2025-07-17
        df["period_display"] = df[date_col].dt.strftime(
            "%d %b %Y"
        )  # Display: 17 Jul 2025
        df["period_sort"] = df[date_col].dt.normalize()

        print(f"   Created DAILY periods")
        print(f"   Sample: {df['period_display'].head(3).tolist()}")

    elif period_label == "Weekly":
        # Calculate week start (Monday)
        df["week_start"] = df[date_col] - pd.to_timedelta(
            df[date_col].dt.weekday, unit="D"
        )
        df["period"] = df["week_start"].dt.strftime("%Y-%W")  # Sortable: 2025-28
        df["period_display"] = df["week_start"].apply(format_weekly_label)
        df["period_sort"] = df["week_start"]

        print(f"   Created WEEKLY periods")
        print(f"   Sample: {df['period_display'].head(3).tolist()}")

    elif period_label == "Monthly":
        # MONTHLY - Most important fix
        df["period"] = df[date_col].dt.strftime("%Y-%m")  # Sortable: 2025-07
        df["period_display"] = df[date_col].dt.strftime(
            "%b-%y"
        )  # Display: Jul-25 (NOT 25-Jul!)
        df["period_sort"] = df[date_col].dt.to_period("M").dt.start_time

        print(f"   Created MONTHLY periods")
        print(f"   Sample periods: {df['period_display'].head(5).tolist()}")
        print(f"   Unique periods: {sorted(df['period_display'].unique())}")

    elif period_label == "Quarterly":
        df["period"] = df[date_col].dt.to_period("Q").astype(str)  # Sortable: 2025Q3
        df["period_display"] = df["period"].apply(format_quarterly_label)
        df["period_sort"] = df[date_col].dt.to_period("Q").dt.start_time

        print(f"   Created QUARTERLY periods")

    else:  # Yearly
        df["period"] = df[date_col].dt.strftime("%Y")  # Sortable: 2025
        df["period_display"] = df["period"]  # Display: 2025
        df["period_sort"] = df[date_col].dt.to_period("Y").dt.start_time

        print(f"   Created YEARLY periods")

    # Sort the dataframe by period_sort to ensure chronological order
    df = df.sort_values("period_sort")

    print(
        f"   ✅ Successfully created {len(df['period_display'].unique())} unique periods\n"
    )
    return df
