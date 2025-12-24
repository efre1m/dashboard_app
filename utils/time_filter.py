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
    Filters out invalid dates like 1970-01-01.
    """
    import datetime
    import pandas as pd

    today = dt.date.today()

    # Handle "All Time" - use REAL data range
    if quick_range == "All Time":
        if not df.empty:
            # Collect ALL VALID dates from ALL columns
            all_valid_dates = []

            for col in df.columns:
                if "date" in col.lower() or "Date" in col:
                    try:
                        # Try to convert to datetime
                        dates = pd.to_datetime(df[col], errors="coerce")
                        # Drop NaT (invalid dates)
                        valid_dates = dates.dropna()

                        # Filter out invalid dates (like 1970-01-01)
                        for date_val in valid_dates:
                            try:
                                year = date_val.year
                                # Only accept dates from year 2000 onward
                                if year >= 2000 and year <= 2030:  # Reasonable range
                                    all_valid_dates.append(date_val)
                            except:
                                continue
                    except:
                        continue

            if all_valid_dates:
                min_date = min(all_valid_dates).date()
                max_date = max(all_valid_dates).date()
                print(
                    f"🔧 get_date_range: All Time valid range = {min_date} to {max_date}"
                )
                return min_date, max_date

        # Fallback to current year if no valid dates
        current_year_start = datetime.date(today.year, 1, 1)
        current_year_end = datetime.date(today.year, 12, 31)
        print(
            f"🔧 get_date_range: All Time fallback = {current_year_start} to {current_year_end}"
        )
        return current_year_start, current_year_end

    # Find which date column exists for other ranges
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
        # For Custom Range in this function (used when quick_range is "Custom Range")
        # We need to get reasonable defaults

        if date_col and not df.empty:
            # Get dates from the specified column
            dates = pd.to_datetime(df[date_col], errors="coerce")
            valid_dates = dates.dropna()

            if not valid_dates.empty:
                # Filter out invalid years
                filtered_dates = []
                for date_val in valid_dates:
                    try:
                        year = date_val.year
                        if 2000 <= year <= 2030:
                            filtered_dates.append(date_val)
                    except:
                        continue

                if filtered_dates:
                    min_date = min(filtered_dates)
                    max_date = max(filtered_dates)
                    start_default = min_date.date()
                    end_default = max_date.date()
                else:
                    # No valid dates, use current year
                    start_default = datetime.date(today.year, 1, 1)
                    end_default = datetime.date(today.year, 12, 31)
            else:
                # No valid dates, use current year
                start_default = datetime.date(today.year, 1, 1)
                end_default = datetime.date(today.year, 12, 31)
        else:
            # No date column, use current year
            start_default = datetime.date(today.year, 1, 1)
            end_default = datetime.date(today.year, 12, 31)

        # For the get_date_range function, we don't show the date inputs here
        # That's handled in render_patient_filter_controls
        return start_default, end_default


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
