from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import streamlit as st


DATA_PATH = Path(__file__).resolve().parents[1] / "imnid_assessment.csv"

WHICH_DATA_COL = "NDQA Which Data are you checking"
MRN_COL = "NDQA MRN of the Mother/newborn record being checked"
PAPER_ENTRY_DATE_COL = (
    "NDQA Paper data entry date of the Mother/newborn record being checked"
)
ENROLLMENT_DATE_COL = (
    "NDQA Enrollment date of the Mother/newborn record being checked"
)

MATERNAL_DOMAINS: List[Dict[str, Optional[str]]] = [
    {
        "id": "birth_outcome",
        "label": "Birth outcome",
        "consistency_indicator": "Birth Outcome Accuracy Rate",
        "completeness_indicator": "Birth Outcome Completeness",
        "paper": "NDQA Birth outcome - Paper Recorded",
        "dhis2": "NDQA Birth outcome - DHIS2 Recorded",
        "consistent": "NDQA Birth outcome - Consistent Across Systems",
        "reason_paper": None,
        "reason_dhis2": None,
    },
    {
        "id": "mode_of_delivery",
        "label": "Mode of delivery",
        "consistency_indicator": "Mode of Delivery Accuracy Rate",
        "completeness_indicator": "Mode of Delivery Completeness",
        "paper": "NDQA Mode of delivery - Paper Recorded",
        "dhis2": "NDQA Mode of delivery - DHIS2 Recorded",
        "consistent": "NDQA Mode of delivery - Consistent Across Systems",
        "reason_paper": None,
        "reason_dhis2": None,
    },
    {
        "id": "maternal_discharge_summary",
        "label": "Maternal discharge summary",
        "consistency_indicator": "Maternal Discharge Summary Accuracy Rate",
        "completeness_indicator": "Maternal Discharge Summary Completeness",
        "paper": "NDQA Maternal discharge summary - Paper Recorded",
        "dhis2": "NDQA Maternal discharge summary - DHIS2 Recorded",
        "consistent": "NDQA Maternal discharge summary - Consistent Across Systems",
        "reason_paper": None,
        "reason_dhis2": "NDQA Maternal discharge summary - Reason not recorded in DHIS2",
    },
]

NEWBORN_DOMAINS: List[Dict[str, Optional[str]]] = [
    {
        "id": "birth_weight",
        "label": "Birth weight",
        "consistency_indicator": "Birth Weight Consistency Rate",
        "completeness_indicator": "Birth Weight Completeness",
        "paper": "NDQA Birth weight - Paper Recorded",
        "dhis2": "NDQA Birth weight - DHIS2 Recorded",
        "consistent": "NDQA Birth weight - Consistent Across Systems",
        "reason_paper": None,
        "reason_dhis2": "NDQA Birth weight - Reason not recorded in DHIS2",
    },
    {
        "id": "cpap",
        "label": "CPAP",
        "consistency_indicator": "CPAP Consistency Rate",
        "completeness_indicator": "CPAP Completeness",
        "paper": "NDQA CPAP - Paper Recorded",
        "dhis2": "NDQA CPAP - DHIS2 Recorded",
        "consistent": "NDQA CPAP - Consistent Across Systems",
        "reason_paper": "NDQA CPAP - Reason not recorded in Paper",
        "reason_dhis2": "NDQA CPAP - Reason not recorded in DHIS2",
    },
    {
        "id": "kmc",
        "label": "KMC (Kangaroo Mother Care)",
        "consistency_indicator": "KMC Consistency Rate",
        "completeness_indicator": "Kangaroo Mother Care (KMC) Completeness",
        "paper": "NDQA KMC (Kangaroo Mother Care) - Paper Recorded",
        "dhis2": "NDQA KMC (Kangaroo Mother Care) - DHIS2 Recorded",
        "consistent": "NDQA KMC (Kangaroo Mother Care) - Consistent Across Systems",
        "reason_paper": "NDQA KMC (Kangaroo Mother Care) - Reason not recorded in Paper",
        "reason_dhis2": "NDQA KMC (Kangaroo Mother Care) - Reason not recorded in DHIS2",
    },
    {
        "id": "neonatal_status",
        "label": "Neonatal status at discharge",
        "consistency_indicator": "Neonatal Status at Discharge Consistency Rate",
        "completeness_indicator": "Neonatal Status at Discharge Completeness",
        "paper": "NDQA Neonatal status at discharge - Paper Recorded",
        "dhis2": "NDQA Neonatal status at discharge - DHIS2 Recorded",
        "consistent": "NDQA Neonatal status at discharge - Consistent Across Systems",
        "reason_paper": "NDQA Neonatal status at discharge - Reason not recorded in Paper",
        "reason_dhis2": "NDQA Neonatal status at discharge - Reason not recorded in DHIS2",
    },
    {
        "id": "temperature_at_admission",
        "label": "Temperature at neonatal admission",
        "consistency_indicator": "Temperature Recording Accuracy Rate",
        "completeness_indicator": "Neonatal Admission Temperature Completeness",
        "paper": "NDQA Temperature at neonatal admission - Paper Recorded",
        "dhis2": "NDQA Temperature at neonatal admission - DHIS2 Recorded",
        "consistent": "NDQA Temperature at neonatal admission - Consistent Across Systems",
        "reason_paper": None,
        "reason_dhis2": "NDQA Temperature at neonatal admission - Reason not recorded in DHIS2",
    },
]

DOMAIN_CONFIG = {
    "maternal": MATERNAL_DOMAINS,
    "newborn": NEWBORN_DOMAINS,
}

DQA_SECTION_SORT = {
    "Consistency Indicators": 0,
    "Completeness Indicators": 1,
    "Timeliness Indicators": 2,
    "Missing Reason Indicators": 3,
}

PLACEHOLDER_TEXT = {"", "nan", "none", "not given", "no", "n/a", "na"}


def _normalize_name(value: object) -> str:
    if pd.isna(value):
        return ""
    text = " ".join(str(value).strip().lower().split())
    return text


def _has_meaningful_text(value: object) -> bool:
    text = _normalize_name(value)
    return text not in PLACEHOLDER_TEXT


def _normalize_binary(value: object) -> Optional[bool]:
    if pd.isna(value):
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)) and not pd.isna(value):
        if float(value) == 1.0:
            return True
        if float(value) == 0.0:
            return False

    text = _normalize_name(value)
    if text in {"1", "1.0", "yes", "y", "true"}:
        return True
    if text in {"0", "0.0", "no", "n", "false"}:
        return False
    return None


def _infer_record_type(row: pd.Series) -> str:
    which_data = _normalize_name(row.get(WHICH_DATA_COL))
    if which_data in {"1", "1.0", "maternal", "mother"}:
        return "maternal"
    if which_data in {"2", "2.0", "newborn", "neonatal", "neonate"}:
        return "newborn"
    return "unknown"


def _domain_column_key(domain_id: str, metric_name: str) -> str:
    return f"__assessment_{domain_id}_{metric_name}"


@st.cache_data(ttl=900, show_spinner=False)
def load_assessment_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Assessment CSV not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df.columns = [str(col).strip() for col in df.columns]

    required_columns = [
        "region_name",
        "facility_name",
        WHICH_DATA_COL,
        MRN_COL,
        PAPER_ENTRY_DATE_COL,
        ENROLLMENT_DATE_COL,
        "eventDate",
    ]
    for column in required_columns:
        if column not in df.columns:
            df[column] = pd.NA

    df["region_name_norm"] = df["region_name"].apply(_normalize_name)
    df["facility_name_norm"] = df["facility_name"].apply(_normalize_name)
    df["record_type"] = df.apply(_infer_record_type, axis=1)

    df["event_date_dt"] = pd.to_datetime(df["eventDate"], errors="coerce")
    df["paper_entry_date_dt"] = pd.to_datetime(df[PAPER_ENTRY_DATE_COL], errors="coerce")
    df["enrollment_date_dt"] = pd.to_datetime(df[ENROLLMENT_DATE_COL], errors="coerce")

    df["mrn_present"] = df[MRN_COL].apply(_has_meaningful_text)
    df["paper_entry_present"] = df[PAPER_ENTRY_DATE_COL].apply(_has_meaningful_text)
    df["enrollment_present"] = df[ENROLLMENT_DATE_COL].apply(_has_meaningful_text)

    df["entry_delay_days"] = (
        df["paper_entry_date_dt"] - df["enrollment_date_dt"]
    ).dt.days
    df["timely_entry"] = df["entry_delay_days"].between(0, 7, inclusive="both")
    df["invalid_date_sequence"] = df["entry_delay_days"] < 0

    enrollment_date_key = df["enrollment_date_dt"].dt.strftime("%Y-%m-%d")
    duplicate_ready = (
        df["facility_name_norm"].ne("")
        & df["mrn_present"]
        & enrollment_date_key.notna()
        & df["record_type"].isin(["maternal", "newborn"])
    )

    df["duplicate_key"] = None
    df.loc[duplicate_ready, "duplicate_key"] = (
        df.loc[duplicate_ready, "facility_name_norm"]
        + "|"
        + df.loc[duplicate_ready, MRN_COL].astype(str).str.strip().str.lower()
        + "|"
        + df.loc[duplicate_ready, "record_type"]
        + "|"
        + enrollment_date_key.loc[duplicate_ready]
    )
    duplicate_counts = df.loc[duplicate_ready, "duplicate_key"].value_counts()
    df["is_duplicate_assessment"] = (
        df["duplicate_key"].map(duplicate_counts).fillna(0).astype(int) > 1
    )
    df["duplicate_ready"] = duplicate_ready

    for domain_list in DOMAIN_CONFIG.values():
        for domain in domain_list:
            for metric_name in ("paper", "dhis2", "consistent"):
                source_col = domain.get(metric_name)
                normalized_col = _domain_column_key(domain["id"], metric_name)
                if source_col and source_col in df.columns:
                    df[normalized_col] = df[source_col].apply(_normalize_binary)
                else:
                    df[normalized_col] = None

            for metric_name in ("reason_paper", "reason_dhis2"):
                source_col = domain.get(metric_name)
                normalized_col = _domain_column_key(domain["id"], metric_name)
                if source_col and source_col in df.columns:
                    df[normalized_col] = df[source_col].apply(_has_meaningful_text)
                else:
                    df[normalized_col] = False

    return df


def filter_assessment_scope(
    df: pd.DataFrame,
    user: Dict,
    dashboard_level: str,
    filter_mode: str = "All Facilities",
    selected_regions: Optional[List[str]] = None,
    selected_facilities: Optional[List[str]] = None,
) -> pd.DataFrame:
    selected_regions = selected_regions or []
    selected_facilities = selected_facilities or []
    filtered_df = df.copy()

    if dashboard_level == "facility":
        facility_norm = _normalize_name(user.get("facility_name"))
        if facility_norm:
            filtered_df = filtered_df[filtered_df["facility_name_norm"] == facility_norm]
        return filtered_df

    if dashboard_level == "regional":
        region_norm = _normalize_name(user.get("region_name"))
        if region_norm:
            filtered_df = filtered_df[filtered_df["region_name_norm"] == region_norm]

        facility_names = [
            facility
            for facility in selected_facilities
            if _normalize_name(facility) and _normalize_name(facility) != "all facilities"
        ]
        if facility_names:
            facility_norms = {_normalize_name(facility) for facility in facility_names}
            filtered_df = filtered_df[
                filtered_df["facility_name_norm"].isin(facility_norms)
            ]
        return filtered_df

    if filter_mode == "By Region" and selected_regions:
        region_norms = {_normalize_name(region) for region in selected_regions if _normalize_name(region)}
        if region_norms:
            filtered_df = filtered_df[filtered_df["region_name_norm"].isin(region_norms)]
    elif filter_mode == "By Facility" and selected_facilities:
        facility_norms = {
            _normalize_name(facility)
            for facility in selected_facilities
            if _normalize_name(facility) and _normalize_name(facility) != "all facilities"
        }
        if facility_norms:
            filtered_df = filtered_df[
                filtered_df["facility_name_norm"].isin(facility_norms)
            ]

    return filtered_df


def _format_percent(numerator: int, denominator: int) -> Optional[float]:
    if denominator == 0:
        return None
    return round((numerator / denominator) * 100, 1)


def _build_indicator_row(
    domain_label: str,
    indicator_id: str,
    indicator_label: str,
    numerator: int,
    denominator: int,
    domain_sort: int,
    indicator_sort: int,
    numerator_label: str,
    denominator_label: str,
    availability_note: str = "",
) -> Dict[str, object]:
    value = _format_percent(numerator, denominator)
    return {
        "domain": domain_label,
        "indicator_id": indicator_id,
        "indicator": indicator_label,
        "numerator": int(numerator),
        "denominator": int(denominator),
        "value_pct": value,
        "value_display": "N/A" if value is None else f"{value:.1f}%",
        "numerator_label": numerator_label,
        "denominator_label": denominator_label,
        "availability_note": availability_note,
        "domain_sort": domain_sort,
        "indicator_sort": indicator_sort,
    }


def _build_summary_text(row: pd.Series) -> str:
    numerator = int(row["numerator"])
    denominator = int(row["denominator"])
    value_display = row["value_display"]
    numerator_label = str(row["numerator_label"])
    denominator_label = str(row["denominator_label"])
    indicator_id = str(row["indicator_id"])
    indicator_name = str(row["indicator"])
    availability_note = str(row.get("availability_note", "") or "").strip()

    if availability_note:
        return availability_note

    if indicator_id == "overall_accuracy_index":
        return (
            f"{numerator} consistent values from `{numerator_label}` out of "
            f"{denominator} total expected fields ({value_display})."
        )

    if indicator_id == "timely_entry_rate":
        return (
            f"{numerator} records entered on the same day out of "
            f"{denominator} total records reviewed "
            f"({value_display})."
        )

    if indicator_id.endswith("_missing_in_dhis2_rate"):
        return (
            f"{numerator} records with a documented reason for missing in DHIS2 for "
            f"{indicator_name.replace(' Missing Reason in DHIS2 Rate', '')} out of "
            f"{denominator} records missing in DHIS2 "
            f"({value_display})."
        )

    if indicator_id.endswith("_missing_in_paper_rate"):
        return (
            f"{numerator} records with a documented reason for missing on paper for "
            f"{indicator_name.replace(' Missing Reason in Paper Rate', '')} out of "
            f"{denominator} records missing on paper "
            f"({value_display})."
        )

    if indicator_id.endswith("_completeness_rate"):
        return (
            f"{numerator} records recorded in DHIS2 out of "
            f"{denominator} records recorded on paper ({value_display})."
        )

    if denominator_label == "Total Records Reviewed":
        return (
            f"{numerator} records where {numerator_label} out of "
            f"{denominator} total records reviewed ({value_display})."
        )

    return (
        f"{numerator} records where {numerator_label} out of "
        f"{denominator} records where {denominator_label} "
        f"({value_display})."
    )


def _collect_missing_reason_breakdown(
    df: pd.DataFrame,
    record_type: str,
    indicator_id: str,
) -> pd.DataFrame:
    if indicator_id.endswith("_missing_in_dhis2_rate"):
        reason_key = "reason_dhis2"
        source_key = "dhis2"
        domain_id = indicator_id[: -len("_missing_in_dhis2_rate")]
    elif indicator_id.endswith("_missing_in_paper_rate"):
        reason_key = "reason_paper"
        source_key = "paper"
        domain_id = indicator_id[: -len("_missing_in_paper_rate")]
    else:
        return pd.DataFrame()

    domain = next(
        (item for item in DOMAIN_CONFIG.get(record_type, []) if item["id"] == domain_id),
        None,
    )
    if not domain:
        return pd.DataFrame()

    reason_column = domain.get(reason_key)
    source_column_key = _domain_column_key(domain_id, source_key)
    if not reason_column or reason_column not in df.columns or source_column_key not in df.columns:
        return pd.DataFrame()

    reason_rows: List[Dict[str, object]] = []
    eligible_df = df[df[source_column_key] != True].copy()
    series = eligible_df[reason_column].dropna()
    for raw_value in series.tolist():
        display_value = str(raw_value).strip()
        if not _has_meaningful_text(display_value):
            continue
        reason_rows.append(
            {
                "reason_norm": _normalize_name(display_value),
                "reason_display": display_value,
            }
        )

    if not reason_rows:
        return pd.DataFrame()

    reason_df = pd.DataFrame(reason_rows)
    summary_rows: List[Dict[str, object]] = []
    for reason_norm, group in reason_df.groupby("reason_norm"):
        display_label = (
            group["reason_display"].value_counts().sort_values(ascending=False).index[0]
        )
        count = int(len(group))
        summary_rows.append(
            {
                "reason_norm": reason_norm,
                "reason_display": display_label,
                "count": count,
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=["count", "reason_display"], ascending=[False, True]
    )
    total_count = int(summary_df["count"].sum())
    summary_df["share_pct"] = round((summary_df["count"] / total_count) * 100, 1)
    return summary_df.reset_index(drop=True)


def _collapse_reason_breakdown(
    reason_df: pd.DataFrame,
    max_categories: int = 8,
) -> pd.DataFrame:
    if reason_df.empty or len(reason_df) <= max_categories:
        return reason_df

    keep_count = max_categories - 1
    top_df = reason_df.head(keep_count).copy()
    other_df = reason_df.iloc[keep_count:]
    other_count = int(other_df["count"].sum())
    total_count = int(reason_df["count"].sum())

    collapsed = pd.concat(
        [
            top_df,
            pd.DataFrame(
                [
                    {
                        "reason_norm": "__other__",
                        "reason_display": "Other recorded reasons",
                        "count": other_count,
                        "share_pct": round((other_count / total_count) * 100, 1),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    return collapsed


def _is_completeness_indicator(indicator_id: str) -> bool:
    return indicator_id.endswith("_completeness_rate")


def _completeness_domain_id(indicator_id: str) -> Optional[str]:
    suffix = "_completeness_rate"
    if indicator_id.endswith(suffix):
        return indicator_id[: -len(suffix)]
    return None


def compute_indicator_rows(df: pd.DataFrame, record_type: str) -> pd.DataFrame:
    subset = df[df["record_type"] == record_type].copy()
    if subset.empty:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    domains = DOMAIN_CONFIG[record_type]
    total_records = len(subset)

    consistency_domain = "Consistency Indicators"
    consistency_sort = DQA_SECTION_SORT[consistency_domain]

    for indicator_sort, domain in enumerate(domains):
        consistent_col = _domain_column_key(domain["id"], "consistent")
        rows.append(
            _build_indicator_row(
                consistency_domain,
                f"{domain['id']}_consistency_rate",
                domain["consistency_indicator"] or f"{domain['label']} Consistency Rate",
                int((subset[consistent_col] == True).sum()),
                int(total_records),
                consistency_sort,
                indicator_sort,
                "Consistent = 1",
                "Total Records Reviewed",
            )
        )

    rows.append(
        _build_indicator_row(
            consistency_domain,
            "overall_accuracy_index",
            "Overall Accuracy Index",
            int(
                sum(
                    int((subset[_domain_column_key(domain["id"], "consistent")] == True).sum())
                    for domain in domains
                )
            ),
            int(total_records * len(domains)),
            consistency_sort,
            len(domains),
            "Consistent fields",
            "Total expected fields",
        )
    )

    completeness_domain = "Completeness Indicators"
    completeness_sort = DQA_SECTION_SORT[completeness_domain]

    for indicator_sort, domain in enumerate(domains):
        paper_col = _domain_column_key(domain["id"], "paper")
        dhis2_col = _domain_column_key(domain["id"], "dhis2")
        rows.append(
            _build_indicator_row(
                completeness_domain,
                f"{domain['id']}_completeness_rate",
                domain["completeness_indicator"] or f"{domain['label']} Completeness",
                int((subset[dhis2_col] == True).sum()),
                int((subset[paper_col] == True).sum()),
                completeness_sort,
                indicator_sort,
                "Recorded (DHIS2)",
                "Recorded (Paper)",
            )
        )

    timeliness_domain = "Timeliness Indicators"
    timeliness_sort = DQA_SECTION_SORT[timeliness_domain]
    timely_entry_same_day = subset["entry_delay_days"] == 0
    rows.append(
        _build_indicator_row(
            timeliness_domain,
            "timely_entry_rate",
            "Timely Entry Rate",
            int((timely_entry_same_day == True).sum()),
            int(total_records),
            timeliness_sort,
            0,
            "Entered on same day",
            "Total Records Reviewed",
        )
    )

    missing_domain = "Missing Reason Indicators"
    missing_sort = DQA_SECTION_SORT[missing_domain]
    missing_indicator_sort = 0
    for domain in domains:
        dhis2_col = _domain_column_key(domain["id"], "dhis2")
        paper_col = _domain_column_key(domain["id"], "paper")
        reason_dhis2_col = _domain_column_key(domain["id"], "reason_dhis2")
        reason_paper_col = _domain_column_key(domain["id"], "reason_paper")

        if domain.get("reason_dhis2"):
            reason_dhis2_present = subset[reason_dhis2_col].fillna(False)
            missing_in_dhis2 = subset[dhis2_col] != True
            rows.append(
                _build_indicator_row(
                    missing_domain,
                    f"{domain['id']}_missing_in_dhis2_rate",
                    f"{domain['label']} Missing Reason in DHIS2 Rate",
                    int((reason_dhis2_present & missing_in_dhis2).sum()),
                    int(missing_in_dhis2.sum()),
                    missing_sort,
                    missing_indicator_sort,
                    "Documented reason for missing in DHIS2",
                    "Records missing in DHIS2",
                )
            )
        else:
            rows.append(
                _build_indicator_row(
                    missing_domain,
                    f"{domain['id']}_missing_in_dhis2_rate",
                    f"{domain['label']} Missing Reason in DHIS2 Rate",
                    0,
                    0,
                    missing_sort,
                    missing_indicator_sort,
                    "Documented reason for missing in DHIS2",
                    "Records missing in DHIS2",
                    "This variable does not have a DHIS2 missing-reason field in the assessment file, so this rate cannot be shown.",
                )
            )
        missing_indicator_sort += 1

        if domain.get("reason_paper"):
            reason_paper_present = subset[reason_paper_col].fillna(False)
            missing_in_paper = subset[paper_col] != True
            rows.append(
                _build_indicator_row(
                    missing_domain,
                    f"{domain['id']}_missing_in_paper_rate",
                    f"{domain['label']} Missing Reason in Paper Rate",
                    int((reason_paper_present & missing_in_paper).sum()),
                    int(missing_in_paper.sum()),
                    missing_sort,
                    missing_indicator_sort,
                    "Documented reason for missing on paper",
                    "Records missing on paper",
                )
            )
        else:
            rows.append(
                _build_indicator_row(
                    missing_domain,
                    f"{domain['id']}_missing_in_paper_rate",
                    f"{domain['label']} Missing Reason in Paper Rate",
                    0,
                    0,
                    missing_sort,
                    missing_indicator_sort,
                    "Documented reason for missing on paper",
                    "Records missing on paper",
                    "This variable does not have a paper missing-reason field in the assessment file, so this rate cannot be shown.",
                )
            )
        missing_indicator_sort += 1

    indicator_df = pd.DataFrame(rows)
    indicator_df = indicator_df.sort_values(
        by=["domain_sort", "indicator_sort", "domain", "indicator"]
    ).reset_index(drop=True)
    return indicator_df


def _scope_label(
    dashboard_level: str,
    filter_mode: str,
    selected_regions: Optional[List[str]],
    selected_facilities: Optional[List[str]],
    user: Dict,
) -> str:
    selected_regions = selected_regions or []
    selected_facilities = selected_facilities or []

    if dashboard_level == "facility":
        return user.get("facility_name", "Selected Facility")

    if dashboard_level == "regional":
        if selected_facilities and selected_facilities != ["All Facilities"]:
            if len(selected_facilities) > 3:
                return f"{len(selected_facilities)} facilities in {user.get('region_name', 'Selected Region')}"
            return ", ".join(selected_facilities)
        return user.get("region_name", "Selected Region")

    if filter_mode == "By Region" and selected_regions:
        if len(selected_regions) > 3:
            return f"{len(selected_regions)} regions"
        return ", ".join(selected_regions)
    if filter_mode == "By Facility" and selected_facilities:
        if len(selected_facilities) > 3:
            return f"{len(selected_facilities)} facilities"
        return ", ".join(selected_facilities)
    return user.get("country_name", "Overall")


def _clean_options(series: pd.Series) -> List[str]:
    options = []
    for value in series.dropna().tolist():
        text = str(value).strip()
        if text:
            options.append(text)
    return sorted(dict.fromkeys(options))


def _select_many_with_all(label: str, options: List[str], key: str, all_token: str) -> List[str]:
    if not options:
        st.multiselect(label, options=[], default=[], key=key, disabled=True)
        return []

    selector_options = [all_token] + options
    if key not in st.session_state or not st.session_state.get(key):
        st.session_state[key] = [all_token]

    st.multiselect(label, options=selector_options, key=key)
    selected_values = st.session_state.get(key, [all_token])
    if all_token in selected_values or len(selected_values) == 0:
        return options
    return [value for value in selected_values if value in options]


def _find_indicator_row(
    df: pd.DataFrame,
    record_type: str,
    domain_label: str,
    indicator_label: str,
) -> Optional[pd.Series]:
    indicator_df = compute_indicator_rows(df, record_type)
    if indicator_df.empty:
        return None

    matched = indicator_df[
        (indicator_df["domain"] == domain_label)
        & (indicator_df["indicator"] == indicator_label)
    ]
    if matched.empty:
        return None
    return matched.iloc[0]


def _build_plot_rows(
    df: pd.DataFrame,
    record_type: str,
    domain_label: str,
    indicator_label: str,
    group_mode: str,
    entity_col: Optional[str],
    selected_entities: List[str],
    overall_label: str,
) -> pd.DataFrame:
    if group_mode == "Overall" or not entity_col:
        row = _find_indicator_row(df, record_type, domain_label, indicator_label)
        if row is None:
            return pd.DataFrame()
        return pd.DataFrame(
            [
                {
                    "entity": overall_label,
                    "value_pct": row["value_pct"],
                    "value_display": row["value_display"],
                    "numerator": row["numerator"],
                    "denominator": row["denominator"],
                }
            ]
        )

    rows: List[Dict[str, object]] = []
    for entity in selected_entities:
        entity_df = df[df[entity_col] == entity].copy()
        if entity_df.empty:
            continue
        row = _find_indicator_row(entity_df, record_type, domain_label, indicator_label)
        if row is None:
            continue
        rows.append(
            {
                "entity": entity,
                "value_pct": row["value_pct"],
                "value_display": row["value_display"],
                "numerator": row["numerator"],
                "denominator": row["denominator"],
            }
        )
    return pd.DataFrame(rows)


def _build_completeness_plot_rows(
    df: pd.DataFrame,
    record_type: str,
    indicator_id: str,
    group_mode: str,
    entity_col: Optional[str],
    selected_entities: List[str],
    overall_label: str,
) -> pd.DataFrame:
    domain_id = _completeness_domain_id(indicator_id)
    if not domain_id:
        return pd.DataFrame()

    domain = next(
        (item for item in DOMAIN_CONFIG.get(record_type, []) if item["id"] == domain_id),
        None,
    )
    if not domain:
        return pd.DataFrame()

    paper_col = _domain_column_key(domain["id"], "paper")
    dhis2_col = _domain_column_key(domain["id"], "dhis2")

    def build_entity_rows(entity_name: str, entity_df: pd.DataFrame) -> List[Dict[str, object]]:
        paper_count = int((entity_df[paper_col] == True).sum())
        dhis2_count = int((entity_df[dhis2_col] == True).sum())
        return [
            {
                "entity": entity_name,
                "series": "Recorded (DHIS2)",
                "count": dhis2_count,
            },
            {
                "entity": entity_name,
                "series": "Recorded (Paper)",
                "count": paper_count,
            },
        ]

    if group_mode == "Overall" or not entity_col:
        return pd.DataFrame(build_entity_rows(overall_label, df))

    rows: List[Dict[str, object]] = []
    for entity in selected_entities:
        entity_df = df[df[entity_col] == entity].copy()
        if entity_df.empty:
            continue
        rows.extend(build_entity_rows(entity, entity_df))
    return pd.DataFrame(rows)


def _resolve_assessment_view_context(
    record_df: pd.DataFrame,
    dashboard_level: str,
    user: Dict,
    key_prefix: str,
) -> Dict[str, object]:
    if dashboard_level == "facility":
        facility_name = user.get("facility_name", "Selected Facility")
        st.caption(f"Facility scope: {facility_name}")
        return {
            "group_mode": "Overall",
            "entity_col": None,
            "selected_entities": [facility_name],
            "filtered_df": record_df,
            "scope_name": facility_name,
            "overall_label": facility_name,
            "plot_title_suffix": "Overall",
        }

    if dashboard_level == "regional":
        region_name = user.get("region_name", "Selected Region")
        group_mode = st.radio(
            "Group By",
            options=["Overall", "Facility"],
            key=f"{key_prefix}_group_mode",
        )
        st.caption(f"Regional scope: {region_name}")

        if group_mode == "Overall":
            return {
                "group_mode": group_mode,
                "entity_col": None,
                "selected_entities": [region_name],
                "filtered_df": record_df,
                "scope_name": region_name,
                "overall_label": region_name,
                "plot_title_suffix": "Overall",
            }

        facility_options = _clean_options(record_df["facility_name"])
        selected_facilities = _select_many_with_all(
            "Select Facilities",
            facility_options,
            key=f"{key_prefix}_facility_selection",
            all_token="All Facilities in Region",
        )
        filtered_df = record_df[record_df["facility_name"].isin(selected_facilities)].copy()
        return {
            "group_mode": group_mode,
            "entity_col": "facility_name",
            "selected_entities": selected_facilities,
            "filtered_df": filtered_df,
            "scope_name": region_name,
            "overall_label": region_name,
            "plot_title_suffix": "Facilities",
        }

    group_mode = st.radio(
        "Group By",
        options=["Overall", "Multi Regional", "Regional"],
        key=f"{key_prefix}_group_mode",
    )

    if group_mode == "Overall":
        return {
            "group_mode": group_mode,
            "entity_col": None,
            "selected_entities": [user.get("country_name", "Overall")],
            "filtered_df": record_df,
            "scope_name": user.get("country_name", "Overall"),
            "overall_label": user.get("country_name", "Overall"),
            "plot_title_suffix": "Overall",
        }

    if group_mode == "Multi Regional":
        region_options = _clean_options(record_df["region_name"])
        selected_regions = _select_many_with_all(
            "Select Regions",
            region_options,
            key=f"{key_prefix}_region_selection",
            all_token="All Regions",
        )
        filtered_df = record_df[record_df["region_name"].isin(selected_regions)].copy()
        return {
            "group_mode": group_mode,
            "entity_col": "region_name",
            "selected_entities": selected_regions,
            "filtered_df": filtered_df,
            "scope_name": f"{len(selected_regions)} region(s)" if selected_regions else "No regions selected",
            "overall_label": user.get("country_name", "Overall"),
            "plot_title_suffix": "Regions",
        }

    region_options = _clean_options(record_df["region_name"])
    if not region_options:
        return {
            "group_mode": group_mode,
            "entity_col": "facility_name",
            "selected_entities": [],
            "filtered_df": record_df.iloc[0:0].copy(),
            "scope_name": "No regions available",
            "overall_label": user.get("country_name", "Overall"),
            "plot_title_suffix": "Facilities",
        }

    selected_region = st.selectbox(
        "Select Region",
        options=region_options,
        key=f"{key_prefix}_selected_region",
    )
    region_df = record_df[record_df["region_name"] == selected_region].copy()
    facility_options = _clean_options(region_df["facility_name"])
    selected_facilities = _select_many_with_all(
        "Select Facilities",
        facility_options,
        key=f"{key_prefix}_facility_selection",
        all_token="All Facilities in Region",
    )
    filtered_df = region_df[region_df["facility_name"].isin(selected_facilities)].copy()
    return {
        "group_mode": group_mode,
        "entity_col": "facility_name",
        "selected_entities": selected_facilities,
        "filtered_df": filtered_df,
        "scope_name": selected_region,
        "overall_label": selected_region,
        "plot_title_suffix": "Facilities",
    }


def _render_indicator_detail(
    record_df: pd.DataFrame,
    selected_row: pd.Series,
    record_type: str,
    scope_name: str,
    plot_df: pd.DataFrame,
    plot_title_suffix: str,
    group_mode: str,
    entity_col: Optional[str],
    selected_entities: List[str],
    overall_label: str,
) -> None:
    numerator_label = str(selected_row["numerator_label"])
    denominator_label = str(selected_row["denominator_label"])
    indicator_id = str(selected_row["indicator_id"])

    st.markdown(
        f"""
        <div class="assessment-detail-card">
            <div class="assessment-detail-eyebrow">{selected_row["domain"]}</div>
            <div class="assessment-detail-title">{selected_row["indicator"]}</div>
            <div class="assessment-detail-subtitle">Scope: {scope_name}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(_build_summary_text(selected_row))

    if not plot_df.empty:
        if _is_completeness_indicator(indicator_id):
            fig = None
        elif plot_df["value_pct"].notna().any():
            fig = px.bar(
                plot_df,
                x="entity",
                y="value_pct",
                text="value_display",
                title=f"{selected_row['indicator']} by {plot_title_suffix}",
                labels={"entity": plot_title_suffix.rstrip("s"), "value_pct": "Percent"},
                height=360,
            )
            fig.update_traces(
                textposition="outside",
                customdata=plot_df[["numerator", "denominator"]],
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Value: %{y:.1f}%<br>"
                    f"{numerator_label}: %{{customdata[0]}}<br>"
                    f"{denominator_label}: %{{customdata[1]}}<extra></extra>"
                ),
            )
            fig.update_layout(
                yaxis=dict(range=[0, 100], ticksuffix="%"),
                xaxis_tickangle=-25,
                margin=dict(l=20, r=20, t=60, b=20),
            )
        else:
            fig = None

        if fig is not None:
            st.plotly_chart(
                fig,
                use_container_width=True,
                key=f"{record_type}_{selected_row['indicator_id']}_{plot_title_suffix}_{scope_name}",
            )

    if _is_completeness_indicator(indicator_id):
        completeness_plot_df = _build_completeness_plot_rows(
            record_df,
            record_type,
            indicator_id,
            group_mode,
            entity_col,
            selected_entities,
            overall_label,
        )
        if not completeness_plot_df.empty:
            completeness_fig = px.bar(
                completeness_plot_df,
                x="entity",
                y="count",
                color="series",
                barmode="stack",
                title=f"{selected_row['indicator']} Recorded Counts",
                labels={
                    "entity": plot_title_suffix.rstrip("s"),
                    "count": "Recorded Count",
                    "series": "Recorded",
                },
                height=360,
                color_discrete_map={
                    "Recorded (DHIS2)": "#2563eb",
                    "Recorded (Paper)": "#f59e0b",
                },
            )
            completeness_fig.update_traces(
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "%{fullData.name}: %{y}<extra></extra>"
                ),
            )
            completeness_fig.update_layout(
                xaxis_tickangle=-25,
                margin=dict(l=20, r=20, t=60, b=20),
                legend_title_text="Recorded",
            )
            st.plotly_chart(
                completeness_fig,
                use_container_width=True,
                key=f"{record_type}_{indicator_id}_{scope_name}_completeness_counts",
            )

    reason_breakdown_df = _collect_missing_reason_breakdown(
        record_df,
        record_type,
        indicator_id,
    )
    if not reason_breakdown_df.empty:
        display_reason_df = _collapse_reason_breakdown(reason_breakdown_df)
        pie_fig = px.pie(
            display_reason_df,
            names="reason_display",
            values="count",
            title=f"Reason Breakdown for {selected_row['indicator']}",
        )
        pie_fig.update_traces(
            textinfo="percent" if len(display_reason_df) > 5 else "percent+label",
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Entries: %{value}<br>"
                "Share: %{percent}<extra></extra>"
            ),
        )
        pie_fig.update_layout(
            margin=dict(l=20, r=20, t=60, b=20),
            legend_title_text="Reason",
        )
        st.plotly_chart(
            pie_fig,
            use_container_width=True,
            key=f"{record_type}_{selected_row['indicator_id']}_{scope_name}_reasons",
        )
        st.caption(
            "Pie chart shows the distribution of documented missing-reason entries in the current scope."
        )


def _render_record_type_tab(
    df: pd.DataFrame,
    record_type: str,
    scope_name: str,
    dashboard_level: str,
    user: Dict,
    key_prefix: str,
) -> None:
    record_df = df[df["record_type"] == record_type].copy()
    if record_df.empty:
        st.info(f"No {record_type} assessment records found for {scope_name}.")
        return

    indicator_df = compute_indicator_rows(df, record_type)
    if indicator_df.empty:
        st.info(f"No {record_type} indicators could be calculated for {scope_name}.")
        return

    left_col, right_col = st.columns([3, 2], gap="large")

    domain_options = indicator_df["domain"].drop_duplicates().tolist()
    domain_key = f"{key_prefix}_selected_domain"
    indicator_key = f"{key_prefix}_selected_indicator"
    filter_card_key = f"{key_prefix}_filters_card"

    if domain_key not in st.session_state or st.session_state[domain_key] not in domain_options:
        st.session_state[domain_key] = domain_options[0]

    with right_col:
        with st.container(key=filter_card_key):
            st.markdown('<div class="assessment-filter-box">', unsafe_allow_html=True)
            st.markdown(
                '<div class="assessment-filter-title">Assessment Filters</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="assessment-filter-subtitle">Choose comparison scope, then select one domain and one indicator.</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="assessment-filter-divider"></div>',
                unsafe_allow_html=True,
            )

            view_context = _resolve_assessment_view_context(
                record_df,
                dashboard_level=dashboard_level,
                user=user,
                key_prefix=key_prefix,
            )

            filtered_scope_df = view_context["filtered_df"]
            if filtered_scope_df.empty:
                st.warning("No data for the selected filter combination.")
                st.markdown("</div>", unsafe_allow_html=True)
                return

            scoped_indicator_df = compute_indicator_rows(filtered_scope_df, record_type)
            if scoped_indicator_df.empty:
                st.warning("No indicators available for the selected filter combination.")
                st.markdown("</div>", unsafe_allow_html=True)
                return

            selected_domain = st.selectbox(
                "Domain",
                options=domain_options,
                key=domain_key,
            )

            domain_indicator_df = scoped_indicator_df[
                scoped_indicator_df["domain"] == selected_domain
            ].copy()
            indicator_options = domain_indicator_df["indicator"].tolist()

            if (
                indicator_key not in st.session_state
                or st.session_state[indicator_key] not in indicator_options
            ):
                st.session_state[indicator_key] = indicator_options[0]

            selected_indicator = st.selectbox(
                "Indicator",
                options=indicator_options,
                key=indicator_key,
            )

            st.caption(f"{len(filtered_scope_df)} {record_type} assessment records in scope")
            st.markdown("</div>", unsafe_allow_html=True)

    selected_row = domain_indicator_df[
        domain_indicator_df["indicator"] == st.session_state[indicator_key]
    ].iloc[0]
    plot_df = _build_plot_rows(
        filtered_scope_df,
        record_type,
        selected_domain,
        st.session_state[indicator_key],
        view_context["group_mode"],
        view_context["entity_col"],
        view_context["selected_entities"],
        view_context["overall_label"],
    )

    with left_col:
        _render_indicator_detail(
            filtered_scope_df,
            selected_row,
            record_type,
            view_context["scope_name"],
            plot_df,
            view_context["plot_title_suffix"],
            view_context["group_mode"],
            view_context["entity_col"],
            view_context["selected_entities"],
            view_context["overall_label"],
        )


def render_assessment_tab(
    user: Dict,
    dashboard_level: str,
    filter_mode: str = "All Facilities",
    selected_regions: Optional[List[str]] = None,
    selected_facilities: Optional[List[str]] = None,
) -> None:
    st.markdown(
        """
        <style>
        .assessment-filter-box {
            margin-bottom: 10px;
        }
        .assessment-filter-title {
            font-size: 1rem;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 4px;
        }
        .assessment-filter-subtitle {
            font-size: 0.78rem;
            color: #475569;
            margin-bottom: 10px;
        }
        .assessment-filter-divider {
            height: 1px;
            background: #bfdbfe;
            margin: 8px 0 12px 0;
        }
        [class*="filters_card"] {
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            border: 3px solid #1d4ed8;
            border-radius: 14px;
            padding: 14px 14px 10px 14px;
            box-shadow: 0 8px 18px rgba(2, 6, 23, 0.12);
        }
        .assessment-detail-card {
            border: 1px solid #cbd5e1;
            border-radius: 14px;
            background: linear-gradient(135deg, #ffffff, #f8fafc);
            padding: 0.85rem 1rem;
            margin-bottom: 0.8rem;
            box-shadow: 0 8px 18px rgba(2, 6, 23, 0.08);
        }
        .assessment-detail-eyebrow {
            color: #2563eb;
            font-size: 0.74rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 0.2rem;
        }
        .assessment-detail-title {
            color: #0f172a;
            font-size: 1.1rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
        }
        .assessment-detail-subtitle {
            color: #475569;
            font-size: 0.86rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    try:
        assessment_df = load_assessment_data()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.error(f"Unable to load assessment data: {exc}")
        return

    filtered_df = filter_assessment_scope(
        assessment_df,
        user=user,
        dashboard_level=dashboard_level,
    )

    scope_name = _scope_label(
        dashboard_level,
        filter_mode,
        selected_regions,
        selected_facilities,
        user,
    )

    if filtered_df.empty:
        st.info(f"No assessment records found for {scope_name}.")
        return

    maternal_tab, newborn_tab = st.tabs(["Maternal", "Newborn"])

    with maternal_tab:
        _render_record_type_tab(
            filtered_df,
            record_type="maternal",
            scope_name=scope_name,
            dashboard_level=dashboard_level,
            user=user,
            key_prefix=f"{dashboard_level}_assessment_maternal",
        )

    with newborn_tab:
        _render_record_type_tab(
            filtered_df,
            record_type="newborn",
            scope_name=scope_name,
            dashboard_level=dashboard_level,
            user=user,
            key_prefix=f"{dashboard_level}_assessment_newborn",
        )
