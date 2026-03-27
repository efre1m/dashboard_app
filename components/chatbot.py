import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import json
import re
import random
import difflib
import logging
import calendar
from datetime import datetime, timedelta
from utils.llm_utils import (
    query_llm,
    query_llm_detailed,
    generate_chatbot_insight,
    get_llm_provider_and_model,
    format_llm_label,
)
from utils.config import settings
from utils import kpi_utils
from utils.indicator_definitions import KPI_DEFINITIONS as MATERNAL_KPI_DEFINITIONS

# Import logic from dashboards to ensure data availability
from dashboards import facility, regional, national
import utils.kpi_utils as kpi_utils
from utils.kpi_utils import (
    prepare_data_for_trend_chart,
    compute_kpis
)
from utils.kpi_admitted_mothers import get_numerator_denominator_for_admitted_mothers
from utils.queries import get_facility_mapping_for_user, get_facilities_grouped_by_region, get_all_facilities_flat
from utils.dash_co import KPI_OPTIONS as MATERNAL_KPI_OPTIONS, KPI_MAPPING as MATERNAL_KPI_MAPPING
from components.chatbot_newborn import (
    get_newborn_chatbot_config,
    get_newborn_welcome_message,
)
from utils.chatbot_manual import (
    build_chatbot_manual_markdown,
    generate_chatbot_manual_doc_bytes,
    generate_chatbot_manual_pdf_bytes,
    get_chatbot_manual_sections,
)

KPI_OPTIONS = MATERNAL_KPI_OPTIONS
KPI_MAPPING = MATERNAL_KPI_MAPPING

PROGRAM_SELECTION_TERMS = {
    "maternal": {"maternal", "maternal health", "maternal indicators", "matenal", "materanl", "maternl", "materna", "mother", "mothers", "mothr", "mothrs", "mothres"},
    "newborn": {"newborn", "newborns", "newborn care", "newborn indicators", "neonatal", "neonate", "newbonr", "newbrn", "newborm", "newbron", "newbrons", "nebrn", "nebrons"},
}

MATERNAL_KPI_ALIASES = {
    "csection": "C-Section Rate (%)",
    "section": "C-Section Rate (%)",
    "caesarean": "C-Section Rate (%)",
    "cesarean": "C-Section Rate (%)",
    "cesarian": "C-Section Rate (%)",
    "ceasarean": "C-Section Rate (%)",
    "maternal death": "Maternal Death Rate (per 100,000)",
    "death": "Maternal Death Rate (per 100,000)",
    "mortality": "Maternal Death Rate (per 100,000)",
    "stillbirth": "Stillbirth Rate (%)",
    "stil birth": "Stillbirth Rate (%)",
    "stillbrith": "Stillbirth Rate (%)",
    "pph": "Postpartum Hemorrhage (PPH) Rate (%)",
    "hemorrhage": "Postpartum Hemorrhage (PPH) Rate (%)",
    "hemorage": "Postpartum Hemorrhage (PPH) Rate (%)",
    "hemorrage": "Postpartum Hemorrhage (PPH) Rate (%)",
    "bleeding": "Postpartum Hemorrhage (PPH) Rate (%)",
    "postpartum hemorrhage": "Postpartum Hemorrhage (PPH) Rate (%)",
    "uterotonic": "Delivered women who received uterotonic (%)",
    "uterotonc": "Delivered women who received uterotonic (%)",
    "utertonic": "Delivered women who received uterotonic (%)",
    "oxytocin": "Delivered women who received uterotonic (%)",
    "missing obstetric condition": "Missing Obstetric Condition at Delivery",
    "missing condition at delivery": "Missing Obstetric Condition at Delivery",
    "missing postpartum": "Missing Obstetric Condition at Delivery",
    "missing post": "Missing Obstetric Condition at Delivery",
    "missing complications diagnosis": "Missing Obstetric Complications Diagnosis",
    "missing obstetric complications": "Missing Obstetric Complications Diagnosis",
    "missing antepartum": "Missing Obstetric Complications Diagnosis",
    "missing ante": "Missing Obstetric Complications Diagnosis",
    "missing uterotonics given": "Missing Uterotonics Given at Delivery",
    "missing oxytocin given": "Missing Uterotonics Given at Delivery",
    "missing uterotonic at delivery": "Missing Uterotonics Given at Delivery",
    "missing uterotonic": "Missing Uterotonics Given at Delivery",
    "missing utertonic": "Missing Uterotonics Given at Delivery",
    "missing mode": "Missing Mode of Delivery",
    "missing birth": "Missing Birth Outcome",
    "missing outcome": "Missing Birth Outcome",
    "missing condition": "Missing Condition of Discharge",
    "missing discharge": "Missing Condition of Discharge",
    "ippcar": "Immediate Postpartum Contraceptive Acceptance Rate (IPPCAR %)",
    "contraceptive": "Immediate Postpartum Contraceptive Acceptance Rate (IPPCAR %)",
    "family planning": "Immediate Postpartum Contraceptive Acceptance Rate (IPPCAR %)",
    "family plainnig": "Immediate Postpartum Contraceptive Acceptance Rate (IPPCAR %)",
    "fp": "Immediate Postpartum Contraceptive Acceptance Rate (IPPCAR %)",
    "contraception": "Immediate Postpartum Contraceptive Acceptance Rate (IPPCAR %)",
    "pnc": "Early Postnatal Care (PNC) Coverage (%)",
    "postnatal": "Early Postnatal Care (PNC) Coverage (%)",
    "post natal": "Early Postnatal Care (PNC) Coverage (%)",
    "arv": "ARV Prophylaxis Rate (%)",
    "antiretroviral": "ARV Prophylaxis Rate (%)",
    "hiv prophylaxis": "ARV Prophylaxis Rate (%)",
    "assisted delivery": "Assisted Delivery Rate (%)",
    "assisted": "Assisted Delivery Rate (%)",
    "instrumental": "Assisted Delivery Rate (%)",
    "vacuum": "Assisted Delivery Rate (%)",
    "forceps": "Assisted Delivery Rate (%)",
    "svd": "Normal Vaginal Delivery (SVD) Rate (%)",
    "vaginal delivery": "Normal Vaginal Delivery (SVD) Rate (%)",
    "normal delivery": "Normal Vaginal Delivery (SVD) Rate (%)",
    "spontaneous": "Normal Vaginal Delivery (SVD) Rate (%)",
    "vaginal": "Normal Vaginal Delivery (SVD) Rate (%)",
    "episiotomy": "Episiotomy Rate (%)",
    "episitomy": "Episiotomy Rate (%)",
    "episotomy": "Episiotomy Rate (%)",
    "episotomi": "Episiotomy Rate (%)",
    "episiotmoy": "Episiotomy Rate (%)",
    "episo": "Episiotomy Rate (%)",
    "antepartum": "Antepartum Complications Rate (%)",
    "ante partum": "Antepartum Complications Rate (%)",
    "antipartum": "Antepartum Complications Rate (%)",
    "antenatal complications": "Antepartum Complications Rate (%)",
    "antenatal": "Antepartum Complications Rate (%)",
    "antepartum complication": "Antepartum Complications Rate (%)",
    "antepart": "Antepartum Complications Rate (%)",
    "antepar": "Antepartum Complications Rate (%)",
    "postpartum": "Postpartum Complications Rate (%)",
    "post partum": "Postpartum Complications Rate (%)",
    "postpartum complications": "Postpartum Complications Rate (%)",
    "postpar": "Postpartum Complications Rate (%)",
    "postpart": "Postpartum Complications Rate (%)",
    "postpra": "Postpartum Complications Rate (%)",
    "postpartum complicaiton": "Postpartum Complications Rate (%)",
    "postpartum complicaiotns": "Postpartum Complications Rate (%)",
    "postpartum complication": "Postpartum Complications Rate (%)",
    "admitted mothers": "Admitted Mothers",
    "admitted": "Admitted Mothers",
    "admissions": "Admitted Mothers",
    "admission": "Admitted Mothers",
    "total mothers": "Admitted Mothers",
    "mothrs": "Admitted Mothers",
    "mothr": "Admitted Mothers",
    "mothres": "Admitted Mothers",
    "enrollment": "Admitted Mothers",
    "total enrollments": "Admitted Mothers",
    "mothers": "Admitted Mothers",
    "total deliveries": "Total Deliveries",
    "deliveries": "Total Deliveries",
    "births": "Total Deliveries",
    "maternal coverage": "Maternal Coverage Rate",
    "maternal coverage rate": "Maternal Coverage Rate",
    "coverage of mothers": "Maternal Coverage Rate",
    "mothers coverage": "Maternal Coverage Rate",
}

MATERNAL_HELP_EXAMPLES = [
    "Plot C-Section Rate last year",
    "Show me Admitted Mothers",
    "Plot Maternal Coverage Rate last year",
    "Compare Admitted Mothers for Adigrat and Suhul",
    "Define PPH rate",
    "List maternal indicators",
]

NEWBORN_CHATBOT_CONFIG = get_newborn_chatbot_config()

MATERNAL_PROGRAM_HINTS = set(k.lower() for k in MATERNAL_KPI_ALIASES.keys()) | PROGRAM_SELECTION_TERMS["maternal"]
NEWBORN_PROGRAM_HINTS = set(k.lower() for k in NEWBORN_CHATBOT_CONFIG["kpi_aliases"].keys()) | PROGRAM_SELECTION_TERMS["newborn"]

PROGRAM_CONFIGS = {
    "maternal": {
        "program_key": "maternal",
        "label": "Maternal",
        "kpi_mapping": MATERNAL_KPI_MAPPING,
        "kpi_options": MATERNAL_KPI_OPTIONS,
        "kpi_aliases": MATERNAL_KPI_ALIASES,
        "kpi_definitions": MATERNAL_KPI_DEFINITIONS,
        "count_indicators": {"Admitted Mothers", "Total Deliveries"},
        "examples": MATERNAL_HELP_EXAMPLES,
    },
    "newborn": NEWBORN_CHATBOT_CONFIG,
}


def detect_program_from_text(text):
    query = re.sub(r"[^a-z0-9\\s]", " ", str(text or "").lower())
    query = re.sub(r"\s+", " ", query).strip()
    if not query:
        return None

    # 1) Explicit program terms always win.
    if any(term in query for term in PROGRAM_SELECTION_TERMS["newborn"]):
        return "newborn"
    if any(term in query for term in PROGRAM_SELECTION_TERMS["maternal"]):
        return "maternal"

    def _term_in_query(term: str) -> bool:
        term = str(term or "").strip().lower()
        if not term:
            return False
        if " " in term:
            return term in query
        return re.search(rf"\\b{re.escape(term)}\\b", query) is not None

    # 2) Heuristic tie-break: choose the program whose *best* matched hint is more specific (longer).
    # This prevents generic overlaps (e.g., "admission") from overriding newborn-specific terms
    # (e.g., "hypothermia") in phrases like "hypothermia on admission".
    best_newborn = 0
    for term in NEWBORN_PROGRAM_HINTS:
        if _term_in_query(term):
            best_newborn = max(best_newborn, len(term))

    best_maternal = 0
    for term in MATERNAL_PROGRAM_HINTS:
        if _term_in_query(term):
            best_maternal = max(best_maternal, len(term))

    if best_newborn and not best_maternal:
        return "newborn"
    if best_maternal and not best_newborn:
        return "maternal"
    if best_newborn > best_maternal:
        return "newborn"
    if best_maternal > best_newborn:
        return "maternal"
    return None


def get_program_selection_message(role):
    return (
        "**Which program do you want to analyze first?**\n\n"
        "Reply with `maternal` or `newborn`.\n\n"
        "After that, I will use the matching indicators, definitions, and charts for your current access level."
    )


def get_maternal_welcome_message(role):
    examples = "\n".join(f"- `{example}`" for example in MATERNAL_HELP_EXAMPLES)
    return (
        "**Maternal program selected.**\n\n"
        "I am now using maternal indicators and maternal dashboard logic for the facilities you can access.\n\n"
        "Available maternal indicators include C-Section, PPH, Maternal Death, Stillbirth, PNC, IPPCAR, uterotonic, admitted mothers, complications, and data quality indicators.\n\n"
        f"Try asking:\n{examples}\n\n"
        "Type `newborn` any time to switch programs."
    )


def get_program_welcome_message(role, program_key):
    if program_key == "newborn":
        return get_newborn_welcome_message(role)
    return get_maternal_welcome_message(role)

def ensure_data_loaded():
    """
    Optimized data loading that REUSES dashboard's cached data.
    """
    user = st.session_state.get("user", {})
    role = user.get("role", "")
    
    # Fallback: treat missing role as national to avoid first-run empty state
    if not role:
        user = {"role": "national"}
        role = "national"
    
    if not role:
        return None
    
    # TRY TO USE DASHBOARD'S ALREADY-LOADED DATA FIRST
    if role == "facility":
        # Check if facility dashboard has already loaded data
        if hasattr(st.session_state, "cached_shared_data_facility"):
            return st.session_state.cached_shared_data_facility
        
        maternal_df = (
            st.session_state.maternal_patients_df
            if hasattr(st.session_state, "maternal_patients_df")
            and not st.session_state.maternal_patients_df.empty
            else None
        )
        newborn_df = (
            st.session_state.newborn_patients_df
            if hasattr(st.session_state, "newborn_patients_df")
            and not st.session_state.newborn_patients_df.empty
            else None
        )

        if maternal_df is not None or newborn_df is not None:
            logging.info("✅ Chatbot: Using dashboard's already-loaded facility data")
            data = {}
            if maternal_df is not None:
                data["maternal"] = {"patients": maternal_df}
            if newborn_df is not None:
                data["newborn"] = {"patients": newborn_df}
            return data
    
    elif role == "regional":
        if hasattr(st.session_state, "cached_shared_data_regional"):
            return st.session_state.cached_shared_data_regional
        
        maternal_df = (
            st.session_state.regional_patients_df
            if hasattr(st.session_state, "regional_patients_df")
            and not st.session_state.regional_patients_df.empty
            else None
        )
        newborn_df = (
            st.session_state.newborn_patients_df
            if hasattr(st.session_state, "newborn_patients_df")
            and not st.session_state.newborn_patients_df.empty
            else None
        )

        if maternal_df is not None or newborn_df is not None:
            logging.info("✅ Chatbot: Using dashboard's already-loaded regional data")
            data = {}
            if maternal_df is not None:
                data["maternal"] = {"patients": maternal_df}
            if newborn_df is not None:
                data["newborn"] = {"patients": newborn_df}
            return data
    
    elif role == "national":
        if hasattr(st.session_state, "cached_shared_data_national"):
            return st.session_state.cached_shared_data_national
        
        maternal_df = (
            st.session_state.maternal_patients_df
            if hasattr(st.session_state, "maternal_patients_df")
            and not st.session_state.maternal_patients_df.empty
            else None
        )
        newborn_df = (
            st.session_state.newborn_patients_df
            if hasattr(st.session_state, "newborn_patients_df")
            and not st.session_state.newborn_patients_df.empty
            else None
        )

        if maternal_df is not None or newborn_df is not None:
            logging.info("✅ Chatbot: Using dashboard's already-loaded national data")
            data = {}
            if maternal_df is not None:
                data["maternal"] = {"patients": maternal_df}
            if newborn_df is not None:
                data["newborn"] = {"patients": newborn_df}
            return data
        if hasattr(st.session_state, "cached_shared_data"):
            return st.session_state.cached_shared_data
    
    # Fallback: load fresh data (should rarely happen)
    logging.warning("Chatbot: Loading fresh data (dashboard cache not available)")
    
    if role == "facility":
        with st.spinner("Initializing chatbot data access..."):
            static_data = facility.get_static_data_facility(user)
            program_uid_map = static_data["program_uid_map"]
            data = facility.get_shared_program_data_facility(user, program_uid_map, show_spinner=False)
            st.session_state.cached_shared_data_facility = data
            return data
    
    elif role == "regional":
        with st.spinner("Initializing chatbot data access..."):
            static_data = regional.get_static_data(user)
            program_uid_map = static_data["program_uid_map"]
            data = regional.get_shared_program_data_optimized(user, program_uid_map, show_spinner=False)
            st.session_state.cached_shared_data_regional = data
            return data
    
    elif role == "national":
        with st.spinner("Initializing chatbot data access..."):
            static_data = national.get_static_data(user)
            program_uid_map = static_data["program_uid_map"]
            data = national.get_shared_program_data_optimized(user, program_uid_map, show_spinner=False)
            st.session_state.cached_shared_data_national = data
            return data
            
    return None

class ChatbotLogic:
    def __init__(self, data):
        self.data = data
        self.user = st.session_state.get("user", {})
        self.facility_mapping = get_facility_mapping_for_user(self.user)
        # --- UNIVERSAL REGISTRY (FOR EXTRACTION) ---
        # Get ALL facilities regardless of user role for name detection purposes
        self.universal_facility_mapping = get_facility_mapping_for_user({"role": "national"})
        
        # Reverse mapping for easy lookup
        # Revised mapping to match current file state: self.uid_to_name
        self.uid_to_name = {v: k for k, v in self.universal_facility_mapping.items()}
        
        # --- MANUAL AMBIGUOUS FACILITY UID MAPPING ---
        # Add specific UIDs for ambiguous facilities
        self.AMBIGUOUS_FACILITY_UIDS = {
            "Ambo General Hospital": "LoTq3j2nraN",
            "Ambo university Hospital": "yhH6cXWdGgT",
            "Debere Markos referral hospital": "f0Uu4cbX6Oo",
            "Debre Tabor referral Hospital": "SVhhOYrCDnf",
            "Debrebirhan CSH": "D1a4DrXEGPF",
            "Debresina Primary Hospital": "UPMjQlkUysO",
            "Addis Alem PH": "cdYYBziDw9F",
            "Addis Zemen PH": "wNjwwsxFHtX"
        }
        
        # Update the universal mapping with these UIDs
        for name, uid in self.AMBIGUOUS_FACILITY_UIDS.items():
            if name in self.universal_facility_mapping:
                # Override the existing UID if different
                self.universal_facility_mapping[name] = uid
                self.uid_to_name[uid] = name
        
        # Maternal and Newborn data
        self.maternal_df = data.get("maternal", {}).get("patients", pd.DataFrame()).copy().reset_index(drop=True) if data.get("maternal") else pd.DataFrame()
        self.newborn_df = data.get("newborn", {}).get("patients", pd.DataFrame()).copy().reset_index(drop=True) if data.get("newborn") else pd.DataFrame()
        
        # Default to maternal for backward compatibility or if not specified
        self.df = self.maternal_df

        # --- COMPREHENSIVE TYPO DICTIONARY ---
        # Maps common misspellings to correct spellings
        self.COMMON_TYPOS = {
            # Episiotomy variations
            "episotomy": "episiotomy",
            "episotomi": "episiotomy",
            "episiotmoy": "episiotomy",
            "episotomie": "episiotomy",
            "episotomoy": "episiotomy",
            "episiotiomy": "episiotomy",
            "episotommy": "episiotomy",
            "episo": "episiotomy",
            
            # Hemorrhage variations
            "hemorage": "hemorrhage",
            "hemorrage": "hemorrhage",
            "hemmorhage": "hemorrhage",
            "hemmorrhage": "hemorrhage",
            "hemorrage": "hemorrhage",
            "hemmorage": "hemorrhage",
            "hemor": "hemorrhage",
            
            # Uterotonic variations
            "uterotoncic": "uterotonic",
            "uterotonc": "uterotonic",
            "utertonic": "uterotonic",
            "uterotoniic": "uterotonic",
            "uterotonnic": "uterotonic",
            "uterotnic": "uterotonic",
            
            # Stillbirth variations
            "still birth": "stillbirth",
            "stil birth": "stillbirth",
            "stillbrith": "stillbirth",
            "stil brith": "stillbirth",
            "stilbirth": "stillbirth",
            "stillbith": "stillbirth",
            
            # C-Section variations
            "c section": "csection",
            "c-section": "csection",
            "cesarian": "csection",
            "ceasarean": "csection",
            "cesarean": "csection",
            "caesarean": "csection",
            "sectioin": "section",
            "sectionn": "section",
            "cesearian": "csection",
            "cesection": "csection",
            
            # Antepartum variations
            "ante partum": "antepartum",
            "antipartum": "antepartum",
            "antepartam": "antepartum",
            "antpartem": "antepartum",
            
            # Postpartum variations
            "post partum": "postpartum",
            "postpartem": "postpartum",
            "postpartam": "postpartum",
            
            # ARV variations
            "anti retroviral": "antiretroviral",
            "antiretro viral": "antiretroviral",
            "antiretro": "antiretroviral",
            
            # Hypothermia variations (newborn)
            "hypthermia": "hypothermia",
            "hypothemia": "hypothermia",
            "hypo thermia": "hypothermia",
            "hipothermia": "hypothermia",
            "hypothrmia": "hypothermia",
            
            # Missing variations
            "misssing": "missing",
            "mising": "missing",
            "mssing": "missing",
            
            # Neonatal variations
            "neo natal": "neonatal",
            "neonatel": "neonatal",
            "neonate": "neonatal",
            
            # CPAP variations
            "c pap": "cpap",
            "c-pap": "cpap",
            
            # KMC variations
            "kangaro": "kangaroo",
            "kangaroo care": "kmc",
            "skin to skin": "kmc",
            
            # General typos
            "sitllbirth": "stillbirth",
            "sitllbirht": "stillbirth",
            "stillbirht": "stillbirth",
            "birht": "birth",
            "wieght": "weight",
            "weigth": "weight",
            "weigt": "weight",
            "mothrs": "mothers",
            "mothr": "mothers",
            "mothres": "mothers",
            "newbron": "newborn",
            "newbonr": "newborn",
            "newbrn": "newborn",
            "newborm": "newborn",
            "newbrons": "newborns",
            "nebrn": "newborn",
            "nebrons": "newborn",
            "oucome": "outcome",
            "indicatofrs": "indicators",
            "totaly": "totally",
            "wome": "women",
            "whor": "who",
            "abot": "about",
            "abut": "about",
            "hte": "the",
            "eht": "the",
            "teh": "the",
            "enrollmet": "enrollment",
            "admited": "admitted",
            "admision": "admission",
            "admittted": "admitted",
            "admitteed": "admitted",
            "adimtted": "admitted",
            "admitd": "admitted",
            "delivry": "delivery",
            "vagnal": "vaginal",
            "materna": "maternal",
            "materanl": "maternal",
            "matenal": "maternal",
            "unversity": "university",
            "univeristy": "university",
            "hospitel": "hospital",
            "changi": "chagni",
            "dangla": "dangela",
            "plainnig": "planning",
            "postpar": "postpartum",
            "postpart": "postpartum",
            "postpra": "postpartum",
            "antepar": "antepartum",
            "antepart": "antepartum",
            "complicaiton": "complication",
            "complicaiotns": "complications",
            "dist": "distribution",
            
            # Intent/Action Variations
            "inidcatorys": "indicators",
            "indicatorys": "indicators",
            "indicater": "indicator",
            "indicatry": "indicator",
            "indicotrs": "indicators",
            "indcators": "indicators",
            "indictors": "indicators",
            "indictors": "indicators",
            "lsit": "list",
            "listt": "list",
            "shw": "show",
            "sho": "show",
            "reioings": "regions",
            "reigons": "regions",
            "reginos": "regions",
            "faciltiy": "facility",
            "faciliteis": "facilities",
            "facilites": "facilities",
            "faciltiyies": "facilities",
            "faclities": "facilities",
            "fac": "facility",
            "facs": "facilities",
            "regijon": "region",
            "regjon": "region",
            
            # Shorthand & Comparison Triggers
            "all fac": "all facilities",
            "all reg": "all regions",
            "per fac": "per facility",
            "per reg": "per region",
            "all facs": "all facilities",
            "all regs": "all regions",
            "compair": "compare",
            "compr": "compare",
            "vs": " vs ",
        }

        # --- FACILITY RESOLUTION ENGINE ---
        # 1. Normalized Mapping (for direct exact matches)
        self.normalized_facility_mapping = {re.sub(r'\s+', ' ', k).strip().lower(): v for k, v in self.universal_facility_mapping.items()}
        
        # 2. Suffix-Blind Mapping (e.g. "Ambo University" -> "Ambo University Hospital")
        self.suffix_blind_mapping = {}
        suffixes = [' hospital', ' primary hospital', ' general hospital', ' referral hospital', ' teaching hospital', ' specialized hospital', ' ph', ' gh', ' csh']
        for k in self.universal_facility_mapping.keys():
            k_norm = re.sub(r'\s+', ' ', k).strip().lower()
            self.suffix_blind_mapping[k_norm] = k
            for s in suffixes:
                if k_norm.endswith(s):
                    stripped = k_norm[: -len(s)].strip()
                    if stripped not in self.suffix_blind_mapping:
                        self.suffix_blind_mapping[stripped] = k
        
        # 3. First-Word Disambiguation Index (for fragments like "Ambo")
        self.facility_search_index = {}
        for full_name in self.universal_facility_mapping.keys():
             clean_name = re.sub(r'\s+', ' ', full_name).strip()
             first_word = clean_name.split(' ')[0].lower()
             if first_word not in self.facility_search_index:
                  self.facility_search_index[first_word] = []
             self.facility_search_index[first_word].append(full_name)
        
        # 5. Ambiguous Prefixes (Terms that are too generic to match directly if found alone)
        self.AMBIGUOUS_PREFIXES = ["ambo", "debre", "st", "saint", "mary", "kidane", "black", "lion", "felege", "gandhi", "yekatit", "alert", "menelik", "paul", "peter"]
        
        # 3.5 Unique first-word aliases (safe only when first word is unique and non-ambiguous)
        for first_word, names in self.facility_search_index.items():
            if (
                len(names) == 1
                and first_word not in self.AMBIGUOUS_PREFIXES
                and first_word not in self.suffix_blind_mapping
            ):
                # Add a single-word alias so "adigrat" resolves even without suffix
                self.suffix_blind_mapping[first_word] = names[0]

        # 4. Greedy Scan List (Sorted by length to catch longest phrases first)
        # We use keys from suffix_blind_mapping because it contains BOTH full names AND short names
        self.sorted_scan_names = sorted(self.suffix_blind_mapping.keys(), key=len, reverse=True)
        
        # 6. Add special handling for full ambiguous facility names with their UIDs
        for full_name in self.AMBIGUOUS_FACILITY_UIDS.keys():
            norm_name = re.sub(r'\s+', ' ', full_name).strip().lower()
            if norm_name not in self.suffix_blind_mapping:
                self.suffix_blind_mapping[norm_name] = full_name

        # Compact index (remove spaces) to handle typos like "abiadi" -> "Abi Adi Hospital".
        # Values can collide (different facilities compacting to the same key), so we store lists.
        _compact_tmp: dict[str, set[str]] = {}
        for k_norm, official in self.suffix_blind_mapping.items():
            compact = str(k_norm or "").replace(" ", "")
            if not compact:
                continue
            if compact not in _compact_tmp:
                _compact_tmp[compact] = set()
            _compact_tmp[compact].add(str(official))
        self.suffix_blind_compact_index: dict[str, list[str]] = {
            k: sorted(v) for k, v in _compact_tmp.items()
        }
        self.suffix_blind_compact_keys = list(self.suffix_blind_compact_index.keys())
        
        logging.info(f"🏥 Facility Resolution Engine Initialized: {len(self.normalized_facility_mapping)} direct, {len(self.suffix_blind_mapping)} suffix-blind, {len(self.facility_search_index)} index groups.")


        # --- SPECIALIZED KPI MAPPING ---
        # Maps full KPI names to their internal utility script suffixes
        # Keys MUST match active_kpi_name (which comes from KPI_MAPPING/KPI_OPTIONS)
        self.SPECIALIZED_KPI_MAP = {
            # Maternal Indicators
            "Total Admitted Mothers": "admitted_mothers",
            "Admitted Mothers": "admitted_mothers",
            "Maternal Coverage Rate": "maternal_coverage_rate",
            "Postpartum Hemorrhage (PPH) Rate (%)": "pph",
            "Normal Vaginal Delivery (SVD) Rate (%)": "svd",
            "ARV Prophylaxis Rate (%)": "arv",
            "Assisted Delivery Rate (%)": "assisted",
            "Delivered women who received uterotonic (%)": "uterotonic",
            "Missing Birth Outcome": "missing_bo",
            "Missing Condition of Discharge": "missing_cod",
            "Missing Mode of Delivery": "missing_md",
            "Missing Obstetric Condition at Delivery": "missing_postpartum",
            "Missing Obstetric Complications Diagnosis": "missing_antepartum",
            "Missing Uterotonics Given at Delivery": "missing_uterotonic",
            "Episiotomy Rate (%)": "episiotomy",
            "Antepartum Complications Rate (%)": "antipartum_compl",
            "Postpartum Complications Rate (%)": "postpartum_compl",
            
            # Newborn Indicators
            "Inborn Rate (%)": "newborn",
            "Outborn Rate (%)": "newborn",
            "Hypothermia on Admission Rate (%)": "newborn",
            "Inborn Hypothermia Rate (%)": "newborn",
            "Outborn Hypothermia Rate (%)": "newborn",
            "Neonatal Mortality Rate (%)": "newborn",
            "Admitted Newborns": "newborn",
            "Newborn Coverage Rate": "newborn",
            "Birth Weight Rate": "newborn_simplified",
            "KMC Coverage by Birth Weight": "newborn_simplified",
            "General CPAP Coverage": "newborn_simplified",
            "CPAP for RDS": "newborn_simplified",
            "CPAP Coverage by Birth Weight": "newborn_simplified",
            "Missing Temperature (%)": "newborn",
            "Missing Birth Weight (%)": "newborn",
            "Missing Discharge Status (%)": "newborn",
            "Missing Status of Discharge (%)": "newborn",
            "Missing Birth Location (%)": "newborn",

            # Standard KPIs that use kpi_utils
            "C-Section Rate (%)": "utils",
            "Maternal Death Rate (per 100,000)": "utils",
            "Stillbirth Rate (%)": "utils",
            "Early Postnatal Care (PNC) Coverage (%)": "utils",
            "Immediate Postpartum Contraceptive Acceptance Rate (IPPCAR %)": "utils"
        }

    def get_selected_program(self):
        return st.session_state.get("chatbot_program")

    def get_program_config(self, program_key=None):
        active_program = program_key or self.get_selected_program() or "maternal"
        return PROGRAM_CONFIGS.get(active_program, PROGRAM_CONFIGS["maternal"])

    def apply_active_program_globals(self, program_key=None):
        global KPI_MAPPING, KPI_OPTIONS

        config = self.get_program_config(program_key)
        KPI_MAPPING = config["kpi_mapping"]
        KPI_OPTIONS = config["kpi_options"]
        self.df = self.newborn_df if config["program_key"] == "newborn" else self.maternal_df
        return config

    def set_selected_program(self, program_key, *, clear_context: bool = True):
        if program_key not in PROGRAM_CONFIGS:
            return False

        previous_program = self.get_selected_program()
        st.session_state["chatbot_program"] = program_key
        if clear_context:
            st.session_state["chatbot_context"] = {}
        self.apply_active_program_globals(program_key)
        return previous_program != program_key

    def is_program_selection_only(self, query):
        query_lower = re.sub(r"\s+", " ", str(query or "").lower()).strip()
        if not query_lower:
            return False

        analysis_tokens = [
            "plot",
            "show",
            "compare",
            "define",
            "what",
            "how",
            "trend",
            "rate",
            "chart",
            "graph",
            "table",
            "list",
            "value",
            "count",
            "number",
            "indicator",
            "kpi",
        ]
        return not any(token in query_lower for token in analysis_tokens)

    def detect_cross_program(self, query, current_program):
        other_program = "newborn" if current_program == "maternal" else "maternal"
        other_aliases = self.get_program_config(other_program)["kpi_aliases"]
        normalized_query = re.sub(r"\s+", " ", str(query or "").lower()).strip()

        for alias in sorted(other_aliases.keys(), key=len, reverse=True):
            if alias in normalized_query:
                return other_program
        return None

    def _silent_prepare_data(self, df, kpi_name, facility_uids=None, date_range_filters=None):
        """
        Use EXACT SAME logic as dashboard's national.py for consistency and performance.
        Optimized for batch processing.
        """
        # Import dashboard functions
        from utils.dash_co import normalize_patient_dates
        
        if df.empty:
            return pd.DataFrame(), None
        
        # STEP 1: Filter by UIDs FIRST (EXACTLY like dashboard)
        working_df = df.copy()
        
        if facility_uids and "orgUnit" in working_df.columns:
            working_df = working_df[working_df["orgUnit"].isin(facility_uids)].copy()
            logging.info(f"Chatbot: Filtered to {len(working_df)} rows for {len(facility_uids)} facilities")
        
        # STEP 2: Use normalize_patient_dates (EXACTLY like dashboard)
        working_df = normalize_patient_dates(working_df)
        
        if working_df.empty:
            logging.info(f"Chatbot: No data after normalization")
            return pd.DataFrame(), "enrollment_date"
        
        # STEP 3: Apply date filters SIMPLY (no assign_period here!)
        if date_range_filters:
            start_date = date_range_filters.get("start_date")
            end_date = date_range_filters.get("end_date")
            
            if start_date and end_date:
                # Filter by date range directly
                mask = (
                    (working_df["enrollment_date"] >= pd.Timestamp(start_date)) &
                    (working_df["enrollment_date"] <= pd.Timestamp(end_date))
                )
                working_df = working_df[mask].copy()
                logging.info(f"Chatbot: Date filtered to {len(working_df)} rows")
        
        # STEP 4: Add period columns ONLY ONCE at the end (not per facility!)
        # We'll let the calling function handle this for the ENTIRE dataset
        return working_df, "enrollment_date"


    def parse_query(self, query):
        """
        Parses the user query to extract intent, kpi, filters.
        Tries LLM first, falls back to regex.
        """
        query_lower = query.lower()

        llm_enabled = bool(getattr(settings, "CHATBOT_USE_LLM", False))
        llm_provider, llm_model = get_llm_provider_and_model()
        self.last_parse_trace = {
            "enabled": llm_enabled,
            "provider": llm_provider,
            "model": llm_model,
            "parser_mode": str(getattr(settings, "CHATBOT_LLM_PARSER_MODE", "fallback") or "fallback").strip().lower(),
            "attempted": False,
            "used": False,
            "failed": False,
            "error": None,
        }
        
        # --- PRE-PROCESS QUERY (Normalization & Typo Correction) ---
        # Move this to the top so it filters early keyword checks
        query_norm = re.sub(r'[^a-z0-9\s]', '', query_lower)
        for typo, correct in self.COMMON_TYPOS.items():
            # Use word-boundary matching to avoid corrupting valid words
            # e.g. 'missin' should NOT match inside 'missing'
            query_norm = re.sub(r'\b' + re.escape(typo) + r'\b', correct, query_norm)
        query_norm = re.sub(r'\s+', ' ', query_norm).strip()
        active_program = self.get_selected_program() or "maternal"
        active_config = self.get_program_config(active_program)
        active_kpi_mapping = active_config["kpi_mapping"]
        active_kpi_options = active_config["kpi_options"]

        # --- MISSING BIRTH OUTCOME (ISOLATED FIX ONLY) ---
        # If the query contains BOTH "birth" and a variation of "outcome/out come",
        # always lock the metric to "Missing Birth Outcome" (never substitute, never reuse context).
        _compact_query = query_norm.replace(" ", "")
        missing_birth_outcome_lock = active_program == "maternal" and ("birth" in _compact_query and "outcome" in _compact_query)
        
        # Initialize variables for parsing
        selected_facility_uids = []
        selected_facility_names = []
        found_regions = []
        region_filter = None
        assistant_response = None
        
        # --- AMBIGUITY RESOLUTION (Numeric Selection) ---
        context = st.session_state.get("chatbot_context", {})
        ambiguity_options = context.get("ambiguity_options")
        
        if ambiguity_options:
            selection = query_norm
            # Handle "option 1", "number 1", or just "1"
            if "option" in selection:
                selection = selection.replace("option", "").strip()
            if "number" in selection:
                selection = selection.replace("number", "").strip()

            def _norm_text(val):
                return re.sub(r"\s+", " ", str(val or "")).strip().lower()

            selected_name = None
            if selection in ambiguity_options:
                selected_name = ambiguity_options[selection]
            else:
                # Allow the user to reply with the facility name instead of a number
                sel_norm = _norm_text(query)
                for opt_name in ambiguity_options.values():
                    if _norm_text(opt_name) == sel_norm:
                        selected_name = opt_name
                        break

            if selected_name:
                # Clear ambiguity options + pending context (we are fulfilling the original request)
                st.session_state.chatbot_context["ambiguity_options"] = None

                pending_parse = context.get("pending_parse")
                if not isinstance(pending_parse, dict):
                    pending_parse = {}

                pending_kpi = pending_parse.get("kpi") or context.get("pending_kpi")
                if isinstance(pending_kpi, str):
                    raw_kpi = pending_kpi.strip()
                    if raw_kpi and raw_kpi not in active_kpi_mapping:
                        ci_map = {k.lower(): k for k in active_kpi_mapping.keys()}
                        exact_ci = ci_map.get(raw_kpi.lower())
                        if exact_ci:
                            pending_kpi = exact_ci
                        else:
                            matches = difflib.get_close_matches(
                                raw_kpi,
                                list(active_kpi_mapping.keys()),
                                n=1,
                                cutoff=0.85,
                            )
                            if matches:
                                pending_kpi = matches[0]

                uid = self.universal_facility_mapping.get(selected_name)

                # Mark the LLM trace as used if we already parsed the original request via LLM
                if context.get("pending_llm_used") or pending_parse:
                    self.last_parse_trace["attempted"] = True
                    self.last_parse_trace["used"] = True

                for key in ("pending_parse", "pending_kpi", "pending_llm_used", "pending_query"):
                    if key in st.session_state.chatbot_context:
                        st.session_state.chatbot_context[key] = None

                resolved_intent = pending_parse.get("intent")
                if resolved_intent not in {
                    "plot",
                    "distribution",
                    "text",
                    "definition",
                    "metadata_query",
                    "chat",
                    "list_kpis",
                    "clear",
                }:
                    resolved_intent = None

                # Facility disambiguation is only triggered when we detected a location token.
                # If we have a KPI (or KPI-like intent), never allow the follow-up selection to derail
                # into metadata listing.
                if isinstance(pending_kpi, str) and pending_kpi.strip():
                    if resolved_intent in {None, "chat", "metadata_query", "list_kpis"}:
                        resolved_intent = "plot"
                if resolved_intent is None:
                    resolved_intent = "plot"

                return {
                    "intent": resolved_intent,
                    "chart_type": pending_parse.get("chart_type") or "line",
                    "orientation": pending_parse.get("orientation") or None,
                    "analysis_type": pending_parse.get("analysis_type") or None,
                    "kpi": pending_kpi if isinstance(pending_kpi, str) else None,
                    "facility_uids": [uid] if uid else [],
                    "facility_names": [selected_name],
                    "date_range": pending_parse.get("date_range") if isinstance(pending_parse.get("date_range"), dict) else None,
                    "period_label": pending_parse.get("period_label") if pending_parse.get("period_label") in {"Daily", "Weekly", "Monthly", "Quarterly", "Yearly"} else None,
                    "entity_type": pending_parse.get("entity_type") if pending_parse.get("entity_type") in {"region", "facility"} else None,
                    "region_filter": pending_parse.get("region_filter") if isinstance(pending_parse.get("region_filter"), str) else None,
                    "count_requested": pending_parse.get("count_requested") if isinstance(pending_parse.get("count_requested"), bool) else False,
                    "comparison_mode": pending_parse.get("comparison_mode") if isinstance(pending_parse.get("comparison_mode"), bool) else False,
                    "comparison_entity": pending_parse.get("comparison_entity") if pending_parse.get("comparison_entity") in {"region", "facility"} else None,
                    "comparison_targets": pending_parse.get("comparison_targets") if isinstance(pending_parse.get("comparison_targets"), list) else [],
                    "is_drill_down": bool(pending_parse.get("is_drill_down")) if isinstance(pending_parse.get("is_drill_down"), bool) else False,
                    "response": pending_parse.get("response") if isinstance(pending_parse.get("response"), str) else None,
                    "fulfillment_requested": True,
                }
        
        # Handle follow-up responses FIRST (before LLM)
        if query_norm in ["yes", "yeah", "sure", "ok", "okay", "yep", "do it"]:
            context = st.session_state.get("chatbot_context", {})
            last_question = context.get("last_question", "")
            if "region" in last_question:
                return {"intent": "metadata_query", "entity_type": "region", "count_requested": False, "fulfillment_requested": True}
            elif "facility" in last_question or "facilities" in last_question:
                # Check if specific region was in last question
                target_reg = None
                if "in " in last_question:
                    target_reg = last_question.split("in ")[-1]
                return {"intent": "metadata_query", "entity_type": "facility", "count_requested": False, "region_filter": target_reg, "fulfillment_requested": True}
            elif "indicator" in last_question:
                return {"intent": "list_kpis"}
            elif last_question == "plot_kpi" and context.get("pending_kpi"):
                 return {
                     "intent": "plot",
                     "kpi": context.get("pending_kpi"),
                     "fulfillment_requested": True
                 }
            elif "mean" in last_question and "facility" in last_question:
                # Handle facility disambiguation pick
                picked_fac = query.strip()
                if picked_fac in self.universal_facility_mapping:
                     return {
                         "intent": "plot",
                         "kpi": context.get("pending_kpi"),
                         "facility_uids": [self.universal_facility_mapping[picked_fac]],
                         "facility_names": [picked_fac],
                         "fulfillment_requested": True
                     }
        
        
        # Handle explicit list requests - OPTIMIZED FOR PLURALS & VARIATIONS
        # NOTE: Keep this detection STRICT. Users often mention "<region> region" inside KPI
        # plot requests, so we only treat it as a metadata list when the query explicitly starts
        # as a list request (e.g., "list regions", "show facilities", "display indicators").
        explicit_list_intent = None  # "metadata_query" | "list_kpis" | None
        explicit_list_entity_type = None  # "region" | "facility" | None
        explicit_list_count_requested = False
        explicit_list_region_filter = None

        if re.search(
            r"^\s*(please\s+)?(list|show|shw|sho|display|tell)(\s+me)?(\s+all)?\s+(facilities|facility|hospitals?|facilti?yies?|faclities?|units?)\b",
            query_norm,
        ):
            explicit_list_entity_type = "facility"
        elif re.search(
            r"^\s*(please\s+)?(list|show|shw|sho|display|tell)(\s+me)?(\s+all)?\s+(regions?|reioings?|reigons?|reginos?|territory|territories)\b",
            query_norm,
        ):
            explicit_list_entity_type = "region"

        # PRIORITY CHECK: If KPI is present, do NOT treat as metadata list
        # "Show C-Section Rate by facility" -> PLOT intent, not LIST intent
        active_aliases = active_config.get("kpi_aliases") or {}
        kpi_found_early = (
            any(kpi.lower() in query_norm for kpi in active_kpi_mapping.keys())
            or any(opt.lower() in query_norm for opt in active_kpi_options)
            or any(str(alias).lower() in query_norm for alias in active_aliases.keys())
            or missing_birth_outcome_lock
        )

        if explicit_list_entity_type and not kpi_found_early:
            explicit_list_intent = "metadata_query"
            explicit_list_count_requested = "how many" in query_lower

            # Extract Region if mentioned (for "list facilities in <region>")
            regions_data = get_facilities_grouped_by_region(self.user)
            for r_name in regions_data.keys():
                if r_name.lower() in query_norm:
                    explicit_list_region_filter = r_name
                    break

        if re.search(
            r"^\s*(please\s+)?(list|show|shw|sho|display|tell)(\s+me)?(\s+all)?\s+(indicators?|indicaters?|kpis?|measures?|metrics?)\b",
            query_norm,
        ):
            explicit_list_intent = "list_kpis"
        
        # --- GREEDY FACILITY SCAN (Prioritize Full Names in Query) ---
        # Instead of just relying on the LLM, scan the RAW query for any known facility names

        # RE-IMPL GREEDY SCAN with Ambiguity Check
        greedy_matches = []
        # Use normalized query so common typos (e.g. "unversity", "hospitel") still match facilities.
        q_norm_for_scan = query_norm
        
        for norm_name in self.sorted_scan_names:
            if norm_name in q_norm_for_scan:
                # AMBIGUITY CHECK START
                if norm_name in self.AMBIGUOUS_PREFIXES:
                    # Safety Net: Check if this ambiguous term is part of a LONGER facility name 
                    # that is ALSO in the query (and hence hasn't been matched/consumed yet).
                    # This protects against sorting issues or partial matches.
                    is_part_of_larger = False
                    for existing_long_name in self.sorted_scan_names:
                        if len(existing_long_name) <= len(norm_name):
                            break # List is sorted by length DESC, so we can stop early
                        if norm_name in existing_long_name and existing_long_name in q_norm_for_scan:
                            is_part_of_larger = True
                            logging.info(f"Skipping ambiguity '{norm_name}' because longer match '{existing_long_name}' is detected in query.")
                            break
                    
                    if is_part_of_larger:
                        continue
                        
                    # Check if this "short" name has confusion in the index
                    first_word = norm_name.split(' ')[0] # usually itself
                    hits = self.facility_search_index.get(first_word, [])
                    if len(hits) > 1:
                        # FOUND AMBIGUOUS TERM (e.g. "ambo")
                        options_map = {str(i+1): m for i, m in enumerate(hits[:8])}
                        st.session_state.chatbot_context["ambiguity_options"] = options_map
                        st.session_state.chatbot_context["pending_query"] = query
                        st.session_state.chatbot_context["pending_parse"] = None
                        st.session_state.chatbot_context["pending_kpi"] = None
                        st.session_state.chatbot_context["pending_llm_used"] = False

                        # Try to parse the original intent/KPI via LLM so the follow-up selection
                        # can immediately execute the user's request.
                        llm_mode_early = self.last_parse_trace.get("parser_mode") or "fallback"
                        if llm_enabled and llm_mode_early in {"always", "fallback"}:
                            self.last_parse_trace["attempted"] = True
                            llm_regions_list = None
                            try:
                                llm_regions_list = list(get_facilities_grouped_by_region(self.user).keys())
                            except Exception:
                                llm_regions_list = None

                            llm_result, llm_error = query_llm_detailed(
                                query,
                                facilities_list=list(self.facility_mapping.items()),
                                regions_list=llm_regions_list,
                                kpi_mapping=active_kpi_mapping,
                                program_label=active_config.get("label"),
                            )
                            if llm_error:
                                self.last_parse_trace["failed"] = True
                                self.last_parse_trace["error"] = llm_error
                            elif isinstance(llm_result, dict):
                                self.last_parse_trace["used"] = True
                                st.session_state.chatbot_context["pending_llm_used"] = True
                                st.session_state.chatbot_context["pending_parse"] = llm_result

                                llm_kpi = llm_result.get("kpi")
                                if isinstance(llm_kpi, str) and llm_kpi.strip():
                                    raw_candidate = llm_kpi.strip()
                                    resolved = None
                                    if raw_candidate in active_kpi_mapping:
                                        resolved = raw_candidate
                                    else:
                                        ci_map = {k.lower(): k for k in active_kpi_mapping.keys()}
                                        exact_ci = ci_map.get(raw_candidate.lower())
                                        if exact_ci:
                                            resolved = exact_ci
                                        else:
                                            matches = difflib.get_close_matches(
                                                raw_candidate,
                                                list(active_kpi_mapping.keys()),
                                                n=1,
                                                cutoff=0.85,
                                            )
                                            if matches:
                                                resolved = matches[0]

                                    if resolved:
                                        st.session_state.chatbot_context["pending_kpi"] = resolved

                        # Lightweight fallback KPI detection (aliases) if LLM is off/failed.
                        if not st.session_state.chatbot_context.get("pending_kpi"):
                            alias_map = active_config.get("kpi_aliases") or {}
                            best_alias = None
                            best_kpi = None
                            for alias, kpi_name in alias_map.items():
                                a = str(alias or "").lower()
                                if a and a in query_norm and (best_alias is None or len(a) > len(best_alias)):
                                    best_alias = a
                                    best_kpi = kpi_name
                            if isinstance(best_kpi, str) and best_kpi in active_kpi_mapping:
                                st.session_state.chatbot_context["pending_kpi"] = best_kpi

                        options_str = "\n".join([f"{i}. {m}" for i, m in options_map.items()])
                        return {
                            "intent": "chat",
                            "response": (
                                f"Which **{norm_name.capitalize()}** do you mean?\n\n"
                                f"{options_str}\n\n"
                                "Reply with a number (e.g. `1`) or type the full facility name."
                            ),
                            "pending_kpi": st.session_state.chatbot_context.get("pending_kpi"),
                        }
                # AMBIGUITY CHECK END
                
                official_name = self.suffix_blind_mapping[norm_name]
                uid = self.universal_facility_mapping[official_name]
                greedy_matches.append((official_name, uid))
                q_norm_for_scan = q_norm_for_scan.replace(norm_name, " [MATCHED] ")

        if greedy_matches:
            logging.info(f"🚀 Greedy Scan Found: {[m[0] for m in greedy_matches]}")

        # --- EXPLICIT FACILITY DETECTION (Use for Scope Protection) ---
        # Strong direct matches (name fully contained in query_norm)
        strong_matches = []
        for name, uid in self.universal_facility_mapping.items():
            n_lower = name.lower()
            if n_lower in query_norm:
                strong_matches.append((name, uid))

        # Facility explicitly mentioned ONLY if a facility name is found
        facility_explicitly_mentioned = bool(greedy_matches or strong_matches)

        # Optional LLM fallback is applied near the end of parsing (only to fill missing slots),
        # keeping facility resolution + access control deterministic.

        # --- FALLBACK TO REGEX / FUZZY MATCHING (Existing Logic) ---
        # query_lower is already defined at top of function
        
        # Check for Clear Chat
        if "clear chat" in query_lower or "reset chat" in query_lower:
            return {"intent": "clear"}
        
        # 1. Detect Intent and Chart Type
        intent = "text"
        chart_type = "line" # Default
        entity_type = None
        count_requested = False
        comparison_mode = False
        comparison_entity = None # Initialize to prevent UnboundLocalError
        if any(w in query_lower for x in ["distribution", "breakdown", "pie", "proportion", "dist", "by category", "by group"] for w in [x]):
            intent = "distribution"
        elif any(
            w in query_lower
            for w in [
                "plot",
                "graph",
                "chart",
                "trend",
                "visualize",
                "show me",
                "show",
                "see",
                "display",
                "view",
            ]
        ):
            intent = "plot"
        elif any(w in query_lower for w in ["define", "meaning", "definition", "how is", "computed", "calculation", "formula"]):
            intent = "definition"
            
        # INTEGRATE GREEDY MATCHES (No wipe!)
        if greedy_matches:
            for name, uid in greedy_matches:
                 selected_facility_names.append(name)
                 selected_facility_uids.append(uid)

        # INTEGRATE STRONG MATCHES (No wipe!)
        if strong_matches:
            for name, uid in strong_matches:
                if uid not in selected_facility_uids:
                    selected_facility_uids.append(uid)
                    selected_facility_names.append(name)
        
        # --- COMPARISON CATCH-UP: try to resolve additional facilities mentioned with "and"/commas ---
        if comparison_mode and len(selected_facility_uids) < 2:
            # First-word fallback: if a single unique facility shares the first word, select it
            if len(selected_facility_uids) < 2:
                for word in query_norm.split():
                    if len(word) < 3:
                        continue
                    candidates = self.facility_search_index.get(word)
                    if candidates:
                        if len(candidates) == 1:
                            official = candidates[0]
                        else:
                            # Pick the closest match among candidates instead of bailing
                            lower_candidates = [c.lower() for c in candidates]
                            best = difflib.get_close_matches(word, lower_candidates, n=1, cutoff=0.6)
                            official = candidates[lower_candidates.index(best[0])] if best else None
                        if official:
                            uid = self.universal_facility_mapping.get(official)
                            if uid and uid not in selected_facility_uids:
                                selected_facility_uids.append(uid)
                                selected_facility_names.append(official)
                    if len(selected_facility_uids) >= 2:
                        break

            # Best-match fallback: pick closest facility name for stray tokens (even if not unique)
            if len(selected_facility_uids) < 2:
                norm_names_all = list(self.normalized_facility_mapping.keys())
                for word in query_norm.split():
                    if len(word) < 4:  # skip very short/ambiguous tokens
                        continue
                    matches = difflib.get_close_matches(word, norm_names_all, n=1, cutoff=0.68)
                    if matches:
                        official = self.suffix_blind_mapping.get(matches[0], self.normalized_facility_mapping.get(matches[0], matches[0]))
                        uid = self.universal_facility_mapping.get(official)
                        if uid and uid not in selected_facility_uids:
                            selected_facility_uids.append(uid)
                            selected_facility_names.append(official)
                    if len(selected_facility_uids) >= 2:
                        break

            tokens = re.split(r"[,&]| and ", query_norm)
            norm_keys = list(self.suffix_blind_mapping.keys())
            for tok in tokens:
                t = tok.strip()
                if len(t) < 3:
                    continue
                matches = difflib.get_close_matches(t, norm_keys, n=1, cutoff=0.72)
                if matches:
                    official = self.suffix_blind_mapping[matches[0]]
                    uid = self.universal_facility_mapping.get(official)
                    if uid and uid not in selected_facility_uids:
                        selected_facility_uids.append(uid)
                        selected_facility_names.append(official)
            # Additional scan: pick any facility name substring present in query
            if len(selected_facility_uids) < 2:
                for norm_name in self.sorted_scan_names:
                    if norm_name in query_norm:
                        official = self.suffix_blind_mapping[norm_name]
                        uid = self.universal_facility_mapping.get(official)
                        if uid and uid not in selected_facility_uids:
                            selected_facility_uids.append(uid)
                            selected_facility_names.append(official)
                    if len(selected_facility_uids) >= 2:
                        break
            # Final fuzzy fallback across all facilities when still <2
            if len(selected_facility_uids) < 2:
                all_names = list(self.normalized_facility_mapping.keys())
                matches = difflib.get_close_matches(query_norm, all_names, n=3, cutoff=0.55)
                for m in matches:
                    official = self.suffix_blind_mapping.get(m, m)
                    uid = self.universal_facility_mapping.get(official)
                    if uid and uid not in selected_facility_uids:
                        selected_facility_uids.append(uid)
                        selected_facility_names.append(official)
                    if len(selected_facility_uids) >= 2:
                        break
            # Deep fuzzy over full facility list per token/word if still <2
            if len(selected_facility_uids) < 2:
                all_facilities = get_all_facilities_flat(self.user)
                fac_names = [f[0] for f in all_facilities]
                fac_name_to_uid = {f[0]: f[1] for f in all_facilities}
                search_terms = tokens + query_norm.split()
                for term in search_terms:
                    term = term.strip()
                    if len(term) < 3:
                        continue
                    m = difflib.get_close_matches(term, [n.lower() for n in fac_names], n=1, cutoff=0.65)
                    if m:
                        # find original name
                        for name in fac_names:
                            if name.lower() == m[0]:
                                uid = fac_name_to_uid.get(name)
                                if uid and uid not in selected_facility_uids:
                                    selected_facility_uids.append(uid)
                                    selected_facility_names.append(name)
                                break
                    if len(selected_facility_uids) >= 2:
                        break
                  
        if "table" in query_lower:
            chart_type = "table"
            intent = "plot" # Treat table requests as plot/data requests
        elif "bar" in query_lower:
            chart_type = "bar"
        elif "area" in query_lower:
            chart_type = "area"
        elif "line" in query_lower:
            chart_type = "line"
            
        # Detect Chart Option Parsing
        if "which charts" in query_lower or "types of charts" in query_lower or ("available" in query_lower and "chart" in query_lower):
            return {
                "intent": "chart_options",
                "kpi": None # Will be filled by context merging if exists
            }
            
        # Detect Comparison Mode
        if any(x in query_lower for x in ["compare", "comparison", " vs ", "versus", "benchmark"]):
            comparison_mode = True
            intent = "plot" # Comparison implies visual
            
            # Detect "all facilities" or "all regions"
            if any(x in query_lower for x in ["all facilities", "all facility", "every facility", "all hospitals"]):
                comparison_entity = "facility"
                # Don't set specific facilities - will be populated later
            elif any(x in query_lower for x in ["all regions", "all region", "every region"]):
                comparison_entity = "region"
                # Don't set specific regions - will be populated later
            # Detect Entity
            elif "region" in query_lower:
                comparison_entity = "region"
            elif "facilit" in query_lower or "hospital" in query_lower or "clinic" in query_lower:
                comparison_entity = "facility"
            else:
                # Default to whatever is selected or infer
                # If specific facilities are mentioned later, it will be facility comparison
                # If "regions" not explicitly said but implied? Let's default to region if no facilities found later?
                # Actually, let's wait to see if multiple facilities are detected.
                pass
            
        # 2. Detect KPI
        selected_kpi = None
        # ... (rest of KPI detection)

        # query_norm is already computed at the start of parse_query
        # Normalization logic moved to top
        
        kpi_map = active_config["kpi_aliases"]
        
        # Stop Word Removal for scanning
        stop_words = ["what", "is", "the", "are", "of", "in", "show", "me", "tell", "about", "rate", "value", "number", "total", "how", "many"]
        query_words = query_norm.split()
        filtered_words = [w for w in query_words if w not in stop_words]
        filtered_query = " ".join(filtered_words)

        def _extract_location_candidates(text: str) -> list[str]:
            cleaned = re.sub(r"\s+", " ", str(text or "")).strip().lower()
            if not cleaned:
                return []

            tokens = [t for t in cleaned.split(" ") if t]
            stop_tokens = {
                "from",
                "to",
                "between",
                "since",
                "last",
                "this",
                "today",
                "yesterday",
                "overall",
                "all",
                "compare",
                "vs",
                "versus",
                "by",
                "per",
                "as",
                "with",
                "and",
                "or",
            }

            candidates: list[str] = []

            # "<...> region" (capture up to 3 tokens before "region")
            for idx, tok in enumerate(tokens):
                if tok in {"region", "regions"} and idx > 0:
                    start = max(0, idx - 3)
                    phrase = " ".join(tokens[start:idx]).strip()
                    if phrase:
                        candidates.append(phrase)

            # "for|in|at|within|inside|around|near <phrase>"
            for idx, tok in enumerate(tokens):
                if tok in {"for", "in", "at", "within", "inside", "around", "near"} and idx + 1 < len(tokens):
                    phrase_tokens: list[str] = []
                    for t in tokens[idx + 1 : idx + 7]:
                        if t in stop_tokens:
                            break
                        phrase_tokens.append(t)
                    while phrase_tokens and phrase_tokens[-1] in {"region", "regions"}:
                        phrase_tokens.pop()
                    if phrase_tokens:
                        candidates.append(" ".join(phrase_tokens))

            # De-dupe while preserving order
            seen = set()
            out: list[str] = []
            for c in candidates:
                c = re.sub(r"\s+", " ", str(c or "")).strip().lower()
                if not c or c in seen:
                    continue
                seen.add(c)
                out.append(c)
            return out

        # Check for direct containment first (using filtered query for better precision)
        # But we must check against 'query_norm' too because some maps have multiple words
        # CRITICAL FIX: If query contains "missing", only match against "missing" KPIs
        # This ensures "missing antepartum" doesn't match "antepartum"
        
        has_missing_keyword = "missing" in query_norm
        missing_birth_weight_flag = has_missing_keyword and "birth weight" in query_norm

        if has_missing_keyword:
            # Filter to only "missing" KPIs
            filtered_kpi_map = {k: v for k, v in kpi_map.items() if "missing" in k}
        else:
            # Exclude "missing" KPIs to avoid false matches
            filtered_kpi_map = {k: v for k, v in kpi_map.items() if "missing" not in k}
        
        # Sort keys by length (longest first) to prioritize specific matches
        sorted_keys = sorted(filtered_kpi_map.keys(), key=len, reverse=True)
        
        for key in sorted_keys:
            if key in query_norm:
                # Protect maternal alias "missing birth" from swallowing "missing birth weight"
                if key.startswith("missing birth") and "weight" in query_norm and key != "missing birth weight (%)":
                    continue
                selected_kpi = filtered_kpi_map[key]
                break
        
        # If not found, try fuzzy matching on filtered words (skip fuzzy when explicitly about missing to avoid cross-missing confusion)
        if not selected_kpi and filtered_query:
            if not has_missing_keyword:
                keys = list(kpi_map.keys())
                matches = difflib.get_close_matches(filtered_query, keys, n=1, cutoff=0.6) # Increased cutoff
                if matches:
                     # Check if this match might be a false positive for a facility list
                     # e.g. "failiteis" -> "episiotomy" 
                     # If "list" or "show" is in query, we are very skeptical of fuzzy clinical matches
                     is_list_query = any(w in query_lower for w in ["list", "show", "all"])
                     if is_list_query and any(w in query_lower for w in ["fac", "hosp", "reg", "fail"]):
                          pass # Don't set selected_kpi, let metadata_query catch it
                     else:
                          selected_kpi = kpi_map[matches[0]]
                
                # Sliding window backup
                if not selected_kpi:
                     if "eciton" in query_lower or "c -s" in query_lower:
                         selected_kpi = "C-Section Rate (%)"
                     
                     
        if not selected_kpi:
            # Fallback: check exact strings in KPI_OPTIONS (ignoring case)
            # CRITICAL: If query has "missing", only match "missing" KPIs to prevent
            # e.g. "Birth Weight Rate" matching "missing birth weight rate"
            for kpi in active_kpi_options:
                kpi_l = kpi.lower()
                if kpi_l in query_lower:
                    if has_missing_keyword and "missing" not in kpi_l:
                        continue  # Skip non-missing KPIs when query mentions "missing"
                    selected_kpi = kpi
                    break

        # MISSING BIRTH OUTCOME LOCK (ISOLATED FIX ONLY)
        if missing_birth_outcome_lock:
            selected_kpi = "Missing Birth Outcome"

        # NEWBORN: hard-lock missing birth weight queries to the missing KPI (avoid fallback to Birth Weight Rate)
        if (self.get_selected_program() or "maternal") == "newborn" and selected_kpi is None:
            if "missing birth weight" in query_norm or "missing birthweight" in query_norm:
                selected_kpi = "Missing Birth Weight (%)"
        # Maternal: if user asked missing birth weight, push cross-program flow
        if (self.get_selected_program() or "maternal") == "maternal" and missing_birth_weight_flag and not selected_kpi:
            selected_kpi = None  # leave unset to trigger cross-program suggestion later
                     
        # Force Bar Chart for Counts
        if selected_kpi in {"Admitted Mothers", "Admitted Newborns"} and chart_type == "line":
            chart_type = "bar"

        # --- REGION-ONLY CONTEXT (Manual GUI Behavior) ---
        # Applies ONLY when the user is asking to group/compare by region AND no facility name is explicitly mentioned.
        region_grouping_requested = any(
            x in query_norm
            for x in [
                "by region",
                "per region",
                "by reg",
                "per reg",
                "compare regions",
                "compare region",
                "all regions",
                "all region",
                "every region",
            ]
        )
        region_only_context = region_grouping_requested and not facility_explicitly_mentioned
        force_region_comparison = region_only_context
        if region_only_context:
            comparison_mode = True
            comparison_entity = "region"
        
        # First, check if user mentioned a facility in the query
        facility_mentioned = any(word in query_lower for word in ["facility", "hospital", "clinic", "center", "health"])
        
        # Refined guard: check for "for ", "at ", "in ", "from " 
        # Refined guard: check for "for ", "at ", "in ", "from ", "i "
        # but EXCLUDE common time-based phrases to avoid "For this year" being caught as a facility request
        time_triggers = ["year", "month", "week", "today", "yesterday", "overall", "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec", "202", "201"]
        is_time_query = any(t in query_lower for t in time_triggers)
        
        # Use regex with word boundaries to avoid matching substrings like "at" in "what"
        # NOTE: "from" is commonly used in date ranges ("from Jan 16 to Jan 31"), not just locations.
        specific_location_regex = r'\b(at)\b'
        specific_facility_requested = bool(re.search(specific_location_regex, query_lower))
        
        # "for" and "in" are common, only treat as facility trigger if NO time word present.
        # NOTE: Do NOT treat the pronoun "I" as a location keyword (it caused many false positives).
        if not is_time_query:
            flexible_location_regex = r'\b(for|in)\b'
            if re.search(flexible_location_regex, query_lower):
                specific_facility_requested = True

        facility_requested_but_unresolved = False
        if (
            (facility_mentioned or specific_facility_requested)
            and not region_only_context
            and not selected_facility_uids
        ):
            # Try to find the facility name
            found_facility = False
            
            # Check each word in query for facility matches
            for i in range(len(query_words)):
                # Check up to 4-word combinations (prefer longest match first)
                for j in range(min(i+4, len(query_words)+1), i, -1):
                    possible_name = " ".join(query_words[i:j]).lower()
                    
                    # Check in suffix_blind_mapping
                    if possible_name in self.suffix_blind_mapping:
                        official_name = self.suffix_blind_mapping[possible_name]
                        uid = self.universal_facility_mapping.get(official_name)
                        if uid:
                            selected_facility_uids.append(uid)
                            selected_facility_names.append(official_name)
                            found_facility = True
                            break
            
            # If no facility found and user specifically asked for a facility, RETURN ERROR
            if not found_facility and specific_facility_requested:
                # Check if it might be a region request (e.g. "for Tigray")
                regions_data = get_facilities_grouped_by_region(self.user)
                region_found = any(r.lower() in query_lower for r in regions_data.keys())

                # Check for generic "all" request (Expanded with Robust Regex)
                # Matches: all fac, all reg, all hospitals, all reioings, every facility, etc.
                all_regex = r'\b(all|every|each)\s+(fac|reg|hosp|unit|reio|reig|territory|faic|faict|faiciliti)'
                is_all_request = bool(re.search(all_regex, query_lower))

                if not region_found and not is_all_request:
                    # Final check: Is this a general value query without a specific location trigger?
                    # If the user just said "What is PPH rate", specific_facility_requested should be False.
                    # If specific_facility_requested is False, we don't return an error.
                    # Defer the error until after we attempt disambiguation / fuzzy matching.
                    facility_requested_but_unresolved = True
        
        # Weak facility fallback: never auto-select multiple facilities from partial words like
        # "addis", "university", etc. If ambiguous, ask the user to choose.
        if (
            facility_requested_but_unresolved
            and not selected_facility_uids
            and not region_only_context
        ):
            weak_candidates = []
            for name, uid in self.universal_facility_mapping.items():
                n_lower = name.lower()
                for word in filtered_words:
                    if len(word) > 3 and (
                        n_lower.startswith(word) or (word in n_lower and len(word) > 5)
                    ):
                        weak_candidates.append((name, uid))
                        break

            # Deduplicate by UID while preserving order
            unique_candidates = []
            seen_uids = set()
            for name, uid in weak_candidates:
                if uid in seen_uids:
                    continue
                seen_uids.add(uid)
                unique_candidates.append((name, uid))

            # If we found nothing via weak word matching, try fuzzy matching on the extracted
            # location phrase (helps with general typos, not just hard-coded COMMON_TYPOS).
            if not unique_candidates:
                location_candidates = _extract_location_candidates(query_norm)

                # If the candidate looks like a region (even with typos), let the region resolver handle it.
                try:
                    regions_data_hint = get_facilities_grouped_by_region(self.user)
                except Exception:
                    regions_data_hint = {}
                region_key_map_hint = {r.lower(): r for r in (regions_data_hint or {}).keys()}
                region_like = False
                if region_key_map_hint:
                    for cand in location_candidates:
                        if cand in region_key_map_hint:
                            region_like = True
                            break
                        if difflib.get_close_matches(
                            cand,
                            list(region_key_map_hint.keys()),
                            n=1,
                            cutoff=0.65,
                        ):
                            region_like = True
                            break

                if not region_like and location_candidates:
                    norm_facility_keys = list(self.suffix_blind_mapping.keys())
                    fuzzy_hits: list[tuple[str, str]] = []
                    for cand in location_candidates:
                        if not cand or cand in self.AMBIGUOUS_PREFIXES:
                            continue

                        if len(cand) <= 4:
                            cutoff = 0.90
                        elif len(cand) <= 6:
                            cutoff = 0.82
                        else:
                            cutoff = 0.75

                        matches = difflib.get_close_matches(
                            cand,
                            norm_facility_keys,
                            n=5,
                            cutoff=cutoff,
                        )
                        for m in matches:
                            official = self.suffix_blind_mapping.get(m)
                            uid = (
                                self.universal_facility_mapping.get(official)
                                if official
                                else None
                            )
                            if uid:
                                fuzzy_hits.append((official, uid))

                    # Deduplicate by UID while preserving order
                    seen_uids = set()
                    for name, uid in fuzzy_hits:
                        if uid in seen_uids:
                            continue
                        seen_uids.add(uid)
                        unique_candidates.append((name, uid))

            if len(unique_candidates) == 1:
                name, uid = unique_candidates[0]
                selected_facility_uids.append(uid)
                selected_facility_names.append(name)
                facility_requested_but_unresolved = False
            elif len(unique_candidates) > 1:
                options_map = {
                    str(i + 1): n for i, (n, _) in enumerate(unique_candidates[:8])
                }
                st.session_state.chatbot_context["ambiguity_options"] = options_map
                st.session_state.chatbot_context["pending_query"] = query
                st.session_state.chatbot_context["pending_parse"] = None
                st.session_state.chatbot_context["pending_kpi"] = selected_kpi
                st.session_state.chatbot_context["pending_llm_used"] = False

                # Try to parse the original intent/KPI via LLM so the follow-up selection
                # can immediately execute the user's request.
                llm_mode_early = self.last_parse_trace.get("parser_mode") or "fallback"
                if llm_enabled and llm_mode_early in {"always", "fallback"}:
                    self.last_parse_trace["attempted"] = True

                    llm_regions_list = None
                    try:
                        llm_regions_list = list(get_facilities_grouped_by_region(self.user).keys())
                    except Exception:
                        llm_regions_list = None

                    llm_result, llm_error = query_llm_detailed(
                        query,
                        facilities_list=list(self.facility_mapping.items()),
                        regions_list=llm_regions_list,
                        kpi_mapping=active_kpi_mapping,
                        program_label=active_config.get("label"),
                    )
                    if llm_error:
                        self.last_parse_trace["failed"] = True
                        self.last_parse_trace["error"] = llm_error
                    elif isinstance(llm_result, dict):
                        st.session_state.chatbot_context["pending_llm_used"] = True
                        st.session_state.chatbot_context["pending_parse"] = llm_result

                        resolved = None
                        raw_candidate = llm_result.get("kpi")
                        if isinstance(raw_candidate, str) and raw_candidate.strip():
                            raw_candidate = raw_candidate.strip()
                            if raw_candidate in active_kpi_mapping:
                                resolved = raw_candidate
                            else:
                                ci_map = {k.lower(): k for k in active_kpi_mapping.keys()}
                                exact_ci = ci_map.get(raw_candidate.lower())
                                if exact_ci:
                                    resolved = exact_ci
                                else:
                                    matches = difflib.get_close_matches(
                                        raw_candidate,
                                        list(active_kpi_mapping.keys()),
                                        n=1,
                                        cutoff=0.88,
                                    )
                                    if matches:
                                        resolved = matches[0]
                        if resolved:
                            st.session_state.chatbot_context["pending_kpi"] = resolved

                options_str = "\n".join([f"{i}. {n}" for i, n in options_map.items()])
                return {
                    "intent": "chat",
                    "response": f"Which **facility** do you mean?\n\n{options_str}",
                    "pending_kpi": selected_kpi,
                }
        
        # If no facility found, check REGIONS
        if not selected_facility_uids:
            regions_data = get_facilities_grouped_by_region(self.user)
            found_regions = []
            
            # Check Match for Regions (Multiple allowed for comparison)
            found_regions = []
            
            # Fallback for Region/Facility extraction if not from LLM
            # Check for region names in query
            found_regions = []
            
            # Direct match
            for region_name in regions_data.keys():
                if region_name.lower() in query_norm:
                    found_regions.append(region_name)

            # Candidate-based fuzzy match (handles typos like "tigry" -> "Tigray") and supports multiple regions.
            # Run even when we already have 1 direct hit, so queries like "Amhara and Tigra" can still resolve
            # the second region via fuzzy matching.
            if not region_only_context:
                region_key_map = {r.lower(): r for r in regions_data.keys()}
                region_compact_map = {
                    re.sub(r"\s+", "", str(r).strip().lower()): r
                    for r in regions_data.keys()
                    if str(r).strip()
                }

                def _extract_region_candidates(text: str) -> list[str]:
                    tokens = [t for t in re.sub(r"\s+", " ", str(text or "")).strip().split(" ") if t]
                    if not tokens:
                        return []

                    stop_tokens = {
                        "from",
                        "to",
                        "between",
                        "since",
                        "last",
                        "this",
                        "today",
                        "yesterday",
                        "overall",
                        "all",
                        "compare",
                        "vs",
                        "versus",
                        "by",
                        "per",
                        "as",
                        "with",
                        "and",
                        "or",
                    }

                    candidates: list[str] = []

                    # "<...> region" (capture up to 3 tokens before "region")
                    for idx, tok in enumerate(tokens):
                        if tok in {"region", "regions"} and idx > 0:
                            start = max(0, idx - 3)
                            phrase = " ".join(tokens[start:idx]).strip()
                            if phrase:
                                candidates.append(phrase)

                    # "for|in|at|within|inside|around|near <phrase>"
                    for idx, tok in enumerate(tokens):
                        if tok in {"for", "in", "at", "within", "inside", "around", "near"} and idx + 1 < len(tokens):
                            phrase_tokens: list[str] = []
                            for t in tokens[idx + 1 : idx + 7]:
                                if t in stop_tokens:
                                    break
                                phrase_tokens.append(t)
                            while phrase_tokens and phrase_tokens[-1] in {"region", "regions"}:
                                phrase_tokens.pop()
                            if phrase_tokens:
                                candidates.append(" ".join(phrase_tokens))

                    # Location lists like: "for amhara and tigra" / "in oromia vs amhara"
                    separators = {"and", "or", "vs", "versus"}
                    hard_stops = stop_tokens - separators
                    location_markers = {"for", "in", "at", "within", "inside", "around", "near"}
                    marker_indexes = [i for i, t in enumerate(tokens) if t in location_markers]
                    if marker_indexes:
                        start_idx = marker_indexes[-1] + 1
                        tail: list[str] = []
                        for t in tokens[start_idx:]:
                            if t in hard_stops:
                                break
                            tail.append(t)

                        seg: list[str] = []
                        for t in tail:
                            if t in separators:
                                if seg:
                                    candidates.append(" ".join(seg))
                                    seg = []
                                continue
                            seg.append(t)
                        if seg:
                            candidates.append(" ".join(seg))

                    # De-dupe while preserving order
                    seen = set()
                    out: list[str] = []
                    for c in candidates:
                        c = c.strip().lower()
                        if not c or c in seen:
                            continue
                        seen.add(c)
                        out.append(c)
                    return out

                def _resolve_region_candidate(cand: str) -> str | None:
                    cand = re.sub(r"[^a-z0-9\\s]", " ", str(cand or "").lower())
                    cand = re.sub(r"\s+", " ", cand).strip()
                    if not cand:
                        return None

                    direct = region_key_map.get(cand)
                    if direct:
                        return direct

                    cand_compact = cand.replace(" ", "")
                    if cand_compact and region_compact_map:
                        direct_compact = region_compact_map.get(cand_compact)
                        if direct_compact:
                            return direct_compact

                    r_matches = difflib.get_close_matches(
                        cand,
                        list(region_key_map.keys()),
                        n=1,
                        cutoff=0.65,
                    )
                    if r_matches:
                        return region_key_map[r_matches[0]]

                    if cand_compact and region_compact_map:
                        cm = difflib.get_close_matches(
                            cand_compact,
                            list(region_compact_map.keys()),
                            n=1,
                            cutoff=0.65,
                        )
                        if cm:
                            return region_compact_map[cm[0]]

                    return None

                # Add fuzzy matches without wiping direct hits (supports multiple regions).
                for cand in _extract_region_candidates(query_norm):
                    resolved = _resolve_region_candidate(cand)
                    if resolved and resolved not in found_regions:
                        found_regions.append(resolved)
                    if len(found_regions) >= 4:
                        break

            # REGION-ONLY CONTEXT: If no regions are explicitly mentioned, select ALL available regions.
            if region_only_context and not found_regions:
                found_regions = list(regions_data.keys())
                comparison_mode = True
                comparison_entity = "region"
            
            # Fuzzy match if none found (only try to find one primary if none explicit)
            if not found_regions and not region_only_context:
                  r_matches = difflib.get_close_matches(query_lower, [r.lower() for r in regions_data.keys()], n=1, cutoff=0.6)
                  if r_matches:
                       for r in regions_data.keys():
                           if r.lower() == r_matches[0]:
                               found_regions.append(r)
                               break
            
            # --- COMPARISON MODE FALLBACK DETECTION ---
            if "compare" in query_lower or " vs " in query_lower or "versus" in query_lower:
                comparison_mode = True
                if found_regions:
                     comparison_entity = "region"
                elif selected_facility_uids:
                     # If we found facility names/uids via earlier regex/fuzzy match
                     comparison_entity = "facility"
            
            if found_regions:
                # If Comparison Mode (explicit or implicit), store these regions
                if comparison_mode:
                     comparison_entity = "region"
                     
                # Collect UIDs from ALL found regions (ALWAYS collect for data fetching)
                all_uids = []
                all_names = []
                comparison_targets = []
                for r_name in found_regions:
                    f_list = regions_data[r_name]
                    all_uids.extend([f[1] for f in f_list])
                    all_names.append(f"{r_name} (Region)")
                    comparison_targets.append(r_name)
                
                selected_facility_uids = all_uids
                selected_facility_names = all_names
                if comparison_mode:
                     selected_comparison_targets = comparison_targets

                # Collect UIDs from ALL found regions
                all_uids = []
                all_names = []
                for r_name in found_regions:
                    f_list = regions_data[r_name]
                    all_uids.extend([f[1] for f in f_list])
                    all_names.append(f"{r_name} (Region)")
                
                selected_facility_uids = all_uids
                selected_facility_names = all_names

            # If still no facility/region found but user says "all facilities" or doesn't specify, 
            # for Facility user it's always their facility.
            if not selected_facility_uids and not facility_requested_but_unresolved:
                if self.user.get("role") == "facility":
                    # Default to user's facility
                    selected_facility_uids = list(self.facility_mapping.values())
                    selected_facility_names = list(self.facility_mapping.keys())
                # For regional/national, if no specific facility, we might mean "overall" or "all"

            # If the user explicitly tried to specify a facility but we couldn't resolve it,
            # defer the error until after we give the LLM a chance to interpret the location.
        
        # 4. Detect Time Period
        start_date = None
        end_date = None
        today = datetime.now()
        reset_date = False # Initialize to avoid UnboundLocalError
        
        # Explicitly check for clearing dates
        if any(x in query_lower for x in ["overall", "all time", "since beginning", "from start", "total", "entire period"]):
             # Only reset if we are talking about time, or if broadly applied
             # "total" is tricky because "total admitted mothers" could mean count for THIS period.
             # So we look for time-bound phrases specifically or "overall"
             if "overall" in query_lower or "all time" in query_lower or "start" in query_lower:
                 reset_date = True
                 start_date = None
                 end_date = None

        # --- Custom Date Range Detection (Must run BEFORE quick ranges) ---
        # If the user doesn't specify a year, assume the current year.
        DEFAULT_YEAR_IF_MISSING = today.year

        def _parse_day_month_token(token: str):
            t = token.strip().lower().replace(",", " ")
            # Handle "jan1" / "1jan" style tokens
            t = re.sub(r"([a-z])(\d)", r"\1 \2", t)
            t = re.sub(r"(\d)([a-z])", r"\1 \2", t)
            t = re.sub(r"\s+", " ", t).strip()
            month_map = {
                "jan": 1,
                "feb": 2,
                "mar": 3,
                "apr": 4,
                "may": 5,
                "jun": 6,
                "jul": 7,
                "aug": 8,
                "sep": 9,
                "oct": 10,
                "nov": 11,
                "dec": 12,
            }

            # "12 dec 2025"
            m = re.match(
                r"^(?P<day>\d{1,2})\s+(?P<month>[a-z]{3,})(?:\s+(?P<year>\d{4}))?$",
                t,
            )
            if not m:
                # "dec 12 2025"
                m = re.match(
                    r"^(?P<month>[a-z]{3,})\s+(?P<day>\d{1,2})(?:\s+(?P<year>\d{4}))?$",
                    t,
                )
            if not m:
                return None

            month_key = m.group("month")[:3]
            if month_key not in month_map:
                return None

            day = int(m.group("day"))
            month = month_map[month_key]
            year = int(m.group("year")) if m.group("year") else None
            return day, month, year

        if not start_date:
            try:
                date_token = r"(?:\d{1,2}\s*[a-z]{3,}|[a-z]{3,}\s*\d{1,2})(?:\s+\d{4})?"
                range_pattern = re.compile(
                    rf"(?:(?:from|form)\s+)?(?P<d1>{date_token})\s*(?:to|[\-\u2013\u2014])\s*(?P<d2>{date_token})",
                    re.IGNORECASE,
                )
                m = range_pattern.search(query_lower)
                if m:
                    t1 = _parse_day_month_token(m.group("d1"))
                    t2 = _parse_day_month_token(m.group("d2"))
                    if t1 and t2:
                        d1, mo1, y1 = t1
                        d2, mo2, y2 = t2
                        
                        def _safe_date(y, m, d):
                            last = calendar.monthrange(y, m)[1]
                            return datetime(y, m, min(d, last))

                        if y1 and y2:
                            pass
                        elif y1 and not y2:
                            y2 = y1 + 1 if mo1 > mo2 else y1
                        elif y2 and not y1:
                            y1 = y2 - 1 if mo1 > mo2 else y2
                        else:
                            # No years specified
                            if mo1 > mo2:
                                # Example: "Dec 12 to Jan 14" -> Dec 2025 to Jan 2026
                                y1 = DEFAULT_YEAR_IF_MISSING - 1
                                y2 = DEFAULT_YEAR_IF_MISSING
                            else:
                                y1 = DEFAULT_YEAR_IF_MISSING
                                y2 = DEFAULT_YEAR_IF_MISSING

                        start_date = _safe_date(y1, mo1, d1).strftime("%Y-%m-%d")
                        end_date = _safe_date(y2, mo2, d2).strftime("%Y-%m-%d")
            except Exception as e:
                logging.warning(f"Custom date range parsing failed: {e}")

        # Quick ranges (ONLY if no custom range detected)
        if not start_date:
            if "this year" in query_lower:
                start_date = f"{today.year}-01-01"
                end_date = today.strftime("%Y-%m-%d")
            elif "last year" in query_lower:
                # Use Calendar Year logic
                last_year = today.year - 1
                start_date = f"{last_year}-01-01"
                end_date = f"{last_year}-12-31"
            elif "this month" in query_lower:
                start_date = today.replace(day=1).strftime("%Y-%m-%d")
                end_date = today.strftime("%Y-%m-%d")
            elif "last month" in query_lower:
                formatted_today = today.replace(day=1)
                last_month_end = formatted_today - timedelta(days=1)
                last_month_start = last_month_end.replace(day=1)
                start_date = last_month_start.strftime("%Y-%m-%d")
                end_date = last_month_end.strftime("%Y-%m-%d")
            elif "this week" in query_lower:
                # Monday of current week
                weekday = today.weekday()
                start_date_dt = today - timedelta(days=weekday)
                start_date = start_date_dt.strftime("%Y-%m-%d")
                end_date = today.strftime("%Y-%m-%d")
            elif "last week" in query_lower:
                # Previous week Monday-Sunday
                weekday = today.weekday()
                end_date_dt = today - timedelta(days=weekday + 1)
                start_date_dt = end_date_dt - timedelta(days=6)
                start_date = start_date_dt.strftime("%Y-%m-%d")
                end_date = end_date_dt.strftime("%Y-%m-%d")
        
        # --- NEW: Robust Multi-Format Date Extraction ---
        if not start_date:
            try:
                # 1. Extract ALL date fragments (Day Month Year, Month Day Year, Month Year)
                # First, extract any "Day Month Year" or "Month Day Year"
                full_date_pattern = re.compile(r'(\d{1,2}\s+[a-z]{3,}\s+\d{4})|([a-z]{3,}\s+\d{1,2},?\s+\d{4})|([a-z]{3,}\s+\d{4})', re.IGNORECASE)
                date_strings = [m.group(0) for m in full_date_pattern.finditer(query)]
                
                def parse_any_date(ds):
                    # Try various formats
                    formats = ["%d %b %Y", "%b %d %Y", "%b %d, %Y", "%b %Y"]
                    for fmt in formats:
                        try:
                            # Pre-clean the string to 3-char month
                            parts = ds.split()
                            cleaned_parts = []
                            for p in parts:
                                if p.lower()[:3] in ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]:
                                    cleaned_parts.append(p[:3])
                                else:
                                    cleaned_parts.append(p.rstrip(','))
                            
                            cleaned_ds = " ".join(cleaned_parts)
                            dt_obj = datetime.strptime(cleaned_ds, fmt)
                            if fmt == "%b %Y":
                                # If only month/year, return (start_of_month, end_of_month)
                                return dt_obj, (dt_obj.replace(month=dt_obj.month % 12 + 1, day=1) - timedelta(days=1)) if dt_obj.month < 12 else dt_obj.replace(day=31)
                            return dt_obj, dt_obj
                        except:
                            continue
                    
                    # Fallback: manual parse with day clamping (handles Feb 29 on non-leap years)
                    m = re.search(r'([a-z]{3,})\\s+(\\d{1,2})(?:,)?\\s+(\\d{4})', ds, re.IGNORECASE)
                    if m:
                        mon_txt, day_txt, year_txt = m.groups()
                        mon_num = datetime.strptime(mon_txt[:3], "%b").month
                        year_num = int(year_txt)
                        day_num = int(day_txt)
                        last = calendar.monthrange(year_num, mon_num)[1]
                        safe_dt = datetime(year_num, mon_num, min(day_num, last))
                        return safe_dt, safe_dt
                    return None, None

                if len(date_strings) >= 2:
                    d1_start, _ = parse_any_date(date_strings[0])
                    _, d2_end = parse_any_date(date_strings[1])
                    if d1_start and d2_end:
                        start_date = d1_start.strftime("%Y-%m-%d")
                        end_date = d2_end.strftime("%Y-%m-%d")
                elif len(date_strings) == 1:
                    d1_start, d1_end = parse_any_date(date_strings[0])
                    if d1_start:
                        start_date = d1_start.strftime("%Y-%m-%d")
                        end_date = d1_end.strftime("%Y-%m-%d")
            except Exception as e:
                logging.warning(f"Hyper-flexible date parsing failed: {e}")

        # Fallback for "YYYY/MM/DD - YYYY/MM/DD"
        if not start_date:
             try:
                 # Pattern: YYYY/MM/DD or YYYY-MM-DD
                 iso_matches = re.findall(r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})', query)
                 if len(iso_matches) >= 2:
                      y1, m1, d1 = iso_matches[0]
                      y2, m2, d2 = iso_matches[1]
                      start_date = f"{y1}-{int(m1):02d}-{int(d1):02d}"
                      end_date = f"{y2}-{int(m2):02d}-{int(d2):02d}"
             except Exception as e:
                 logging.warning(f"ISO Date regex parse failed: {e}")
                 
        # 4b. NEW: Handle "Today" logic explicitly with typos (e.g. "ot today", "to today")
        # Pattern: "Jan 1 to today" or "from Jan 1 - today"
        if not start_date and "today" in query_lower:
             try:
                 # Clean up typical typos for "to" -> "ot", "too"
                 cleaned_query_date = (
                     query_lower.replace(" ot ", " to ")
                     .replace(" too ", " to ")
                     .replace(" from ", "")
                     .replace(" form ", " ")
                 )
                 # Clean up month typos for this specific check
                 for typo, corr in [("jna", "jan"), ("fbe", "feb"), ("mrc", "mar"), ("arp", "apr"), ("my", "may"), ("jnu", "jun"), ("july", "jul"), ("augst", "aug"), ("sept", "sep"), ("octber", "oct"), ("novmber", "nov"), ("dec", "dec")]:
                      if typo in cleaned_query_date:
                           cleaned_query_date = cleaned_query_date.replace(typo, corr)
                 
                 today_str = today.strftime("%Y-%m-%d")
                 
                 # Pattern: (Month DD) ... today
                 pattern_today = re.compile(r"([a-z]{3,})\s*(\d{1,2}).*?today", re.IGNORECASE)
                 match_today = pattern_today.search(cleaned_query_date)
                  
                 if match_today:
                      m, d = match_today.groups()
                      # Assume current year first, fallback logic for rollovers could be added
                      y = today.year
                      start_date = datetime.strptime(f"{m[:3]} {d} {y}", "%b %d %Y").strftime("%Y-%m-%d")
                      end_date = today_str
             except Exception as e:
                  logging.warning(f"Date regex 'today' failed: {e}")
        
        # 5. Detect Aggregation Period
        period_label = None
        if "daily" in query_lower:
            period_label = "Daily"
        elif "weekly" in query_lower:
            period_label = "Weekly"
        elif "monthly" in query_lower:
            period_label = "Monthly"
        elif "quarterly" in query_lower:
            period_label = "Quarterly"
        elif "yearly" in query_lower:
            period_label = "Yearly"
            
        if period_label and intent != "list_kpis" and "define" not in query_lower and "what is" not in query_lower:
            intent = "plot" # Aggregation usually implies visual trend
              
        # 6. Detect Analysis Type (Max/Min)
        analysis_type = None
        if any(x in query_lower for x in ["highest", "max", "peak", "most"]):
            analysis_type = "max"
            intent = "text" # Force text response for analysis
            reset_date = True # Analysis usually implies searching history
        elif any(x in query_lower for x in ["lowest", "min", "least", "minimum"]):
            analysis_type = "min"
            intent = "text" # Force text response for analysis
            reset_date = True # Analysis usually implies searching history

        # --- REFINE INTENT ---
        # If user asks "what is" or "value", assume text.
        # If user asks "what is" or "plot", "show", "trend", assume plot.
        # If ambiguous, prefer text if specific date/value requested, plot if trend.
        if "what is" in query_lower or "value of" in query_lower:
             intent = "text"
        
        # Use already detected selected_kpi for intent refinement
        kpi_detected = selected_kpi is not None

        # Definition Detection
        definition_keywords = ["define", "meaning", "definition", "how is", "computed", "formula", "calculation"]
        is_definition_request = any(d in query_lower for d in definition_keywords)

        if "what is" in query_lower or "how many" in query_lower or is_definition_request:
             # If user asks for "value", "rate", "count", "number", they want DATA, not definition.
             # Strict check: If KPI name is present, assume DATA intent not definition
             # Also exclude "total"
             data_keywords = ["value", "rate", "count", "number", "score", "percentage", "trend", "plot", "total"]
             
             # Check if this is a data query vs a definition query
             # A query is "data" if it has data keywords or a KPI name AND doesn't have definition keywords
             if (any(x in query_lower for x in data_keywords) or kpi_detected) and not is_definition_request:
                 # It's likely a data query like "What is the total admitted mothers..."
                 # FORCE override even if LLM said metadata_query (common error for "how many")
                 if intent == "metadata_query" and kpi_detected:
                      intent = "text"
                 elif intent == "metadata_query" and not kpi_detected:
                      pass # Valid metadata query like "how many facilities"
                 else: 
                      intent = "text"
             
             elif is_definition_request:
                 intent = "definition"

        # Default to plotting when the user mentions a KPI without explicitly
        # asking for a single value/definition.
        if (
            intent == "text"
            and kpi_detected
            and not analysis_type
            and not is_definition_request
            and not any(
                token in query_lower
                for token in [
                    "what is",
                    "how many",
                    "value",
                    "number",
                    "count",
                ]
            )
        ):
            intent = "plot"

        # Robust List Detection
        if "indicator" in query_lower or "kpi" in query_lower or "all indicators" in query_lower:
            if any(x in query_lower for x in ["what", "list", "show", "available", "options", "help", "how many", "total"]):
                intent = "list_kpis"
        
        # Explicit handling for direct program references in current context
        if active_program == "maternal" and query_lower in ["maternal", "maternal indicators", "maternal health", "mothers", "matenal"]:
            intent = "list_kpis"
        if active_program == "newborn" and query_lower in ["newborn", "newborn indicators", "newborn care", "neonatal"]:
            intent = "list_kpis"
        if "options" in query_lower or "capabilities" in query_lower:
            intent = "list_kpis"

        # Robust Scope Error Detection
        if any(x in query_lower for x in ["color", "style", "background", "theme", "dark mode", "appearance"]):
            intent = "scope_error"
             
        # Hallucination Guard: Check for specific out-of-scope terms that fuzzily match KPIs
        # e.g. "temperature" -> "Antepartum" or "Outborn" (incorrectly)
        hallucination_terms = ["temperature", "temprature", "weather", "climate", "hot", "cold", "fever"]
        if any(term in query_lower for term in hallucination_terms):
             # Only trigger if NO valid KPI was explicitly found to override this
             # But "temperature" might match "hypothermia" (valid). 
             # We need to be careful. "Hypothermia" is in KPI_MAPPING (or Newborn).
             # If exact KPI name isn't there, blocking "temperature" is safer.
             # Let's check if a STRONG match exists first.
             is_valid_context = False
             if "hypothermia" in query_lower or "kmc" in query_lower: 
                 is_valid_context = True # Allow if specific medical term used
                 
             if not is_valid_context:
                 intent = "scope_error_hallucination"
             
        # Robust Chat/Greeting Detection
        chat_patterns = ["hi", "hello", "hey", "greetings", "who are you", "thanks", "thank you", "help", "good morning", "good afternoon", "how can you help", "what can you help"]
        # Check for exact matches or start of string to avoid false positives
        cleaned_words = re.sub(r'[^a-z\s]', '', query_lower).split()
        if any(word in chat_patterns for word in cleaned_words):
             intent = "chat"
             
        # Off-Topic / System Admin Detection
        if any(x in query_lower for x in ["password", "login", "admin", "access", "credential", "sign in", "log in"]):
             intent = "chat"
             # We will handle this intent in generate_response by checking the query content again or via LLM response if present
             # If LLM didn't catch it, we force it here. But we need a way to pass the response.
             # Since 'response' key is None here, generate_response will fallback to welcome message.
             # We should probably set a custom response or handling flag.
             # Or rely on generate_response generic handler to be smarter.
             # Let's set a specific response here? No, parse_query returns structure.
             # Let's rely on generate_response.
             pass
             # We can optionally set a specific response here if we refactor return, 
             # but generate_response handles generic chat well.
            
             
        # Explicit list KPIs/indicators request (keep deterministic even if LLM is enabled)
        if explicit_list_intent == "list_kpis":
            intent = "list_kpis"

        # Metadata / Counts Detection (Regex)
        # ONLY if no KPI selected, otherwise assume data query.
        # Facility listing must take precedence over region listing when both words exist
        # (e.g. "list facilities in Tigray region").
        if not selected_kpi and (
            explicit_list_intent == "metadata_query"
            or ("how many" in query_lower or "list" in query_lower or "show me" in query_lower or "show all" in query_lower)
        ):
            if explicit_list_intent == "metadata_query":
                intent = "metadata_query"
                entity_type = explicit_list_entity_type
                count_requested = explicit_list_count_requested
                if explicit_list_region_filter:
                    region_filter = explicit_list_region_filter
            elif any(x in query_lower for x in ["facilit", "hospital", " fac ", " failit"]):
                intent = "metadata_query"
                entity_type = "facility"
                count_requested = "how many" in query_lower

                # Try to detect a region focus for facility lists (e.g. "facilities in Tigray")
                regions_data = get_facilities_grouped_by_region(self.user)
                for r_name in regions_data.keys():
                    if r_name.lower() in query_norm:
                        region_filter = r_name
                        break
            elif "region" in query_lower or "reigon" in query_lower or " reg " in query_lower:
                intent = "metadata_query"
                entity_type = "region"
                count_requested = "how many" in query_lower

        # --- CONTEXT MERGING (FOLLOW-UP MEMORY) ---
        # Reuse the previous query's scope/date/plot settings ONLY for follow-up questions.
        raw_context = st.session_state.get("chatbot_context", {}) or {}
        if not isinstance(raw_context, dict):
            raw_context = {}

        current_program = self.get_selected_program() or "maternal"
        context_program = raw_context.get("source")
        cross_program_context = bool(context_program and context_program != current_program)

        # If the previous message came from a different program, keep only program-agnostic context
        # (location/date/plot prefs). Never carry KPI names across programs.
        if cross_program_context:
            context = {
                "facility_uids": raw_context.get("facility_uids"),
                "facility_names": raw_context.get("facility_names"),
                "date_range": raw_context.get("date_range"),
                "entity_type": raw_context.get("entity_type"),
                "region_filter": raw_context.get("region_filter"),
                "comparison_mode": raw_context.get("comparison_mode"),
                "comparison_entity": raw_context.get("comparison_entity"),
                "comparison_targets": raw_context.get("comparison_targets"),
                "intent": raw_context.get("intent"),
                "chart_type": raw_context.get("chart_type"),
                "orientation": raw_context.get("orientation"),
                "period_label": raw_context.get("period_label"),
                "source": raw_context.get("source"),
            }
        else:
            context = raw_context

        ctx_has_state = bool(
            isinstance(context, dict)
            and (
                context.get("facility_uids")
                or context.get("facility_names")
                or context.get("date_range")
                or context.get("kpi")
                or context.get("region_filter")
            )
        )
        prev_ctx_intent = context.get("intent") if isinstance(context, dict) else None

        follow_up = False
        if ctx_has_state:
            q_follow = query_norm.strip()
            follow_prefixes = (
                "what about",
                "how about",
                "and",
                "also",
                "same",
                "same for",
                "instead",
                "now",
                "then",
                "but",
                "please",
                "i need",
                "i want",
            )
            if any(q_follow == p or q_follow.startswith(p + " ") for p in follow_prefixes):
                follow_up = True
            elif re.fullmatch(r"(it|that|this|those|them|same|previous|earlier)(\\s+.*)?", q_follow):
                follow_up = True
            else:
                tokens = q_follow.split()
                if tokens:
                    # "KPI-only" follow-up like: "admitted mothers"
                    has_new_scope = bool(
                        re.search(
                            r"\b(for|in|at|from|to|between|since|region|facility|hospital)\b",
                            q_follow,
                        )
                    )
                    has_new_time = bool(
                        re.search(
                            r"\b(19|20)\d{2}\b",
                            q_follow,
                        )
                        or any(m in q_follow for m in ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
                    )
                    if len(tokens) <= 3 and not has_new_scope and not has_new_time:
                        follow_up = True
                    # Longer follow-ups like: "but i need the comparison plot of both"
                    elif not has_new_scope and not has_new_time:
                        if re.search(r"\b(both|same|previous|earlier|again|comparison|compare|plot|chart|graph|them|it|that|this)\b", q_follow):
                            follow_up = True

        # Merge KPI (same program only; cross-program context does not include a KPI)
        if (
            follow_up
            and intent not in ["list_kpis", "metadata_query", "scope_error", "chart_options", "clear"]
            and not selected_kpi
            and isinstance(context, dict)
            and context.get("kpi")
        ):
            selected_kpi = context.get("kpi")

        # Merge Facilities / Region scope (follow-ups only)
        if (
            follow_up
            and not selected_facility_uids
            and not region_only_context
            and not facility_requested_but_unresolved
            and intent not in {"metadata_query", "list_kpis"}
            and isinstance(context, dict)
            and context.get("facility_uids")
        ):
            selected_facility_uids = context.get("facility_uids")
            selected_facility_names = context.get("facility_names")

            prev_region_filter = context.get("region_filter")
            if isinstance(prev_region_filter, str) and prev_region_filter.strip() and not region_filter:
                region_filter = prev_region_filter.strip()

        # Merge comparison context (follow-ups): preserve the previous compare targets/entities when the
        # user asks a generic follow-up like "show the comparison plot of both".
        if follow_up and isinstance(context, dict):
            prev_comp_mode = context.get("comparison_mode")
            prev_comp_entity = context.get("comparison_entity")
            prev_comp_targets = context.get("comparison_targets")

            if prev_comp_mode and not comparison_mode:
                comparison_mode = True
            if prev_comp_entity and not comparison_entity:
                comparison_entity = prev_comp_entity
            if (
                prev_comp_entity == "region"
                and isinstance(prev_comp_targets, list)
                and prev_comp_targets
                and not found_regions
            ):
                found_regions = [
                    r.strip()
                    for r in prev_comp_targets
                    if isinstance(r, str) and r.strip()
                ][:5]

        # Merge Date Range (follow-ups only)
        if follow_up and not start_date and not reset_date and isinstance(context, dict):
            prev_dr = context.get("date_range")
            if isinstance(prev_dr, dict):
                prev_start = prev_dr.get("start_date")
                prev_end = prev_dr.get("end_date")
                if isinstance(prev_start, str) and isinstance(prev_end, str):
                    start_date = prev_start
                    end_date = prev_end

        # Follow-up intent inheritance: if the user doesn't ask for a single value, keep plotting.
        if follow_up and intent == "text" and isinstance(context, dict):
            prev_intent = context.get("intent")
            explicit_value_question = bool(
                "what is" in query_lower
                or re.search(r"\b(how many|count|number|total|value)\b", query_lower)
            )
            if prev_intent == "plot" and not explicit_value_question:
                intent = "plot"

            prev_chart_type = context.get("chart_type")
            if prev_chart_type in {"line", "bar", "area", "table"} and chart_type == "line":
                chart_type = prev_chart_type

            prev_period = context.get("period_label")
            if not period_label and prev_period in {"Daily", "Weekly", "Monthly", "Quarterly", "Yearly"}:
                period_label = prev_period
             
        # Merge Entity Type (For "Name them" queries)
        if intent == "text" and not selected_kpi and not entity_type:
            # Check if user says "list them" or "name them"
            if any(x in query_lower for x in ["name them", "list them", "what are they", "show them"]):
                if context.get("entity_type"):
                    intent = "metadata_query"
                    entity_type = context.get("entity_type")
                    count_requested = False
        
        
        # Explicitly check for clearing facilities/regions ("all regions", "overall")
        # BUT respect found_regions if they exist (e.g. "Tigray all facilities")
        if ("all region" in query_lower or "all facilities" in query_lower) and not found_regions:
            selected_facility_uids = []
            selected_facility_names = []
        
        # --- ROBUST COMPARISON DETECTION ---
        # Handle explicitly requested comparisons, including common typos
        comparison_keywords = ["compare", "comparison", "comparisoin", "compariosn", " vs ", "versus", "difference", "benchmark"]
        if any(x in query_lower for x in comparison_keywords):
             comparison_mode = True
             # Force plot intent if we are comparing, unless it's a list or definition or scope error
             if intent not in ["list_kpis", "definition", "scope_error"]:
                  intent = "plot"

        # Detect "By Facility" intent (Drill-down / Disaggregation)
        # Triggers if:
        # 1. "by facility" explicit phrase
        # 2. "facilities" mentioned along with a region in comparison mode (e.g. "Compare Tigray facilities")
                # NEW: Implicit Comparison Triggers (Handle "by facility" at top level)
        if not comparison_mode:
            if any(x in query_lower for x in ["by facility", "per facility", "by local", "per local", "by fac", "per fac"]):
                comparison_mode = True
                comparison_entity = "facility"
                if not selected_facility_uids and not found_regions:
                     # Check context for previous region
                     prev_context = st.session_state.get("chatbot_context", {})
                     if prev_context.get("facility_names") and "(Region)" in prev_context.get("facility_names")[0]:
                          found_regions = [n.replace(" (Region)", "") for n in prev_context.get("facility_names")]
            elif any(x in query_lower for x in ["by region", "per region", "by reg", "per reg"]):
                comparison_mode = True
                comparison_entity = "region"
        drill_down_phrases = [
            "by facility",
            "per facility",
            "under",
            "breakdown by facility",
            "compare facilities",
            "by region",
            "per region",
            "breakdown by region",
            "by fac",
            "per fac",
            "by reg",
            "per reg",
        ]
        explicit_drill_down = any(p in query_lower for p in drill_down_phrases)

        # Treat "compare <region> facilities" as a drill-down request even if it doesn't match the exact
        # phrase "compare facilities". This enables prompts like "Compare Tigray facilities".
        if (
            not explicit_drill_down
            and comparison_mode
            and found_regions
            and len(found_regions) == 1
            and ("facility" in query_lower or "facilities" in query_lower or "hospitals" in query_lower)
            and not facility_explicitly_mentioned
        ):
            explicit_drill_down = True

        # IMPORTANT: do NOT treat every comparison as drill-down; otherwise a typo that leaves only one
        # region detected will expand to facilities and break region-vs-region comparisons.
        is_drill_down = explicit_drill_down
        
        if is_drill_down:
             # Force Plot intent unless it's a definition
             if intent != "definition":
                 intent = "plot"
             
             # NEW: If multiple regions are found, we likely want to compare REGIONS, not facilities within them
             # UNLESS user explicitly said "by facility"
             if len(found_regions) > 1 and not any(x in query_lower for x in ["by facility", "per facility", "by fac", "per fac"]):
                  comparison_mode = True
                  comparison_entity = "region"
                  # selected_comparison_targets = found_regions # This is handled by the return statement
             # If we have found regions (e.g. "Tigray by facility"), we want to compare facilities within that region
             elif found_regions and not force_region_comparison:
                 comparison_mode = True
                 comparison_entity = "facility"
                 
                 # EXPAND the region into individual facilities for the comparison logic
                 new_names = []
                 new_uids = []
                 seen_uids = set()
                 for r_name in found_regions:
                     facs = regions_data.get(r_name, [])
                     for f_name, f_uid in facs:
                         if f_uid not in seen_uids:
                             new_names.append(f_name)
                             new_uids.append(f_uid)
                             seen_uids.add(f_uid)
                 
                 # Limit expansion to prevent UI overload (e.g. max 50?)
                 if len(new_names) > 0:
                     selected_facility_names = new_names
                     selected_facility_uids = new_uids
                     
                 # Important: Clear found_regions so it doesn't default to region aggregation logic below
                 found_regions = []
             
             # Handle "By Region" - Expand ALL regions if none specified
             elif "region" in query_lower and ("by" in query_lower or "per" in query_lower or "all" in query_lower) and not found_regions:
                 comparison_mode = True
                 comparison_entity = "region"
                 # Auto-populate all regions
                 regions_data = get_facilities_grouped_by_region(self.user)
                 found_regions = list(regions_data.keys())
              
             # Handle "By Facility" - Expand ALL facilities if none specified
             elif ("facilit" in query_lower or "hospital" in query_lower or "fac " in query_lower) and ("by" in query_lower or "per" in query_lower) and not selected_facility_uids:
                 comparison_mode = True
                 comparison_entity = "facility"
                 # Auto-populate all facilities
                 all_facs = get_all_facilities_flat(self.user)
                 selected_facility_names = [f[0] for f in all_facs]
                 selected_facility_uids = [f[1] for f in all_facs]

             # If we already have selected facilities (e.g. "Adigrat and Abiadi by facility"), 
             # just ensure comparison mode is on so we see them side-by-side
             elif selected_facility_uids:
                 comparison_mode = True
                 comparison_entity = "facility" 

        # --- OPTIONAL LLM ASSIST (Fill Missing KPI/Intent/Date/Facilities) ---
        llm_mode = self.last_parse_trace.get("parser_mode") or "fallback"

        month_tokens = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
        has_date_hint = (
            bool(re.search(r"\b(19|20)\d{2}\b", query_lower))
            or any(tok in query_lower for tok in month_tokens)
            or any(
                tok in query_lower
                for tok in (
                    "last ",
                    "this ",
                    "since ",
                    "from ",
                    " to ",
                    "between ",
                    "today",
                    "yesterday",
                )
            )
        )

        excluded_intents_for_llm = (
            {"clear", "scope_error", "scope_error_hallucination", "chart_options"}
            if llm_mode == "always"
            else {
                "metadata_query",
                "list_kpis",
                "clear",
                "scope_error",
                "scope_error_hallucination",
                "chart_options",
                "chat",
            }
        )

        wants_visual = bool(
            re.search(
                r"\b(show|see|display|view|plot|graph|chart|trend|visualize)\b",
                query_lower,
            )
        )
        wants_single_value = any(
            token in query_lower
            for token in [
                "what is",
                "how many",
                "value",
                "number",
                "count",
            ]
        )

        should_try_llm = (
            llm_enabled
            and llm_mode in {"always", "fallback"}
            and assistant_response is None
            and intent not in excluded_intents_for_llm
            and (
                llm_mode == "always"
                or not selected_kpi
                or (intent == "text" and selected_kpi and wants_visual and not wants_single_value)
                or (not start_date and has_date_hint)
                or (comparison_mode and comparison_entity == "facility" and len(selected_facility_uids) < 2)
                or (facility_requested_but_unresolved and not selected_facility_uids)
            )
        )

        llm_orientation_hint = None
        if should_try_llm:
            self.last_parse_trace["attempted"] = True

            # Provide lightweight prior context to the LLM for follow-up questions.
            llm_user_query = query
            prev_ctx = st.session_state.get("chatbot_context", {})
            if isinstance(prev_ctx, dict) and prev_ctx:
                prev_summary = {}
                prev_source = prev_ctx.get("source")
                if isinstance(prev_source, str) and prev_source.strip():
                    prev_summary["source"] = prev_source.strip()

                # Always include program-agnostic scope/time so follow-ups can inherit even across programs.
                if prev_ctx.get("facility_names"):
                    prev_summary["facility_names"] = list(prev_ctx.get("facility_names") or [])[:5]
                if prev_ctx.get("date_range"):
                    prev_summary["date_range"] = prev_ctx.get("date_range")
                if prev_ctx.get("region_filter"):
                    prev_summary["region_filter"] = prev_ctx.get("region_filter")
                if prev_ctx.get("intent"):
                    prev_summary["intent"] = prev_ctx.get("intent")
                if prev_ctx.get("chart_type"):
                    prev_summary["chart_type"] = prev_ctx.get("chart_type")
                if prev_ctx.get("period_label"):
                    prev_summary["period_label"] = prev_ctx.get("period_label")
                if prev_ctx.get("comparison_mode") is not None:
                    prev_summary["comparison_mode"] = prev_ctx.get("comparison_mode")
                if prev_ctx.get("comparison_entity"):
                    prev_summary["comparison_entity"] = prev_ctx.get("comparison_entity")
                if prev_ctx.get("comparison_targets"):
                    prev_summary["comparison_targets"] = list(prev_ctx.get("comparison_targets") or [])[:5]

                # Only carry KPI names when it is the same active program.
                if prev_source == active_program and prev_ctx.get("kpi"):
                    prev_summary["kpi"] = prev_ctx.get("kpi")

                if prev_summary and (follow_up or ctx_has_state):
                    llm_user_query = (
                        f"PREVIOUS_CONTEXT: {json.dumps(prev_summary, ensure_ascii=False)}\n\n"
                        f"USER_QUERY: {query}"
                    )

            llm_regions_list = None
            try:
                llm_regions_list = list(get_facilities_grouped_by_region(self.user).keys())
            except Exception:
                llm_regions_list = None

            llm_result, llm_error = query_llm_detailed(
                llm_user_query,
                facilities_list=list(self.facility_mapping.items()),
                regions_list=llm_regions_list,
                kpi_mapping=active_kpi_mapping,
                program_label=active_config.get("label"),
            )

            if llm_error:
                self.last_parse_trace["failed"] = True
                self.last_parse_trace["error"] = llm_error
            elif isinstance(llm_result, dict):
                self.last_parse_trace["used"] = True
                llm_primary = llm_mode == "always"
                llm_intent = llm_result.get("intent")
                llm_kpi = llm_result.get("kpi")

                llm_kpi_resolved = None
                if isinstance(llm_kpi, str):
                    raw_candidate = llm_kpi.strip()
                    if raw_candidate:
                        resolved = None
                        if raw_candidate in active_kpi_mapping:
                            resolved = raw_candidate
                        else:
                            def _norm_kpi(val: str) -> str:
                                text = re.sub(r"[^a-z0-9\\s%]", " ", str(val or "").lower())
                                return re.sub(r"\s+", " ", text).strip()

                            raw_norm = _norm_kpi(raw_candidate)
                            raw_compact = raw_norm.replace(" ", "") if raw_norm else ""

                            # Exact (normalized) match against KPI names
                            kpi_norm_map = {_norm_kpi(k): k for k in active_kpi_mapping.keys()}
                            if raw_norm:
                                resolved = kpi_norm_map.get(raw_norm)

                            # Exact alias match (helps if the LLM returns a synonym)
                            if not resolved and raw_norm:
                                alias_map = active_config.get("kpi_aliases") or {}
                                alias_norm_map = {
                                    _norm_kpi(a): v
                                    for a, v in (alias_map or {}).items()
                                    if isinstance(v, str) and v in active_kpi_mapping
                                }
                                resolved = alias_norm_map.get(raw_norm)

                            # Compact match (remove spaces / punctuation)
                            if not resolved and raw_compact:
                                kpi_compact_map = {
                                    _norm_kpi(k).replace(" ", ""): k for k in active_kpi_mapping.keys()
                                }
                                resolved = kpi_compact_map.get(raw_compact)

                            # Case-insensitive exact match
                            if not resolved:
                                ci_map = {k.lower(): k for k in active_kpi_mapping.keys()}
                                resolved = ci_map.get(raw_candidate.lower())

                            # Fuzzy match (exact KPI keys)
                            if not resolved:
                                matches = difflib.get_close_matches(
                                    raw_candidate,
                                    list(active_kpi_mapping.keys()),
                                    n=1,
                                    cutoff=0.82,
                                )
                                if matches:
                                    resolved = matches[0]

                            # Fuzzy match (normalized KPI keys)
                            if not resolved and raw_norm:
                                n_matches = difflib.get_close_matches(
                                    raw_norm,
                                    list(kpi_norm_map.keys()),
                                    n=1,
                                    cutoff=0.72,
                                )
                                if n_matches:
                                    resolved = kpi_norm_map.get(n_matches[0])

                        llm_kpi_resolved = resolved

                # Guardrail: if the user asked "what is <term>" (or similar) and the LLM guessed a KPI
                # that isn't actually referenced by the term, treat it as a general chat/out-of-scope question.
                # This prevents cases like: "what is imnid" -> random KPI definition.
                term_question = bool(
                    re.search(
                        r"\b(what is|what's|whats|meaning of|define|definition of)\b",
                        query_norm,
                    )
                )
                if llm_primary and term_question and isinstance(llm_kpi_resolved, str) and llm_kpi_resolved in active_kpi_mapping:
                    def _norm_topic(val: str) -> str:
                        text = re.sub(r"[^a-z0-9\\s%]", " ", str(val or "").lower())
                        return re.sub(r"\s+", " ", text).strip()

                    def _extract_term_topic(text: str) -> str | None:
                        q = _norm_topic(text)
                        if not q:
                            return None
                        for pat in (
                            r"\bwhat is\b\s+(.+)$",
                            r"\bwhat'?s\b\s+(.+)$",
                            r"\bmeaning of\b\s+(.+)$",
                            r"\bdefinition of\b\s+(.+)$",
                            r"\bdefine\b\s+(.+)$",
                        ):
                            m = re.search(pat, q)
                            if m:
                                topic = m.group(1).strip()
                                topic = re.sub(r"\b(in|for|at|from|to|between|by)\b.*$", "", topic).strip()
                                return topic or None
                        return None

                    def _topic_matches_kpi(topic: str, kpi_name: str) -> bool:
                        topic_norm = _norm_topic(topic)
                        if not topic_norm:
                            return True
                        topic_compact = topic_norm.replace(" ", "")

                        kpi_norm = _norm_topic(kpi_name)
                        kpi_compact = kpi_norm.replace(" ", "") if kpi_norm else ""

                        if kpi_norm and (topic_norm in kpi_norm or kpi_norm in topic_norm):
                            return True
                        if kpi_compact and (topic_compact in kpi_compact or kpi_compact in topic_compact):
                            return True

                        # Alias-based match (tolerant to typos)
                        alias_map = active_config.get("kpi_aliases") or {}
                        for alias, mapped in (alias_map or {}).items():
                            if mapped != kpi_name:
                                continue
                            a_norm = _norm_topic(alias)
                            if not a_norm:
                                continue
                            a_compact = a_norm.replace(" ", "")
                            if a_norm in topic_norm or topic_norm in a_norm:
                                return True
                            if topic_compact and a_compact:
                                if difflib.SequenceMatcher(None, topic_compact, a_compact).ratio() >= 0.70:
                                    return True

                        # Fuzzy against KPI name as a fallback
                        if topic_compact and kpi_compact:
                            if difflib.SequenceMatcher(None, topic_compact, kpi_compact).ratio() >= 0.62:
                                return True

                        return False

                    topic = _extract_term_topic(query_norm)
                    if topic and not _topic_matches_kpi(topic, llm_kpi_resolved):
                        topic_compact = topic.replace(" ", "")
                        if "imnid" in topic_compact:
                            assistant_response = (
                                "**IMNID** is the name of this dashboard/program.\n\n"
                                "I can help you plot maternal/newborn indicators, or list facilities/regions.\n"
                                "Try: `list indicators` or `plot PPH rate for Tigray`."
                            )
                        else:
                            assistant_response = (
                                f"I don't have dashboard knowledge to answer **{topic}**.\n\n"
                                "I can help you with IMNID dashboard analysis (plot KPIs, list indicators, facilities, regions). "
                                "Try: `list indicators`."
                            )

                        intent = "chat"
                        llm_intent = "chat"
                        llm_kpi_resolved = None
                        selected_kpi = None

                if (
                    isinstance(llm_kpi_resolved, str)
                    and llm_kpi_resolved in active_kpi_mapping
                    and not missing_birth_outcome_lock
                    and (llm_primary or not selected_kpi)
                ):
                    selected_kpi = llm_kpi_resolved

                force_list_kpis = explicit_list_intent == "list_kpis"
                force_metadata_list = (
                    explicit_list_intent == "metadata_query"
                    and not selected_kpi  # no KPI found even after LLM resolution above
                )

                if llm_primary and llm_intent in {
                    "plot",
                    "distribution",
                    "definition",
                    "text",
                    "metadata_query",
                    "chat",
                    "list_kpis",
                }:
                    if (
                        selected_kpi
                        and llm_intent in {"metadata_query", "list_kpis"}
                        and not force_list_kpis
                        and not force_metadata_list
                    ):
                        # If we already have a KPI, do not let the LLM derail into listing metadata.
                        pass
                    elif force_list_kpis and llm_intent != "list_kpis":
                        pass
                    elif force_metadata_list and llm_intent not in {"metadata_query", "chat"}:
                        pass
                    else:
                        intent = llm_intent
                elif intent == "text" and llm_intent in {"plot", "distribution", "definition", "text"}:
                    if (
                        selected_kpi
                        and llm_intent in {"metadata_query", "list_kpis"}
                        and not force_list_kpis
                        and not force_metadata_list
                    ):
                        pass
                    elif force_list_kpis:
                        pass
                    elif force_metadata_list and llm_intent not in {"metadata_query", "chat"}:
                        pass
                    else:
                        intent = llm_intent

                if llm_intent == "list_kpis" and not force_metadata_list and not selected_kpi:
                    intent = "list_kpis"

                llm_response = llm_result.get("response")
                if llm_intent == "chat" and isinstance(llm_response, str) and llm_response.strip():
                    intent = "chat"
                    assistant_response = llm_response.strip()

                if llm_primary and intent == "metadata_query":
                    llm_entity_type = llm_result.get("entity_type")
                    if llm_entity_type in {"facility", "region"}:
                        entity_type = llm_entity_type

                    llm_count_requested = llm_result.get("count_requested")
                    if isinstance(llm_count_requested, bool):
                        count_requested = llm_count_requested

                # Region filter / region-as-location (works for plot + metadata_query).
                llm_region_filter_raw = llm_result.get("region_filter")
                if isinstance(llm_region_filter_raw, str) and llm_region_filter_raw.strip():
                    try:
                        llm_regions_mapping = get_facilities_grouped_by_region(self.user)
                    except Exception:
                        llm_regions_mapping = {}

                    region_key_map = {r.lower(): r for r in (llm_regions_mapping or {}).keys()}
                    region_compact_map = {
                        re.sub(r"\s+", "", str(r).strip().lower()): r
                        for r in (llm_regions_mapping or {}).keys()
                        if str(r).strip()
                    }

                    candidate_region = re.sub(r"\s+", " ", llm_region_filter_raw).strip().lower()
                    candidate_region = re.sub(r"[^a-z0-9\\s]", " ", candidate_region)
                    candidate_region = re.sub(r"\s+", " ", candidate_region).strip()
                    matched_region = None

                    if candidate_region and region_key_map:
                        matched_region = region_key_map.get(candidate_region)
                        if not matched_region:
                            cand_compact = candidate_region.replace(" ", "")
                            if cand_compact and region_compact_map:
                                matched_region = region_compact_map.get(cand_compact)

                        if not matched_region:
                            r_matches = difflib.get_close_matches(
                                candidate_region,
                                list(region_key_map.keys()),
                                n=1,
                                cutoff=0.65,
                            )
                            if r_matches:
                                matched_region = region_key_map.get(r_matches[0])

                        if not matched_region and region_compact_map:
                            cand_compact = candidate_region.replace(" ", "")
                            if cand_compact:
                                cm = difflib.get_close_matches(
                                    cand_compact,
                                    list(region_compact_map.keys()),
                                    n=1,
                                    cutoff=0.65,
                                )
                                if cm:
                                    matched_region = region_compact_map.get(cm[0])

                    if matched_region:
                        region_filter = matched_region
                        if matched_region not in found_regions:
                            found_regions.append(matched_region)

                        use_region_scope = (
                            (llm_primary and not facility_explicitly_mentioned)
                            or not selected_facility_uids
                        )
                        if use_region_scope and (llm_regions_mapping or {}).get(matched_region):
                            region_uids = []
                            for fac in llm_regions_mapping.get(matched_region, []) or []:
                                uid = (
                                    fac[1]
                                    if isinstance(fac, (list, tuple)) and len(fac) > 1
                                    else fac
                                )
                                if uid and uid not in region_uids:
                                    region_uids.append(uid)
                            selected_facility_uids = region_uids
                            selected_facility_names = [f"{matched_region} (Region)"]
                            facility_requested_but_unresolved = False

                llm_period_label = llm_result.get("period_label")
                if (llm_primary or not period_label) and llm_period_label in {
                    "Daily",
                    "Weekly",
                    "Monthly",
                    "Quarterly",
                    "Yearly",
                }:
                    period_label = llm_period_label

                llm_analysis_type = llm_result.get("analysis_type")
                if not analysis_type and llm_analysis_type in {"max", "min"}:
                    analysis_type = llm_analysis_type
                    intent = "text"

                llm_orientation = llm_result.get("orientation")
                if llm_orientation in {"h", "v"}:
                    llm_orientation_hint = llm_orientation

                if not start_date:
                    llm_date_range = llm_result.get("date_range")
                    if isinstance(llm_date_range, dict):
                        llm_start = llm_date_range.get("start_date")
                        llm_end = llm_date_range.get("end_date")
                        if isinstance(llm_start, str) and isinstance(llm_end, str):
                            try:
                                datetime.strptime(llm_start, "%Y-%m-%d")
                                datetime.strptime(llm_end, "%Y-%m-%d")
                                start_date = llm_start
                                end_date = llm_end
                            except Exception:
                                pass

                llm_chart_type = llm_result.get("chart_type")
                if llm_chart_type in {"line", "bar", "area", "table"} and (llm_primary or chart_type == "line"):
                    chart_type = llm_chart_type

                if not comparison_mode and llm_result.get("comparison_mode") is True:
                    comparison_mode = True
                    llm_comp_entity = llm_result.get("comparison_entity")
                    if llm_comp_entity in {"facility", "region"}:
                        comparison_entity = llm_comp_entity
                elif llm_primary and comparison_mode and not comparison_entity:
                    llm_comp_entity = llm_result.get("comparison_entity")
                    if llm_comp_entity in {"facility", "region"}:
                        comparison_entity = llm_comp_entity

                llm_facility_names = llm_result.get("facility_names")
                should_fill_facilities = not selected_facility_uids or (
                    comparison_mode and comparison_entity == "facility" and len(selected_facility_uids) < 2
                )
                if should_fill_facilities and isinstance(llm_facility_names, list):
                    seen = set(selected_facility_uids)
                    norm_candidates = list(self.suffix_blind_mapping.keys())

                    llm_regions_mapping = {}
                    region_key_map = {}
                    try:
                        llm_regions_mapping = get_facilities_grouped_by_region(self.user)
                        region_key_map = {r.lower(): r for r in (llm_regions_mapping or {}).keys()}
                        region_compact_map = {
                            re.sub(r"\s+", "", str(r).strip().lower()): r
                            for r in (llm_regions_mapping or {}).keys()
                            if str(r).strip()
                        }
                    except Exception:
                        llm_regions_mapping = {}
                        region_key_map = {}
                        region_compact_map = {}

                    matched_llm_regions: list[str] = []

                    for raw_name in llm_facility_names[:8]:
                        candidate = re.sub(r"\s+", " ", str(raw_name or "")).strip().lower()
                        candidate = re.sub(r"[^a-z0-9\\s]", " ", candidate)
                        candidate = re.sub(r"\s+", " ", candidate).strip()
                        if not candidate:
                            continue
                        if candidate in self.AMBIGUOUS_PREFIXES:
                            continue

                        # Allow region names (and typos) in the facility_names list.
                        if region_key_map:
                            region_match = region_key_map.get(candidate)
                            if not region_match:
                                cand_compact = candidate.replace(" ", "")
                                if cand_compact and region_compact_map:
                                    region_match = region_compact_map.get(cand_compact)

                            if not region_match:
                                r_matches = difflib.get_close_matches(
                                    candidate,
                                    list(region_key_map.keys()),
                                    n=1,
                                    cutoff=0.65,
                                )
                                if r_matches:
                                    region_match = region_key_map.get(r_matches[0])

                            if not region_match and region_compact_map:
                                cand_compact = candidate.replace(" ", "")
                                if cand_compact:
                                    cm = difflib.get_close_matches(
                                        cand_compact,
                                        list(region_compact_map.keys()),
                                        n=1,
                                        cutoff=0.65,
                                    )
                                    if cm:
                                        region_match = region_compact_map.get(cm[0])

                            if region_match:
                                matched_llm_regions.append(region_match)
                                region_label = f"{region_match} (Region)"
                                if region_label not in selected_facility_names:
                                    selected_facility_names.append(region_label)

                                for fac in (llm_regions_mapping or {}).get(region_match, []) or []:
                                    uid = (
                                        fac[1]
                                        if isinstance(fac, (list, tuple)) and len(fac) > 1
                                        else fac
                                    )
                                    if uid and uid not in seen:
                                        selected_facility_uids.append(uid)
                                        seen.add(uid)

                                facility_requested_but_unresolved = False
                                continue

                        official_name = None
                        if candidate in self.suffix_blind_mapping:
                            official_name = self.suffix_blind_mapping[candidate]
                        else:
                            # Handle compact typos (missing spaces) e.g. "abiadi" -> "abi adi"
                            cand_compact = candidate.replace(" ", "")
                            if cand_compact:
                                direct = (self.suffix_blind_compact_index or {}).get(cand_compact)
                                if direct:
                                    official_name = direct[0] if isinstance(direct, list) and direct else None

                            # Fuzzy match (dynamic cutoff by length to tolerate typos safely)
                            if not official_name:
                                if len(candidate) <= 4:
                                    cutoff = 0.90
                                elif len(candidate) <= 6:
                                    cutoff = 0.82
                                else:
                                    cutoff = 0.75

                                matches = difflib.get_close_matches(
                                    candidate,
                                    norm_candidates,
                                    n=1,
                                    cutoff=cutoff,
                                )
                                if matches:
                                    official_name = self.suffix_blind_mapping.get(matches[0])

                            # Compact fuzzy match as a last resort
                            if not official_name and cand_compact and getattr(self, "suffix_blind_compact_keys", None):
                                if len(cand_compact) <= 4:
                                    cutoff = 0.90
                                elif len(cand_compact) <= 6:
                                    cutoff = 0.80
                                else:
                                    cutoff = 0.72
                                cm = difflib.get_close_matches(
                                    cand_compact,
                                    self.suffix_blind_compact_keys,
                                    n=1,
                                    cutoff=cutoff,
                                )
                                if cm:
                                    opts = (self.suffix_blind_compact_index or {}).get(cm[0]) or []
                                    if isinstance(opts, list) and opts:
                                        official_name = opts[0]

                        if not official_name:
                            continue

                        uid = self.universal_facility_mapping.get(official_name)
                        if uid and uid not in seen:
                            selected_facility_uids.append(uid)
                            selected_facility_names.append(official_name)
                            seen.add(uid)

                    if matched_llm_regions:
                        for r_name in matched_llm_regions:
                            if r_name not in found_regions:
                                found_regions.append(r_name)
                        if comparison_mode and comparison_entity in {None, "region"}:
                            comparison_entity = "region"
            else:
                self.last_parse_trace["failed"] = True
                self.last_parse_trace["error"] = "LLM returned no result"

        # Follow-up override (post-LLM): if the last action was a plot and the user asks
        # "what about <another KPI>" without requesting a single number, keep plotting.
        if follow_up and prev_ctx_intent == "plot":
            explicit_value_question = bool(
                "what is" in query_lower
                or re.search(r"\b(how many|count|number|total|value)\b", query_lower)
            )
            if not explicit_value_question and intent == "text" and not analysis_type:
                intent = "plot"

        final_date_range = {"start_date": start_date, "end_date": end_date} if start_date else None
        # Do NOT reuse prior context date range; default to all time unless explicitly provided

        # Follow-up: if we previously asked the user to specify a KPI, reuse prior filters
        # (location/date range) when the new message supplies only the indicator.
        ctx_follow = st.session_state.get("chatbot_context") or {}
        if isinstance(ctx_follow, dict) and ctx_follow.get("last_question") == "need_kpi":
            if selected_kpi:
                if not selected_facility_uids and not region_filter:
                    prev_uids = ctx_follow.get("facility_uids")
                    prev_names = ctx_follow.get("facility_names")
                    if isinstance(prev_uids, list) and prev_uids:
                        selected_facility_uids = list(prev_uids)
                    if isinstance(prev_names, list) and prev_names:
                        selected_facility_names = list(prev_names)

                    prev_region = ctx_follow.get("region_filter")
                    if isinstance(prev_region, str) and prev_region.strip():
                        region_filter = prev_region.strip()

                if not final_date_range:
                    prev_dr = ctx_follow.get("date_range")
                    if (
                        isinstance(prev_dr, dict)
                        and prev_dr.get("start_date")
                        and prev_dr.get("end_date")
                    ):
                        final_date_range = {
                            "start_date": prev_dr.get("start_date"),
                            "end_date": prev_dr.get("end_date"),
                        }
                        start_date = final_date_range["start_date"]
                        end_date = final_date_range["end_date"]

                prev_chart_type = ctx_follow.get("chart_type")
                if prev_chart_type in {"line", "bar", "area", "table"} and chart_type == "line":
                    chart_type = prev_chart_type

                prev_period = ctx_follow.get("period_label")
                if (
                    not period_label
                    and prev_period in {"Daily", "Weekly", "Monthly", "Quarterly", "Yearly"}
                ):
                    period_label = prev_period

                try:
                    st.session_state.chatbot_context["last_question"] = None
                except Exception:
                    pass
         
        # Merge Entity Type (For "Name them" queries)
        if intent == "text" and not selected_kpi and not entity_type:
             pass 

        # --- ACCESS CONTROL CHECK ---
        # Ensure user has access to ALL selected facilities
        # self.facility_mapping contains ONLY allowed facilities for this user.
        # self.universal_facility_mapping contains ALL facilities.
        # detected uids are in selected_facility_uids.
        
        # Identify denied UIDs
        denied_uids = []
        allowed_uids = list(self.facility_mapping.values())
        
        # Only check if user is NOT national/admin (who usually sees all)
        # But self.facility_mapping should already reflect their role limits via get_facility_mapping_for_user
        # So we just check against self.facility_mapping.
        
        # However, for regional/national users, facility_mapping might be large.
        # Let's trust self.facility_mapping as the source of truth for access.
        
        # Important: selected_facility_uids might contain duplicates
        for uid in set(selected_facility_uids):
            if uid not in allowed_uids:
                denied_uids.append(uid)
                
        if denied_uids:
            # We found facilities the user is NOT allowed to see
            # Get their names for the error message
            denied_names = []
            for uid in denied_uids:
                # Use universal map to enforce finding the name even if not in allowed map
                name = self.uid_to_name.get(uid, "Unknown Facility")
                denied_names.append(name)
            
            denied_list_str = ", ".join([f"**{n}**" for n in denied_names])
            
            return {
                "intent": "chat",
                "response": f"🚫 **Access Denied**: You do not have permission to view data for: {denied_list_str}.\n\nPlease contact your administrator if you believe this is an error."
            }

        # Final unresolved location guard (after region detection + LLM attempt).
        if facility_requested_but_unresolved and not selected_facility_uids:
            location_candidates: list[str] = []
            try:
                location_candidates = _extract_location_candidates(query_norm)
            except Exception:
                location_candidates = []

            region_suggestions: list[str] = []
            facility_suggestions: list[str] = []

            try:
                _regions_mapping = get_facilities_grouped_by_region(self.user)
            except Exception:
                _regions_mapping = {}

            region_key_map = {str(r).strip().lower(): r for r in (_regions_mapping or {}).keys()}
            region_compact_map = {
                re.sub(r"\s+", "", str(r).strip().lower()): r
                for r in (_regions_mapping or {}).keys()
                if str(r).strip()
            }

            def _norm_loc(val: str) -> str:
                text = re.sub(r"[^a-z0-9\\s]", " ", str(val or "").lower())
                return re.sub(r"\s+", " ", text).strip()

            for raw in location_candidates[:3]:
                cand = _norm_loc(raw)
                if not cand:
                    continue

                # Region suggestions (closest match)
                if region_key_map:
                    direct = region_key_map.get(cand)
                    if direct and direct not in region_suggestions:
                        region_suggestions.append(direct)
                    else:
                        cand_compact = cand.replace(" ", "")
                        direct_c = region_compact_map.get(cand_compact) if cand_compact else None
                        if direct_c and direct_c not in region_suggestions:
                            region_suggestions.append(direct_c)
                        else:
                            r_matches = difflib.get_close_matches(
                                cand,
                                list(region_key_map.keys()),
                                n=3,
                                cutoff=0.55,
                            )
                            for m in r_matches:
                                reg = region_key_map.get(m)
                                if reg and reg not in region_suggestions:
                                    region_suggestions.append(reg)

                            if cand_compact and region_compact_map:
                                rc = difflib.get_close_matches(
                                    cand_compact,
                                    list(region_compact_map.keys()),
                                    n=3,
                                    cutoff=0.55,
                                )
                                for m in rc:
                                    reg = region_compact_map.get(m)
                                    if reg and reg not in region_suggestions:
                                        region_suggestions.append(reg)

                # Facility suggestions (closest match)
                if cand in self.suffix_blind_mapping:
                    off = self.suffix_blind_mapping.get(cand)
                    if off and off not in facility_suggestions:
                        facility_suggestions.append(off)
                else:
                    cand_compact = cand.replace(" ", "")
                    if cand_compact and (self.suffix_blind_compact_index or {}).get(cand_compact):
                        opts = (self.suffix_blind_compact_index or {}).get(cand_compact) or []
                        for off in (opts[:3] if isinstance(opts, list) else []):
                            if off and off not in facility_suggestions:
                                facility_suggestions.append(off)

                    if len(cand) <= 4:
                        cutoff = 0.88
                    elif len(cand) <= 6:
                        cutoff = 0.80
                    else:
                        cutoff = 0.70

                    f_matches = difflib.get_close_matches(
                        cand,
                        list(self.suffix_blind_mapping.keys()),
                        n=5,
                        cutoff=cutoff,
                    )
                    for m in f_matches:
                        off = self.suffix_blind_mapping.get(m)
                        if off and off not in facility_suggestions:
                            facility_suggestions.append(off)

                    if cand_compact and getattr(self, "suffix_blind_compact_keys", None):
                        if len(cand_compact) <= 4:
                            cutoff = 0.88
                        elif len(cand_compact) <= 6:
                            cutoff = 0.78
                        else:
                            cutoff = 0.68
                        fc = difflib.get_close_matches(
                            cand_compact,
                            self.suffix_blind_compact_keys,
                            n=5,
                            cutoff=cutoff,
                        )
                        for m in fc:
                            opts = (self.suffix_blind_compact_index or {}).get(m) or []
                            for off in (opts[:2] if isinstance(opts, list) else []):
                                if off and off not in facility_suggestions:
                                    facility_suggestions.append(off)

            suggestion_lines: list[str] = []
            for r in region_suggestions[:3]:
                suggestion_lines.append(f"- `{r}` (region)")
            for f in facility_suggestions[:3]:
                suggestion_lines.append(f"- `{f}` (facility)")

            suggestion_block = ""
            if suggestion_lines:
                suggestion_block = "Closest matches:\n" + "\n".join(suggestion_lines) + "\n\n"

            return {
                "intent": "chat",
                "response": (
                    "⚠️ **Facility/region not found!**\n\n"
                    "I couldn't match the location you typed to a known **facility** or **region**.\n\n"
                    f"{suggestion_block}"
                    "Please try:\n"
                    "1. Checking the spelling\n"
                    "2. Using the full facility name (e.g., `Adigrat Hospital`)\n"
                    "3. Saying `list regions` or `list facilities` to see available options"
                ),
            }

        # Horizontal Chart Detection
        orientation = "v"
        if "horizontal" in query_lower:
            orientation = "h"
        elif llm_orientation_hint in {"h", "v"}:
            orientation = llm_orientation_hint

        if orientation == "h" and chart_type == "line":
            chart_type = "bar"  # line charts don't render well horizontally
        
        # Auto-detect Granularity for Short Date Ranges
        # if final_date_range and not period_label:
        #     try:
        #         s = datetime.strptime(final_date_range["start_date"], "%Y-%m-%d")
        #         e = datetime.strptime(final_date_range["end_date"], "%Y-%m-%d")
        #         delta = (e - s).days
        #         if delta <= 45: # If range is 1.5 months or less
        #             period_label = "Daily"
        #     except:
        #         pass

        # Infer Comparison Entity if not explicitly set (e.g. "by facility" not used)
        # If multiple regions were identified, treat it as a region comparison even if the user
        # didn't explicitly say "compare" (e.g., "Tigray and Amhara").
        if (
            not comparison_mode
            and intent in {"plot", "distribution", "text"}
            and isinstance(found_regions, list)
            and len(found_regions) >= 2
        ):
            comparison_mode = True
            comparison_entity = "region"

        if comparison_mode and not comparison_entity:
            if selected_facility_uids:
                comparison_entity = "facility"
            elif found_regions:
                # Prioritize Facility if both present? usually facility is more specific.
                # But if we found regions and NO facility UIDs, then region.
                comparison_entity = "region"
        
        # If comparison by facility was requested but no facilities resolved, only auto-expand to
        # "all accessible facilities" when the user explicitly asked for an all/by-facility view.
        if comparison_mode and comparison_entity == "facility" and not selected_facility_uids:
            wants_all_facilities = any(
                token in query_lower
                for token in (
                    "all facilities",
                    "all facility",
                    "every facility",
                    "each facility",
                    "all hospitals",
                    "every hospital",
                    "each hospital",
                    "by facility",
                    "per facility",
                    "breakdown by facility",
                )
            )
            if wants_all_facilities:
                all_facs = get_all_facilities_flat(self.user)
                selected_facility_names = [f[0] for f in all_facs]
                selected_facility_uids = [f[1] for f in all_facs]

        return {
            "intent": intent,
            "chart_type": chart_type,
            "orientation": orientation,
            "analysis_type": None, # Start with None
            "kpi": selected_kpi,
            "facility_uids": selected_facility_uids,
            "facility_names": selected_facility_names,
            "date_range": final_date_range,
            "period_label": period_label,
            "analysis_type": analysis_type,
            "entity_type": entity_type,
            "region_filter": region_filter,
            "count_requested": count_requested,
            "comparison_mode": comparison_mode,
            "comparison_entity": comparison_entity,
            "comparison_targets": found_regions if comparison_mode and comparison_entity == "region" and found_regions else [],
            "is_drill_down": is_drill_down,
            "response": assistant_response
        }

    def _get_cache_key(self, parsed_query, facility_uids=None):
        """Create a unique cache key for the query"""
        import hashlib
        import json
        
        # Sort UIDs for consistent hashing
        sorted_uids = sorted(facility_uids) if facility_uids else []
        
        key_data = {
            "program": self.get_selected_program(),
            "kpi": parsed_query.get("kpi"),
            "facility_count": len(sorted_uids),
            "facility_hash": hashlib.md5(str(sorted_uids).encode()).hexdigest(),
            "date_range": parsed_query.get("date_range"),
            "intent": parsed_query.get("intent"),
            "comparison_mode": parsed_query.get("comparison_mode"),
            "user_role": self.user.get("role")
        }
        
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key):
        """Get cached result if not expired"""
        import time
        
        # Initialize cache if needed
        if not hasattr(self, "query_cache"):
            self.query_cache = {}
            self.cache_expiry = 30  # seconds
            
        if cache_key in self.query_cache:
            result, timestamp = self.query_cache[cache_key]
            if time.time() - timestamp < self.cache_expiry:
                logging.info(f"✅ Chatbot: Using cached result for {cache_key[:8]}")
                return result
            else:
                del self.query_cache[cache_key]
        return None
    
    def _cache_result(self, cache_key, result):
        """Cache the query result"""
        import time
        
        # Initialize cache if needed
        if not hasattr(self, "query_cache"):
            self.query_cache = {}
            self.cache_expiry = 30
            
        self.query_cache[cache_key] = (result, time.time())
        # Limit cache size
        if len(self.query_cache) > 50:
            # Remove oldest entry
            oldest_key = min(self.query_cache.items(), key=lambda x: x[1][1])[0]
            del self.query_cache[oldest_key]

    def generate_response(self, query):
        global KPI_MAPPING, KPI_OPTIONS
        query_lower = query.lower()
        role = self.user.get("role", "facility")
        selected_program = self.get_selected_program()
        detected_program = detect_program_from_text(query)

        # Auto-select program: prefer detected hints; otherwise default to maternal.
        #
        # IMPORTANT: Do NOT clear conversation context on *auto-switch* (this breaks follow-up questions
        # like "what about admitted mothers" after a newborn plot). Only clear context when the user
        # explicitly switches programs (selection-only input like "newborn" / "maternal").
        if detected_program and detected_program != selected_program:
            clear_ctx = bool(self.is_program_selection_only(query))
            self.set_selected_program(detected_program, clear_context=clear_ctx)
            selected_program = detected_program
        if not selected_program:
            self.set_selected_program(detected_program or "maternal", clear_context=True)
            selected_program = self.get_selected_program()

        active_program_config = self.apply_active_program_globals(selected_program)
        parsed = self.parse_query(query)
        llm_provider, llm_model = get_llm_provider_and_model()
        self.last_trace = {
            "parser": getattr(self, "last_parse_trace", None),
            "insight": {
                "enabled": bool(getattr(settings, "CHATBOT_USE_LLM_INSIGHTS", False)),
                "attempted": False,
                "used": False,
                "failed": False,
                "provider": llm_provider,
                "model": llm_model,
            },
        }

        # Guard: if selected program is newborn, prevent maternal-only KPIs from proceeding
        if selected_program == "newborn" and parsed.get("kpi"):
            from components.chatbot_newborn import validate_newborn_indicator
            is_valid, err_msg = validate_newborn_indicator(parsed["kpi"], raw_query=query)
            if not is_valid:
                return None, err_msg
        # Guard: if selected program is maternal, prevent newborn-only KPIs from proceeding
        if selected_program == "maternal" and parsed.get("kpi") and parsed["kpi"] not in active_program_config["kpi_mapping"]:
            if parsed["kpi"] in PROGRAM_CONFIGS["newborn"]["kpi_mapping"]:
                return None, "That looks like a newborn indicator. Type `newborn` to switch programs, then ask again."
            # If user clearly asked for missing birth weight, guide to newborn
            if "missing birth weight" in query_lower or "missing birthweight" in query_lower:
                return None, "Missing Birth Weight is tracked in the newborn program. Type `newborn` then ask again, or say `list indicators` to see maternal options."
            return None, "I couldn't match that to a maternal indicator. Say `list indicators` to see maternal options."

        # If the user did not specify a facility/region in their prompt, optionally default to the
        # selected dashboard scope (so chatbot plots match the manually navigated dashboard views).
        # Default behavior is "all accessible" unless CHATBOT_DEFAULT_SCOPE="dashboard".
        default_scope = str(getattr(settings, "CHATBOT_DEFAULT_SCOPE", "all") or "all").strip().lower()
        if (
            default_scope == "dashboard"
            and isinstance(parsed, dict)
            and parsed.get("intent") not in {"metadata_query", "list_kpis", "clear", "chat"}
        ):
            explicit_all_scope = any(
                token in query_lower
                for token in (
                    "all facilities",
                    "all facility",
                    "all regions",
                    "all region",
                    "every facility",
                    "every region",
                    "across all",
                    "national",
                    "countrywide",
                )
            )

            has_location = bool(
                parsed.get("facility_uids")
                or parsed.get("facility_names")
                or parsed.get("comparison_targets")
                or parsed.get("region_filter")
            )

            if not explicit_all_scope and not has_location:
                filter_mode = str(st.session_state.get("filter_mode") or "").strip()
                selected_facilities = st.session_state.get("selected_facilities") or []
                selected_regions = st.session_state.get("selected_regions") or []

                if isinstance(selected_facilities, str):
                    selected_facilities = [selected_facilities]
                if isinstance(selected_regions, str):
                    selected_regions = [selected_regions]

                selected_facilities = [
                    f
                    for f in selected_facilities
                    if isinstance(f, str) and f.strip() and f.strip() != "All Facilities"
                ]
                selected_regions = [r for r in selected_regions if isinstance(r, str) and r.strip()]

                def _dedupe_keep_order(values):
                    seen = set()
                    result = []
                    for value in values:
                        if value in seen:
                            continue
                        seen.add(value)
                        result.append(value)
                    return result

                def _map_facility_names(names):
                    resolved_names = []
                    resolved_uids = []
                    for name in names:
                        uid = self.facility_mapping.get(name)
                        if uid and uid not in resolved_uids:
                            resolved_uids.append(uid)
                            resolved_names.append(name)
                    return resolved_names, resolved_uids

                if parsed.get("comparison_mode"):
                    if parsed.get("comparison_entity") == "facility" and selected_facilities:
                        # Regional dashboards typically don't set filter_mode, so treat blank as "By Facility".
                        if not filter_mode or filter_mode == "By Facility":
                            fac_names, fac_uids = _map_facility_names(selected_facilities)
                            if len(fac_uids) >= 2:
                                parsed["facility_names"] = fac_names
                                parsed["facility_uids"] = fac_uids
                    elif (
                        parsed.get("comparison_entity") == "region"
                        and selected_regions
                        and filter_mode == "By Region"
                    ):
                        if len(selected_regions) >= 2:
                            parsed["comparison_targets"] = _dedupe_keep_order(selected_regions)
                else:
                    if selected_facilities and (not filter_mode or filter_mode == "By Facility"):
                        fac_names, fac_uids = _map_facility_names(selected_facilities)
                        if fac_uids:
                            parsed["facility_names"] = fac_names
                            parsed["facility_uids"] = fac_uids
                    elif filter_mode == "By Region" and selected_regions:
                        try:
                            regions_mapping = get_facilities_grouped_by_region(self.user)
                        except Exception:
                            regions_mapping = {}

                        flat_uids = []
                        for region in _dedupe_keep_order(selected_regions):
                            for fac in regions_mapping.get(region, []) or []:
                                uid = (
                                    fac[1]
                                    if isinstance(fac, (list, tuple)) and len(fac) > 1
                                    else fac
                                )
                                if uid and uid not in flat_uids:
                                    flat_uids.append(uid)

                if flat_uids:
                    parsed["facility_uids"] = flat_uids
                    parsed["facility_names"] = [
                        f"{r} (Region)" for r in _dedupe_keep_order(selected_regions)
                    ]

        # Guard: if user requested facility comparison but we resolved fewer than 2 facilities, ask to clarify
        if parsed.get("comparison_mode") and parsed.get("comparison_entity") == "facility":
            if len(parsed.get("facility_uids", [])) < 2:
                found = parsed.get("facility_names") or []
                found_str = ", ".join(found) if found else "none"
                return None, (
                    f"I couldn't identify at least two facilities for comparison (found: {found_str}). "
                    "Please specify the full facility names or say `list facilities` to view available options."
                )
        
        # --- PERFORMANCE CACHE ---
        cache_key = self._get_cache_key(parsed, parsed.get("facility_uids"))
        cached = self._get_cached_result(cache_key)
        if cached: return cached
        

        # Handle List KPIs Intent (Consolidated below)


        # Handle KPI Definition intent
        if parsed.get("intent") == "definition":
            kpi_name = parsed.get("kpi")
            if not kpi_name:
                return None, "I understand you're asking for a definition, but I couldn't identify which indicator you're referring to. Could you please specify a name like 'PPH rate' or 'C-Section'?"
            
            # Try to get comprehensive definition first
            active_definitions = active_program_config.get("kpi_definitions", {})
            kpi_def = active_definitions.get(kpi_name)
            if not kpi_def and hasattr(self, "KB_DEFINITIONS"):
                basic_definition = self.KB_DEFINITIONS.get(kpi_name)
                if basic_definition:
                    kpi_def = {"description": basic_definition}
            
            if kpi_def:
                # Build comprehensive response
                response = f"**{kpi_name}**\n\n"
                response += f"📋 **Definition**: {kpi_def.get('description', 'No description available.')}\n\n"
                
                if 'numerator' in kpi_def and 'denominator' in kpi_def:
                    response += f"🔢 **Calculation**:\n"
                    response += f"- **Numerator**: {kpi_def['numerator']}\n"
                    response += f"- **Denominator**: {kpi_def['denominator']}\n\n"
                elif 'value_name' in kpi_def:
                    response += f"📊 **Measure**: {kpi_def['value_name']}\n\n"
                
                if 'interpretation' in kpi_def:
                    response += f"💡 **Interpretation**: {kpi_def['interpretation']}"
                
                # Store context for follow-up
                st.session_state.chatbot_context["last_question"] = "plot_kpi"
                st.session_state.chatbot_context["pending_kpi"] = kpi_name
                
                return None, response
            else:
                # Fallback to basic config from dash_co
                kpi_config = KPI_MAPPING.get(kpi_name, {})
                num = kpi_config.get("numerator_name")
                den = kpi_config.get("denominator_name")
                
                if num and den:
                    calculation_logic = f"**{kpi_name}** is computed as:\n\n**Numerator**: {num}\n**Denominator**: {den}"
                    
                    # Store context for follow-up
                    st.session_state.chatbot_context["last_question"] = "plot_kpi"
                    st.session_state.chatbot_context["pending_kpi"] = kpi_name
                    
                    return None, f"{calculation_logic}"
                else:
                    # Fallback for volume indicators (e.g. Admitted Mothers)
                    val_name = kpi_config.get("value_name")
                    if val_name:
                        msg = f"**{kpi_name}** is a volume indicator based on: **{val_name}**."
                    else:
                        msg = f"I found the indicator **{kpi_name}**, but I don't have a specific calculation formula on file for it yet."
                    
                    # Store context for follow-up
                    st.session_state.chatbot_context["last_question"] = "plot_kpi"
                    st.session_state.chatbot_context["pending_kpi"] = kpi_name
                    
                    return None, f"{msg}"


        # Handle General Chat
        if parsed.get("intent") == "chat":
            response = parsed.get("response")
            
            if response:
                # If the assistant is asking for an indicator/KPI, persist partial filters so the
                # next user message can be treated as a follow-up (KPI selection).
                try:
                    resp_low = str(response).lower()
                except Exception:
                    resp_low = ""

                needs_kpi = bool(
                    re.search(r"\b(which|specify|choose|select)\b", resp_low)
                    and re.search(r"\b(kpi|indicator|metric|measure)\b", resp_low)
                )

                if needs_kpi:
                    ctx = st.session_state.get("chatbot_context") or {}
                    if not isinstance(ctx, dict):
                        ctx = {}

                    ctx["last_question"] = "need_kpi"
                    ctx["source"] = selected_program

                    for key in (
                        "facility_uids",
                        "facility_names",
                        "date_range",
                        "region_filter",
                        "comparison_mode",
                        "comparison_entity",
                        "comparison_targets",
                        "period_label",
                        "chart_type",
                        "orientation",
                    ):
                        val = parsed.get(key)
                        if val:
                            ctx[key] = val

                    st.session_state.chatbot_context = ctx

                return None, response
            
            # Fallback for local detection (e.g. password)
            q_low = query.lower()
            if any(x in q_low for x in ["password", "login", "admin", "credential"]):
                return None, "I'm your dashboard assistant. I don't handle system passwords or administrative access. Please contact your system administrator if you're having login issues! 🔒"
                
            return None, get_program_welcome_message(role, selected_program)
            
        # Handle Chart Options Parsing
        if parsed.get("intent") == "chart_options":
             kpi_concern = parsed.get("kpi") or st.session_state.get("chatbot_context", {}).get("kpi")
              
             if kpi_concern in {"Admitted Mothers", "Admitted Newborns"}:
                 return None, f"For **{kpi_concern}**, the available charts are:\n- **Vertical Bar Chart** (Default)\n- **Horizontal Bar Chart** (Say 'plot horizontal bar')\n- **Data Table**"
             elif kpi_concern:
                 return None, f"For **{kpi_concern}**, I can generate:\n- **Line Chart**: Best for trends over time.\n- **Bar Chart**: Good for comparison.\n- **Area Chart**: Visualizes volume over time.\n- **Data Table**: Detailed numbers."
             else:
                 return None, "I can generate the following charts for any indicator:\n- **Line Chart** (Default): 'Plot PPH trend'\n- **Bar Chart**: 'Show Admitted Mothers as bar chart'\n- **Area Chart**: 'PPH Rate area chart'\n- **Data Table**: 'Show table for C-Section'"
            
        # Handle Scope Error
        if parsed.get("intent") == "scope_error":
             return None, "I'm focused on data analysis and visualization. I cannot change the dashboard's appearance or colors, but I can help you plot trends or find specific values."

        # Handle Hallucination Scope Error
        if parsed.get("intent") == "scope_error_hallucination":
             return None, f"I detected a term that is not mapped directly in the **{active_program_config['label']}** program.\n\nTry asking for a specific indicator, or say `list indicators` to see what I can analyze."
              
        # List KPIs handler at top (Consolidated)
        if parsed.get("intent") == "list_kpis":
             program_label = active_program_config["label"]
             # Check if asking for count only
             if "how many" in query_lower or "count" in query_lower:
                 msg = f"I currently have **{len(KPI_MAPPING)}** {program_label.lower()} indicators available.\n\n"
                 msg += "Would you like me to list all of them? Just say 'yes' or 'list all indicators'."
                 # Store context for follow-up
                 st.session_state["chatbot_context"]["last_question"] = "indicators"
                 return None, msg
             else:
                 # List all indicators
                 kpi_list = "\n".join([f"- **{k}**" for k in KPI_MAPPING.keys()])
                 msg = f"Here are all the {program_label.lower()} indicators I can provide information about:\n\n{kpi_list}"
                 return None, msg
        
        # Handle Clear Chat
        if parsed.get("intent") == "clear":
             st.session_state.messages = []
             st.session_state.chatbot_context = {}
             st.session_state["chatbot_program"] = None
             st.session_state.messages.append({
                "role": "assistant",
                "content": get_program_selection_message(role)
              })
             # Reset period filter to default
             if "filters" in st.session_state:
                 st.session_state.filters["period_label"] = "Monthly"
             return None, "Chat history cleared."

        if parsed.get("intent") == "metadata_query":
             entity_type = parsed.get("entity_type")
             count_requested = parsed.get("count_requested")
             facility_names = parsed.get("facility_names", [])
             region_filter = parsed.get("region_filter")
             fulfillment_requested = parsed.get("fulfillment_requested", False)
             
             # Resolve chatbot context (guaranteed by app.py)
             context = st.session_state.get("chatbot_context", {})

             # 1. Handle Regions
             if entity_type == "region":
                 regions_data = get_facilities_grouped_by_region(self.user)
                 region_names = sorted(list(regions_data.keys()))
                 
                 if count_requested and not fulfillment_requested:
                     msg = f"There are **{len(region_names)}** regions available.\n\n"
                     msg += "Would you like me to list them? Just say 'yes' or 'list regions'."
                     context["last_question"] = "regions"
                     return None, msg
                 else:
                     return None, f"The available regions are:\n- " + "\n- ".join(region_names)
                     
             # 2. Handle Facilities
             elif entity_type == "facility":
                 regions_data = get_facilities_grouped_by_region(self.user)
                 target_region = None
                 
                 # Resolve region focus if mentioned
                 if region_filter:
                      matches = difflib.get_close_matches(region_filter, regions_data.keys(), n=1, cutoff=0.6)
                      if matches: target_region = matches[0]
                 
                 if not target_region and facility_names:
                     for potential in facility_names:
                         matches = difflib.get_close_matches(potential, regions_data.keys(), n=1, cutoff=0.6)
                         if matches:
                             target_region = matches[0]
                             break
                 
                 if target_region:
                     # Specific region list
                     facilities = regions_data.get(target_region, [])
                     fac_names = sorted([f[0] for f in facilities])
                     
                     if count_requested and not fulfillment_requested:
                         msg = f"There are **{len(fac_names)}** facilities in **{target_region}**.\n\n"
                         msg += "Would you like me to list them? Just say 'yes' or 'list facilities'."
                         context["last_question"] = f"facilities in {target_region}"
                         return None, msg
                     else:
                         msg = f"Here are the facilities in **{target_region}**:\n- " + "\n- ".join(fac_names)
                         if len(fac_names) > 50:
                              msg = f"Here are the first 50 facilities in **{target_region}**:\n- " + "\n- ".join(fac_names[:50]) + "\n...(and more)"
                         return None, msg
                 else:
                     # Global facilities
                     all_facilities = get_all_facilities_flat(self.user)
                     if (count_requested or not fulfillment_requested) and not fulfillment_requested:
                         msg = f"There are **{len(all_facilities)}** facilities available in total.\n\n"
                         msg += "To see a more specific list, you can ask:\n"
                         msg += "- 'list facilities for **Tigray** region'\n"
                         msg += "- 'show facilities in **Amhara**'\n\n"
                         msg += "Would you like me to list all of them anyway? (This might be a long list!)"
                         context["last_question"] = "facilities"
                         return None, msg
                     else:
                         # Full List (Confirmated)
                         all_fac_names = sorted([f[0] for f in all_facilities])
                         msg = f"Here are all **{len(all_fac_names)}** facilities:\n- " + "\n- ".join(all_fac_names[:50])
                         if len(all_fac_names) > 50:
                             msg += "\n...(list truncated for length)"
                         return None, msg

             return None, "I'm not sure which entity (region or facility) you are asking about."
        
        # Update Context even for Metadata queries (so "Total" can be answered next)
        if parsed.get("intent") == "metadata_query":
             st.session_state["chatbot_context"]["entity_type"] = entity_type
             return None, "Please specify whether you want a region or facility list so I can proceed."
        
        if not parsed["kpi"]:
            # Enforce newborn strict fallback
            if selected_program == "newborn":
                from components.chatbot_newborn import validate_newborn_indicator
                _, msg = validate_newborn_indicator(None, raw_query=query)
                return None, msg

            cross_program = self.detect_cross_program(query, selected_program)
            if cross_program:
                cross_label = self.get_program_config(cross_program)["label"]
                msg = f"That looks like a **{cross_label}** indicator. Type `{cross_program}` to switch programs, then ask again."
            else:
                msg = f"I could not match that request to a **{active_program_config['label']}** indicator. Say `list indicators` to see the available {active_program_config['label'].lower()} indicators."
            return None, msg

        # --- SPECIALIZATION CHECK: Use the explicitly selected program ---
        use_newborn_data = selected_program == "newborn"
        kpi_name = parsed["kpi"]

        active_df = self.newborn_df if use_newborn_data else self.maternal_df
        
        # Update session state with correct collection
        ctx = st.session_state.get("chatbot_context") or {}
        if not isinstance(ctx, dict):
            ctx = {}
        ctx.update(
            {
                "kpi": parsed.get("kpi"),
                "facility_uids": parsed.get("facility_uids") or [],
                "facility_names": parsed.get("facility_names") or [],
                "date_range": parsed.get("date_range"),
                "entity_type": parsed.get("entity_type"),  # Persist for follow-up
                "region_filter": parsed.get("region_filter"),
                "intent": parsed.get("intent"),
                "chart_type": parsed.get("chart_type"),
                "orientation": parsed.get("orientation"),
                "period_label": parsed.get("period_label"),
                "comparison_mode": parsed.get("comparison_mode"),
                "comparison_entity": parsed.get("comparison_entity"),
                "comparison_targets": parsed.get("comparison_targets") or [],
                "source": "newborn" if use_newborn_data else "maternal",
                # Successful execution clears pending conversational questions by default.
                "last_question": None,
            }
        )
        # Drop stale disambiguation state (if present).
        for key in (
            "ambiguity_options",
            "pending_query",
            "pending_parse",
            "pending_kpi",
            "pending_llm_used",
        ):
            if key in ctx:
                ctx[key] = None
        st.session_state["chatbot_context"] = ctx
        
        # Apply Aggregation Filter
        if parsed.get("period_label"):
            if "filters" not in st.session_state: st.session_state.filters = {}
            st.session_state.filters["period_label"] = parsed["period_label"]

        # Prepare Filters
        kpi_name = parsed["kpi"]
        facility_uids = parsed["facility_uids"]
        date_range = parsed["date_range"]
        # If the user did not specify a date range, optionally default to the active dashboard filters (if any).
        # Default behavior is "all time" unless CHATBOT_DEFAULT_DATE_RANGE="dashboard".
        default_date_range = str(getattr(settings, "CHATBOT_DEFAULT_DATE_RANGE", "all_time") or "all_time").strip().lower()
        if not date_range and default_date_range == "dashboard":
            explicit_all_time = any(
                token in query_lower
                for token in [
                    "all time",
                    "since beginning",
                    "from start",
                    "entire period",
                    "overall",
                ]
            )
            if not explicit_all_time:
                filters_ctx = st.session_state.get("filters") or {}
                start_ctx = filters_ctx.get("start_date")
                end_ctx = filters_ctx.get("end_date")
                if start_ctx and end_ctx:
                    date_range = {"start_date": start_ctx, "end_date": end_ctx}

        chart_type = parsed.get("chart_type", "line") # Safe get
        
        # Enforce Bar Chart constraint for count indicators if not specified otherwise
        if kpi_name in {"Admitted Mothers", "Admitted Newborns"} and chart_type == "line":
             chart_type = "bar"
        
        # PROXY KPI MAPPING: Handle components (Numerator/Denominator)
        # Some requests like "Total Deliveries" or "Maternal Deaths" are not KPIs but components.
        # We Map them to a 'host' KPI and extract the component.
        KPI_COMPONENT_MAPPING = {
            "Total Deliveries": {"host_kpi": "C-Section Rate (%)", "component": "denominator"},
            "Maternal Deaths": {"host_kpi": "Institutional Maternal Death Rate (%)", "component": "numerator"},
        }
        
        target_component = "value" # default
        active_kpi_name = kpi_name
        
        if kpi_name in KPI_COMPONENT_MAPPING:
             active_kpi_name = KPI_COMPONENT_MAPPING[kpi_name]["host_kpi"]
             target_component = KPI_COMPONENT_MAPPING[kpi_name]["component"]
        
        # DEBUG: Log parsed entities
        logging.info(f"DEBUG: Chatbot Intent: {parsed['intent']}")
        logging.info(f"DEBUG: Parsed Facility Names: {parsed['facility_names']}")
        logging.info(f"DEBUG: Parsed Facility UIDs: {parsed['facility_uids']}")
        logging.info(f"DEBUG: Active KPI: {active_kpi_name}")

        # Proactive Suggestion
        suggestion = ""
        import random
        tips = [
            f"You can also ask: 'Show me {kpi_name} for last year'",
            f"Try asking: 'Plot {kpi_name} as a bar chart'",
            "Tip: You can say 'in table format' to see the raw numbers.",
            "Did you know? You can filter by specific facilities."
        ]
        if random.random() < 0.3:
            suggestion = f"\n\n💡 *{random.choice(tips)}*"
        
        # Initialize navigation feedback
        nav_feedback = ""

        # --- CONTEXT DESCRIPTION BUILDER ---
        context_desc = ""
        if parsed.get("comparison_mode"):
            if parsed.get("comparison_entity") == "facility":
                 if any(x in query_lower for x in ["all facilities", "all facility", "every facility", "all hospitals"]):
                     context_desc = " for **All Facilities**"
                 elif parsed.get("facility_names"):
                     c_names = list(dict.fromkeys(parsed["facility_names"]))
                     if len(c_names) > 3:
                         context_desc = f" for **{', '.join(c_names[:3])}** and {len(c_names)-3} others"
                     else:
                         context_desc = f" for **{', '.join(c_names)}**"
            elif parsed.get("comparison_entity") == "region":
                 if any(x in query_lower for x in ["all regions", "all region", "every region"]):
                     context_desc = " for **All Regions**"
                 elif parsed.get("comparison_targets"):
                     c_names = parsed["comparison_targets"]
                     if len(c_names) > 3:
                         context_desc = f" for regions **{', '.join(c_names[:3])}**..."
                     else:
                         context_desc = f" for regions **{', '.join(c_names)}**"
        else:
            if parsed.get("facility_names"):
                 c_names = list(dict.fromkeys(parsed["facility_names"]))
                 if len(c_names) > 3:
                     context_desc = f" for **{', '.join(c_names[:3])}**..."
                 else:
                     context_desc = f" for **{', '.join(c_names)}**"
            elif parsed.get("comparison_targets"): 
                 c_names = parsed["comparison_targets"]
                 context_desc = f" for **{c_names[0]}**" 
            else:
                 if self.user.get("role") in ["national", "regional", "admin"]:
                     context_desc = " for **All Facilities**"

        # Handle Distribution Intent
        if parsed["intent"] == "distribution":
            kpi_name = parsed["kpi"]
            facility_uids = parsed["facility_uids"]
            date_range = parsed["date_range"]
            
            # Prepare data
            if use_newborn_data:
                # Newborn distribution not handled specifically yet, fallback to plot or text
                return None, "Distribution views for Newborn indicators are coming soon! Try asking for a trend plot instead."
            
            prepared_df, _ = self._silent_prepare_data(active_df, kpi_name, facility_uids, date_range)
            
            if prepared_df is None or prepared_df.empty:
                return None, f"No data found to show the distribution for **{kpi_name}**."

            fig = None
            title = f"Distribution of {kpi_name}"
            
            try:
                if kpi_name == "Postpartum Hemorrhage (PPH) Rate (%)":
                    from utils.kpi_pph import compute_obstetric_condition_distribution
                    dist = compute_obstetric_condition_distribution(prepared_df, facility_uids)
                    data = [{"Category": k, "Count": v} for k, v in dist.items() if k != "Total Complications" and v > 0]
                    title = "Distribution of Obstetric Conditions (PPH)"
                elif kpi_name == "Postpartum Complications Rate (%)":
                    from utils.kpi_postpartum_compl import compute_postpartum_distribution
                    dist = compute_postpartum_distribution(prepared_df, facility_uids)
                    data = [{"Category": k, "Count": v} for k, v in dist.items() if k != "Total Complication Instances" and v > 0]
                    title = "Distribution of Postpartum Complication Types"
                elif kpi_name == "Antepartum Complications Rate (%)":
                    from utils.kpi_antipartum_compl import compute_complication_distribution
                    dist = compute_complication_distribution(prepared_df, facility_uids)
                    data = [{"Category": k, "Count": v} for k, v in dist.items() if k != "Total Complication Instances" and v > 0]
                    title = "Distribution of Antepartum Complication Types"
                elif kpi_name == "Immediate Postpartum Contraceptive Acceptance Rate (IPPCAR %)":
                    from utils.kpi_utils import compute_fp_distribution
                    dist = compute_fp_distribution(prepared_df, facility_uids)
                    data = [{"Category": k, "Count": v} for k, v in dist.items() if v > 0]
                    title = "Distribution of Family Planning Methods"
                else:
                    return None, f"I don't have a categorical distribution view for **{kpi_name}** yet. I can show you the trend plot instead!"

                if not data:
                    return None, f"I searched the data for **{kpi_name}**, but no categorical breakdowns were found for this period."

                df_pie = pd.DataFrame(data)
                fig = px.pie(
                    df_pie, values="Count", names="Category", 
                    title=title, height=450,
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
                
                return fig, f"Here is the categorical breakdown for **{kpi_name}**{context_desc}."
            except Exception as e:
                logging.error(f"Distribution rendering failed: {e}")
                return None, f"Sorry, I encountered an error while generating the distribution chart: {str(e)}"

        # If Intent is Plot OR Analysis is requested (to get trend data)
        if parsed["intent"] == "plot" or parsed.get("analysis_type"):
            # CRITICAL FIX: Check if facilities exist BEFORE attempting to plot
            if parsed["facility_names"] and not parsed["facility_uids"]:
                return None, "⚠️ **Facility not found!**\n\nI couldn't find that facility in the system. Please check:\n1. **Spelling** of the facility name\n2. Try saying **'list facilities'** to see all available facilities\n3. Use the full facility name (e.g., 'Adigrat Hospital' not just 'Adigrat')"
            
            # 2. Get Data for Active KPI (using SILENT PREPARE)
            if use_newborn_data:
                from newborns_dashboard.kpi_utils_newborn import prepare_data_for_newborn_trend_chart
                prepared_df, date_col = prepare_data_for_newborn_trend_chart(active_df, active_kpi_name, facility_uids, date_range)
            else:
                prepared_df, date_col = self._silent_prepare_data(active_df, active_kpi_name, facility_uids, date_range)
            
            # Check if data exists - if not, we can't plot much (unless it's comparison where we might look broader?)
            # But prepare_data filters by facility_uids. If comparison mode is "facility comparison", 
            # facility_uids should contain the targets.
            
            # If no data found for SPECIFIC facilities, normally we warn. 
            # But let's check comparison groups later.
            if (prepared_df is None or prepared_df.empty) and not parsed.get("comparison_mode"):
                 # Basic check failed
                 pass # Will handle below with "No data" message logic
                 
            # Initialize Responses
            fig = None
            
            # If initial fetch failed, it might be due to filters.
            # If comparison mode, we ignore this initial check and let the loop handle it
            if not parsed.get("comparison_mode"):
                if prepared_df is None or prepared_df.empty:
                    # SPECIAL CASE: Coverage-rate KPIs have an external denominator source,
                    # so we still need to show periods with 0 numerator when denominator exists.
                    if not (
                        (use_newborn_data and active_kpi_name == "Newborn Coverage Rate")
                        or ((not use_newborn_data) and active_kpi_name == "Maternal Coverage Rate")
                    ):
                        fac_names = parsed.get("facility_names") or []
                        if fac_names:
                            shown = ", ".join(fac_names[:3])
                            scope_part = f" for **{shown}**" + ("..." if len(fac_names) > 3 else "")
                        else:
                            scope_part = " across **all accessible facilities**"

                        if isinstance(date_range, dict) and date_range.get("start_date") and date_range.get("end_date"):
                            date_part = f" from **{date_range['start_date']}** to **{date_range['end_date']}**"
                            if not re.search(r"\b(19|20)\d{2}\b", query_lower):
                                date_part += f" (assumed year {str(date_range['start_date'])[:4]})"
                        else:
                            date_part = " for **all time**"

                        return None, (
                            f"I found no data for **{kpi_name}**{scope_part}{date_part}.\n\n"
                            "Try widening the date range (e.g. `all time`), checking the facility spelling, or trying a different indicator."
                        )

            # Generate Plot
            # Group by period
            if prepared_df is not None and "period_display" not in prepared_df.columns:
                # Logic from render_trend_chart_section
                if "event_date" in prepared_df.columns:
                    prepared_df["period_display"] = prepared_df["event_date"].dt.strftime("%b-%y").str.capitalize()
                    prepared_df["period"] = prepared_df["period_display"]
                    # Add Sort Column (Default Monthly logic for chatbot basics)
                    prepared_df["period_sort"] = prepared_df["event_date"].dt.to_period("M").dt.start_time

            # --- COMPARISON MODE LOGIC ---
            comparison_mode = parsed.get("comparison_mode")
            comparison_entity = parsed.get("comparison_entity")
            all_comparison_uids = [] # To store all UIDs for specialized scripts
            
            comparison_groups = [] # List of (EntityName, FilteredDF)
            is_compare_all = False  # Flag to track if comparing all entities
            
            if comparison_mode:
                # Check if "all" comparison was requested in the query
                query_lower_check = query.lower()
                if any(x in query_lower_check for x in ["all facilities", "all facility", "every facility", "all hospitals"]) and not parsed.get("facility_uids") and not parsed.get("comparison_targets"):
                    is_compare_all = True
                    comparison_entity = "facility"
                elif any(x in query_lower_check for x in ["all regions", "all region", "every region", "all reioings", "all reigons", "all reginos"]) and not parsed.get("comparison_targets"):
                    is_compare_all = True
                    comparison_entity = "region"
                
                # --- CLARIFICATION CHECK ---
                if not parsed.get("facility_names") and not parsed.get("comparison_targets") and not is_compare_all:
                     return None, "Which **facilities** or **regions** would you like to compare? Please specify at least two, or say 'all facilities' or 'all regions'."

                if comparison_entity == "region":
                    regions_data = get_facilities_grouped_by_region(self.user)
                    comp_targets = parsed.get("comparison_targets", [])
                    
                    # If "all regions", populate all
                    if is_compare_all:
                        comp_targets = list(regions_data.keys())
                    
                    for r_name, facilities in regions_data.items():
                         if comp_targets and r_name not in comp_targets:
                             continue
                             
                         r_uids = [f[1] for f in facilities]
                         comparison_groups.append((r_name, r_uids))
                         # For region comparison, targets are the region names
                         all_comparison_uids.extend(r_uids) 
                         
                elif comparison_entity == "facility":
                    # If "all facilities", populate all
                    if is_compare_all:
                        all_facilities_flat = get_all_facilities_flat(self.user)
                        for fac_name, fac_uid in all_facilities_flat:
                            comparison_groups.append((fac_name, [fac_uid]))
                            all_comparison_uids.append(fac_uid)
                    else:
                        for name, uid in zip(parsed["facility_names"], parsed["facility_uids"]):
                            comparison_groups.append((name, [uid]))
                            all_comparison_uids.append(uid)
                
                # Force table-only rendering for "all" comparisons OR drill-down OR massive comparisons
                # USER RULE UPDATE: If comparison > 5 facilities, FORCE TABLE (ignore "plot")
                force_table_threshold = 5
                
                # Check for region plotting specifically
                allow_region_plot = comparison_entity == "region" and len(comparison_groups) < 15 # Allow up to 15 regions (usually ~11)
                
                if comparison_entity == "facility" and len(comparison_groups) > force_table_threshold:
                        chart_type = "table"
                        suggestion = f"\n\n💡 *Comparison involves **{len(comparison_groups)}** facilities. Showing table for clarity. Ask for a specific chart type (e.g. 'bar chart') if you really want a plot.*"
                
                # Fallback for "all" or "drill down" if threshold not met but logic applies?
                elif (is_compare_all or parsed.get("is_drill_down")) and not allow_region_plot:
                     # Check if specific chart type mentioned
                    explicit_chart_type = any(k in query.lower() for k in ["bar", "line", "area", "column"])
                    
                    if not explicit_chart_type: # Force table if just "plot" or "show"
                        chart_type = "table"
                        suggestion = f"\n\n💡 *Showing comparison table. Ask for a specific chart type (e.g. 'bar chart') to see a plot.*"
            
            # If NOT comparison mode (or failed setup), default to single group
            if not comparison_groups:
                 comparison_groups.append(("Overall", facility_uids))
                 all_comparison_uids = facility_uids
            
            # If NOT comparison mode (or failed setup), default to single group
            if not comparison_groups:
                 comparison_groups.append(("Overall", facility_uids))
            
            # --- NAVIGATION CONTROL: Update Dashboard State ---
            # Side-effect: Update the main dashboard filters based on this query
            # Only do this if specific entities were found
            try:
                # 1. Facility Mode
                if comparison_entity == "facility" and parsed.get("facility_names"):
                    # Only switch if allowed (National/Regional/Admin)
                    if self.user.get("role") in ["national", "regional", "admin"]:
                        st.session_state["filter_mode"] = "By Facility"
                        st.session_state["selected_facilities"] = parsed["facility_names"]
                        st.session_state["selected_kpi"] = active_kpi_name # Match dashboard to requested KPI
                        st.session_state["selection_applied"] = True # Trigger refresh
                        st.session_state["refresh_trigger"] = True
                        nav_feedback = f"\n*(Dashboard updated to show: {', '.join(parsed['facility_names'][:3])})*"

                # 2. Region Mode
                elif comparison_entity == "region" and parsed.get("comparison_targets"):
                    if self.user.get("role") in ["national", "admin"]: # Regional user can't switch regions usually
                         st.session_state["filter_mode"] = "By Region"
                         st.session_state["selected_regions"] = parsed["comparison_targets"]
                         st.session_state["selected_kpi"] = active_kpi_name # Match dashboard to requested KPI
                         st.session_state["selection_applied"] = True
                         st.session_state["refresh_trigger"] = True
                         nav_feedback = f"\n*(Dashboard updated to show regions: {', '.join(parsed['comparison_targets'][:3])})*"
                
                # 3. Overall / Reset
                elif not parsed.get("facility_names") and not parsed.get("comparison_targets") and "all facilities" in query_lower:
                     if self.user.get("role") in ["national", "regional", "admin"]:
                         st.session_state["filter_mode"] = "All Facilities"
                         st.session_state["selected_facilities"] = ["All Facilities"]
                         st.session_state["selected_kpi"] = active_kpi_name # Match dashboard to requested KPI
                         st.session_state["selection_applied"] = True
                         st.session_state["refresh_trigger"] = True
                         nav_feedback = "\n*(Dashboard reset to All Facilities)*"
                         
            except Exception as e:
                logging.warning(f"Navigation update failed: {e}")

            # --- BATCH PROCESSING ---
            # Optimize: Fetch data for ALL entities at once, then filter in memory.
            
            # 1. Collect all valid targets
            flat_uids = []
            if is_compare_all:
                flat_uids = all_comparison_uids
            else:
                for _, uids in comparison_groups:
                    if uids: flat_uids.extend(uids)
                flat_uids = list(set(flat_uids))
            
            # 2. Add current filters to context
            # (Already in active_df but we pass flat_uids to filter efficiently)
            
            # 3. Batch Fetch
            if use_newborn_data:
                 from newborns_dashboard.kpi_utils_newborn import prepare_data_for_newborn_trend_chart
                 all_data_df, date_col = prepare_data_for_newborn_trend_chart(active_df, active_kpi_name, flat_uids, date_range)
            else:
                 all_data_df, date_col = self._silent_prepare_data(active_df, active_kpi_name, flat_uids, date_range)

            # 4. Global Period Assignment
            if not all_data_df.empty:
                 # Check if period labels needed
                 from utils.time_filter import assign_period
                 filters_ctx = st.session_state.get("filters") or {}
                 p_label = (
                     parsed.get("period_label")
                     or filters_ctx.get("period_label")
                     or st.session_state.get("period_label", "Monthly")
                 )
                 
                 # Ensure date column exists as datetime
                 if "event_date" not in all_data_df.columns and date_col in all_data_df.columns:
                     all_data_df["event_date"] = pd.to_datetime(all_data_df[date_col], errors='coerce')
                 
                 # Assign period
                 if "event_date" in all_data_df.columns:
                     all_data_df = assign_period(all_data_df, "event_date", p_label)

            # Fast-path: simplified newborn KPIs need patient-level rows (not aggregated num/den values).
            kpi_suffix = self.SPECIALIZED_KPI_MAP.get(active_kpi_name) or self.SPECIALIZED_KPI_MAP.get(kpi_name)
            if use_newborn_data and kpi_suffix == "newborn_simplified" and target_component == "value":
                category_map = {
                    "Birth Weight Rate": "render_birth_weight",
                    "KMC Coverage by Birth Weight": "render_kmc_coverage",
                    # NOTE: The newborn dashboard routes "General CPAP Coverage" to the by-weight visualization.
                    "General CPAP Coverage": "render_cpap_by_weight",
                    "CPAP for RDS": "render_cpap_rds",
                    "CPAP Coverage by Birth Weight": "render_cpap_by_weight",
                }
                func_prefix = category_map.get(active_kpi_name, "render_birth_weight")

                resolved_kpi_name = active_kpi_name
                if resolved_kpi_name == "Birth Weight Distribution":
                    resolved_kpi_name = "Birth Weight Rate"
                chart_title = (
                    (active_program_config.get("kpi_mapping") or {})
                    .get(resolved_kpi_name, {})
                    .get("title", active_kpi_name)
                )

                render_source_df = all_data_df if comparison_mode else prepared_df
                if not isinstance(render_source_df, pd.DataFrame):
                    render_source_df = pd.DataFrame()

                render_df = render_source_df.copy().reset_index(drop=True)

                spec = {
                    "type": "specialized",
                    "suffix": kpi_suffix,
                    "func_prefix": func_prefix,
                    "comparison_mode": bool(comparison_mode),
                    "comparison_entity": comparison_entity,
                    "is_compare_all": is_compare_all,
                    "params": {
                        "active_kpi_name": active_kpi_name,
                        "chart_title": chart_title,
                        "facility_names": parsed.get("facility_names"),
                        "facility_uids": facility_uids,
                        "all_comparison_uids": all_comparison_uids,
                        "comparison_targets": parsed.get("comparison_targets"),
                    },
                    "data": render_df,
                }

                return spec, (
                    f"I've rendered the specialized dashboard visualization for **{kpi_name}**{context_desc}.{nav_feedback}"
                )

            # Loop through Groups and Build Data using MEMORY FILTERING
            chart_data = []

            # SPECIAL HANDLING: Newborn Coverage Rate should include periods even when numerator is zero.
            # The denominator comes from an external aggregated admissions file, so months/years with no
            # patient rows still need to appear as 0% when the denominator exists (>0).
            coverage_rate_handled = False
            if use_newborn_data and active_kpi_name == "Newborn Coverage Rate":
                try:
                    from newborns_dashboard.kpi_newborn_coverage_rate import (
                        load_newborn_coverage_denominator,
                        _sum_denominator_for_regions,
                        _sum_denominator_for_facilities,
                        _resolve_facility_names,
                    )
                except Exception as e:
                    logging.error(
                        f"Failed to load Newborn Coverage Rate denominator utilities: {e}"
                    )
                else:
                    den_long = load_newborn_coverage_denominator()
                    if den_long is None or den_long.empty:
                        return None, (
                            "Newborn Coverage Rate denominator data is missing or empty. "
                            "Please check `utils/aggregated_admission_newborn.xlsx`."
                        )

                    # Only Monthly/Yearly are supported for this KPI (match dashboard behavior)
                    period_label = parsed.get("period_label") or st.session_state.get(
                        "period_label", "Monthly"
                    )
                    if (
                        "filters" in st.session_state
                        and "period_label" in st.session_state.filters
                    ):
                        period_label = st.session_state.filters["period_label"]
                    if period_label not in ["Monthly", "Yearly"]:
                        period_label = "Monthly"

                    start_date = date_range.get("start_date") if date_range else None
                    end_date = date_range.get("end_date") if date_range else None

                    month_periods = None
                    try:
                        if start_date and end_date:
                            start_ts = pd.Timestamp(start_date)
                            end_ts = pd.Timestamp(end_date)
                            month_periods = pd.period_range(
                                start=start_ts, end=end_ts, freq="M"
                            )
                    except Exception:
                        month_periods = None

                    if month_periods is None or len(month_periods) == 0:
                        ym_vals = pd.to_numeric(
                            den_long.get("yearmonth"), errors="coerce"
                        ).dropna()
                        if ym_vals.empty:
                            return None, (
                                "No denominator periods available for Newborn Coverage Rate."
                            )
                        ym_min = int(ym_vals.min())
                        ym_max = int(ym_vals.max())
                        start_ts = pd.Timestamp(
                            year=ym_min // 100, month=ym_min % 100, day=1
                        )
                        end_ts = pd.Timestamp(
                            year=ym_max // 100, month=ym_max % 100, day=1
                        )
                        month_periods = pd.period_range(
                            start=start_ts, end=end_ts, freq="M"
                        )

                    period_defs = []
                    if period_label == "Monthly":
                        for p in month_periods:
                            ym = int(p.year * 100 + p.month)
                            period_defs.append(
                                {
                                    "period_display": pd.Timestamp(p.start_time).strftime(
                                        "%b-%y"
                                    ),
                                    "period_sort": pd.Timestamp(p.start_time),
                                    "yearmonths": [ym],
                                    "year": int(p.year),
                                }
                            )
                    else:  # Yearly
                        year_to_yms = {}
                        for p in month_periods:
                            ym = int(p.year * 100 + p.month)
                            year_to_yms.setdefault(int(p.year), []).append(ym)
                        for year in sorted(year_to_yms.keys()):
                            period_defs.append(
                                {
                                    "period_display": str(year),
                                    "period_sort": pd.Timestamp(year=year, month=1, day=1),
                                    "yearmonths": year_to_yms[year],
                                    "year": year,
                                }
                            )

                    # Prepare numerator base (may be empty; still OK)
                    numerator_base_df = all_data_df.copy()
                    if not numerator_base_df.empty:
                        if (
                            "event_date" not in numerator_base_df.columns
                            and date_col
                            and date_col in numerator_base_df.columns
                        ):
                            numerator_base_df["event_date"] = pd.to_datetime(
                                numerator_base_df[date_col], errors="coerce"
                            )
                        if "event_date" in numerator_base_df.columns:
                            numerator_base_df = numerator_base_df[
                                numerator_base_df["event_date"].notna()
                            ].copy()
                            numerator_base_df["_yearmonth"] = (
                                numerator_base_df["event_date"].dt.year.astype(int) * 100
                                + numerator_base_df["event_date"].dt.month.astype(int)
                            )
                            numerator_base_df["_year"] = (
                                numerator_base_df["event_date"].dt.year.astype(int)
                            )

                    # Determine overall denominator scope for non-comparison mode
                    overall_region_scope = None
                    overall_facility_scope_names = None
                    if not comparison_mode:
                        facility_names_ctx = parsed.get("facility_names") or []
                        region_names_ctx = [
                            n.replace(" (Region)", "").strip()
                            for n in facility_names_ctx
                            if isinstance(n, str) and n.strip().lower().endswith("(region)")
                        ]

                        if not facility_uids:
                            # Interpret empty UIDs as "all accessible" for national/regional/admin
                            try:
                                regions_mapping = get_facilities_grouped_by_region(
                                    self.user
                                )
                                if isinstance(regions_mapping, dict) and regions_mapping:
                                    overall_region_scope = list(regions_mapping.keys())
                            except Exception:
                                overall_region_scope = None
                        elif region_names_ctx:
                            overall_region_scope = region_names_ctx
                        else:
                            # Prefer explicit facility names when they align with UIDs; otherwise resolve.
                            if (
                                facility_names_ctx
                                and facility_uids
                                and len(facility_names_ctx) == len(facility_uids)
                                and all(
                                    "(region)" not in str(n).lower() for n in facility_names_ctx
                                )
                            ):
                                overall_facility_scope_names = list(facility_names_ctx)
                            else:
                                overall_facility_scope_names = _resolve_facility_names(
                                    facility_uids, df=numerator_base_df
                                )

                    for entity_name, entity_uids in comparison_groups:
                        # Filter numerator DF for this entity (empty DF is OK)
                        if numerator_base_df.empty:
                            entity_num_df = pd.DataFrame()
                        else:
                            if entity_uids:
                                if "orgUnit" in numerator_base_df.columns:
                                    entity_num_df = numerator_base_df[
                                        numerator_base_df["orgUnit"].isin(entity_uids)
                                    ].copy()
                                else:
                                    entity_num_df = pd.DataFrame()
                            else:
                                entity_num_df = numerator_base_df

                        for pdef in period_defs:
                            # Numerator
                            if entity_num_df.empty:
                                numerator = 0
                            elif period_label == "Monthly":
                                period_df = entity_num_df[
                                    entity_num_df["_yearmonth"] == pdef["yearmonths"][0]
                                ]
                                numerator = (
                                    int(period_df["tei_id"].dropna().nunique())
                                    if "tei_id" in period_df.columns
                                    else int(len(period_df))
                                )
                            else:  # Yearly
                                y_df = entity_num_df[entity_num_df["_year"] == pdef["year"]]
                                numerator = (
                                    int(y_df["tei_id"].dropna().nunique())
                                    if "tei_id" in y_df.columns
                                    else int(len(y_df))
                                )

                            # Denominator
                            if comparison_mode and comparison_entity == "region":
                                denominator = _sum_denominator_for_regions(
                                    den_long, [entity_name], yearmonths=pdef["yearmonths"]
                                )
                            elif comparison_mode and comparison_entity == "facility":
                                denominator = _sum_denominator_for_facilities(
                                    den_long, [entity_name], yearmonths=pdef["yearmonths"]
                                )
                            else:
                                if overall_region_scope:
                                    denominator = _sum_denominator_for_regions(
                                        den_long,
                                        overall_region_scope,
                                        yearmonths=pdef["yearmonths"],
                                    )
                                else:
                                    denominator = _sum_denominator_for_facilities(
                                        den_long,
                                        overall_facility_scope_names or [],
                                        yearmonths=pdef["yearmonths"],
                                    )

                            value = (
                                (numerator / denominator * 100) if denominator > 0 else 0.0
                            )
                            if np.isnan(value) or np.isinf(value):
                                value = 0.0

                            # Resolve Component
                            plot_value = value
                            if target_component == "numerator":
                                plot_value = numerator
                            elif target_component == "denominator":
                                plot_value = denominator

                            chart_data.append(
                                {
                                    "Period": pdef["period_display"],
                                    "Entity": entity_name,
                                    "Value": plot_value,
                                    "Numerator": int(numerator),
                                    "Denominator": int(denominator),
                                    "SortDate": pdef["period_sort"],
                                    "orgUnit": entity_uids[0] if entity_uids else None,
                                }
                            )

                    coverage_rate_handled = True
             
            # SPECIAL HANDLING: Maternal Coverage Rate should include periods even when numerator is zero.
            # The denominator comes from an external aggregated admissions file, so months/years with no
            # patient rows still need to appear as 0% when the denominator exists (>0).
            if (
                (not coverage_rate_handled)
                and (not use_newborn_data)
                and active_kpi_name == "Maternal Coverage Rate"
            ):
                try:
                    from utils.kpi_maternal_coverage_rate import (
                        load_maternal_coverage_denominator,
                        _sum_denominator_for_regions,
                        _sum_denominator_for_facilities,
                        _resolve_facility_names,
                    )
                except Exception as e:
                    logging.error(
                        f"Failed to load Maternal Coverage Rate denominator utilities: {e}"
                    )
                else:
                    den_long = load_maternal_coverage_denominator()
                    if den_long is None or den_long.empty:
                        return None, (
                            "Maternal Coverage Rate denominator data is missing or empty. "
                            "Please check `utils/aggregated_admission_mothers.xlsx`."
                        )

                    # Only Monthly/Yearly are supported for this KPI (match dashboard behavior)
                    period_label = parsed.get("period_label") or st.session_state.get(
                        "period_label", "Monthly"
                    )
                    if (
                        "filters" in st.session_state
                        and "period_label" in st.session_state.filters
                    ):
                        period_label = st.session_state.filters["period_label"]
                    if period_label not in ["Monthly", "Yearly"]:
                        period_label = "Monthly"

                    start_date = date_range.get("start_date") if date_range else None
                    end_date = date_range.get("end_date") if date_range else None

                    month_periods = None
                    try:
                        if start_date and end_date:
                            start_ts = pd.Timestamp(start_date)
                            end_ts = pd.Timestamp(end_date)
                            month_periods = pd.period_range(
                                start=start_ts, end=end_ts, freq="M"
                            )
                    except Exception:
                        month_periods = None

                    if month_periods is None or len(month_periods) == 0:
                        ym_vals = pd.to_numeric(
                            den_long.get("yearmonth"), errors="coerce"
                        ).dropna()
                        if ym_vals.empty:
                            return None, (
                                "No denominator periods available for Maternal Coverage Rate."
                            )
                        ym_min = int(ym_vals.min())
                        ym_max = int(ym_vals.max())
                        start_ts = pd.Timestamp(
                            year=ym_min // 100, month=ym_min % 100, day=1
                        )
                        end_ts = pd.Timestamp(
                            year=ym_max // 100, month=ym_max % 100, day=1
                        )
                        month_periods = pd.period_range(
                            start=start_ts, end=end_ts, freq="M"
                        )

                    period_defs = []
                    if period_label == "Monthly":
                        for p in month_periods:
                            ym = int(p.year * 100 + p.month)
                            period_defs.append(
                                {
                                    "period_display": pd.Timestamp(p.start_time).strftime(
                                        "%b-%y"
                                    ),
                                    "period_sort": pd.Timestamp(p.start_time),
                                    "yearmonths": [ym],
                                    "year": int(p.year),
                                }
                            )
                    else:  # Yearly
                        year_to_yms = {}
                        for p in month_periods:
                            ym = int(p.year * 100 + p.month)
                            year_to_yms.setdefault(int(p.year), []).append(ym)
                        for year in sorted(year_to_yms.keys()):
                            period_defs.append(
                                {
                                    "period_display": str(year),
                                    "period_sort": pd.Timestamp(year=year, month=1, day=1),
                                    "yearmonths": year_to_yms[year],
                                    "year": year,
                                }
                            )

                    # Prepare numerator base (may be empty; still OK)
                    numerator_base_df = all_data_df.copy()
                    if not numerator_base_df.empty:
                        if (
                            "event_date" not in numerator_base_df.columns
                            and date_col
                            and date_col in numerator_base_df.columns
                        ):
                            numerator_base_df["event_date"] = pd.to_datetime(
                                numerator_base_df[date_col], errors="coerce"
                            )
                        if "event_date" in numerator_base_df.columns:
                            numerator_base_df = numerator_base_df[
                                numerator_base_df["event_date"].notna()
                            ].copy()
                            numerator_base_df["_yearmonth"] = (
                                numerator_base_df["event_date"].dt.year.astype(int) * 100
                                + numerator_base_df["event_date"].dt.month.astype(int)
                            )
                            numerator_base_df["_year"] = (
                                numerator_base_df["event_date"].dt.year.astype(int)
                            )

                    # Determine overall denominator scope for non-comparison mode
                    overall_region_scope = None
                    overall_facility_scope_names = None
                    if not comparison_mode:
                        facility_names_ctx = parsed.get("facility_names") or []
                        region_names_ctx = [
                            n.replace(" (Region)", "").strip()
                            for n in facility_names_ctx
                            if isinstance(n, str)
                            and n.strip().lower().endswith("(region)")
                        ]

                        if not facility_uids:
                            # Interpret empty UIDs as "all accessible" for national/regional/admin
                            try:
                                regions_mapping = get_facilities_grouped_by_region(
                                    self.user
                                )
                                if (
                                    isinstance(regions_mapping, dict)
                                    and regions_mapping
                                ):
                                    overall_region_scope = list(regions_mapping.keys())
                            except Exception:
                                overall_region_scope = None
                        elif region_names_ctx:
                            overall_region_scope = region_names_ctx
                        else:
                            # Prefer explicit facility names when they align with UIDs; otherwise resolve.
                            if (
                                facility_names_ctx
                                and facility_uids
                                and len(facility_names_ctx) == len(facility_uids)
                                and all(
                                    "(region)" not in str(n).lower()
                                    for n in facility_names_ctx
                                )
                            ):
                                overall_facility_scope_names = list(facility_names_ctx)
                            else:
                                overall_facility_scope_names = _resolve_facility_names(
                                    facility_uids, df=numerator_base_df
                                )

                    for entity_name, entity_uids in comparison_groups:
                        # Filter numerator DF for this entity (empty DF is OK)
                        if numerator_base_df.empty:
                            entity_num_df = pd.DataFrame()
                        else:
                            if entity_uids:
                                if "orgUnit" in numerator_base_df.columns:
                                    entity_num_df = numerator_base_df[
                                        numerator_base_df["orgUnit"].isin(entity_uids)
                                    ].copy()
                                else:
                                    entity_num_df = pd.DataFrame()
                            else:
                                entity_num_df = numerator_base_df

                        for pdef in period_defs:
                            # Numerator
                            if entity_num_df.empty:
                                numerator = 0
                            elif period_label == "Monthly":
                                period_df = entity_num_df[
                                    entity_num_df["_yearmonth"] == pdef["yearmonths"][0]
                                ]
                                numerator = (
                                    int(period_df["tei_id"].dropna().nunique())
                                    if "tei_id" in period_df.columns
                                    else int(len(period_df))
                                )
                            else:  # Yearly
                                y_df = entity_num_df[entity_num_df["_year"] == pdef["year"]]
                                numerator = (
                                    int(y_df["tei_id"].dropna().nunique())
                                    if "tei_id" in y_df.columns
                                    else int(len(y_df))
                                )

                            # Denominator
                            if comparison_mode and comparison_entity == "region":
                                denominator = _sum_denominator_for_regions(
                                    den_long, [entity_name], yearmonths=pdef["yearmonths"]
                                )
                            elif comparison_mode and comparison_entity == "facility":
                                denominator = _sum_denominator_for_facilities(
                                    den_long, [entity_name], yearmonths=pdef["yearmonths"]
                                )
                            else:
                                if overall_region_scope:
                                    denominator = _sum_denominator_for_regions(
                                        den_long,
                                        overall_region_scope,
                                        yearmonths=pdef["yearmonths"],
                                    )
                                else:
                                    denominator = _sum_denominator_for_facilities(
                                        den_long,
                                        overall_facility_scope_names or [],
                                        yearmonths=pdef["yearmonths"],
                                    )

                            value = (
                                (numerator / denominator * 100) if denominator > 0 else 0.0
                            )
                            if np.isnan(value) or np.isinf(value):
                                value = 0.0

                            # Resolve Component
                            plot_value = value
                            if target_component == "numerator":
                                plot_value = numerator
                            elif target_component == "denominator":
                                plot_value = denominator

                            chart_data.append(
                                {
                                    "Period": pdef["period_display"],
                                    "Entity": entity_name,
                                    "Value": plot_value,
                                    "Numerator": int(numerator),
                                    "Denominator": int(denominator),
                                    "SortDate": pdef["period_sort"],
                                    "orgUnit": entity_uids[0] if entity_uids else None,
                                }
                            )

                    coverage_rate_handled = True

            if not coverage_rate_handled:
                for entity_name, entity_uids in comparison_groups:
                    # Filter this group's data from the batch
                    if all_data_df.empty:
                        entity_df = pd.DataFrame()
                    else:
                        if "orgUnit" in all_data_df.columns and entity_uids:
                            entity_df = all_data_df[
                                all_data_df["orgUnit"].isin(entity_uids)
                            ].copy()
                        else:
                            # Fallback for national/region if specific UID column logic differs
                            # For now assuming orgUnit is key
                            if not entity_uids:  # Overall
                                entity_df = all_data_df.copy()
                            else:
                                entity_df = pd.DataFrame()

                    if entity_df.empty:
                        continue

                    # Determine Period Order (Already done globally but ensuring display col)
                    if "period_display" not in entity_df.columns:
                        if "event_date" in entity_df.columns:
                            entity_df["period_display"] = (
                                entity_df["event_date"]
                                .dt.strftime("%b-%y")
                                .str.capitalize()
                            )
                            entity_df["period_sort"] = (
                                entity_df["event_date"].dt.to_period("M").dt.start_time
                            )

                    grouped = entity_df.groupby("period_display")

                    # Sort groups by period_sort
                    time_groups = []
                    for name, group in grouped:
                        # Get sort value from first row
                        sort_val = (
                            group["period_sort"].iloc[0]
                            if "period_sort" in group.columns
                            else (
                                group["event_date"].min()
                                if "event_date" in group.columns
                                else datetime.min
                            )
                        )
                        time_groups.append((name, group, sort_val))

                    # Sort by the extracted sort_val
                    time_groups.sort(key=lambda x: x[2])

                    for period_name, group_df, sort_val in time_groups:
                        # Generic KPI calculation fallback
                        kpi_suffix = self.SPECIALIZED_KPI_MAP.get(active_kpi_name)

                        if use_newborn_data:
                            from newborns_dashboard.dash_co_newborn import (
                                get_numerator_denominator_for_newborn_kpi_with_all,
                            )

                            numerator, denominator, value = (
                                get_numerator_denominator_for_newborn_kpi_with_all(
                                    group_df, active_kpi_name, entity_uids, date_range
                                )
                            )
                        elif kpi_suffix and kpi_suffix != "utils":
                            try:
                                module = __import__(
                                    f"utils.kpi_{kpi_suffix}",
                                    fromlist=[f"get_numerator_denominator_for_{kpi_suffix}"],
                                )
                                get_nd_func = getattr(
                                    module, f"get_numerator_denominator_for_{kpi_suffix}"
                                )
                                numerator, denominator, value = get_nd_func(
                                    group_df, entity_uids, date_range
                                )
                            except Exception as e:
                                logging.error(
                                    f"Failed to call specialized function for {active_kpi_name}: {e}"
                                )
                                numerator, denominator, value = (
                                    kpi_utils.get_numerator_denominator_for_kpi(
                                        group_df, active_kpi_name, entity_uids, date_range
                                    )
                                )
                        else:
                            # Use standard kpi_utils for "utils" suffix or no suffix
                            numerator, denominator, value = (
                                kpi_utils.get_numerator_denominator_for_kpi(
                                    group_df, active_kpi_name, entity_uids, date_range
                                )
                            )

                        # Resolve Component
                        plot_value = value
                        if target_component == "numerator":
                            plot_value = numerator
                        elif target_component == "denominator":
                            plot_value = denominator

                        chart_data.append(
                            {
                                "Period": period_name,
                                "Entity": entity_name,
                                "Value": plot_value,
                                "Numerator": numerator,
                                "Denominator": denominator,
                                "SortDate": sort_val,
                                "orgUnit": entity_uids[0] if entity_uids else None,
                            }
                        )
            
            plot_df = pd.DataFrame(chart_data)
            if not plot_df.empty:
                plot_df = plot_df.reset_index(drop=True) # Ensure unique index
                if "SortDate" in plot_df.columns:
                    plot_df.sort_values("SortDate", inplace=True)
                    plot_df = plot_df.reset_index(drop=True) # Final clean index
            
            if plot_df.empty:
                return None, "Data processing resulted in empty dataset."
                
            # Prepare for rendering
            render_plot_df = plot_df.copy()

            # --- SPECIALIZED RENDERING (Move BEFORE Generic Logic) ---
            # Check for specialized KPI using both the potentially mapped 'active' name and original name
            kpi_suffix = self.SPECIALIZED_KPI_MAP.get(active_kpi_name) or self.SPECIALIZED_KPI_MAP.get(kpi_name)
            
            # Skip specialized rendering if we are in "table" mode for broad comparison (Force generic table)
            if kpi_suffix:
                try:
                    # Determine module and function prefix
                    if kpi_suffix == "utils":
                        func_prefix = "render"
                    elif kpi_suffix == "newborn":
                        func_prefix = "render_newborn"
                        if active_kpi_name == "Admitted Newborns":
                            func_prefix = "render_admitted_newborns"
                    elif kpi_suffix == "newborn_simplified":
                        # Map category for func prefix
                        category_map = {
                            "Birth Weight Rate": "render_birth_weight",
                            "KMC Coverage by Birth Weight": "render_kmc_coverage",
                            # Keep parity with newborn dashboard behavior (route to by-weight chart).
                            "General CPAP Coverage": "render_cpap_by_weight",
                            "CPAP for RDS": "render_cpap_rds",
                            "CPAP Coverage by Birth Weight": "render_cpap_by_weight",
                        }
                        func_prefix = category_map.get(active_kpi_name, "render_birth_weight")
                    else:
                        func_prefix = f"render_{kpi_suffix}"
                    
                    # Prepare DF for specialized scripts (without Overall row)
                    render_df = render_plot_df.copy().reset_index(drop=True)
                    render_df = render_df.rename(columns={
                        "Period": "period_display",
                        "Value": "value",
                        "Numerator": "numerator",
                        "Denominator": "denominator",
                        "SortDate": "period_sort"
                    })
                    
                    # Add name column based on entity type
                    name_col = "Facility" if comparison_entity == "facility" else "Region"
                    render_df[name_col] = render_df["Entity"]
                    
                    # Final reset_index to ensure absolutely clean index before passing to rendering functions
                    render_df = render_df.reset_index(drop=True)
                    
                    # Get labels for kpi_utils functions
                    kpi_info = KPI_MAPPING.get(active_kpi_name, {})
                    chart_title = kpi_info.get("title") or active_kpi_name
                    num_label = kpi_info.get("numerator_name", "Numerator")
                    den_label = kpi_info.get("denominator_name", "Denominator")
                    
                    # Return specialized rendering specification
                    spec = {
                        "type": "specialized",
                        "suffix": kpi_suffix,
                        "func_prefix": func_prefix,
                        "comparison_mode": comparison_mode,
                        "comparison_entity": comparison_entity,
                        "is_compare_all": is_compare_all,  # Add flag to spec
                        "params": {
                            "active_kpi_name": active_kpi_name,
                            "chart_title": chart_title,
                            "facility_names": parsed.get("facility_names"),
                            "facility_uids": facility_uids,
                            "all_comparison_uids": all_comparison_uids,
                            "comparison_targets": parsed.get("comparison_targets"),
                            "num_label": num_label,
                            "den_label": den_label
                        },
                        "data": render_df
                    }
                    
                    return spec, f"I've rendered the specialized dashboard visualization for **{kpi_name}**{context_desc}.{nav_feedback}"
                except Exception as e:
                    logging.error(f"Specialized rendering spec generation failed for {active_kpi_name}: {e}")
                    # Fallback to generic plot below if not handled
            
             # Handle Analysis Request (Max/Min) - NEW
            analysis_type = parsed.get("analysis_type")
            if analysis_type and not plot_df.empty:
                if analysis_type == "max":
                    # Filter out "Overall" row if present to avoid skewed max
                    trend_df = plot_df[plot_df["Period"] != "Overall"]
                    if trend_df.empty: trend_df = plot_df 
                    
                    best_row = trend_df.loc[trend_df['Value'].idxmax()]
                    val = best_row['Value']
                    period = best_row['Period']
                    # Check if percentage
                    val_str = f"{val:.2f}%" if "Rate" in kpi_name or "%" in kpi_name else f"{int(val):,}"
                    return None, f"The **highest** {kpi_name} was recorded in **{period}** with a value of **{val_str}**."
                elif analysis_type == "min":
                    worst_row = plot_df.loc[plot_df['Value'].idxmin()]
                    val = worst_row['Value']
                    period = worst_row['Period']
                    # Check if percentage
                    val_str = f"{val:.2f}%" if "Rate" in kpi_name or "%" in kpi_name else f"{int(val):,}"
                    return None, f"The **lowest** {kpi_name} was recorded in **{period}** with a value of **{val_str}**."
            
            # --- GENERIC FALLBACK (Only if no specialized rendering was returned) ---
            # ADD OVERALL ROW for generic table/chart (Standard Mode)
            if not plot_df.empty and not parsed.get("comparison_mode"):
                total_numerator = plot_df["Numerator"].sum()
                total_denominator = plot_df["Denominator"].sum()
                total_kpi_value = (total_numerator / total_denominator * 100) if total_denominator > 0 else 0
                if target_component != "value":
                    total_kpi_value = total_numerator if target_component == "numerator" else total_denominator
                
                overall_row = {"Period": "Overall", "Entity": "Overall", "Value": total_kpi_value, "Numerator": total_numerator, "Denominator": total_denominator}
                plot_df = pd.concat([plot_df, pd.DataFrame([overall_row])], ignore_index=True)

            # Determine Plot Params for Generic Charts
            color_col = None
            if "Entity" in plot_df.columns:
                 if parsed.get("comparison_mode") or plot_df["Entity"].nunique() > 1:
                     color_col = "Entity"

            # Dynamic Chart Generation
            if chart_type == "table":
                # Comparison Mode Table: Pivot for better readability
                if parsed.get("comparison_mode"):
                     # Pivot: Index=Period, Cols=Entity, Values=Value
                     try:
                         pivot_df = plot_df.pivot(index='Period', columns='Entity', values='Value')
                         # Optional: Add mean/total row? simpler is better for now.
                         # Reset index to make Period a column again for display
                         pivot_df.reset_index(inplace=True)
                         return pivot_df, f"Here is the comparison table for **{kpi_name}**{context_desc}."
                     except Exception as e:
                         # Fallback if pivot fails (e.g. duplicates)
                         return plot_df[["Period", "Entity", "Value"]], f"Here is the comparison data for **{kpi_name}**."
                         
                # CLEAN UP INTERNAL COLUMNS before returning
                for col in ["SortDate", "orgUnit"]:
                    if col in plot_df.columns:
                        plot_df = plot_df.drop(columns=[col])
                
                # Format Percentage columns if needed
                if "Rate" in kpi_name or "%" in kpi_name:
                    for col in plot_df.columns:
                        if col not in ["Period", "Entity", "Month", "Facility", "Region"]:
                            try:
                                plot_df[col] = plot_df[col].apply(lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else x)
                            except: pass

                return plot_df, f"Here is the data table for **{kpi_name}**{context_desc}."
                
            # Step 10b: Render Plot (Multi-trace / Drill-down aware)
            line_plot_df = plot_df
            if chart_type == "line" or chart_type not in {"bar", "area"}:
                if "Denominator" in line_plot_df.columns:
                    den_vals = pd.to_numeric(
                        line_plot_df["Denominator"], errors="coerce"
                    ).fillna(0)
                    line_plot_df = line_plot_df.copy()
                    line_plot_df["Value"] = pd.to_numeric(
                        line_plot_df["Value"], errors="coerce"
                    )
                    line_plot_df.loc[den_vals <= 0, "Value"] = np.nan

            if chart_type == "line":
                fig = px.line(line_plot_df, x="Period", y="Value", color=color_col, title=f"{kpi_name}", markers=False, line_shape="spline", height=400, custom_data=["Numerator", "Denominator"])
            elif chart_type == "bar":
                if parsed.get("orientation") == "h":
                    fig = px.bar(plot_df, x="Value", y="Period", color=color_col,  title=f"{kpi_name}", height=400, orientation='h', barmode='group', custom_data=["Numerator", "Denominator"])
                else:
                    fig = px.bar(plot_df, x="Period", y="Value", color=color_col, title=f"{kpi_name}", height=400, barmode='group', custom_data=["Numerator", "Denominator"])
            elif chart_type == "area":
                fig = px.area(plot_df, x="Period", y="Value", color=color_col, title=f"{kpi_name}", markers=True, height=400, custom_data=["Numerator", "Denominator"])
            else: # Default to line
                fig = px.line(line_plot_df, x="Period", y="Value", color=color_col, title=f"{kpi_name}", markers=False, line_shape="spline", height=400, custom_data=["Numerator", "Denominator"])
                
            if chart_type == "line":
                fig.update_traces(mode="lines", cliponaxis=False)
                fig.update_xaxes(layer="below traces")
                fig.update_yaxes(layer="below traces")
            
            # Add custom hover template to show numerator and denominator
            # Treat "Missing" indicators as rates if they have a denominator
            is_rate = "Rate" in kpi_name or "%" in kpi_name or "Missing" in kpi_name or "missing" in kpi_name.lower()
            
            kpi_info = KPI_MAPPING.get(kpi_name, {})
            # Formulate title for hover (use mapped title if possible)
            hover_title = kpi_info.get("title", kpi_name)
            
            num_label = kpi_info.get("numerator_name", "Numerator")
            den_label = kpi_config.get("numerator_name", "Numerator") if 'kpi_config' in locals() else num_label # Safer fallback
            den_label = kpi_info.get("denominator_name", "Denominator")
            
            if is_rate:
                hover_template = (
                    f"<b>%{{x}}</b><br>"
                    f"{hover_title}: <b>%{{y:.2f}}%</b><br>"
                    f"{num_label}: %{{customdata[0]:,.0f}}<br>"
                    f"{den_label}: %{{customdata[1]:,.0f}}<extra></extra>"
                )
            else:
                hover_template = (
                    f"<b>%{{x}}</b><br>"
                    f"{hover_title}: <b>%{{y:,.0f}}</b><br>"
                    f"{num_label}: %{{customdata[0]:,.0f}}<br>"
                    f"{den_label}: %{{customdata[1]:,.0f}}<extra></extra>"
                )
            
            fig.update_traces(hovertemplate=hover_template)
            if is_rate:
                if isinstance(plot_df, pd.DataFrame) and "Value" in plot_df.columns:
                    values = pd.to_numeric(plot_df["Value"], errors="coerce")
                else:
                    values = pd.Series(dtype="float64")
                y_lower = -0.5
                y_upper = 100.5
                if not values.empty:
                    vmin = values.min(skipna=True)
                    vmax = values.max(skipna=True)

                    if vmin is not None and not pd.isna(vmin) and float(vmin) < y_lower:
                        pad = max(5.0, abs(float(vmin)) * 0.05)
                        y_lower = float(vmin) - pad
                    if vmax is not None and not pd.isna(vmax) and float(vmax) > y_upper:
                        pad = max(5.0, abs(float(vmax)) * 0.05)
                        y_upper = float(vmax) + pad

                y_dtick = 25
                if y_upper > 200:
                    y_dtick = 50
                if y_upper > 500:
                    y_dtick = 100

                if chart_type == "bar" and parsed.get("orientation") == "h":
                    fig.update_layout(xaxis=dict(range=[y_lower, y_upper], dtick=y_dtick))
                else:
                    fig.update_layout(yaxis=dict(range=[y_lower, y_upper], dtick=y_dtick))
            
            # Refine title
            title_text = f"{kpi_name}"
            if parsed["facility_names"]:
                title_text += f" - {', '.join(parsed['facility_names'][:2])}"
                if len(parsed['facility_names']) > 2: title_text += "..."
            fig.update_layout(title_text=title_text, margin=dict(l=20, r=20, t=40, b=20))
            
            response_text = f"Here is the {chart_type} chart for **{kpi_name}**{context_desc}."

            insight = None
            if settings.CHATBOT_USE_LLM_INSIGHTS:
                self.last_trace["insight"]["attempted"] = True
                try:
                    insight_df = line_plot_df if chart_type == "line" else plot_df
                    insight_df = insight_df.copy()
                    if "Value" in insight_df.columns:
                        insight_df["Value"] = pd.to_numeric(insight_df["Value"], errors="coerce")
                    insight_df = insight_df.dropna(subset=["Value"]) if "Value" in insight_df.columns else insight_df

                    latest_values = []
                    if not insight_df.empty and "Entity" in insight_df.columns and "Value" in insight_df.columns:
                        sort_col = "SortDate" if "SortDate" in insight_df.columns else None
                        if sort_col:
                            insight_df = insight_df.sort_values(sort_col)
                        latest_rows = insight_df.groupby("Entity", as_index=False).tail(1)
                        for _, row in latest_rows.head(5).iterrows():
                            latest_values.append(
                                {
                                    "entity": str(row.get("Entity")),
                                    "period": str(row.get("Period")),
                                    "value": float(row.get("Value")) if pd.notna(row.get("Value")) else None,
                                }
                            )

                    kpi_def = (active_program_config.get("kpi_definitions") or {}).get(kpi_name, {})
                    stats = {
                        "kpi": kpi_name,
                        "chart_type": chart_type,
                        "date_range": date_range,
                        "entity_count": int(insight_df["Entity"].nunique()) if not insight_df.empty and "Entity" in insight_df.columns else 0,
                        "period_count": int(insight_df["Period"].nunique()) if not insight_df.empty and "Period" in insight_df.columns else 0,
                        "latest_values": latest_values,
                        "definition": kpi_def.get("description"),
                        "interpretation": kpi_def.get("interpretation"),
                    }
                    insight = generate_chatbot_insight(stats=stats, program_label=active_program_config.get("label", "Dashboard"))
                    if insight:
                        self.last_trace["insight"]["used"] = True
                    else:
                        self.last_trace["insight"]["failed"] = True
                except Exception:
                    self.last_trace["insight"]["failed"] = True
                    insight = None

            if insight:
                response_text += f"\n\n{insight}"

            result = (fig, response_text + suggestion + nav_feedback)
            self._cache_result(cache_key, result)
            return result
            
        else:
            # Text Response (Single value)
            if use_newborn_data:
                 from newborns_dashboard.kpi_utils_newborn import prepare_data_for_newborn_trend_chart
                 prepared_df, date_col = prepare_data_for_newborn_trend_chart(active_df, active_kpi_name, facility_uids, date_range)
            else:
                 prepared_df, date_col = self._silent_prepare_data(
                     active_df, 
                     active_kpi_name, 
                     facility_uids, 
                     date_range
                 )
            
            if prepared_df is None or prepared_df.empty:
                return None, f"No data found for **{kpi_name}**."
            
            # Check for specialized KPI
            kpi_suffix = self.SPECIALIZED_KPI_MAP.get(active_kpi_name)
            
            if use_newborn_data:
                from newborns_dashboard.dash_co_newborn import get_numerator_denominator_for_newborn_kpi_with_all
                numerator, denominator, value = get_numerator_denominator_for_newborn_kpi_with_all(active_df, active_kpi_name, facility_uids, date_range)
            elif kpi_suffix and kpi_suffix != "utils":
                try:
                    module = __import__(f"utils.kpi_{kpi_suffix}", fromlist=[f"get_numerator_denominator_for_{kpi_suffix}"])
                    get_nd_func = getattr(module, f"get_numerator_denominator_for_{kpi_suffix}")
                    numerator, denominator, value = get_nd_func(prepared_df, facility_uids, date_range)
                except Exception as e:
                    logging.error(f"Failed to call specialized function for {active_kpi_name}: {e}")
                    numerator, denominator, value = kpi_utils.get_numerator_denominator_for_kpi(prepared_df, active_kpi_name, facility_uids)
            else:
                  # Use standard kpi_utils for "utils" suffix or no suffix
                  numerator, denominator, value = kpi_utils.get_numerator_denominator_for_kpi(prepared_df, active_kpi_name, facility_uids)

            if kpi_suffix == "newborn_simplified" and target_component == "value":
                return None, f"**{kpi_name}** is best viewed as a chart or table. Ask `plot {kpi_name}` or `show {kpi_name} in table format`."
             
            # Resolve Component
            display_value = value
            if target_component == "numerator": display_value = numerator
            elif target_component == "denominator": display_value = denominator
             
            # Format display logic
            value_indicator_names = active_program_config.get("count_indicators", set())
            if active_kpi_name in value_indicator_names and target_component == "value":
                response_text = f"The **{active_kpi_name}** count is **{int(display_value):,}**"
            elif target_component == "value":
                response_text = f"The **{kpi_name}** is **{display_value:.2f}%**"
            else:
                response_text = f"The **{kpi_name}** count is **{int(display_value):,}**"

            if parsed.get("facility_names"):
                 names = list(dict.fromkeys(parsed["facility_names"]))
                 if len(names) > 3:
                     response_text += f" for *{', '.join(names[:3])}*..."
                 else:
                     response_text += f" for *{', '.join(names)}*"
            elif self.user.get("role") in ["national", "regional", "admin"] and not parsed.get("comparison_targets"):
                 response_text += " for *All Facilities*"
            if date_range:
                response_text += f" during the period {date_range['start_date']} to {date_range['end_date']}."
            else:
                response_text += "."
            
            if target_component == "value":
                response_text += f"\n\n(Based on {int(numerator)} cases out of {int(denominator)})"
            
            insight = None
            if settings.CHATBOT_USE_LLM_INSIGHTS:
                self.last_trace["insight"]["attempted"] = True
                try:
                    kpi_def = (active_program_config.get("kpi_definitions") or {}).get(kpi_name, {})
                    stats = {
                        "kpi": kpi_name,
                        "value": float(display_value) if isinstance(display_value, (int, float, np.floating)) else str(display_value),
                        "numerator": int(numerator) if numerator is not None else None,
                        "denominator": int(denominator) if denominator is not None else None,
                        "component": target_component,
                        "date_range": date_range,
                        "facilities": list(dict.fromkeys(parsed.get("facility_names") or [])),
                        "definition": kpi_def.get("description"),
                        "interpretation": kpi_def.get("interpretation"),
                    }
                    insight = generate_chatbot_insight(stats=stats, program_label=active_program_config.get("label", "Dashboard"))
                    if insight:
                        self.last_trace["insight"]["used"] = True
                    else:
                        self.last_trace["insight"]["failed"] = True
                except Exception:
                    self.last_trace["insight"]["failed"] = True
                    insight = None

            if insight:
                response_text += f"\n\n{insight}"

            # Cache result
            result = (None, response_text + suggestion + nav_feedback)
            self._cache_result(cache_key, result)
            return result


def _format_llm_trace_caption(trace):
    if not isinstance(trace, dict):
        return None

    parser = trace.get("parser")
    insight = trace.get("insight")
    if not isinstance(parser, dict):
        parser = {}
    if not isinstance(insight, dict):
        insight = {}

    parts = []

    enabled = bool(parser.get("enabled"))
    attempted = bool(parser.get("attempted"))
    used = bool(parser.get("used"))
    failed = bool(parser.get("failed"))
    provider = parser.get("provider")
    model = parser.get("model")

    provider_label = None
    if provider == "gemini":
        provider_label = "Gemini"
    elif provider == "openai":
        provider_label = "OpenAI"
    elif isinstance(provider, str) and provider.strip():
        provider_label = provider.strip()

    model_label = str(model).strip() if isinstance(model, str) and model.strip() else None
    llm_label = None
    if provider_label and model_label:
        llm_label = f"{provider_label} / {model_label}"
    else:
        llm_label = provider_label or model_label

    if not enabled:
        parts.append("Parser: rule-based (LLM off)")
    else:
        if used:
            parts.append(f"Parser: LLM ({llm_label})" if llm_label else "Parser: LLM")
        elif attempted and failed:
            err = parser.get("error")
            if isinstance(err, str) and err.strip():
                cleaned = " ".join(err.split())
                cleaned = re.sub(r"AIza[0-9A-Za-z\-_]{20,}", "<redacted>", cleaned)
                cleaned = re.sub(r"(api_key[:=])\s*[0-9A-Za-z\-_]+", r"\1<redacted>", cleaned, flags=re.IGNORECASE)
                cleaned = re.sub(r"(Bearer\s+)[0-9A-Za-z\-_\.]+", r"\1<redacted>", cleaned, flags=re.IGNORECASE)
                cleaned = re.sub(r"sk-[0-9A-Za-z]{20,}", "sk-<redacted>", cleaned)
                if len(cleaned) > 140:
                    cleaned = cleaned[:139] + "…"
                parts.append(f"Parser: rule-based (LLM failed -> fallback: {cleaned})")
            else:
                parts.append("Parser: rule-based (LLM failed -> fallback)")
        else:
            parts.append("Parser: rule-based")

    insight_enabled = bool(insight.get("enabled"))
    if insight_enabled:
        insight_attempted = bool(insight.get("attempted"))
        insight_used = bool(insight.get("used"))
        insight_failed = bool(insight.get("failed"))

        if insight_used:
            parts.append("Insight: LLM")
        elif insight_attempted and insight_failed:
            parts.append("Insight: off (LLM failed)")

    return " | ".join(parts) if parts else None


def _compact_plotly_figure(fig):
    """
    Normalize Plotly figure sizing for the chat UI.

    Streamlit renders Plotly figures at their layout height, which can feel oversized in chat.
    We cap the height to a small, readable default.
    """
    try:
        target_height = int(getattr(settings, "CHATBOT_CHART_HEIGHT", 420) or 420)
    except Exception:
        target_height = 420

    try:
        if fig is None or not hasattr(fig, "update_layout"):
            return fig

        current_height = getattr(getattr(fig, "layout", None), "height", None)
        if not isinstance(current_height, (int, float)) or current_height > target_height:
            fig.update_layout(height=target_height)

        fig.update_layout(margin=dict(l=20, r=20, t=50, b=30))
    except Exception:
        return fig

    return fig


def render_chatbot():
    """
    Renders an attractive chat bot interface in a single window mode.
    """
        # Custom CSS for the chat interface - PROFESSIONAL DARK BLUE THEME
    st.markdown("""
    <style>
        .stChatInput {
            bottom: 20px;
        }
        .main-chat-container {
            max-width: 800px;
            margin: auto;
            padding-top: 2rem;
            border-radius: 10px;
            background-color: #f8f9fa; 
            padding-bottom: 50px;
        }
        .chat-header {
            text-align: center;
            margin-bottom: 2rem;
            color: #0f172a; /* Dark Blue */
            font-family: 'Helvetica Neue', sans-serif;
            font-weight: 700;
        }
        /* Message Bubbles */
        /* Sidebar Chat Styling - REVERTED TO WHITE */
        section[data-testid="stSidebar"] {
             background-color: #ffffff !important;
             color: #0f172a !important;
        }
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: #0f172a !important;
        }
        section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
            color: #0f172a !important;
        }
        /* Improve Visibility of Toggle/Chat Container */
        .stChatInputContainer {
             background-color: #ffffff;
             border: 1px solid #e2e8f0;
             border-radius: 8px;
        }
        div[data-testid="stChatMessage"] {
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            border-left: 5px solid #0f172a; /* Dark Blue Accent */
        }
        /* User Message Accent */
        div[data-testid="stChatMessage"]:nth-child(even) {
             border-left: 5px solid #3b82f6; /* Lighter Blue for User */
             background-color: #f1f5f9;
        }
        
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        user_role = st.session_state.get("user", {}).get("role", "national")
        if not st.session_state.get("chatbot_program"):
            st.session_state["chatbot_program"] = "maternal"
        welcome_msg = get_program_welcome_message(user_role, st.session_state["chatbot_program"])
        st.session_state.messages.append({
            "role": "assistant",
            "content": welcome_msg
        })

    # SIDEBAR: Clear Chat Button - ADDED KEY
    if st.sidebar.button("🗑️ Clear Chat History", key="clear_chat_history_btn"):
         st.session_state.messages = []
         st.session_state.chatbot_context = {}
         st.session_state["chatbot_program"] = None
         st.rerun()

    # SIDEBAR: Chatbot manual (view + downloads)
    with st.sidebar.expander("📘 Chatbot manual", expanded=False):
        manual_md = build_chatbot_manual_markdown()

        if hasattr(st, "dialog"):

            @st.dialog("IMNID Chatbot Manual")
            def _show_manual_dialog():
                sections = get_chatbot_manual_sections()
                if len(sections) >= 2:
                    tabs = st.tabs([section.title for section in sections])
                    for tab, section in zip(tabs, sections):
                        with tab:
                            st.markdown(section.markdown)
                else:
                    st.markdown(sections[0].markdown)

            if st.button("Open manual", use_container_width=True, key="open_chatbot_manual_btn"):
                _show_manual_dialog()
        else:
            # Fallback for older Streamlit: show inline in the main page.
            if st.button("Show manual", use_container_width=True, key="show_chatbot_manual_inline_btn"):
                st.session_state["chatbot_show_manual_inline"] = True

        st.caption("Download (offline use):")
        pdf_bytes = generate_chatbot_manual_pdf_bytes(manual_md)
        if pdf_bytes:
            st.download_button(
                "Download PDF",
                data=pdf_bytes,
                file_name="IMNID_Chatbot_Manual.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="download_chatbot_manual_pdf_btn",
            )
        else:
            st.caption("PDF export unavailable (missing reportlab).")

        doc_bytes = generate_chatbot_manual_doc_bytes(manual_md)
        st.download_button(
            "Download Word (.doc)",
            data=doc_bytes,
            file_name="IMNID_Chatbot_Manual.doc",
            mime="application/msword",
            use_container_width=True,
            key="download_chatbot_manual_doc_btn",
        )

    with st.sidebar.expander("LLM status", expanded=False):
        llm_label = format_llm_label()
        if settings.CHATBOT_USE_LLM:
            st.markdown(f"**Parser:** LLM ({llm_label or 'not configured'})")
            st.caption(f"Parser mode: {getattr(settings, 'CHATBOT_LLM_PARSER_MODE', 'fallback')}")
        else:
            st.markdown("**Parser:** rule-based (LLM off)")

        if settings.CHATBOT_USE_LLM_INSIGHTS:
            st.markdown("**Insights:** LLM on")
        else:
            st.markdown("**Insights:** off")

    st.markdown('<div class="main-chat-container">', unsafe_allow_html=True)
    st.markdown('<h1 class="chat-header">🤖 IMNID Chatbot</h1>', unsafe_allow_html=True)
    
    # Fallback inline manual viewer (only used if `st.dialog` is unavailable)
    if st.session_state.get("chatbot_show_manual_inline"):
        with st.expander("📘 Chatbot manual", expanded=True):
            sections = get_chatbot_manual_sections()
            if len(sections) >= 2:
                tabs = st.tabs([section.title for section in sections])
                for tab, section in zip(tabs, sections):
                    with tab:
                        st.markdown(section.markdown)
            else:
                st.markdown(sections[0].markdown)

            if st.button("Close manual", key="close_chatbot_manual_inline_btn"):
                st.session_state["chatbot_show_manual_inline"] = False
                st.rerun()

    llm_label = format_llm_label()
    insights_status = "ON" if settings.CHATBOT_USE_LLM_INSIGHTS else "OFF"
    if settings.CHATBOT_USE_LLM:
        st.caption(
            f"LLM parser: ON ({llm_label or 'not configured'}) | mode: {getattr(settings, 'CHATBOT_LLM_PARSER_MODE', 'fallback')} | insights: {insights_status}"
        )
    else:
        st.caption(f"LLM parser: OFF (rule-based) | insights: {insights_status}")

    # Ensure Data Availability
    data = ensure_data_loaded()
    
    # FIX: Robust check for data validity (Handles DataFrames correctly)
    data_invalid = False
    if data is None:
        data_invalid = True
    elif isinstance(data, pd.DataFrame) and data.empty:
        data_invalid = True
    
    if data_invalid:
        st.error("Unable to load data. Please ensure you are logged in correctly.")
        return

    # CACHE CHATBOT LOGIC INSTANCE (To avoid slow __init__ on every rerun)
    if "chatbot_instance" not in st.session_state:
        st.session_state.chatbot_instance = ChatbotLogic(data)
    
    chatbot_logic = st.session_state.chatbot_instance
    
    # Update data if it changed (pointer update is fast)
    chatbot_logic.data = data
    chatbot_logic.maternal_df = data.get("maternal", {}).get("patients", pd.DataFrame()) if data.get("maternal") else pd.DataFrame()
    chatbot_logic.newborn_df = data.get("newborn", {}).get("patients", pd.DataFrame()) if data.get("newborn") else pd.DataFrame()
    
    # Defensive fix: Ensure unique index from the start
    if not chatbot_logic.maternal_df.empty:
        chatbot_logic.maternal_df = chatbot_logic.maternal_df.reset_index(drop=True)
    if not chatbot_logic.newborn_df.empty:
        chatbot_logic.newborn_df = chatbot_logic.newborn_df.reset_index(drop=True)
    chatbot_logic.apply_active_program_globals(st.session_state.get("chatbot_program"))
    
    # Inject KB_DEFINITIONS for Definition Intent
    chatbot_logic.KB_DEFINITIONS = {
        "C-Section Rate (%)": "A Caesarean section (C-section) is a surgical procedure used to deliver a baby through incisions in the abdomen and uterus. The rate is the percentage of deliveries performed via C-section out of total deliveries.",
        "Postpartum Hemorrhage (PPH) Rate (%)": "Postpartum Hemorrhage (PPH) is defined as excessive bleeding after childbirth (usually >500ml for vaginal, >1000ml for C-section). It is a leading cause of maternal mortality.",
        "Maternal Death Rate (per 100,000)": "Maternal death refers to the death of a woman while pregnant or within 42 days of termination of pregnancy, from any cause related to or aggravated by the pregnancy or its management but not from accidental or incidental causes. The rate is calculated per 100,000 live births.",
        "Stillbirth Rate (%)": "A stillbirth is the death or loss of a baby before or during delivery. The rate typically measures stillbirths per 1,000 total births, but here it is presented as a percentage.",
        "Early Postnatal Care (PNC) Coverage (%)": "Early Postnatal Care refers to the medical care given to the mother and newborn within the first 24-48 hours after delivery, crucial for detecting complications.",
        "Maternal Coverage Rate": "Maternal Coverage Rate measures the percentage of admitted mothers captured in the system compared to the aggregated maternal admissions denominator (from `utils/aggregated_admission_mothers.xlsx`).",
        "Admitted Mothers": "The total count of mothers admitted to the facility for delivery or pregnancy-related care.",
        "Total Deliveries": "The total number of delivery events recorded, regardless of the outcome (live birth or stillbirth).",
        # Expanded Definitions
        "Immediate Postpartum Contraceptive Acceptance Rate (IPPCAR %)": "IPPCAR measures the percentage of women who accept a modern family planning method immediately after delivery (before discharge). High coverage indicates effective family planning counseling.",
        "Delivered women who received uterotonic (%)": "This indicator measures the percentage of women who received a uterotonic drug (like oxytocin) immediately after delivery to prevent Postpartum Hemorrhage (PPH).",
        "ARV Prophylaxis Rate (%)": "ARV Prophylaxis Rate tracks the percentage of HIV-exposed infants who received antiretroviral drugs to prevent Mother-to-Child Transmission (PMTCT).",
        "Assisted Delivery Rate (%)": "Assisted delivery refers to vaginal deliveries assisted by instruments like vacuum extractors or forceps. It is an alternative to C-section for prolonged labor.",
        "Normal Vaginal Delivery (SVD) Rate (%)": "SVD refers to Spontaneous Vaginal Delivery without instrumental assistance or surgery. It is the natural mode of birth.",
        "Missing Mode of Delivery": "Data Quality Metric: The percentage of delivery records where the 'Mode of Delivery' (e.g., SVD, C-Section) was not recorded.",
        "Missing Birth Outcome": "Data Quality Metric: The percentage of delivery records where the 'Birth Outcome' (e.g., Live Birth, Stillbirth) was not recorded.",
        "Missing Condition of Discharge": "Data Quality Metric: The percentage of maternal discharge records where the mother's condition (e.g., Discharged Healthy, Referred, Death) was not recorded.",
        "Episiotomy Rate (%)": "An episiotomy is a surgical cut made in the muscle between the vagina and the anus during childbirth to assist delivery. The rate measures how often this procedure is performed.",
        "Antepartum Complications Rate (%)": "Antepartum complications refer to medical conditions that arise during pregnancy before labor, such as hypertension or hemorrhage.",
        "Postpartum Complications Rate (%)": "Postpartum complications refer to medical conditions that arise after delivery, such as infections, hemorrhage (non-PPH specific), or other maternal distress issues.",
        # Newborn Definitions
        "Inborn Rate (%)": "The percentage of babies born within the current facility out of total newborn admissions.",
        "Outborn Rate (%)": "The percentage of babies born outside (e.g., at home or another facility) and then admitted to this facility.",
        "Hypothermia on Admission Rate (%)": "Hypothermia is a dangerously low body temperature (<36.5°C). This rate tracks how many newborns are cold when first admitted to the NICU.",
        "Neonatal Mortality Rate (%)": "The percentage of newborns who died while under care in the facility.",
        "KMC Coverage by Birth Weight": "Kangaroo Mother Care (KMC) involves skin-to-skin contact and exclusive breastfeeding. This metric tracks coverage among low birth weight infants.",
        "General CPAP Coverage": "Continuous Positive Airway Pressure (CPAP) is a breathing support for newborns. This tracks the percentage of all admitted newborns who received CPAP."
    }

    # Function to execute specialized rendering
    def execute_specialized(spec):
        suffix = spec["suffix"]
        func_prefix = spec["func_prefix"]
        render_df = spec["data"]
        params = spec["params"]
        comparison_mode = spec["comparison_mode"]
        comparison_entity = spec["comparison_entity"]
        is_compare_all = spec.get("is_compare_all", False)  # Get the flag
        message_index = spec.get("message_index", 0)
        
        # If comparing all entities, only show the table, not the plot
        suppress_plot = is_compare_all
        base_title = params.get("chart_title") or params.get("active_kpi_name") or ""
         
        try:
            if suffix == "utils":
                module = kpi_utils
            elif suffix == "newborn":
                import newborns_dashboard.kpi_utils_newborn as module
            elif suffix == "newborn_simplified":
                import newborns_dashboard.kpi_utils_newborn_simplified as module
            else:
                module = __import__(f"utils.kpi_{suffix}", fromlist=[f"render_{suffix}_trend_chart"])

            if comparison_mode:
                 # Newborn simplified KPIs need to use the SAME comparison charts as the newborn dashboard
                 # (and we must ensure widget keys are unique across chat history).
                 if suffix == "newborn_simplified":
                     from utils.queries import (
                         get_facilities_grouped_by_region,
                         get_all_facilities_flat,
                     )
 
                     user_ctx = st.session_state.get("user", {}) or {}
                     regions_mapping = get_facilities_grouped_by_region(user_ctx)
                     active_kpi = params.get("active_kpi_name")
                     chart_title = params.get("chart_title") or active_kpi
                     key_suffix = f"msg_{message_index}"
 
                     if comparison_entity == "facility":
                         facility_uids = (
                             params.get("facility_uids")
                             or params.get("all_comparison_uids")
                             or []
                         )
                         facility_names = params.get("facility_names") or []
 
                         if facility_uids and (
                             (not facility_names)
                             or len(facility_names) != len(facility_uids)
                         ):
                             uid_to_name = {
                                 uid: name
                                 for name, uid in get_all_facilities_flat(user_ctx)
                             }
                             facility_names = [
                                 uid_to_name.get(uid, uid) for uid in facility_uids
                             ]
 
                         if active_kpi in {"Birth Weight Rate", "Birth Weight Distribution"}:
                              module.render_birth_weight_facility_comparison(
                                  render_df,
                                  period_col="period_display",
                                  title=f"{chart_title} - Facility Comparison",
                                  facility_names=facility_names,
                                  facility_uids=facility_uids,
                                  key_suffix=key_suffix,
                              )
                         elif active_kpi == "KMC Coverage by Birth Weight":
                              module.render_kmc_coverage_comparison_chart(
                                  render_df,
                                  comparison_mode="facility",
                                  display_names=facility_names,
                                  facility_uids=facility_uids,
                                  facilities_by_region=regions_mapping,
                                  region_names=params.get("comparison_targets"),
                                  period_col="period_display",
                                  title=f"{chart_title} - Facility Comparison",
                                  key_suffix=key_suffix,
                              )
                         elif active_kpi == "CPAP for RDS":
                              module.render_cpap_rds_comparison_line_chart(
                                  render_df,
                                  comparison_mode="facility",
                                  display_names=facility_names,
                                  facility_uids=facility_uids,
                                  facilities_by_region=regions_mapping,
                                  region_names=params.get("comparison_targets"),
                                  period_col="period_display",
                                  title=f"{chart_title} - Facility Comparison",
                                  key_suffix=key_suffix,
                              )
                         elif active_kpi in {
                              "CPAP Coverage by Birth Weight",
                              "General CPAP Coverage",
                          }:
                              module.render_cpap_by_weight_comparison_chart(
                                  render_df,
                                  comparison_mode="facility",
                                  display_names=facility_names,
                                  facility_uids=facility_uids,
                                  facilities_by_region=regions_mapping,
                                  region_names=params.get("comparison_targets"),
                                  period_col="period_display",
                                  title=f"{chart_title} - Facility Comparison",
                                  key_suffix=key_suffix,
                              )
                         else:
                             # Fallback to legacy name pattern if present.
                             func_name = f"{func_prefix}_facility_comparison"
                             render_func = getattr(module, func_name)
                             render_func(
                                 render_df,
                                 facility_uids=facility_uids,
                                 facility_names=facility_names,
                                 key_suffix=key_suffix,
                             )
                     else:
                         region_names = params.get("comparison_targets") or []
 
                         if active_kpi in {"Birth Weight Rate", "Birth Weight Distribution"}:
                              module.render_birth_weight_region_comparison(
                                  render_df,
                                  period_col="period_display",
                                  title=f"{chart_title} - Region Comparison",
                                  region_names=region_names,
                                  region_mapping=regions_mapping,
                                  facilities_by_region=regions_mapping,
                                  key_suffix=key_suffix,
                              )
                         elif active_kpi == "KMC Coverage by Birth Weight":
                              module.render_kmc_coverage_comparison_chart(
                                  render_df,
                                  comparison_mode="region",
                                  display_names=region_names,
                                  facilities_by_region=regions_mapping,
                                  region_names=region_names,
                                  period_col="period_display",
                                  title=f"{chart_title} - Region Comparison",
                                  key_suffix=key_suffix,
                              )
                         elif active_kpi == "CPAP for RDS":
                              module.render_cpap_rds_comparison_line_chart(
                                  render_df,
                                  comparison_mode="region",
                                  display_names=region_names,
                                  facilities_by_region=regions_mapping,
                                  region_names=region_names,
                                  period_col="period_display",
                                  title=f"{chart_title} - Region Comparison",
                                  key_suffix=key_suffix,
                              )
                         elif active_kpi in {
                              "CPAP Coverage by Birth Weight",
                              "General CPAP Coverage",
                          }:
                              module.render_cpap_by_weight_comparison_chart(
                                  render_df,
                                  comparison_mode="region",
                                  display_names=region_names,
                                  facilities_by_region=regions_mapping,
                                  region_names=region_names,
                                  period_col="period_display",
                                  title=f"{chart_title} - Region Comparison",
                                  key_suffix=key_suffix,
                              )
                         else:
                             func_name = f"{func_prefix}_region_comparison"
                             render_func = getattr(module, func_name)
                             render_func(
                                 render_df,
                                 region_names=region_names,
                                 region_mapping=regions_mapping,
                                 facilities_by_region=regions_mapping,
                                 key_suffix=key_suffix,
                             )
 
                     return
                 if comparison_entity == "facility":
                     if suffix == "newborn_simplified":
                          func_name = f"{func_prefix}_facility_comparison"
                     else:
                         func_name = f"{func_prefix}_facility_comparison_chart"
                         
                     render_func = getattr(module, func_name)
                    
                     if suffix == "utils":
                        render_func(render_df, "period_display", "value", base_title, "#FFFFFF", None, params["facility_names"], params["facility_uids"], params["num_label"], params["den_label"], suppress_plot=suppress_plot, key_suffix=f"msg_{message_index}")
                     elif suffix == "newborn" and func_prefix.startswith("render_admitted_newborns"):
                        render_func(render_df, "period_display", "value", base_title, "#FFFFFF", None, params["facility_names"], params["facility_uids"], value_name=params["active_kpi_name"], suppress_plot=suppress_plot, key_suffix=f"msg_{message_index}")
                     elif suffix == "newborn":
                        render_func(render_df, "period_display", "value", base_title, "#FFFFFF", None, params["facility_names"], params["facility_uids"], params["num_label"], params["den_label"], suppress_plot=suppress_plot, key_suffix=f"msg_{message_index}")
                     elif suffix == "newborn_simplified":
                        render_func(render_df, facility_uids=params["all_comparison_uids"], facility_names=params["facility_names"], key_suffix=f"msg_{message_index}")
                     elif suffix == "admitted_mothers":
                        # admitted_mothers has a different signature (no num/den labels)
                        render_func(render_df, "period_display", "value", base_title, "#FFFFFF", None, params["facility_names"], params["facility_uids"], suppress_plot=suppress_plot, key_suffix=f"msg_{message_index}")
                     else:
                        # All other specialized maternal modules (pph, svd, arv, etc.) use the same signature as utils
                        render_func(render_df, "period_display", "value", base_title, "#FFFFFF", None, params["facility_names"], params["facility_uids"], params["num_label"], params["den_label"], suppress_plot=suppress_plot, key_suffix=f"msg_{message_index}")
                 else:
                     if suffix == "newborn_simplified":
                          func_name = f"{func_prefix}_region_comparison"
                     else:
                          func_name = f"{func_prefix}_region_comparison_chart"
                         
                     render_func = getattr(module, func_name)
                    
                     if suffix == "utils":
                        from utils.queries import get_facilities_grouped_by_region
                        regions_mapping = get_facilities_grouped_by_region(st.session_state.get("user", {}))
                        # FIX: Pass region names (comparison_targets) instead of UIDs
                        render_func(render_df, "period_display", "value", base_title, "#FFFFFF", None, params["comparison_targets"], regions_mapping, regions_mapping, params["num_label"], params["den_label"], suppress_plot=suppress_plot, key_suffix=f"msg_{message_index}")
                     elif suffix == "newborn" and func_prefix.startswith("render_admitted_newborns"):
                        from utils.queries import get_facilities_grouped_by_region
                        regions_mapping = get_facilities_grouped_by_region(st.session_state.get("user", {}))
                        render_func(render_df, "period_display", "value", base_title, "#FFFFFF", None, params["comparison_targets"], regions_mapping, regions_mapping, value_name=params["active_kpi_name"], suppress_plot=suppress_plot, key_suffix=f"msg_{message_index}")
                     elif suffix == "newborn":
                        from utils.queries import get_facilities_grouped_by_region
                        regions_mapping = get_facilities_grouped_by_region(st.session_state.get("user", {}))
                        # FIX: Pass region names (comparison_targets) instead of UIDs
                        render_func(render_df, "period_display", "value", base_title, "#FFFFFF", None, params["comparison_targets"], regions_mapping, regions_mapping, params["num_label"], params["den_label"], suppress_plot=suppress_plot, key_suffix=f"msg_{message_index}")
                     elif suffix == "newborn_simplified":
                        # FIX: Use region names and region/facility mappings
                        from utils.queries import get_facilities_grouped_by_region
                        regions_mapping = get_facilities_grouped_by_region(st.session_state.get("user", {}))
                        render_func(render_df, region_names=params["comparison_targets"], region_mapping=regions_mapping, facilities_by_region=regions_mapping)
                     elif suffix == "admitted_mothers":
                        # admitted_mothers has a different signature
                        # FIX: Use region names
                        render_func(render_df, "period_display", "value", base_title, "#FFFFFF", None, params["comparison_targets"], suppress_plot=suppress_plot, key_suffix=f"msg_{message_index}")
                     else:
                        # All other specialized maternal modules use the same signature as utils
                        from utils.queries import get_facilities_grouped_by_region
                        regions_mapping = get_facilities_grouped_by_region(st.session_state.get("user", {}))
                        # FIX: Pass region names (comparison_targets) instead of UIDs
                        render_func(render_df, "period_display", "value", base_title, "#FFFFFF", None, params["comparison_targets"], regions_mapping, regions_mapping, params["num_label"], params["den_label"], suppress_plot=suppress_plot, key_suffix=f"msg_{message_index}")
            else:
                render_func = getattr(module, f"{func_prefix}_trend_chart")
                if suffix == "utils":
                    render_func(render_df, "period_display", "value", base_title, "#FFFFFF", None, params["facility_names"], params["num_label"], params["den_label"], facility_uids=[f"msg_{message_index}"], key_suffix=f"msg_{message_index}")
                elif suffix == "newborn" and func_prefix.startswith("render_admitted_newborns"):
                    render_func(render_df, "period_display", "value", base_title, "#FFFFFF", None, params["facility_names"], value_name=params["active_kpi_name"], facility_uids=params.get("facility_uids"), key_suffix=f"msg_{message_index}")
                elif suffix == "newborn":
                    render_func(render_df, "period_display", "value", base_title, "#FFFFFF", None, params["facility_names"], params["num_label"], params["den_label"], facility_uids=params.get("facility_uids"), key_suffix=f"msg_{message_index}")
                elif suffix == "newborn_simplified":
                     render_func(
                         render_df,
                         period_col="period_display",
                         title=base_title,
                         facility_uids=params.get("facility_uids"),
                         key_suffix=f"msg_{message_index}",
                     )
                elif suffix == "admitted_mothers":
                     # admitted_mothers has a different signature
                     render_func(render_df, "period_display", "value", base_title, "#FFFFFF", None, params["facility_names"], facility_uids=[f"msg_{message_index}"], key_suffix=f"msg_{message_index}")
                else:
                    # All other specialized maternal modules (pph, svd, arv, etc.) use the same signature as utils
                    render_func(render_df, "period_display", "value", base_title, "#FFFFFF", None, params["facility_names"], params["num_label"], params["den_label"], facility_uids=[f"msg_{message_index}"], key_suffix=f"msg_{message_index}")
        except Exception as e:
            st.error(f"Error rendering specialized content: {e}")

    # Display chat messages from history on app rerun
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message.get("role") == "assistant":
                hint_text = _format_llm_trace_caption(message.get("trace"))
                if hint_text:
                    st.caption(hint_text)
            if "content" in message:
                st.markdown(message["content"])
            
            # Handle specialized rendering spec
            if "specialized_spec" in message:
                # Add message index to spec for unique widget keys
                spec_with_index = message["specialized_spec"].copy()
                spec_with_index["message_index"] = i
                execute_specialized(spec_with_index)
            elif "figure" in message:
                # Handle DataFrame vs Plotly Figure
                if isinstance(message["figure"], pd.DataFrame):
                     st.dataframe(message["figure"])
                else:
                     fig_to_show = _compact_plotly_figure(message["figure"])
                     st.plotly_chart(fig_to_show, use_container_width=True, key=f"chat_chart_{i}")

    # Accept user input
    selected_program = st.session_state.get("chatbot_program")
    if selected_program:
        chat_placeholder = f"Ask about {selected_program} indicators..."
    else:
        chat_placeholder = "Start by typing `maternal` or `newborn`..."
    if prompt := st.chat_input(chat_placeholder):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            hint_placeholder = st.empty()
            message_placeholder = st.empty()
            
            with st.spinner("Analyzing data..."):
                # Run Logic
                try:
                    trace = None
                    # Check for help intent explicitly here or rely on chatbot_logic
                    if prompt.lower().strip() in ["help", "info", "usage"]:
                         user_role = st.session_state.get("user", {}).get("role", "national")
                         selected_program = st.session_state.get("chatbot_program")
                         if selected_program:
                              response_text = get_program_welcome_message(user_role, selected_program)
                         else:
                              response_text = get_program_selection_message(user_role)
                         fig = None
                    else:
                         # Rerurns (fig, text)
                          fig, response_text = chatbot_logic.generate_response(prompt)
                          trace = getattr(chatbot_logic, "last_trace", None)
                          if fig is None and response_text is None:
                              response_text = "I couldn't parse that yet. Please mention a program (maternal/newborn) and an indicator, or say `help`."
                    
                    hint_text = _format_llm_trace_caption(trace)
                    if hint_text:
                        hint_placeholder.caption(hint_text)
                    message_placeholder.markdown(response_text)
                    
                    msg_obj = {"role": "assistant", "content": response_text}
                    if trace:
                        msg_obj["trace"] = trace
                    
                    if fig is not None:
                        if isinstance(fig, dict) and fig.get("type") == "specialized":
                            # Execute Specialized Rendering
                            # FIX: Pass unique index (length of current history + 1 or just current length since we appended prompt already?)
                            # We appended prompt (msg-1) at line 2359.
                            # So current length used for next message is len(st.session_state.messages).
                            # Wait, we haven't appended the ASSISTANT message yet (happens at 2397).
                            # So collision checks needs to be careful.
                            # History loop ran on 0..N.
                            # We are making N+1.
                            current_msg_index = len(st.session_state.messages) 
                            
                            # Update Spec with Index
                            fig["message_index"] = current_msg_index
                            execute_specialized(fig)
                            
                            # Save Spec to history
                            msg_obj["specialized_spec"] = fig
                        elif isinstance(fig, pd.DataFrame):
                            st.dataframe(fig)
                            msg_obj["figure"] = fig
                        else:
                            fig_to_show = _compact_plotly_figure(fig)
                            st.plotly_chart(fig_to_show, use_container_width=True, key=f"chat_chart_new_{len(st.session_state.messages)}")
                            msg_obj["figure"] = fig_to_show
                        
                    # Save to history
                    st.session_state.messages.append(msg_obj)
                    
                    if response_text == "Chat history cleared.":
                        st.rerun()
                        
                except Exception as e:
                    logging.error(f"Chatbot Error: {e}", exc_info=True)
                    
                    # USER FRIENDLY ERROR GUIDANCE
                    error_msg = "**I had some trouble processing that request.** 🧐\n\n"
                    
                    e_str = str(e).lower()
                    if "kpi" in e_str or "indicator" in e_str:
                        selected_program = st.session_state.get("chatbot_program")
                        if selected_program:
                            program_label = PROGRAM_CONFIGS[selected_program]["label"]
                            error_msg += (
                                f"It seems I couldn't find the specific indicator you're looking for in the **{program_label}** program. "
                                "Say `list indicators` to see what I can analyze."
                            )
                        else:
                            error_msg += "It seems I couldn't find the specific indicator you're looking for. Say `list indicators` to see what I can analyze."
                    elif "facility" in e_str or "region" in e_str:
                        error_msg += "I had trouble identifying the location (facility or region) in your prompt."
                    elif "local variable" in e_str:
                        error_msg += (
                            "I couldn't resolve that request cleanly. "
                            "Say `list indicators` to see what I can analyze, or type `maternal`/`newborn` to switch programs."
                        )
                    else:
                        error_msg += (
                            "I'm not quite sure how to handle that specific question yet. "
                            "Say `list indicators` to see the indicators I can analyze."
                        )
                    
                    error_msg += "\n\n---\n"
                    error_msg += "💡 **How to Prompt for Best Results:**\n"
                    error_msg += "- **Be Specific**: Mention an indicator and a location. *(e.g., 'Plot PPH for Tigray')*\n"
                    error_msg += "- **Ask for Lists**: If you're unsure, ask 'list indicators' or 'show all regions'.\n"
                    error_msg += "- **Keep it Simple**: Ask one question at a time.\n"
                    
                    message_placeholder.markdown(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

    st.markdown('</div>', unsafe_allow_html=True)

    # Auto-scroll to bottom on each rerun
    st.markdown(
        "<script>window.scrollTo(0, document.body.scrollHeight);</script>",
        unsafe_allow_html=True,
    )


def _legacy_get_welcome_message(role):
    """Generates a dynamic welcome message based on user role."""
    
    # Dashboard List
    dashboards = ["Maternal", "Newborn", "Summary", "Resources"] # Default/Facility
    if role != "facility":
        dashboards = ["Maternal", "Newborn", "Summary", "Mentorship", "Resources", "Data Quality"]
        
    dashboard_str = ", ".join(dashboards)
    
    msg = f"""**Hello! I'm your IMNID dashboard assistant.** 🤖
    
I can help you analyze data across the **{dashboard_str}** dashboards.

**What I can do for you:**
- **Plot Charts**: I can generate Line, Bar, and Area charts for Maternal Health indicators.
- **Quick Values**: Ask for a specific value (e.g. "What is the PPH rate?") and I'll provide the overall value.
- **Comparisons**: Ask to compare facilities or regions! (e.g., "Compare Admitted Mothers for Adigrat and Suhul")
- **Definitions**: Ask "Define [Indicator]" to get a medical definition.
- **List Indicators**: Ask "list all indicators" to see all available health indicators.
- **List Facilities**: Ask "list all facilities" to see all available health facilities.

**How to Compare:**
1. **By Facility**: "Compare [KPI] for [Facility A] and [Facility B]"
   *(Example: "Compare Admitted Mothers for Adigrat and Suhul General Hospital")*
2. **By Region**: "Compare [KPI] for [Region A] and [Region B]"
   *(Example: "Compare C-Section Rate for Tigray and Amhara")*
3. **Drill Down**: "Compare [KPI] for [Region] by facility"
   *(Example: "Compare Admitted Mothers for Tigray by facility")*

**What I cannot do:**
- 🚫 I **cannot** generate, update, or modify health data. All data is read-only.
- 🚫 I cannot predict future data (forecasting is not yet enabled).

**Try asking:**
- "Plot C-Section Rate last year"
- "Show me Admitted Mothers"
- "Compare Admitted Mothers for Adigrat and Suhul"

Type **'Help'** at any time to see this message again.
"""
    return msg


def get_welcome_message(role):
    selected_program = st.session_state.get("chatbot_program")
    if selected_program:
        return get_program_welcome_message(role, selected_program)
    return get_program_selection_message(role)
