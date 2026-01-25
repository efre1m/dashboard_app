import streamlit as st
import pandas as pd
import plotly.express as px
import re
import random
import difflib
import logging
from datetime import datetime, timedelta
from utils.llm_utils import query_llm
from utils import kpi_utils

# Import logic from dashboards to ensure data availability
from dashboards import facility, regional, national
import utils.kpi_utils as kpi_utils
from utils.kpi_utils import (
    prepare_data_for_trend_chart,
    compute_kpis
)
from utils.kpi_admitted_mothers import get_numerator_denominator_for_admitted_mothers
from utils.queries import get_facility_mapping_for_user, get_facilities_grouped_by_region, get_all_facilities_flat
from utils.dash_co import KPI_OPTIONS, KPI_MAPPING

def ensure_data_loaded():
    """
    Ensures that the necessary data is loaded into session state based on the user's role.
    Returns the shared_data dict.
    """
    user = st.session_state.get("user", {})
    role = user.get("role", "")
    
    if not role:
        return None

    if role == "facility":
        # Check if data is already in session state
        if hasattr(st.session_state, "cached_shared_data_facility") and st.session_state.cached_shared_data_facility:
            return st.session_state.cached_shared_data_facility
            
        # Load data if not present
        with st.spinner("Initializing chatbot data access..."):
            static_data = facility.get_static_data_facility(user)
            program_uid_map = static_data["program_uid_map"]
            data = facility.get_shared_program_data_facility(user, program_uid_map, show_spinner=False)
            return data
            
    elif role == "regional":
        if hasattr(st.session_state, "cached_shared_data_regional") and st.session_state.cached_shared_data_regional:
            return st.session_state.cached_shared_data_regional
            
        with st.spinner("Initializing chatbot data access..."):
            static_data = regional.get_static_data(user)
            program_uid_map = static_data["program_uid_map"]
            data = regional.get_shared_program_data_optimized(user, program_uid_map, show_spinner=False)
            return data
            
    elif role == "national":
        if hasattr(st.session_state, "cached_shared_data_national") and st.session_state.cached_shared_data_national:
            return st.session_state.cached_shared_data_national
            
        with st.spinner("Initializing chatbot data access..."):
            static_data = national.get_static_data(user)
            program_uid_map = static_data["program_uid_map"]
            data = national.get_shared_program_data_optimized(user, program_uid_map, show_spinner=False)
            return data
            
    return None

class ChatbotLogic:
    def __init__(self, data):
        self.data = data
        self.user = st.session_state.get("user", {})
        self.facility_mapping = get_facility_mapping_for_user(self.user)
        # Reverse mapping for easy lookup
        # Revised mapping to match current file state: self.uid_to_name
        self.uid_to_name = {v: k for k, v in self.facility_mapping.items()}
        
        # Maternal and Newborn data
        self.maternal_df = data.get("maternal", {}).get("patients", pd.DataFrame()) if data.get("maternal") else pd.DataFrame()
        self.newborn_df = data.get("newborn", {}).get("patients", pd.DataFrame()) if data.get("newborn") else pd.DataFrame()
        
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
            "birht": "birth",
            "oucome": "outcome",
            "indicatofrs": "indicators",
            "totaly": "totally",
            "wome": "women",
            "whor": "who",
            "abot": "about",
            "abut": "about",
            "enrollmet": "enrollment",
            "admited": "admitted",
            "admision": "admission",
            "delivry": "delivery",
            "vagnal": "vaginal",
            "materna": "maternal",
            "materanl": "maternal",
            "matenal": "maternal",
            
            # Intent/Action Variations
            "inidcatorys": "indicators",
            "indicatorys": "indicators",
            "indicater": "indicator",
            "indicatry": "indicator",
            "indicotrs": "indicators",
            "indcators": "indicators",
            "indictors": "indicators",
            "lsit": "list",
            "listt": "list",
            "shw": "show",
            "sho": "show",
            "reioings": "regions",
            "reigons": "regions",
            "reginos": "regions",
            "faciltiy": "facility",
            "faciltiyies": "facilities",
            "faclities": "facilities",
            "regijon": "region",
            "regjon": "region",
            "indicatorys": "indicators",
        }

        # --- SPECIALIZED KPI MAPPING ---
        # Maps full KPI names to their internal utility script suffixes
        # Keys MUST match active_kpi_name (which comes from KPI_MAPPING/KPI_OPTIONS)
        self.SPECIALIZED_KPI_MAP = {
            # Maternal Indicators
            "Total Admitted Mothers": "admitted_mothers",
            "Admitted Mothers": "admitted_mothers",
            "Postpartum Hemorrhage (PPH) Rate (%)": "pph",
            "Normal Vaginal Delivery (SVD) Rate (%)": "svd",
            "ARV Prophylaxis Rate (%)": "arv",
            "Assisted Delivery Rate (%)": "assisted",
            "Delivered women who received uterotonic (%)": "uterotonic",
            "Missing Birth Outcome": "missing_bo",
            "Missing Condition of Discharge": "missing_cod",
            "Missing Mode of Delivery": "missing_md",
            "Episiotomy Rate (%)": "episiotomy",
            "Antepartum Complications Rate (%)": "antipartum_compl",
            
            # Newborn Indicators
            "Inborn Rate (%)": "newborn",
            "Outborn Rate (%)": "newborn",
            "Hypothermia on Admission Rate (%)": "newborn",
            "Inborn Hypothermia Rate (%)": "newborn",
            "Outborn Hypothermia Rate (%)": "newborn",
            "Neonatal Mortality Rate (%)": "newborn",
            "Admitted Newborns": "newborn",
            "Birth Weight Rate": "newborn_simplified",
            "KMC Coverage by Birth Weight": "newborn_simplified",
            "General CPAP Coverage": "newborn_simplified",
            "CPAP for RDS": "newborn_simplified",
            "CPAP Coverage by Birth Weight": "newborn_simplified",
            "Missing Temperature (%)": "newborn",
            "Missing Birth Weight (%)": "newborn",
            "Missing Discharge Status (%)": "newborn",

            # Standard KPIs that use kpi_utils
            "C-Section Rate (%)": "utils",
            "Institutional Maternal Death Rate (%)": "utils",
            "Stillbirth Rate (%)": "utils",
            "Early Postnatal Care (PNC) Coverage (%)": "utils",
            "Immediate Postpartum Contraceptive Acceptance Rate (IPPCAR %)": "utils"
        }

    def _silent_prepare_data(self, df, kpi_name, facility_uids=None, date_range_filters=None):
        """
        Silent version of prepare_data_for_trend_chart that uses logging instead of st.info/warning.
        Copied logic to ensure chatbot doesn't spam UI.
        """
        from utils.kpi_utils import get_relevant_date_column_for_kpi
        from utils.time_filter import assign_period
        
        if df.empty:
            return pd.DataFrame(), None

        filtered_df = df.copy()

        # Filter by facility UIDs if provided
        if facility_uids and "orgUnit" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

        # Get the SPECIFIC date column for this KPI
        date_column = get_relevant_date_column_for_kpi(kpi_name)

        # Check if the SPECIFIC date column exists
        if date_column not in filtered_df.columns:
            # Try to use event_date as fallback
            if "event_date" in filtered_df.columns:
                date_column = "event_date"
                logging.warning(f"Chatbot: KPI-specific date column not found for {kpi_name}, using 'event_date'")
            else:
                logging.warning(f"Chatbot: Required date column '{date_column}' not found for {kpi_name}")
                return pd.DataFrame(), date_column

        # Create result dataframe
        result_df = filtered_df.copy()

        # Convert to datetime
        result_df["event_date"] = pd.to_datetime(result_df[date_column], errors="coerce")
        # Filter out rows without valid dates (Logic from kpi_utils)
        result_df = result_df[result_df["event_date"].notna()].copy()

        # CRITICAL: Apply date range filtering
        if date_range_filters:
            start_date = date_range_filters.get("start_date")
            end_date = date_range_filters.get("end_date")

            if start_date and end_date:
                # Convert to datetime for comparison
                start_dt = pd.Timestamp(start_date)
                end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1)  # Include end date

                # Filter by date range
                result_df = result_df[
                    (result_df["event_date"] >= start_dt)
                    & (result_df["event_date"] < end_dt)
                ].copy()

        if result_df.empty:
            # Silent log instead of st.info
            logging.info(f"Chatbot: No data with valid dates in '{date_column}' for {kpi_name}")
            return pd.DataFrame(), date_column

        # Get period label (default to Monthly if not set)
        period_label = st.session_state.get("period_label", "Monthly")
        if "filters" in st.session_state and "period_label" in st.session_state.filters:
            period_label = st.session_state.filters["period_label"]

        # Create period columns
        result_df = assign_period(result_df, "event_date", period_label)

        # Filter by facility if needed (redundant usually but safe)
        if facility_uids and "orgUnit" in result_df.columns:
             result_df = result_df[result_df["orgUnit"].isin(facility_uids)].copy()

        # FIX: Ensure unique index before specialized rendering
        result_df = result_df.reset_index(drop=True)

        return result_df, date_column


    def parse_query(self, query):
        """
        Parses the user query to extract intent, kpi, filters.
        Tries LLM first, falls back to regex.
        """
        query_lower = query.lower()
        
        # --- PRE-PROCESS QUERY (Normalization & Typo Correction) ---
        # Move this to the top so it filters early keyword checks
        query_norm = re.sub(r'[^a-z0-9\s]', '', query_lower)
        for typo, correct in self.COMMON_TYPOS.items():
            query_norm = query_norm.replace(typo, correct)
        query_norm = re.sub(r'\s+', ' ', query_norm).strip()
        
        
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
        
        
        # Handle explicit list requests - OPTIMIZED FOR PLURALS & VARIATIONS
        # PRIORITIZE Facility keyword to avoid "list facilities in Tigray region" matching regions regex first
        entity_type = None
        count_requested = False
        
        if re.search(r'(list|show|shw|sho|display|tell).*(facilities|facility|hospitals?|facilti?yies?|faclities?|units?)', query_norm):
            entity_type = "facility"
        elif re.search(r'(list|show|shw|sho|display|tell).*(regions?|reioings?|reigons?|territory|territories)', query_norm):
            entity_type = "region"
            
        if entity_type:
            # Extract Region if mentioned (even in rule-based mode)
            from utils.queries import get_facilities_grouped_by_region
            regions_data = get_facilities_grouped_by_region(self.user)
            region_found = None
            for r_name in regions_data.keys():
                if r_name.lower() in query_norm:
                    region_found = r_name
                    break
            
            return {
                "intent": "metadata_query", 
                "entity_type": entity_type, 
                "count_requested": False,
                "region_filter": region_found
            }

        if re.search(r'(list|show|shw|sho|display|tell).*(indicators?|indicaters?|kpis?|measures?|metrics?)', query_norm):
            return {"intent": "list_kpis"}
        
        # 0. Try LLM Parsing
        from utils.llm_utils import query_llm
        
        # Prepare list of facility names for context
        facility_names_list = list(self.facility_mapping.keys())
        
        llm_result = query_llm(query, facility_names_list)
        
        if llm_result:
            # Handle "chat" intent from LLM
            if llm_result.get("intent") == "chat":
                 return {
                     "response": llm_result.get("response")
                 }
            
            # Special Handling for Generic "What indicators" query via LLM
            if llm_result.get("intent") == "list_kpis":
                 # Check if we should ask clarification
                 if not any(x in query.lower() for x in ["maternal", "newborn", "all"]):
                      return {
                          "intent": "chat",
                          "response": "Are you interested in **Maternal** or **Newborn** health indicators?"
                      }
            if llm_result.get("intent") == "clear":
                 return {"intent": "clear"}
                 
            # If LLM identified a KPI, use it
            if llm_result.get("kpi"):
                # Map facility names to UIDs
                selected_facility_uids = []
                selected_facility_names = []
                
                llm_facs = llm_result.get("facility_names", [])
                
                # --- FUZZY MATCHING & RESOLUTION ---
                # Resolve Facility Names (Handle Typos) AND Region Names
                regions_data = get_facilities_grouped_by_region(self.user)
                all_regions = list(regions_data.keys())
                
                found_regions = []
                
                if llm_facs:
                    for fname in llm_facs:
                        fname_clean = fname.strip().lower()
                        match_found = False

                        # 1. Try Direct Match (Facility) via UID mapping
                        if fname in self.facility_mapping:
                            selected_facility_uids.append(self.facility_mapping[fname])
                            selected_facility_names.append(fname)
                            match_found = True
                            continue
                        
                        # 2. Try Robust Partial Match against ALL facilities (Prioritize this over Region)
                        all_facilities = list(self.facility_mapping.keys())
                        
                        starts_with_match = None
                        contains_match = None
                        
                        for f in all_facilities:
                            f_clean = f.lower()
                            if f_clean == fname_clean: 
                                 starts_with_match = f
                                 break
                            if f_clean.startswith(fname_clean):
                                 starts_with_match = f
                                 break
                            if fname_clean in f_clean and not contains_match:
                                 contains_match = f
                        
                        final_match = starts_with_match or contains_match
                        
                        if final_match:
                             logging.info(f"Chatbot: Resolved '{fname}' to Facility: '{final_match}'")
                             selected_facility_uids.append(self.facility_mapping[final_match])
                             selected_facility_names.append(final_match)
                             match_found = True
                        
                        # 3. Fallback to Strict Fuzzy Match (Facility)
                        if not match_found:
                             f_matches = difflib.get_close_matches(fname, all_facilities, n=1, cutoff=0.5)
                             if f_matches:
                                 matched_name = f_matches[0]
                                 logging.info(f"Chatbot: Fuzzy Matched '{fname}' to Facility: '{matched_name}'")
                                 selected_facility_uids.append(self.facility_mapping[matched_name])
                                 selected_facility_names.append(matched_name)
                                 match_found = True
                        
                        # 4. If NOT a facility, check Regions
                        if not match_found:
                            # Direct Region Match
                            if fname in regions_data:
                                found_regions.append(fname)
                                match_found = True
                            else:
                                 # Fuzzy Region Match
                                 r_matches = difflib.get_close_matches(fname, all_regions, n=1, cutoff=0.6)
                                 if r_matches:
                                     found_regions.append(r_matches[0])
                                     match_found = True
                
                # --- DRILL-DOWN / DRILL-UP LOGIC (New) ---
                if "by facility" in query_lower or "per facility" in query_lower:
                     # Force facility comparison if region is present
                     if found_regions:
                         selected_facility_uids = []
                         selected_facility_names = []
                         for r in found_regions:
                              facs_in_region = regions_data.get(r, [])
                              selected_facility_uids.extend([f[1] for f in facs_in_region])
                              selected_facility_names.extend([f[0] for f in facs_in_region])
                         llm_result["comparison_mode"] = True
                         llm_result["comparison_entity"] = "facility"

                # If no facilities filtered by LLM but user is Facility Role, assume their facility
                if not selected_facility_uids and not found_regions and self.user.get("role") == "facility":
                     selected_facility_uids = list(self.facility_mapping.values())
                     selected_facility_names = list(self.facility_mapping.keys())
                
                # Populate associated facilities if only Region was found
                if found_regions and not selected_facility_uids:
                     # Get all facilities in these regions
                     for r in found_regions:
                          facs_in_region = regions_data.get(r, [])
                          selected_facility_uids.extend([f[1] for f in facs_in_region])
                          selected_facility_names.append(f"{r} (Region)")

                return {
                    "intent": llm_result.get("intent", "text"),
                    "chart_type": llm_result.get("chart_type", "line"),
                    "kpi": llm_result.get("kpi"),
                    "facility_uids": selected_facility_uids,
                    "facility_names": selected_facility_names,
                    "date_range": llm_result.get("date_range"),
                    "entity_type": llm_result.get("entity_type"),
                    "count_requested": llm_result.get("count_requested"),
                    "comparison_mode": llm_result.get("comparison_mode", False),
                    "comparison_entity": llm_result.get("comparison_entity"),
                    "comparison_targets": found_regions if llm_result.get("comparison_entity") == "region" else selected_facility_names, 
                    "region_filter": llm_result.get("region_filter"),
                    "response": llm_result.get("response")
                }

        # --- FALLBACK TO REGEX / FUZZY MATCHING (Existing Logic) ---
        # query_lower is already defined at top of function
        
        # Check for Clear Chat
        
        # Check for Clear Chat
        if "clear chat" in query_lower or "reset chat" in query_lower:
            return {"intent": "clear"}
        
        # 1. Detect Intent and Chart Type
        intent = "text"
        chart_type = "line" # Default
        entity_type = None
        count_requested = False
        comparison_mode = False
        comparison_entity = None
        
        if any(w in query_lower for w in ["plot", "graph", "chart", "trend", "visualize", "show me"]):
            intent = "plot"
        
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
        
        # Comprehensive KPI Map based on dash_co.KPI_MAPPING
        # Now includes phonetic variations and common typos
        kpi_map = {
            # C-Section (with typos)
            "csection": "C-Section Rate (%)",
            "section": "C-Section Rate (%)",
            "caesarean": "C-Section Rate (%)",
            "cesarean": "C-Section Rate (%)",
            "cesarian": "C-Section Rate (%)",
            "ceasarean": "C-Section Rate (%)",
            
            # Maternal Death
            "maternal death": "Institutional Maternal Death Rate (%)",
            "death": "Institutional Maternal Death Rate (%)",
            "mortality": "Institutional Maternal Death Rate (%)",
            
            # Stillbirth (with typos)
            "stillbirth": "Stillbirth Rate (%)",
            "stil birth": "Stillbirth Rate (%)",
            "stillbrith": "Stillbirth Rate (%)",
            
            # PPH (with typos)
            "pph": "Postpartum Hemorrhage (PPH) Rate (%)",
            "hemorrhage": "Postpartum Hemorrhage (PPH) Rate (%)",
            "hemorage": "Postpartum Hemorrhage (PPH) Rate (%)",
            "hemorrage": "Postpartum Hemorrhage (PPH) Rate (%)",
            "bleeding": "Postpartum Hemorrhage (PPH) Rate (%)",
            "postpartum hemorrhage": "Postpartum Hemorrhage (PPH) Rate (%)",
            
            # Uterotonic (with typos)
            "uterotonic": "Delivered women who received uterotonic (%)",
            "uterotonc": "Delivered women who received uterotonic (%)",
            "utertonic": "Delivered women who received uterotonic (%)",
            "oxytocin": "Delivered women who received uterotonic (%)",
            
            # IPPCAR
            "ippcar": "Immediate Postpartum Contraceptive Acceptance Rate (IPPCAR %)",
            "contraceptive": "Immediate Postpartum Contraceptive Acceptance Rate (IPPCAR %)",
            "family planning": "Immediate Postpartum Contraceptive Acceptance Rate (IPPCAR %)",
            "fp": "Immediate Postpartum Contraceptive Acceptance Rate (IPPCAR %)",
            
            # PNC
            "pnc": "Early Postnatal Care (PNC) Coverage (%)",
            "postnatal": "Early Postnatal Care (PNC) Coverage (%)",
            "post natal": "Early Postnatal Care (PNC) Coverage (%)",
            
            # ARV
            "arv": "ARV Prophylaxis Rate (%)",
            "antiretroviral": "ARV Prophylaxis Rate (%)",
            "hiv prophylaxis": "ARV Prophylaxis Rate (%)",
            
            # Assisted Delivery
            "assisted delivery": "Assisted Delivery Rate (%)",
            "assisted": "Assisted Delivery Rate (%)",
            "instrumental": "Assisted Delivery Rate (%)",
            "vacuum": "Assisted Delivery Rate (%)",
            "forceps": "Assisted Delivery Rate (%)",
            
            # SVD
            "svd": "Normal Vaginal Delivery (SVD) Rate (%)",
            "vaginal delivery": "Normal Vaginal Delivery (SVD) Rate (%)",
            "normal delivery": "Normal Vaginal Delivery (SVD) Rate (%)",
            "spontaneous": "Normal Vaginal Delivery (SVD) Rate (%)",
            "vaginal": "Normal Vaginal Delivery (SVD) Rate (%)",
            
            # Episiotomy (with typos)
            "episiotomy": "Episiotomy Rate (%)",
            "episotomy": "Episiotomy Rate (%)",
            "episotomi": "Episiotomy Rate (%)",
            "episiotmoy": "Episiotomy Rate (%)",
            "episo": "Episiotomy Rate (%)",
            
            # Antepartum (with typos)
            "antepartum": "Antepartum Complications Rate (%)",
            "ante partum": "Antepartum Complications Rate (%)",
            "antipartum": "Antepartum Complications Rate (%)",
            "antenatal complications": "Antepartum Complications Rate (%)",
            "antenatal": "Antepartum Complications Rate (%)",
            
            # Data Quality / Counts
            "missing mode": "Missing Mode of Delivery",
            "missing birth": "Missing Birth Outcome",
            "birth outcome": "Missing Birth Outcome",
            "missing outcome": "Missing Birth Outcome",
            "missing condition": "Missing Condition of Discharge",
            "missing discharge": "Missing Condition of Discharge",
            
            # Admitted Mothers
            "admitted mothers": "Admitted Mothers",
            "admitted": "Admitted Mothers",
            "admissions": "Admitted Mothers",
            "admission": "Admitted Mothers",
            "total mothers": "Admitted Mothers",
            "enrollment": "Admitted Mothers",
            "total enrollments": "Admitted Mothers",
            "mothers": "Admitted Mothers",
            
            # Total Deliveries
            "total deliveries": "Total Deliveries",
            "deliveries": "Total Deliveries",
            "births": "Total Deliveries",
            
            # Newborn indicators (with typos)
            "inborn": "Inborn Rate (%)",
            "outborn": "Outborn Rate (%)",
            "neonatal death": "Neonatal Mortality Rate (%)",
            "neonatal mortality": "Neonatal Mortality Rate (%)",
            "nmr": "Neonatal Mortality Rate (%)",
            "admitted newborns": "Admitted Newborns",
            "newborn admissions": "Admitted Newborns",
            "newborn admission": "Admitted Newborns",
            
            # KMC (with typos)
            "kmc": "KMC Coverage by Birth Weight",
            "kangaroo": "KMC Coverage by Birth Weight",
            "kangaro": "KMC Coverage by Birth Weight",
            "skin to skin": "KMC Coverage by Birth Weight",
            
            # CPAP (with typos)
            "cpap": "General CPAP Coverage",
            "c pap": "General CPAP Coverage",
            "rds": "CPAP for RDS",
            "respiratory distress": "CPAP for RDS",
            
            # Hypothermia (with typos)
            "hypothermia": "Hypothermia on Admission Rate (%)",
            "hypthermia": "Hypothermia on Admission Rate (%)",
            "hipothermia": "Hypothermia on Admission Rate (%)",
            "hypo thermia": "Hypothermia on Admission Rate (%)",

            # Birth Weight & Other Newborn (with typos)
            "birth weight": "Birth Weight Rate",
            "birthweight": "Birth Weight Rate",
            "birht weight": "Birth Weight Rate",
            "bw": "Birth Weight Rate",
            "kmc": "KMC Coverage by Birth Weight",
            "kangaroo": "KMC Coverage by Birth Weight",
            "cpap": "General CPAP Coverage",
            "c-pap": "General CPAP Coverage",
            "rds": "CPAP for RDS",
            "respiratory distress": "CPAP for RDS",
        }
        
        # Stop Word Removal for scanning
        stop_words = ["what", "is", "the", "are", "of", "in", "show", "me", "tell", "about", "rate", "value", "number", "total", "how", "many"]
        query_words = query_norm.split()
        filtered_words = [w for w in query_words if w not in stop_words]
        filtered_query = " ".join(filtered_words)

        # Check for direct containment first (using filtered query for better precision)
        # But we must check against 'query_norm' too because some maps have multiple words
        for key, val in kpi_map.items():
            if key in query_norm:
                selected_kpi = val
                break
        
        # If not found, try fuzzy matching on filtered words
        if not selected_kpi and filtered_query:
            keys = list(kpi_map.keys())
            matches = difflib.get_close_matches(filtered_query, keys, n=1, cutoff=0.6) # Increased cutoff
            if matches:
                 selected_kpi = kpi_map[matches[0]]
            
            # Sliding window backup
            if not selected_kpi:
                 if "eciton" in query_lower or "c -s" in query_lower:
                     selected_kpi = "C-Section Rate (%)"
                     
                     
        if not selected_kpi:
            # Fallback: check exact strings in KPI_OPTIONS (ignoring case)
            for kpi in KPI_OPTIONS:
                if kpi.lower() in query_lower:
                    selected_kpi = kpi
                    break
                    
        # Force Bar Chart for Counts
        if selected_kpi == "Admitted Mothers" and chart_type == "line":
            chart_type = "bar"
        
        # 3. Detect Facility
        selected_facility_uids = []
        selected_facility_names = []
        
        # Check against available facilities
        for name, uid in self.facility_mapping.items():
            n_lower = name.lower()
            # 1. Full name in query
            if n_lower in query_norm:
                selected_facility_uids.append(uid)
                selected_facility_names.append(name)
                continue
            
            # 2. Robust Partial Match (starts with specific word in query)
            # Use filtered_words to avoid common stopwords
            for word in filtered_words:
                if len(word) > 3 and n_lower.startswith(word):
                    selected_facility_uids.append(uid)
                    selected_facility_names.append(name)
                    break
        
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
            
            # Fuzzy match if none found (only try to find one primary if none explicit)
            if not found_regions:
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
                if comparison_mode and comparison_entity == "region":
                     # We will use 'comparison_targets' to store the list
                     pass 
                else:
                     # Standard mode - treat as aggregation filter (only if NOT comparison)
                     pass

                # Collect UIDs from ALL found regions (ALWAYS collect for data fetching)
                all_uids = []
                all_names = []
                for r_name in found_regions:
                    f_list = regions_data[r_name]
                    all_uids.extend([f[1] for f in f_list])
                    all_names.append(f"{r_name} (Region)")
                
                selected_facility_uids = all_uids
                selected_facility_names = all_names

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
            if not selected_facility_uids:
                if self.user.get("role") == "facility":
                    # Default to user's facility
                    selected_facility_uids = list(self.facility_mapping.values())
                    selected_facility_names = list(self.facility_mapping.keys())
                # For regional/national, if no specific facility, we might mean "overall" or "all"
        
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

        # NEW: Handle "this year" and "last year" FIRST (before month/week)
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

        # Fallback Strict Regex Date Parsing
        if not start_date:
            try:
                # 1. "Month DD, YYYY" or "Month DD YYYY" ranges (Explicit 2 years)
                month_pattern = r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*'
                range_pattern = re.compile(f"({month_pattern})[\s,]+(\d{{1,2}})[\s,]+(\d{{4}})\s*(?:to|-)?\s*({month_pattern})[\s,]+(\d{{1,2}})[\s,]+(\d{{4}})", re.IGNORECASE)
                matches = range_pattern.search(query)
                
                if matches:
                    m1, d1, y1, m2, d2, y2 = matches.groups()
                    start_date = datetime.strptime(f"{m1[:3]} {d1} {y1}", "%b %d %Y").strftime("%Y-%m-%d")
                    end_date = datetime.strptime(f"{m2[:3]} {d2} {y2}", "%b %d %Y").strftime("%Y-%m-%d")
            except Exception as e:
                logging.warning(f"Date regex 1 failed: {e}")

        # 2. Check for "Jan 1 - Jan 7 2026" (Year only at end)
        if not start_date:
             try:
                 month_pattern = r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*'
                 range_pattern_end_year = re.compile(f"({month_pattern})[\s,]+(\d{{1,2}})\s*(?:to|-)?\s*({month_pattern})[\s,]+(\d{{1,2}})[\s,]+(\d{{4}})", re.IGNORECASE)
                 range_match = range_pattern_end_year.search(query)
                 
                 if range_match:
                     m1, d1, m2, d2, y = range_match.groups()
                     start_date = datetime.strptime(f"{m1[:3]} {d1} {y}", "%b %d %Y").strftime("%Y-%m-%d")
                     end_date = datetime.strptime(f"{m2[:3]} {d2} {y}", "%b %d %Y").strftime("%Y-%m-%d")
             except Exception as e:
                 logging.warning(f"Date regex range pattern 2 failed: {e}")
        
        # 2b. NEW: Check for "Jan 1-5" (no year, assume current year)
        if not start_date:
             try:
                 month_pattern = r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*'
                 range_pattern_no_year = re.compile(f"({month_pattern})[\s,]+(\d{{1,2}})\s*-\s*(\d{{1,2}})", re.IGNORECASE)
                 range_match = range_pattern_no_year.search(query)
                 
                 if range_match:
                     m1, d1, d2 = range_match.groups()
                     current_year = today.year
                     start_date = datetime.strptime(f"{m1[:3]} {d1} {current_year}", "%b %d %Y").strftime("%Y-%m-%d")
                     end_date = datetime.strptime(f"{m1[:3]} {d2} {current_year}", "%b %d %Y").strftime("%Y-%m-%d")
             except Exception as e:
                 logging.warning(f"Date regex range pattern no year failed: {e}")

        # 3. Fallback to extracting ANY single dates with years
        if not start_date:
             try:
                single_date_pattern = re.compile(f"({month_pattern})[\s,]+(\d{{1,2}})[\s,]+(\d{{4}})", re.IGNORECASE)
                all_dates = single_date_pattern.findall(query)
                if len(all_dates) >= 2:
                    m1, d1, y1 = all_dates[0]
                    m2, d2, y2 = all_dates[1]
                    start_date = datetime.strptime(f"{m1[:3]} {d1} {y1}", "%b %d %Y").strftime("%Y-%m-%d")
                    end_date = datetime.strptime(f"{m2[:3]} {d2} {y2}", "%b %d %Y").strftime("%Y-%m-%d")
                elif len(all_dates) == 1:
                    m1, d1, y1 = all_dates[0]
                    start_date = datetime.strptime(f"{m1[:3]} {d1} {y1}", "%b %d %Y").strftime("%Y-%m-%d")
                    end_date = start_date 
             except Exception as e:
                logging.warning(f"Date regex 3 failed: {e}")

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
        # If user asks "plot", "show", "trend", assume plot.
        # If ambiguous, prefer text if specific date/value requested, plot if trend.
        if "what is" in query_lower or "value of" in query_lower:
             intent = "text"
        
        if "how is" in query_lower or "explain" in query_lower or "computation" in query_lower:
             intent = "explain"
             
        # Definition Detection
        if "what is" in query_lower or "define" in query_lower or "meaning of" in query_lower or "how many" in query_lower:
             # If user asks for "value", "rate", "count", "number", they want DATA, not definition.
             # Strict check: If KPI name is present, assume DATA intent not definition
             # Also exclude "total"
             data_keywords = ["value", "rate", "count", "number", "score", "percentage", "trend", "plot", "total"]
             
             # Check if ANY KPI from our mapping is in the query
             kpi_found = any(kpi.lower() in query_lower for kpi in KPI_MAPPING.keys())
             
             if any(x in query_lower for x in data_keywords) or kpi_found:
                 # It's likely a data query like "What is the total admitted mothers..."
                 # FORCE override even if LLM said metadata_query (common error for "how many")
                 if intent == "metadata_query" and kpi_found:
                      intent = "text"
                 elif intent == "metadata_query" and not kpi_found:
                      pass # Valid metadata query like "how many facilities"
                 else: 
                      intent = "text"
             
             elif "define" in query_lower or "meaning" in query_lower:
                 intent = "definition"

        # Robust List Detection
        if "indicator" in query_lower or "kpi" in query_lower:
            if any(x in query_lower for x in ["what", "list", "show", "available", "options", "help", "how many", "total"]):
                 # Clarification Check
                scope_keywords = ["maternal", "newborn", "all", "matenal", "materanl"] # added typos
                if not any(x in query_lower for x in scope_keywords):
                     return {
                         "intent": "chat",
                         "response": "Are you interested in **Maternal** or **Newborn** health indicators? (Currently I specialize in Maternal health!)"
                     }
                intent = "list_kpis"
        
        # Explicit handling for "maternal" answer to clarification
        if query_lower in ["maternal", "maternal indicators", "maternal health", "mothers", "matenal"]:
             intent = "list_kpis"
        if "options" in query_lower or "capabilities" in query_lower:
             intent = "list_kpis"
             
        # Newborn Scope Detection
        if "newborn" in query_lower:
            intent = "scope_error_newborn"

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
            
             
        # Metadata / Counts Detection Fallback (Regex)
        # ONLY if no KPI selected, otherwise assume data query
        if not selected_kpi and ("how many" in query_lower or "list" in query_lower or "show me" in query_lower):
             if "region" in query_lower:
                 intent = "metadata_query"
                 entity_type = "region"
                 count_requested = "how many" in query_lower
             elif "facilit" in query_lower or "hospital" in query_lower:
                  intent = "metadata_query"
                  entity_type = "facility"
                  count_requested = "how many" in query_lower

        # --- CONTEXT MERGING ---
        # If we have a stored context and current query is missing KPI or Facility, use context.
        context = st.session_state.get("chatbot_context", {})
        
        # Merge KPI
        # Prevent context merging if intent is list or scope error
        if intent not in ["list_kpis", "scope_error"] and not selected_kpi:
            if context.get("kpi"):
                selected_kpi = context.get("kpi")
                
        # Merge Facilities
        if not selected_facility_uids:
             # Only inherit facilities if the user didn't specify any NEW ones.
             if context.get("facility_uids"):
                 selected_facility_uids = context.get("facility_uids")
                 selected_facility_names = context.get("facility_names")
        
        # Merge Date Range
        if not start_date and context.get("date_range"):
            if reset_date:
                # User explicitly asked for overall, so we IGNORE context date
                pass
            else:
                # Only inherit if user didn't specify new date AND didn't ask to reset
                 pass # Logic below uses context.get("date_range")
        else:
             # If new date specified, use it.
             pass 
             
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
             # Force plot intent if we are comparing, unless user explicitly asks for "table" (which generate_response handles inside plot logic)
             if intent != "list_kpis" and intent != "scope_error":
                  intent = "plot"

        # Detect "By Facility" intent (Drill-down / Disaggregation)
        # Triggers if:
        # 1. "by facility" explicit phrase
        # 2. "facilities" mentioned along with a region in comparison mode (e.g. "Compare Tigray facilities")
        is_drill_down = "by facility" in query_lower or ("facilit" in query_lower and comparison_mode and found_regions)
        
        if is_drill_down:
             # Force Plot intent
             intent = "plot"
             
             # If we have found regions (e.g. "Tigray by facility"), we want to compare facilities within that region
             if found_regions:
                 comparison_mode = True
                 comparison_entity = "facility"
                 
                 # EXPAND the region into individual facilities for the comparison logic
                 # currently selected_facility_names might be ["Tigray (Region)"] -> We need ["Fac A", "Fac B"...]
                 new_names = []
                 new_uids = []
                 for r_name in found_regions:
                     facs = regions_data.get(r_name, [])
                     for f_name, f_uid in facs:
                         new_names.append(f_name)
                         new_uids.append(f_uid)
                 
                 # Limit expansion to prevent UI overload (e.g. max 50?)
                 if len(new_names) > 0:
                     selected_facility_names = new_names
                     selected_facility_uids = new_uids
             
             # If we already have selected facilities (e.g. "Adigrat and Abiadi by facility"), 
             # just ensure comparison mode is on so we see them side-by-side
             elif selected_facility_uids:
                 comparison_mode = True
                 comparison_entity = "facility" 

        final_date_range = {"start_date": start_date, "end_date": end_date} if start_date else (None if reset_date else context.get("date_range"))
        
        # Merge Entity Type (For "Name them" queries)
        if intent == "text" and not selected_kpi and not entity_type:
             pass 

        # Horizontal Chart Detection
        orientation = "v"
        if "horizontal" in query_lower:
            orientation = "h"
            if chart_type == "line": chart_type = "bar" # Line cannot be horizontal effectively usually
        
        # Auto-detect Granularity for Short Date Ranges
        if final_date_range and not period_label:
            try:
                s = datetime.strptime(final_date_range["start_date"], "%Y-%m-%d")
                e = datetime.strptime(final_date_range["end_date"], "%Y-%m-%d")
                delta = (e - s).days
                if delta <= 45: # If range is 1.5 months or less
                    period_label = "Daily"
            except:
                pass

        # Infer Comparison Entity if not explicitly set (e.g. "by facility" not used)
        if comparison_mode and not comparison_entity:
            if selected_facility_uids:
                comparison_entity = "facility"
            elif found_regions:
                # Prioritize Facility if both present? usually facility is more specific.
                # But if we found regions and NO facility UIDs, then region.
                comparison_entity = "region"

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
            "count_requested": count_requested,
            "comparison_mode": comparison_mode,
            "comparison_entity": comparison_entity,
            "comparison_targets": found_regions if comparison_mode and comparison_entity == "region" and found_regions else [],
            "response": None
        }

    def generate_response(self, query):
        global KPI_MAPPING, KPI_OPTIONS
        parsed = self.parse_query(query)
        
        # Handle List KPIs Intent (New)
        if parsed.get("intent") == "list_kpis":
            # Include both Maternal and Newborn
            from newborns_dashboard.dash_co_newborn import NEWBORN_KPI_MAPPING
            
            response = "Here are the available **Health Indicators** in this dashboard:\n\n"
            response += "🤰 **Maternal Indicators**:\n"
            for k in KPI_MAPPING.keys():
                response += f"- {k}\n"
            
            response += "\n👶 **Newborn Indicators**:\n"
            for k in NEWBORN_KPI_MAPPING.keys():
                response += f"- {k}\n"
            
            response += "\nYou can ask me to **plot** any of these or show their **stats**!"
            return None, response


        # Handle General Chat
        if parsed.get("intent") == "chat":
            if parsed.get("response"):
                return None, parsed.get("response")
            
            # Fallback for local detection (e.g. password)
            q_low = query.lower()
            if any(x in q_low for x in ["password", "login", "admin", "credential"]):
                return None, "I'm your Data Analytics Assistant. I don't handle system passwords or administrative access. Please contact your system administrator if you're having login issues! 🔒"
                
            role = self.user.get("role", "facility") # Default to facility if unknown
            return None, get_welcome_message(role)
            
        # Handle Chart Options Parsing
        if parsed.get("intent") == "chart_options":
             kpi_concern = parsed.get("kpi") or st.session_state.get("chatbot_context", {}).get("kpi")
             
             if kpi_concern == "Admitted Mothers":
                 return None, "For **Admitted Mothers**, the available charts are:\n- **Vertical Bar Chart** (Default)\n- **Horizontal Bar Chart** (Say 'plot horizontal bar')\n- **Data Table**"
             elif kpi_concern:
                 return None, f"For **{kpi_concern}**, I can generate:\n- **Line Chart**: Best for trends over time.\n- **Bar Chart**: Good for comparison.\n- **Area Chart**: Visualizes volume over time.\n- **Data Table**: Detailed numbers."
             else:
                 return None, "I can generate the following charts for any indicator:\n- **Line Chart** (Default): 'Plot PPH trend'\n- **Bar Chart**: 'Show Admitted Mothers as bar chart'\n- **Area Chart**: 'PPH Rate area chart'\n- **Data Table**: 'Show table for C-Section'"
            
        # Handle Scope Error
        if parsed.get("intent") == "scope_error":
             return None, "I'm focused on data analysis and visualization. I cannot change the dashboard's appearance or colors, but I can help you plot trends or find specific values."

        # Handle Hallucination Scope Error
        if parsed.get("intent") == "scope_error_hallucination":
             return None, "I detected a term (like 'temperature') that I don't track directly. I specialize in **Maternal** and **Newborn** health indicators.\n\nTry asking about 'Hypothermia' or 'KMC' if you are interested in thermal care, or say 'list indicators' to see what I can do."
             
        # Handle List KPIs
        if parsed.get("intent") == "list_kpis":
             # Check if asking for count only
             if "how many" in query_lower or "count" in query_lower:
                 msg = f"I currently have **{len(KPI_MAPPING)}** maternal health indicators available.\n\n"
                 msg += "Would you like me to list all of them? Just say 'yes' or 'list all indicators'."
                 # Store context for follow-up
                 st.session_state["chatbot_context"]["last_question"] = "indicators"
                 return None, msg
             else:
                 # List all indicators
                 kpi_list = "\n".join([f"- **{k}**" for k in KPI_MAPPING.keys()])
                 msg = f"Here are all the maternal health indicators I can provide information about:\n\n{kpi_list}"
                 return None, msg
             
        # Handle Newborn Scope Error
        if parsed.get("intent") == "scope_error_newborn":
            return None, "I now have access to **Newborn Care** data! Try asking about 'CPAP', 'KMC', 'NMR', or 'Birth Weights'."
        
        # Handle Clear Chat
        if parsed.get("intent") == "clear":
             st.session_state.messages = []
             st.session_state.chatbot_context = {} 
             st.session_state.messages.append({
                "role": "assistant",
                "content": "Hello! I'm your AI health assistant. You can ask me to plot trends like 'Plot C-Section Rate this month' or ask for specific values."
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
             return None, None # Should have returned above, but just in case
        
        if not parsed["kpi"]:
            # Smart response for out-of-scope queries
            msg = "I couldn't identify a specific health indicator in your question.\n\n"
            msg += "I currently provide information about **maternal health indicators**.\n\n"
            msg += "Would you like to see all available indicators? Just say 'show all indicators' or 'list indicators'."
            return None, msg

        # --- SPECIALIZATION CHECK: Restrict to Maternal Indicators for Plotting/Explaining ---
        # 1. Determine Data Source (Maternal vs Newborn)
        use_newborn_data = False
        kpi_name = parsed["kpi"]
        kpi_lower = kpi_name.lower()
        
        # Comprehensive check for newborn indicators
        newborn_indicators = [
            "inborn", "outborn", "neonatal", "newborn", "nmr", "kmc", "cpap", 
            "birth weight", "rds", "temperature", "missing temperature", 
            "missing birth weight", "missing discharge status"
        ]
        
        if any(x in kpi_lower for x in newborn_indicators):
             use_newborn_data = True
        
        # If user wants to PLOT or EXPLAIN a newborn indicator, intercept with specialization message
        if use_newborn_data and parsed["intent"] in ["plot", "explain"]:
             return None, "I am currently able to plot for **Maternal indicators**. You may say 'list indicators' to see what I can help you with."
             
        active_df = self.newborn_df if use_newborn_data else self.maternal_df
        
        # Update session state with correct collection
        st.session_state["chatbot_context"] = {
            "kpi": parsed["kpi"],
            "facility_uids": parsed["facility_uids"],
            "facility_names": parsed["facility_names"],
            "date_range": parsed["date_range"],
            "entity_type": parsed.get("entity_type"), # Persist for follow-up
            "source": "newborn" if use_newborn_data else "maternal"
        }
        
        # Apply Aggregation Filter
        if parsed.get("period_label"):
            if "filters" not in st.session_state: st.session_state.filters = {}
            st.session_state.filters["period_label"] = parsed["period_label"]

        # Prepare Filters
        kpi_name = parsed["kpi"]
        facility_uids = parsed["facility_uids"]
        date_range = parsed["date_range"]
        chart_type = parsed.get("chart_type", "line") # Safe get
        
        # Enforce Bar Chart constraint for Admitted Mothers if not specified otherwise
        if kpi_name == "Admitted Mothers" and chart_type == "line":
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

        # --- HANDLE EXPLAIN INTENT ---
        if parsed["intent"] == "explain":
            # Use KB_DEFINITIONS for explanation
            kpi_info = KPI_MAPPING.get(active_kpi_name, {})
            numerator_desc = kpi_info.get("numerator_name", "the numerator")
            denominator_desc = kpi_info.get("denominator_name", "the denominator")
            
            # Check if we have a detailed definition
            definition = getattr(self, "KB_DEFINITIONS", {}).get(active_kpi_name)
            
            if definition:
                explanation = f"**{kpi_name}** explanation:\n\n{definition}\n\n**Calculation**:\n- **Numerator**: {numerator_desc}\n- **Denominator**: {denominator_desc}"
            else:
                explanation = f"**{kpi_name}** explanation:\n- **Numerator**: {numerator_desc}\n- **Denominator**: {denominator_desc}"
            
            if target_component != "value":
                explanation += f"\n\n*Note: Since you asked for **{kpi_name}**, I'm extracting the **{target_component}** from this indicator.*"
                
            return None, explanation

        # --- HANDLE DEFINITION INTENT ---
        if parsed["intent"] == "definition":
            # Use KB_DEFINITIONS
            definition = getattr(self, "KB_DEFINITIONS", {}).get(active_kpi_name)
            
            if not definition:
                # Basic fuzzy search in KB keys
                for k, v in getattr(self, "KB_DEFINITIONS", {}).items():
                    if kpi_name.lower() in k.lower():
                        definition = v
                        break
            
            if definition:
                return None, f"**Definition of {active_kpi_name}**:\n\n{definition}"
            else:
                 # Suggest available indicators
                 return None, f"I don't have a definition for **{kpi_name}** yet. Try asking about indicators like 'PPH', 'C-Section', 'NMR', 'Episiotomy', or 'Admitted Mothers'."

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

        # If Intent is Plot OR Analysis is requested (to get trend data)
        if parsed["intent"] == "plot" or parsed.get("analysis_type"):
            # 2. Get Data for Active KPI (using SILENT PREPARE)
            if use_newborn_data:
                from newborns_dashboard.dash_co_newborn import prepare_data_for_newborn_trend_chart
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
                     return None, f"I found no data for **{kpi_name}** matching your criteria."
            
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
                if any(x in query_lower_check for x in ["all facilities", "all facility", "every facility", "all hospitals"]):
                    is_compare_all = True
                    comparison_entity = "facility"
                elif any(x in query_lower_check for x in ["all regions", "all region", "every region", "all reioings", "all reigons", "all reginos"]):
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
                         all_comparison_uids.append(r_name) 
                         
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
                
                # Force table-only rendering for "all" comparisons unless explicit plot request
                if is_compare_all:
                    explicit_plot_keywords = ["plot", "graph", "chart", "visualize", "trend"]
                    if not any(k in query.lower() for k in explicit_plot_keywords):
                        chart_type = "table"
                        suggestion = f"\n\n💡 *Showing comparison table for all {comparison_entity}s. Ask to 'plot' if you want a chart.*"
            
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
                elif not parsed.get("facility_names") and not parsed.get("comparison_targets") and "all facilities" in prompt.lower():
                     if self.user.get("role") in ["national", "regional", "admin"]:
                         st.session_state["filter_mode"] = "All Facilities"
                         st.session_state["selected_facilities"] = ["All Facilities"]
                         st.session_state["selected_kpi"] = active_kpi_name # Match dashboard to requested KPI
                         st.session_state["selection_applied"] = True
                         st.session_state["refresh_trigger"] = True
                         nav_feedback = "\n*(Dashboard reset to All Facilities)*"
                         
            except Exception as e:
                logging.warning(f"Navigation update failed: {e}")

            # Loop through Groups and Build Data
            chart_data = []
            
            for entity_name, entity_uids in comparison_groups:
                 # Re-fetch/Filter data for this entity
                 if use_newborn_data:
                      from newborns_dashboard.dash_co_newborn import prepare_data_for_newborn_trend_chart
                      entity_df, _ = prepare_data_for_newborn_trend_chart(active_df, active_kpi_name, entity_uids, date_range)
                 else:
                      entity_df, _ = self._silent_prepare_data(active_df, active_kpi_name, entity_uids, date_range)
                 
                 if entity_df is None or entity_df.empty:
                     continue

                 # Determine Period Order
                 if "period_display" not in entity_df.columns:
                     if "event_date" in entity_df.columns:
                        entity_df["period_display"] = entity_df["event_date"].dt.strftime("%b-%y").str.capitalize()
                        entity_df["period_sort"] = entity_df["event_date"].dt.to_period("M").dt.start_time
                 
                 grouped = entity_df.groupby("period_display")
                 
                 # Sort groups by period_sort
                 time_groups = []
                 for name, group in grouped: 
                     # Get sort value from first row
                     sort_val = group['period_sort'].iloc[0] if 'period_sort' in group.columns else (group['event_date'].min() if 'event_date' in group.columns else datetime.min)
                     time_groups.append((name, group, sort_val))
                 
                 # Sort by the extracted sort_val
                 time_groups.sort(key=lambda x: x[2])

                 for period_name, group_df, sort_val in time_groups:
                     # Generic KPI calculation fallback
                     kpi_suffix = self.SPECIALIZED_KPI_MAP.get(active_kpi_name)
                     
                     if use_newborn_data:
                         from newborns_dashboard.dash_co_newborn import get_numerator_denominator_for_newborn_kpi_with_all
                         numerator, denominator, value = get_numerator_denominator_for_newborn_kpi_with_all(group_df, active_kpi_name, entity_uids, date_range)
                     elif kpi_suffix and kpi_suffix != "utils":
                         try:
                             module = __import__(f"utils.kpi_{kpi_suffix}", fromlist=[f"get_numerator_denominator_for_{kpi_suffix}"])
                             get_nd_func = getattr(module, f"get_numerator_denominator_for_{kpi_suffix}")
                             numerator, denominator, value = get_nd_func(group_df, entity_uids, date_range)
                         except Exception as e:
                             logging.error(f"Failed to call specialized function for {active_kpi_name}: {e}")
                             numerator, denominator, value = kpi_utils.get_numerator_denominator_for_kpi(group_df, active_kpi_name, entity_uids, date_range)
                     else:
                         # Use standard kpi_utils for "utils" suffix or no suffix
                         numerator, denominator, value = kpi_utils.get_numerator_denominator_for_kpi(group_df, active_kpi_name, entity_uids, date_range)
                     
                     # Resolve Component
                     plot_value = value
                     if target_component == "numerator": plot_value = numerator
                     elif target_component == "denominator": plot_value = denominator
                     
                     chart_data.append({
                         "Period": period_name,
                         "Entity": entity_name,
                         "Value": plot_value,
                         "Numerator": numerator,
                         "Denominator": denominator,
                         "SortDate": sort_val,
                         "orgUnit": entity_uids[0] if entity_uids else None
                     })
            
            plot_df = pd.DataFrame(chart_data)
            if not plot_df.empty and "SortDate" in plot_df.columns:
                plot_df.sort_values("SortDate", inplace=True)
            
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
                            "General CPAP Coverage": "render_cpap_general",
                            "CPAP for RDS": "render_cpap_rds",
                            "CPAP Coverage by Birth Weight": "render_cpap_by_weight"
                        }
                        func_prefix = category_map.get(active_kpi_name, "render_birth_weight")
                    else:
                        func_prefix = f"render_{kpi_suffix}"
                    
                    # Prepare DF for specialized scripts (without Overall row)
                    render_df = render_plot_df.copy()
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
                    
                    # Get labels for kpi_utils functions
                    kpi_info = KPI_MAPPING.get(active_kpi_name, {})
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
                            "facility_names": parsed.get("facility_names"),
                            "facility_uids": facility_uids,
                            "all_comparison_uids": all_comparison_uids,
                            "num_label": num_label,
                            "den_label": den_label
                        },
                        "data": render_df
                    }
                    
                    return spec, f"I've rendered the specialized dashboard visualization for **{kpi_name}**.{nav_feedback}"
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
                         return pivot_df, f"Here is the comparison table for **{kpi_name}**."
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

                return plot_df, f"Here is the data table for **{kpi_name}**."
                
            elif chart_type == "bar":
                if parsed.get("orientation") == "h":
                    fig = px.bar(plot_df, x="Value", y="Period", color=color_col,  title=f"{kpi_name}", height=400, orientation='h', barmode='group', hover_data=["Numerator", "Denominator"])
                else:
                    fig = px.bar(plot_df, x="Period", y="Value", color=color_col, title=f"{kpi_name}", height=400, barmode='group', hover_data=["Numerator", "Denominator"])
            elif chart_type == "area":
                fig = px.area(plot_df, x="Period", y="Value", color=color_col, title=f"{kpi_name}", markers=True, height=400, hover_data=["Numerator", "Denominator"])
            else: # Default to line
                fig = px.line(plot_df, x="Period", y="Value", color=color_col, title=f"{kpi_name}", markers=True, height=400, hover_data=["Numerator", "Denominator"])
                
            fig.update_traces(marker=dict(size=8)) if chart_type != "bar" else None
            
            # Refine title
            title_text = f"{kpi_name}"
            if parsed["facility_names"]:
                title_text += f" - {', '.join(parsed['facility_names'][:2])}"
                if len(parsed['facility_names']) > 2: title_text += "..."
            fig.update_layout(title_text=title_text, margin=dict(l=20, r=20, t=40, b=20))
            
            return fig, f"Here is the {chart_type} chart for **{kpi_name}**.{suggestion}{nav_feedback}"
            
        else:
            # Text Response (Single value)
            if use_newborn_data:
                 from newborns_dashboard.dash_co_newborn import prepare_data_for_newborn_trend_chart
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
            
            # Resolve Component
            display_value = value
            if target_component == "numerator": display_value = numerator
            elif target_component == "denominator": display_value = denominator
            
            # Format display logic
            if active_kpi_name == "Admitted Mothers":
                response_text = f"The **Total Admitted Mothers** count is **{int(display_value):,}**"
            elif target_component == "value":
                response_text = f"The **{kpi_name}** is **{display_value:.2f}%**"
            else:
                response_text = f"The **{kpi_name}** count is **{int(display_value):,}**"

            if parsed["facility_names"]:
                response_text += f" for *{', '.join(parsed['facility_names'])}*"
            if date_range:
                response_text += f" during the period {date_range['start_date']} to {date_range['end_date']}."
            else:
                response_text += "."
            
            if target_component == "value":
                response_text += f"\n\n(Based on {int(numerator)} cases out of {int(denominator)})"
            
            return None, response_text + suggestion + nav_feedback


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
        # Dynamic Welcome Message
        user_role = st.session_state.get("user", {}).get("role", "national")
        welcome_msg = get_welcome_message(user_role)
        st.session_state.messages.append({
            "role": "assistant",
            "content": welcome_msg
        })

    # SIDEBAR: Clear Chat Button - ADDED KEY
    if st.sidebar.button("🗑️ Clear Chat History", key="clear_chat_history_btn"):
         st.session_state.messages = []
         st.session_state.chatbot_context = {}
         st.rerun()

    st.markdown('<div class="main-chat-container">', unsafe_allow_html=True)
    st.markdown('<h1 class="chat-header">🤖 IMNID AI Assistant</h1>', unsafe_allow_html=True)
    
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

    chatbot_logic = ChatbotLogic(data)
    
    # Inject KB_DEFINITIONS for Definition Intent
    chatbot_logic.KB_DEFINITIONS = {
        "C-Section Rate (%)": "A Caesarean section (C-section) is a surgical procedure used to deliver a baby through incisions in the abdomen and uterus. The rate is the percentage of deliveries performed via C-section out of total deliveries.",
        "Postpartum Hemorrhage (PPH) Rate (%)": "Postpartum Hemorrhage (PPH) is defined as excessive bleeding after childbirth (usually >500ml for vaginal, >1000ml for C-section). It is a leading cause of maternal mortality.",
        "Institutional Maternal Death Rate (%)": "Maternal death refers to the death of a woman while pregnant or within 42 days of termination of pregnancy, irrespective of the duration and site of the pregnancy, from any cause related to or aggravated by the pregnancy or its management but not from accidental or incidental causes.",
        "Stillbirth Rate (%)": "A stillbirth is the death or loss of a baby before or during delivery. The rate typically measures stillbirths per 1,000 total births, but here it is presented as a percentage.",
        "Early Postnatal Care (PNC) Coverage (%)": "Early Postnatal Care refers to the medical care given to the mother and newborn within the first 24-48 hours after delivery, crucial for detecting complications.",
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
                if comparison_entity == "facility":
                    if suffix == "newborn_simplified":
                         func_name = f"{func_prefix}_facility_comparison"
                    else:
                         func_name = f"{func_prefix}_facility_comparison_chart"
                         
                    render_func = getattr(module, func_name)
                    
                    if suffix == "utils" or suffix == "newborn":
                        render_func(render_df, "period_display", "value", params["active_kpi_name"], "#FFFFFF", None, params["facility_names"], params["facility_uids"], params["num_label"], params["den_label"], suppress_plot=suppress_plot, key_suffix=f"msg_{message_index}")
                    elif suffix == "newborn_simplified":
                        render_func(render_df, facility_uids=params["all_comparison_uids"], facility_names=params["facility_names"], key_suffix=f"msg_{message_index}")
                    elif suffix == "admitted_mothers":
                        # admitted_mothers has a different signature (no num/den labels)
                        render_func(render_df, "period_display", "value", params["active_kpi_name"], "#FFFFFF", None, params["facility_names"], params["facility_uids"], suppress_plot=suppress_plot, key_suffix=f"msg_{message_index}")
                    else:
                        # All other specialized maternal modules (pph, svd, arv, etc.) use the same signature as utils
                        render_func(render_df, "period_display", "value", params["active_kpi_name"], "#FFFFFF", None, params["facility_names"], params["facility_uids"], params["num_label"], params["den_label"], suppress_plot=suppress_plot, key_suffix=f"msg_{message_index}")
                else:
                    if suffix == "newborn_simplified":
                         func_name = f"{func_prefix}_region_comparison"
                    else:
                         func_name = f"{func_prefix}_region_comparison_chart"
                         
                    render_func = getattr(module, func_name)
                    
                    if suffix == "utils":
                        from utils.queries import get_facilities_grouped_by_region
                        regions_mapping = get_facilities_grouped_by_region(st.session_state.get("user", {}))
                        render_func(render_df, "period_display", "value", params["active_kpi_name"], "#FFFFFF", None, params["all_comparison_uids"], regions_mapping, regions_mapping, params["num_label"], params["den_label"], suppress_plot=suppress_plot, key_suffix=f"msg_{message_index}")
                    elif suffix == "newborn":
                        regions_mapping = {} # Fallback
                        render_func(render_df, "period_display", "value", params["active_kpi_name"], "#FFFFFF", None, params["all_comparison_uids"], regions_mapping, params["all_comparison_uids"], params["num_label"], params["den_label"], suppress_plot=suppress_plot, key_suffix=f"msg_{message_index}")
                    elif suffix == "newborn_simplified":
                        render_func(render_df, region_names=params["all_comparison_uids"])
                    elif suffix == "admitted_mothers":
                        # admitted_mothers has a different signature
                        render_func(render_df, "period_display", "value", params["active_kpi_name"], "#FFFFFF", None, params["all_comparison_uids"], suppress_plot=suppress_plot, key_suffix=f"msg_{message_index}")
                    else:
                        # All other specialized maternal modules use the same signature as utils
                        from utils.queries import get_facilities_grouped_by_region
                        regions_mapping = get_facilities_grouped_by_region(st.session_state.get("user", {}))
                        render_func(render_df, "period_display", "value", params["active_kpi_name"], "#FFFFFF", None, params["all_comparison_uids"], regions_mapping, regions_mapping, params["num_label"], params["den_label"], suppress_plot=suppress_plot, key_suffix=f"msg_{message_index}")
            else:
                render_func = getattr(module, f"{func_prefix}_trend_chart")
                if suffix == "utils" or suffix == "newborn":
                    render_func(render_df, "period_display", "value", params["active_kpi_name"], "#FFFFFF", None, params["facility_names"], params["num_label"], params["den_label"], facility_uids=[f"msg_{message_index}"], key_suffix=f"msg_{message_index}")
                elif suffix == "newborn_simplified":
                    render_func(render_df, period_col="period_display", title=params["active_kpi_name"], facility_uids=params.get("facility_uids"))
                elif suffix == "admitted_mothers":
                    # admitted_mothers has a different signature
                    render_func(render_df, "period_display", "value", params["active_kpi_name"], "#FFFFFF", None, params["facility_names"], facility_uids=[f"msg_{message_index}"], key_suffix=f"msg_{message_index}")
                else:
                    # All other specialized maternal modules (pph, svd, arv, etc.) use the same signature as utils
                    render_func(render_df, "period_display", "value", params["active_kpi_name"], "#FFFFFF", None, params["facility_names"], params["num_label"], params["den_label"], facility_uids=[f"msg_{message_index}"], key_suffix=f"msg_{message_index}")
        except Exception as e:
            st.error(f"Error rendering specialized content: {e}")

    # Display chat messages from history on app rerun
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
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
                     st.plotly_chart(message["figure"], use_container_width=True, key=f"chat_chart_{i}")

    # Accept user input
    if prompt := st.chat_input("Ask about KPIs (e.g. 'Plot PPH rate for Facility A')..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            with st.spinner("Analyzing data..."):
                # Run Logic
                try:
                    # Check for help intent explicitly here or rely on chatbot_logic
                    if prompt.lower().strip() in ["help", "info", "usage"]:
                         user_role = st.session_state.get("user", {}).get("role", "national")
                         response_text = get_welcome_message(user_role)
                         fig = None
                    else:
                        # Rerurns (fig, text)
                        fig, response_text = chatbot_logic.generate_response(prompt)
                    
                    message_placeholder.markdown(response_text)
                    
                    msg_obj = {"role": "assistant", "content": response_text}
                    
                    if fig is not None:
                        if isinstance(fig, dict) and fig.get("type") == "specialized":
                            # Execute Specialized Rendering
                            execute_specialized(fig)
                            # Save Spec to history
                            msg_obj["specialized_spec"] = fig
                        elif isinstance(fig, pd.DataFrame):
                            st.dataframe(fig)
                            msg_obj["figure"] = fig
                        else:
                            st.plotly_chart(fig, use_container_width=True, key=f"chat_chart_new_{len(st.session_state.messages)}")
                            msg_obj["figure"] = fig
                        
                    # Save to history
                    st.session_state.messages.append(msg_obj)
                    
                    if response_text == "Chat history cleared.":
                        st.rerun()
                        
                except Exception as e:
                    logging.error(f"Chatbot Error: {e}", exc_info=True)
                    
                    # Provide smart error messages
                    error_type = type(e).__name__
                    if "KeyError" in error_type:
                        error_msg = "I couldn't find the requested data field. This indicator might not be available for your selection. Try asking about a different indicator or time period."
                    elif "AttributeError" in error_type:
                        error_msg = "I had trouble processing this request. The data structure doesn't support this query. Try rephrasing your question."
                    else:
                        error_msg = f"I encountered an issue: {str(e)}\n\nTry asking in a different way or about a different indicator."
                    
                    message_placeholder.markdown(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

    st.markdown('</div>', unsafe_allow_html=True)


def get_welcome_message(role):
    """Generates a dynamic welcome message based on user role."""
    
    # Dashboard List
    dashboards = ["Maternal", "Newborn", "Summary", "Resources"] # Default/Facility
    if role != "facility":
        dashboards = ["Maternal", "Newborn", "Summary", "Mentorship", "Resources", "Data Quality"]
        
    dashboard_str = ", ".join(dashboards)
    
    msg = f"""**Hello! I'm your IMNID AI Assistant.** 🤖
    
I can help you analyze data across the **{dashboard_str}** dashboards.

**What I can do for you:**
- **📊 Plot Charts**: I can generate Line, Bar, and Area charts for Maternal Health indicators (Newborn coming soon).
- **🔢 Quick Values**: Ask for a specific value (e.g. "What is the PPH rate?") and I'll provide the latest figure.
- **🗺️ Comparisons**: Ask to compare facilities or regions! (e.g., "Compare Admitted Mothers for Adigrat and Suhul")
- **📚 Definitions**: Ask "What is [Indicator]?" to get a medical definition.
- **📥 Data Tables**: Ask for "table format" to see the raw numbers.

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