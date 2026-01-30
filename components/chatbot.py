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
    Optimized data loading that REUSES dashboard's cached data.
    """
    user = st.session_state.get("user", {})
    role = user.get("role", "")
    
    if not role:
        return None
    
    # TRY TO USE DASHBOARD'S ALREADY-LOADED DATA FIRST
    if role == "facility":
        # Check if facility dashboard has already loaded data
        if hasattr(st.session_state, "cached_shared_data_facility"):
            return st.session_state.cached_shared_data_facility
        
        # Also check if maternal_patients_df exists (dashboard's filtered data)
        if hasattr(st.session_state, "maternal_patients_df") and not st.session_state.maternal_patients_df.empty:
            logging.info("✅ Chatbot: Using dashboard's already-loaded facility data")
            return {"maternal": {"patients": st.session_state.maternal_patients_df}}
    
    elif role == "regional":
        if hasattr(st.session_state, "cached_shared_data_regional"):
            return st.session_state.cached_shared_data_regional
        
        # Use regional dashboard's data if available
        if hasattr(st.session_state, "regional_patients_df") and not st.session_state.regional_patients_df.empty:
            logging.info("✅ Chatbot: Using dashboard's already-loaded regional data")
            return {"maternal": {"patients": st.session_state.regional_patients_df}}
    
    elif role == "national":
        if hasattr(st.session_state, "cached_shared_data_national"):
            return st.session_state.cached_shared_data_national
        
        # Use national dashboard's data if available
        if hasattr(st.session_state, "maternal_patients_df") and not st.session_state.maternal_patients_df.empty:
            logging.info("✅ Chatbot: Using dashboard's already-loaded national data")
            return {"maternal": {"patients": st.session_state.maternal_patients_df}}
        elif hasattr(st.session_state, "cached_shared_data"):
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
            "sitllbirth": "stillbirth",
            "sitllbirht": "stillbirth",
            "stillbirht": "stillbirth",
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

        # 4. Greedy Scan List (Sorted by length to catch longest phrases first)
        # We use keys from suffix_blind_mapping because it contains BOTH full names AND short names
        self.sorted_scan_names = sorted(self.suffix_blind_mapping.keys(), key=len, reverse=True)
        
        # 5. Ambiguous Prefixes (Terms that are too generic to match directly if found alone)
        self.AMBIGUOUS_PREFIXES = ["ambo", "debre", "st", "saint", "mary", "kidane", "black", "lion", "felege", "gandhi", "yekatit", "alert", "menelik", "paul", "peter"]
        
        # 6. Add special handling for full ambiguous facility names with their UIDs
        for full_name in self.AMBIGUOUS_FACILITY_UIDS.keys():
            norm_name = re.sub(r'\s+', ' ', full_name).strip().lower()
            if norm_name not in self.suffix_blind_mapping:
                self.suffix_blind_mapping[norm_name] = full_name
        
        logging.info(f"🏥 Facility Resolution Engine Initialized: {len(self.normalized_facility_mapping)} direct, {len(self.suffix_blind_mapping)} suffix-blind, {len(self.facility_search_index)} index groups.")


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
            "Postpartum Complications Rate (%)": "postpartum_compl",
            
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
            "Maternal Death Rate (per 100,000)": "utils",
            "Stillbirth Rate (%)": "utils",
            "Early Postnatal Care (PNC) Coverage (%)": "utils",
            "Immediate Postpartum Contraceptive Acceptance Rate (IPPCAR %)": "utils"
        }

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
        
        # --- PRE-PROCESS QUERY (Normalization & Typo Correction) ---
        # Move this to the top so it filters early keyword checks
        query_norm = re.sub(r'[^a-z0-9\s]', '', query_lower)
        for typo, correct in self.COMMON_TYPOS.items():
            query_norm = query_norm.replace(typo, correct)
        query_norm = re.sub(r'\s+', ' ', query_norm).strip()
        
        
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
                
            if selection in ambiguity_options:
                selected_name = ambiguity_options[selection]
                # Clear ambiguity options
                st.session_state.chatbot_context["ambiguity_options"] = None
                
                # Resolving to the selected facility
                return {
                     "intent": "plot",
                     "kpi": context.get("pending_kpi"),
                     "facility_uids": [self.universal_facility_mapping[selected_name]],
                     "facility_names": [selected_name],
                     "fulfillment_requested": True
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
        # PRIORITIZE Facility keyword to avoid "list facilities in Tigray region" matching regions regex first
        entity_type = None
        count_requested = False
        
        if re.search(r'(list|show|shw|sho|display|tell).*(facilities|facility|hospitals?|facilti?yies?|faclities?|units?)', query_norm):
            entity_type = "facility"
        elif re.search(r'(list|show|shw|sho|display|tell).*(regions?|reioings?|reigons?|territory|territories)', query_norm):
            entity_type = "region"
            
        # 0.5. PRIORITY CHECK: If KPI is present, do NOT treat as metadata list
        # "Show C-Section Rate by facility" -> PLOT intent, not LIST intent
        kpi_found_early = any(kpi.lower() in query_norm for kpi in KPI_MAPPING.keys())
        
        if entity_type and not kpi_found_early:
            # Extract Region if mentioned (even in rule-based mode)
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
        
        # --- GREEDY FACILITY SCAN (Prioritize Full Names in Query) ---
        # Instead of just relying on the LLM, scan the RAW query for any known facility names

        # RE-IMPL GREEDY SCAN with Ambiguity Check
        greedy_matches = []
        q_norm_for_scan = re.sub(r'\s+', ' ', query_lower).strip()
        
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
                        
                        options_str = "\n".join([f"{i}. {m}" for i, m in options_map.items()])
                        return {
                              "intent": "chat",
                              "response": f"Which **{norm_name.capitalize()}** do you mean?\n\n{options_str}",
                              "pending_kpi": None
                         }
                # AMBIGUITY CHECK END
                
                official_name = self.suffix_blind_mapping[norm_name]
                uid = self.universal_facility_mapping[official_name]
                greedy_matches.append((official_name, uid))
                q_norm_for_scan = q_norm_for_scan.replace(norm_name, " [MATCHED] ")

        if greedy_matches:
            logging.info(f"🚀 Greedy Scan Found: {[m[0] for m in greedy_matches]}")

        # 0. Try LLM Parsing - DISABLED BY USER REQUEST
        # from utils.llm_utils import query_llm
        llm_result = None # Force fallback to rule-based logic
        
        if llm_result:
                # ... (existing LLM logic remains the same)
                pass

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
        elif any(w in query_lower for w in ["plot", "graph", "chart", "trend", "visualize", "show me"]):
            intent = "plot"
        elif any(w in query_lower for w in ["define", "meaning", "definition", "how is", "computed", "calculation", "formula"]):
            intent = "definition"
            
        # INTEGRATE GREEDY MATCHES
        selected_facility_uids = []
        selected_facility_names = []
        if greedy_matches:
            for name, uid in greedy_matches:
                 selected_facility_names.append(name)
                 selected_facility_uids.append(uid)
                 
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
            "maternal death": "Maternal Death Rate (per 100,000)",
            "death": "Maternal Death Rate (per 100,000)",
            "mortality": "Maternal Death Rate (per 100,000)",
            
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
            "family plainnig": "Immediate Postpartum Contraceptive Acceptance Rate (IPPCAR %)",
            "fp": "Immediate Postpartum Contraceptive Acceptance Rate (IPPCAR %)",
            "contraception": "Immediate Postpartum Contraceptive Acceptance Rate (IPPCAR %)",
            
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
            "ante": "Antepartum Complications Rate (%)",
            "anit": "Antepartum Complications Rate (%)",
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
            for kpi in KPI_OPTIONS:
                if kpi.lower() in query_lower:
                    selected_kpi = kpi
                    break
                    
        # Force Bar Chart for Counts
        if selected_kpi == "Admitted Mothers" and chart_type == "line":
            chart_type = "bar"
        
        # 3. Detect Facility - CRITICAL FIX: Check if facility name exists
        selected_facility_uids = []
        selected_facility_names = []
        
        # First, check if user mentioned a facility in the query
        facility_mentioned = any(word in query_lower for word in ["facility", "hospital", "clinic", "center", "health"])
        
        # Refined guard: check for "for ", "at ", "in ", "from " 
        # Refined guard: check for "for ", "at ", "in ", "from ", "i "
        # but EXCLUDE common time-based phrases to avoid "For this year" being caught as a facility request
        time_triggers = ["year", "month", "week", "today", "yesterday", "overall", "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec", "202", "201"]
        is_time_query = any(t in query_lower for t in time_triggers)
        
        # Use regex with word boundaries to avoid matching substrings like "at" in "what"
        specific_location_regex = r'\b(at|from)\b'
        specific_facility_requested = bool(re.search(specific_location_regex, query_lower))
        
        # "for", "in", "i " are common, only treat as facility trigger if NO time word present
        if not is_time_query:
            flexible_location_regex = r'\b(for|in|i)\b'
            if re.search(flexible_location_regex, query_lower):
                specific_facility_requested = True

        if facility_mentioned or specific_facility_requested:
            # Try to find the facility name
            found_facility = False
            
            # Check each word in query for facility matches
            for i in range(len(query_words)):
                for j in range(i+1, min(i+4, len(query_words)+1)):  # Check up to 4-word combinations
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
                    if specific_facility_requested:
                         return {
                             "intent": "chat",
                             "response": "⚠️ **Facility not found!**\n\nI couldn't find that facility in the system. Please check:\n1. **Spelling** of the facility name\n2. Try saying **'list facilities'** to see all available facilities\n3. Use the full facility name (e.g., 'Adigrat Hospital' not just 'Adigrat')"
                         }
        
        # Two-pass approach to avoid ambiguous prefix matches
        # Pass 1: Strong Matches (Name strictly contained in query)
        strong_matches = []
        for name, uid in self.facility_mapping.items():
            n_lower = name.lower()
            if n_lower in query_norm:
                strong_matches.append((name, uid))
        
        if strong_matches:
            for name, uid in strong_matches:
                selected_facility_uids.append(uid)
                selected_facility_names.append(name)
        else:
            # Pass 2: Weak Matches (StartsWith) - ONLY if no strong matches
            # Use filtered_words to avoid common stopwords
            for name, uid in self.facility_mapping.items():
                n_lower = name.lower()
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
                 cleaned_query_date = query_lower.replace(" ot ", " to ").replace(" too ", " to ").replace(" from ", "")
                 # Clean up month typos for this specific check
                 for typo, corr in [("jna", "jan"), ("fbe", "feb"), ("mrc", "mar"), ("arp", "apr"), ("my", "may"), ("jnu", "jun"), ("july", "jul"), ("augst", "aug"), ("sept", "sep"), ("octber", "oct"), ("novmber", "nov"), ("dec", "dec")]:
                      if typo in cleaned_query_date:
                           cleaned_query_date = cleaned_query_date.replace(typo, corr)
                 
                 today_str = today.strftime("%Y-%m-%d")
                 
                 # Pattern: (Month DD) ... today
                 pattern_today = re.compile(r"([a-z]{3})\s+(\d{1,2}).*?today", re.IGNORECASE)
                 match_today = pattern_today.search(cleaned_query_date)
                 
                 if match_today:
                      m, d = match_today.groups()
                      # Assume current year first, fallback logic for rollovers could be added
                      y = today.year
                      start_date = datetime.strptime(f"{m} {d} {y}", "%b %d %Y").strftime("%Y-%m-%d")
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
             
             if (any(x in query_lower for x in data_keywords) or kpi_found) and not any(d in query_lower for d in ["define", "meaning", "definition", "list", "show", "available"]):
                 # It's likely a data query like "What is the total admitted mothers..."
                 # FORCE override even if LLM said metadata_query (common error for "how many")
                 if intent == "metadata_query" and kpi_found:
                      intent = "text"
                 elif intent == "metadata_query" and not kpi_found:
                      pass # Valid metadata query like "how many facilities"
                 else: 
                      intent = "text"
             
             elif any(d in query_lower for d in ["define", "meaning", "definition"]):
                 intent = "definition"

        # Robust List Detection
        if "indicator" in query_lower or "kpi" in query_lower or "all indicators" in query_lower:
            if any(x in query_lower for x in ["what", "list", "show", "available", "options", "help", "how many", "total"]):
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
        if not selected_kpi and ("how many" in query_lower or "list" in query_lower or "show me" in query_lower or "show all" in query_lower):
            if "region" in query_lower or "reigon" in query_lower or " reg " in query_lower:
                intent = "metadata_query"
                entity_type = "region"
                count_requested = "how many" in query_lower
            elif any(x in query_lower for x in ["facilit", "hospital", " fac ", " failit"]):
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
             # Force plot intent if we are comparing, unless it's a list or definition or scope error
             if intent not in ["list_kpis", "definition", "scope_error"]:
                  intent = "plot"

        # Detect "By Facility" intent (Drill-down / Disaggregation)
        # Triggers if:
        # 1. "by facility" explicit phrase
        # 2. "facilities" mentioned along with a region in comparison mode (e.g. "Compare Tigray facilities")
                # NEW: Implicit Comparison Triggers (Handle "by facility" at top level)
        if not comparison_mode:
            if any(x in query_lower for x in ["by facility", "per facility", "by fac", "per fac"]):
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
        drill_down_phrases = ["by facility", "per facility", "under", "in ", "i ", "breakdown by facility", "compare facilities", "by region", "per region", "breakdown by region"]
        is_drill_down = any(p in query_lower for p in drill_down_phrases) and ("facilit" in query_lower or "hospital" in query_lower or "region" in query_lower)
        
        if is_drill_down:
             # Force Plot intent unless it's a definition
             if intent != "definition":
                 intent = "plot"
             
             # If we have found regions (e.g. "Tigray by facility"), we want to compare facilities within that region
             if found_regions:
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

        final_date_range = {"start_date": start_date, "end_date": end_date} if start_date else (None if reset_date else context.get("date_range"))
        
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

        # Horizontal Chart Detection
        orientation = "v"
        if "horizontal" in query_lower:
            orientation = "h"
            if chart_type == "line": chart_type = "bar" # Line cannot be horizontal effectively usually
        
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
            "is_drill_down": is_drill_down,
            "response": None
        }

    def _get_cache_key(self, parsed_query, facility_uids=None):
        """Create a unique cache key for the query"""
        import hashlib
        import json
        
        # Sort UIDs for consistent hashing
        sorted_uids = sorted(facility_uids) if facility_uids else []
        
        key_data = {
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
        parsed = self.parse_query(query)
        query_lower = query.lower()
        
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
            
            # Fetch config from dash_co (logic imported in generate_response)
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
                return None, response
            
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
             
        # List KPIs handler at top (Consolidated)
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
            msg = "I have no information on these currently. If you want to know what I am capable of, you can list indicators for Maternal. Currently, I do not have info on other programs or on Newborn data."
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

                import plotly.express as px
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
                 from newborns_dashboard.dash_co_newborn import prepare_data_for_newborn_trend_chart
                 all_data_df, date_col = prepare_data_for_newborn_trend_chart(active_df, active_kpi_name, flat_uids, date_range)
            else:
                 all_data_df, date_col = self._silent_prepare_data(active_df, active_kpi_name, flat_uids, date_range)

            # 4. Global Period Assignment
            if not all_data_df.empty:
                 # Check if period labels needed
                 from utils.time_filter import assign_period
                 p_label = parsed.get("period_label") or st.session_state.get("period_label", "Monthly")
                 
                 # Ensure date column exists as datetime
                 if "event_date" not in all_data_df.columns and date_col in all_data_df.columns:
                     all_data_df["event_date"] = pd.to_datetime(all_data_df[date_col], errors='coerce')
                 
                 # Assign period
                 if "event_date" in all_data_df.columns:
                     all_data_df = assign_period(all_data_df, "event_date", p_label)

            # Loop through Groups and Build Data using MEMORY FILTERING
            chart_data = []
            
            for entity_name, entity_uids in comparison_groups:
                 # Filter this group's data from the batch
                 if all_data_df.empty:
                     entity_df = pd.DataFrame()
                 else:
                     if "orgUnit" in all_data_df.columns and entity_uids:
                         entity_df = all_data_df[all_data_df["orgUnit"].isin(entity_uids)].copy()
                     else:
                         # Fallback for national/region if specific UID column logic differs
                         # For now assuming orgUnit is key
                         if not entity_uids: # Overall
                             entity_df = all_data_df.copy()
                         else:
                             entity_df = pd.DataFrame()
                 
                 if entity_df.empty:
                     continue

                 # Determine Period Order (Already done globally but ensuring display col)
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
            
            result = (fig, f"Here is the {chart_type} chart for **{kpi_name}**{context_desc}.{suggestion}{nav_feedback}")
            self._cache_result(cache_key, result)
            return result
            
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
            
            # Cache result
            result = (None, response_text + suggestion + nav_feedback)
            self._cache_result(cache_key, result)
            return result


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
    st.markdown('<h1 class="chat-header">🤖 IMNID Chatbot</h1>', unsafe_allow_html=True)
    
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
    
    # Inject KB_DEFINITIONS for Definition Intent
    chatbot_logic.KB_DEFINITIONS = {
        "C-Section Rate (%)": "A Caesarean section (C-section) is a surgical procedure used to deliver a baby through incisions in the abdomen and uterus. The rate is the percentage of deliveries performed via C-section out of total deliveries.",
        "Postpartum Hemorrhage (PPH) Rate (%)": "Postpartum Hemorrhage (PPH) is defined as excessive bleeding after childbirth (usually >500ml for vaginal, >1000ml for C-section). It is a leading cause of maternal mortality.",
        "Maternal Death Rate (per 100,000)": "Maternal death refers to the death of a woman while pregnant or within 42 days of termination of pregnancy, from any cause related to or aggravated by the pregnancy or its management but not from accidental or incidental causes. The rate is calculated per 100,000 live births.",
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
                        # FIX: Pass region names (comparison_targets) instead of UIDs
                        render_func(render_df, "period_display", "value", params["active_kpi_name"], "#FFFFFF", None, params["comparison_targets"], regions_mapping, regions_mapping, params["num_label"], params["den_label"], suppress_plot=suppress_plot, key_suffix=f"msg_{message_index}")
                    elif suffix == "newborn":
                        from utils.queries import get_facilities_grouped_by_region
                        regions_mapping = get_facilities_grouped_by_region(st.session_state.get("user", {}))
                        # FIX: Pass region names (comparison_targets) instead of UIDs
                        render_func(render_df, "period_display", "value", params["active_kpi_name"], "#FFFFFF", None, params["comparison_targets"], regions_mapping, regions_mapping, params["num_label"], params["den_label"], suppress_plot=suppress_plot, key_suffix=f"msg_{message_index}")
                    elif suffix == "newborn_simplified":
                        # FIX: Use region names
                        render_func(render_df, region_names=params["comparison_targets"])
                    elif suffix == "admitted_mothers":
                        # admitted_mothers has a different signature
                        # FIX: Use region names
                        render_func(render_df, "period_display", "value", params["active_kpi_name"], "#FFFFFF", None, params["comparison_targets"], suppress_plot=suppress_plot, key_suffix=f"msg_{message_index}")
                    else:
                        # All other specialized maternal modules use the same signature as utils
                        from utils.queries import get_facilities_grouped_by_region
                        regions_mapping = get_facilities_grouped_by_region(st.session_state.get("user", {}))
                        # FIX: Pass region names (comparison_targets) instead of UIDs
                        render_func(render_df, "period_display", "value", params["active_kpi_name"], "#FFFFFF", None, params["comparison_targets"], regions_mapping, regions_mapping, params["num_label"], params["den_label"], suppress_plot=suppress_plot, key_suffix=f"msg_{message_index}")
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
                            st.plotly_chart(fig, use_container_width=True, key=f"chat_chart_new_{len(st.session_state.messages)}")
                            msg_obj["figure"] = fig
                        
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
                        error_msg += "It seems I couldn't find the specific health indicator you're looking for. I currently specialize in **Maternal Health indicators**."
                    elif "facility" in e_str or "region" in e_str:
                        error_msg += "I had trouble identifying the location (facility or region) in your prompt."
                    elif "local variable" in e_str:
                        error_msg += "I encountered a technical glitch while resolving names. Please try your request again."
                    else:
                        error_msg += "I'm not quite sure how to handle that specific question yet."
                    
                    error_msg += "\n\n---\n"
                    error_msg += "💡 **How to Prompt for Best Results:**\n"
                    error_msg += "- **Be Specific**: Mention an indicator and a location. *(e.g., 'Plot PPH for Tigray')*\n"
                    error_msg += "- **Ask for Lists**: If you're unsure, ask 'list indicators' or 'show all regions'.\n"
                    error_msg += "- **Keep it Simple**: Ask one question at a time.\n"
                    
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
- **Plot Charts**: I can generate Line, Bar, and Area charts for Maternal Health indicators.
- **Distributions**: Ask for a **breakdown** or **distribution** (e.g., "show distribution of complications") to see categorical pie charts.
- **Quick Values**: Ask for a specific value (e.g. "What is the PPH rate?") and I'll provide the latest figure.
- **Comparisons**: Ask to compare facilities or regions! (e.g., "Compare Admitted Mothers for Adigrat and Suhul")
- **Definitions**: Ask "Define [Indicator]" to get a medical definition.
- **Data Tables**: Ask for "table format" to see the raw numbers.

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