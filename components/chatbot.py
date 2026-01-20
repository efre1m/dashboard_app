# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import re
# import random
# import difflib
# import logging
# from datetime import datetime, timedelta
# from collections import defaultdict
# from utils.llm_utils import query_llm
# from utils import kpi_utils

# # Import logic from dashboards to ensure data availability
# from dashboards import facility, regional, national
# import utils.kpi_utils as kpi_utils
# from utils.kpi_utils import (
#     prepare_data_for_trend_chart,
#     compute_kpis
# )
# from utils.kpi_admitted_mothers import get_numerator_denominator_for_admitted_mothers
# from utils.queries import get_facility_mapping_for_user, get_facilities_grouped_by_region, get_all_facilities_flat
# from utils.dash_co import KPI_OPTIONS, KPI_MAPPING

# def ensure_data_loaded():
#     """
#     Ensures that the necessary data is loaded into session state based on the user's role.
#     Returns the shared_data dict.
#     """
#     user = st.session_state.get("user", {})
#     role = user.get("role", "")
    
#     if not role:
#         return None

#     if role == "facility":
#         # Check if data is already in session state
#         if hasattr(st.session_state, "cached_shared_data_facility") and st.session_state.cached_shared_data_facility:
#             return st.session_state.cached_shared_data_facility
            
#         # Load data if not present
#         with st.spinner("Initializing chatbot data access..."):
#             static_data = facility.get_static_data_facility(user)
#             program_uid_map = static_data["program_uid_map"]
#             data = facility.get_shared_program_data_facility(user, program_uid_map, show_spinner=False)
#             return data
            
#     elif role == "regional":
#         if hasattr(st.session_state, "cached_shared_data_regional") and st.session_state.cached_shared_data_regional:
#             return st.session_state.cached_shared_data_regional
            
#         with st.spinner("Initializing chatbot data access..."):
#             static_data = regional.get_static_data(user)
#             program_uid_map = static_data["program_uid_map"]
#             data = regional.get_shared_program_data_optimized(user, program_uid_map, show_spinner=False)
#             return data
            
#     elif role == "national":
#         if hasattr(st.session_state, "cached_shared_data_national") and st.session_state.cached_shared_data_national:
#             return st.session_state.cached_shared_data_national
            
#         with st.spinner("Initializing chatbot data access..."):
#             static_data = national.get_static_data(user)
#             program_uid_map = static_data["program_uid_map"]
#             data = national.get_shared_program_data_optimized(user, program_uid_map, show_spinner=False)
#             return data
            
#     return None

# class ChatbotLogic:
#     def __init__(self, data):
#         self.data = data
#         self.user = st.session_state.get("user", {})
#         self.facility_mapping = get_facility_mapping_for_user(self.user)
#         # Reverse mapping for easy lookup
#         self.uid_to_name = {v: k for k, v in self.facility_mapping.items()}
        
#         # Get all facilities with region information for smarter matching
#         self.all_facilities_with_regions = self._get_all_facilities_with_regions()
        
#         # Track ambiguous facility names
#         self.ambiguous_facilities = self._identify_ambiguous_facilities()
        
#         # Common typos mapping
#         self.COMMON_TYPOS = {
#             # Facility name typos
#             'ambo': 'Ambo',
#             'ambbo': 'Ambo',
#             'adigrat': 'Adigrat',
#             'adigudom': 'Adigudom',
#             'mekelle': 'Mekelle',
#             'axum': 'Axum',
#             'dilla': 'Dilla',
#             'jimma': 'Jimma',
#             'adama': 'Adama',
#             'hawassa': 'Hawassa',
#             'dessie': 'Dessie',
#             'woldia': 'Woldia',
            
#             # Region name typos
#             'tigray': 'Tigray',
#             'tigrai': 'Tigray',
#             'oromia': 'Oromia',
#             'oromya': 'Oromia',
#             'amhara': 'Amhara',
#             'sidama': 'Sidama',
#             'afar': 'Afar',
#             'south ethiopia': 'South Ethiopia',
#             'central ethiopia': 'Central Ethiopia',
#             'south west ethiopia': 'South West Ethiopia',
            
#             # KPI typos
#             'csection': 'C-Section Rate (%)',
#             'c section': 'C-Section Rate (%)',
#             'caesarean': 'C-Section Rate (%)',
#             'section': 'C-Section Rate (%)',
#             'pph': 'Postpartum Hemorrhage (PPH) Rate (%)',
#             'hemorrhage': 'Postpartum Hemorrhage (PPH) Rate (%)',
#             'maternal death': 'Institutional Maternal Death Rate (%)',
#             'stillbirth': 'Stillbirth Rate (%)',
#             'svd': 'Normal Vaginal Delivery (SVD) Rate (%)',
#             'vaginal delivery': 'Normal Vaginal Delivery (SVD) Rate (%)',
#             'normal delivery': 'Normal Vaginal Delivery (SVD) Rate (%)',
#             'admitted': 'Admitted Mothers',
#             'admissions': 'Admitted Mothers',
#             'deliveries': 'Total Deliveries',
#         }

#         # Combine maternal and newborn data
#         maternal_df = data.get("maternal", {}).get("patients", pd.DataFrame()) if data.get("maternal") else pd.DataFrame()
#         self.df = maternal_df

#         # --- SPECIALIZED KPI MAPPING ---
#         # Maps full KPI names to their internal utility script suffixes
#         # Keys MUST match active_kpi_name (which comes from KPI_MAPPING/KPI_OPTIONS)
#         self.SPECIALIZED_KPI_MAP = {
#             "Total Admitted Mothers": "admitted_mothers",
#             "Admitted Mothers": "admitted_mothers",
#             "Postpartum Hemorrhage (PPH) Rate (%)": "pph",
#             "Normal Vaginal Delivery (SVD) Rate (%)": "svd",
#             "ARV Prophylaxis Rate (%)": "arv",
#             "Assisted Delivery Rate (%)": "assisted",
#             "Delivered women who received uterotonic (%)": "uterotonic",
#             "Missing Birth Outcome": "missing_bo",
#             "Missing Condition of Discharge": "missing_cod",
#             "Missing Mode of Delivery": "missing_md",
#             # Standard KPIs that use kpi_utils
#             "C-Section Rate (%)": "utils",
#             "Institutional Maternal Death Rate (%)": "utils",
#             "Stillbirth Rate (%)": "utils",
#             "Early Postnatal Care (PNC) Coverage (%)": "utils",
#             "Immediate Postpartum Contraceptive Acceptance Rate (IPPCAR %)": "utils"
#         }
    
#     def _get_all_facilities_with_regions(self):
#         """Get all facilities with their region information"""
#         regions_data = get_facilities_grouped_by_region(self.user)
#         facilities_with_regions = []
#         for region, fac_list in regions_data.items():
#             for fac_name, fac_uid in fac_list:
#                 facilities_with_regions.append({
#                     'name': fac_name,
#                     'uid': fac_uid,
#                     'region': region
#                 })
#         return facilities_with_regions
    
#     def _identify_ambiguous_facilities(self):
#         """Identify facilities with similar names that could be ambiguous"""
#         name_to_facilities = defaultdict(list)
#         for fac in self.all_facilities_with_regions:
#             # Extract first word for checking ambiguity
#             first_word = fac['name'].split()[0]
#             name_to_facilities[first_word].append(fac)
        
#         # Return only names with multiple facilities
#         ambiguous = {}
#         for name, facilities in name_to_facilities.items():
#             if len(facilities) > 1:
#                 ambiguous[name] = facilities
        
#         # Also check for specific known ambiguous cases
#         additional_ambiguous = {
#             'Ambo': [fac for fac in self.all_facilities_with_regions if 'Ambo' in fac['name']],
#             'Leku': [fac for fac in self.all_facilities_with_regions if 'Leku' in fac['name']],
#             'Robe': [fac for fac in self.all_facilities_with_regions if 'Robe' in fac['name']],
#         }
        
#         ambiguous.update(additional_ambiguous)
#         return ambiguous
    
#     def _handle_follow_up_response(self, query, context):
#         """Handle user's response to a follow-up question"""
#         if context.get("type") == "ambiguous_facility":
#             options = context.get("options", [])
#             query_lower = query.lower()
            
#             # Check for region mention
#             for option in options:
#                 if option['region'].lower() in query_lower:
#                     return {
#                         "facility_uids": [option['uid']],
#                         "facility_names": [option['name']],
#                         "region": option['region']
#                     }
            
#             # Check for full facility name
#             for option in options:
#                 if option['name'].lower() in query_lower:
#                     return {
#                         "facility_uids": [option['uid']],
#                         "facility_names": [option['name']],
#                         "region": option['region']
#                     }
            
#             # Check for "first", "second", etc.
#             if "first" in query_lower and len(options) >= 1:
#                 return {
#                     "facility_uids": [options[0]['uid']],
#                     "facility_names": [options[0]['name']],
#                     "region": options[0]['region']
#                 }
#             elif "second" in query_lower and len(options) >= 2:
#                 return {
#                     "facility_uids": [options[1]['uid']],
#                     "facility_names": [options[1]['name']],
#                     "region": options[1]['region']
#                 }
#             elif "third" in query_lower and len(options) >= 3:
#                 return {
#                     "facility_uids": [options[2]['uid']],
#                     "facility_names": [options[2]['name']],
#                     "region": options[2]['region']
#                 }
#             elif "1" in query_lower and len(options) >= 1:
#                 return {
#                     "facility_uids": [options[0]['uid']],
#                     "facility_names": [options[0]['name']],
#                     "region": options[0]['region']
#                 }
#             elif "2" in query_lower and len(options) >= 2:
#                 return {
#                     "facility_uids": [options[1]['uid']],
#                     "facility_names": [options[1]['name']],
#                     "region": options[1]['region']
#                 }
#             elif "3" in query_lower and len(options) >= 3:
#                 return {
#                     "facility_uids": [options[2]['uid']],
#                     "facility_names": [options[2]['name']],
#                     "region": options[2]['region']
#                 }
        
#         return None
    
#     def _silent_prepare_data(self, df, kpi_name, facility_uids=None, date_range_filters=None):
#         """
#         Silent version of prepare_data_for_trend_chart that uses logging instead of st.info/warning.
#         Copied logic to ensure chatbot doesn't spam UI.
#         """
#         from utils.kpi_utils import get_relevant_date_column_for_kpi
#         from utils.time_filter import assign_period
        
#         if df.empty:
#             return pd.DataFrame(), None

#         filtered_df = df.copy()

#         # Filter by facility UIDs if provided
#         if facility_uids and "orgUnit" in filtered_df.columns:
#             filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

#         # Get the SPECIFIC date column for this KPI
#         date_column = get_relevant_date_column_for_kpi(kpi_name)

#         # Check if the SPECIFIC date column exists
#         if date_column not in filtered_df.columns:
#             # Try to use event_date as fallback
#             if "event_date" in filtered_df.columns:
#                 date_column = "event_date"
#                 logging.warning(f"Chatbot: KPI-specific date column not found for {kpi_name}, using 'event_date'")
#             else:
#                 logging.warning(f"Chatbot: Required date column '{date_column}' not found for {kpi_name}")
#                 return pd.DataFrame(), date_column

#         # Create result dataframe
#         result_df = filtered_df.copy()

#         # Convert to datetime
#         result_df["event_date"] = pd.to_datetime(result_df[date_column], errors="coerce")
#         # Filter out rows without valid dates (Logic from kpi_utils)
#         result_df = result_df[result_df["event_date"].notna()].copy()

#         # CRITICAL: Apply date range filtering
#         if date_range_filters:
#             start_date = date_range_filters.get("start_date")
#             end_date = date_range_filters.get("end_date")

#             if start_date and end_date:
#                 # Convert to datetime for comparison
#                 start_dt = pd.Timestamp(start_date)
#                 end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1)  # Include end date

#                 # Filter by date range
#                 result_df = result_df[
#                     (result_df["event_date"] >= start_dt)
#                     & (result_df["event_date"] < end_dt)
#                 ].copy()

#         if result_df.empty:
#             # Silent log instead of st.info
#             logging.info(f"Chatbot: No data with valid dates in '{date_column}' for {kpi_name}")
#             return pd.DataFrame(), date_column

#         # Get period label (default to Monthly if not set)
#         period_label = st.session_state.get("period_label", "Monthly")
#         if "filters" in st.session_state and "period_label" in st.session_state.filters:
#             period_label = st.session_state.filters["period_label"]

#         # Create period columns
#         result_df = assign_period(result_df, "event_date", period_label)

#         # Filter by facility if needed (redundant usually but safe)
#         if facility_uids and "orgUnit" in result_df.columns:
#              result_df = result_df[result_df["orgUnit"].isin(facility_uids)].copy()

#         return result_df, date_column

#     def parse_query(self, query):
#         """
#         Parses the user query to extract intent, kpi, filters.
#         Tries LLM first, falls back to enhanced regex.
#         """
#         query_lower = query.lower()
        
#         # Check if this is a follow-up response
#         if st.session_state.get("waiting_for_clarification"):
#             follow_up_context = st.session_state.get("clarification_context", {})
#             follow_up_result = self._handle_follow_up_response(query, follow_up_context)
#             if follow_up_result:
#                 # Clear waiting state
#                 st.session_state.waiting_for_clarification = False
#                 st.session_state.clarification_context = {}
                
#                 # Get the original context to merge with follow-up
#                 original_context = st.session_state.get("original_query_context", {})
                
#                 # Update with follow-up result
#                 original_context.update({
#                     "facility_uids": follow_up_result["facility_uids"],
#                     "facility_names": follow_up_result["facility_names"],
#                     "is_follow_up": True
#                 })
#                 return original_context
        
#         # 0. Try LLM Parsing
#         try:
#             from utils.llm_utils import query_llm
#             facility_names_list = list(self.facility_mapping.keys())
#             llm_result = query_llm(query, facility_names_list)
            
#             if llm_result:
#                 # Handle "chat" intent from LLM
#                 if llm_result.get("intent") == "chat":
#                     return {
#                         "response": llm_result.get("response")
#                     }
                
#                 # Special Handling for Generic "What indicators" query via LLM
#                 if llm_result.get("intent") == "list_kpis":
#                     # Check if we should ask clarification
#                     if not any(x in query.lower() for x in ["maternal", "newborn", "all"]):
#                         return {
#                             "intent": "chat",
#                             "response": "Are you interested in **Maternal** or **Newborn** health indicators?"
#                         }
#                 if llm_result.get("intent") == "clear":
#                     return {"intent": "clear"}
                
#                 # If LLM identified a KPI, use it
#                 if llm_result.get("kpi"):
#                     # Map facility names to UIDs
#                     selected_facility_uids = []
#                     selected_facility_names = []
                    
#                     llm_facs = llm_result.get("facility_names", [])
                    
#                     # Check for ambiguous facilities in LLM result
#                     for fname in llm_facs:
#                         fname_lower = fname.lower()
#                         fname_title = fname.title()
                        
#                         # Check if this is an ambiguous facility
#                         if fname_lower in self.ambiguous_facilities or fname_title in self.ambiguous_facilities:
#                             facilities = self.ambiguous_facilities.get(fname_lower, 
#                                                                     self.ambiguous_facilities.get(fname_title, []))
#                             if len(facilities) > 1:
#                                 # Need clarification
#                                 st.session_state.waiting_for_clarification = True
#                                 st.session_state.clarification_context = {
#                                     "type": "ambiguous_facility",
#                                     "query": fname,
#                                     "options": facilities
#                                 }
#                                 st.session_state.original_query_context = {
#                                     "intent": llm_result.get("intent", "text"),
#                                     "chart_type": llm_result.get("chart_type", "line"),
#                                     "kpi": llm_result.get("kpi"),
#                                     "date_range": llm_result.get("date_range"),
#                                     "entity_type": llm_result.get("entity_type"),
#                                     "count_requested": llm_result.get("count_requested"),
#                                     "comparison_mode": llm_result.get("comparison_mode", False),
#                                     "comparison_entity": llm_result.get("comparison_entity"),
#                                 }
                                
#                                 return {
#                                     "intent": "clarification",
#                                     "question": f"I found multiple facilities named '{fname}'. Which one do you mean?\n\n" + 
#                                               "\n".join([f"{i+1}. {fac['name']} ({fac['region']})" 
#                                                        for i, fac in enumerate(facilities)])
#                                 }
                        
#                         # Regular facility matching
#                         if fname in self.facility_mapping:
#                             selected_facility_uids.append(self.facility_mapping[fname])
#                             selected_facility_names.append(fname)
#                         else:
#                             # Try fuzzy matching
#                             matches = difflib.get_close_matches(fname, list(self.facility_mapping.keys()), n=1, cutoff=0.6)
#                             if matches:
#                                 selected_facility_uids.append(self.facility_mapping[matches[0]])
#                                 selected_facility_names.append(matches[0])
                    
#                     # Process regions from LLM
#                     regions_data = get_facilities_grouped_by_region(self.user)
#                     all_regions = list(regions_data.keys())
#                     found_regions = []
                    
#                     for fname in llm_facs:
#                         fname_clean = fname.strip().lower()
                        
#                         # Check if it's a region
#                         if fname in regions_data:
#                             found_regions.append(fname)
#                         else:
#                             # Fuzzy Region Match
#                             r_matches = difflib.get_close_matches(fname, all_regions, n=1, cutoff=0.6)
#                             if r_matches:
#                                 found_regions.append(r_matches[0])
                    
#                     # --- DRILL-DOWN / DRILL-UP LOGIC (New) ---
#                     if "by facility" in query_lower or "per facility" in query_lower:
#                         # Force facility comparison if region is present
#                         if found_regions:
#                             selected_facility_uids = []
#                             selected_facility_names = []
#                             for r in found_regions:
#                                 facs_in_region = regions_data.get(r, [])
#                                 selected_facility_uids.extend([f[1] for f in facs_in_region])
#                                 selected_facility_names.extend([f[0] for f in facs_in_region])
#                             llm_result["comparison_mode"] = True
#                             llm_result["comparison_entity"] = "facility"

#                     # If no facilities filtered by LLM but user is Facility Role, assume their facility
#                     if not selected_facility_uids and not found_regions and self.user.get("role") == "facility":
#                         selected_facility_uids = list(self.facility_mapping.values())
#                         selected_facility_names = list(self.facility_mapping.keys())
                    
#                     # Populate associated facilities if only Region was found
#                     if found_regions and not selected_facility_uids:
#                         # Get all facilities in these regions
#                         for r in found_regions:
#                             facs_in_region = regions_data.get(r, [])
#                             selected_facility_uids.extend([f[1] for f in facs_in_region])
#                             selected_facility_names.append(f"{r} (Region)")

#                     return {
#                         "intent": llm_result.get("intent", "text"),
#                         "chart_type": llm_result.get("chart_type", "line"),
#                         "kpi": llm_result.get("kpi"),
#                         "facility_uids": selected_facility_uids,
#                         "facility_names": selected_facility_names,
#                         "date_range": llm_result.get("date_range"),
#                         "entity_type": llm_result.get("entity_type"),
#                         "count_requested": llm_result.get("count_requested"),
#                         "comparison_mode": llm_result.get("comparison_mode", False),
#                         "comparison_entity": llm_result.get("comparison_entity"),
#                         "comparison_targets": found_regions if llm_result.get("comparison_entity") == "region" else selected_facility_names, 
#                         "region_filter": llm_result.get("region_filter"),
#                         "response": llm_result.get("response")
#                     }
#         except Exception as e:
#             logging.warning(f"LLM parsing failed: {e}")
        
#         # --- ENHANCED REGEX / FUZZY MATCHING ---
        
#         # Check for Clear Chat
#         if "clear chat" in query_lower or "reset chat" in query_lower:
#             return {"intent": "clear"}
        
#         # 1. Detect Intent and Chart Type
#         intent = "text"
#         chart_type = "line" # Default
#         entity_type = None
#         count_requested = False
#         comparison_mode = False
#         comparison_entity = None
        
#         if any(w in query_lower for w in ["plot", "graph", "chart", "trend", "visualize", "show me"]):
#             intent = "plot"
        
#         if "table" in query_lower:
#             chart_type = "table"
#             intent = "plot" # Treat table requests as plot/data requests
#         elif "bar" in query_lower:
#             chart_type = "bar"
#         elif "area" in query_lower:
#             chart_type = "area"
#         elif "line" in query_lower:
#             chart_type = "line"
            
#         # Detect Chart Option Parsing
#         if "which charts" in query_lower or "types of charts" in query_lower or ("available" in query_lower and "chart" in query_lower):
#             return {
#                 "intent": "chart_options",
#                 "kpi": None # Will be filled by context merging if exists
#             }
            
#         # Detect Comparison Mode
#         if any(x in query_lower for x in ["compare", "comparison", " vs ", "versus", "benchmark"]):
#             comparison_mode = True
#             intent = "plot" # Comparison implies visual
            
#             # Detect Entity
#             if "region" in query_lower:
#                 comparison_entity = "region"
#             elif "facilit" in query_lower or "hospital" in query_lower or "clinic" in query_lower:
#                 comparison_entity = "facility"
#             else:
#                 # Default to whatever is selected or infer
#                 pass
            
#         # 2. Detect KPI with enhanced typos handling
#         selected_kpi = None
        
#         # Normalize spaces and remove some punctuation for fuzzy matching
#         query_norm = re.sub(r'[^a-z0-9\s]', '', query_lower)
        
#         # First check common typos
#         for typo, correction in self.COMMON_TYPOS.items():
#             if typo in query_norm and correction in KPI_OPTIONS:
#                 selected_kpi = correction
#                 break
        
#         # If not found via typos, use existing KPI map
#         if not selected_kpi:
#             # Fix common typos manually
#             query_norm = query_norm.replace("sectioin", "section").replace("sectionn", "section")
#             query_norm = query_norm.replace("still birth", "stillbirth").replace("c section", "csection")
#             query_norm = query_norm.replace("birht", "birth").replace("oucome", "outcome")
#             query_norm = query_norm.replace("indicatofrs", "indicators").replace("totaly", "totally")
#             query_norm = query_norm.replace("uterotoncic", "uterotonic").replace("wome", "women").replace("whor", "who")
#             query_norm = query_norm.replace("abot", "about").replace("abut", "about")
            
#             # Comprehensive KPI Map based on dash_co.KPI_MAPPING
#             kpi_map = {
#                 # Standard KPIs
#                 "csection": "C-Section Rate (%)",
#                 "c section": "C-Section Rate (%)",
#                 "caesarean": "C-Section Rate (%)",
#                 "maternal death": "Institutional Maternal Death Rate (%)",
#                 "stillbirth": "Stillbirth Rate (%)",
#                 "pph": "Postpartum Hemorrhage (PPH) Rate (%)",
#                 "hemorrhage": "Postpartum Hemorrhage (PPH) Rate (%)",
#                 "bleeding": "Postpartum Hemorrhage (PPH) Rate (%)",
#                 "uterotonic": "Delivered women who received uterotonic (%)",
#                 "oxytocin": "Delivered women who received uterotonic (%)",
#                 "ippcar": "Immediate Postpartum Contraceptive Acceptance Rate (IPPCAR %)",
#                 "contraceptive": "Immediate Postpartum Contraceptive Acceptance Rate (IPPCAR %)",
#                 "family planning": "Immediate Postpartum Contraceptive Acceptance Rate (IPPCAR %)",
#                 "pnc": "Early Postnatal Care (PNC) Coverage (%)",
#                 "postnatal": "Early Postnatal Care (PNC) Coverage (%)",
#                 "arv": "ARV Prophylaxis Rate (%)",
#                 "antiretroviral": "ARV Prophylaxis Rate (%)",
#                 "assisted delivery": "Assisted Delivery Rate (%)",
#                 "instrumental": "Assisted Delivery Rate (%)",
#                 "vacuum": "Assisted Delivery Rate (%)",
#                 "forceps": "Assisted Delivery Rate (%)",
#                 "svd": "Normal Vaginal Delivery (SVD) Rate (%)",
#                 "vaginal delivery": "Normal Vaginal Delivery (SVD) Rate (%)",
#                 "normal delivery": "Normal Vaginal Delivery (SVD) Rate (%)",
#                 "spontaneous": "Normal Vaginal Delivery (SVD) Rate (%)",
                
#                 # Data Quality / Counts
#                 "missing mode": "Missing Mode of Delivery",
#                 "missing birth": "Missing Birth Outcome",
#                 "birth outcome": "Missing Birth Outcome", # Explicit map
#                 "missing outcome": "Missing Birth Outcome",
#                 "missing condition": "Missing Condition of Discharge",
#                 "missing discharge": "Missing Condition of Discharge",
#                 "admitted mothers": "Admitted Mothers",
#                 "admitted": "Admitted Mothers", # Explicit map
#                 "admissions": "Admitted Mothers",
#                 "admission": "Admitted Mothers", # Added singular
#                 "total mothers": "Admitted Mothers",
#                 "enrollment": "Admitted Mothers",
#                 "enrollmet": "Admitted Mothers",
#                 "total enrollments": "Admitted Mothers",
                
#                 # Additional commonly requested counts
#                 "total deliveries": "Total Deliveries",
#                 "deliveries": "Total Deliveries",
#                 "births": "Total Deliveries"
#             }
            
#             # Check for direct containment first
#             for key, val in kpi_map.items():
#                 if key in query_norm:
#                     selected_kpi = val
#                     break
            
#             # If not found, try fuzzy matching on filtered words
#             if not selected_kpi:
#                 stop_words = ["what", "is", "the", "are", "of", "in", "show", "me", "tell", "about", "rate", "value", "number", "total", "how", "many"]
#                 query_words = query_norm.split()
#                 filtered_words = [w for w in query_words if w not in stop_words]
#                 filtered_query = " ".join(filtered_words)
                
#                 if filtered_query:
#                     keys = list(kpi_map.keys())
#                     matches = difflib.get_close_matches(filtered_query, keys, n=1, cutoff=0.6)
#                     if matches:
#                         selected_kpi = kpi_map[matches[0]]
                    
#                     # Sliding window backup
#                     if not selected_kpi:
#                         if "eciton" in query_lower or "c -s" in query_lower:
#                             selected_kpi = "C-Section Rate (%)"
            
#             if not selected_kpi:
#                 # Fallback: check exact strings in KPI_OPTIONS (ignoring case)
#                 for kpi in KPI_OPTIONS:
#                     if kpi.lower() in query_lower:
#                         selected_kpi = kpi
#                         break
        
#         # 3. Enhanced Facility Detection with Clarification
#         selected_facility_uids = []
#         selected_facility_names = []
        
#         # Extract words from query
#         words = re.findall(r'\b\w+\b', query)
        
#         # Check for ambiguous facility names
#         for word in words:
#             word_lower = word.lower()
#             word_title = word.title()
            
#             # Skip common stop words
#             if word_lower in ["the", "and", "for", "with", "about", "show", "me", "plot", "chart", "rate", "data", "what", "is", "are"]:
#                 continue
            
#             # Check if this is an ambiguous facility name
#             if word_lower in self.ambiguous_facilities or word_title in self.ambiguous_facilities:
#                 facilities = self.ambiguous_facilities.get(word_lower, 
#                                                          self.ambiguous_facilities.get(word_title, []))
#                 if len(facilities) > 1:
#                     # Need clarification
#                     st.session_state.waiting_for_clarification = True
#                     st.session_state.clarification_context = {
#                         "type": "ambiguous_facility",
#                         "query": word,
#                         "options": facilities
#                     }
#                     st.session_state.original_query_context = {
#                         "intent": intent,
#                         "chart_type": chart_type,
#                         "kpi": selected_kpi,
#                         "date_range": None,
#                         "comparison_mode": comparison_mode,
#                         "comparison_entity": comparison_entity,
#                     }
                    
#                     return {
#                         "intent": "clarification",
#                         "question": f"I found multiple facilities named '{word}'. Which one do you mean?\n\n" + 
#                                   "\n".join([f"{i+1}. {fac['name']} ({fac['region']})" 
#                                            for i, fac in enumerate(facilities)])
#                     }
#                 elif facilities:
#                     # Only one facility, use it
#                     fac = facilities[0]
#                     selected_facility_uids.append(fac['uid'])
#                     selected_facility_names.append(fac['name'])
#                     continue
            
#             # Try to match against all facilities
#             for fac in self.all_facilities_with_regions:
#                 fac_name_lower = fac['name'].lower()
                
#                 # Multiple matching strategies
#                 if (word_lower in fac_name_lower or 
#                     fac_name_lower.startswith(word_lower) or
#                     difflib.SequenceMatcher(None, word_lower, fac_name_lower).ratio() > 0.7):
                    
#                     if fac['uid'] not in selected_facility_uids:
#                         selected_facility_uids.append(fac['uid'])
#                         selected_facility_names.append(fac['name'])
#                         break
        
#         # If no facility found, check REGIONS (existing code)
#         if not selected_facility_uids:
#             regions_data = get_facilities_grouped_by_region(self.user)
#             found_regions = []
            
#             # Check Match for Regions (Multiple allowed for comparison)
#             found_regions = []
            
#             # Fallback for Region/Facility extraction if not from LLM
#             # Check for region names in query
#             found_regions = []
            
#             # Direct match
#             for region_name in regions_data.keys():
#                 if region_name.lower() in query_norm:
#                     found_regions.append(region_name)
            
#             # Fuzzy match if none found
#             if not found_regions:
#                 r_matches = difflib.get_close_matches(query_lower, [r.lower() for r in regions_data.keys()], n=1, cutoff=0.6)
#                 if r_matches:
#                     for r in regions_data.keys():
#                         if r.lower() == r_matches[0]:
#                             found_regions.append(r)
#                             break
            
#             # --- COMPARISON MODE FALLBACK DETECTION ---
#             if "compare" in query_lower or " vs " in query_lower or "versus" in query_lower:
#                 comparison_mode = True
#                 if found_regions:
#                     comparison_entity = "region"
#                 elif selected_facility_uids:
#                     comparison_entity = "facility"
            
#             if found_regions:
#                 # If Comparison Mode (explicit or implicit), store these regions
#                 if comparison_mode and comparison_entity == "region":
#                     # We will use 'comparison_targets' to store the list
#                     pass 
#                 else:
#                     # Standard mode - treat as aggregation filter (only if NOT comparison)
#                     pass

#                 # Collect UIDs from ALL found regions
#                 all_uids = []
#                 all_names = []
#                 for r_name in found_regions:
#                     f_list = regions_data[r_name]
#                     all_uids.extend([f[1] for f in f_list])
#                     all_names.append(f"{r_name} (Region)")
                
#                 selected_facility_uids = all_uids
#                 selected_facility_names = all_names
        
#         # If still no facility/region found but user is Facility Role, assume their facility
#         if not selected_facility_uids:
#             if self.user.get("role") == "facility":
#                 # Default to user's facility
#                 selected_facility_uids = list(self.facility_mapping.values())
#                 selected_facility_names = list(self.facility_mapping.keys())
        
#         # 4. Detect Time Period (existing code)
#         start_date = None
#         end_date = None
#         today = datetime.now()
#         reset_date = False
        
#         # Explicitly check for clearing dates
#         if any(x in query_lower for x in ["overall", "all time", "since beginning", "from start", "total", "entire period"]):
#             if "overall" in query_lower or "all time" in query_lower or "start" in query_lower:
#                 reset_date = True
#                 start_date = None
#                 end_date = None

#         if "this month" in query_lower:
#             start_date = today.replace(day=1).strftime("%Y-%m-%d")
#             end_date = today.strftime("%Y-%m-%d")
#         elif "last month" in query_lower:
#             formatted_today = today.replace(day=1)
#             last_month_end = formatted_today - timedelta(days=1)
#             last_month_start = last_month_end.replace(day=1)
#             start_date = last_month_start.strftime("%Y-%m-%d")
#             end_date = last_month_end.strftime("%Y-%m-%d")
#         elif "last year" in query_lower:
#             # Use Calendar Year logic (User Request)
#             last_year = today.year - 1
#             start_date = f"{last_year}-01-01"
#             end_date = f"{last_year}-12-31"
#         elif "this week" in query_lower:
#             # Monday of current week
#             weekday = today.weekday()
#             start_date_dt = today - timedelta(days=weekday)
#             start_date = start_date_dt.strftime("%Y-%m-%d")
#             end_date = today.strftime("%Y-%m-%d")
#         elif "last week" in query_lower:
#             # Previous week Monday-Sunday
#             weekday = today.weekday()
#             end_date_dt = today - timedelta(days=weekday + 1)
#             start_date_dt = end_date_dt - timedelta(days=6)
#             start_date = start_date_dt.strftime("%Y-%m-%d")
#             end_date = end_date_dt.strftime("%Y-%m-%d")

#         # Fallback Strict Regex Date Parsing (existing code)
#         if not start_date:
#             try:
#                 # 1. "Month DD, YYYY" or "Month DD YYYY" ranges (Explicit 2 years)
#                 month_pattern = r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*'
#                 range_pattern = re.compile(f"({month_pattern})[\s,]+(\d{{1,2}})[\s,]+(\d{{4}})\s*(?:to|-)?\s*({month_pattern})[\s,]+(\d{{1,2}})[\s,]+(\d{{4}})", re.IGNORECASE)
#                 matches = range_pattern.search(query)
                
#                 if matches:
#                     m1, d1, y1, m2, d2, y2 = matches.groups()
#                     start_date = datetime.strptime(f"{m1[:3]} {d1} {y1}", "%b %d %Y").strftime("%Y-%m-%d")
#                     end_date = datetime.strptime(f"{m2[:3]} {d2} {y2}", "%b %d %Y").strftime("%Y-%m-%d")
#             except Exception as e:
#                 logging.warning(f"Date regex 1 failed: {e}")

#         # 2. Check for "Jan 1 - Jan 7 2026" (Year only at end)
#         if not start_date:
#             try:
#                 range_pattern_end_year = re.compile(f"({month_pattern})[\s,]+(\d{{1,2}})\s*(?:to|-)?\s*({month_pattern})[\s,]+(\d{{1,2}})[\s,]+(\d{{4}})", re.IGNORECASE)
#                 range_match = range_pattern_end_year.search(query)
                
#                 if range_match:
#                     m1, d1, m2, d2, y = range_match.groups()
#                     start_date = datetime.strptime(f"{m1[:3]} {d1} {y}", "%b %d %Y").strftime("%Y-%m-%d")
#                     end_date = datetime.strptime(f"{m2[:3]} {d2} {y}", "%b %d %Y").strftime("%Y-%m-%d")
#             except Exception as e:
#                 logging.warning(f"Date regex range pattern 2 failed: {e}")

#         # 3. Fallback to extracting ANY single dates with years
#         if not start_date:
#             try:
#                 single_date_pattern = re.compile(f"({month_pattern})[\s,]+(\d{{1,2}})[\s,]+(\d{{4}})", re.IGNORECASE)
#                 all_dates = single_date_pattern.findall(query)
#                 if len(all_dates) >= 2:
#                     m1, d1, y1 = all_dates[0]
#                     m2, d2, y2 = all_dates[1]
#                     start_date = datetime.strptime(f"{m1[:3]} {d1} {y1}", "%b %d %Y").strftime("%Y-%m-%d")
#                     end_date = datetime.strptime(f"{m2[:3]} {d2} {y2}", "%b %d %Y").strftime("%Y-%m-%d")
#                 elif len(all_dates) == 1:
#                     m1, d1, y1 = all_dates[0]
#                     start_date = datetime.strptime(f"{m1[:3]} {d1} {y1}", "%b %d %Y").strftime("%Y-%m-%d")
#                     end_date = start_date 
#             except Exception as e:
#                 logging.warning(f"Date regex 3 failed: {e}")

#         # Fallback for "YYYY/MM/DD - YYYY/MM/DD"
#         if not start_date:
#             try:
#                 # Pattern: YYYY/MM/DD or YYYY-MM-DD
#                 iso_matches = re.findall(r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})', query)
#                 if len(iso_matches) >= 2:
#                     y1, m1, d1 = iso_matches[0]
#                     y2, m2, d2 = iso_matches[1]
#                     start_date = f"{y1}-{int(m1):02d}-{int(d1):02d}"
#                     end_date = f"{y2}-{int(m2):02d}-{int(d2):02d}"
#             except Exception as e:
#                 logging.warning(f"ISO Date regex parse failed: {e}")
        
#         # 5. Detect Aggregation Period
#         period_label = None
#         if "daily" in query_lower:
#             period_label = "Daily"
#         elif "weekly" in query_lower:
#             period_label = "Weekly"
#         elif "monthly" in query_lower:
#             period_label = "Monthly"
#         elif "quarterly" in query_lower:
#             period_label = "Quarterly"
#         elif "yearly" in query_lower:
#             period_label = "Yearly"
            
#         if period_label and intent != "list_kpis" and "define" not in query_lower and "what is" not in query_lower:
#             intent = "plot" # Aggregation usually implies visual trend
        
#         # 6. Detect Analysis Type (Max/Min)
#         analysis_type = None
#         if any(x in query_lower for x in ["highest", "max", "peak", "most"]):
#             analysis_type = "max"
#             intent = "text" # Force text response for analysis
#             reset_date = True # Analysis usually implies searching history
#         elif any(x in query_lower for x in ["lowest", "min", "least", "minimum"]):
#             analysis_type = "min"
#             intent = "text" # Force text response for analysis
#             reset_date = True # Analysis usually implies searching history

#         # --- REFINE INTENT ---
#         # If user asks "what is" or "value", assume text.
#         # If user asks "plot", "show", "trend", assume plot.
#         # If ambiguous, prefer text if specific date/value requested, plot if trend.
#         if "what is" in query_lower or "value of" in query_lower:
#             intent = "text"
        
#         if "how is" in query_lower or "explain" in query_lower or "computation" in query_lower:
#             intent = "explain"
        
#         # Definition Detection
#         if "what is" in query_lower or "define" in query_lower or "meaning of" in query_lower or "how many" in query_lower:
#             # If user asks for "value", "rate", "count", "number", they want DATA, not definition.
#             data_keywords = ["value", "rate", "count", "number", "score", "percentage", "trend", "plot", "total"]
            
#             # Check if ANY KPI from our mapping is in the query
#             kpi_found = any(kpi.lower() in query_lower for kpi in KPI_MAPPING.keys())
            
#             if any(x in query_lower for x in data_keywords) or kpi_found:
#                 # It's likely a data query like "What is the total admitted mothers..."
#                 if intent == "metadata_query" and kpi_found:
#                     intent = "text"
#                 elif intent == "metadata_query" and not kpi_found:
#                     pass # Valid metadata query like "how many facilities"
#                 else: 
#                     intent = "text"
            
#             elif "define" in query_lower or "meaning" in query_lower:
#                 intent = "definition"

#         # Robust List Detection
#         if "indicator" in query_lower or "kpi" in query_lower:
#             if any(x in query_lower for x in ["what", "list", "show", "available", "options", "help", "how many", "total"]):
#                 # Clarification Check
#                 scope_keywords = ["maternal", "newborn", "all", "matenal", "materanl"]
#                 if not any(x in query_lower for x in scope_keywords):
#                     return {
#                         "intent": "chat",
#                         "response": "Are you interested in **Maternal** or **Newborn** health indicators? (Currently I specialize in Maternal health!)"
#                     }
#                 intent = "list_kpis"
        
#         # Explicit handling for "maternal" answer to clarification
#         if query_lower in ["maternal", "maternal indicators", "maternal health", "mothers", "matenal"]:
#             intent = "list_kpis"
#         if "options" in query_lower or "capabilities" in query_lower:
#             intent = "list_kpis"
        
#         # Newborn Scope Detection
#         if "newborn" in query_lower:
#             intent = "scope_error_newborn"

#         # Robust Scope Error Detection
#         if any(x in query_lower for x in ["color", "style", "background", "theme", "dark mode", "appearance"]):
#             intent = "scope_error"
        
#         # Robust Chat/Greeting Detection
#         chat_patterns = ["hi", "hello", "hey", "greetings", "who are you", "thanks", "thank you", "help", "good morning", "good afternoon", "how can you help", "what can you help"]
#         cleaned_words = re.sub(r'[^a-z\s]', '', query_lower).split()
#         if any(word in chat_patterns for word in cleaned_words):
#             intent = "chat"
        
#         # Off-Topic / System Admin Detection
#         if any(x in query_lower for x in ["password", "login", "admin", "access", "credential", "sign in", "log in"]):
#             intent = "chat"
        
#         # Metadata / Counts Detection Fallback (Regex)
#         if not selected_kpi and ("how many" in query_lower or "list" in query_lower or "show me" in query_lower):
#             if "region" in query_lower:
#                 intent = "metadata_query"
#                 entity_type = "region"
#                 count_requested = "how many" in query_lower
#             elif "facilit" in query_lower or "hospital" in query_lower:
#                 intent = "metadata_query"
#                 entity_type = "facility"
#                 count_requested = "how many" in query_lower

#         # --- CONTEXT MERGING ---
#         context = st.session_state.get("chatbot_context", {})
        
#         # Merge KPI
#         if intent not in ["list_kpis", "scope_error"] and not selected_kpi:
#             if context.get("kpi"):
#                 selected_kpi = context.get("kpi")
        
#         # Merge Facilities
#         if not selected_facility_uids:
#             if context.get("facility_uids"):
#                 selected_facility_uids = context.get("facility_uids")
#                 selected_facility_names = context.get("facility_names")
        
#         # Merge Date Range
#         if not start_date and context.get("date_range"):
#             if reset_date:
#                 pass
#             else:
#                 pass
#         else:
#             pass
        
#         # Merge Entity Type (For "Name them" queries)
#         if intent == "text" and not selected_kpi and not entity_type:
#             if any(x in query_lower for x in ["name them", "list them", "what are they", "show them"]):
#                 if context.get("entity_type"):
#                     intent = "metadata_query"
#                     entity_type = context.get("entity_type")
#                     count_requested = False
        
#         # Explicitly check for clearing facilities/regions ("all regions", "overall")
#         if ("all region" in query_lower or "all facilities" in query_lower):
#             selected_facility_uids = []
#             selected_facility_names = []
        
#         # --- ROBUST COMPARISON DETECTION ---
#         comparison_keywords = ["compare", "comparison", "comparisoin", "compariosn", " vs ", "versus", "difference", "benchmark"]
#         if any(x in query_lower for x in comparison_keywords):
#             comparison_mode = True
#             if intent != "list_kpis" and intent != "scope_error":
#                 intent = "plot"

#         # Detect "By Facility" intent (Drill-down / Disaggregation)
#         is_drill_down = "by facility" in query_lower or ("facilit" in query_lower and comparison_mode and found_regions)
        
#         if is_drill_down:
#             intent = "plot"
            
#             if found_regions:
#                 comparison_mode = True
#                 comparison_entity = "facility"
                
#                 new_names = []
#                 new_uids = []
#                 for r_name in found_regions:
#                     facs = regions_data.get(r_name, [])
#                     for f_name, f_uid in facs:
#                         new_names.append(f_name)
#                         new_uids.append(f_uid)
                
#                 if len(new_names) > 0:
#                     selected_facility_names = new_names
#                     selected_facility_uids = new_uids
            
#             elif selected_facility_uids:
#                 comparison_mode = True
#                 comparison_entity = "facility"

#         final_date_range = {"start_date": start_date, "end_date": end_date} if start_date else (None if reset_date else context.get("date_range"))
        
#         # Horizontal Chart Detection
#         orientation = "v"
#         if "horizontal" in query_lower:
#             orientation = "h"
#             if chart_type == "line": chart_type = "bar"
        
#         # Auto-detect Granularity for Short Date Ranges
#         if final_date_range and not period_label:
#             try:
#                 s = datetime.strptime(final_date_range["start_date"], "%Y-%m-%d")
#                 e = datetime.strptime(final_date_range["end_date"], "%Y-%m-%d")
#                 delta = (e - s).days
#                 if delta <= 45:
#                     period_label = "Daily"
#             except:
#                 pass

#         # Infer Comparison Entity if not explicitly set
#         if comparison_mode and not comparison_entity:
#             if selected_facility_uids:
#                 comparison_entity = "facility"
#             elif found_regions:
#                 comparison_entity = "region"

#         return {
#             "intent": intent,
#             "chart_type": chart_type,
#             "orientation": orientation,
#             "analysis_type": None,
#             "kpi": selected_kpi,
#             "facility_uids": selected_facility_uids,
#             "facility_names": selected_facility_names,
#             "date_range": final_date_range,
#             "period_label": period_label,
#             "analysis_type": analysis_type,
#             "entity_type": entity_type,
#             "count_requested": count_requested,
#             "comparison_mode": comparison_mode,
#             "comparison_entity": comparison_entity,
#             "comparison_targets": found_regions if comparison_mode and comparison_entity == "region" and found_regions else [],
#             "response": None
#         }

#     def generate_response(self, query):
#         global KPI_MAPPING, KPI_OPTIONS
#         parsed = self.parse_query(query)
        
#         # Handle clarification requests
#         if parsed.get("intent") == "clarification":
#             return None, parsed.get("question")
        
#         # Handle List KPIs Intent
#         if parsed.get("intent") == "list_kpis":
#             if "newborn" in query.lower():
#                 return None, "I currently specialize in **Maternal Health** data. Newborn indicators are coming soon in the next update! 👶"
            
#             kpi_list = [k for k in KPI_MAPPING.keys()]
#             response = "Here are the available **Maternal Health Indicators**:\n\n"
#             for kpi in kpi_list:
#                 response += f"- {kpi}\n"
#             response += "\nYou can ask me to **plot** any of these or show their **stats**!"
#             return None, response

#         # Handle General Chat
#         if parsed.get("intent") == "chat":
#             if parsed.get("response"):
#                 return None, parsed.get("response")
            
#             q_low = query.lower()
#             if any(x in q_low for x in ["password", "login", "admin", "credential"]):
#                 return None, "I'm your Data Analytics Assistant. I don't handle system passwords or administrative access. Please contact your system administrator if you're having login issues! 🔒"
            
#             role = self.user.get("role", "facility")
#             return None, get_welcome_message(role)
        
#         # Handle Chart Options Parsing
#         if parsed.get("intent") == "chart_options":
#             kpi_concern = parsed.get("kpi") or st.session_state.get("chatbot_context", {}).get("kpi")
            
#             if kpi_concern == "Admitted Mothers":
#                 return None, "For **Admitted Mothers**, the available charts are:\n- **Vertical Bar Chart** (Default)\n- **Horizontal Bar Chart** (Say 'plot horizontal bar')\n- **Data Table**"
#             elif kpi_concern:
#                 return None, f"For **{kpi_concern}**, I can generate:\n- **Line Chart**: Best for trends over time.\n- **Bar Chart**: Good for comparison.\n- **Area Chart**: Visualizes volume over time.\n- **Data Table**: Detailed numbers."
#             else:
#                 return None, "I can generate the following charts for any indicator:\n- **Line Chart** (Default): 'Plot PPH trend'\n- **Bar Chart**: 'Show Admitted Mothers as bar chart'\n- **Area Chart**: 'PPH Rate area chart'\n- **Data Table**: 'Show table for C-Section'"
        
#         # Handle Scope Error
#         if parsed.get("intent") == "scope_error":
#             return None, "I'm focused on data analysis and visualization. I cannot change the dashboard's appearance or colors, but I can help you plot trends or find specific values."
        
#         # Handle Newborn Scope Error
#         if parsed.get("intent") == "scope_error_newborn":
#             return None, "I am currently optimized for **Maternal Health** indicators. Access to Newborn data is being integrated and will be available soon! 👶"
        
#         # Handle Clear Chat
#         if parsed.get("intent") == "clear":
#             st.session_state.messages = []
#             st.session_state.chatbot_context = {} 
#             st.session_state.waiting_for_clarification = False
#             st.session_state.clarification_context = {}
#             st.session_state.original_query_context = {}
#             st.session_state.messages.append({
#                 "role": "assistant",
#                 "content": "Hello! I'm your AI health assistant. You can ask me to plot trends like 'Plot C-Section Rate this month' or ask for specific values."
#             })
#             if "filters" in st.session_state:
#                 st.session_state.filters["period_label"] = "Monthly"
#             return None, "Chat history cleared."

#         # Handle Metadata Query (existing code)
#         if parsed.get("intent") == "metadata_query":
#             # ... [existing metadata query handling code]
#             pass
        
#         # Continue with existing generate_response logic...
#         # ... [ALL YOUR EXISTING generate_response CODE GOES HERE]
#         # I'm keeping the structure but won't repeat 1000+ lines
        
#         # At the beginning of the function, add this check:
#         if not parsed["kpi"]:
#             return None, "I couldn't identify the specific health indicator (KPI) you're asking about. Try phrases like 'C-Section Rate', 'PPH Rate', or 'Total Deliveries'."

#         # Store context
#         st.session_state["chatbot_context"] = {
#             "kpi": parsed["kpi"],
#             "facility_uids": parsed["facility_uids"],
#             "facility_names": parsed["facility_names"],
#             "date_range": parsed["date_range"],
#             "entity_type": parsed.get("entity_type")
#         }
        
#         # Continue with your existing generate_response logic...
#         # The rest of your 1500+ lines of generate_response code should follow here
        
#         # IMPORTANT: You need to keep all your existing generate_response code
#         # I've only shown the beginning where I added the clarification handling
#         # The rest of your function should remain exactly as it was
        
#         # Since I can't show 1500+ lines here, I'll indicate where your code continues:
#         # YOUR EXISTING generate_response CODE CONTINUES FROM HERE...
#         # [All your existing code for handling plots, data fetching, chart generation, etc.]

# def render_chatbot():
#     """
#     Renders an attractive chat bot interface in a single window mode.
#     """
#     # Custom CSS
#     st.markdown("""
#     <style>
#         .stChatInput {
#             bottom: 20px;
#         }
#         .main-chat-container {
#             max-width: 800px;
#             margin: auto;
#             padding-top: 2rem;
#             border-radius: 10px;
#             background-color: #f8f9fa; 
#             padding-bottom: 50px;
#         }
#         .chat-header {
#             text-align: center;
#             margin-bottom: 2rem;
#             color: #0f172a;
#             font-family: 'Helvetica Neue', sans-serif;
#             font-weight: 700;
#         }
#         div[data-testid="stChatMessage"] {
#             border-radius: 12px;
#             padding: 1rem;
#             margin-bottom: 1rem;
#             box-shadow: 0 2px 5px rgba(0,0,0,0.05);
#             border-left: 5px solid #0f172a;
#         }
#         div[data-testid="stChatMessage"]:nth-child(even) {
#              border-left: 5px solid #3b82f6;
#              background-color: #f1f5f9;
#         }
#     </style>
#     """, unsafe_allow_html=True)

#     # Initialize chat history and clarification state
#     if "messages" not in st.session_state:
#         st.session_state.messages = []
#         user_role = st.session_state.get("user", {}).get("role", "national")
#         welcome_msg = get_welcome_message(user_role)
#         st.session_state.messages.append({
#             "role": "assistant",
#             "content": welcome_msg
#         })
    
#     # Initialize clarification state variables
#     if "waiting_for_clarification" not in st.session_state:
#         st.session_state.waiting_for_clarification = False
#     if "clarification_context" not in st.session_state:
#         st.session_state.clarification_context = {}
#     if "original_query_context" not in st.session_state:
#         st.session_state.original_query_context = {}

#     # Clear chat button
#     if st.sidebar.button("🗑️ Clear Chat History", key="clear_chat_history_btn"):
#         st.session_state.messages = []
#         st.session_state.chatbot_context = {}
#         st.session_state.waiting_for_clarification = False
#         st.session_state.clarification_context = {}
#         st.session_state.original_query_context = {}
#         st.rerun()

#     st.markdown('<div class="main-chat-container">', unsafe_allow_html=True)
#     st.markdown('<h1 class="chat-header">🤖 IMNID AI Assistant</h1>', unsafe_allow_html=True)
    
#     # Ensure Data Availability
#     data = ensure_data_loaded()
    
#     if data is None or (isinstance(data, pd.DataFrame) and data.empty):
#         st.error("Unable to load data. Please ensure you are logged in correctly.")
#         return

#     chatbot_logic = ChatbotLogic(data)
    
#     # Inject KB_DEFINITIONS for Definition Intent
#     chatbot_logic.KB_DEFINITIONS = {
#         "C-Section Rate (%)": "A Caesarean section (C-section) is a surgical procedure used to deliver a baby through incisions in the abdomen and uterus. The rate is the percentage of deliveries performed via C-section out of total deliveries.",
#         "Postpartum Hemorrhage (PPH) Rate (%)": "Postpartum Hemorrhage (PPH) is defined as excessive bleeding after childbirth (usually >500ml for vaginal, >1000ml for C-section). It is a leading cause of maternal mortality.",
#         "Institutional Maternal Death Rate (%)": "Maternal death refers to the death of a woman while pregnant or within 42 days of termination of pregnancy, irrespective of the duration and site of the pregnancy, from any cause related to or aggravated by the pregnancy or its management but not from accidental or incidental causes.",
#         "Stillbirth Rate (%)": "A stillbirth is the death or loss of a baby before or during delivery. The rate typically measures stillbirths per 1,000 total births, but here it is presented as a percentage.",
#         "Early Postnatal Care (PNC) Coverage (%)": "Early Postnatal Care refers to the medical care given to the mother and newborn within the first 24-48 hours after delivery, crucial for detecting complications.",
#         "Admitted Mothers": "The total count of mothers admitted to the facility for delivery or pregnancy-related care.",
#         "Total Deliveries": "The total number of delivery events recorded, regardless of the outcome (live birth or stillbirth).",
#         "Immediate Postpartum Contraceptive Acceptance Rate (IPPCAR %)": "IPPCAR measures the percentage of women who accept a modern family planning method immediately after delivery (before discharge). High coverage indicates effective family planning counseling.",
#         "Delivered women who received uterotonic (%)": "This indicator measures the percentage of women who received a uterotonic drug (like oxytocin) immediately after delivery to prevent Postpartum Hemorrhage (PPH).",
#         "ARV Prophylaxis Rate (%)": "ARV Prophylaxis Rate tracks the percentage of HIV-exposed infants who received antiretroviral drugs to prevent Mother-to-Child Transmission (PMTCT).",
#         "Assisted Delivery Rate (%)": "Assisted delivery refers to vaginal deliveries assisted by instruments like vacuum extractors or forceps. It is an alternative to C-section for prolonged labor.",
#         "Normal Vaginal Delivery (SVD) Rate (%)": "SVD refers to Spontaneous Vaginal Delivery without instrumental assistance or surgery. It is the natural mode of birth.",
#         "Missing Mode of Delivery": "Data Quality Metric: The percentage of delivery records where the 'Mode of Delivery' (e.g., SVD, C-Section) was not recorded.",
#         "Missing Birth Outcome": "Data Quality Metric: The percentage of delivery records where the 'Birth Outcome' (e.g., Live Birth, Stillbirth) was not recorded.",
#         "Missing Condition of Discharge": "Data Quality Metric: The percentage of maternal discharge records where the mother's condition (e.g., Discharged Healthy, Referred, Death) was not recorded."
#     }

#     # Function to execute specialized rendering (keep your existing function)
#     def execute_specialized(spec):
#         # ... [your existing execute_specialized function code]
#         pass

#     # Display chat messages from history
#     for i, message in enumerate(st.session_state.messages):
#         with st.chat_message(message["role"]):
#             if "content" in message:
#                 st.markdown(message["content"])
            
#             # Handle specialized rendering spec
#             if "specialized_spec" in message:
#                 execute_specialized(message["specialized_spec"])
#             elif "figure" in message:
#                 if isinstance(message["figure"], pd.DataFrame):
#                     st.dataframe(message["figure"])
#                 else:
#                     st.plotly_chart(message["figure"], use_container_width=True, key=f"chat_chart_{i}")

#     # Accept user input
#     if prompt := st.chat_input("Ask about KPIs (e.g. 'Plot PPH rate for Ambo')..."):
#         # Add user message to chat history
#         st.session_state.messages.append({"role": "user", "content": prompt})
        
#         with st.chat_message("user"):
#             st.markdown(prompt)

#         with st.chat_message("assistant"):
#             message_placeholder = st.empty()
            
#             with st.spinner("Analyzing..."):
#                 try:
#                     # Check for help intent
#                     if prompt.lower().strip() in ["help", "info", "usage"]:
#                         user_role = st.session_state.get("user", {}).get("role", "national")
#                         response_text = get_welcome_message(user_role)
#                         fig = None
#                     else:
#                         fig, response_text = chatbot_logic.generate_response(prompt)
                    
#                     message_placeholder.markdown(response_text)
                    
#                     msg_obj = {"role": "assistant", "content": response_text}
                    
#                     if fig is not None:
#                         if isinstance(fig, dict) and fig.get("type") == "specialized":
#                             execute_specialized(fig)
#                             msg_obj["specialized_spec"] = fig
#                         elif isinstance(fig, pd.DataFrame):
#                             st.dataframe(fig)
#                             msg_obj["figure"] = fig
#                         else:
#                             st.plotly_chart(fig, use_container_width=True, key=f"chat_chart_new_{len(st.session_state.messages)}")
#                             msg_obj["figure"] = fig
                    
#                     st.session_state.messages.append(msg_obj)
                    
#                     if response_text == "Chat history cleared.":
#                         st.rerun()
                        
#                 except Exception as e:
#                     logging.error(f"Chatbot Error: {e}", exc_info=True)
#                     error_msg = f"I encountered an error analyzing your request: {str(e)}"
#                     message_placeholder.markdown(error_msg)
#                     st.session_state.messages.append({"role": "assistant", "content": error_msg})

#     st.markdown('</div>', unsafe_allow_html=True)

# def get_welcome_message(role):
#     """Generates a dynamic welcome message based on user role."""
    
#     dashboards = ["Maternal", "Newborn", "Summary", "Resources"]
#     if role != "facility":
#         dashboards = ["Maternal", "Newborn", "Summary", "Mentorship", "Resources", "Data Quality"]
        
#     dashboard_str = ", ".join(dashboards)
    
#     msg = f"""**Hello! I'm your IMNID AI Assistant.** 🤖
    
# I can help you analyze data across the **{dashboard_str}** dashboards.

# **Smart Features:**
# - ✅ **Understands typos**: "ambo", "tigrai", "oromya" → "Ambo", "Tigray", "Oromia"
# - ✅ **Handles ambiguous names**: If you say "Ambo", I'll ask which one (General Hospital or University Hospital)
# - ✅ **Region-aware matching**: Knows which facilities are in which regions
# - ✅ **Follow-up questions**: Asks for clarification when needed

# **What I can do for you:**
# - **📊 Plot Charts**: Line, Bar, Area charts and Data Tables
# - **🔢 Quick Values**: Ask for specific values like "What is the PPH rate?"
# - **🗺️ Comparisons**: Compare facilities or regions
# - **📚 Definitions**: Ask "What is [Indicator]?" for medical definitions
# - **📥 Data Tables**: Ask for "table format" to see the raw numbers

# **Examples:**
# - "Plot C-Section Rate for Ambo" → I'll ask which Ambo
# - "Show me Admitted Mothers in Tigray last month"
# - "Compare PPH rate for Adigrat and Mekelle"
# - "What is the definition of C-Section Rate?"

# Type **'Help'** at any time to see this message again.
# """
#     return msg