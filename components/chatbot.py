import streamlit as st
import pandas as pd
import plotly.express as px
import time
from datetime import datetime, timedelta
import re
import difflib
import logging

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
        self.uid_to_name = {v: k for k, v in self.facility_mapping.items()}
        
        # Combine maternal and newborn data
        maternal_df = data.get("maternal", {}).get("patients", pd.DataFrame()) if data.get("maternal") else pd.DataFrame()
        # For now, we focus on maternal KPIs as per user request example, but we could merge if needed.
        # kpi_utils works with the maternal dataframe structure usually.
        self.df = maternal_df


    def parse_query(self, query):
        """
        Parses the user query to extract intent, kpi, filters.
        Tries LLM first, falls back to regex.
        """
        query_lower = query.lower() # Defined globally for check
        
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
                if llm_facs:
                    for fname in llm_facs:
                        # Find closest match in our mapping
                        # (LLM should be good, but let's double check matching)
                        matches = difflib.get_close_matches(fname, self.facility_mapping.keys(), n=1, cutoff=0.6)
                        if matches:
                            matched_name = matches[0]
                            selected_facility_uids.append(self.facility_mapping[matched_name])
                            selected_facility_names.append(matched_name)
                
                # If no facilities filtered by LLM but user is Facility Role, assume their facility
                if not selected_facility_uids and self.user.get("role") == "facility":
                     selected_facility_uids = list(self.facility_mapping.values())
                     selected_facility_names = list(self.facility_mapping.keys())

                return {
                    "intent": llm_result.get("intent", "text"),
                    "chart_type": llm_result.get("chart_type", "line"),
                    "kpi": llm_result.get("kpi"),
                    "facility_uids": selected_facility_uids,
                    "facility_names": selected_facility_names,
                    "date_range": llm_result.get("date_range"),
                    "entity_type": llm_result.get("entity_type"),
                    "count_requested": llm_result.get("count_requested"),
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
            
            # Detect Entity
            if "region" in query_lower:
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

        # Normalize spaces and remove some punctuation for fuzzy matching
        # Fix common typos manually
        query_norm = re.sub(r'[^a-z0-9\s]', '', query_lower)
        # Phase 7 Typos
        query_norm = query_norm.replace("sectioin", "section").replace("sectionn", "section")
        query_norm = query_norm.replace("still birth", "stillbirth").replace("c section", "csection")
        query_norm = query_norm.replace("birht", "birth").replace("oucome", "outcome")
        query_norm = query_norm.replace("indicatofrs", "indicators").replace("totaly", "totally")
        query_norm = query_norm.replace("uterotoncic", "uterotonic").replace("wome", "women").replace("whor", "who")
        query_norm = query_norm.replace("abot", "about").replace("abut", "about")
        
        # Comprehensive KPI Map based on dash_co.KPI_MAPPING
        kpi_map = {
            # Standard KPIs
            "csection": "C-Section Rate (%)",
            "c section": "C-Section Rate (%)",
            "caesarean": "C-Section Rate (%)",
            "maternal death": "Institutional Maternal Death Rate (%)",
            "stillbirth": "Stillbirth Rate (%)",
            "pph": "Postpartum Hemorrhage (PPH) Rate (%)",
            "hemorrhage": "Postpartum Hemorrhage (PPH) Rate (%)",
            "bleeding": "Postpartum Hemorrhage (PPH) Rate (%)",
            "uterotonic": "Delivered women who received uterotonic (%)",
            "oxytocin": "Delivered women who received uterotonic (%)",
            "ippcar": "Immediate Postpartum Contraceptive Acceptance Rate (IPPCAR %)",
            "contraceptive": "Immediate Postpartum Contraceptive Acceptance Rate (IPPCAR %)",
            "family planning": "Immediate Postpartum Contraceptive Acceptance Rate (IPPCAR %)",
            "pnc": "Early Postnatal Care (PNC) Coverage (%)",
            "postnatal": "Early Postnatal Care (PNC) Coverage (%)",
            "arv": "ARV Prophylaxis Rate (%)",
            "antiretroviral": "ARV Prophylaxis Rate (%)",
            "assisted delivery": "Assisted Delivery Rate (%)",
            "instrumental": "Assisted Delivery Rate (%)",
            "vacuum": "Assisted Delivery Rate (%)",
            "forceps": "Assisted Delivery Rate (%)",
            "svd": "Normal Vaginal Delivery (SVD) Rate (%)",
            "vaginal delivery": "Normal Vaginal Delivery (SVD) Rate (%)",
            "normal delivery": "Normal Vaginal Delivery (SVD) Rate (%)",
            "spontaneous": "Normal Vaginal Delivery (SVD) Rate (%)",
            
            # Data Quality / Counts
            "missing mode": "Missing Mode of Delivery",
            "missing birth": "Missing Birth Outcome",
            "birth outcome": "Missing Birth Outcome", # Explicit map
            "missing outcome": "Missing Birth Outcome",
            "missing condition": "Missing Condition of Discharge",
            "missing discharge": "Missing Condition of Discharge",
            "admitted mothers": "Admitted Mothers",
            "admitted": "Admitted Mothers", # Explicit map
            "admissions": "Admitted Mothers",
            "admission": "Admitted Mothers", # Added singular
            "total mothers": "Admitted Mothers",
            
            # Additional commonly requested counts
            "total deliveries": "Total Deliveries",
            "deliveries": "Total Deliveries",
            "births": "Total Deliveries"
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
            # Basic check: if facility name is in query
            if name.lower() in query_norm:
                selected_facility_uids.append(uid)
                selected_facility_names.append(name)
        
        # If no facility found, check REGIONS
        if not selected_facility_uids:
            regions_data = get_facilities_grouped_by_region(self.user)
            found_regions = []
            
            # Check Match for Regions (Multiple allowed for comparison)
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
            
            if found_regions:
                # If comparison mode is active (via keywords), align entity
                if comparison_mode and not comparison_entity:
                    comparison_entity = "region"
                
                # If Comparison Mode (explicit or implicit), store these regions
                if comparison_mode and comparison_entity == "region":
                     # We will use 'comparison_targets' to store the list
                     pass 
                else:
                     # Standard mode - treat as aggregation filter
                     pass

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
        
        if "this month" in query_lower:
            start_date = today.replace(day=1).strftime("%Y-%m-%d")
            end_date = today.strftime("%Y-%m-%d")
        elif "last month" in query_lower:
            formatted_today = today.replace(day=1)
            last_month_end = formatted_today - timedelta(days=1)
            last_month_start = last_month_end.replace(day=1)
            start_date = last_month_start.strftime("%Y-%m-%d")
            end_date = last_month_end.strftime("%Y-%m-%d")
        elif "last year" in query_lower:
            # Use Calendar Year logic (User Request)
            last_year = today.year - 1
            start_date = f"{last_year}-01-01"
            end_date = f"{last_year}-12-31"
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
            end_date = end_date_dt.strftime("%Y-%m-%d")

        # Fallback Strict Regex Date Parsing (e.g. "Jan 1 2026 to Jan 18 2026")
        if not start_date:
            try:
                # Pattern: "from Jan 1 2026 to Jan 18 2026" or "Jan 1 2026"
                # Need a simple parser or regex. 
                # Let's cover "Month DD YYYY" specifically as per user example.
                date_matches = re.findall(r'([a-zA-Z]{3,9})\s+(\d{1,2})\s+(\d{4})', query)
                if len(date_matches) >= 2:
                    # Assume first is start, second is end
                    m1, d1, y1 = date_matches[0]
                    m2, d2, y2 = date_matches[1]
                    start_date = datetime.strptime(f"{m1} {d1} {y1}", "%b %d %Y").strftime("%Y-%m-%d")
                    end_date = datetime.strptime(f"{m2} {d2} {y2}", "%b %d %Y").strftime("%Y-%m-%d")
                elif len(date_matches) == 1:
                    # Single date? Maybe specific day.
                    m1, d1, y1 = date_matches[0]
                    start_date = datetime.strptime(f"{m1} {d1} {y1}", "%b %d %Y").strftime("%Y-%m-%d")
                    end_date = start_date # Single day
            except Exception as e:
                logging.warning(f"Date regex parse failed: {e}")

        # Fallback for "Jan 1 - Jan 18 2026" (Year only at end)
        if not start_date:
             try:
                 # Pattern: Month DD [-/to] Month DD YYYY
                 # Catch "Jan 1 - Jan 18 2026"
                 range_match = re.search(r'([a-zA-Z]{3,9})\s+(\d{1,2})\s*[-to]+\s*([a-zA-Z]{3,9})\s+(\d{1,2})[\s,]+(\d{4})', query, re.IGNORECASE)
                 if range_match:
                     m1, d1, m2, d2, y = range_match.groups()
                     start_date = datetime.strptime(f"{m1} {d1} {y}", "%b %d %Y").strftime("%Y-%m-%d")
                     end_date = datetime.strptime(f"{m2} {d2} {y}", "%b %d %Y").strftime("%Y-%m-%d")
             except Exception as e:
                 logging.warning(f"Date regex range parse failed: {e}")

        # Fallback for "Jan 1 - 18 2026" (Month occurring once, two days)
        if not start_date:
             try:
                 # Pattern: Month DD - DD YYYY
                 short_range_match = re.search(r'([a-zA-Z]{3,9})\s+(\d{1,2})\s*[-to]+\s*(\d{1,2})[\s,]+(\d{4})', query, re.IGNORECASE)
                 if short_range_match:
                     m, d1, d2, y = short_range_match.groups()
                     start_date = datetime.strptime(f"{m} {d1} {y}", "%b %d %Y").strftime("%Y-%m-%d")
                     end_date = datetime.strptime(f"{m} {d2} {y}", "%b %d %Y").strftime("%Y-%m-%d")
             except Exception as e:
                 logging.warning(f"Date short range regex parse failed: {e}")

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
        
        # Explicitly check for clearing dates
        reset_date = False
        if any(x in query_lower for x in ["overall", "all time", "since beginning", "from start", "total", "entire period"]):
             # Only reset if we are talking about time, or if broadly applied
             # "total" is tricky because "total admitted mothers" could mean count for THIS period.
             # So we look for time-bound phrases specifically or "overall"
             if "overall" in query_lower or "all time" in query_lower or "start" in query_lower:
                 reset_date = True
                 start_date = None
                 end_date = None
            
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
        if "what is" in query_lower or "define" in query_lower or "meaning of" in query_lower:
             # If user asks for "value", "rate", "count", "number", they want DATA, not definition.
             if not any(x in query_lower for x in ["value", "rate", "count", "number", "score", "percentage", "trend", "plot"]):
                 intent = "definition"

        # Robust List Detection
        # Robust List Detection
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
        if "how many" in query_lower or "list" in query_lower or "show me" in query_lower:
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
        if "all region" in query_lower or "all facilities" in query_lower:
            selected_facility_uids = []
            selected_facility_names = []
            
        # Detect "By Facility" intent if KPI is present (Disaggregation)
        # If user says "by facility" but intent was detected as "text" or "plot", 
        # normally the graph handles it by showing a line. But if they want a breakdown/list of values...
        if "by facility" in query_lower and (selected_kpi or context.get("kpi")):
             # This is a plot/data request, NOT a metadata listing of just names.
             # We ensure intent is plot so we get a chart/table.
             intent = "plot"
             # We might want to force a specific chart type or ensure title reflects it.
             # For now, let the standard plot logic handle it (it filters by selected facilities).
             # If no facilities selected, it plots aggregate? 
             # Actually, the user wants disaggregation. Our current logic aggregates unless we have a breakdown feature.
             # If we don't have Disaggregation logic, we can at least ensure we don't treat it as metadata query.
             if entity_type == "facility": 
                  # If regex caught "facility" and made it metadata_query, revert it if we have a KPI
                  entity_type = None 
                  intent = "plot" 

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
        parsed = self.parse_query(query)
        
        # Handle List KPIs Intent (New)
        if parsed.get("intent") == "list_kpis":
            # If we are here, we passed the scope check or explicit maternal check
            if "newborn" in query.lower():
                 return None, "I currently specialize in **Maternal Health** data. Newborn indicators are coming soon in the next update! 👶"
            
            # List Maternal KPIs
            from utils.dash_co import KPI_MAPPING
            kpi_list = [k for k in KPI_MAPPING.keys()]
            # Format nicely
            response = "Here are the available **Maternal Health Indicators**:\n\n"
            for kpi in kpi_list:
                response += f"- {kpi}\n"
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
             
        # Handle List KPIs
        if parsed.get("intent") == "list_kpis":
             from utils.dash_co import KPI_MAPPING
             kpi_list = "\n".join([f"- **{k}**" for k in KPI_MAPPING.keys()])
             msg = f"Here are the available health indicators in this dashboard:\n\n{kpi_list}"
             if "how many" in query_lower or "total" in query_lower:
                 msg = f"There are **{len(KPI_MAPPING)}** available indicators:\n\n{kpi_list}"
             return None, msg
             
        # Handle Newborn Scope Error
        if parsed.get("intent") == "scope_error_newborn":
            return None, "I am currently optimized for **Maternal Health** indicators. Access to Newborn data is being integrated and will be available soon! 👶"
        
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
             facility_names = parsed.get("facility_names", []) # May contain region names
             
             # If asking about Regions
             if entity_type == "region":
                 regions_data = get_facilities_grouped_by_region(self.user)
                 region_names = list(regions_data.keys())
                 
                 if count_requested:
                     return None, f"There are **{len(region_names)}** regions available."
                 else:
                     return None, f"The available regions are:\n- " + "\n- ".join(region_names)
                     
             # If asking about Facilities
             elif entity_type == "facility":
                 regions_data = get_facilities_grouped_by_region(self.user)
                 
                 # Check if specific region mentioned
                 target_region = None
                 # Try to match facility_names (which might contain region name from LLM) to region list
                 if facility_names:
                     # Check direct match or fuzzy
                     for potential_region in facility_names:
                         matches = difflib.get_close_matches(potential_region, regions_data.keys(), n=1, cutoff=0.6)
                         if matches:
                             target_region = matches[0]
                             break
                 
                 if target_region:
                     facilities = regions_data.get(target_region, [])
                     if count_requested:
                         return None, f"There are **{len(facilities)}** facilities in **{target_region}**."
                     else:
                         fac_names = [f[0] for f in facilities]
                         return None, f"Here are the facilities in **{target_region}**:\n- " + "\n- ".join(fac_names)
                 else:
                     # Global
                     all_facilities = get_all_facilities_flat(self.user)
                     if count_requested:
                         return None, f"There are a total of **{len(all_facilities)}** facilities."
                     else:
                         # Too many to list?
                         if len(all_facilities) > 50:
                             return None, f"There are {len(all_facilities)} facilities. That's too many to list here! Can you specify a region?"
                         else:
                             fac_names = [f[0] for f in all_facilities]
                             return None, f"Here are the available facilities:\n- " + "\n- ".join(fac_names)
             
             return None, "I'm not sure which entity (region or facility) you are asking about."
        
        # Update Context even for Metadata queries (so "Total" can be answered next)
        if parsed.get("intent") == "metadata_query":
             st.session_state["chatbot_context"]["entity_type"] = entity_type
             return None, None # Should have returned above, but just in case
        
        if not parsed["kpi"]:
            return None, "I couldn't identify the specific health indicator (KPI) you're asking about. Try phrases like 'C-Section Rate', 'PPH Rate', or 'Total Deliveries'."

        # Update Session State Context & Filters
        st.session_state["chatbot_context"] = {
            "kpi": parsed["kpi"],
            "facility_uids": parsed["facility_uids"],
            "facility_names": parsed["facility_names"],
            "date_range": parsed["date_range"],
            "entity_type": parsed.get("entity_type") # Persist for follow-up
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
            "Admitted Mothers": {"host_kpi": "Institutional Maternal Death Rate (%)", "component": "denominator"},
            "Maternal Deaths": {"host_kpi": "Institutional Maternal Death Rate (%)", "component": "numerator"},
        }
        
        target_component = "value" # default
        active_kpi_name = kpi_name
        
        if kpi_name in KPI_COMPONENT_MAPPING:
             active_kpi_name = KPI_COMPONENT_MAPPING[kpi_name]["host_kpi"]
             target_component = KPI_COMPONENT_MAPPING[kpi_name]["component"]
        
        # --- HANDLE EXPLAIN INTENT ---
        if parsed["intent"] == "explain":
            from utils.dash_co import KPI_MAPPING
            kpi_info = KPI_MAPPING.get(active_kpi_name, {})
            numerator_desc = kpi_info.get("numerator_name", "the numerator")
            denominator_desc = kpi_info.get("denominator_name", "the denominator")
            
            explanation = f"**{kpi_name}** explanation:\n- **Numerator**: {numerator_desc}\n- **Denominator**: {denominator_desc}"
            if target_component != "value":
                explanation += f"\n\nSince you asked for **{kpi_name}**, I'm extracting the **{target_component}** from this indicator."
                
            return None, explanation

        # --- HANDLE DEFINITION INTENT ---
        if parsed["intent"] == "definition":
            # Lookup in Static KB
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
                 # Fallback (Simulated LLM response for now, to be robust against 429s)
                 return None, f"I couldn't find a specific definition for **{kpi_name}** in my knowledge base. However, generally in this dashboard, it refers to the tracked health indicator for {kpi_name}."

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
        
        # If Intent is Plot OR Analysis is requested (to get trend data)
        if parsed["intent"] == "plot" or parsed.get("analysis_type"):
            prepared_df, date_col = prepare_data_for_trend_chart(
                self.df, 
                active_kpi_name, 
                facility_uids, 
                date_range
            )
            
            if prepared_df is None or prepared_df.empty:
                # If comparison mode, we might still have data for other regions?
                # Actually if initial fetch failed, it might be due to filters.
                # If comparison mode, we ignore this initial check and let the loop handle it
                if not parsed.get("comparison_mode"):
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
            
            comparison_groups = [] # List of (EntityName, FilteredDF)
            
            if comparison_mode:
                if comparison_entity == "region":
                    # Fetch all regions
                    regions_data = get_facilities_grouped_by_region(self.user)
                    comp_targets = parsed.get("comparison_targets", [])
                    
                    for r_name, facilities in regions_data.items():
                         # Filter if specific targets requested
                         if comp_targets and r_name not in comp_targets:
                             continue
                             
                         # Filter main DF by these facilities
                         r_uids = [f[1] for f in facilities]
                         comparison_groups.append((r_name, r_uids))
                         
                elif comparison_entity == "facility":
                    # Use selected facilities
                    # IF only 1 facility selected, comparison mode implies we want to compare it vs others?
                    # Or maybe user listed specific facilities.
                    for name, uid in zip(parsed["facility_names"], parsed["facility_uids"]):
                        comparison_groups.append((name, [uid]))
            
            # If NOT comparison mode (or failed setup), default to single group
            if not comparison_groups:
                 comparison_groups.append(("Overall", facility_uids))

            # Loop through Groups and Build Data
            chart_data = []
            
            for entity_name, entity_uids in comparison_groups:
                 # Re-fetch/Filter data for this entity
                 # We need to call prepare_data_for_trend_chart again OR filter the raw DF?
                 # Calling prepare_data is safer as it handles dates/missing cols.
                 # Optimization: If prepared_df already has everything, filtering is faster.
                 # But prepared_df was built with initial 'facility_uids'. 
                 # If comparison, initial uids might have been empty (global).
                 
                 entity_df, _ = prepare_data_for_trend_chart(self.df, active_kpi_name, entity_uids, date_range)
                 
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
                     if active_kpi_name == "Admitted Mothers":
                         from utils.kpi_admitted_mothers import get_numerator_denominator_for_admitted_mothers
                         numerator, denominator, value = get_numerator_denominator_for_admitted_mothers(group_df, entity_uids, date_range)
                     else:
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
                         "SortDate": sort_val
                     })
            
            plot_df = pd.DataFrame(chart_data)
            if not plot_df.empty and "SortDate" in plot_df.columns:
                plot_df.sort_values("SortDate", inplace=True)
            
            if plot_df.empty:
                return None, "Data processing resulted in empty dataset."
                
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

                # ADD OVERALL ROW (Standard Mode)
                total_kpi_value = 0
                total_numerator = plot_df["Numerator"].sum()
                total_denominator = plot_df["Denominator"].sum()
                
                # Re-calculate overall rate
                if total_denominator > 0:
                     if target_component == "value":
                         # Default is Rate %
                         total_kpi_value = (total_numerator / total_denominator) * 100
                     else:
                         # Component requested (Count)
                         if target_component == "numerator": total_kpi_value = total_numerator
                         elif target_component == "denominator": total_kpi_value = total_denominator
                else:
                    total_kpi_value = 0 if target_component == "value" else total_numerator # simplified
                    
                # Append Row
                overall_row = {
                    "Period": "Overall",
                    "Value": total_kpi_value,
                    "Numerator": total_numerator,
                    "Denominator": total_denominator
                }
                # Use concat to append
                overall_df = pd.DataFrame([overall_row])
                plot_df = pd.concat([plot_df, overall_df], ignore_index=True)
            
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
            
            # If no analysis type, proceed to standard chart/table response
            # Determine Plot Params
            color_col = "Entity" if "Entity" in plot_df.columns and plot_df["Entity"].nunique() > 1 else None
            
            # Dynamic Chart Generation
            if chart_type == "table":
                # ... (Keep existing table logic, maybe adapt for comparison later or let it show raw long-form table)
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
            
            return fig, f"Here is the {chart_type} chart for **{kpi_name}**.{suggestion}"
            
        else:
            # Text Response (Single value)
            prepared_df, date_col = prepare_data_for_trend_chart(
                self.df, 
                active_kpi_name, 
                facility_uids, 
                date_range
            )
            
            if prepared_df is None or prepared_df.empty:
                return None, f"No data found for **{kpi_name}**."
                
            # SPECIAL: Admitted Mothers (Count based)
            if active_kpi_name == "Admitted Mothers":
                 numerator, denominator, value = get_numerator_denominator_for_admitted_mothers(self.df, facility_uids, date_range)
                 # Re-assign if date_range was passed to function manually, or use prepared_df logic?
                 # Actually, generic get_n_d uses prepared_df. Specific one uses DF + Filters.
                 # Let's trust generic flow for 'value' extraction if possible, BUT admitted uses specific logic.
                 # Let's use the VALUE returned by the specific function.
            else:
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
            
            return None, response_text + suggestion


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
        .stChatMessage {
            background-color: #ffffff;
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
        "Missing Condition of Discharge": "Data Quality Metric: The percentage of maternal discharge records where the mother's condition (e.g., Discharged Healthy, Referred, Death) was not recorded."
    }

    # Display chat messages from history on app rerun
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if "content" in message:
                st.markdown(message["content"])
            if "figure" in message:
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
                    if fig is not None:
                        if isinstance(fig, pd.DataFrame):
                            st.dataframe(fig)
                        else:
                            st.plotly_chart(fig, use_container_width=True, key=f"chat_chart_new_{len(st.session_state.messages)}")
                        
                    # Save to history
                    msg_obj = {"role": "assistant", "content": response_text}
                    if fig is not None: 
                        # Save Dataframe compatible object logic
                        # If fig is DF, we save it. If Plotly, save it.
                         msg_obj["figure"] = fig
                    st.session_state.messages.append(msg_obj)
                    
                    if response_text == "Chat history cleared.":
                        st.rerun()
                        
                except Exception as e:
                    logging.info(f"Error details: {e}")
                    error_msg = f"I encountered an error analyzing your request: {str(e)}"
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
- **🗺️ Comparisons**: I can compare performance between different regions or facilities for a specific indicator.
- **📚 Definitions**: Ask "What is [Indicator]?" to get a medical definition.
- **📥 Data Tables**: Ask for "table format" to see the raw numbers.

**What I cannot do:**
- 🚫 I **cannot** generate, update, or modify health data. All data is read-only.
- 🚫 I cannot predict future data (forecasting is not yet enabled).

**Try asking:**
- "Plot C-Section Rate last year"
- "Show me Admitted Mothers"
- "Compare PPH Rate for all facilities"

Type **'Help'** at any time to see this message again.
"""
    return msg
