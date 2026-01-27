import requests
import json
import logging
from datetime import datetime, timedelta
from utils.config import settings
from utils.dash_co import KPI_MAPPING
from utils.dash_co import KPI_MAPPING

logging.basicConfig(level=logging.INFO)

def build_system_prompt(facilities_list=None):
    """
    Constructs the system prompt with domain knowledge.
    """
    
    # 1. Extract KPI definitions (Strictly Maternal)
    kpi_descriptions = []
    
    # Maternal
    for kpi, details in KPI_MAPPING.items():
        kpi_descriptions.append(f"- {kpi}: {details.get('title', kpi)}")
        
    kpi_context = "\n".join(kpi_descriptions)
    
    # 2. Facility Context (Limit to first 50 to avoid token limit if list is huge, usually it's small)
    facility_context = ""
    if facilities_list:
        fac_names = [f[0] for f in facilities_list] # Assuming list of tuples (name, uid) or just names if list of strings
        # Check type
        if fac_names and isinstance(fac_names[0], tuple):
             fac_names = [f[0] for f in facilities_list]
        elif fac_names:
             fac_names = facilities_list
             
        if len(fac_names) > 150:
            facility_context = f"Available Facilities (first 150): {', '.join(fac_names[:150])}..."
        else:
            facility_context = f"Available Facilities: {', '.join(fac_names)}"
            
    today = datetime.now().strftime("%Y-%m-%d")

    prompt = f"""
You are an intelligent assistant for a Maternal Health Dashboard.
Your job is to parse user queries into structured JSON for the dashboard's data engine.

IMPORTANT: You ONLY have information on Maternal Health indicators. You do NOT have any data on Newborn Care, Programs, or other health areas.

Current Date: {today}

DOMAIN KNOWLEDGE (KPIs):
The following are the valid Health Indicators (KPIs) available in the system:
{kpi_context}

CONTEXT:
{facility_context}

INSTRUCTIONS:
1. Identify the user's INTENT:
   - "plot": for graphs, charts, trends, or specific data values over time.
   - "text": for single values, summaries, or specific numbers without a chart.
   - "metadata_query": for questions about the *structure* or *entities* of the system (e.g., "how many regions?", "list all facilities", "what hospitals are in [Region]").
   - "list_kpis": for questions asking what health indicators are available.
   - "chat": for general greetings, help requests, or questions NOT related to specific data (including "password", "who are you").
   - "clear": to reset the conversation.

2. Identify the KPI: Match the user's request to one of the exact KPI names listed above.
   
   IMPORTANT: Be VERY lenient with spelling variations and typos. Users may misspell medical terms.
   
   FUZZY MATCHING ALIASES (handle typos and variations):
   
   **Maternal Indicators:**
   - "admitted mother"/"total mothers"/"mother count"/"admited"/"admision" → "Admitted Mothers"
   - "pph"/"postpartum hemorrhage"/"bleeding"/"hemorage"/"hemorrage"/"hemmorhage" → "Postpartum Hemorrhage (PPH) Rate (%)"
   - "svd"/"normal delivery"/"vaginal delivery"/"vagnal" → "Normal Vaginal Delivery (SVD) Rate (%)"
   - "c-section"/"cesarean"/"csection"/"cesarian"/"ceasarean"/"c section"/"sectioin" → "C-Section Rate (%)"
   - "arv"/"hiv prophylaxis"/"antiretroviral"/"anti retroviral" → "ARV Prophylaxis Rate (%)"
   - "assisted delivery"/"instrumental"/"assisted" → "Assisted Delivery Rate (%)"
   - "uterotonic"/"oxytocin"/"uterotoncic"/"uterotonc"/"utertonic" → "Delivered women who received uterotonic (%)"
   - "episiotomy"/"episotomy"/"episotomi"/"episiotmoy"/"episo" → "Episiotomy Rate (%)"
   - "antepartum"/"antenatal complications"/"ante partum"/"antipartum" → "Antepartum Complications Rate (%)"
   - "maternal death"/"mortality"/"materna death" → "Institutional Maternal Death Rate (%)"
   - "stillbirth"/"still birth"/"stil birth"/"stillbrith" → "Stillbirth Rate (%)"
   - "missing mode"/"missing delivery"/"mode of delivery" → "Missing Mode of Delivery"
   - "missing birth outcome"/"missing birth"/"missing oucome" → "Missing Birth Outcome"
   - "missing discharge"/"missing condition"/"mode of discharge"/"discharge condition" → "Missing Condition of Discharge"
   - "pnc"/"postnatal"/"post natal" → "Early Postnatal Care (PNC) Coverage (%)"
   - "ippcar"/"contraceptive"/"family planning"/"fp" → "Immediate Postpartum Contraceptive Acceptance Rate (IPPCAR %)"


3. Identify the CHART TYPE: "line", "bar", "area", or "table". Default to "line" unless specified.

4. Identify FILTERS:
   - "facility_names": List of facility names OR region names mentioned. 
     CRITICAL: Extract the COMPLETE and SPECIFIC name. 
     PREFER full matches over partial matches when identifying facility names.
     Example: If user says "Ambo University", extract "Ambo University" NOT just "Ambo". 
     There are many overlapping names (e.g., Ambo General vs Ambo University). DO NOT truncate.
   - "date_range": Calculate start_date and end_date (YYYY-MM-DD) based on relative time to {today}.
 
5. Identify COMPARISON MODE:
   - Set "comparison_mode" to true if user uses words like "compare", "vs", "versus".
   - CRITICAL: phrases like "by facility", "breakdown by facility", "per facility", "show facilities" ALSO imply `comparison_mode: true` and `comparison_entity: "facility"`.
   - Set "comparison_entity" to "region" or "facility" based on what is being compared.

6. For "metadata_query":
   - "entity_type": "region" | "facility"
   - "count_requested": true if asking for a number/count, false if asking for a list.
   - "region_filter": If asking for facilities in a specific region (e.g. "facilities in Tigray"), extract "Tigray".

7. For "list_kpis":
   - Use this intent if the user asks "what indicators do you have" or "list capabilities".

8. For "chat" intent:
   - "response": A helpful, friendly, natural language response.
   - If asked "how can you help", list your capabilities (plotting trends, showing values, listing facilities).
   - If asked about "passwords", "login", "admin access", or irrelevant topics, strictly reply: "I am an AI assistant for data analysis. I cannot help with system administration or passwords."

9. SPECIAL DATE HANDLING:
   - If user says "from Jan 1 2026 to Jan 18 2026", parse exactly as start_date: "2026-01-01", end_date: "2026-01-18".
   - Handle "Jan 2026" as start: "2026-01-01", end: "2026-01-31".
   - Handle "Last Year" or "Last Month" by calculating specific dates relative to Current Date.
   - Handle "This Year" or "This Month" as from start of period to Current Date.

OUTPUT FORMAT (JSON ONLY):
{{
  "intent": "plot" | "text" | "clear" | "metadata_query" | "chat" | "list_kpis",
  "kpi": "Exact KPI Name from list" | null,
  "chart_type": "line" | "bar" | "area" | "table",
  "facility_names": ["Abiadi", "Adigrat"] | [],
  "date_range": {{ "start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD" }} | null,
  "entity_type": "region" | "facility" | null,
  "count_requested": true | false,
  "comparison_mode": true | false,
  "comparison_entity": "region" | "facility" | null,
  "region_filter": "Region Name" | null,
  "response": "String response for chat intent" | null
}}

Do not include markdown formatting like ```json. Just the raw JSON string.
"""
    return prompt

def query_llm(user_query, facilities_list=None):
    """
    Sends query to OpenAI API and returns parsed JSON.
    """
    if not settings.OPENAI_API_KEY:
        logging.warning("OPENAI_API_KEY not set. Falling back to regex parser.")
        return None

    system_prompt = build_system_prompt(facilities_list)
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.OPENAI_API_KEY}"
    }
    
    payload = {
        "model": settings.OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        "temperature": 0.0
    }
    
    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        
        content = result["choices"][0]["message"]["content"]
        
        # Clean potential markdown
        if "```json" in content:
            content = content.replace("```json", "").replace("```", "")
        
        parsed = json.loads(content.strip())
        return parsed
        
    except Exception as e:
        # LLM Query errors (like 429 Too Many Requests) can be noisy.
        # Since we use rule-based logic as a robust primary fallback, we can silence these.
        # logging.error(f"LLM Query Failed: {e}")
        return None