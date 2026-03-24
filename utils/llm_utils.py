import requests
import json
import logging
import re
from datetime import datetime
from utils.config import settings

logging.basicConfig(level=logging.INFO)

def build_system_prompt_compact(*, facilities_list=None, kpi_mapping=None, program_label=None):
    """
    Compact prompt to reduce token usage and avoid program-specific alias noise.
    """
    if kpi_mapping is None:
        # Backward-compatible default (maternal)
        from utils.dash_co import KPI_MAPPING as kpi_mapping  # local import to avoid import-time cost

    program_label = program_label or "Health"
    today = datetime.now().strftime("%Y-%m-%d")

    kpi_descriptions = [f"- {kpi}: {details.get('title', kpi)}" for kpi, details in kpi_mapping.items()]
    kpi_context = "\n".join(kpi_descriptions)

    facility_context = ""
    if facilities_list:
        if isinstance(facilities_list, (list, tuple)) and facilities_list and isinstance(facilities_list[0], (list, tuple)):
            fac_names = [str(f[0]) for f in facilities_list]
        else:
            fac_names = [str(f) for f in facilities_list]

        if len(fac_names) > 150:
            facility_context = f"Available Facilities (first 150): {', '.join(fac_names[:150])}..."
        else:
            facility_context = f"Available Facilities: {', '.join(fac_names)}"

    phrase_hints = []

    def _hint(phrase: str, kpi_name: str):
        if kpi_name in kpi_mapping:
            phrase_hints.append(f'- "{phrase}" -> "{kpi_name}"')

    # High-value, low-noise hints for the LLM (keep this list short)
    _hint("c-section / cesarean", "C-Section Rate (%)")
    _hint("pph", "Postpartum Hemorrhage (PPH) Rate (%)")
    _hint("stillbirth", "Stillbirth Rate (%)")
    _hint("pnc", "Early Postnatal Care (PNC) Coverage (%)")
    _hint("ippcar / family planning", "Immediate Postpartum Contraceptive Acceptance Rate (IPPCAR %)")
    _hint("cpap coverage", "CPAP Coverage by Birth Weight")
    _hint("cpap rds", "CPAP for RDS")
    _hint("kmc coverage", "KMC Coverage by Birth Weight")
    _hint("birth weight distribution", "Birth Weight Rate")

    hints_section = ""
    if phrase_hints:
        hints_section = "COMMON PHRASES (choose the exact KPI name):\n" + "\n".join(phrase_hints)

    return f"""
You are an intelligent assistant for a {program_label} Dashboard.
Convert the user's message into a single JSON object that the dashboard can execute.

Current Date: {today}

VALID KPIs (use exact names; pick one or null):
{kpi_context}

FACILITY CONTEXT (names only; optional):
{facility_context}

{hints_section}

RULES:
- Output MUST be valid JSON only (no markdown, no backticks).
- The user message may include `PREVIOUS_CONTEXT: {...}` then `USER_QUERY: ...`. Use PREVIOUS_CONTEXT only for follow-ups; prioritize USER_QUERY.
- If the user asks what indicators/KPIs exist -> intent="list_kpis".
- If the user asks to list/show facilities or regions (or asks "how many") -> intent="metadata_query", set entity_type, set count_requested, and set region_filter when asking for facilities in a specific region.
- If the user greets / asks for help / asks unrelated questions -> intent="chat" and include a short "response" that redirects to dashboard analysis.
- If asked about passwords/login/admin access -> intent="chat" and respond that you cannot help with passwords or system administration.
- If the user asks to reset/clear -> intent="clear".
- If the user asks to see/show/plot/graph/chart/trend/visualize a KPI (or mentions a KPI without asking for a single number) -> intent="plot".
- If the user asks "what is", "how many", "count", "number", or asks for a single value -> intent="text" (unless they also asked to plot).
- If the user asks for a definition/meaning/formula -> intent="definition".
- chart_type: "line" | "bar" | "area" | "table" (default "line").
- period_label: null or "Daily" | "Weekly" | "Monthly" | "Quarterly" | "Yearly".
- analysis_type: null or "max" | "min" (highest/lowest). If set, intent should usually be "text".
- orientation: null or "h" | "v" (use "h" when user asks for a horizontal bar chart).
- date_range: null or {{ "start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD" }}.
- comparison_mode: true when the user compares facilities/regions ("compare", "vs", "by facility", "by region").

OUTPUT JSON SCHEMA:
{{
  "intent": "plot" | "distribution" | "text" | "definition" | "clear" | "metadata_query" | "chat" | "list_kpis",
  "kpi": "Exact KPI Name from list" | null,
  "chart_type": "line" | "bar" | "area" | "table",
  "facility_names": [string] | [],
  "date_range": {{ "start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD" }} | null,
  "period_label": "Daily" | "Weekly" | "Monthly" | "Quarterly" | "Yearly" | null,
  "analysis_type": "max" | "min" | null,
  "orientation": "h" | "v" | null,
  "entity_type": "region" | "facility" | null,
  "count_requested": true | false,
  "comparison_mode": true | false,
  "comparison_entity": "region" | "facility" | null,
  "region_filter": string | null,
  "response": string | null
}}
""".strip()


def build_system_prompt(*, facilities_list=None, kpi_mapping=None, program_label=None):
    """
    Constructs the system prompt with domain knowledge.
    """
    
    if kpi_mapping is None:
        # Backward-compatible default (maternal)
        from utils.dash_co import KPI_MAPPING as kpi_mapping  # local import to avoid import-time cost

    program_label = program_label or "Health"

    # 1. Extract KPI definitions
    kpi_descriptions = [f"- {kpi}: {details.get('title', kpi)}" for kpi, details in kpi_mapping.items()]
         
    kpi_context = "\n".join(kpi_descriptions)
    
    # 2. Facility Context (Limit to first 50 to avoid token limit if list is huge, usually it's small)
    facility_context = ""
    if facilities_list:
        if isinstance(facilities_list, (list, tuple)) and facilities_list and isinstance(facilities_list[0], (list, tuple)):
            fac_names = [str(f[0]) for f in facilities_list]
        else:
            fac_names = [str(f) for f in facilities_list]
              
        if len(fac_names) > 150:
            facility_context = f"Available Facilities (first 150): {', '.join(fac_names[:150])}..."
        else:
            facility_context = f"Available Facilities: {', '.join(fac_names)}"
            
    today = datetime.now().strftime("%Y-%m-%d")

    prompt = f"""
You are an intelligent assistant for a {program_label} Dashboard.
Your job is to parse user queries into structured JSON for the dashboard's data engine.

IMPORTANT: Only use the KPI list provided below. If the user asks for something outside these KPIs, set kpi=null and intent="chat" (with a helpful response) or intent="list_kpis" when appropriate.

Current Date: {today}

DOMAIN KNOWLEDGE (KPIs):
The following are the valid Health Indicators (KPIs) available in the system:
{kpi_context}

CONTEXT:
{facility_context}

INSTRUCTIONS:
1. Identify the user's INTENT:
   - "plot": for graphs, charts, trends, or specific data values over time.
   - "distribution": for pie charts, breakdowns by category, proportions, or composition (e.g., "what is the breakdown of complications").
   - "text": for single values, summaries, or specific numbers without a chart.
   - "definition": for questions asking what an indicator means or how it is calculated.
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
    - "maternal death"/"mortality"/"materna death" → "Maternal Death Rate (per 100,000)"
   - "stillbirth"/"still birth"/"stil birth"/"stillbrith" → "Stillbirth Rate (%)"
   - "missing mode"/"missing delivery"/"mode of delivery" → "Missing Mode of Delivery"
    - "missing birth outcome"/"missing birth"/"missing oucome" → "Missing Birth Outcome"
    - "missing discharge"/"missing condition"/"mode of discharge"/"discharge condition" → "Missing Condition of Discharge"
    - "missing obstetric condition"/"missing condition at delivery"/"missing postpartum" → "Missing Obstetric Condition at Delivery"
    - "missing complications diagnosis"/"missing antepartum"/"missing diagnosis"/"missing ante" → "Missing Obstetric Complications Diagnosis"
    - "missing uterotonics given"/"missing oxytocin given"/"missing uterotonic at delivery"/"missing utertonic"/"missing uterotonic" → "Missing Uterotonics Given at Delivery"
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
  "intent": "plot" | "distribution" | "text" | "definition" | "clear" | "metadata_query" | "chat" | "list_kpis",
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

def _extract_first_json_object(text: str):
    """Best-effort extraction of the first JSON object in a string."""
    if not text:
        return None

    start = text.find("{")
    if start == -1:
        return None

    brace_depth = 0
    end = None
    for idx in range(start, len(text)):
        ch = text[idx]
        if ch == "{":
            brace_depth += 1
        elif ch == "}":
            brace_depth -= 1
            if brace_depth == 0:
                end = idx
                break

    if end is None:
        return None

    candidate = text[start : end + 1].strip()
    try:
        return json.loads(candidate)
    except Exception:
        return None


def _parse_json_from_model_text(text: str):
    if not text:
        return None

    cleaned = str(text).replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        parsed = _extract_first_json_object(cleaned)
        return parsed if isinstance(parsed, dict) else None


def _get_llm_provider():
    provider = getattr(settings, "LLM_PROVIDER", None)
    if provider:
        return str(provider).strip().lower()

    # Auto-detect when provider isn't explicitly configured
    if getattr(settings, "GEMINI_API_KEY", None):
        return "gemini"
    if getattr(settings, "OPENAI_API_KEY", None):
        return "openai"
    return None


def get_llm_provider_and_model():
    """
    Return (provider, model) for the active LLM configuration.

    provider is one of: "gemini", "openai", or None.
    model is a string (or None if unknown/not configured).
    """
    provider = _get_llm_provider()
    if provider == "gemini":
        model = (getattr(settings, "GEMINI_MODEL", None) or "gemini-2.5-flash-lite").strip()
        return provider, model
    if provider == "openai":
        model = (getattr(settings, "OPENAI_MODEL", None) or "gpt-4o-mini").strip()
        return provider, model
    return provider, None


def format_llm_label():
    """User-facing label like 'Gemini / gemini-2.0-flash' (or None if not configured)."""
    provider, model = get_llm_provider_and_model()
    if not provider:
        return None

    if provider == "openai":
        provider_label = "OpenAI"
    else:
        provider_label = provider.capitalize()

    if model:
        return f"{provider_label} / {model}"
    return provider_label


def _query_openai_chat_completions(*, user_query: str, system_prompt: str):
    api_key = getattr(settings, "OPENAI_API_KEY", None)
    if not api_key:
        return None

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    payload = {
        "model": getattr(settings, "OPENAI_MODEL", "gpt-4o-mini"),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
        "temperature": 0.0,
    }

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=getattr(settings, "OPENAI_TIMEOUT", 10),
        )
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        return _parse_json_from_model_text(content)
    except Exception:
        return None


def _query_gemini_generate_content(*, user_query: str, system_prompt: str):
    api_key = getattr(settings, "GEMINI_API_KEY", None)
    if not api_key:
        return None

    model = (getattr(settings, "GEMINI_MODEL", None) or "gemini-2.5-flash-lite").strip()
    model_resource = model.lstrip("/")
    if "/" not in model_resource:
        model_resource = f"models/{model_resource}"

    url = f"https://generativelanguage.googleapis.com/v1beta/{model_resource}:generateContent"

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }

    payload = {
        # Gemini API expects camelCase field name.
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_query}],
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json",
        },
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=getattr(settings, "GEMINI_TIMEOUT", 10),
        )
        response.raise_for_status()
        result = response.json()

        candidates = result.get("candidates") or []
        if not candidates:
            return None

        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
        return _parse_json_from_model_text(text)
    except Exception:
        return None


def _redact_secrets(text: str):
    if not text:
        return text

    redacted = str(text)

    for secret in (getattr(settings, "GEMINI_API_KEY", None), getattr(settings, "OPENAI_API_KEY", None)):
        if isinstance(secret, str) and secret:
            redacted = redacted.replace(secret, "<redacted>")

    # Generic patterns (defense-in-depth)
    redacted = re.sub(r"AIza[0-9A-Za-z\-_]{20,}", "<redacted>", redacted)
    redacted = re.sub(r"(api_key[:=])\s*[0-9A-Za-z\-_]+", r"\1<redacted>", redacted, flags=re.IGNORECASE)
    redacted = re.sub(r"(Bearer\s+)[0-9A-Za-z\-_\.]+", r"\1<redacted>", redacted, flags=re.IGNORECASE)
    redacted = re.sub(r"sk-[0-9A-Za-z]{20,}", "sk-<redacted>", redacted)

    return redacted


def _format_http_error(prefix: str, response: requests.Response):
    try:
        payload = response.json()
    except Exception:
        payload = None

    message = None
    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, dict):
            message = err.get("message") or err.get("status")
        if not message:
            message = payload.get("message")

    if not message:
        message = (response.text or "").strip()

    message = " ".join(str(message).split())
    message = _redact_secrets(message)
    if len(message) > 240:
        message = message[:239] + "…"

    return f"{prefix} (HTTP {response.status_code}): {message}"


def _query_openai_chat_completions_detailed(*, user_query: str, system_prompt: str):
    api_key = getattr(settings, "OPENAI_API_KEY", None)
    if not api_key:
        return None, "OPENAI_API_KEY is not set"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    payload = {
        "model": getattr(settings, "OPENAI_MODEL", "gpt-4o-mini"),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
        "temperature": 0.0,
    }

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=getattr(settings, "OPENAI_TIMEOUT", 10),
        )
    except Exception as exc:
        return None, _redact_secrets(f"OpenAI request failed: {exc}")

    if response.status_code >= 400:
        return None, _format_http_error("OpenAI error", response)

    try:
        result = response.json()
        content = result["choices"][0]["message"]["content"]
    except Exception:
        return None, "OpenAI response parse error"

    parsed = _parse_json_from_model_text(content)
    if not isinstance(parsed, dict):
        return None, "OpenAI returned non-JSON output"
    return parsed, None


def _query_gemini_generate_content_detailed(*, user_query: str, system_prompt: str):
    api_key = getattr(settings, "GEMINI_API_KEY", None)
    if not api_key:
        return None, "GEMINI_API_KEY is not set"

    model = (getattr(settings, "GEMINI_MODEL", None) or "gemini-2.5-flash-lite").strip()
    model_resource = model.lstrip("/")
    if "/" not in model_resource:
        model_resource = f"models/{model_resource}"

    url = f"https://generativelanguage.googleapis.com/v1beta/{model_resource}:generateContent"

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }

    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_query}],
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json",
        },
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=getattr(settings, "GEMINI_TIMEOUT", 10),
        )
    except Exception as exc:
        return None, _redact_secrets(f"Gemini request failed: {exc}")

    if response.status_code >= 400:
        return None, _format_http_error("Gemini error", response)

    try:
        result = response.json()
    except Exception:
        return None, "Gemini response parse error"

    candidates = result.get("candidates") or []
    if not candidates:
        return None, "Gemini returned no candidates"

    parts = (candidates[0].get("content") or {}).get("parts") or []
    text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
    parsed = _parse_json_from_model_text(text)
    if not isinstance(parsed, dict):
        return None, "Gemini returned non-JSON output"

    return parsed, None


def query_llm_detailed(user_query, facilities_list=None, *, kpi_mapping=None, program_label=None):
    """
    Sends query to the configured LLM provider and returns (parsed_json, error_message).

    - parsed_json is a dict on success, otherwise None.
    - error_message is a short string on failure, otherwise None.
    """
    if not getattr(settings, "CHATBOT_USE_LLM", False):
        return None, "CHATBOT_USE_LLM is disabled"

    system_prompt = build_system_prompt_compact(
        facilities_list=facilities_list,
        kpi_mapping=kpi_mapping,
        program_label=program_label,
    )

    provider = _get_llm_provider()
    if provider == "gemini":
        return _query_gemini_generate_content_detailed(user_query=user_query, system_prompt=system_prompt)
    if provider == "openai":
        return _query_openai_chat_completions_detailed(user_query=user_query, system_prompt=system_prompt)
    return None, "No LLM provider configured (set LLM_PROVIDER + API key)"


def query_llm_with_prompt(*, user_prompt: str, system_prompt: str):
    """
    Provider-agnostic helper that returns a parsed JSON dict (or None).
    """
    if not getattr(settings, "CHATBOT_USE_LLM", False):
        return None

    provider = _get_llm_provider()
    if provider == "gemini":
        return _query_gemini_generate_content(user_query=user_prompt, system_prompt=system_prompt)
    if provider == "openai":
        return _query_openai_chat_completions(user_query=user_prompt, system_prompt=system_prompt)
    return None


def generate_chatbot_insight(*, stats: dict, program_label: str):
    """
    Generate a short, safe narrative insight from already-computed stats.

    Returns a string or None. Never sends raw patient-level data.
    """
    if not getattr(settings, "CHATBOT_USE_LLM_INSIGHTS", False):
        return None

    safe_stats = stats if isinstance(stats, dict) else {"stats": str(stats)}

    system_prompt = f"""
You are a dashboard assistant for the {program_label} program.
You will receive computed summary statistics (these are the source of truth).
Write 1-3 short sentences that describe what the data shows, and suggest 1 useful follow-up question.

Rules:
- Do NOT invent numbers, facilities, regions, or time periods.
- Do NOT provide medical advice; keep it to data interpretation and next analysis steps.
- Keep it concise.

Output JSON only:
{{"insight": "..."}}
""".strip()

    user_prompt = json.dumps(safe_stats, ensure_ascii=False)
    result = query_llm_with_prompt(user_prompt=user_prompt, system_prompt=system_prompt)
    if not isinstance(result, dict):
        return None

    insight = result.get("insight")
    if not isinstance(insight, str):
        return None

    return insight.strip() or None


def query_llm(user_query, facilities_list=None, *, kpi_mapping=None, program_label=None):
    """
    Sends query to the configured LLM provider and returns parsed JSON.
    """
    if not getattr(settings, "CHATBOT_USE_LLM", False):
        return None

    system_prompt = build_system_prompt_compact(
        facilities_list=facilities_list,
        kpi_mapping=kpi_mapping,
        program_label=program_label,
    )

    result = query_llm_with_prompt(user_prompt=user_query, system_prompt=system_prompt)
    if isinstance(result, dict):
        return result

    if not getattr(settings, "OPENAI_API_KEY", None) and not getattr(settings, "GEMINI_API_KEY", None):
        logging.warning("No LLM API key set (OPENAI_API_KEY or GEMINI_API_KEY). Falling back to rule-based parser.")
    return None
