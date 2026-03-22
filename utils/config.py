import os

class Settings:
    """
    Centralized application settings.
    All sensitive values must come from .env file.
    """

    # Database configuration (required)
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = int(os.getenv("DB_PORT", "5432"))  # default PostgreSQL port
    DB_NAME = os.getenv("DB_NAME")
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")

    # Application secret key (required for sessions or JWT)
    APP_SECRET_KEY = os.getenv("APP_SECRET_KEY")

    # DHIS2 API configuration
    DHIS2_BASE_URL = os.getenv("DHIS2_BASE_URL", "").rstrip("/")
    DHIS2_USERNAME = os.getenv("DHIS2_USERNAME")
    DHIS2_PASSWORD = os.getenv("DHIS2_PASSWORD")
    DHIS2_TIMEOUT = int(os.getenv("DHIS2_TIMEOUT", "30"))

    # LLM Configuration
    # NOTE: This module intentionally does NOT call load_dotenv() (per repo requirement).
    # Set environment variables via Streamlit secrets, the process environment, or load .env in the app entrypoint.
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "").strip().lower() or None
    CHATBOT_USE_LLM = os.getenv("CHATBOT_USE_LLM", "0").strip().lower() in {"1", "true", "yes", "y", "on"}
    CHATBOT_LLM_PARSER_MODE = os.getenv("CHATBOT_LLM_PARSER_MODE", "fallback").strip().lower()  # off|fallback|always
    CHATBOT_USE_LLM_INSIGHTS = os.getenv("CHATBOT_USE_LLM_INSIGHTS", "0").strip().lower() in {"1", "true", "yes", "y", "on"}

    # OpenAI (optional)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "10"))

    # Gemini (optional)
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    GEMINI_TIMEOUT = float(os.getenv("GEMINI_TIMEOUT", str(OPENAI_TIMEOUT)))

# Single settings instance to import anywhere
settings = Settings()
