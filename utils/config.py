import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

# Single settings instance to import anywhere
settings = Settings()
