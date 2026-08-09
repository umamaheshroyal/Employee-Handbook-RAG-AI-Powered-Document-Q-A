import os
from pathlib import Path

from dotenv import load_dotenv

# ------------------------------------
# Load .env file
# ------------------------------------

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_FILE)

# ------------------------------------
# OpenRouter Configuration
# ------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

OPENAI_BASE_URL = os.getenv(
    "OPENAI_BASE_URL",
    "https://openrouter.ai/api/v1"
)

# ------------------------------------
# Embedding Model
# ------------------------------------

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "text-embedding-3-small"
)

# ------------------------------------
# LLM Model
# ------------------------------------

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "openai/gpt-4.1-mini"
)

# ------------------------------------
# Generation Settings
# ------------------------------------

TEMPERATURE = float(
    os.getenv("TEMPERATURE", "0")
)

MAX_TOKENS = int(
    os.getenv("MAX_TOKENS", "1024")
)