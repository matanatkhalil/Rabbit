"""Runtime configuration: ../config.yaml, overridden per value by RABBIT_* environment
variables. Secrets (the LLM API key) come from the environment, loaded from ../.env."""
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent  # backend/
PROJECT_ROOT = ROOT.parent  # Rabbit/Rabbit/, where config.yaml lives

load_dotenv(PROJECT_ROOT / ".env")  # never overrides variables already set

EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 3072


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    database_url: str
    data_dir: Path
    embed_url: str
    embed_api_key: str | None = field(repr=False)
    embed_timeout_s: float
    embed_batch_size: int
    chunk_size: int
    chunk_overlap: int
    empty_page_ratio: float
    chat_top_k: int
    llm_base_url: str
    llm_model: str
    llm_api_key: str | None
    llm_temperature: float
    llm_max_tokens: int
    llm_timeout_s: float


def load(path: Path | None = None, env: dict | None = None) -> Settings:
    env = os.environ if env is None else env
    path = Path(path or env.get("RABBIT_CONFIG") or PROJECT_ROOT / "config.yaml")
    cfg = (yaml.safe_load(path.read_text()) or {}) if path.is_file() else {}

    def get(key: str, var: str, default, cast=str):
        if var in env:
            return cast(env[var])
        node = cfg
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return cast(default)
            node = node[part]
        return cast(node)

    data_dir = Path(get("storage.data_dir", "RABBIT_DATA_DIR", "backend/data"))
    key_var = get("chat.llm.api_key_env", "RABBIT_LLM_API_KEY_ENV", "AZURE_CHAT_API_KEY")
    return Settings(
        host=get("server.host", "RABBIT_HOST", "127.0.0.1"),
        port=get("server.port", "RABBIT_PORT", 8000, int),
        database_url=get(
            "database.url", "RABBIT_DATABASE_URL", "postgresql://rabbit:rabbit@localhost:5432/orange_rabbit"
        ),
        data_dir=data_dir if data_dir.is_absolute() else PROJECT_ROOT / data_dir,
        embed_url=get(
            "embedder.url", "RABBIT_EMBED_URL", "https://agro-llm.services.ai.azure.com/openai/v1/embeddings"
        ),
        embed_api_key=env.get("AZURE_EMBEDDING_API_KEY") or None,
        embed_timeout_s=get("embedder.timeout_s", "RABBIT_EMBED_TIMEOUT_S", 60, float),
        embed_batch_size=get("embedder.batch_size", "RABBIT_EMBED_BATCH_SIZE", 16, int),
        chunk_size=get("ingest.chunk_size", "RABBIT_CHUNK_SIZE", 500, int),
        chunk_overlap=get("ingest.chunk_overlap", "RABBIT_CHUNK_OVERLAP", 75, int),
        # clean_text: a book is "scanned" when more than this fraction of pages is empty.
        empty_page_ratio=get("ingest.empty_page_ratio", "RABBIT_EMPTY_PAGE_RATIO", 0.8, float),
        chat_top_k=get("chat.top_k", "RABBIT_CHAT_TOP_K", 8, int),
        llm_base_url=get(
            "chat.llm.base_url", "RABBIT_LLM_BASE_URL", "https://agro-llm.services.ai.azure.com/openai/v1"
        ),
        llm_model=get("chat.llm.model", "RABBIT_LLM_MODEL", "DeepSeek-V4.1-Flash"),
        llm_api_key=env.get(key_var) or None,
        llm_temperature=get("chat.llm.temperature", "RABBIT_LLM_TEMPERATURE", 0.2, float),
        llm_max_tokens=get("chat.llm.max_tokens", "RABBIT_LLM_MAX_TOKENS", 1024, int),
        llm_timeout_s=get("chat.llm.timeout_s", "RABBIT_LLM_TIMEOUT_S", 60, float),
    )


SETTINGS = load()

HOST = SETTINGS.host
PORT = SETTINGS.port
DATABASE_URL = SETTINGS.database_url
DATA_DIR = SETTINGS.data_dir
BOOKS_DIR = DATA_DIR / "books"

EMBED_BATCH_SIZE = SETTINGS.embed_batch_size

CHUNK_SIZE = SETTINGS.chunk_size
CHUNK_OVERLAP = SETTINGS.chunk_overlap
EMPTY_PAGE_RATIO = SETTINGS.empty_page_ratio

CHAT_TOP_K = SETTINGS.chat_top_k
