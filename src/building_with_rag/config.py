from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings from environment / .env. Every value has a default so the app starts bare."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    mongodb_uri: str = ""
    mongodb_db_name: str = "building_with_rag"
    mongodb_test_db_name: str = "building_with_rag_test"
    voyage_api_key: str = ""
    capstone_api_key: str = ""
    webui_demo_caller_id: str = "demo-public"
    generation_api_base_url: str = ""
    generation_api_key: str = ""
    generation_model_name: str = "gpt-4o-mini"
    rerank_api_base_url: str = "https://api.voyageai.com/v1"
    rerank_api_key: str = ""
    rerank_model_name: str = "rerank-2.5"
    rerank_request_timeout_seconds: float = 30
    rerank_candidate_limit: int = 20
    rerank_send_limit: int = 10
    rerank_return_limit: int = 5

    # Fixed course embedding choice (not configurable).
    embedding_model: str = "voyage-3.5"
    embedding_model_version: str = "voyage-3.5"
    embedding_dimensions: int = 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
