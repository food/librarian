"""Configuration: .env (secrets) + config.yaml (non-secret)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Env(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    litellm_base_url: str = "http://localhost:4000/v1"
    litellm_api_key: str = "sk-anything"
    ollama_base_url: str = "http://localhost:11434"
    bundle_path: Path = Path("/bundle")
    data_path: Path = Path("/data")
    # Reference shelf: read-only docs mirror; if the folder is missing the
    # feature is simply off
    docs_path: Path = Path("/docs")
    # "name:key,name:key" -- the name identifies the client in traces/commits
    client_keys: str = ""
    # LiteLLM model name that overrides the ModelSelector completely --
    # e.g. FORCE_MODEL=cloud-pool when something else hogs the GPU. Empty = off.
    force_model: str = ""
    # YAML file with prompt overrides (key -> text), e.g. to run the
    # librarian in another language. Empty or missing file = built-in prompts.
    prompts_file: str = ""

    def parsed_client_keys(self) -> dict[str, str]:
        """key -> client name"""
        out: dict[str, str] = {}
        for pair in self.client_keys.split(","):
            pair = pair.strip()
            if not pair or ":" not in pair:
                continue
            name, key = pair.split(":", 1)
            out[key.strip()] = name.strip()
        return out


class Config(BaseSettings):
    """Tuning knobs. Defaults here < config.yaml < .env / environment."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings,
                                   dotenv_settings, file_secret_settings):
        # environment beats config.yaml (init kwargs); lists/dicts in .env are JSON
        return (env_settings, dotenv_settings, init_settings)

    # Heading of the backlink section in concept bodies. Change it only to
    # keep an existing bundle parsing whose concepts use a different heading.
    related_heading: str = "## Related"
    librarian_models: list[str] = ["qwen3.6:35b-a3b", "gemma4:12b", "qwen3:30b-a3b"]
    litellm_model_map: dict[str, str] = {}
    litellm_fallback_model: str = "librarian-chat"
    model_ps_ttl: int = 30
    llm_timeout: int = 120
    llm_retries: int = 2
    fts_candidates: int = 8
    # Full-corpus sweep (librarian.sweep): cheap model for the many map calls
    # (LiteLLM name). Empty = normal model selection (ModelSelector).
    sweep_model: str = ""
    query_max_iterations: int = 4
    hot_cache_ttl_days: int = 7
    hot_cache_max_entries: int = 30
    server_host: str = "0.0.0.0"
    server_port: int = 8765


class Settings(BaseModel):
    env: Env
    cfg: Config


@lru_cache
def get_settings(config_file: str = "config.yaml") -> Settings:
    env = Env()
    path = Path(config_file)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    return Settings(env=env, cfg=Config(**(data or {})))
