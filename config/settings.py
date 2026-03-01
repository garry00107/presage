from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Literal


class PPMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PPM_", env_file=".env")

    # Embedder
    embedder_backend: Literal["openai", "nomic", "bge", "nvidia"] = "openai"
    embedder_model: str = "text-embedding-3-small"
    embedder_dim: int = 1536
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")

    # LLM
    llm_backend: Literal["anthropic", "openai", "ollama", "nvidia"] = "anthropic"
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    nvidia_api_key: str = Field(default="", alias="NVIDIA_API_KEY")
    llm_model: str = "claude-sonnet-4-20250514"

    # Storage paths
    sqlite_path: str = "./ppm_data/ppm.db"
    qdrant_path: str = "./ppm_data/qdrant"
    kuzu_path: str = "./ppm_data/kuzu"

    # Momentum math
    decay_lambda_base: float = 0.85       # exponential decay base
    decay_lambda_min: float = 0.60        # minimum after velocity modulation
    decay_lambda_max: float = 0.95        # maximum after context-switch reset
    momentum_beta: float = 0.90           # smoothing factor (Adam-style)
    state_window_max: int = 6             # hard cap on lookback turns
    context_switch_threshold: float = 0.40 # cosine distance → trigger reset
    velocity_alpha: float = 0.10          # velocity → lambda modulation strength
    slerp_step_size: float = 0.30         # geodesic step size per turn

    # Staging
    slot_count: int = 10
    auto_inject_threshold: float = 0.80
    hot_threshold: float = 0.50
    warm_threshold: float = 0.30
    slot_ttl_seconds: float = 120.0

    # Context budget
    max_inject_tokens: int = 4096

    # Outbox worker
    outbox_poll_interval_s: float = 0.10
    outbox_max_attempts: int = 5
    outbox_backoff_base_s: float = 2.0
    read_your_writes_window_s: float = 5.0

    # Observability
    log_level: str = "INFO"
    metrics_port: int = 9090


settings = PPMSettings()

