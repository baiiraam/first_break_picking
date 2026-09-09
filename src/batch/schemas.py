"""
Typed configuration schemas using Pydantic.
"""

from typing import Any

from pydantic import BaseModel, Field, field_validator


class GlobalConfig(BaseModel):
    """Global batch training configuration."""

    epochs: int = Field(default=30, ge=1)
    device: str = Field(default="mps", pattern="^(cpu|cuda|mps)$")
    log_memory: bool = False
    verbose: bool = False
    log_level: str = Field(
        default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$"
    )
    preprocess: bool = False
    checkpoint_every: int = Field(default=5, ge=1)
    early_stopping: int = Field(default=5, ge=1)
    timeout_seconds: int = Field(default=7200, ge=60)
    skip_failed: bool = True
    max_retries: int = Field(default=3, ge=0)
    clear_memory_between_datasets: bool = True
    pause_between_datasets: int = Field(default=2, ge=0)
    concurrent: bool = False
    max_workers: int = Field(default=2, ge=1, le=8)


class MonitoringConfig(BaseModel):
    """Monitoring and notification configuration."""

    memory_warning_threshold_gb: float = Field(default=16.0, ge=0)
    memory_critical_threshold_gb: float = Field(default=20.0, ge=0)
    system_memory_percent_warning: int = Field(default=80, ge=0, le=100)
    system_memory_percent_critical: int = Field(default=90, ge=0, le=100)
    email: dict[str, Any] | None = None
    slack: dict[str, Any] | None = None


class AutoConfig(BaseModel):
    """Auto-configuration settings."""

    strategy: str = Field(default="smart", pattern="^(smart|greedy|conservative)$")
    memory_usage: float = Field(default=0.85, ge=0.5, le=0.95)
    loss_overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)
    model_order: list[str] = Field(
        default=[
            "pico",
            "nano",
            "tiny",
            "mpslight",
            "light",
            "mobile",
            "efficient",
            "unet",
        ]
    )
    skip_for_large: list[str] = Field(default=["unet"])


class BatchConfig(BaseModel):
    """Complete batch configuration."""

    global_: GlobalConfig = Field(alias="global")
    datasets: dict[str, dict[str, Any]] = Field(default_factory=dict)
    variants: list[dict[str, Any]] = Field(default_factory=list)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    auto: AutoConfig | None = None

    @field_validator("global_", mode="before")
    @classmethod
    def validate_global(cls, v):
        """Handle legacy config format."""
        if isinstance(v, dict):
            return v
        return {}

    class Config:
        populate_by_name = True
