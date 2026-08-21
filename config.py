"""Environment-backed runtime configuration."""

from __future__ import annotations

import os
import base64
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _secret(env_name: str, filename: str) -> str | None:
    value = os.getenv(env_name, "").strip()
    if value:
        return value
    path = ROOT / filename
    if path.exists():
        value = path.read_text(encoding="utf-8").strip()
        if value.startswith("b64:"):
            value = base64.b64decode(value[4:]).decode().strip()
        if value:
            return value
    return None


def _integer(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _number(name: str, default: float, minimum: float = 0.1) -> float:
    raw = os.getenv(name, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


@dataclass(frozen=True)
class Settings:
    strategy: str
    l2_backend: str
    lunit_api_url: str
    lunit_api_key: str | None
    lunit_model: str
    patient_simulator_url: str
    patient_simulator_model: str
    openai_api_url: str
    openai_api_key: str | None
    sol_model: str
    sol_reasoning_effort: str
    profile_count: int
    max_parallel_l2_calls: int
    min_successful_profiles: int
    l2_timeout_seconds: float
    sol_timeout_seconds: float
    pipeline_timeout_seconds: float

    @classmethod
    def from_env(cls) -> "Settings":
        profile_count = _integer("PATIENT_PROFILE_COUNT", 3)
        minimum = _integer("MIN_SUCCESSFUL_PROFILES", min(2, profile_count))
        if minimum > profile_count:
            raise ValueError("MIN_SUCCESSFUL_PROFILES cannot exceed PATIENT_PROFILE_COUNT")
        return cls(
            strategy=os.getenv("MODEL_STRATEGY", "direct_l2"),
            l2_backend=os.getenv("L2_BACKEND", "direct_l2").strip().lower(),
            lunit_api_url=os.getenv("LUNIT_FM_API_URL", "https://model.hackathon.lunit.io").rstrip("/"),
            lunit_api_key=_secret("LUNIT_FM_API_KEY", "submission_api_key"),
            lunit_model=os.getenv("LUNIT_FM_MODEL", "Lunit/L2-preview"),
            patient_simulator_url=os.getenv(
                "PATIENT_SIMULATOR_URL", "https://patient.hackathon.lunit.io"
            ).rstrip("/"),
            patient_simulator_model=os.getenv("PATIENT_SIMULATOR_MODEL", "patient-simulator-ko"),
            openai_api_url=os.getenv("OPENAI_API_URL", "https://api.openai.com/v1").rstrip("/"),
            openai_api_key=_secret("OPENAI_API_KEY", "submission_openai_key"),
            sol_model=os.getenv("SOL_MODEL", "gpt-5.6-sol"),
            sol_reasoning_effort=os.getenv("SOL_REASONING_EFFORT", "medium"),
            profile_count=profile_count,
            max_parallel_l2_calls=_integer("MAX_PARALLEL_L2_CALLS", profile_count),
            min_successful_profiles=minimum,
            l2_timeout_seconds=_number("L2_TIMEOUT_SECONDS", 180),
            sol_timeout_seconds=_number("SOL_TIMEOUT_SECONDS", 180),
            pipeline_timeout_seconds=_number("PIPELINE_TIMEOUT_SECONDS", 240),
        )
