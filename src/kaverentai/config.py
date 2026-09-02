"""Load and validate shared project configuration."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from kaverentai.paths import PROJECT_ROOT


MODEL_CONFIG_PATH = PROJECT_ROOT / "configs" / "model_config.yaml"


@lru_cache(maxsize=None)
def load_model_config(path: str | Path | None = None) -> dict[str, Any]:
    """Return the model configuration from YAML."""
    config_path = Path(path) if path is not None else MODEL_CONFIG_PATH
    if not config_path.is_file():
        raise FileNotFoundError(f"Model configuration was not found: {config_path}")

    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError("Model configuration must contain a YAML mapping")
    return config


def _required_setting(config: dict[str, Any], *keys: str) -> Any:
    """Read a required nested setting and report its complete path."""
    value: Any = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            setting_path = ".".join(keys)
            raise ValueError(f"Missing required model setting: {setting_path}")
        value = value[key]
    return value


_MODEL_CONFIG = load_model_config()
RANDOM_SEED = int(_required_setting(_MODEL_CONFIG, "random_seed"))
DEFAULT_TRAIN_END = str(
    _required_setting(_MODEL_CONFIG, "time_split", "train_end")
)
DEFAULT_VALIDATION_END = str(
    _required_setting(_MODEL_CONFIG, "time_split", "validation_end")
)
RENEWAL_LOGISTIC_C = float(
    _required_setting(
        _MODEL_CONFIG,
        "models",
        "renewal",
        "binomial_logistic_c",
    )
)
RENEWAL_SELECTED_MODEL = str(
    _required_setting(_MODEL_CONFIG, "models", "renewal", "selected")
)
LEASE_PROBABILITY_HORIZON_WEEKS = int(
    _required_setting(
        _MODEL_CONFIG,
        "models",
        "lease_probability",
        "horizon_weeks",
    )
)
LEASE_PROBABILITY_TARGET = str(
    _required_setting(
        _MODEL_CONFIG,
        "models",
        "lease_probability",
        "target",
    )
)
LEASE_PROBABILITY_SELECTED_MODEL = str(
    _required_setting(
        _MODEL_CONFIG,
        "models",
        "lease_probability",
        "selected",
    )
)
LEASE_CATBOOST_ITERATIONS = int(
    _required_setting(
        _MODEL_CONFIG,
        "models",
        "lease_probability",
        "catboost_iterations",
    )
)
LEASE_CATBOOST_DEPTH = int(
    _required_setting(
        _MODEL_CONFIG,
        "models",
        "lease_probability",
        "catboost_depth",
    )
)
LEASE_CATBOOST_LEARNING_RATE = float(
    _required_setting(
        _MODEL_CONFIG,
        "models",
        "lease_probability",
        "catboost_learning_rate",
    )
)

if RENEWAL_LOGISTIC_C <= 0:
    raise ValueError("models.renewal.binomial_logistic_c must be greater than zero")
if LEASE_PROBABILITY_HORIZON_WEEKS <= 0:
    raise ValueError("models.lease_probability.horizon_weeks must be positive")
_EXPECTED_LEASE_TARGET = (
    f"leased_within_{LEASE_PROBABILITY_HORIZON_WEEKS}_weeks"
)
if LEASE_PROBABILITY_TARGET != _EXPECTED_LEASE_TARGET:
    raise ValueError(
        "models.lease_probability.target must match its horizon_weeks setting"
    )
