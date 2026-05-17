"""Central model artifact paths and loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "data"

MODEL_PATHS = {
    "demand_classifier": MODELS_DIR / "demand_classifier.pkl",
    "sellthrough_regressor": MODELS_DIR / "sellthrough_regressor.pkl",
    "scalper_risk_classifier": MODELS_DIR / "scalper_risk_classifier.pkl",
    "preprocessor": MODELS_DIR / "preprocessor.pkl",
    "legacy_demand_model": MODELS_DIR / "demand_model.pkl",
}


def required_model_paths() -> list[Path]:
    return [
        MODEL_PATHS["demand_classifier"],
        MODEL_PATHS["sellthrough_regressor"],
        MODEL_PATHS["scalper_risk_classifier"],
    ]


def artifacts_exist() -> bool:
    return all(path.exists() for path in required_model_paths())


def load_artifact(name: str) -> Any:
    if name not in MODEL_PATHS:
        raise KeyError(f"Unknown SeatSense model artifact: {name}")
    return joblib.load(MODEL_PATHS[name])


def save_artifact(name: str, artifact: Any) -> Path:
    if name not in MODEL_PATHS:
        raise KeyError(f"Unknown SeatSense model artifact: {name}")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_PATHS[name]
    joblib.dump(artifact, path)
    return path


def load_models() -> dict[str, Any]:
    return {
        "demand_classifier": load_artifact("demand_classifier"),
        "sellthrough_regressor": load_artifact("sellthrough_regressor"),
        "scalper_risk_classifier": load_artifact("scalper_risk_classifier"),
    }
