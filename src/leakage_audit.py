"""Leakage audit utilities for SeatSense AI."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.preprocess import FAIRNESS_ONLY_COLUMNS, FEATURE_COLUMNS, FORBIDDEN_FEATURES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def run_leakage_audit(
    frame: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> dict[str, Any]:
    features = feature_columns or FEATURE_COLUMNS
    forbidden_found = sorted(set(features).intersection(FORBIDDEN_FEATURES))
    missing_features = [col for col in features if col not in frame.columns]
    available_post_event = [col for col in FORBIDDEN_FEATURES if col in frame.columns]

    audit = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": not forbidden_found and not missing_features,
        "strategy": (
            "Model features are checked against target, post-event outcome, "
            "recommended-price, and fairness-only columns before training."
        ),
        "feature_count": len(features),
        "forbidden_columns_found_in_features": forbidden_found,
        "missing_feature_columns": missing_features,
        "forbidden_columns_available_in_dataset": available_post_event,
        "fairness_only_columns": FAIRNESS_ONLY_COLUMNS,
        "message": (
            "No forbidden target/post-event/fairness-only columns are used as model features."
            if not forbidden_found and not missing_features
            else "Review feature configuration before trusting model metrics."
        ),
    }
    return audit


def save_leakage_audit(
    frame: pd.DataFrame,
    path: Path | None = None,
    feature_columns: list[str] | None = None,
) -> dict[str, Any]:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    audit = run_leakage_audit(frame, feature_columns=feature_columns)
    output_path = path or OUTPUTS_DIR / "leakage_audit.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    return audit
