"""Explainability helpers for model outputs and Streamlit views."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def load_feature_importance(path: Path | None = None) -> pd.DataFrame:
    path = path or OUTPUTS_DIR / "feature_importance.csv"
    if not path.exists():
        return pd.DataFrame(columns=["feature", "importance"])
    return pd.read_csv(path)


def top_global_drivers(n: int = 8) -> pd.DataFrame:
    importance = load_feature_importance()
    if importance.empty:
        return importance
    return importance.sort_values("importance", ascending=False).head(n)


def local_driver_summary(input_row: dict) -> list[dict]:
    """Create business-readable local drivers without requiring SHAP."""

    drivers: list[dict] = []
    if input_row.get("opponent_strength", 0) >= 82:
        drivers.append({"driver": "Elite opponent", "direction": "Raises demand"})
    elif input_row.get("opponent_strength", 0) <= 55:
        drivers.append({"driver": "Lower-strength opponent", "direction": "Lowers demand"})

    if input_row.get("rivalry_flag"):
        drivers.append({"driver": "Rivalry matchup", "direction": "Raises demand"})
    if input_row.get("weekend_flag"):
        drivers.append({"driver": "Weekend timing", "direction": "Raises demand"})
    if not input_row.get("star_player_available", 1):
        drivers.append({"driver": "Star player unavailable", "direction": "Lowers demand"})
    if input_row.get("weather_severity", 0) >= 0.65:
        drivers.append({"driver": "Severe weather", "direction": "Lowers attendance confidence"})
    if input_row.get("website_traffic_index", 0) >= 155:
        drivers.append({"driver": "High ticketing traffic", "direction": "Raises demand"})
    if input_row.get("search_interest_index", 0) >= 130:
        drivers.append({"driver": "Elevated search interest", "direction": "Raises demand"})
    if input_row.get("social_volume_index", 0) >= 130:
        drivers.append({"driver": "High social volume", "direction": "Raises demand"})
    if input_row.get("social_sentiment_score", 0) >= 0.45:
        drivers.append({"driver": "Positive social sentiment", "direction": "Raises demand"})
    if input_row.get("secondary_market_avg_price", 0) > input_row.get("current_ticket_price", 0) * 1.18:
        drivers.append({"driver": "Secondary-market price gap", "direction": "Suggests underpricing"})
    if input_row.get("inventory_remaining", 10**9) < input_row.get("section_capacity", 1) * 0.30:
        drivers.append({"driver": "Limited inventory remaining", "direction": "Raises sellout pressure"})
    if input_row.get("playoff_implication_score", 0) >= 0.55:
        drivers.append({"driver": "Playoff implication", "direction": "Raises urgency"})
    if input_row.get("days_before_game", 99) <= 10:
        drivers.append({"driver": "Event is soon", "direction": "Raises urgency"})

    return drivers[:6] or [{"driver": "Balanced signal mix", "direction": "No single signal dominates"}]
