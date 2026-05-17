"""Prediction interface used by the Streamlit app and tests."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.generate_ticket_data import SECTIONS, days_bucket, generate_all  # noqa: E402
from src.model_registry import MODEL_PATHS, artifacts_exist  # noqa: E402
from src.preprocess import FEATURE_COLUMNS  # noqa: E402


DATA_PATH = PROJECT_ROOT / "data" / "model_training_data.csv"
MODEL_PATH = MODEL_PATHS["demand_classifier"]
METRICS_PATH = PROJECT_ROOT / "outputs" / "model_metrics.json"

SECTION_ALIASES = {
    "Upper": "Upper Sideline",
    "Lower": "Lower Sideline",
    "Club": "Club Level",
    "Premium": "Courtside/Premium",
    "Courtside": "Courtside/Premium",
}

DEFAULT_INPUT: dict[str, Any] = {
    "season": 2025,
    "game_id": "DEMO_GAME",
    "game_date": "2026-01-24",
    "day_of_week": "Saturday",
    "weekend_flag": 1,
    "holiday_flag": 0,
    "rivalry_flag": 1,
    "division_matchup_flag": 1,
    "opponent": "Boston Celtics",
    "opponent_strength": 92,
    "home_team_win_rate": 0.61,
    "away_team_win_rate": 0.67,
    "home_team_recent_form": 0.64,
    "away_team_recent_form": 0.66,
    "playoff_implication_score": 0.62,
    "premium_game_flag": 1,
    "seat_section": "Lower Sideline",
    "section_capacity": 1900,
    "inventory_remaining": 530,
    "inventory_remaining_pct": 0.279,
    "current_sell_through_rate": 0.721,
    "seat_quality_score": 0.72,
    "base_ticket_price": 138.0,
    "current_ticket_price": 148.0,
    "price_change_from_base_pct": 0.0725,
    "days_before_game": 14,
    "pricing_window": "14_days_out",
    "days_until_game_bucket": "8_to_20_days",
    "month": 1,
    "season_stage": "mid",
    "website_traffic_index": 188,
    "search_interest_index": 146,
    "social_sentiment_score": 0.52,
    "social_volume_index": 154,
    "email_campaign_active": 1,
    "promotion_flag": 0,
    "marketing_spend_index": 58,
    "star_player_available": 1,
    "injury_news_severity": 0.05,
    "weather_severity": 0.20,
    "temperature_score": 0.86,
    "demand_signal_score": 0.79,
    "secondary_market_avg_price": 218.0,
    "secondary_market_listing_count": 325,
    "resale_price_gap_pct": 0.473,
    "resale_velocity_index": 96,
    "scalper_pressure_score": 0.72,
    "rolling_3_game_attendance_rate": 0.86,
    "rolling_5_game_sell_through_rate": 0.84,
    "previous_similar_game_sell_through": 0.88,
    "same_opponent_last_season_attendance": 0.91,
    "section_historical_fill_rate": 0.84,
    "historical_price_elasticity_section": -0.82,
    "historical_attendance_rate": 0.86,
    "tickets_sold_pct": 0.72,
    "affordability_segment": "Mainstream",
    "affordability_index": 72,
    "local_income_proxy": 72,
}


def ensure_project_ready(force_train: bool = False) -> None:
    if not DATA_PATH.exists():
        generate_all(PROJECT_ROOT / "data")
    if force_train or not artifacts_exist() or not METRICS_PATH.exists():
        from src.train_model import train

        train()


def load_model():
    ensure_project_ready()
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return joblib.load(MODEL_PATHS["legacy_demand_model"])


def load_all_models() -> dict[str, Any]:
    ensure_project_ready()
    return {
        "demand_classifier": joblib.load(MODEL_PATHS["demand_classifier"]),
        "sellthrough_regressor": joblib.load(MODEL_PATHS["sellthrough_regressor"]),
        "scalper_risk_classifier": joblib.load(MODEL_PATHS["scalper_risk_classifier"]),
    }


def _canonical_section(value: Any) -> str:
    section = str(value or DEFAULT_INPUT["seat_section"])
    return SECTION_ALIASES.get(section, section if section in SECTIONS else DEFAULT_INPUT["seat_section"])


def normalize_input(input_data: dict[str, Any]) -> dict[str, Any]:
    normalized = DEFAULT_INPUT.copy()
    normalized.update(input_data or {})
    normalized["seat_section"] = _canonical_section(normalized.get("seat_section"))
    config = SECTIONS[normalized["seat_section"]]

    normalized["day_of_week"] = str(normalized.get("day_of_week", DEFAULT_INPUT["day_of_week"]))
    normalized["weekend_flag"] = int(normalized["day_of_week"] in {"Friday", "Saturday", "Sunday"})
    normalized["rivalry_flag"] = int(bool(normalized.get("rivalry_flag", 0)))
    normalized["division_matchup_flag"] = int(bool(normalized.get("division_matchup_flag", normalized["rivalry_flag"])))
    normalized["holiday_flag"] = int(bool(normalized.get("holiday_flag", 0)))
    normalized["premium_game_flag"] = int(bool(normalized.get("premium_game_flag", 0)))
    normalized["star_player_available"] = int(bool(normalized.get("star_player_available", 1)))
    normalized["email_campaign_active"] = int(bool(normalized.get("email_campaign_active", 0)))
    normalized["promotion_flag"] = int(bool(normalized.get("promotion_flag", 0)))

    normalized["section_capacity"] = int(float(normalized.get("section_capacity") or config.capacity))
    normalized["seat_quality_score"] = float(normalized.get("seat_quality_score") or config.seat_quality_score)
    normalized["base_ticket_price"] = float(normalized.get("base_ticket_price") or config.base_price)
    normalized["current_ticket_price"] = float(normalized.get("current_ticket_price") or normalized["base_ticket_price"])
    normalized["secondary_market_avg_price"] = float(
        normalized.get("secondary_market_avg_price") or normalized["current_ticket_price"]
    )
    normalized["days_before_game"] = int(float(normalized.get("days_before_game", 14)))
    normalized["pricing_window"] = str(
        normalized.get(
            "pricing_window",
            "day_of_game"
            if normalized["days_before_game"] == 0
            else "2_days_out"
            if normalized["days_before_game"] <= 2
            else "7_days_out"
            if normalized["days_before_game"] <= 7
            else "14_days_out"
            if normalized["days_before_game"] <= 14
            else "30_days_out",
        )
    )
    normalized["days_until_game_bucket"] = days_bucket(normalized["days_before_game"])

    normalized["price_change_from_base_pct"] = (
        normalized["current_ticket_price"] / max(normalized["base_ticket_price"], 1) - 1
    )
    normalized["resale_price_gap_pct"] = (
        normalized["secondary_market_avg_price"] / max(normalized["current_ticket_price"], 1) - 1
    )

    historical_fill = float(
        normalized.get(
            "section_historical_fill_rate",
            normalized.get("historical_attendance_rate", DEFAULT_INPUT["historical_attendance_rate"]),
        )
    )
    normalized["section_historical_fill_rate"] = historical_fill
    normalized["historical_attendance_rate"] = float(normalized.get("historical_attendance_rate", historical_fill))
    normalized["rolling_3_game_attendance_rate"] = float(
        normalized.get("rolling_3_game_attendance_rate", historical_fill)
    )
    normalized["rolling_5_game_sell_through_rate"] = float(
        normalized.get("rolling_5_game_sell_through_rate", historical_fill)
    )
    normalized["previous_similar_game_sell_through"] = float(
        normalized.get("previous_similar_game_sell_through", min(0.98, historical_fill + 0.03))
    )
    normalized["same_opponent_last_season_attendance"] = float(
        normalized.get("same_opponent_last_season_attendance", min(0.98, historical_fill + 0.04))
    )
    normalized["historical_price_elasticity_section"] = float(
        normalized.get("historical_price_elasticity_section") or config.elasticity
    )

    tickets_sold_pct = float(normalized.get("tickets_sold_pct", max(0.10, historical_fill - 0.08)))
    normalized["tickets_sold_pct"] = tickets_sold_pct
    normalized["inventory_remaining"] = int(
        float(
            normalized.get(
                "inventory_remaining",
                normalized["section_capacity"] * max(0, 1 - tickets_sold_pct),
            )
        )
    )
    normalized["inventory_remaining_pct"] = float(
        normalized.get(
            "inventory_remaining_pct",
            normalized["inventory_remaining"] / max(normalized["section_capacity"], 1),
        )
    )
    normalized["current_sell_through_rate"] = float(
        normalized.get(
            "current_sell_through_rate",
            1 - normalized["inventory_remaining_pct"],
        )
    )

    normalized["search_interest_index"] = float(
        normalized.get("search_interest_index", normalized["website_traffic_index"] * 0.78)
    )
    normalized["social_volume_index"] = float(
        normalized.get("social_volume_index", normalized["website_traffic_index"] * 0.68)
    )
    normalized["secondary_market_listing_count"] = float(
        normalized.get(
            "secondary_market_listing_count",
            max(5, normalized["section_capacity"] * (0.03 + max(normalized["resale_price_gap_pct"], 0) * 0.07)),
        )
    )
    normalized["resale_velocity_index"] = float(
        normalized.get(
            "resale_velocity_index",
            np.clip(30 + 90 * max(normalized["resale_price_gap_pct"], 0) + 22 * tickets_sold_pct, 5, 145),
        )
    )
    normalized["scalper_pressure_score"] = float(
        normalized.get(
            "scalper_pressure_score",
            np.clip(
                0.42 * max(normalized["resale_price_gap_pct"], 0)
                + 0.22 * min(normalized["secondary_market_listing_count"] / 520, 1.4)
                + 0.28 * min(normalized["resale_velocity_index"] / 120, 1.4),
                0,
                1,
            ),
        )
    )
    normalized["demand_signal_score"] = float(
        normalized.get(
            "demand_signal_score",
            np.clip(
                0.20 * min(normalized["website_traffic_index"] / 220, 1.2)
                + 0.16 * min(normalized["search_interest_index"] / 190, 1.2)
                + 0.12 * min(normalized["social_volume_index"] / 210, 1.2)
                + 0.11 * ((float(normalized.get("social_sentiment_score", 0)) + 1) / 2)
                + 0.16 * normalized["current_sell_through_rate"]
                + 0.10 * min(max(normalized["resale_price_gap_pct"], 0) / 0.50, 1.2)
                + 0.06 * int(normalized.get("premium_game_flag", 0))
                + 0.05 * int(normalized.get("rivalry_flag", 0))
                + 0.04 * (1 - float(normalized.get("weather_severity", 0))),
                0,
                1.15,
            ),
        )
    )

    normalized["affordability_segment"] = normalized.get(
        "affordability_segment", config.affordability_segment
    )
    normalized["affordability_index"] = float(normalized.get("affordability_index", config.affordability_index))
    normalized["local_income_proxy"] = float(normalized.get("local_income_proxy", normalized["affordability_index"]))

    for column in FEATURE_COLUMNS:
        if column not in normalized:
            normalized[column] = DEFAULT_INPUT.get(column, 0)
    return normalized


def _probability_map(model, probabilities: np.ndarray) -> dict[str, float]:
    return {str(label): float(probabilities[idx]) for idx, label in enumerate(model.classes_)}


def predict_demand(input_data: dict[str, Any], model=None) -> dict[str, Any]:
    model = model or load_model()
    normalized = normalize_input(input_data)
    frame = pd.DataFrame([normalized])[FEATURE_COLUMNS]
    prediction = str(model.predict(frame)[0])
    probabilities = model.predict_proba(frame)[0]
    probability_map = _probability_map(model, probabilities)
    confidence = probability_map[prediction]
    return {
        "demand_tier": prediction,
        "demand_probability": confidence,
        "probabilities": probability_map,
        "input": normalized,
    }


def predict_full(input_data: dict[str, Any], models: dict[str, Any] | None = None) -> dict[str, Any]:
    models = models or load_all_models()
    demand = predict_demand(input_data, models["demand_classifier"])
    normalized = demand["input"]
    frame = pd.DataFrame([normalized])[FEATURE_COLUMNS]
    sellthrough = float(np.clip(models["sellthrough_regressor"].predict(frame)[0], 0, 1))
    scalper_prediction = str(models["scalper_risk_classifier"].predict(frame)[0])
    scalper_probabilities = models["scalper_risk_classifier"].predict_proba(frame)[0]
    scalper_probability_map = _probability_map(models["scalper_risk_classifier"], scalper_probabilities)
    return {
        "demand": demand,
        "sellthrough": {
            "predicted_sell_through": sellthrough,
            "sell_through_rate": sellthrough,
        },
        "scalper": {
            "scalper_risk_tier": scalper_prediction,
            "scalper_risk_probability": scalper_probability_map[scalper_prediction],
            "probabilities": scalper_probability_map,
        },
        "input": normalized,
    }
