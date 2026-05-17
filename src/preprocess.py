"""Preprocessing definitions for SeatSense AI models.

This module is the contract between the synthetic data generator, model
training pipeline, prediction interface, and Streamlit pages. Post-event
outcomes and fairness-only affordability columns are intentionally excluded
from ``FEATURE_COLUMNS`` to reduce leakage and fairness risk.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "demand_tier"
SELLTHROUGH_TARGET_COLUMN = "final_sell_through_rate"
SCALPER_TARGET_COLUMN = "scalper_risk_tier"

CLASS_ORDER = ["Low", "Medium", "High"]
SCALPER_CLASS_ORDER = ["Low", "Medium", "High"]

POST_EVENT_OUTCOME_COLUMNS = [
    "final_sell_through_rate",
    "final_attendance_rate",
    "realized_revenue",
    "revenue_per_available_seat",
    "unsold_inventory_pct",
    "actual_secondary_market_gap",
    "realized_demand_score",
    "demand_index",
    "demand_tier",
    "scalper_risk_tier",
    "scalper_risk_label",
    "optimal_price_oracle",
    "recommended_price",
    "recommended_price_target",
]

FAIRNESS_ONLY_COLUMNS = [
    "affordability_index",
    "affordability_segment",
    "local_income_proxy",
]

FORBIDDEN_FEATURES = sorted(set(POST_EVENT_OUTCOME_COLUMNS + FAIRNESS_ONLY_COLUMNS))

NUMERIC_FEATURES = [
    "season",
    "weekend_flag",
    "holiday_flag",
    "rivalry_flag",
    "division_matchup_flag",
    "opponent_strength",
    "home_team_win_rate",
    "away_team_win_rate",
    "home_team_recent_form",
    "away_team_recent_form",
    "playoff_implication_score",
    "premium_game_flag",
    "section_capacity",
    "inventory_remaining",
    "inventory_remaining_pct",
    "current_sell_through_rate",
    "seat_quality_score",
    "base_ticket_price",
    "current_ticket_price",
    "price_change_from_base_pct",
    "days_before_game",
    "month",
    "website_traffic_index",
    "search_interest_index",
    "social_sentiment_score",
    "social_volume_index",
    "email_campaign_active",
    "promotion_flag",
    "marketing_spend_index",
    "star_player_available",
    "injury_news_severity",
    "weather_severity",
    "temperature_score",
    "demand_signal_score",
    "secondary_market_avg_price",
    "secondary_market_listing_count",
    "resale_price_gap_pct",
    "resale_velocity_index",
    "scalper_pressure_score",
    "rolling_3_game_attendance_rate",
    "rolling_5_game_sell_through_rate",
    "previous_similar_game_sell_through",
    "same_opponent_last_season_attendance",
    "section_historical_fill_rate",
    "historical_price_elasticity_section",
    "historical_attendance_rate",
]

CATEGORICAL_FEATURES = [
    "day_of_week",
    "opponent",
    "seat_section",
    "pricing_window",
    "days_until_game_bucket",
    "season_stage",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

BUSINESS_METRIC_COLUMNS = [
    "game_id",
    "game_date",
    "affordability_segment",
    "affordability_index",
    "tickets_sold_pct",
    "sold_tickets",
    "attendance_rate",
    "revenue",
    "recommended_price_target",
    "optimal_price_oracle",
]

REQUIRED_TRAINING_COLUMNS = sorted(
    set(
        FEATURE_COLUMNS
        + BUSINESS_METRIC_COLUMNS
        + [
            TARGET_COLUMN,
            SELLTHROUGH_TARGET_COLUMN,
            SCALPER_TARGET_COLUMN,
            "final_attendance_rate",
            "realized_revenue",
            "revenue_per_available_seat",
            "unsold_inventory_pct",
            "actual_secondary_market_gap",
            "realized_demand_score",
        ]
    )
)


def _one_hot_encoder() -> OneHotEncoder:
    """Return a scikit-learn-compatible one-hot encoder across versions."""

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", _one_hot_encoder(), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def validate_training_frame(frame) -> None:
    missing = [col for col in REQUIRED_TRAINING_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"Training data is missing required columns: {missing}")

    forbidden_in_features = sorted(set(FEATURE_COLUMNS).intersection(FORBIDDEN_FEATURES))
    if forbidden_in_features:
        raise ValueError(
            "Forbidden leakage or fairness-only columns are configured as model features: "
            f"{forbidden_in_features}"
        )
