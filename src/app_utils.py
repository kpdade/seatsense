"""Shared Streamlit UI helpers and demo scenarios."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from data.generate_ticket_data import SECTIONS
from src.explainability import local_driver_summary
from src.generative_explainer import (
    fallback_pricing_explanation,
    generate_pricing_explanation_with_source,
    openai_explanations_enabled,
)
from src.predict import DEFAULT_INPUT, SECTION_ALIASES, ensure_project_ready, load_all_models, load_model, predict_full
from src.pricing_optimizer import optimize_price
from src.ui_components import apply_global_styles, render_sidebar_branding


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "model_training_data.csv"
METRICS_PATH = PROJECT_ROOT / "outputs" / "model_metrics.json"
SECTION_CAPACITY = {section: config.capacity for section, config in SECTIONS.items()}
AFFORDABILITY_SEGMENT = {section: config.affordability_segment for section, config in SECTIONS.items()}
AFFORDABILITY_INDEX = {section: config.affordability_index for section, config in SECTIONS.items()}


def configure_page(title: str = "SeatSense AI") -> None:
    st.set_page_config(page_title=title, page_icon="🎟️", layout="wide")
    apply_global_styles()
    render_sidebar_branding()


@st.cache_resource(show_spinner="Preparing SeatSense model artifacts...")
def cached_model():
    try:
        ensure_project_ready()
        return load_model()
    except Exception as exc:
        st.warning(
            "SeatSense could not load the saved model. Attempting to regenerate the demo artifacts."
        )
        ensure_project_ready(force_train=True)
        try:
            return load_model()
        except Exception as retry_exc:
            raise RuntimeError(
                "Model artifacts are unavailable. Run `python -m src.train_model` from the project root."
            ) from retry_exc


@st.cache_resource(show_spinner="Loading SeatSense model portfolio...")
def cached_models():
    try:
        ensure_project_ready()
        return load_all_models()
    except Exception:
        ensure_project_ready(force_train=True)
        return load_all_models()


@st.cache_data(show_spinner=False)
def _load_training_data_cached(data_mtime: float) -> pd.DataFrame:
    del data_mtime
    return pd.read_csv(DATA_PATH)


def load_training_data() -> pd.DataFrame:
    try:
        ensure_project_ready()
        return _load_training_data_cached(DATA_PATH.stat().st_mtime)
    except Exception:
        st.warning("Demo data was missing or unreadable, so SeatSense is regenerating it.")
        ensure_project_ready(force_train=True)
        return _load_training_data_cached(DATA_PATH.stat().st_mtime)


@st.cache_data(show_spinner=False)
def _load_metrics_cached(metrics_mtime: float) -> dict[str, Any]:
    del metrics_mtime
    with METRICS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_metrics() -> dict[str, Any]:
    try:
        ensure_project_ready()
        return _load_metrics_cached(METRICS_PATH.stat().st_mtime)
    except Exception:
        st.warning("Model metrics were missing or unreadable, so SeatSense is retraining the pricing model.")
        ensure_project_ready(force_train=True)
        return _load_metrics_cached(METRICS_PATH.stat().st_mtime)


def clear_cached_data_loaders() -> None:
    _load_training_data_cached.clear()
    _load_metrics_cached.clear()


def metric_card(label: str, value: str, note: str = "") -> None:
    from src.ui_components import render_kpi_card

    render_kpi_card(label=label, value=value, note=note)


def build_prediction_form(defaults: dict[str, Any] | None = None, key_prefix: str = "") -> dict[str, Any]:
    values = DEFAULT_INPUT.copy()
    if defaults:
        values.update(defaults)
    values["seat_section"] = SECTION_ALIASES.get(values.get("seat_section"), values.get("seat_section"))
    if values["seat_section"] not in SECTION_CAPACITY:
        values["seat_section"] = DEFAULT_INPUT["seat_section"]

    game_tab, demand_tab, pricing_tab, market_tab = st.tabs(
        ["Game context", "Demand signals", "Pricing context", "Market signals"]
    )

    with game_tab:
        left, right = st.columns(2)
        with left:
            day_of_week = st.selectbox(
                "Day of week",
                ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                index=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(values["day_of_week"]),
                key=f"{key_prefix}day",
            )
            days_before_game = st.slider(
                "Days before game",
                1,
                45,
                int(values["days_before_game"]),
                key=f"{key_prefix}days",
            )
        with right:
            rivalry = st.toggle("Rivalry game", value=bool(values["rivalry_flag"]), key=f"{key_prefix}rivalry")
            star_available = st.toggle("Star player available", value=bool(values["star_player_available"]), key=f"{key_prefix}star")
            premium_game = st.toggle("Premium game", value=bool(values["premium_game_flag"]), key=f"{key_prefix}premium")

    with demand_tab:
        left, right = st.columns(2)
        with left:
            opponent_strength = st.slider(
                "Opponent strength",
                30,
                100,
                int(values["opponent_strength"]),
                key=f"{key_prefix}opp_strength",
            )
            home_win = st.slider(
                "Home team win rate",
                0.20,
                0.90,
                float(values["home_team_win_rate"]),
                0.01,
                key=f"{key_prefix}home_win",
            )
        with right:
            away_win = st.slider(
                "Away team win rate",
                0.20,
                0.90,
                float(values["away_team_win_rate"]),
                0.01,
                key=f"{key_prefix}away_win",
            )
            historical_attendance = st.slider(
                "Historical attendance rate",
                0.30,
                1.00,
                float(values["historical_attendance_rate"]),
                0.01,
                key=f"{key_prefix}hist_attendance",
            )

    with pricing_tab:
        left, right = st.columns(2)
        with left:
            seat_section = st.selectbox(
                "Seat section",
                list(SECTION_CAPACITY),
                index=list(SECTION_CAPACITY).index(values["seat_section"]),
                key=f"{key_prefix}seat_section",
            )
            current_price = st.number_input(
                "Current ticket price",
                min_value=15.0,
                max_value=1000.0,
                value=float(values["current_ticket_price"]),
                step=5.0,
                key=f"{key_prefix}current",
            )
        with right:
            secondary_price = st.number_input(
                "Secondary market average",
                min_value=15.0,
                max_value=1500.0,
                value=float(values["secondary_market_avg_price"]),
                step=5.0,
                key=f"{key_prefix}secondary",
            )
            promotion = st.toggle("Promotion active", value=bool(values["promotion_flag"]), key=f"{key_prefix}promotion")

    with market_tab:
        left, right = st.columns(2)
        with left:
            traffic = st.slider(
                "Website traffic index",
                35,
                230,
                int(values["website_traffic_index"]),
                key=f"{key_prefix}traffic",
            )
            sentiment = st.slider(
                "Social sentiment",
                -1.0,
                1.0,
                float(values["social_sentiment_score"]),
                0.05,
                key=f"{key_prefix}sentiment",
            )
        with right:
            weather = st.slider(
                "Weather severity",
                0.0,
                1.0,
                float(values["weather_severity"]),
                0.05,
                key=f"{key_prefix}weather",
            )
            st.info("Traffic and sentiment should be aggregated signals in production.")

    affordability_segment = AFFORDABILITY_SEGMENT[seat_section]
    section_config = SECTIONS[seat_section]
    return {
        "season": values.get("season", DEFAULT_INPUT["season"]),
        "game_id": values.get("game_id", "DEMO_GAME"),
        "game_date": values.get("game_date", DEFAULT_INPUT["game_date"]),
        "opponent": values.get("opponent", DEFAULT_INPUT["opponent"]),
        "opponent_strength": opponent_strength,
        "home_team_win_rate": home_win,
        "away_team_win_rate": away_win,
        "home_team_recent_form": values.get("home_team_recent_form", home_win),
        "away_team_recent_form": values.get("away_team_recent_form", away_win),
        "day_of_week": day_of_week,
        "weekend_flag": int(day_of_week in {"Friday", "Saturday", "Sunday"}),
        "holiday_flag": int(values.get("holiday_flag", 0)),
        "rivalry_flag": int(rivalry),
        "division_matchup_flag": int(values.get("division_matchup_flag", rivalry)),
        "playoff_implication_score": values.get("playoff_implication_score", 0.35),
        "star_player_available": int(star_available),
        "injury_news_severity": values.get("injury_news_severity", 0.08 if star_available else 0.65),
        "weather_severity": weather,
        "temperature_score": values.get("temperature_score", max(0.1, 1 - weather * 0.55)),
        "social_sentiment_score": sentiment,
        "website_traffic_index": traffic,
        "search_interest_index": values.get("search_interest_index", traffic * 0.78),
        "social_volume_index": values.get("social_volume_index", traffic * 0.68),
        "email_campaign_active": int(values.get("email_campaign_active", 0)),
        "marketing_spend_index": values.get("marketing_spend_index", 58),
        "historical_attendance_rate": historical_attendance,
        "rolling_3_game_attendance_rate": values.get("rolling_3_game_attendance_rate", historical_attendance),
        "rolling_5_game_sell_through_rate": values.get("rolling_5_game_sell_through_rate", historical_attendance),
        "previous_similar_game_sell_through": values.get("previous_similar_game_sell_through", min(0.98, historical_attendance + 0.03)),
        "same_opponent_last_season_attendance": values.get("same_opponent_last_season_attendance", min(0.98, historical_attendance + 0.04)),
        "section_historical_fill_rate": values.get("section_historical_fill_rate", historical_attendance),
        "historical_price_elasticity_section": values.get("historical_price_elasticity_section", section_config.elasticity),
        "base_ticket_price": values.get("base_ticket_price", section_config.base_price),
        "section_capacity": SECTION_CAPACITY[seat_section],
        "inventory_remaining": values.get("inventory_remaining", int(SECTION_CAPACITY[seat_section] * max(0.05, 1 - historical_attendance))),
        "inventory_remaining_pct": values.get("inventory_remaining_pct", max(0.05, 1 - historical_attendance)),
        "current_sell_through_rate": values.get("current_sell_through_rate", historical_attendance),
        "seat_quality_score": section_config.seat_quality_score,
        "current_ticket_price": current_price,
        "secondary_market_avg_price": secondary_price,
        "secondary_market_listing_count": values.get("secondary_market_listing_count", SECTION_CAPACITY[seat_section] * 0.08),
        "resale_velocity_index": values.get("resale_velocity_index", 78),
        "scalper_pressure_score": values.get("scalper_pressure_score", 0.50),
        "days_before_game": days_before_game,
        "pricing_window": values.get("pricing_window", DEFAULT_INPUT["pricing_window"]),
        "month": values.get("month", DEFAULT_INPUT["month"]),
        "season_stage": values.get("season_stage", DEFAULT_INPUT["season_stage"]),
        "seat_section": seat_section,
        "promotion_flag": int(promotion),
        "premium_game_flag": int(premium_game),
        "affordability_segment": affordability_segment,
        "affordability_index": AFFORDABILITY_INDEX[seat_section],
        "local_income_proxy": AFFORDABILITY_INDEX[seat_section],
    }


def run_full_recommendation(
    input_row: dict[str, Any],
    use_openai_explanation: bool = False,
) -> dict[str, Any]:
    full_prediction = predict_full(input_row, cached_models())
    prediction = full_prediction["demand"]
    normalized = prediction["input"]
    recommendation = optimize_price(
        normalized,
        demand_output=prediction,
        sellthrough_output=full_prediction["sellthrough"],
        scalper_output=full_prediction["scalper"],
    )
    model_drivers = local_driver_summary(normalized)
    explanation_context = {
        **prediction,
        "sellthrough": full_prediction["sellthrough"],
        "scalper": full_prediction["scalper"],
    }
    if use_openai_explanation:
        generated_explanation, explanation_source = generate_pricing_explanation_with_source(
            input_data=normalized,
            prediction_output=explanation_context,
            pricing_output=recommendation,
            model_drivers=model_drivers,
        )
    else:
        generated_explanation = fallback_pricing_explanation(
            input_data=normalized,
            prediction_output=explanation_context,
            pricing_output=recommendation,
            model_drivers=model_drivers,
        )
        explanation_source = (
            "Template fallback · OpenAI ready"
            if openai_explanations_enabled()
            else "Template fallback"
        )
    return {
        "prediction": prediction,
        "sellthrough": full_prediction["sellthrough"],
        "scalper": full_prediction["scalper"],
        "recommendation": recommendation,
        "generated_explanation": generated_explanation,
        "model_drivers": model_drivers,
        "explanation_source": explanation_source,
    }


def demo_scenarios() -> dict[str, dict[str, Any]]:
    return {
        "High-demand rivalry game": {
            **DEFAULT_INPUT,
            "seat_section": "Lower",
            "day_of_week": "Saturday",
            "opponent_strength": 94,
            "rivalry_flag": 1,
            "star_player_available": 1,
            "weather_severity": 0.18,
            "website_traffic_index": 205,
            "social_sentiment_score": 0.62,
            "historical_attendance_rate": 0.94,
            "current_ticket_price": 132,
            "secondary_market_avg_price": 205,
            "days_before_game": 17,
            "premium_game_flag": 1,
        },
        "Low-demand weekday game": {
            **DEFAULT_INPUT,
            "seat_section": "Upper",
            "day_of_week": "Tuesday",
            "opponent_strength": 46,
            "rivalry_flag": 0,
            "star_player_available": 1,
            "weather_severity": 0.35,
            "website_traffic_index": 72,
            "social_sentiment_score": -0.18,
            "historical_attendance_rate": 0.54,
            "current_ticket_price": 58,
            "secondary_market_avg_price": 52,
            "days_before_game": 8,
            "premium_game_flag": 0,
            "affordability_segment": "Value-sensitive",
            "local_income_proxy": 52,
        },
        "Star player injured": {
            **DEFAULT_INPUT,
            "seat_section": "Club",
            "day_of_week": "Friday",
            "opponent_strength": 80,
            "rivalry_flag": 0,
            "star_player_available": 0,
            "weather_severity": 0.22,
            "website_traffic_index": 118,
            "social_sentiment_score": -0.38,
            "historical_attendance_rate": 0.72,
            "current_ticket_price": 238,
            "secondary_market_avg_price": 221,
            "days_before_game": 11,
            "premium_game_flag": 0,
            "affordability_segment": "Affluent",
            "local_income_proxy": 113,
        },
        "Bad weather game": {
            **DEFAULT_INPUT,
            "seat_section": "Lower",
            "day_of_week": "Wednesday",
            "opponent_strength": 72,
            "rivalry_flag": 0,
            "star_player_available": 1,
            "weather_severity": 0.86,
            "website_traffic_index": 96,
            "social_sentiment_score": 0.06,
            "historical_attendance_rate": 0.66,
            "current_ticket_price": 118,
            "secondary_market_avg_price": 111,
            "days_before_game": 5,
            "premium_game_flag": 0,
        },
        "Premium playoff-style game": {
            **DEFAULT_INPUT,
            "seat_section": "Courtside/Premium",
            "day_of_week": "Sunday",
            "opponent_strength": 97,
            "rivalry_flag": 1,
            "star_player_available": 1,
            "weather_severity": 0.05,
            "website_traffic_index": 224,
            "social_sentiment_score": 0.74,
            "historical_attendance_rate": 0.98,
            "current_ticket_price": 585,
            "secondary_market_avg_price": 835,
            "days_before_game": 24,
            "premium_game_flag": 1,
            "affordability_segment": "Premium",
            "local_income_proxy": 165,
        },
    }
