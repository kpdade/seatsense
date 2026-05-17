"""Flexible client CSV adapter for SeatSense training data.

Real client exports rarely match a prototype schema. This module maps common
ticketing column names into SeatSense concepts, derives safe defaults for
optional features, and preserves the no-demographics model boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import numpy as np
import pandas as pd

from data.generate_ticket_data import SECTIONS, days_bucket
from src.predict import DEFAULT_INPUT, SECTION_ALIASES
from src.preprocess import REQUIRED_TRAINING_COLUMNS
from src.schema_validation import SchemaValidationReport, validate_seatsense_training_schema


NO_COLUMN = "(not provided)"

SENSITIVE_SOURCE_PATTERNS = [
    "customer_id",
    "crm_customer",
    "age_group",
    "household_income",
    "income_band",
    "zip_region",
    "zip",
    "postal",
    "loyalty_tier",
    "membership_status",
    "historical_avg_spend",
    "season_ticket_holder",
    "distance_to_venue",
]

AGENCY_FEED_COLUMNS = {
    "eventid",
    "datetimeutc",
    "averageprice",
    "lowestprice",
    "highestprice",
    "medianprice",
    "listingcount",
    "ticketcount",
    "eventscore",
    "popularityscore",
    "ticketmasterid",
    "stubhubid",
    "integratedprovider",
}

CONCEPTS: dict[str, dict[str, Any]] = {
    "game_id": {
        "label": "Game ID",
        "required": False,
        "aliases": ["game_id", "event_id", "eventId", "match_id", "game", "event"],
    },
    "game_date": {
        "label": "Game date",
        "required": False,
        "aliases": ["game_date", "event_date", "date", "datetime", "datetimeUtc", "start_date"],
    },
    "season": {
        "label": "Season",
        "required": False,
        "aliases": ["season", "year", "season_year"],
    },
    "seat_section": {
        "label": "Seat section",
        "required": True,
        "aliases": ["seat_section", "section", "zone", "price_level", "seat_zone"],
    },
    "section_capacity": {
        "label": "Section capacity",
        "required": False,
        "aliases": ["section_capacity", "capacity", "available_seats", "total_seats", "inventory_total"],
    },
    "current_ticket_price": {
        "label": "Current ticket price",
        "required": True,
        "aliases": [
            "current_ticket_price",
            "price",
            "ticket_price",
            "face_value",
            "primary_price",
            "avg_price",
            "lowestSgBasePrice",
            "lowest_base_price",
        ],
    },
    "base_ticket_price": {
        "label": "Base ticket price",
        "required": False,
        "aliases": ["base_ticket_price", "base_price", "list_price", "initial_price", "lowestSgBasePrice"],
    },
    "secondary_market_avg_price": {
        "label": "Secondary market avg price",
        "required": False,
        "aliases": [
            "secondary_market_avg_price",
            "resale_price",
            "secondary_price",
            "stubhub_price",
            "seatgeek_price",
            "averagePrice",
            "medianPrice",
            "average_price",
            "median_price",
        ],
    },
    "secondary_market_listing_count": {
        "label": "Secondary-market listing count",
        "required": False,
        "aliases": ["secondary_market_listing_count", "listing_count", "listingCount", "listings"],
    },
    "tickets_sold": {
        "label": "Tickets sold",
        "required": False,
        "aliases": ["tickets_sold", "sold_tickets", "sales", "qty_sold", "units_sold"],
    },
    "tickets_sold_pct": {
        "label": "Current sell-through %",
        "required": False,
        "aliases": ["tickets_sold_pct", "sell_through", "sell_through_rate", "sold_pct", "current_sell_through_rate"],
    },
    "final_sell_through_rate": {
        "label": "Final sell-through / outcome",
        "required": False,
        "aliases": ["final_sell_through_rate", "final_sell_through", "final_sold_pct", "final_sales_rate"],
    },
    "demand_tier": {
        "label": "Demand tier target",
        "required": False,
        "aliases": ["demand_tier", "demand_label", "demand", "target", "class"],
    },
    "opponent": {
        "label": "Opponent",
        "required": False,
        "aliases": ["opponent", "away_team", "visiting_team", "visitor", "team", "name", "shortName"],
    },
    "opponent_strength": {
        "label": "Opponent strength",
        "required": False,
        "aliases": [
            "opponent_strength",
            "opponent_rating",
            "team_strength",
            "opponent_rank_score",
            "eventScore",
            "event_score",
        ],
    },
    "day_of_week": {
        "label": "Day of week",
        "required": False,
        "aliases": ["day_of_week", "weekday"],
    },
    "days_before_game": {
        "label": "Days before game",
        "required": False,
        "aliases": ["days_before_game", "days_out", "lead_time", "days_until_event"],
    },
    "website_traffic_index": {
        "label": "Website traffic index",
        "required": False,
        "aliases": [
            "website_traffic_index",
            "traffic",
            "site_traffic",
            "pageviews",
            "sessions",
            "popularityScore",
            "popularity_score",
        ],
    },
    "social_sentiment_score": {
        "label": "Social sentiment score",
        "required": False,
        "aliases": ["social_sentiment_score", "sentiment", "social_sentiment"],
    },
}


@dataclass
class AdaptationResult:
    frame: pd.DataFrame
    warnings: list[str]
    mapping: dict[str, str]
    validation_report: SchemaValidationReport | None = None
    feed_profile: dict[str, Any] | None = None
    excluded_sensitive_columns: list[str] | None = None


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower().strip())


def sensitive_source_columns(columns: list[str]) -> list[str]:
    flagged: list[str] = []
    for column in columns:
        normalized = _normalize_name(column)
        if any(pattern.replace("_", "") in normalized for pattern in SENSITIVE_SOURCE_PATTERNS):
            flagged.append(column)
    return flagged


def detect_agency_feed_profile(frame: pd.DataFrame) -> dict[str, Any]:
    normalized = {_normalize_name(column) for column in frame.columns}
    matched = sorted(column for column in AGENCY_FEED_COLUMNS if column in normalized)
    provider_columns = [
        column
        for column in ["ticketmasterId", "stubhubId", "integratedProvider", "integratedProviderId"]
        if column in frame.columns
    ]
    has_market_prices = bool(
        {"averageprice", "medianprice", "lowestprice", "highestprice"}.intersection(normalized)
    )
    score = len(matched)
    is_agency_feed = score >= 4 or (bool(provider_columns) and has_market_prices)
    likely_trainable = bool(
        {
            "demandtier",
            "demandlabel",
            "finalsellthroughrate",
            "finalsalesrate",
            "ticketssoldpct",
            "sellthroughrate",
            "soldpct",
            "ticketssold",
        }.intersection(normalized)
    )
    return {
        "is_agency_feed": is_agency_feed,
        "matched_signature_columns": matched,
        "provider_columns": provider_columns,
        "has_market_prices": has_market_prices,
        "likely_trainable": likely_trainable,
        "recommended_use": "Retraining-ready after mapping outcomes"
        if likely_trainable
        else "Market-signal enrichment unless historical outcomes are added",
    }


def infer_column_mapping(columns: list[str]) -> dict[str, str]:
    sensitive_columns = set(sensitive_source_columns(columns))
    normalized = {_normalize_name(column): column for column in columns if column not in sensitive_columns}
    mapping: dict[str, str] = {}
    for concept, spec in CONCEPTS.items():
        match = NO_COLUMN
        for alias in spec["aliases"]:
            alias_key = _normalize_name(alias)
            if alias_key in normalized:
                match = normalized[alias_key]
                break
        if match == NO_COLUMN:
            for alias in spec["aliases"]:
                alias_key = _normalize_name(alias)
                if len(alias_key) < 8:
                    continue
                fuzzy = next(
                    (original for key, original in normalized.items() if alias_key in key or key in alias_key),
                    None,
                )
                if fuzzy:
                    match = fuzzy
                    break
        mapping[concept] = match
    return mapping


def _validate_sensitive_mapping_boundary(mapping: dict[str, str], columns: list[str]) -> list[str]:
    sensitive_columns = sensitive_source_columns(columns)
    sensitive_set = set(sensitive_columns)
    unsafe = {
        concept: column
        for concept, column in mapping.items()
        if column in sensitive_set and concept in CONCEPTS
    }
    if unsafe:
        details = ", ".join(f"{concept} ← {column}" for concept, column in unsafe.items())
        raise ValueError(
            "Sensitive CRM/demographic fields cannot be mapped into model-training concepts. "
            f"Remove these mappings: {details}. These fields may be used only for audit/guardrail analysis."
        )
    return sensitive_columns


def agency_signal_preview(uploaded_frame: pd.DataFrame, mapping: dict[str, str] | None = None) -> pd.DataFrame:
    """Return a compact event-market signal preview from an agency-style feed."""

    mapping = {**infer_column_mapping(list(uploaded_frame.columns)), **(mapping or {})}
    preview = pd.DataFrame(index=uploaded_frame.index)
    preview["game_id"] = _series(uploaded_frame, mapping, "game_id", "").astype(str)
    preview["event_name"] = _series(uploaded_frame, mapping, "opponent", "Unknown event").astype(str)
    preview["game_date"] = pd.to_datetime(
        _series(uploaded_frame, mapping, "game_date", pd.NaT),
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")
    preview["secondary_market_avg_price"] = _numeric(
        _series(uploaded_frame, mapping, "secondary_market_avg_price", np.nan),
        np.nan,
    )
    preview["current_price_proxy"] = _numeric(
        _series(uploaded_frame, mapping, "current_ticket_price", np.nan),
        np.nan,
    )
    preview["listing_count"] = _numeric(
        _series(uploaded_frame, mapping, "secondary_market_listing_count", np.nan),
        np.nan,
    )
    preview["popularity_proxy"] = _numeric(
        _series(uploaded_frame, mapping, "website_traffic_index", np.nan),
        np.nan,
    )
    return preview.replace([np.inf, -np.inf], np.nan)


def _series(frame: pd.DataFrame, mapping: dict[str, str], concept: str, default: Any) -> pd.Series:
    column = mapping.get(concept, NO_COLUMN)
    if column and column != NO_COLUMN and column in frame.columns:
        return frame[column]
    return pd.Series([default] * len(frame), index=frame.index)


def _numeric(series: pd.Series, default: float) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default).astype(float)


def _canonical_section(value: Any) -> str:
    text = str(value or DEFAULT_INPUT["seat_section"]).strip()
    if text in SECTIONS:
        return text
    alias = SECTION_ALIASES.get(text)
    if alias in SECTIONS:
        return alias
    lowered = text.lower()
    for section in SECTIONS:
        if lowered in section.lower() or section.lower() in lowered:
            return section
    if "upper" in lowered:
        return "Upper Sideline"
    if "lower" in lowered:
        return "Lower Sideline"
    if "club" in lowered:
        return "Club Level"
    if "suite" in lowered:
        return "Suite"
    if "premium" in lowered or "court" in lowered:
        return "Courtside/Premium"
    return DEFAULT_INPUT["seat_section"]


def _normalize_tier(value: Any) -> str | None:
    text = str(value).strip().lower()
    if text in {"high", "h", "3", "premium"}:
        return "High"
    if text in {"medium", "med", "m", "2", "mid"}:
        return "Medium"
    if text in {"low", "l", "1"}:
        return "Low"
    return None


def _tier_from_sellthrough(rate: pd.Series) -> pd.Series:
    return pd.cut(
        rate,
        bins=[-np.inf, 0.64, 0.84, np.inf],
        labels=["Low", "Medium", "High"],
    ).astype(str)


def _derive_pricing_window(days: pd.Series) -> pd.Series:
    def label(value: float) -> str:
        if value <= 0:
            return "day_of_game"
        if value <= 2:
            return "2_days_out"
        if value <= 7:
            return "7_days_out"
        if value <= 14:
            return "14_days_out"
        return "30_days_out"

    return days.apply(label)


def _normalize_day_of_week(series: pd.Series) -> pd.Series:
    valid_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    lookup = {day.lower(): day for day in valid_days}
    normalized = series.astype(str).str.strip().str.lower().map(lookup)
    return normalized.fillna(DEFAULT_INPUT["day_of_week"])


def adapt_uploaded_training_data(
    uploaded_frame: pd.DataFrame,
    mapping: dict[str, str] | None = None,
) -> AdaptationResult:
    if uploaded_frame.empty:
        raise ValueError("Uploaded CSV is empty.")

    mapping = {**infer_column_mapping(list(uploaded_frame.columns)), **(mapping or {})}
    warnings: list[str] = []
    feed_profile = detect_agency_feed_profile(uploaded_frame)
    excluded_sensitive_columns = _validate_sensitive_mapping_boundary(mapping, list(uploaded_frame.columns))
    if excluded_sensitive_columns:
        warnings.append(
            "Excluded CRM/demographic fields from model training: "
            + ", ".join(excluded_sensitive_columns[:12])
            + ("..." if len(excluded_sensitive_columns) > 12 else "")
        )
    if feed_profile["is_agency_feed"] and not feed_profile["likely_trainable"]:
        warnings.append(
            "This looks like a ticketing-agency or marketplace feed. It can enrich event and market signals, "
            "but supervised retraining still requires historical sales outcomes such as final sell-through, "
            "demand tier, or tickets sold with capacity."
        )
    frame = pd.DataFrame(index=uploaded_frame.index)

    section_raw = _series(uploaded_frame, mapping, "seat_section", DEFAULT_INPUT["seat_section"])
    frame["seat_section"] = section_raw.apply(_canonical_section)
    section_config = frame["seat_section"].map(SECTIONS)

    game_date_raw = _series(uploaded_frame, mapping, "game_date", pd.NaT)
    game_dates = pd.to_datetime(game_date_raw, errors="coerce")
    if game_dates.notna().any():
        frame["game_date"] = game_dates.dt.date.astype(str)
        frame["season"] = _numeric(_series(uploaded_frame, mapping, "season", np.nan), np.nan)
        frame["season"] = frame["season"].where(frame["season"].notna(), game_dates.dt.year)
        frame["season"] = frame["season"].fillna(DEFAULT_INPUT["season"]).astype(int)
        frame["month"] = game_dates.dt.month.fillna(DEFAULT_INPUT["month"]).astype(int)
        day_names = game_dates.dt.day_name()
        frame["day_of_week"] = _series(uploaded_frame, mapping, "day_of_week", "").astype(str)
        frame["day_of_week"] = frame["day_of_week"].where(frame["day_of_week"].str.len() > 0, day_names)
        frame["day_of_week"] = _normalize_day_of_week(frame["day_of_week"])
    else:
        warnings.append("No usable game date found; using provided season if available and default weekday/month.")
        frame["game_date"] = DEFAULT_INPUT["game_date"]
        frame["season"] = _numeric(_series(uploaded_frame, mapping, "season", DEFAULT_INPUT["season"]), DEFAULT_INPUT["season"]).astype(int)
        frame["month"] = DEFAULT_INPUT["month"]
        frame["day_of_week"] = _normalize_day_of_week(
            _series(uploaded_frame, mapping, "day_of_week", DEFAULT_INPUT["day_of_week"])
        )

    game_id = _series(uploaded_frame, mapping, "game_id", "")
    if game_id.astype(str).str.strip().replace("", np.nan).isna().all():
        warnings.append("No game_id mapped; generated grouped event IDs from row order. Real deployment should provide event IDs.")
        frame["game_id"] = [f"CLIENT_GAME_{idx // 10:06d}" for idx in range(len(frame))]
    else:
        frame["game_id"] = game_id.astype(str).replace("", np.nan).ffill().fillna("CLIENT_GAME")

    frame["opponent"] = _series(uploaded_frame, mapping, "opponent", DEFAULT_INPUT["opponent"]).astype(str)
    frame["opponent_strength"] = _numeric(
        _series(uploaded_frame, mapping, "opponent_strength", DEFAULT_INPUT["opponent_strength"]),
        DEFAULT_INPUT["opponent_strength"],
    ).clip(30, 100)
    frame["home_team_win_rate"] = DEFAULT_INPUT["home_team_win_rate"]
    frame["away_team_win_rate"] = DEFAULT_INPUT["away_team_win_rate"]
    frame["home_team_recent_form"] = DEFAULT_INPUT["home_team_recent_form"]
    frame["away_team_recent_form"] = DEFAULT_INPUT["away_team_recent_form"]
    frame["weekend_flag"] = frame["day_of_week"].isin(["Friday", "Saturday", "Sunday"]).astype(int)
    frame["holiday_flag"] = 0
    frame["rivalry_flag"] = (frame["opponent_strength"] >= 88).astype(int)
    frame["division_matchup_flag"] = frame["rivalry_flag"]
    frame["playoff_implication_score"] = np.where(frame["season"].rank(pct=True) > 0.75, 0.42, 0.25)
    frame["premium_game_flag"] = ((frame["opponent_strength"] >= 88) | (frame["weekend_flag"].eq(1) & (frame["opponent_strength"] >= 82))).astype(int)

    frame["section_capacity"] = _numeric(_series(uploaded_frame, mapping, "section_capacity", np.nan), np.nan)
    default_capacity = frame["seat_section"].map(lambda sec: SECTIONS[sec].capacity)
    frame["section_capacity"] = frame["section_capacity"].where(frame["section_capacity"].notna(), default_capacity).clip(lower=50).astype(int)
    frame["seat_quality_score"] = frame["seat_section"].map(lambda sec: SECTIONS[sec].seat_quality_score)
    frame["base_ticket_price"] = _numeric(
        _series(uploaded_frame, mapping, "base_ticket_price", np.nan),
        np.nan,
    )
    frame["current_ticket_price"] = _numeric(
        _series(uploaded_frame, mapping, "current_ticket_price", DEFAULT_INPUT["current_ticket_price"]),
        DEFAULT_INPUT["current_ticket_price"],
    ).clip(lower=1)
    frame["base_ticket_price"] = frame["base_ticket_price"].where(
        frame["base_ticket_price"].notna(),
        frame["current_ticket_price"] * 0.94,
    ).clip(lower=1)
    frame["price_change_from_base_pct"] = frame["current_ticket_price"] / frame["base_ticket_price"].clip(lower=1) - 1

    days = _numeric(_series(uploaded_frame, mapping, "days_before_game", DEFAULT_INPUT["days_before_game"]), DEFAULT_INPUT["days_before_game"]).clip(0, 90)
    frame["days_before_game"] = days.astype(int)
    frame["pricing_window"] = _derive_pricing_window(days)
    frame["days_until_game_bucket"] = frame["days_before_game"].apply(days_bucket)
    frame["season_stage"] = pd.cut(
        frame.groupby("season").cumcount() / frame.groupby("season")["season"].transform("count").clip(lower=1),
        bins=[-np.inf, 0.25, 0.68, 0.90, np.inf],
        labels=["early", "mid", "late", "playoff_push"],
    ).astype(str)

    sold_pct_raw = _numeric(_series(uploaded_frame, mapping, "tickets_sold_pct", np.nan), np.nan)
    sold_units = _numeric(_series(uploaded_frame, mapping, "tickets_sold", np.nan), np.nan)
    observed_sellthrough_available = bool(
        sold_pct_raw.notna().any() or (sold_units.notna() & frame["section_capacity"].notna()).any()
    )
    sold_pct = sold_pct_raw.where(sold_pct_raw.notna(), sold_units / frame["section_capacity"].clip(lower=1))
    sold_pct = sold_pct.where(sold_pct.notna(), DEFAULT_INPUT["tickets_sold_pct"]).clip(0.02, 0.995)
    frame["tickets_sold_pct"] = sold_pct
    frame["current_sell_through_rate"] = sold_pct
    frame["sold_tickets"] = (sold_pct * frame["section_capacity"]).round().astype(int)
    frame["inventory_remaining"] = (frame["section_capacity"] - frame["sold_tickets"]).clip(lower=0).astype(int)
    frame["inventory_remaining_pct"] = frame["inventory_remaining"] / frame["section_capacity"].clip(lower=1)

    final_sell = _numeric(_series(uploaded_frame, mapping, "final_sell_through_rate", np.nan), np.nan)
    demand_raw = _series(uploaded_frame, mapping, "demand_tier", "")
    demand_from_raw = demand_raw.apply(_normalize_tier)
    if final_sell.isna().all() and demand_from_raw.isna().all() and not observed_sellthrough_available:
        raise ValueError(
            "Client training requires a historical outcome: map final sell-through, demand tier, "
            "or enough sales/capacity fields to derive demand."
        )
    final_sell = final_sell.where(final_sell.notna(), demand_from_raw.map({"Low": 0.56, "Medium": 0.76, "High": 0.91}))
    final_sell = final_sell.where(final_sell.notna(), sold_pct.clip(lower=0.05)).clip(0.05, 0.995)
    frame["final_sell_through_rate"] = final_sell
    frame["demand_tier"] = demand_from_raw.where(demand_from_raw.notna(), _tier_from_sellthrough(final_sell))

    frame["website_traffic_index"] = _numeric(
        _series(uploaded_frame, mapping, "website_traffic_index", np.nan),
        np.nan,
    )
    frame["website_traffic_index"] = frame["website_traffic_index"].where(
        frame["website_traffic_index"].notna(),
        55 + 135 * frame["tickets_sold_pct"] + 18 * frame["premium_game_flag"],
    ).clip(20, 260)
    frame["search_interest_index"] = frame["website_traffic_index"] * 0.78
    frame["social_sentiment_score"] = _numeric(
        _series(uploaded_frame, mapping, "social_sentiment_score", 0.0),
        0.0,
    ).clip(-1, 1)
    frame["social_volume_index"] = (frame["website_traffic_index"] * 0.68 + 12 * frame["premium_game_flag"]).clip(5, 235)
    frame["email_campaign_active"] = 0
    frame["promotion_flag"] = (frame["tickets_sold_pct"] < 0.55).astype(int)
    frame["marketing_spend_index"] = 58 + 18 * frame["promotion_flag"]
    frame["star_player_available"] = 1
    frame["injury_news_severity"] = 0.08
    frame["weather_severity"] = 0.18
    frame["temperature_score"] = 0.88

    frame["secondary_market_avg_price"] = _numeric(
        _series(uploaded_frame, mapping, "secondary_market_avg_price", np.nan),
        np.nan,
    )
    frame["secondary_market_avg_price"] = frame["secondary_market_avg_price"].where(
        frame["secondary_market_avg_price"].notna(),
        frame["current_ticket_price"] * (1 + 0.08 + 0.28 * frame["premium_game_flag"] + 0.18 * frame["tickets_sold_pct"]),
    ).clip(lower=1)
    frame["resale_price_gap_pct"] = frame["secondary_market_avg_price"] / frame["current_ticket_price"].clip(lower=1) - 1
    mapped_listing_count = _numeric(
        _series(uploaded_frame, mapping, "secondary_market_listing_count", np.nan),
        np.nan,
    )
    frame["secondary_market_listing_count"] = mapped_listing_count.where(
        mapped_listing_count.notna(),
        frame["section_capacity"] * (0.025 + 0.09 * frame["resale_price_gap_pct"].clip(lower=0)),
    ).clip(2, 950)
    frame["resale_velocity_index"] = (
        25 + 75 * frame["resale_price_gap_pct"].clip(lower=0) + 28 * frame["tickets_sold_pct"]
    ).clip(5, 145)
    frame["scalper_pressure_score"] = (
        0.42 * (frame["resale_price_gap_pct"].clip(lower=0) / 0.45).clip(upper=1.5)
        + 0.22 * (frame["secondary_market_listing_count"] / 520).clip(upper=1.5)
        + 0.28 * (frame["resale_velocity_index"] / 120).clip(upper=1.5)
    ).clip(0, 1)
    frame["scalper_risk_tier"] = pd.cut(
        frame["scalper_pressure_score"],
        bins=[-np.inf, 0.42, 0.72, np.inf],
        labels=["Low", "Medium", "High"],
    ).astype(str)
    frame["scalper_risk_label"] = frame["scalper_risk_tier"]

    frame["rolling_3_game_attendance_rate"] = frame.groupby("seat_section")["final_sell_through_rate"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    ).fillna(frame["tickets_sold_pct"]).clip(0.05, 0.995)
    frame["rolling_5_game_sell_through_rate"] = frame.groupby("seat_section")["final_sell_through_rate"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    ).fillna(frame["tickets_sold_pct"]).clip(0.05, 0.995)
    frame["previous_similar_game_sell_through"] = frame["rolling_5_game_sell_through_rate"]
    frame["same_opponent_last_season_attendance"] = frame["rolling_3_game_attendance_rate"]
    frame["section_historical_fill_rate"] = frame.groupby("seat_section")["final_sell_through_rate"].transform("mean").clip(0.05, 0.995)
    frame["historical_attendance_rate"] = frame["rolling_3_game_attendance_rate"]
    frame["historical_price_elasticity_section"] = frame["seat_section"].map(lambda sec: SECTIONS[sec].elasticity)

    frame["demand_signal_score"] = (
        0.20 * (frame["website_traffic_index"] / 220).clip(0, 1.2)
        + 0.16 * (frame["search_interest_index"] / 190).clip(0, 1.2)
        + 0.12 * (frame["social_volume_index"] / 210).clip(0, 1.2)
        + 0.11 * ((frame["social_sentiment_score"] + 1) / 2)
        + 0.16 * frame["current_sell_through_rate"]
        + 0.10 * (frame["resale_price_gap_pct"].clip(lower=0) / 0.50).clip(0, 1.2)
        + 0.06 * frame["premium_game_flag"]
        + 0.05 * frame["rivalry_flag"]
        + 0.04 * (1 - frame["weather_severity"])
    ).clip(0, 1.15)

    frame["affordability_segment"] = frame["seat_section"].map(lambda sec: SECTIONS[sec].affordability_segment)
    frame["affordability_index"] = frame["seat_section"].map(lambda sec: SECTIONS[sec].affordability_index)
    frame["local_income_proxy"] = frame["affordability_index"]

    frame["final_attendance_rate"] = (frame["final_sell_through_rate"] - 0.02).clip(0.05, 0.995)
    frame["realized_revenue"] = frame["final_sell_through_rate"] * frame["section_capacity"] * frame["current_ticket_price"]
    frame["revenue_per_available_seat"] = frame["realized_revenue"] / frame["section_capacity"].clip(lower=1)
    frame["unsold_inventory_pct"] = 1 - frame["final_sell_through_rate"]
    frame["actual_secondary_market_gap"] = (frame["secondary_market_avg_price"] - frame["current_ticket_price"]).clip(lower=0)
    frame["realized_demand_score"] = (
        0.50 * frame["final_sell_through_rate"] + 0.35 * frame["demand_signal_score"] + 0.15 * frame["tickets_sold_pct"]
    ).clip(0, 1.15)
    frame["demand_index"] = frame["realized_demand_score"]
    frame["optimal_price_oracle"] = frame["current_ticket_price"] * (
        1
        + np.where(frame["demand_tier"].eq("High"), 0.12, np.where(frame["demand_tier"].eq("Low"), -0.08, 0.04))
        + 0.08 * frame["resale_price_gap_pct"].clip(lower=0)
    )
    frame["recommended_price_target"] = frame["optimal_price_oracle"]
    frame["attendance_rate"] = frame["final_attendance_rate"]
    frame["revenue"] = frame["current_ticket_price"] * frame["sold_tickets"]

    for column in REQUIRED_TRAINING_COLUMNS:
        if column not in frame.columns:
            frame[column] = DEFAULT_INPUT.get(column, 0)

    frame = frame[REQUIRED_TRAINING_COLUMNS].copy()
    frame = frame.replace([np.inf, -np.inf], np.nan)
    numeric_columns = frame.select_dtypes(include=["number"]).columns
    frame[numeric_columns] = frame[numeric_columns].fillna(frame[numeric_columns].median(numeric_only=True)).fillna(0)
    frame = frame.fillna("Unknown")
    validation_report = validate_seatsense_training_schema(frame)
    if not validation_report.passed:
        raise ValueError(
            "Converted CSV failed the SeatSense Pydantic schema gate: "
            + "; ".join(validation_report.errors[:6])
        )
    warnings.extend(validation_report.warnings)
    return AdaptationResult(
        frame=frame,
        warnings=warnings,
        mapping=mapping,
        validation_report=validation_report,
        feed_profile=feed_profile,
        excluded_sensitive_columns=excluded_sensitive_columns,
    )
