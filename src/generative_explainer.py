"""Business-ready recommendation explanation generation.

SeatSense AI can use OpenAI for concise executive explanations, but it must
remain fully functional without paid API access. This module therefore loads
credentials safely, calls the OpenAI Responses API only when a key is available,
and otherwise returns a deterministic explanation grounded in the same model and
pricing outputs.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency safety
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "gpt-4.1-mini"
FALLBACK_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_OPENAI_TIMEOUT_SECONDS = 30.0
SECRET_PLACEHOLDERS = {
    "",
    "your_api_key_here",
    "your_openai_api_key_here",
    "sk-your_api_key_here",
}

if load_dotenv:
    load_dotenv(PROJECT_ROOT / ".env")


def _get_streamlit_secret(name: str) -> str | None:
    """Safely read a Streamlit secret when running inside or outside Streamlit."""

    try:
        import streamlit as st

        value = st.secrets.get(name)
        return str(value) if value else None
    except Exception:
        return None


def get_openai_api_key() -> str | None:
    """Return the OpenAI API key without logging, printing, or exposing it."""

    value = os.getenv("OPENAI_API_KEY") or _get_streamlit_secret("OPENAI_API_KEY")
    if not value:
        return None
    cleaned = str(value).strip().strip('"').strip("'")
    if cleaned.lower() in SECRET_PLACEHOLDERS:
        return None
    return cleaned


def get_openai_model() -> str:
    return os.getenv("OPENAI_MODEL") or _get_streamlit_secret("OPENAI_MODEL") or DEFAULT_MODEL


def get_openai_timeout_seconds() -> float:
    value = os.getenv("OPENAI_TIMEOUT_SECONDS") or _get_streamlit_secret("OPENAI_TIMEOUT_SECONDS")
    try:
        return max(5.0, float(value)) if value else DEFAULT_OPENAI_TIMEOUT_SECONDS
    except Exception:
        return DEFAULT_OPENAI_TIMEOUT_SECONDS


def openai_explanations_enabled() -> bool:
    return bool(get_openai_api_key())


def _safe_openai_error_message(exc: Exception) -> str:
    """Return a non-secret diagnostic that is safe to show in the UI."""

    error_type = exc.__class__.__name__
    status_code = getattr(exc, "status_code", None)
    raw_message = str(exc)
    lowered = raw_message.lower()
    if "api key" in lowered or "incorrect api key" in lowered or "invalid_api_key" in lowered:
        reason = "invalid or revoked API key"
    elif "insufficient_quota" in lowered or "quota" in lowered or "billing" in lowered:
        reason = "quota or billing issue"
    elif status_code == 401:
        reason = "authentication failed"
    elif status_code == 403:
        reason = "model or project access denied"
    elif status_code == 404 or "model" in lowered and "not" in lowered:
        reason = "configured model is unavailable"
    elif "timeout" in lowered or error_type.lower().endswith("timeout"):
        reason = "request timed out"
    elif "responses" in lowered and "attribute" in lowered:
        reason = "OpenAI SDK version does not support Responses API"
    else:
        reason = error_type
    return f"{reason} ({error_type}{f', HTTP {status_code}' if status_code else ''})"


def _to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return value
    return {
        key: getattr(value, key)
        for key in dir(value)
        if not key.startswith("_") and not callable(getattr(value, key))
    }


def _money(value: Any) -> str:
    try:
        return f"${float(value):,.0f}"
    except Exception:
        return "not available"


def _price(value: Any) -> str:
    try:
        return f"${float(value):,.2f}"
    except Exception:
        return "not available"


def _pct(value: Any) -> str:
    try:
        return f"{float(value):.1%}"
    except Exception:
        return "not available"


def _format_drivers(model_drivers: Any) -> list[str]:
    if not model_drivers:
        return ["Balanced signal mix: no single driver dominates the decision."]
    formatted: list[str] = []
    for item in model_drivers:
        if isinstance(item, dict):
            driver = item.get("driver") or item.get("feature") or "Model signal"
            direction = item.get("direction") or item.get("impact") or item.get("importance") or ""
            formatted.append(f"{driver}: {direction}".strip(": "))
        else:
            formatted.append(str(item))
    return formatted[:6]


def _context_payload(
    input_data: dict[str, Any],
    prediction_output: dict[str, Any],
    pricing_output: Any,
    model_drivers: Any,
) -> dict[str, Any]:
    pricing = _to_dict(pricing_output)
    normalized_input = dict(input_data or {})
    drivers = _format_drivers(model_drivers)
    return {
        "game_context": {
            "seat_section": normalized_input.get("seat_section"),
            "day_of_week": normalized_input.get("day_of_week"),
            "opponent_strength": normalized_input.get("opponent_strength"),
            "rivalry_flag": bool(normalized_input.get("rivalry_flag")),
            "star_player_available": bool(normalized_input.get("star_player_available")),
            "weather_severity": normalized_input.get("weather_severity"),
            "days_before_game": normalized_input.get("days_before_game"),
            "premium_game_flag": bool(normalized_input.get("premium_game_flag")),
        },
        "market_signals": {
            "current_price": normalized_input.get("current_ticket_price"),
            "secondary_market_avg_price": normalized_input.get("secondary_market_avg_price"),
            "website_traffic_index": normalized_input.get("website_traffic_index"),
            "social_sentiment_score": normalized_input.get("social_sentiment_score"),
            "historical_attendance_rate": normalized_input.get("historical_attendance_rate"),
        },
        "demand_prediction": {
            "demand_tier": prediction_output.get("demand_tier"),
            "confidence": prediction_output.get("demand_probability"),
            "probabilities": prediction_output.get("probabilities"),
        },
        "sellthrough_prediction": prediction_output.get("sellthrough", {}),
        "scalper_risk_prediction": prediction_output.get("scalper", {}),
        "pricing_recommendation": {
            "recommended_price": pricing.get("recommended_price"),
            "price_change_pct": pricing.get("price_change_pct"),
            "revenue_uplift_estimate": pricing.get("revenue_uplift_estimate"),
            "secondary_market_leakage_estimate": pricing.get(
                "secondary_market_leakage_estimate"
            ),
            "unsold_inventory_risk": pricing.get("unsold_inventory_risk"),
            "risk_category": pricing.get("risk_category"),
            "business_action": pricing.get("business_action"),
        },
        "responsible_ai_guardrails": {
            "human_approval_required": pricing.get("human_approval_required"),
            "affordability_risk_flag": pricing.get("affordability_risk_flag"),
            "guardrails": [
                "Price increase caps",
                "Value-ticket affordability protection",
                "Human review for premium games and high-impact changes",
                "Manual override by revenue manager",
                "Use aggregated traffic and sentiment signals",
            ],
        },
        "top_feature_drivers": drivers,
    }


def fallback_pricing_explanation(
    input_data: dict[str, Any],
    prediction_output: dict[str, Any],
    pricing_output: Any,
    model_drivers: Any,
) -> str:
    pricing = _to_dict(pricing_output)
    drivers = _format_drivers(model_drivers)
    demand_tier = prediction_output.get("demand_tier", "Unknown")
    confidence = prediction_output.get("demand_probability", 0)
    sellthrough = prediction_output.get("sellthrough", {}).get("predicted_sell_through")
    scalper_risk = prediction_output.get("scalper", {}).get("scalper_risk_tier")
    current_price = input_data.get("current_ticket_price")
    resale_price = input_data.get("secondary_market_avg_price")
    recommended_price = pricing.get("recommended_price")
    approval_required = bool(pricing.get("human_approval_required"))
    affordability_flag = bool(pricing.get("affordability_risk_flag"))

    if demand_tier == "High":
        tradeoff = (
            "The larger risk is underpricing: inventory may sell out quickly while resale markets "
            "capture the remaining willingness to pay."
        )
    elif demand_tier == "Low":
        tradeoff = (
            "The larger risk is overpricing: empty seats can reduce ticket revenue, concessions, "
            "and in-venue energy."
        )
    else:
        tradeoff = (
            "The recommendation is intentionally measured because demand signals are mixed and "
            "overreacting could create unnecessary price volatility."
        )

    driver_lines = "\n".join(f"- {driver}" for driver in drivers)
    approval_note = (
        "Human approval is recommended before publishing this change."
        if approval_required
        else "This recommendation is within the configured auto-approval guardrails."
    )
    fairness_note = (
        "The fan affordability guardrail is active; preserve value-priced inventory and review the change manually."
        if affordability_flag
        else "No affordability exception was triggered, but value-priced inventory should continue to be monitored."
    )

    return f"""### 1. Executive Summary
SeatSense classifies this opportunity as **{demand_tier} demand** with **{_pct(confidence)} confidence**. The current primary price is **{_price(current_price)}**, the secondary-market average is **{_price(resale_price)}**, and the recommended price is **{_price(recommended_price)}**.

### 2. Pricing Recommendation
Set the {input_data.get("seat_section", "selected")} section price to **{_price(recommended_price)}**, a **{_pct(pricing.get("price_change_pct"))}** change from the current price. {pricing.get("business_action", "Continue monitoring demand signals.")}

### 3. Key Drivers
{driver_lines}

### 4. Business Impact
Expected section revenue impact is **{_money(pricing.get("revenue_uplift_estimate"))}** with an estimated **{_money(pricing.get("secondary_market_leakage_estimate"))}** reduction in secondary-market leakage. The sell-through model estimates **{_pct(sellthrough)}** final sell-through, scalper risk is **{scalper_risk or "not available"}**, and unsold inventory risk is estimated at **{_pct(pricing.get("unsold_inventory_risk"))}**.

### 5. Risks & Guardrails
{tradeoff} {fairness_note} SeatSense applies price caps, premium-game review, value-ticket protection, and manual override controls.

### 6. Human Review Decision
**{"Human approval required" if approval_required else "Auto-approval acceptable"}**. {approval_note}
"""


def _generate_openai_pricing_explanation(
    input_data: dict[str, Any],
    prediction_output: dict[str, Any],
    pricing_output: Any,
    model_drivers: Any,
) -> tuple[str | None, str | None]:
    try:
        from openai import OpenAI
    except Exception as exc:
        return None, _safe_openai_error_message(exc)

    api_key = get_openai_api_key()
    if not api_key:
        return None, "no OpenAI key detected"
    context = _context_payload(
        input_data=input_data,
        prediction_output=prediction_output,
        pricing_output=pricing_output,
        model_drivers=model_drivers,
    )
    system_prompt = (
        "You are a senior revenue strategy analyst for a professional sports venue. "
        "You explain AI-powered ticket pricing recommendations to executives and revenue managers. "
        "Be concise, business-focused, transparent about uncertainty, and include responsible AI considerations. "
        "Do not invent numbers beyond the provided inputs."
    )
    user_prompt = f"""
Create a concise, professional pricing recommendation in markdown for a venue revenue manager.

Use exactly these section headings:
1. Executive Summary
2. Pricing Recommendation
3. Key Drivers
4. Business Impact
5. Risks & Guardrails
6. Human Review Decision

    Structured context:
{json.dumps(context, indent=2, default=str)}
"""

    errors: list[str] = []
    configured_model = get_openai_model()
    candidate_models = [configured_model]
    if configured_model != FALLBACK_OPENAI_MODEL:
        candidate_models.append(FALLBACK_OPENAI_MODEL)

    try:
        try:
            client = OpenAI(api_key=api_key, timeout=get_openai_timeout_seconds())
        except TypeError:
            client = OpenAI(api_key=api_key)
    except Exception as exc:
        return None, _safe_openai_error_message(exc)

    for model_name in candidate_models:
        try:
            response = client.responses.create(
                model=model_name,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_output_tokens=700,
            )
            output_text = getattr(response, "output_text", None)
            if output_text and output_text.strip():
                source = "OpenAI Responses API"
                if model_name != configured_model:
                    source = f"OpenAI Responses API · {model_name} fallback"
                return output_text.strip(), source
        except Exception as exc:
            errors.append(_safe_openai_error_message(exc))

    return None, "; ".join(dict.fromkeys(errors)) or "OpenAI call failed"


def generate_pricing_explanation_with_source(
    input_data: dict[str, Any],
    prediction_output: dict[str, Any],
    pricing_output: Any,
    model_drivers: Any,
) -> tuple[str, str]:
    """Return explanation text plus the source used to generate it."""

    if get_openai_api_key():
        openai_text, openai_source_or_error = _generate_openai_pricing_explanation(
            input_data, prediction_output, pricing_output, model_drivers
        )
        if openai_text:
            return openai_text, openai_source_or_error or "OpenAI Responses API"
        fallback_text = fallback_pricing_explanation(
            input_data, prediction_output, pricing_output, model_drivers
        )
        diagnostic = openai_source_or_error or "OpenAI call failed"
        return fallback_text, f"Template fallback · {diagnostic}"

    fallback_text = fallback_pricing_explanation(
        input_data, prediction_output, pricing_output, model_drivers
    )
    return fallback_text, "Template fallback · no OpenAI key"


def generate_pricing_explanation(
    input_data: dict[str, Any],
    prediction_output: dict[str, Any],
    pricing_output: Any,
    model_drivers: Any,
) -> str:
    """Generate an executive-ready pricing explanation.

    Uses OpenAI when `OPENAI_API_KEY` is available from the environment or
    Streamlit secrets. Falls back to deterministic markdown on missing keys,
    missing SDK, or API failures.
    """

    explanation, _source = generate_pricing_explanation_with_source(
        input_data, prediction_output, pricing_output, model_drivers
    )
    return explanation


def generate_recommendation_explanation(context: dict[str, Any]) -> str:
    """Backward-compatible wrapper for older app code."""

    pricing_context = {
        "recommended_price": context.get("recommended_price"),
        "price_change_pct": context.get("price_change_pct"),
        "revenue_uplift_estimate": context.get("revenue_uplift_estimate"),
        "secondary_market_leakage_estimate": context.get(
            "secondary_market_leakage_estimate"
        ),
        "unsold_inventory_risk": context.get("unsold_inventory_risk"),
        "risk_category": context.get("risk_category"),
        "human_approval_required": context.get("human_approval_required"),
        "affordability_risk_flag": context.get("affordability_risk_flag"),
        "business_action": context.get("business_action"),
    }
    prediction_context = {
        "demand_tier": context.get("demand_tier"),
        "demand_probability": context.get("demand_probability"),
        "probabilities": context.get("probabilities"),
    }
    drivers = context.get("model_drivers", [])
    return generate_pricing_explanation(
        input_data=context,
        prediction_output=prediction_context,
        pricing_output=pricing_context,
        model_drivers=drivers,
    )
