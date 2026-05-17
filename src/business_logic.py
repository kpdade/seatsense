"""Business pricing logic and responsible guardrails for SeatSense AI."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DEMAND_MULTIPLIER = {"Low": -1, "Medium": 0, "High": 1}


@dataclass
class PricingRecommendation:
    recommended_price: float
    price_change_pct: float
    revenue_uplift_estimate: float
    secondary_market_leakage_estimate: float
    unsold_inventory_risk: float
    risk_category: str
    human_approval_required: bool
    affordability_risk_flag: bool
    business_action: str
    explanation: str


def estimate_unsold_inventory_risk(demand_tier: str, tickets_sold_pct: float, days_before_game: int) -> float:
    demand_penalty = {"Low": 0.26, "Medium": 0.12, "High": 0.04}[demand_tier]
    time_penalty = max(0, 16 - days_before_game) / 16 * 0.16
    sell_through_penalty = max(0, 0.72 - tickets_sold_pct) * 0.70
    return float(np.clip(demand_penalty + time_penalty + sell_through_penalty, 0.02, 0.88))


def calculate_price_change(
    current_price: float,
    secondary_market_price: float,
    demand_tier: str,
    demand_probability: float,
    days_before_game: int,
    seat_section: str,
    affordability_segment: str,
    premium_game_flag: int,
) -> float:
    secondary_ratio = secondary_market_price / max(current_price, 1)
    confidence_adjustment = np.clip((demand_probability - 0.50) * 0.35, -0.08, 0.10)

    if demand_tier == "High":
        change = 0.06 + max(0, secondary_ratio - 1.0) * 0.20 + confidence_adjustment
    elif demand_tier == "Medium":
        change = (secondary_ratio - 1.0) * 0.07 + confidence_adjustment * 0.35
    else:
        urgency = 0.10 if days_before_game <= 10 else 0.05
        change = -urgency + min(0, secondary_ratio - 1.0) * 0.08 - abs(confidence_adjustment) * 0.20

    # Guardrail caps keep the recommendation commercially useful but not extreme.
    max_increase = 0.18
    max_decrease = -0.20
    if seat_section == "Courtside/Premium" or premium_game_flag:
        max_increase = 0.16
    if affordability_segment == "Value-sensitive":
        max_increase = min(max_increase, 0.08)
    return float(np.clip(change, max_decrease, max_increase))


def recommend_price(
    current_price: float,
    demand_tier: str,
    demand_probability: float,
    secondary_market_price: float,
    days_before_game: int,
    seat_section: str,
    tickets_sold_pct: float,
    section_capacity: int = 5000,
    affordability_segment: str = "Mainstream",
    premium_game_flag: int = 0,
    local_income_proxy: float = 75,
) -> PricingRecommendation:
    change_pct = calculate_price_change(
        current_price=current_price,
        secondary_market_price=secondary_market_price,
        demand_tier=demand_tier,
        demand_probability=demand_probability,
        days_before_game=days_before_game,
        seat_section=seat_section,
        affordability_segment=affordability_segment,
        premium_game_flag=premium_game_flag,
    )
    recommended_price = round(current_price * (1 + change_pct), 2)
    expected_units = max(1, section_capacity * min(0.97, max(0.25, tickets_sold_pct)))

    demand_elasticity = {"Low": -1.25, "Medium": -0.85, "High": -0.45}[demand_tier]
    expected_volume_change = float(np.clip(demand_elasticity * change_pct, -0.12, 0.18))
    projected_units = expected_units * (1 + expected_volume_change)
    current_revenue = current_price * expected_units
    projected_revenue = recommended_price * projected_units
    revenue_uplift = round(projected_revenue - current_revenue, 2)

    leakage_before = max(0, secondary_market_price - current_price) * expected_units * 0.14
    leakage_after = max(0, secondary_market_price - recommended_price) * projected_units * 0.10
    leakage_estimate = round(max(0, leakage_before - leakage_after), 2)

    unsold_inventory_risk = estimate_unsold_inventory_risk(
        demand_tier, tickets_sold_pct, days_before_game
    )

    affordability_risk = bool(
        (affordability_segment == "Value-sensitive" and change_pct > 0.06)
        or (local_income_proxy < 58 and change_pct > 0.05)
        or (seat_section == "Upper" and recommended_price > 85 and change_pct > 0)
    )
    large_change = abs(change_pct) >= 0.15
    premium_decision = bool(premium_game_flag or seat_section == "Courtside/Premium")
    human_approval = bool(large_change or premium_decision or affordability_risk)

    if demand_tier == "High" and change_pct > 0:
        action = "Increase price within the approved cap and monitor resale gaps."
    elif demand_tier == "Low" and change_pct < 0:
        action = "Launch a targeted promotion or controlled discount to protect attendance."
    elif demand_tier == "Medium":
        action = "Hold close to current price and keep monitoring demand signals."
    else:
        action = "Keep price stable until more demand signal confidence is available."

    risk_category = "High" if human_approval or unsold_inventory_risk > 0.50 else "Medium" if abs(change_pct) > 0.08 else "Low"
    explanation = (
        f"SeatSense predicts {demand_tier.lower()} demand with {demand_probability:.0%} confidence. "
        f"The current primary price is ${current_price:,.2f} and the secondary-market average is "
        f"${secondary_market_price:,.2f}, which implies a resale gap of "
        f"{(secondary_market_price / max(current_price, 1) - 1):.1%}. "
        f"The recommended price is ${recommended_price:,.2f}, a {change_pct:.1%} change. "
        f"Estimated revenue impact for this section is ${revenue_uplift:,.0f}; estimated leakage reduction is "
        f"${leakage_estimate:,.0f}. Human approval is "
        f"{'required' if human_approval else 'not required'} because of the configured price, premium-game, "
        "and affordability guardrails."
    )

    return PricingRecommendation(
        recommended_price=recommended_price,
        price_change_pct=round(change_pct, 4),
        revenue_uplift_estimate=revenue_uplift,
        secondary_market_leakage_estimate=leakage_estimate,
        unsold_inventory_risk=round(unsold_inventory_risk, 3),
        risk_category=risk_category,
        human_approval_required=human_approval,
        affordability_risk_flag=affordability_risk,
        business_action=action,
        explanation=explanation,
    )


def fan_affordability_score(
    current_price: float,
    recommended_price: float,
    seat_section: str,
    local_income_proxy: float,
) -> float:
    section_weight = {
        "Upper": 1.0,
        "Lower": 0.82,
        "Club": 0.62,
        "Courtside/Premium": 0.42,
    }.get(seat_section, 0.75)
    price_pressure = max(0, recommended_price - current_price) / max(current_price, 1)
    income_buffer = np.clip((local_income_proxy - 45) / 120, 0.10, 1.0)
    score = 100 * section_weight * income_buffer * (1 - 0.72 * price_pressure)
    return float(np.clip(score, 0, 100))

