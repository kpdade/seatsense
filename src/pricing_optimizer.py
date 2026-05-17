"""Constrained ticket price optimizer for SeatSense AI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


VALUE_SEGMENTS = {"Value-sensitive", "Value", "Low affordability"}


@dataclass
class PricingOptimizationResult:
    recommended_price: float
    price_change_pct: float
    expected_sell_through: float
    expected_revenue: float
    revenue_uplift_vs_current: float
    secondary_market_leakage_reduction: float
    affordability_risk_flag: bool
    human_approval_required: bool
    pricing_action: str
    risk_category: str
    unsold_inventory_risk: float
    capped_by_guardrail: bool
    approval_reasons: list[str] = field(default_factory=list)
    candidate_table: list[dict[str, Any]] = field(default_factory=list)
    explanation: str = ""

    @property
    def revenue_uplift_estimate(self) -> float:
        return self.revenue_uplift_vs_current

    @property
    def secondary_market_leakage_estimate(self) -> float:
        return self.secondary_market_leakage_reduction

    @property
    def business_action(self) -> str:
        if self.pricing_action == "Increase":
            return "Increase price within guardrails and monitor resale response."
        if self.pricing_action == "Discount":
            return "Use a controlled discount or promotion to protect attendance."
        if self.pricing_action == "Review":
            return "Route to human review before publishing the price."
        return "Hold current price and continue monitoring demand signals."


def _demand_probability(prediction: dict[str, Any]) -> float:
    return float(prediction.get("demand_probability", prediction.get("confidence", 0.65)) or 0.65)


def _demand_tier(prediction: dict[str, Any]) -> str:
    return str(prediction.get("demand_tier", "Medium"))


def _scalper_tier(scalper_output: dict[str, Any] | None) -> str:
    if not scalper_output:
        return "Medium"
    return str(scalper_output.get("scalper_risk_tier", scalper_output.get("risk_tier", "Medium")))


def _estimate_candidate_sellthrough(
    base_sell_through: float,
    candidate_price: float,
    current_price: float,
    elasticity: float,
    demand_tier: str,
    promotion_flag: int,
) -> float:
    price_change = candidate_price / max(current_price, 1) - 1
    elasticity_effect = elasticity * price_change
    demand_buffer = {"Low": -0.05, "Medium": 0.0, "High": 0.045}.get(demand_tier, 0)
    promotion_lift = 0.025 if promotion_flag and price_change <= 0 else 0
    return float(np.clip(base_sell_through * (1 + elasticity_effect) + demand_buffer + promotion_lift, 0.22, 0.995))


def optimize_price(
    input_data: dict[str, Any],
    demand_output: dict[str, Any],
    sellthrough_output: dict[str, Any] | float | None = None,
    scalper_output: dict[str, Any] | None = None,
) -> PricingOptimizationResult:
    """Recommend a price using model outputs plus business constraints."""

    current_price = float(input_data.get("current_ticket_price", 100))
    base_price = float(input_data.get("base_ticket_price", current_price))
    secondary_price = float(input_data.get("secondary_market_avg_price", current_price))
    section_capacity = int(float(input_data.get("section_capacity", 5000)))
    inventory_remaining = int(float(input_data.get("inventory_remaining", section_capacity * 0.35)))
    demand_tier = _demand_tier(demand_output)
    demand_probability = _demand_probability(demand_output)
    scalper_tier = _scalper_tier(scalper_output)
    premium_game = int(input_data.get("premium_game_flag", 0))
    seat_section = str(input_data.get("seat_section", "Lower Sideline"))
    pricing_window = str(input_data.get("pricing_window", "14_days_out"))
    affordability_segment = str(input_data.get("affordability_segment", "Mainstream"))
    affordability_index = float(input_data.get("affordability_index", input_data.get("local_income_proxy", 75)))
    promotion_flag = int(input_data.get("promotion_flag", 0))
    price_change_from_base = float(input_data.get("price_change_from_base_pct", current_price / max(base_price, 1) - 1))
    elasticity = float(input_data.get("historical_price_elasticity_section", -0.85))

    if isinstance(sellthrough_output, dict):
        base_sell_through = float(
            sellthrough_output.get("predicted_sell_through", sellthrough_output.get("sell_through_rate", 0.78))
        )
    elif sellthrough_output is None:
        known_sell_through = 1 - inventory_remaining / max(section_capacity, 1)
        base_sell_through = float(np.clip(known_sell_through + 0.12, 0.32, 0.96))
    else:
        base_sell_through = float(sellthrough_output)
    base_sell_through = float(np.clip(base_sell_through, 0.25, 0.98))

    normal_cap = 0.20
    high_cap = 0.30
    premium_cap = 0.35
    if premium_game or seat_section in {"Courtside/Premium", "Suite"}:
        max_increase = premium_cap
    elif demand_tier == "High":
        max_increase = high_cap
    else:
        max_increase = normal_cap

    affordability_risk = bool(
        affordability_segment in VALUE_SEGMENTS
        or affordability_index <= 55
        or seat_section in {"Upper Baseline", "Standing Room", "Upper Sideline"}
    )
    if affordability_risk:
        max_increase = min(max_increase, 0.10)

    min_multiplier = 0.70
    max_multiplier = 1.40
    candidate_multipliers = np.arange(min_multiplier, max_multiplier + 0.0001, 0.025)
    candidates: list[dict[str, Any]] = []
    current_expected_units = section_capacity * base_sell_through
    current_expected_revenue = current_price * current_expected_units
    current_leakage = max(0, secondary_price - current_price) * current_expected_units * 0.14

    for multiplier in candidate_multipliers:
        candidate_price = float(base_price * multiplier)
        price_change_pct = candidate_price / max(current_price, 1) - 1
        if price_change_pct > max_increase or price_change_pct < -0.30:
            continue

        candidate_sell_through = _estimate_candidate_sellthrough(
            base_sell_through,
            candidate_price,
            current_price,
            elasticity,
            demand_tier,
            promotion_flag,
        )
        if pricing_window in {"2_days_out", "day_of_game"} and price_change_pct > 0.12:
            candidate_sell_through -= 0.015
        candidate_sell_through = float(np.clip(candidate_sell_through, 0.22, 0.995))

        affordable_reserve_pct = 0.08 if affordability_risk else 0.04
        sellable_units = section_capacity * candidate_sell_through
        reserve_discount_drag = affordable_reserve_pct * max(candidate_price - base_price, 0) * 0.35
        expected_revenue = candidate_price * sellable_units - reserve_discount_drag * section_capacity
        resale_gap_after = max(0, secondary_price - candidate_price)
        leakage_after = resale_gap_after * sellable_units * 0.09
        guardrail_penalty = 0.0
        if affordability_risk and price_change_pct > 0.08:
            guardrail_penalty += expected_revenue * 0.04
        if abs(price_change_from_base) > 0.18 and price_change_pct > 0.08:
            guardrail_penalty += expected_revenue * 0.025
        if demand_probability < 0.58 and abs(price_change_pct) > 0.08:
            guardrail_penalty += expected_revenue * 0.025
        objective_value = expected_revenue - guardrail_penalty + max(0, current_leakage - leakage_after) * 0.18

        candidates.append(
            {
                "candidate_price": round(candidate_price, 2),
                "price_change_pct": float(price_change_pct),
                "expected_sell_through": float(candidate_sell_through),
                "expected_revenue": float(expected_revenue),
                "secondary_market_leakage_reduction": float(max(0, current_leakage - leakage_after)),
                "objective_value": float(objective_value),
            }
        )

    if not candidates:
        candidates = [
            {
                "candidate_price": current_price,
                "price_change_pct": 0.0,
                "expected_sell_through": base_sell_through,
                "expected_revenue": current_expected_revenue,
                "secondary_market_leakage_reduction": 0.0,
                "objective_value": current_expected_revenue,
            }
        ]

    candidate_frame = pd.DataFrame(candidates)
    best = candidate_frame.sort_values("objective_value", ascending=False).iloc[0].to_dict()
    recommended_price = round(float(best["candidate_price"]), 2)
    price_change_pct = float(best["price_change_pct"])
    capped_by_guardrail = bool(
        (price_change_pct >= max_increase - 0.015 and demand_tier == "High")
        or (affordability_risk and price_change_pct >= 0.095)
    )

    approval_reasons: list[str] = []
    if price_change_pct > 0.20:
        approval_reasons.append("price increase above 20%")
    if premium_game or seat_section in {"Courtside/Premium", "Suite"}:
        approval_reasons.append("premium game or premium inventory")
    if affordability_risk and price_change_pct > 0.06:
        approval_reasons.append("fan affordability guardrail")
    if demand_probability < 0.60:
        approval_reasons.append("model confidence below threshold")
    if scalper_tier == "High" and price_change_pct > 0.16:
        approval_reasons.append("high scalper risk with aggressive price increase")
    if capped_by_guardrail:
        approval_reasons.append("recommendation constrained by guardrail cap")

    human_approval = bool(approval_reasons)
    revenue_uplift = float(best["expected_revenue"] - current_expected_revenue)
    unsold_inventory_risk = float(np.clip(1 - float(best["expected_sell_through"]), 0, 1))

    if human_approval and abs(price_change_pct) >= 0.08:
        pricing_action = "Review"
    elif price_change_pct > 0.035:
        pricing_action = "Increase"
    elif price_change_pct < -0.035:
        pricing_action = "Discount"
    else:
        pricing_action = "Hold"

    risk_category = "High" if human_approval or unsold_inventory_risk > 0.34 else "Medium" if abs(price_change_pct) > 0.08 else "Low"

    explanation = (
        f"SeatSense evaluated {len(candidate_frame)} feasible prices from 70% to 140% of base price. "
        f"The selected price is ${recommended_price:,.2f}, a {price_change_pct:.1%} change from the current price. "
        f"It balances {demand_tier.lower()} demand, predicted sell-through of {float(best['expected_sell_through']):.1%}, "
        f"and a resale gap of {(secondary_price / max(current_price, 1) - 1):.1%}. "
        f"Human approval is {'required' if human_approval else 'not required'}."
    )

    return PricingOptimizationResult(
        recommended_price=recommended_price,
        price_change_pct=round(price_change_pct, 4),
        expected_sell_through=round(float(best["expected_sell_through"]), 4),
        expected_revenue=round(float(best["expected_revenue"]), 2),
        revenue_uplift_vs_current=round(revenue_uplift, 2),
        secondary_market_leakage_reduction=round(float(best["secondary_market_leakage_reduction"]), 2),
        affordability_risk_flag=affordability_risk,
        human_approval_required=human_approval,
        pricing_action=pricing_action,
        risk_category=risk_category,
        unsold_inventory_risk=round(unsold_inventory_risk, 4),
        capped_by_guardrail=capped_by_guardrail,
        approval_reasons=approval_reasons,
        candidate_table=candidate_frame.round(4).to_dict(orient="records"),
        explanation=explanation,
    )
