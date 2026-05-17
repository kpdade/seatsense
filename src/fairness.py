"""Responsible AI and fairness audit utilities."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def _price_change_column(frame: pd.DataFrame) -> pd.Series:
    if "price_change_pct" in frame.columns:
        return frame["price_change_pct"].astype(float)
    if "recommended_price" in frame.columns:
        return (frame["recommended_price"] - frame["current_ticket_price"]) / frame[
            "current_ticket_price"
        ].clip(lower=1)
    if "recommended_price_target" in frame.columns:
        return (frame["recommended_price_target"] - frame["current_ticket_price"]) / frame[
            "current_ticket_price"
        ].clip(lower=1)
    return pd.Series(np.zeros(len(frame)), index=frame.index)


def build_fairness_audit_table(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "affordability_segment",
        "current_ticket_price",
        "attendance_rate",
        "tickets_sold_pct",
        "demand_tier",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Cannot build fairness audit. Missing columns: {missing}")

    working = frame.copy()
    working["price_change_pct"] = _price_change_column(working)
    if "human_approval_required" not in working.columns:
        working["human_approval_required"] = (
            (working["price_change_pct"].abs() > 0.20)
            | (
                working["affordability_segment"].eq("Value-sensitive")
                & (working["price_change_pct"] > 0.06)
            )
            | working.get("premium_game_flag", 0).astype(bool)
        )
    if "capped_by_guardrail" not in working.columns:
        working["capped_by_guardrail"] = (
            (working["affordability_segment"].eq("Value-sensitive") & (working["price_change_pct"] > 0.095))
            | (working["price_change_pct"] > 0.30)
        )

    grouped = (
        working.groupby("affordability_segment")
        .agg(
            rows=("game_id", "count"),
            avg_affordability_index=("affordability_index", "mean")
            if "affordability_index" in working.columns
            else ("current_ticket_price", "size"),
            avg_current_price=("current_ticket_price", "mean"),
            avg_recommended_price=(
                "recommended_price",
                "mean",
            )
            if "recommended_price" in working.columns
            else ("recommended_price_target", "mean")
            if "recommended_price_target" in working.columns
            else ("current_ticket_price", "mean"),
            avg_price_change_pct=("price_change_pct", "mean"),
            p90_price_change_pct=("price_change_pct", lambda s: s.quantile(0.90)),
            approval_rate=("human_approval_required", "mean"),
            price_cap_trigger_rate=("capped_by_guardrail", "mean"),
            avg_attendance_rate=("attendance_rate", "mean"),
            avg_sell_through=("tickets_sold_pct", "mean"),
            high_demand_share=("demand_tier", lambda s: (s == "High").mean()),
        )
        .reset_index()
    )
    grouped["affordable_reserve_effect"] = grouped.apply(
        lambda row: "High protection"
        if row["affordability_segment"] == "Value-sensitive"
        else "Standard monitoring",
        axis=1,
    )
    grouped["guardrail_status"] = grouped.apply(_guardrail_status, axis=1)
    return grouped.round(
        {
            "avg_affordability_index": 2,
            "avg_current_price": 2,
            "avg_recommended_price": 2,
            "avg_price_change_pct": 4,
            "p90_price_change_pct": 4,
            "approval_rate": 4,
            "price_cap_trigger_rate": 4,
            "avg_attendance_rate": 3,
            "avg_sell_through": 3,
            "high_demand_share": 3,
        }
    )


def _guardrail_status(row: pd.Series) -> str:
    if row["affordability_segment"] == "Value-sensitive" and row["avg_price_change_pct"] > 0.08:
        return "Warning: value segment average increase above 8% cap"
    if row["affordability_segment"] == "Value-sensitive" and row["p90_price_change_pct"] > 0.12:
        return "Review: high-end value segment increases need approval"
    if row["avg_price_change_pct"] > 0.18:
        return "Review: average increase above governance threshold"
    return "Within configured guardrails"


def build_fairness_audit(
    frame: pd.DataFrame,
    output_path: Path | None = None,
) -> dict[str, Any]:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    table = build_fairness_audit_table(frame)
    value_rows = table[table["affordability_segment"].eq("Value-sensitive")]
    warning = False
    warning_messages: list[str] = []
    if not value_rows.empty:
        value_avg = float(value_rows["avg_price_change_pct"].iloc[0])
        all_avg = float(table["avg_price_change_pct"].mean())
        if value_avg > all_avg + 0.035:
            warning = True
            warning_messages.append(
                "Value-sensitive segment receives materially higher average price increases."
            )
        if float(value_rows["p90_price_change_pct"].iloc[0]) > 0.12:
            warning = True
            warning_messages.append(
                "Upper-tail price increases in value-sensitive inventory exceed the review threshold."
            )

    audit = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protected_attributes_used_in_model": [],
        "fairness_only_columns": ["affordability_index", "affordability_segment"],
        "model_training_policy": (
            "Affordability variables are used only for monitoring, approval guardrails, "
            "and affordable ticket reserve simulation. They are excluded from demand, "
            "sell-through, and scalper-risk model features."
        ),
        "disparate_impact_warning": warning,
        "warnings": warning_messages or ["No disproportionate value-segment price increase detected."],
        "mitigation_suggestions": [
            "Keep value-section price increase caps at or below 8-10%.",
            "Reserve affordable ticket inventory for high-demand games.",
            "Route premium games and large moves to human approval.",
            "Monitor average and p90 price movement by affordability segment.",
            "Use aggregated demand signals and avoid protected attributes.",
        ],
        "summary_table": table.to_dict(orient="records"),
    }
    path = output_path or OUTPUTS_DIR / "fairness_audit.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    table.to_csv(OUTPUTS_DIR / "fairness_audit.csv", index=False)
    return audit


def failure_mode_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Failure mode": "False positive high demand",
                "Business impact": "Price too high, empty seats, fan frustration, lost concession revenue.",
                "Mitigation": "Increase caps, confidence thresholds, approval workflow, attendance monitoring.",
            },
            {
                "Failure mode": "False negative high demand",
                "Business impact": "Price too low, fast sellout, scalpers capture profit.",
                "Mitigation": "High-demand recall monitoring and secondary-market gap alerts.",
            },
            {
                "Failure mode": "Noisy social sentiment spike",
                "Business impact": "Model overreacts to temporary news or bot activity.",
                "Mitigation": "Signal smoothing, traffic confirmation, human review.",
            },
            {
                "Failure mode": "Affordability pressure",
                "Business impact": "Lower-affordability fans may be priced out of popular games.",
                "Mitigation": "Affordable ticket reserve, value bucket cap, segment audits.",
            },
            {
                "Failure mode": "Privacy overreach",
                "Business impact": "Fan trust falls if behavioral tracking is opaque.",
                "Mitigation": "Use aggregated signals, minimize data, disclose pricing factors.",
            },
        ]
    )
