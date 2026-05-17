"""Typed schema contract for imported SeatSense training rows.

Client exports can arrive in many shapes. The adapter converts those exports
into the canonical SeatSense schema, then this module validates the converted
rows before anything reaches model training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from src.preprocess import CLASS_ORDER, REQUIRED_TRAINING_COLUMNS, SCALPER_CLASS_ORDER


DemandTier = Literal["Low", "Medium", "High"]


class SeatSenseTrainingRow(BaseModel):
    """Canonical row schema after client CSV mapping."""

    model_config = ConfigDict(extra="allow")

    game_id: str = Field(min_length=1)
    game_date: str = Field(min_length=1)
    season: int = Field(ge=2000, le=2100)
    day_of_week: str = Field(min_length=3)
    opponent: str = Field(min_length=1)
    seat_section: str = Field(min_length=1)
    pricing_window: str = Field(min_length=1)
    days_until_game_bucket: str = Field(min_length=1)
    season_stage: str = Field(min_length=1)

    section_capacity: int = Field(gt=0)
    current_ticket_price: float = Field(gt=0)
    base_ticket_price: float = Field(gt=0)
    secondary_market_avg_price: float = Field(gt=0)
    tickets_sold_pct: float = Field(ge=0, le=1)
    current_sell_through_rate: float = Field(ge=0, le=1)
    final_sell_through_rate: float = Field(ge=0, le=1)
    final_attendance_rate: float = Field(ge=0, le=1)
    attendance_rate: float = Field(ge=0, le=1)
    inventory_remaining_pct: float = Field(ge=0, le=1)
    unsold_inventory_pct: float = Field(ge=0, le=1)
    demand_tier: DemandTier
    scalper_risk_tier: DemandTier
    affordability_segment: str = Field(min_length=1)
    affordability_index: float = Field(ge=0, le=150)

    @field_validator("day_of_week")
    @classmethod
    def _valid_day(cls, value: str) -> str:
        valid = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}
        if value not in valid:
            raise ValueError(f"day_of_week must be one of {sorted(valid)}")
        return value

    @field_validator("demand_tier")
    @classmethod
    def _valid_demand_tier(cls, value: str) -> str:
        if value not in CLASS_ORDER:
            raise ValueError(f"demand_tier must be one of {CLASS_ORDER}")
        return value

    @field_validator("scalper_risk_tier")
    @classmethod
    def _valid_scalper_tier(cls, value: str) -> str:
        if value not in SCALPER_CLASS_ORDER:
            raise ValueError(f"scalper_risk_tier must be one of {SCALPER_CLASS_ORDER}")
        return value


@dataclass(frozen=True)
class SchemaValidationReport:
    passed: bool
    rows_checked: int
    errors: list[str]
    warnings: list[str]


def _format_validation_error(row_index: int, error: ValidationError) -> str:
    first = error.errors()[0]
    field_path = ".".join(str(part) for part in first.get("loc", []))
    message = first.get("msg", "Invalid value")
    return f"row {row_index}: {field_path} - {message}"


def validate_seatsense_training_schema(
    frame: pd.DataFrame,
    *,
    sample_size: int = 250,
) -> SchemaValidationReport:
    """Validate a converted training frame before model training.

    Pydantic validates a deterministic sample for row-level typing and range
    rules, while vectorized Pandas checks cover the full dataset for missing
    values and impossible ranges. This keeps 100k+ row imports responsive.
    """

    errors: list[str] = []
    warnings: list[str] = []

    missing_columns = [column for column in REQUIRED_TRAINING_COLUMNS if column not in frame.columns]
    if missing_columns:
        errors.append(f"Missing canonical SeatSense columns: {missing_columns[:25]}")
        return SchemaValidationReport(False, 0, errors, warnings)

    if frame.empty:
        return SchemaValidationReport(False, 0, ["Converted training data is empty."], warnings)

    null_critical = [
        column
        for column in [
            "game_id",
            "season",
            "seat_section",
            "current_ticket_price",
            "secondary_market_avg_price",
            "tickets_sold_pct",
            "final_sell_through_rate",
            "demand_tier",
            "scalper_risk_tier",
        ]
        if frame[column].isna().any()
    ]
    if null_critical:
        errors.append(f"Critical canonical columns contain null values: {null_critical}")

    range_checks = {
        "current_ticket_price": frame["current_ticket_price"].astype(float).gt(0),
        "base_ticket_price": frame["base_ticket_price"].astype(float).gt(0),
        "secondary_market_avg_price": frame["secondary_market_avg_price"].astype(float).gt(0),
        "section_capacity": frame["section_capacity"].astype(float).gt(0),
        "tickets_sold_pct": frame["tickets_sold_pct"].astype(float).between(0, 1),
        "current_sell_through_rate": frame["current_sell_through_rate"].astype(float).between(0, 1),
        "final_sell_through_rate": frame["final_sell_through_rate"].astype(float).between(0, 1),
        "final_attendance_rate": frame["final_attendance_rate"].astype(float).between(0, 1),
        "affordability_index": frame["affordability_index"].astype(float).between(0, 150),
    }
    for column, mask in range_checks.items():
        invalid_count = int((~mask).sum())
        if invalid_count:
            errors.append(f"{column} has {invalid_count:,} values outside the allowed range.")

    invalid_demand = sorted(set(frame["demand_tier"].astype(str)) - set(CLASS_ORDER))
    invalid_scalper = sorted(set(frame["scalper_risk_tier"].astype(str)) - set(SCALPER_CLASS_ORDER))
    if invalid_demand:
        errors.append(f"Invalid demand_tier labels: {invalid_demand}")
    if invalid_scalper:
        errors.append(f"Invalid scalper_risk_tier labels: {invalid_scalper}")

    if errors:
        return SchemaValidationReport(False, 0, errors, warnings)

    if len(frame) > sample_size:
        step = max(1, len(frame) // sample_size)
        sample = frame.iloc[::step].head(sample_size)
        warnings.append(
            f"Pydantic row validation sampled {len(sample):,} of {len(frame):,} rows; vectorized checks covered all rows."
        )
    else:
        sample = frame

    row_errors: list[str] = []
    for row_index, row in sample.iterrows():
        try:
            SeatSenseTrainingRow.model_validate(row.to_dict())
        except ValidationError as exc:
            row_errors.append(_format_validation_error(int(row_index), exc))
            if len(row_errors) >= 8:
                row_errors.append("Additional row validation errors truncated.")
                break

    if row_errors:
        errors.extend(row_errors)

    return SchemaValidationReport(
        passed=not errors,
        rows_checked=int(len(sample)),
        errors=errors,
        warnings=warnings,
    )
