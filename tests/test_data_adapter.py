import pandas as pd
import pytest

from src.data_adapter import (
    NO_COLUMN,
    adapt_uploaded_training_data,
    agency_signal_preview,
    detect_agency_feed_profile,
    infer_column_mapping,
)
from src.preprocess import FEATURE_COLUMNS, REQUIRED_TRAINING_COLUMNS


def _agency_frame(rows: int = 90) -> pd.DataFrame:
    tiers = (["Low", "Medium", "High"] * ((rows // 3) + 1))[:rows]
    return pd.DataFrame(
        {
            "eventId": [f"EVT{i:04d}" for i in range(rows)],
            "name": ["Away Team at Home Team"] * rows,
            "type": ["nba"] * rows,
            "datetimeUtc": pd.date_range("2025-01-01", periods=rows, freq="D").astype(str),
            "eventScore": [65 + (i % 30) for i in range(rows)],
            "popularityScore": [70 + (i % 20) for i in range(rows)],
            "listingCount": [100 + i for i in range(rows)],
            "averagePrice": [120 + (i % 25) for i in range(rows)],
            "medianPrice": [105 + (i % 20) for i in range(rows)],
            "lowestSgBasePrice": [55 + (i % 10) for i in range(rows)],
            "ticketmasterId": [f"TM{i:04d}" for i in range(rows)],
            "stubhubId": [1000 + i for i in range(rows)],
            "integratedProvider": ["Ticketmaster"] * rows,
            "crm_customer_id": [f"CRM{i:04d}" for i in range(rows)],
            "crm_age_group": ["25-34"] * rows,
            "crm_household_income_band": ["150k-200k"] * rows,
            "demand_tier": tiers,
        }
    )


def test_agency_feed_is_detected_and_mapped_to_safe_concepts():
    frame = _agency_frame()
    profile = detect_agency_feed_profile(frame)
    mapping = infer_column_mapping(list(frame.columns))

    assert profile["is_agency_feed"] is True
    assert mapping["game_id"] == "eventId"
    assert mapping["secondary_market_avg_price"] == "averagePrice"
    assert mapping["secondary_market_listing_count"] == "listingCount"
    assert mapping["season"] == NO_COLUMN
    assert mapping["day_of_week"] == NO_COLUMN

    signal_preview = agency_signal_preview(frame, mapping)
    assert {"game_id", "secondary_market_avg_price", "listing_count"}.issubset(signal_preview.columns)


def test_agency_adapter_validates_canonical_schema_and_excludes_crm_fields():
    frame = _agency_frame(600)
    adapted = adapt_uploaded_training_data(frame)

    assert set(REQUIRED_TRAINING_COLUMNS).issubset(adapted.frame.columns)
    assert adapted.validation_report is not None
    assert adapted.validation_report.passed is True
    assert "crm_age_group" in (adapted.excluded_sensitive_columns or [])
    assert not set(adapted.excluded_sensitive_columns or []).intersection(FEATURE_COLUMNS)


def test_agency_feed_without_outcome_is_not_used_for_supervised_training():
    frame = _agency_frame(30).drop(columns=["demand_tier"])

    with pytest.raises(ValueError, match="historical outcome"):
        adapt_uploaded_training_data(frame)


def test_sensitive_crm_column_cannot_be_mapped_into_model_concept():
    frame = _agency_frame(30)
    mapping = infer_column_mapping(list(frame.columns))
    mapping["current_ticket_price"] = "crm_household_income_band"

    with pytest.raises(ValueError, match="Sensitive CRM/demographic fields"):
        adapt_uploaded_training_data(frame, mapping)
