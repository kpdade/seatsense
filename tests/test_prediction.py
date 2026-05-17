from src.predict import DEFAULT_INPUT, ensure_project_ready, predict_demand


def test_prediction_returns_supported_demand_tier():
    ensure_project_ready()
    result = predict_demand(DEFAULT_INPUT)
    assert result["demand_tier"] in {"Low", "Medium", "High"}
    assert 0 <= result["demand_probability"] <= 1
    assert set(result["probabilities"]).issubset({"Low", "Medium", "High"})

