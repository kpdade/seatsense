from types import SimpleNamespace

from src.business_logic import recommend_price
from src.generative_explainer import generate_pricing_explanation
from src.predict import DEFAULT_INPUT


def _prediction():
    return {
        "demand_tier": "High",
        "demand_probability": 0.82,
        "probabilities": {"Low": 0.04, "Medium": 0.14, "High": 0.82},
    }


def _pricing():
    return recommend_price(
        current_price=125,
        demand_tier="High",
        demand_probability=0.82,
        secondary_market_price=175,
        days_before_game=14,
        seat_section="Lower",
        tickets_sold_pct=0.86,
        section_capacity=5200,
        affordability_segment="Mainstream",
        premium_game_flag=1,
        local_income_proxy=78,
    )


def test_fallback_explanation_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    explanation = generate_pricing_explanation(
        DEFAULT_INPUT,
        _prediction(),
        _pricing(),
        [{"driver": "Secondary-market price gap", "direction": "Suggests underpricing"}],
    )
    assert "Executive Summary" in explanation
    assert "Human Review Decision" in explanation
    assert "$" in explanation


def test_openai_explanation_path_when_key_exists(monkeypatch):
    calls = {"count": 0}

    class FakeResponses:
        def create(self, **kwargs):
            calls["count"] += 1
            assert kwargs["model"] == "gpt-4.1-mini"
            return SimpleNamespace(output_text="### 1. Executive Summary\nOpenAI path used.")

    class FakeOpenAI:
        def __init__(self, api_key):
            assert api_key == "test-key"
            self.responses = FakeResponses()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4.1-mini")
    monkeypatch.setitem(__import__("sys").modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))

    explanation = generate_pricing_explanation(
        DEFAULT_INPUT,
        _prediction(),
        _pricing(),
        [{"driver": "Elite opponent", "direction": "Raises demand"}],
    )
    assert calls["count"] == 1
    assert "OpenAI path used" in explanation

