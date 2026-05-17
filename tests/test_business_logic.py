from src.business_logic import recommend_price


def test_high_demand_with_resale_gap_increases_price_with_approval_guardrail():
    recommendation = recommend_price(
        current_price=120,
        demand_tier="High",
        demand_probability=0.86,
        secondary_market_price=180,
        days_before_game=15,
        seat_section="Lower",
        tickets_sold_pct=0.88,
        section_capacity=5000,
        affordability_segment="Mainstream",
        premium_game_flag=1,
        local_income_proxy=78,
    )
    assert recommendation.recommended_price > 120
    assert recommendation.price_change_pct <= 0.16
    assert recommendation.human_approval_required is True


def test_low_demand_near_event_discounts_to_reduce_unsold_inventory():
    recommendation = recommend_price(
        current_price=70,
        demand_tier="Low",
        demand_probability=0.78,
        secondary_market_price=62,
        days_before_game=5,
        seat_section="Upper",
        tickets_sold_pct=0.48,
        section_capacity=7600,
        affordability_segment="Value-sensitive",
        premium_game_flag=0,
        local_income_proxy=52,
    )
    assert recommendation.recommended_price < 70
    assert recommendation.unsold_inventory_risk > 0.30
    assert "promotion" in recommendation.business_action.lower() or "discount" in recommendation.business_action.lower()

