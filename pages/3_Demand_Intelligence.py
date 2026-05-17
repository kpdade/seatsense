from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.app_utils import configure_page, demo_scenarios, run_full_recommendation
from src.ui_components import (
    CHART_COLORS,
    render_feature_card,
    render_footer,
    render_kpi_card,
    render_page_header,
    render_section,
    render_status_badge,
    style_plotly,
)


configure_page("SeatSense AI · Demand Intelligence")

render_page_header(
    "Demand Intelligence",
    "Understand why the model sees demand risk or revenue opportunity before the pricing recommendation is applied.",
    kicker="Forecast diagnostics",
)

scenario = st.selectbox("Scenario", list(demo_scenarios().keys()), key="demand_scenario")
result = run_full_recommendation(demo_scenarios()[scenario])
prediction = result["prediction"]
sellthrough = result["sellthrough"]
scalper = result["scalper"]
normalized = prediction["input"]

top_cols = st.columns(4)
with top_cols[0]:
    render_kpi_card("Demand tier", prediction["demand_tier"], note="Forecast class")
    render_status_badge(f"{prediction['demand_tier']} Demand", prediction["demand_tier"])
with top_cols[1]:
    render_kpi_card("Confidence", f"{prediction['demand_probability']:.1%}", note="Class probability")
with top_cols[2]:
    render_kpi_card("Sell-through", f"{sellthrough['predicted_sell_through']:.1%}", note="Regression estimate")
with top_cols[3]:
    render_kpi_card("Scalper risk", scalper["scalper_risk_tier"], note=f"{scalper['scalper_risk_probability']:.0%} confidence")

left, right = st.columns([0.52, 0.48])
with left:
    prob_df = pd.DataFrame(
        [{"Demand tier": k, "Probability": v} for k, v in prediction["probabilities"].items()]
    )
    fig = px.bar(
        prob_df,
        x="Demand tier",
        y="Probability",
        color="Demand tier",
        color_discrete_map={"Low": "#64748B", "Medium": "#F59E0B", "High": "#14B8A6"},
        title="Demand Probability Distribution",
        category_orders={"Demand tier": ["Low", "Medium", "High"]},
    )
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    fig.update_layout(showlegend=False)
    st.plotly_chart(style_plotly(fig, height=380), width="stretch")
with right:
    drivers = pd.DataFrame(result["model_drivers"])
    drivers["Driver rank"] = range(len(drivers), 0, -1)
    fig_driver = px.bar(
        drivers.sort_values("Driver rank"),
        x="Driver rank",
        y="driver",
        color="direction",
        orientation="h",
        color_discrete_sequence=CHART_COLORS,
        title="Top Demand Drivers",
        labels={"driver": "", "Driver rank": "Relative influence"},
    )
    fig_driver.update_layout(showlegend=False)
    st.plotly_chart(style_plotly(fig_driver, height=380), width="stretch")

render_section(
    "Demand signal cards",
    "Signals are business-readable so revenue teams can challenge the recommendation before publishing changes.",
    "Signal review",
)
gap = normalized["secondary_market_avg_price"] - normalized["current_ticket_price"]
cards = [
    ("Opponent strength", f"{normalized['opponent_strength']}/100", "Brand draw and competitive quality."),
    ("Website traffic", f"{normalized['website_traffic_index']:.0f}", "Aggregated ticketing interest signal."),
    ("Sentiment", f"{normalized['social_sentiment_score']:.2f}", "Social buzz and market momentum."),
    ("Secondary gap", f"${gap:,.0f}", "Potential underpricing or resale leakage."),
    ("Star availability", "Available" if normalized["star_player_available"] else "Unavailable", "Expected impact on fan willingness to pay."),
]
cols = st.columns(5)
for col, (title, value, body) in zip(cols, cards):
    with col:
        render_feature_card(title, f"{value}. {body}")

render_footer()
