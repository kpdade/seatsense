from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.app_utils import configure_page, load_metrics, load_training_data
from src.ui_components import (
    CHART_COLORS,
    render_feature_card,
    render_footer,
    render_kpi_card,
    render_page_header,
    render_section,
    style_plotly,
)


configure_page("SeatSense AI · Revenue Impact")

render_page_header(
    "Revenue Impact",
    "Quantify the upside from smarter pricing: retained revenue, lower leakage, better sell-through, and more targeted marketing actions.",
    kicker="Executive KPI dashboard",
)

data = load_training_data()
metrics = load_metrics()
business = metrics["business_metrics"]
section_revenue = (
    data.groupby(["game_id", "seat_section"])["revenue"].mean().reset_index().groupby("game_id")["revenue"].sum()
)

cols = st.columns(5)
with cols[0]:
    render_kpi_card("Revenue per game", f"${section_revenue.mean()/1000:.0f}K", note="Current average")
with cols[1]:
    render_kpi_card("Total uplift", f"${business['estimated_revenue_uplift']/1_000_000:.2f}M", delta=f"{business['estimated_revenue_uplift_pct']:.1%}", note="AI-optimized estimate")
with cols[2]:
    render_kpi_card("Sell-through", f"{data['tickets_sold_pct'].mean():.1%}", note="Average inventory sold")
with cols[3]:
    render_kpi_card("Resale leakage", f"${business['secondary_market_leakage_reduction']/1_000_000:.1f}M", note="Estimated reduction")
with cols[4]:
    render_kpi_card("Affordability score", f"{business['fan_affordability_score']:.0f}/100", note="Guardrail KPI")

guard_cols = st.columns(3)
with guard_cols[0]:
    render_kpi_card("Human review", f"{business.get('percentage_human_approval_required', 0):.1%}", note="Recommendations requiring approval")
with guard_cols[1]:
    render_kpi_card("Guardrail caps", f"{business.get('percentage_capped_by_guardrails', 0):.1%}", note="Price recommendations constrained")
with guard_cols[2]:
    render_kpi_card("Sell-through lift", f"{business.get('estimated_sell_through_improvement', 0):.1%}", note="Expected operational improvement")

render_section(
    "Revenue uplift scenarios",
    "Use conservative, expected, and aggressive realization assumptions for executive discussion.",
    "Scenario model",
)
adoption = st.slider("Recommendation adoption", 0.25, 1.0, 0.75, 0.05)
realization = st.slider("Expected realization", 0.25, 1.0, 0.70, 0.05)
scenario_factor = pd.DataFrame(
    [
        {"Scenario": "Conservative", "Multiplier": 0.45},
        {"Scenario": "Expected", "Multiplier": realization},
        {"Scenario": "Aggressive", "Multiplier": 1.00},
    ]
)
scenario_factor["Revenue uplift"] = scenario_factor["Multiplier"] * adoption * business["estimated_revenue_uplift"]
fig = px.bar(
    scenario_factor,
    x="Scenario",
    y="Revenue uplift",
    text=scenario_factor["Revenue uplift"].map(lambda x: f"${x/1_000_000:.2f}M"),
    color="Scenario",
    color_discrete_sequence=CHART_COLORS,
    title="Projected Revenue Uplift",
)
fig.update_layout(showlegend=False)
st.plotly_chart(style_plotly(fig, height=390), width="stretch")

left, right = st.columns(2)
with left:
    current = business["current_total_revenue"]
    projected = business["projected_total_revenue"]
    fig_rev = go.Figure()
    fig_rev.add_bar(
        x=["Current revenue", "AI-optimized revenue"],
        y=[current, projected],
        marker_color=[CHART_COLORS[5], CHART_COLORS[1]],
        text=[f"${current/1_000_000:.1f}M", f"${projected/1_000_000:.1f}M"],
        textposition="outside",
    )
    fig_rev.update_layout(showlegend=False, title="Current vs AI-Optimized Ticket Revenue")
    st.plotly_chart(style_plotly(fig_rev, height=370), width="stretch")
with right:
    leakage = pd.DataFrame(
        [
            {"Metric": "Before SeatSense", "Value": business["secondary_market_leakage_before"]},
            {"Metric": "After SeatSense", "Value": business["secondary_market_leakage_after"]},
        ]
    )
    fig_leak = px.bar(
        leakage,
        x="Metric",
        y="Value",
        color="Metric",
        color_discrete_sequence=[CHART_COLORS[3], CHART_COLORS[1]],
        text=leakage["Value"].map(lambda x: f"${x/1_000_000:.1f}M"),
        title="Secondary Market Leakage Exposure",
    )
    fig_leak.update_layout(showlegend=False)
    st.plotly_chart(style_plotly(fig_leak, height=370), width="stretch")

render_section(
    "Competitive advantage",
    "The business value is not only higher price. SeatSense improves the operating rhythm for revenue teams.",
    "Why it wins",
)
adv_cols = st.columns(4)
advantages = [
    ("Better revenue capture", "Price premium games closer to true demand before resale markets capture the upside."),
    ("Reduced scalper leakage", "Use resale gaps as signals while keeping primary-market guardrails in place."),
    ("Better inventory management", "Move earlier on low-demand games with targeted discounts and promotions."),
    ("Smarter marketing actions", "Direct campaigns toward games with sell-through risk instead of blanket discounting."),
]
for col, (title, body) in zip(adv_cols, advantages):
    with col:
        render_feature_card(title, body)

render_footer()
