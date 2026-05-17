from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from src.app_utils import configure_page, load_metrics
from src.ui_components import (
    CHART_COLORS,
    render_feature_card,
    render_footer,
    render_kpi_card,
    render_page_header,
    render_section,
    style_plotly,
)


configure_page("SeatSense AI · Executive Overview")

metrics = load_metrics()
business = metrics["business_metrics"]

render_page_header(
    title="SeatSense AI",
    subtitle=(
        "Dynamic pricing intelligence for sports venues. Forecast demand, recommend ticket prices, "
        "quantify revenue opportunity, and route high-impact decisions through human approval."
    ),
    kicker="Executive Overview",
    dark=True,
    badge="AI Demand Forecasting + Revenue Optimization",
)

hero_cols = st.columns([0.72, 0.28])
with hero_cols[0]:
    st.markdown(
        "SeatSense AI is built for revenue managers who need real-time demand intelligence without losing control of fan trust, affordability, or executive accountability."
    )
with hero_cols[1]:
    st.page_link("pages/2_Live_Pricing_Demo.py", label="Open Pricing Workbench")

kpi_cols = st.columns(4)
with kpi_cols[0]:
    render_kpi_card(
        "Revenue uplift opportunity",
        f"{business['estimated_revenue_uplift_pct']:.1%}",
        "+ portfolio upside",
        "Projected lift from guarded price recommendations.",
        "↗",
    )
with kpi_cols[1]:
    render_kpi_card(
        "Sell-through improvement",
        f"{business['unsold_inventory_risk_reduction_pct']:.1%}",
        "+ lower inventory risk",
        "Expected reduction in unsold inventory exposure.",
        "✓",
    )
with kpi_cols[2]:
    render_kpi_card(
        "Leakage reduction",
        f"${business['secondary_market_leakage_reduction']/1_000_000:.1f}M",
        "+ retained value",
        "Estimated revenue recaptured from secondary-market gaps.",
        "$",
    )
with kpi_cols[3]:
    render_kpi_card(
        "Human-reviewed guardrails",
        f"{business['human_approval_rate_proxy']:.1%}",
        "approval workflow",
        "Large or premium changes routed to revenue managers.",
        "H",
    )

render_section(
    "Static pricing leaves revenue and trust exposed",
    "Venue teams need demand intelligence that sees both sides of the risk: underpriced premium games and overpriced low-demand inventory.",
    "Business problem",
)

problem_cols = st.columns(4)
problem_cards = [
    ("Static pricing", "Prices lag changes in opponent strength, timing, weather, player news, and traffic."),
    ("Scalper leakage", "Underpriced games sell out early while secondary markets capture the upside."),
    ("Empty seats", "Overpriced games hurt attendance, concessions, atmosphere, and fan experience."),
    ("Slow decisions", "Revenue managers need clear actions, not another spreadsheet to maintain."),
]
for col, (title, body) in zip(problem_cols, problem_cards):
    with col:
        render_feature_card(title, body)

render_section(
    "A governed AI workflow for ticket pricing",
    "SeatSense connects prediction, recommendation, business impact, explanation, and approval guardrails in one operating view.",
    "Solution",
)

workflow = go.Figure(
    data=[
        go.Scatter(
            x=[1, 2, 3, 4, 5],
            y=[1, 1, 1, 1, 1],
            mode="markers+text+lines",
            marker=dict(size=34, color=CHART_COLORS[:5], line=dict(width=3, color="#FFFFFF")),
            line=dict(color="#CBD5E1", width=4),
            text=[
                "Ingest signals",
                "Predict demand",
                "Recommend price",
                "Explain action",
                "Approve guardrails",
            ],
            textposition="bottom center",
            textfont=dict(size=13, color="#111827"),
            hoverinfo="skip",
        )
    ]
)
workflow.update_xaxes(visible=False, range=[0.6, 5.4])
workflow.update_yaxes(visible=False, range=[0.65, 1.25])
workflow = style_plotly(workflow, height=250)
workflow.update_layout(showlegend=False, title="Product Workflow")
st.plotly_chart(workflow, width="stretch")

feature_cols = st.columns(4)
solution_cards = [
    ("Predict game demand", "Classify demand by game and section with confidence scores."),
    ("Recommend optimal price", "Compare current price, resale gap, urgency, and sell-through risk."),
    ("Estimate impact", "Show revenue uplift, leakage reduction, and inventory risk before action."),
    ("Flag approval risk", "Apply price caps, affordability controls, and human-in-the-loop review."),
]
for col, (title, body) in zip(feature_cols, solution_cards):
    with col:
        render_feature_card(title, body)

render_footer()
