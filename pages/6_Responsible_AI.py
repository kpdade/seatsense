from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st
import json

from src.app_utils import PROJECT_ROOT, configure_page, load_metrics, load_training_data
from src.fairness import build_fairness_audit_table, failure_mode_table
from src.ui_components import (
    CHART_COLORS,
    render_action_card,
    render_feature_card,
    render_footer,
    render_kpi_card,
    render_page_header,
    render_section,
    style_plotly,
)


configure_page("SeatSense AI · Responsible AI")

render_page_header(
    "Responsible AI",
    "Governance dashboard for fan affordability, trust, privacy, human approval, and operational failure modes.",
    kicker="Pricing governance",
)

data = load_training_data()
metrics = load_metrics()
audit_path = PROJECT_ROOT / "outputs" / "fairness_audit.csv"
audit = pd.read_csv(audit_path) if audit_path.exists() else build_fairness_audit_table(data)
fairness_json_path = PROJECT_ROOT / "outputs" / "fairness_audit.json"
fairness_json = {}
if fairness_json_path.exists():
    fairness_json = json.loads(fairness_json_path.read_text(encoding="utf-8"))

cols = st.columns(4)
with cols[0]:
    render_kpi_card("Protected attributes", "0", note="Not used directly")
with cols[1]:
    render_kpi_card("Model demographics", "Excluded", note="Affordability is audit-only")
with cols[2]:
    render_kpi_card("Approval rate", f"{metrics.get('business_metrics', {}).get('percentage_human_approval_required', 0):.1%}", note="Human-in-loop share")
with cols[3]:
    render_kpi_card("Manual override", "Always", note="Revenue manager control")

render_section(
    "Risk register",
    "Dynamic pricing creates trust and access risks. SeatSense makes those risks visible before prices are published.",
    "Responsible AI risks",
)
risk_cols = st.columns(5)
risks = [
    ("Affordability risk", "Value-sensitive fans can be priced out without caps and ticket reserves."),
    ("Geographic bias", "Market proxies can create disparate impact across communities."),
    ("Privacy risk", "Traffic and sentiment signals must remain aggregated and minimized."),
    ("Fan trust risk", "Opaque pricing can feel arbitrary and damage loyalty."),
    ("Overpricing risk", "Noisy signals can push prices too high and create empty seats."),
]
for col, (title, body) in zip(risk_cols, risks):
    with col:
        render_feature_card(title, body)

render_section(
    "Mitigation controls",
    "The system is designed as decision support, not autonomous price publishing.",
    "Control framework",
)
mitigation_cols = st.columns(5)
mitigations = [
    ("Price caps", "Limit large price jumps and reduce fan backlash."),
    ("Affordable reserve", "Preserve value-priced ticket buckets."),
    ("Human approval", "Review premium games and high-impact changes."),
    ("Manual override", "Keep revenue managers accountable for final decisions."),
    ("Monitoring", "Audit by section, segment, attendance, and complaint signals."),
]
for col, (title, body) in zip(mitigation_cols, mitigations):
    with col:
        render_feature_card(title, body)

render_section(
    "Affordability policy",
    "SeatSense does not use protected demographics or income proxies to maximize price. Affordability is monitored only for fairness audit, price caps, approval routing, and affordable ticket reserve simulation.",
    "Model boundary",
)
policy_cols = st.columns(2)
with policy_cols[0]:
    render_action_card(
        "Training exclusion",
        "Race, gender, age, individual income, geography, and affordability segment are not model features.",
        "The demand, sell-through, and scalper-risk models learn from game, inventory, timing, market, and historical behavior signals.",
        "This reduces the risk that dynamic pricing directly monetizes sensitive demographics.",
    )
with policy_cols[1]:
    render_action_card(
        "Guardrail use",
        "Affordability_index appears only after model prediction, inside monitoring and approval controls.",
        "Value-sensitive inventory receives stricter price caps and reserve protection.",
        "Revenue managers can override or reject any recommendation.",
    )

left, right = st.columns([0.55, 0.45])
with left:
    render_section("Fairness audit", "Monitor recommendation patterns by affordability segment.", "Audit")
    st.dataframe(audit, width="stretch", hide_index=True)
with right:
    if "avg_price_change_pct" in audit.columns:
        fig = px.bar(
            audit,
            x="affordability_segment",
            y="avg_price_change_pct",
            color="guardrail_status",
            color_discrete_sequence=CHART_COLORS,
            title="Average Price Change by Segment",
            labels={"affordability_segment": "Segment", "avg_price_change_pct": "Avg change"},
        )
        fig.update_yaxes(tickformat=".1%")
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_plotly(fig, height=360), width="stretch")

with st.expander("Fairness audit JSON"):
    st.json(fairness_json or {"message": "Fairness audit JSON not available. Run python -m src.train_model."})

render_section(
    "API and data boundary",
    "SeatSense can enrich scenarios with optional live data providers while keeping the trained model, pricing engine, and approval guardrails explicit.",
    "Implementation policy",
)
api_cols = st.columns([0.5, 0.5])
with api_cols[0]:
    render_action_card(
        "Live signal enrichment",
        "Ticketmaster, SeatGeek, StubHub, weather, sports stats, social buzz, and website analytics connectors can update demand inputs when configured.",
        "Live signals feed the saved demand model and deterministic pricing engine before the recommendation is calculated.",
        "Every provider is timeout-protected and optional; missing keys fall back to the current scenario values.",
    )
with api_cols[1]:
    render_action_card(
        "Explanation-only OpenAI use",
        "OpenAI is optional and receives only the already-computed scenario, model, pricing, and guardrail outputs.",
        "It creates a revenue-manager briefing; it does not train the model, fetch market data, or decide the price.",
        "If the key is missing or the call fails, SeatSense uses a deterministic fallback explanation.",
    )

render_section(
    "Human-in-the-loop policy",
    "Revenue managers remain in control of recommendations that affect fan access, premium events, or large price moves.",
    "Approval",
)
render_action_card(
    "Approval workflow",
    "Flag recommendations when premium-game status, price movement, affordability risk, or low confidence raises operational risk.",
    "High-impact actions are reviewed before publication; low-risk actions can move faster.",
    "Post-game monitoring compares attendance, sell-through, resale gap, fan sentiment, and fairness metrics.",
)

with st.expander("Failure modes and mitigations"):
    st.dataframe(failure_mode_table(), width="stretch", hide_index=True)

render_footer()
