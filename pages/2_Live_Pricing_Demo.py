from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.app_utils import build_prediction_form, configure_page, demo_scenarios, run_full_recommendation
from src.generative_explainer import openai_explanations_enabled
from src.live_market_integrations import (
    apply_live_signals,
    collect_live_market_signals,
    provider_status_frame,
)
from src.ui_components import (
    CHART_COLORS,
    render_action_card,
    render_explanation_box,
    render_footer,
    render_kpi_card,
    render_page_header,
    render_risk_badge,
    render_section,
    render_status_badge,
    style_plotly,
)


configure_page("SeatSense AI · Pricing Workbench")


def _money_short(value: float) -> str:
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    if abs(value) >= 10_000:
        return f"${value / 1_000:.1f}K"
    if abs(value) >= 1_000:
        return f"${value / 1_000:.0f}K"
    return f"${value:,.0f}"

render_page_header(
    "Pricing Workbench",
    "Evaluate live game scenarios, forecast demand, compare current and recommended pricing, and review the AI-generated executive explanation.",
    kicker="Revenue command center",
    dark=False,
)

scenarios = demo_scenarios()
top_left, top_right = st.columns([0.38, 0.62])
with top_left:
    selected = st.selectbox("Scenario", list(scenarios.keys()), help="Choose a pricing scenario.")
    st.caption("Compare high-demand, low-demand, injury, weather, and premium-event conditions.")

    with st.expander("Adjust scenario inputs", expanded=True):
        input_row = build_prediction_form(defaults=scenarios[selected], key_prefix="live_")
    live_key = f"live_market_signals::{selected}"
    with st.expander("Live API signal mode", expanded=False):
        st.caption(
            "Optional live providers enrich the scenario before the saved demand model and pricing engine run. Missing keys fail gracefully."
        )
        default_query = f"{selected.replace('-', ' ')} basketball tickets"
        market_query = st.text_input("Event / market search query", value=default_query, key="live_market_query")
        city = st.text_input("Market city", value="New York", key="live_city")
        geo_cols = st.columns(2)
        with geo_cols[0]:
            latitude = st.number_input("Venue latitude", value=40.7505, format="%.4f", key="live_latitude")
        with geo_cols[1]:
            longitude = st.number_input("Venue longitude", value=-73.9934, format="%.4f", key="live_longitude")
        sports_team = st.text_input("Sports stats team token", value="NYK", help="Example: NYK, Knicks, Lakers, BOS", key="live_sports_team")

        provider_cols = st.columns(2)
        with provider_cols[0]:
            use_ticketmaster = st.checkbox("Ticketmaster", value=True, key="use_ticketmaster")
            use_seatgeek = st.checkbox("SeatGeek", value=True, key="use_seatgeek")
            use_stubhub = st.checkbox("StubHub", value=True, key="use_stubhub")
            use_weather = st.checkbox("Weather", value=True, key="use_weather")
        with provider_cols[1]:
            use_sports_stats = st.checkbox("Sports stats", value=True, key="use_sports_stats")
            use_social = st.checkbox("Social buzz", value=True, key="use_social")
            use_analytics = st.checkbox("Website analytics", value=True, key="use_analytics")

        action_cols = st.columns(2)
        with action_cols[0]:
            if st.button("Fetch live API signals", key="fetch_live_api_signals"):
                with st.spinner("Calling live data providers..."):
                    st.session_state[live_key] = collect_live_market_signals(
                        input_row,
                        market_query=market_query,
                        city=city,
                        latitude=float(latitude),
                        longitude=float(longitude),
                        sports_team=sports_team,
                        use_ticketmaster=use_ticketmaster,
                        use_seatgeek=use_seatgeek,
                        use_stubhub=use_stubhub,
                        use_weather=use_weather,
                        use_sports_stats=use_sports_stats,
                        use_social=use_social,
                        use_analytics=use_analytics,
                    )
        with action_cols[1]:
            if st.button("Clear live signals", key="clear_live_api_signals"):
                st.session_state.pop(live_key, None)

        live_signals = st.session_state.get(live_key)
        if live_signals:
            status_frame = provider_status_frame(live_signals)
            st.dataframe(status_frame, width="stretch", hide_index=True)
            if live_signals.adjustments:
                st.success("Live API signals are active for this recommendation.")
            else:
                st.info("Providers were checked, but no adjustable live signal was returned.")

live_signals = st.session_state.get(f"live_market_signals::{selected}")
effective_input = apply_live_signals(input_row, live_signals)

with top_right:
    result = run_full_recommendation(effective_input, use_openai_explanation=False)
    prediction = result["prediction"]
    sellthrough = result["sellthrough"]
    scalper = result["scalper"]
    recommendation = result["recommendation"]
    normalized = prediction["input"]

    st.markdown("#### Pricing recommendation")
    badge_cols = st.columns(4)
    with badge_cols[0]:
        render_status_badge(f"{prediction['demand_tier']} demand", prediction["demand_tier"])
    with badge_cols[1]:
        render_status_badge(f"{scalper['scalper_risk_tier']} scalper risk", scalper["scalper_risk_tier"])
    with badge_cols[2]:
        render_risk_badge(recommendation.risk_category)
    with badge_cols[3]:
        render_status_badge(
            "Human Approval Required" if recommendation.human_approval_required else "Within Guardrails",
            "Required" if recommendation.human_approval_required else "Safe",
        )

    price_cols = st.columns(3)
    with price_cols[0]:
        render_kpi_card("Current price", _money_short(normalized["current_ticket_price"]), note="Primary market")
    with price_cols[1]:
        render_kpi_card("Recommended price", _money_short(recommendation.recommended_price), delta=f"{recommendation.price_change_pct:.1%}", note="SeatSense action")
    with price_cols[2]:
        render_kpi_card("Revenue impact", _money_short(recommendation.revenue_uplift_estimate), note="Section estimate")

    risk_cols = st.columns(2)
    with risk_cols[0]:
        render_kpi_card("Sell-through", f"{sellthrough['predicted_sell_through']:.1%}", note="Model estimate")
    with risk_cols[1]:
        render_kpi_card("Leakage reduction", _money_short(recommendation.secondary_market_leakage_estimate), note="Resale upside retained")

    fig = go.Figure()
    fig.add_bar(
        x=["Current", "Secondary Market", "SeatSense"],
        y=[
            normalized["current_ticket_price"],
            normalized["secondary_market_avg_price"],
            recommendation.recommended_price,
        ],
        marker_color=[CHART_COLORS[5], CHART_COLORS[3], CHART_COLORS[1]],
        text=[
            f"${normalized['current_ticket_price']:,.0f}",
            f"${normalized['secondary_market_avg_price']:,.0f}",
            f"${recommendation.recommended_price:,.0f}",
        ],
        textposition="outside",
    )
    fig.update_layout(showlegend=False, title="Current vs Recommended Price")
    st.plotly_chart(style_plotly(fig, height=330), width="stretch")

render_section(
    "Executive explanation",
    "Generated from the model prediction, confidence, recommended price, revenue impact, resale leakage estimate, feature drivers, and approval flag.",
    "AI recommendation",
)
left, right = st.columns([0.64, 0.36])
with left:
    if openai_explanations_enabled():
        explanation_mode = st.radio(
            "Explanation mode",
            ["Fast template", "OpenAI analyst"],
            index=0,
            horizontal=True,
            help="Fast template is instant. OpenAI analyst uses your configured API key and falls back safely if the API call fails.",
        )
        if explanation_mode == "OpenAI analyst":
            with st.spinner("Generating executive briefing with OpenAI. This can take up to 30 seconds on Streamlit Cloud..."):
                result = run_full_recommendation(effective_input, use_openai_explanation=True)
            if result["explanation_source"].startswith("Template fallback"):
                st.warning(
                    f"OpenAI was configured but the API call did not complete: {result['explanation_source'].replace('Template fallback · ', '')}. "
                    "The deterministic fallback remains active for the demo."
                )
    else:
        st.warning(
            "No OpenAI key detected. SeatSense is using the deterministic template explanation. "
            "Add `OPENAI_API_KEY` in `.env` or Streamlit secrets to enable the OpenAI analyst mode."
        )
    render_explanation_box(
        "Revenue manager briefing",
        result["generated_explanation"],
        source_label=result["explanation_source"],
    )
with right:
    render_action_card(
        "Recommended Action",
        recommendation.business_action,
        f"Revenue impact {recommendation.revenue_uplift_estimate:,.0f}; leakage reduction ${recommendation.secondary_market_leakage_estimate:,.0f}.",
        "Human approval required." if recommendation.human_approval_required else "Within configured approval guardrails.",
    )
    st.markdown("#### Demand probability")
    prob_df = pd.DataFrame(
        [{"Demand": k, "Probability": v} for k, v in prediction["probabilities"].items()]
    )
    fig_prob = go.Figure()
    fig_prob.add_bar(
        x=prob_df["Demand"],
        y=prob_df["Probability"],
        marker_color=[
            "#64748B" if demand == "Low" else "#F59E0B" if demand == "Medium" else "#14B8A6"
            for demand in prob_df["Demand"]
        ],
        text=[f"{value:.0%}" for value in prob_df["Probability"]],
        textposition="outside",
    )
    fig_prob.update_layout(showlegend=False, yaxis_tickformat=".0%", title="")
    fig_prob.update_yaxes(range=[0, 1])
    st.plotly_chart(style_plotly(fig_prob, height=260), width="stretch")

with st.expander("Scenario details"):
    scenario_table = pd.DataFrame(
        [
            {"Signal": "Seat section", "Value": str(normalized["seat_section"])},
            {"Signal": "Game day", "Value": str(normalized["day_of_week"])},
            {"Signal": "Opponent strength", "Value": str(normalized["opponent_strength"])},
            {"Signal": "Traffic index", "Value": str(normalized["website_traffic_index"])},
            {"Signal": "Predicted sell-through", "Value": f"{sellthrough['predicted_sell_through']:.1%}"},
            {"Signal": "Scalper risk", "Value": f"{scalper['scalper_risk_tier']} ({scalper['scalper_risk_probability']:.0%})"},
            {"Signal": "Secondary market average", "Value": f"${normalized['secondary_market_avg_price']:,.2f}"},
            {"Signal": "Live API mode", "Value": "Active" if live_signals and live_signals.adjustments else "Scenario only"},
            {"Signal": "Human approval", "Value": "Required" if recommendation.human_approval_required else "Not required"},
        ]
    )
    st.dataframe(scenario_table, width="stretch", hide_index=True)

render_footer()
