"""Production-grade UI components for SeatSense AI."""

from __future__ import annotations

from html import escape
from textwrap import dedent
from typing import Iterable, Mapping

import streamlit as st


PALETTE = {
    "primary_navy": "#0B1020",
    "secondary_navy": "#111827",
    "page_bg": "#F6F8FB",
    "card_bg": "#FFFFFF",
    "accent": "#2563EB",
    "teal": "#14B8A6",
    "warning": "#F59E0B",
    "danger": "#DC2626",
    "success": "#16A34A",
    "text": "#111827",
    "text_secondary": "#475569",
    "muted": "#64748B",
    "border": "#E5E7EB",
}

DEMAND_COLORS = {
    "High": PALETTE["teal"],
    "Medium": PALETTE["warning"],
    "Low": PALETTE["muted"],
}

RISK_COLORS = {
    "High": PALETTE["danger"],
    "Medium": PALETTE["warning"],
    "Low": PALETTE["success"],
    "Required": PALETTE["danger"],
    "Auto": PALETTE["success"],
    "Safe": PALETTE["success"],
    "Yes": PALETTE["danger"],
    "No": PALETTE["success"],
}

CHART_COLORS = [
    PALETTE["accent"],
    PALETTE["teal"],
    PALETTE["warning"],
    PALETTE["danger"],
    "#7C3AED",
    PALETTE["muted"],
]


def load_css() -> None:
    """Load the global SeatSense visual system."""

    st.markdown(
        f"""
        <style>
        :root {{
            --ss-primary-navy: {PALETTE["primary_navy"]};
            --ss-secondary-navy: {PALETTE["secondary_navy"]};
            --ss-page-bg: {PALETTE["page_bg"]};
            --ss-card-bg: {PALETTE["card_bg"]};
            --ss-accent: {PALETTE["accent"]};
            --ss-teal: {PALETTE["teal"]};
            --ss-warning: {PALETTE["warning"]};
            --ss-danger: {PALETTE["danger"]};
            --ss-success: {PALETTE["success"]};
            --ss-text: {PALETTE["text"]};
            --ss-text-secondary: {PALETTE["text_secondary"]};
            --ss-muted: {PALETTE["muted"]};
            --ss-border: {PALETTE["border"]};
            --ss-shadow: 0 18px 46px rgba(15, 23, 42, 0.08);
            --ss-soft-shadow: 0 8px 26px rgba(15, 23, 42, 0.06);
        }}

        html, body, .stApp {{
            background: var(--ss-page-bg) !important;
            color: var(--ss-text) !important;
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }}

        .main .block-container {{
            max-width: 1280px;
            padding: 1.35rem 2.2rem 3rem 2.2rem;
        }}

        h1, h2, h3, h4, h5, h6, p, li, label, span {{
            letter-spacing: 0;
        }}

        h1, h2, h3, h4 {{
            color: var(--ss-text) !important;
        }}

        p, li, label, .stMarkdown, [data-testid="stMarkdownContainer"] {{
            color: var(--ss-text-secondary);
        }}

        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #0B1020 0%, #111827 100%) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }}

        section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {{
            padding: 1.15rem 1rem;
        }}

        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] {{
            color: #CBD5E1 !important;
        }}

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {{
            color: #FFFFFF !important;
        }}

        section[data-testid="stSidebar"] a {{
            color: #E5E7EB !important;
            text-decoration: none;
            border-radius: 10px;
        }}

        section[data-testid="stSidebar"] a:hover {{
            background: rgba(37, 99, 235, 0.22);
            color: #FFFFFF !important;
        }}

        div[data-testid="stPageLink"] a {{
            padding: .55rem .72rem;
            border: 1px solid rgba(255,255,255,.06);
            margin-bottom: .18rem;
        }}

        .stButton > button, .stLinkButton > a {{
            border-radius: 10px !important;
            border: 1px solid var(--ss-accent) !important;
            background: var(--ss-accent) !important;
            color: #FFFFFF !important;
            font-weight: 750 !important;
            min-height: 2.55rem;
            box-shadow: 0 10px 22px rgba(37, 99, 235, .20);
        }}

        .stButton > button:hover, .stLinkButton > a:hover {{
            background: #1D4ED8 !important;
            border-color: #1D4ED8 !important;
            color: #FFFFFF !important;
            transform: translateY(-1px);
        }}

        div[data-testid="stMetric"] {{
            background: var(--ss-card-bg);
            border: 1px solid var(--ss-border);
            border-radius: 14px;
            padding: 1rem 1.05rem;
            box-shadow: var(--ss-soft-shadow);
        }}

        div[data-testid="stMetricLabel"] p {{
            color: var(--ss-muted) !important;
            font-size: .78rem !important;
            font-weight: 800 !important;
            text-transform: uppercase;
        }}

        div[data-testid="stMetricValue"] {{
            color: var(--ss-text) !important;
            font-size: 1.65rem !important;
            font-weight: 850 !important;
        }}

        div[data-testid="stTabs"] [data-baseweb="tab-list"] {{
            gap: .4rem;
            border-bottom: 1px solid var(--ss-border);
        }}

        div[data-testid="stTabs"] [data-baseweb="tab"] {{
            border-radius: 999px 999px 0 0;
            color: var(--ss-text-secondary);
            font-weight: 750;
            padding: .55rem .9rem;
        }}

        div[data-testid="stTabs"] [aria-selected="true"] {{
            color: var(--ss-accent) !important;
            background: #EFF6FF;
        }}

        div[data-testid="stExpander"] {{
            background: var(--ss-card-bg);
            border: 1px solid var(--ss-border) !important;
            border-radius: 14px !important;
            box-shadow: var(--ss-soft-shadow);
        }}

        div[data-testid="stExpander"] summary p {{
            color: var(--ss-text) !important;
            font-weight: 800;
        }}

        div[data-testid="stDataFrame"] {{
            border: 1px solid var(--ss-border);
            border-radius: 14px;
            overflow: hidden;
            box-shadow: var(--ss-soft-shadow);
        }}

        .stSelectbox, .stSlider, .stNumberInput, .stToggle {{
            color: var(--ss-text) !important;
        }}

        div[data-baseweb="select"] > div,
        div[data-testid="stNumberInput"] input {{
            background: #FFFFFF !important;
            border-color: var(--ss-border) !important;
            color: var(--ss-text) !important;
            border-radius: 10px !important;
        }}

        .ss-page-header {{
            background: #FFFFFF;
            border: 1px solid var(--ss-border);
            border-radius: 18px;
            padding: 1.45rem 1.55rem;
            box-shadow: var(--ss-shadow);
            margin-bottom: 1.1rem;
        }}

        .ss-kicker {{
            color: var(--ss-accent);
            text-transform: uppercase;
            font-size: .74rem;
            font-weight: 850;
            letter-spacing: .08rem;
            margin-bottom: .35rem;
        }}

        .ss-page-header h1 {{
            font-size: clamp(2rem, 4vw, 3.35rem);
            line-height: 1.04;
            margin: 0;
        }}

        .ss-page-header p {{
            color: var(--ss-text-secondary);
            max-width: 780px;
            font-size: 1.02rem;
            line-height: 1.6;
            margin: .72rem 0 0 0;
        }}

        .ss-hero-dark {{
            background:
                radial-gradient(circle at 88% 18%, rgba(20, 184, 166, .28), transparent 18rem),
                linear-gradient(135deg, #0B1020 0%, #111827 54%, #1E3A8A 100%);
            border: 1px solid rgba(255,255,255,.12);
            border-radius: 22px;
            padding: 2.35rem 2.45rem;
            margin-bottom: 1.25rem;
            box-shadow: var(--ss-shadow);
            overflow: hidden;
        }}

        .ss-hero-dark h1, .ss-hero-dark h2, .ss-hero-dark h3 {{
            color: #FFFFFF !important;
        }}

        .ss-hero-dark p {{
            color: #D1D5DB !important;
            max-width: 760px;
            font-size: 1.05rem;
            line-height: 1.62;
        }}

        .ss-hero-badge {{
            display: inline-flex;
            align-items: center;
            gap: .35rem;
            padding: .36rem .68rem;
            border: 1px solid rgba(153, 246, 228, .38);
            border-radius: 999px;
            background: rgba(20, 184, 166, .12);
            color: #CCFBF1;
            font-size: .8rem;
            font-weight: 850;
            margin-bottom: .75rem;
        }}

        .ss-grid-card, .ss-kpi-card, .ss-feature-card, .ss-action-card, .ss-explanation-box {{
            background: var(--ss-card-bg);
            border: 1px solid var(--ss-border);
            border-radius: 16px;
            box-shadow: var(--ss-soft-shadow);
        }}

        .ss-kpi-card {{
            padding: 1rem 1.05rem;
            min-height: 132px;
            min-width: 0;
            overflow: hidden;
        }}

        .ss-kpi-topline {{
            display: flex;
            justify-content: space-between;
            gap: .6rem;
            align-items: flex-start;
        }}

        .ss-kpi-label {{
            color: var(--ss-muted);
            text-transform: uppercase;
            letter-spacing: .06rem;
            font-weight: 850;
            font-size: .74rem;
        }}

        .ss-kpi-icon {{
            width: 2rem;
            height: 2rem;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #EFF6FF;
            color: var(--ss-accent);
            font-weight: 850;
        }}

        .ss-kpi-value {{
            color: var(--ss-text);
            font-size: clamp(1.18rem, 1.6vw, 1.82rem);
            font-weight: 900;
            line-height: 1.12;
            margin-top: .45rem;
            max-width: 100%;
            white-space: normal;
            overflow-wrap: break-word;
            word-break: normal;
        }}

        .ss-kpi-value-long {{
            font-size: clamp(.92rem, 1.05vw, 1.18rem);
            overflow-wrap: anywhere;
            word-break: break-word;
        }}

        .ss-kpi-delta {{
            color: var(--ss-success);
            font-size: .84rem;
            font-weight: 850;
            margin-top: .2rem;
        }}

        .ss-kpi-note {{
            color: var(--ss-muted);
            font-size: .86rem;
            line-height: 1.35;
            margin-top: .36rem;
            overflow-wrap: break-word;
            word-break: normal;
            white-space: normal;
        }}

        .ss-kpi-note-long {{
            overflow-wrap: anywhere;
            word-break: break-word;
        }}

        .ss-feature-card {{
            padding: 1.05rem 1.08rem;
            min-height: 152px;
        }}

        .ss-feature-card h3 {{
            font-size: 1.02rem;
            margin: 0 0 .4rem 0;
        }}

        .ss-feature-card p {{
            color: var(--ss-text-secondary) !important;
            line-height: 1.5;
            margin: 0;
            font-size: .94rem;
        }}

        .ss-section {{
            margin: 1.25rem 0 .7rem 0;
        }}

        .ss-section h2 {{
            font-size: 1.45rem;
            margin: .15rem 0 .25rem 0;
        }}

        .ss-section p {{
            margin: 0;
            color: var(--ss-text-secondary) !important;
            max-width: 850px;
        }}

        .ss-badge {{
            display: inline-flex;
            align-items: center;
            padding: .31rem .68rem;
            border-radius: 999px;
            font-size: .76rem;
            font-weight: 900;
            color: #FFFFFF;
            border: 1px solid rgba(255,255,255,.18);
            white-space: nowrap;
        }}

        .ss-explanation-box {{
            padding: 1rem 1.12rem;
        }}

        .ss-explanation-box h3 {{
            margin-top: 0;
        }}

        .ss-action-card {{
            padding: 1rem 1.08rem;
            border-left: 5px solid var(--ss-accent);
        }}

        .ss-action-card h3 {{
            margin: 0 0 .35rem 0;
            font-size: 1.06rem;
        }}

        .ss-action-card p {{
            color: var(--ss-text-secondary) !important;
            margin: .2rem 0;
            line-height: 1.45;
        }}

        .ss-sidebar-brand {{
            padding: .5rem .2rem 1rem .2rem;
            border-bottom: 1px solid rgba(255,255,255,.12);
            margin-bottom: .8rem;
        }}

        .ss-sidebar-brand-title {{
            color: #FFFFFF;
            font-size: 1.28rem;
            line-height: 1.1;
            font-weight: 900;
            margin: .3rem 0 .15rem 0;
        }}

        .ss-sidebar-brand-subtitle {{
            color: #93C5FD !important;
            font-size: .82rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: .06rem;
            margin: 0 0 .45rem 0;
        }}

        .ss-sidebar-copy {{
            color: #CBD5E1 !important;
            font-size: .87rem;
            line-height: 1.45;
            margin: 0;
        }}

        .ss-demo-pill {{
            display: inline-block;
            background: rgba(22, 163, 74, .18);
            border: 1px solid rgba(34, 197, 94, .35);
            color: #BBF7D0;
            padding: .25rem .55rem;
            border-radius: 999px;
            font-size: .72rem;
            font-weight: 900;
            margin-top: .7rem;
        }}

        .ss-sidebar-section {{
            color: #94A3B8 !important;
            text-transform: uppercase;
            letter-spacing: .08rem;
            font-size: .68rem;
            font-weight: 900;
            margin: 1rem .25rem .45rem .25rem;
        }}

        .ss-sidebar-footer {{
            border-top: 1px solid rgba(255,255,255,.12);
            color: #94A3B8 !important;
            font-size: .76rem;
            line-height: 1.4;
            padding-top: .9rem;
            margin-top: 1.2rem;
        }}

        .ss-footer {{
            color: var(--ss-muted);
            border-top: 1px solid var(--ss-border);
            margin-top: 1.8rem;
            padding-top: 1rem;
            font-size: .84rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _html(markup: str) -> None:
    st.html(dedent(markup).strip())


def _badge_html(label: str, color: str) -> str:
    return f'<span class="ss-badge" style="background:{color};">{escape(str(label))}</span>'


def render_sidebar() -> None:
    with st.sidebar:
        st.html(
            """
            <div class="ss-sidebar-brand">
                <div class="ss-sidebar-brand-subtitle">Dynamic Pricing Intelligence</div>
                <div class="ss-sidebar-brand-title">SeatSense AI</div>
                <p class="ss-sidebar-copy">Sports revenue optimization for ticketing teams, executives, and venue operators.</p>
                <span class="ss-demo-pill">Revenue Ops</span>
            </div>
            """,
        )
        st.html('<div class="ss-sidebar-section">Command Center</div>')
        st.page_link("app.py", label="Executive Overview")
        st.page_link("pages/2_Live_Pricing_Demo.py", label="Pricing Workbench")
        st.page_link("pages/3_Demand_Intelligence.py", label="Demand Intelligence")
        st.page_link("pages/4_Revenue_Impact.py", label="Revenue Impact")
        st.html('<div class="ss-sidebar-section">Governance</div>')
        st.page_link("pages/5_Model_Performance.py", label="Model Performance")
        st.page_link("pages/6_Responsible_AI.py", label="Responsible AI")
        st.html(
            """
            <div class="ss-sidebar-footer">
                Enterprise pricing intelligence for sports venues
            </div>
            """,
        )


def render_page_header(
    title: str,
    subtitle: str,
    kicker: str = "SeatSense AI",
    dark: bool = False,
    badge: str | None = None,
) -> None:
    klass = "ss-hero-dark" if dark else "ss-page-header"
    badge_html = f'<div class="ss-hero-badge">{escape(badge)}</div>' if badge else ""
    _html(
        f"""
        <div class="{klass}">
            {badge_html}
            <div class="ss-kicker">{escape(kicker)}</div>
            <h1>{escape(title)}</h1>
            <p>{escape(subtitle)}</p>
        </div>
        """,
    )


def render_kpi_card(
    label: str,
    value: str,
    delta: str = "",
    note: str = "",
    icon: str = "",
) -> None:
    icon_html = f'<div class="ss-kpi-icon">{escape(icon)}</div>' if icon else ""
    delta_html = f'<div class="ss-kpi-delta">{escape(delta)}</div>' if delta else ""
    value_text = str(value)
    value_class = (
        "ss-kpi-value ss-kpi-value-long"
        if len(value_text) > 16 or any(token in value_text for token in ("/", "_"))
        else "ss-kpi-value"
    )
    note_text = str(note)
    note_class = (
        "ss-kpi-note ss-kpi-note-long"
        if len(note_text) > 36 or any(token in note_text for token in ("/", "_"))
        else "ss-kpi-note"
    )
    _html(
        f"""
        <div class="ss-kpi-card">
            <div class="ss-kpi-topline">
                <div class="ss-kpi-label">{escape(label)}</div>
                {icon_html}
            </div>
            <div class="{value_class}">{escape(value_text)}</div>
            {delta_html}
            <div class="{note_class}">{escape(note_text)}</div>
        </div>
        """,
    )


def render_feature_card(title: str, body: str, icon: str = "") -> None:
    heading = f"{escape(icon)} {escape(title)}" if icon else escape(title)
    _html(
        f"""
        <div class="ss-feature-card">
            <h3>{heading}</h3>
            <p>{escape(body)}</p>
        </div>
        """,
    )


def render_status_badge(label: str, status: str | None = None) -> None:
    status = status or label
    color = DEMAND_COLORS.get(status) or RISK_COLORS.get(status) or PALETTE["accent"]
    st.html(_badge_html(label, color))


def render_risk_badge(risk: str) -> None:
    render_status_badge(risk, risk)


def render_section(title: str, body: str = "", kicker: str = "Overview") -> None:
    _html(
        f"""
        <div class="ss-section">
            <div class="ss-kicker">{escape(kicker)}</div>
            <h2>{escape(title)}</h2>
            <p>{escape(body)}</p>
        </div>
        """,
    )


def render_explanation_box(
    title: str,
    explanation: str,
    source_label: str = "Executive explanation",
) -> None:
    with st.container(border=True):
        st.markdown(f"### {title}")
        st.caption(source_label)
        st.markdown(explanation)


def render_action_card(
    title: str,
    reason: str,
    impact: str,
    risk: str,
) -> None:
    _html(
        f"""
        <div class="ss-action-card">
            <h3>{escape(title)}</h3>
            <p><strong>Reason:</strong> {escape(reason)}</p>
            <p><strong>Impact:</strong> {escape(impact)}</p>
            <p><strong>Risk:</strong> {escape(risk)}</p>
        </div>
        """,
    )


def render_footer() -> None:
    st.html(
        """
        <div class="ss-footer">
            SeatSense AI · Dynamic pricing intelligence with human-reviewed guardrails.
        </div>
        """,
    )


def render_metric_grid(metrics: Iterable[Mapping[str, str]], columns: int = 4) -> None:
    cols = st.columns(columns)
    for index, metric in enumerate(metrics):
        with cols[index % columns]:
            render_kpi_card(
                label=str(metric.get("label", "")),
                value=str(metric.get("value", "")),
                delta=str(metric.get("delta", "")),
                note=str(metric.get("note", "")),
                icon=str(metric.get("icon", "")),
            )


def style_plotly(fig, height: int = 380):
    fig.update_layout(
        height=height,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color=PALETTE["text"], size=12),
        title_font=dict(color=PALETTE["text"], size=17),
        margin=dict(l=20, r=20, t=58, b=25),
        legend=dict(font=dict(color=PALETTE["text_secondary"])),
    )
    fig.update_xaxes(
        showgrid=False,
        linecolor=PALETTE["border"],
        tickfont=dict(color=PALETTE["text_secondary"]),
        title_font=dict(color=PALETTE["text_secondary"]),
    )
    fig.update_yaxes(
        gridcolor="#EEF2F7",
        zerolinecolor=PALETTE["border"],
        tickfont=dict(color=PALETTE["text_secondary"]),
        title_font=dict(color=PALETTE["text_secondary"]),
    )
    return fig


# Backward-compatible aliases used by existing imports during transition.
apply_global_styles = load_css
render_sidebar_branding = render_sidebar
render_hero_section = render_page_header
render_section_header = render_section
render_callout = lambda title, body: render_action_card(title, body, "Use as executive decision support.", "Monitor guardrails.")
render_demo_footer = render_footer
render_explanation_panel = lambda explanation, source_label="Executive explanation": render_explanation_box(
    "AI Pricing Explanation", explanation, source_label
)
