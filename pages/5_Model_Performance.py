from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data.generate_ticket_data import generate_all
from src.app_utils import (
    PROJECT_ROOT,
    cached_model,
    cached_models,
    clear_cached_data_loaders,
    configure_page,
    load_metrics,
    load_training_data,
)
from src.data_adapter import (
    CONCEPTS,
    NO_COLUMN,
    adapt_uploaded_training_data,
    agency_signal_preview,
    detect_agency_feed_profile,
    infer_column_mapping,
    sensitive_source_columns,
)
from src.preprocess import CLASS_ORDER, TARGET_COLUMN
from src.train_model import DATA_PATH, REQUIRED_TRAINING_COLUMNS, train, validate_retraining_frame
from src.ui_components import (
    CHART_COLORS,
    render_action_card,
    render_footer,
    render_kpi_card,
    render_page_header,
    render_section,
    style_plotly,
)


configure_page("SeatSense AI · Model Performance")

render_page_header(
    "Model Performance",
    "Executive-readable validation for the demand model, baseline comparison, and business impact of prediction errors.",
    kicker="AI governance",
)

metrics = load_metrics()
data = load_training_data()
final = metrics["final_model"]["test"]
dataset_meta = metrics.get("dataset", {})
active_csv_path = dataset_meta.get("active_csv", "data/model_training_data.csv")
active_csv_name = Path(active_csv_path).name


def _money_short(value: float) -> str:
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    if abs(value) >= 10_000:
        return f"${value / 1_000:.1f}K"
    return f"${value:,.0f}"

cols = st.columns(5)
with cols[0]:
    render_kpi_card("Accuracy", f"{final['accuracy']:.1%}", note="Overall test quality")
with cols[1]:
    render_kpi_card("Precision", f"{final['precision_macro']:.1%}", note="Controls false demand alerts")
with cols[2]:
    render_kpi_card("Recall", f"{final['recall_macro']:.1%}", note="Controls missed demand")
with cols[3]:
    render_kpi_card("F1 Score", f"{final['f1_macro']:.1%}", note="Balanced tier performance")
with cols[4]:
    render_kpi_card("ROC-AUC", f"{final['roc_auc_ovr_macro']:.1%}" if final.get("roc_auc_ovr_macro") else "N/A", note="Probability ranking")

render_section(
    "Training dataset",
    "The model is trained on the active SeatSense game-section CSV. The table shown later is a preview, not the full training set.",
    "Data source",
)
data_cols = st.columns(4)
with data_cols[0]:
    render_kpi_card("Training rows", f"{dataset_meta.get('rows', len(data)):,}", note="Game-section-window observations")
with data_cols[1]:
    render_kpi_card("Games", f"{dataset_meta.get('games', data['game_id'].nunique()):,}", note="Unique simulated games")
with data_cols[2]:
    render_kpi_card("Seasons", f"{dataset_meta.get('seasons', data['season'].nunique()):,}", note="Season coverage")
with data_cols[3]:
    render_kpi_card("Active CSV", active_csv_name, note="Current training file")

quality_cols = st.columns(4)
with quality_cols[0]:
    render_kpi_card("Seat sections", f"{dataset_meta.get('seat_sections_count', data['seat_section'].nunique()):,}", note="Inventory granularity")
with quality_cols[1]:
    render_kpi_card("Pricing windows", f"{dataset_meta.get('pricing_windows_count', data['pricing_window'].nunique() if 'pricing_window' in data else 1):,}", note="Time-to-event snapshots")
with quality_cols[2]:
    render_kpi_card("Observation unit", "Game + section + window", note="No random row leakage")
with quality_cols[3]:
    render_kpi_card("Data status", "Semi-synthetic", note="Designed for private-data prototype")

st.download_button(
    "Download active training CSV",
    data=data.to_csv(index=False).encode("utf-8"),
    file_name="SeatSense_active_training_dataset.csv",
    mime="text/csv",
)
if st.button("Refresh metrics from disk", help="Use this if Streamlit is showing stale row counts after retraining."):
    clear_cached_data_loaders()
    cached_model.clear()
    cached_models.clear()
    st.rerun()

with st.expander("Import client CSV and retrain model", expanded=False):
    st.write(
        "Upload a venue export and map its columns to SeatSense concepts. The app can fill optional "
        "features with defensible defaults, but supervised retraining still needs a historical outcome "
        "such as final sell-through, demand tier, or sold tickets with capacity."
    )
    uploaded_csv = st.file_uploader("Upload client training CSV", type=["csv"], key="model_training_upload")

    if uploaded_csv is not None:
        try:
            if st.session_state.get("uploaded_training_name") != uploaded_csv.name:
                st.session_state["uploaded_training_name"] = uploaded_csv.name
                st.session_state.pop("adapted_training_preview", None)
                st.session_state.pop("adapted_training_warnings", None)
            uploaded_frame = pd.read_csv(uploaded_csv)
            auto_mapping = infer_column_mapping(list(uploaded_frame.columns))
            feed_profile = detect_agency_feed_profile(uploaded_frame)
            sensitive_columns = sensitive_source_columns(list(uploaded_frame.columns))
            upload_cols = st.columns(4)
            with upload_cols[0]:
                render_kpi_card("Uploaded rows", f"{len(uploaded_frame):,}", note=uploaded_csv.name)
            with upload_cols[1]:
                render_kpi_card("Detected columns", f"{len(uploaded_frame.columns):,}", note="Client export")
            with upload_cols[2]:
                render_kpi_card("Auto-mapped", f"{sum(v != NO_COLUMN for v in auto_mapping.values()):,}", note="SeatSense concepts")
            with upload_cols[3]:
                feed_label = "Agency feed" if feed_profile["is_agency_feed"] else "Client export"
                render_kpi_card("Detected feed", feed_label, note=feed_profile["recommended_use"])

            boundary_cols = st.columns(3)
            with boundary_cols[0]:
                projected_split = "time-based" if auto_mapping.get("season") != NO_COLUMN or auto_mapping.get("game_date") != NO_COLUMN else "group-aware"
                render_kpi_card("Split strategy", "Auto", note=projected_split)
            with boundary_cols[1]:
                trainability = "Outcome mapped" if feed_profile["likely_trainable"] else "Needs outcomes"
                render_kpi_card("Training readiness", trainability, note="Sales outcome required")
            with boundary_cols[2]:
                render_kpi_card("Audit-only fields", f"{len(sensitive_columns):,}", note="Excluded from model features")

            if sensitive_columns:
                st.info(
                    "CRM/demographic-style columns were detected and will not be allowed into model-training concepts: "
                    + ", ".join(sensitive_columns[:10])
                    + ("..." if len(sensitive_columns) > 10 else "")
                )
            if feed_profile["is_agency_feed"]:
                signal_preview = agency_signal_preview(uploaded_frame, auto_mapping).head(25)
                st.markdown("##### Agency market-signal preview")
                st.dataframe(signal_preview, width="stretch", hide_index=True)

            st.markdown("##### Column mapping")
            options = [NO_COLUMN] + list(uploaded_frame.columns)
            concept_tabs = st.tabs(["Required", "Outcomes", "Agency feed", "Market signals", "Context"])
            concept_groups = {
                "Required": ["seat_section", "current_ticket_price", "section_capacity", "tickets_sold", "tickets_sold_pct"],
                "Outcomes": ["final_sell_through_rate", "demand_tier", "game_id", "game_date", "season"],
                "Agency feed": ["secondary_market_avg_price", "secondary_market_listing_count", "base_ticket_price"],
                "Market signals": ["website_traffic_index", "social_sentiment_score"],
                "Context": ["opponent", "opponent_strength", "day_of_week", "days_before_game"],
            }
            mapping = auto_mapping.copy()
            for tab, (_, concepts) in zip(concept_tabs, concept_groups.items()):
                with tab:
                    left, right = st.columns(2)
                    for idx, concept in enumerate(concepts):
                        spec = CONCEPTS[concept]
                        default_value = auto_mapping.get(concept, NO_COLUMN)
                        default_index = options.index(default_value) if default_value in options else 0
                        with left if idx % 2 == 0 else right:
                            mapping[concept] = st.selectbox(
                                spec["label"],
                                options,
                                index=default_index,
                                key=f"mapping_{concept}",
                                help="Required" if spec.get("required") else "Optional; SeatSense can derive or default this field.",
                            )

            if st.button("Preview adapted training data", type="secondary"):
                try:
                    adapted = adapt_uploaded_training_data(uploaded_frame, mapping)
                    validate_retraining_frame(adapted.frame)
                    st.session_state["adapted_training_preview"] = adapted.frame
                    st.session_state["adapted_training_warnings"] = adapted.warnings
                    st.session_state["adapted_training_validation"] = adapted.validation_report
                    st.session_state["adapted_feed_profile"] = adapted.feed_profile
                    st.session_state["adapted_sensitive_columns"] = adapted.excluded_sensitive_columns
                    st.success("Client CSV can be adapted into the SeatSense training schema.")
                except Exception as exc:
                    st.session_state.pop("adapted_training_preview", None)
                    st.error(f"Could not adapt CSV for supervised retraining: {exc}")
                    if feed_profile["is_agency_feed"]:
                        st.warning(
                            "This file can still be useful as an agency market-signal feed. To retrain the model, "
                            "join it with historical venue outcomes such as final sell-through, demand tier, "
                            "tickets sold by section, or realized sales."
                        )

            adapted_preview = st.session_state.get("adapted_training_preview")
            if adapted_preview is not None:
                preview_warnings = st.session_state.get("adapted_training_warnings", [])
                if preview_warnings:
                    for warning in preview_warnings:
                        st.warning(warning)
                validation_report = st.session_state.get("adapted_training_validation")
                preview_cols = st.columns(4)
                with preview_cols[0]:
                    render_kpi_card("Adapted rows", f"{len(adapted_preview):,}", note="Ready for training")
                with preview_cols[1]:
                    render_kpi_card("Games", f"{adapted_preview['game_id'].nunique():,}", note="Grouped validation")
                with preview_cols[2]:
                    render_kpi_card("Seasons", f"{adapted_preview['season'].nunique():,}", note="Time split if available")
                with preview_cols[3]:
                    validation_note = f"{validation_report.rows_checked:,} rows checked" if validation_report else "Schema gate passed"
                    render_kpi_card("Pydantic schema", "Passed", note=validation_note)
                tier_cols = st.columns(2)
                with tier_cols[0]:
                    render_kpi_card("Demand tiers", f"{adapted_preview[TARGET_COLUMN].nunique():,}", note=str(adapted_preview[TARGET_COLUMN].value_counts().to_dict())[:54])
                with tier_cols[1]:
                    sensitive_count = len(st.session_state.get("adapted_sensitive_columns") or [])
                    render_kpi_card("Audit-only excluded", f"{sensitive_count:,}", note="CRM/demographic fields")
                st.dataframe(adapted_preview.head(25), width="stretch", hide_index=True)

                if st.button("Replace active dataset and retrain", type="primary"):
                    backups_dir = PROJECT_ROOT / "data" / "backups"
                    backups_dir.mkdir(parents=True, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    if DATA_PATH.exists():
                        shutil.copy2(DATA_PATH, backups_dir / f"model_training_data_backup_{timestamp}.csv")
                    adapted_preview.to_csv(DATA_PATH, index=False)
                    with st.spinner("Training model portfolio on adapted client CSV..."):
                        train(data_path=DATA_PATH, source_name=f"Adapted client CSV: {uploaded_csv.name}")
                    clear_cached_data_loaders()
                    cached_model.clear()
                    cached_models.clear()
                    st.success("Model retrained on adapted client data.")
                    st.rerun()
        except Exception as exc:
            st.error(f"Could not read uploaded CSV: {exc}")

    with st.expander("What happens with a 100,000-row client dataset?"):
        st.write(
            "If the upload has seasons or dates, SeatSense sorts chronologically: roughly the oldest 60% of seasons train, "
            "the next 20% validate, and the newest 20% test. If dates are missing but game IDs exist, it uses a group-aware "
            "split by game ID so the same game never appears in both train and test. Random row splitting is only a last-resort "
            "fallback because it can leak game-level patterns."
        )

    if st.button("Restore 20,500-row demo dataset"):
        with st.spinner("Regenerating demo dataset and retraining model portfolio..."):
            generate_all(PROJECT_ROOT / "data")
            train(data_path=DATA_PATH, source_name="Regenerated 20,500-row demo data")
        clear_cached_data_loaders()
        cached_model.clear()
        cached_models.clear()
        st.success("Demo dataset restored and model retrained.")
        st.rerun()

render_section(
    "Dataset drill-down",
    "Filter the active training portfolio and inspect demand mix, resale gaps, sell-through, and pricing pressure.",
    "Dashboard analytics",
)
filter_cols = st.columns(3)
with filter_cols[0]:
    selected_sections = st.multiselect(
        "Seat sections",
        sorted(data["seat_section"].dropna().unique().tolist()),
        default=sorted(data["seat_section"].dropna().unique().tolist()),
    )
with filter_cols[1]:
    selected_tiers = st.multiselect(
        "Demand tiers",
        CLASS_ORDER,
        default=CLASS_ORDER,
    )
with filter_cols[2]:
    window_options = sorted(data["pricing_window"].dropna().unique().tolist()) if "pricing_window" in data else []
    selected_windows = st.multiselect(
        "Pricing windows",
        window_options,
        default=window_options,
    )

filtered = data.copy()
if selected_sections:
    filtered = filtered[filtered["seat_section"].isin(selected_sections)]
if selected_tiers:
    filtered = filtered[filtered[TARGET_COLUMN].isin(selected_tiers)]
if selected_windows and "pricing_window" in filtered.columns:
    filtered = filtered[filtered["pricing_window"].isin(selected_windows)]

if filtered.empty:
    st.warning("No rows match the current drill-down filters.")
else:
    filtered = filtered.copy()
    if "resale_price_gap_pct" not in filtered.columns:
        filtered["resale_price_gap_pct"] = (
            filtered["secondary_market_avg_price"] / filtered["current_ticket_price"].clip(lower=1) - 1
        )

    drill_cols = st.columns(4)
    with drill_cols[0]:
        render_kpi_card("Filtered rows", f"{len(filtered):,}", note="Current drill-down")
    with drill_cols[1]:
        render_kpi_card("Avg primary price", _money_short(filtered["current_ticket_price"].mean()), note="Filtered inventory")
    with drill_cols[2]:
        render_kpi_card("Avg resale gap", f"{filtered['resale_price_gap_pct'].mean():.1%}", note="Secondary vs primary")
    with drill_cols[3]:
        render_kpi_card("Avg sell-through", f"{filtered['tickets_sold_pct'].mean():.1%}", note="Current window")

    chart_tabs = st.tabs(["Demand mix", "Price gap scatter", "Signal bubble", "Section drill-down"])
    with chart_tabs[0]:
        donut_cols = st.columns([0.42, 0.58])
        with donut_cols[0]:
            fig_donut = px.pie(
                filtered,
                names=TARGET_COLUMN,
                hole=0.58,
                color=TARGET_COLUMN,
                color_discrete_map={"Low": "#64748B", "Medium": "#F59E0B", "High": "#14B8A6"},
                title="Demand Tier Mix",
            )
            st.plotly_chart(style_plotly(fig_donut, height=390), width="stretch")
        with donut_cols[1]:
            tier_window = (
                filtered.groupby(["pricing_window", TARGET_COLUMN])
                .size()
                .reset_index(name="Rows")
                if "pricing_window" in filtered.columns
                else filtered.groupby([TARGET_COLUMN]).size().reset_index(name="Rows")
            )
            fig_window = px.bar(
                tier_window,
                x="pricing_window" if "pricing_window" in tier_window.columns else TARGET_COLUMN,
                y="Rows",
                color=TARGET_COLUMN,
                color_discrete_map={"Low": "#64748B", "Medium": "#F59E0B", "High": "#14B8A6"},
                title="Demand Mix by Pricing Window",
            )
            st.plotly_chart(style_plotly(fig_window, height=390), width="stretch")
    with chart_tabs[1]:
        scatter_sample = filtered.sample(min(len(filtered), 3000), random_state=751)
        fig_scatter = px.scatter(
            scatter_sample,
            x="current_ticket_price",
            y="secondary_market_avg_price",
            color=TARGET_COLUMN,
            size="section_capacity",
            hover_data=["seat_section", "pricing_window", "tickets_sold_pct", "resale_price_gap_pct"],
            color_discrete_map={"Low": "#64748B", "Medium": "#F59E0B", "High": "#14B8A6"},
            title="Primary Price vs Secondary Market Price",
            labels={
                "current_ticket_price": "Primary price",
                "secondary_market_avg_price": "Secondary-market avg",
                "section_capacity": "Section capacity",
            },
        )
        fig_scatter.update_xaxes(tickprefix="$")
        fig_scatter.update_yaxes(tickprefix="$")
        st.plotly_chart(style_plotly(fig_scatter, height=430), width="stretch")
    with chart_tabs[2]:
        bubble_sample = filtered.sample(min(len(filtered), 3000), random_state=752)
        fig_bubble = px.scatter(
            bubble_sample,
            x="website_traffic_index",
            y="current_sell_through_rate" if "current_sell_through_rate" in bubble_sample.columns else "tickets_sold_pct",
            size="secondary_market_listing_count" if "secondary_market_listing_count" in bubble_sample.columns else "section_capacity",
            color="scalper_risk_tier" if "scalper_risk_tier" in bubble_sample.columns else TARGET_COLUMN,
            hover_data=["seat_section", TARGET_COLUMN, "current_ticket_price", "secondary_market_avg_price"],
            color_discrete_sequence=CHART_COLORS,
            title="Demand Signals and Resale Pressure",
            labels={
                "website_traffic_index": "Website traffic index",
                "current_sell_through_rate": "Current sell-through",
                "secondary_market_listing_count": "Secondary listings",
            },
        )
        fig_bubble.update_yaxes(tickformat=".0%")
        st.plotly_chart(style_plotly(fig_bubble, height=430), width="stretch")
    with chart_tabs[3]:
        section_summary = (
            filtered.groupby("seat_section")
            .agg(
                rows=("game_id", "count"),
                avg_price=("current_ticket_price", "mean"),
                avg_resale_gap=("resale_price_gap_pct", "mean"),
                avg_sell_through=("tickets_sold_pct", "mean"),
                high_demand_share=(TARGET_COLUMN, lambda s: (s == "High").mean()),
            )
            .reset_index()
            .sort_values("avg_resale_gap", ascending=False)
        )
        fig_section = px.bar(
            section_summary,
            x="seat_section",
            y="avg_resale_gap",
            color="high_demand_share",
            color_continuous_scale=["#64748B", "#2563EB", "#14B8A6"],
            title="Resale Gap and High-Demand Share by Section",
            labels={"avg_resale_gap": "Avg resale gap", "seat_section": "Seat section", "high_demand_share": "High-demand share"},
        )
        fig_section.update_yaxes(tickformat=".0%")
        st.plotly_chart(style_plotly(fig_section, height=390), width="stretch")
        st.dataframe(section_summary.round(3), width="stretch", hide_index=True)

render_section(
    "Validation and leakage audit",
    "The upgraded training flow uses season-based validation and explicitly checks that target/post-event columns are not model inputs.",
    "Trust controls",
)
split = metrics.get("split", {})
leakage = metrics.get("leakage_audit", {})
split_cols = st.columns(4)
with split_cols[0]:
    render_kpi_card("Train rows", f"{split.get('train_rows', 0):,}", note=f"Seasons {split.get('train_seasons', [])}")
with split_cols[1]:
    render_kpi_card("Validation rows", f"{split.get('validation_rows', 0):,}", note=f"Seasons {split.get('validation_seasons', [])}")
with split_cols[2]:
    render_kpi_card("Test rows", f"{split.get('test_rows', 0):,}", note=f"Seasons {split.get('test_seasons', [])}")
with split_cols[3]:
    render_kpi_card("Leakage audit", "Passed" if leakage.get("passed") else "Review", note=leakage.get("message", "Audit unavailable"))

with st.expander("Leakage audit details"):
    st.json(leakage)

split_distribution = []
for split_name, distribution in split.get("demand_tier_distribution", {}).items():
    for tier, share in distribution.items():
        split_distribution.append({"Split": split_name.title(), "Demand tier": tier, "Share": share})
if split_distribution:
    split_df = pd.DataFrame(split_distribution)
    fig_split = px.bar(
        split_df,
        x="Split",
        y="Share",
        color="Demand tier",
        color_discrete_map={"Low": "#64748B", "Medium": "#F59E0B", "High": "#14B8A6"},
        title="Demand Tier Distribution by Validation Split",
    )
    fig_split.update_yaxes(tickformat=".0%")
    st.plotly_chart(style_plotly(fig_split, height=340), width="stretch")

render_section(
    "Baseline vs final model",
    "The final random forest captures nonlinear interactions between demand signals, pricing context, and seat section.",
    "Evaluation",
)
comparison = pd.DataFrame(
    [
        {
            "Model": metrics["baseline_model"].get("name", "Baseline"),
            "Accuracy": metrics["baseline_model"]["test"]["accuracy"],
            "Precision": metrics["baseline_model"]["test"]["precision_macro"],
            "Recall": metrics["baseline_model"]["test"]["recall_macro"],
            "F1": metrics["baseline_model"]["test"]["f1_macro"],
            "ROC-AUC": metrics["baseline_model"]["test"]["roc_auc_ovr_macro"],
            "High-demand precision": metrics["baseline_model"]["test"]["high_demand_precision"],
            "High-demand recall": metrics["baseline_model"]["test"]["high_demand_recall"],
        },
        {
            "Model": metrics["final_model"].get("name", "Final model"),
            "Accuracy": final["accuracy"],
            "Precision": final["precision_macro"],
            "Recall": final["recall_macro"],
            "F1": final["f1_macro"],
            "ROC-AUC": final["roc_auc_ovr_macro"],
            "High-demand precision": final["high_demand_precision"],
            "High-demand recall": final["high_demand_recall"],
        },
    ]
)
st.dataframe(comparison, width="stretch", hide_index=True)
comparison_long = comparison.melt(id_vars="Model", var_name="Metric", value_name="Score")
comparison_long = comparison_long[comparison_long["Metric"].isin(["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"])]
fig_comparison = px.bar(
    comparison_long,
    x="Metric",
    y="Score",
    color="Model",
    barmode="group",
    color_discrete_sequence=[CHART_COLORS[5], CHART_COLORS[0]],
    title="Baseline vs Final Model Metrics",
)
fig_comparison.update_yaxes(tickformat=".0%", range=[0, 1])
st.plotly_chart(style_plotly(fig_comparison, height=360), width="stretch")

aux_cols = st.columns(3)
with aux_cols[0]:
    sell = metrics.get("sellthrough_regressor", {}).get("final", {}).get("test", {})
    render_kpi_card("Sell-through RMSE", f"{sell.get('rmse', 0):.3f}", note=f"MAE {sell.get('mae', 0):.3f}")
with aux_cols[1]:
    scalper = metrics.get("scalper_risk_classifier", {}).get("final", {}).get("test", {})
    render_kpi_card("Scalper F1", f"{scalper.get('f1_macro', 0):.1%}", note=f"Recall {scalper.get('recall_macro', 0):.1%}")
with aux_cols[2]:
    pricing = metrics.get("business_metrics", {})
    render_kpi_card("Approval rate", f"{pricing.get('percentage_human_approval_required', 0):.1%}", note="Human-in-loop share")

left, right = st.columns(2)
with left:
    matrix = pd.DataFrame(final["confusion_matrix"], index=CLASS_ORDER, columns=CLASS_ORDER)
    fig_matrix = px.imshow(
        matrix,
        text_auto=True,
        color_continuous_scale="Blues",
        labels={"x": "Predicted tier", "y": "Actual tier", "color": "Rows"},
        title="Confusion Matrix",
    )
    st.plotly_chart(style_plotly(fig_matrix, height=420), width="stretch")
with right:
    importance_path = PROJECT_ROOT / "outputs" / "feature_importance.csv"
    if importance_path.exists():
        importance = pd.read_csv(importance_path).head(12)
        fig_importance = px.bar(
            importance.sort_values("importance"),
            x="importance",
            y="feature",
            orientation="h",
            color_discrete_sequence=[CHART_COLORS[0]],
            title="Feature Importance",
        )
        st.plotly_chart(style_plotly(fig_importance, height=420), width="stretch")
    else:
        st.warning("Feature importance output is missing. Run `python -m src.train_model` to regenerate.")

render_section(
    "Business meaning of model errors",
    "The model is evaluated in terms of pricing risk, not only technical classification metrics.",
    "Critical analysis",
)
err_cols = st.columns(2)
with err_cols[0]:
    render_action_card(
        "False positive = overpricing risk",
        "Model predicts high demand when actual demand is lower.",
        "Prices may rise too far, creating empty seats and lost concession revenue.",
        "Mitigate with price caps, confidence thresholds, and approval review.",
    )
with err_cols[1]:
    render_action_card(
        "False negative = underpricing risk",
        "Model misses high-demand inventory.",
        "Tickets sell out too quickly and scalpers capture the resale upside.",
        "Mitigate with high-demand recall monitoring and resale gap alerts.",
    )

render_section(
    "Pricing engine performance",
    "The optimizer simulates candidate prices, applies caps and affordable ticket reserve logic, and routes high-impact decisions to review.",
    "Revenue operations",
)
pricing = metrics.get("business_metrics", {})
pricing_cols = st.columns(4)
with pricing_cols[0]:
    render_kpi_card("Revenue uplift", f"${pricing.get('estimated_revenue_uplift', 0)/1_000_000:.2f}M", delta=f"{pricing.get('estimated_revenue_uplift_pct', 0):.1%}", note="Test-season simulation")
with pricing_cols[1]:
    render_kpi_card("Sell-through lift", f"{pricing.get('estimated_sell_through_improvement', 0):.1%}", note="Expected vs current")
with pricing_cols[2]:
    render_kpi_card("Guardrail caps", f"{pricing.get('percentage_capped_by_guardrails', 0):.1%}", note="Recommendations constrained")
with pricing_cols[3]:
    render_kpi_card("Resale gap after", f"${pricing.get('average_price_gap_vs_resale_after', 0):,.0f}", note="Average underpricing gap")

with st.expander("Training data preview"):
    st.write(
        f"The active model was trained on `{dataset_meta.get('active_csv', 'data/model_training_data.csv')}` "
        f"with {len(data):,} game-section-window rows. The table below is a preview by default."
    )
    show_full_dataset = st.toggle("Show full active training dataset", value=False)
    preview = data if show_full_dataset else data.head(80)
    st.dataframe(preview, width="stretch", hide_index=True)

render_footer()
