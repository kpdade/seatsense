"""Train and evaluate SeatSense AI demand, sell-through, and scalper models."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.generate_ticket_data import generate_all  # noqa: E402
from src.fairness import build_fairness_audit  # noqa: E402
from src.leakage_audit import save_leakage_audit  # noqa: E402
from src.model_registry import MODEL_PATHS  # noqa: E402
from src.preprocess import (  # noqa: E402
    BUSINESS_METRIC_COLUMNS,
    CLASS_ORDER,
    FEATURE_COLUMNS,
    REQUIRED_TRAINING_COLUMNS,
    SCALPER_CLASS_ORDER,
    SCALPER_TARGET_COLUMN,
    SELLTHROUGH_TARGET_COLUMN,
    TARGET_COLUMN,
    build_preprocessor,
    validate_training_frame,
)
from src.pricing_optimizer import optimize_price  # noqa: E402
from src.schema_validation import validate_seatsense_training_schema  # noqa: E402


DATA_PATH = PROJECT_ROOT / "data" / "model_training_data.csv"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DELIVERABLES_DIR = PROJECT_ROOT / "deliverables"
MIN_DEFAULT_ROWS = 10_000


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def validate_retraining_frame(frame: pd.DataFrame) -> None:
    validate_training_frame(frame)
    schema_report = validate_seatsense_training_schema(frame)
    if not schema_report.passed:
        raise ValueError(
            "Training data failed the SeatSense Pydantic schema gate: "
            + "; ".join(schema_report.errors[:6])
        )
    if len(frame) < 500:
        raise ValueError("Training CSV must contain at least 500 rows; 10,000+ is recommended.")
    for target in [TARGET_COLUMN, SCALPER_TARGET_COLUMN]:
        class_counts = frame[target].value_counts()
        missing_classes = [label for label in CLASS_ORDER if label not in class_counts.index]
        if missing_classes:
            raise ValueError(f"Training CSV must include all {target} classes: {missing_classes}")
        small_classes = class_counts[class_counts < 25]
        if not small_classes.empty:
            raise ValueError(
                f"Each {target} class should have at least 25 rows. Low-count classes: "
                f"{small_classes.to_dict()}"
            )
    if frame["game_id"].nunique() < 60:
        raise ValueError("Training CSV needs at least 60 unique games.")
    if frame["season"].nunique() < 3:
        # This is acceptable for a client pilot as long as game groups exist; the
        # split function will fall back to group-aware validation.
        return


def _default_data_needs_refresh(path: Path) -> bool:
    if not path.exists():
        return True
    try:
        sample = pd.read_csv(path, nrows=5)
        missing = [col for col in REQUIRED_TRAINING_COLUMNS if col not in sample.columns]
        row_count = sum(1 for _ in path.open("r", encoding="utf-8")) - 1
        return bool(missing) or row_count < MIN_DEFAULT_ROWS
    except Exception:
        return True


def load_or_generate_data(data_path: Path | None = None) -> pd.DataFrame:
    active_path = Path(data_path) if data_path else DATA_PATH
    if active_path == DATA_PATH and _default_data_needs_refresh(active_path):
        generate_all(PROJECT_ROOT / "data")
    frame = pd.read_csv(active_path)
    validate_retraining_frame(frame)
    return frame


def _group_split(frame: pd.DataFrame, random_state: int = 751) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(random_state)
    games = np.array(sorted(frame["game_id"].astype(str).unique()))
    rng.shuffle(games)
    train_end = max(1, int(len(games) * 0.70))
    validation_end = max(train_end + 1, int(len(games) * 0.85))
    train_games = set(games[:train_end])
    validation_games = set(games[train_end:validation_end])
    test_games = set(games[validation_end:])

    train_frame = frame[frame["game_id"].astype(str).isin(train_games)].copy()
    validation_frame = frame[frame["game_id"].astype(str).isin(validation_games)].copy()
    test_frame = frame[frame["game_id"].astype(str).isin(test_games)].copy()
    split_summary = {
        "strategy": "Group-aware split by game_id fallback; no game appears in more than one split.",
        "grouping": "game_id",
        "train_seasons": sorted([int(s) for s in train_frame["season"].unique()]) if "season" in frame else [],
        "validation_seasons": sorted([int(s) for s in validation_frame["season"].unique()]) if "season" in frame else [],
        "test_seasons": sorted([int(s) for s in test_frame["season"].unique()]) if "season" in frame else [],
        "train_rows": int(len(train_frame)),
        "validation_rows": int(len(validation_frame)),
        "test_rows": int(len(test_frame)),
        "train_games": int(train_frame["game_id"].nunique()),
        "validation_games": int(validation_frame["game_id"].nunique()),
        "test_games": int(test_frame["game_id"].nunique()),
        "demand_tier_distribution": {
            "train": train_frame[TARGET_COLUMN].value_counts(normalize=True).round(4).to_dict(),
            "validation": validation_frame[TARGET_COLUMN].value_counts(normalize=True).round(4).to_dict(),
            "test": test_frame[TARGET_COLUMN].value_counts(normalize=True).round(4).to_dict(),
        },
        "season_coverage": {
            "train": sorted([int(s) for s in train_frame["season"].unique()]) if "season" in frame else [],
            "validation": sorted([int(s) for s in validation_frame["season"].unique()]) if "season" in frame else [],
            "test": sorted([int(s) for s in test_frame["season"].unique()]) if "season" in frame else [],
        },
    }
    return train_frame, validation_frame, test_frame, split_summary


def split_by_time(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    seasons = sorted(frame["season"].unique().tolist())
    if len(seasons) < 3:
        return _group_split(frame)
    if len(seasons) == 3:
        train_seasons = [seasons[0]]
        validation_seasons = [seasons[1]]
        test_seasons = [seasons[2]]
    else:
        train_end = max(1, int(round(len(seasons) * 0.60)))
        validation_end = max(train_end + 1, int(round(len(seasons) * 0.80)))
        validation_end = min(validation_end, len(seasons) - 1)
        train_seasons = seasons[:train_end]
        validation_seasons = seasons[train_end:validation_end]
        test_seasons = seasons[validation_end:]

    train_frame = frame[frame["season"].isin(train_seasons)].copy()
    validation_frame = frame[frame["season"].isin(validation_seasons)].copy()
    test_frame = frame[frame["season"].isin(test_seasons)].copy()

    split_summary = {
        "strategy": "Time-based split by season; all rows for the same game_id remain in one split.",
        "grouping": "game_id",
        "train_seasons": [int(s) for s in train_seasons],
        "validation_seasons": [int(s) for s in validation_seasons],
        "test_seasons": [int(s) for s in test_seasons],
        "train_rows": int(len(train_frame)),
        "validation_rows": int(len(validation_frame)),
        "test_rows": int(len(test_frame)),
        "train_games": int(train_frame["game_id"].nunique()),
        "validation_games": int(validation_frame["game_id"].nunique()),
        "test_games": int(test_frame["game_id"].nunique()),
        "demand_tier_distribution": {
            "train": train_frame[TARGET_COLUMN].value_counts(normalize=True).round(4).to_dict(),
            "validation": validation_frame[TARGET_COLUMN].value_counts(normalize=True).round(4).to_dict(),
            "test": test_frame[TARGET_COLUMN].value_counts(normalize=True).round(4).to_dict(),
        },
        "season_coverage": {
            "train": [int(s) for s in sorted(train_frame["season"].unique())],
            "validation": [int(s) for s in sorted(validation_frame["season"].unique())],
            "test": [int(s) for s in sorted(test_frame["season"].unique())],
        },
    }
    return train_frame, validation_frame, test_frame, split_summary


def build_baseline_classifier() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "model",
                DecisionTreeClassifier(
                    max_depth=1,
                    min_samples_leaf=200,
                    random_state=751,
                ),
            ),
        ]
    )


def build_random_forest_search(random_state: int = 751) -> RandomizedSearchCV:
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", RandomForestClassifier(random_state=random_state, n_jobs=-1)),
        ]
    )
    param_distributions = {
        "model__n_estimators": [180, 220, 260],
        "model__max_depth": [8, 10, 12],
        "model__min_samples_split": [10, 12, 16],
        "model__min_samples_leaf": [8, 10, 12],
        "model__max_features": ["sqrt", 0.65, 0.85],
        "model__class_weight": ["balanced_subsample"],
    }

    def high_demand_recall(y_true, y_pred) -> float:
        y_true_array = np.asarray(y_true)
        y_pred_array = np.asarray(y_pred)
        mask = y_true_array == "High"
        if not bool(mask.any()):
            return 0.0
        return float((y_pred_array[mask] == "High").mean())

    return RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=7,
        scoring={
            "macro_f1": "f1_macro",
            "high_demand_recall": make_scorer(high_demand_recall),
        },
        refit="macro_f1",
        cv=GroupKFold(n_splits=3),
        random_state=random_state,
        n_jobs=-1,
        verbose=0,
    )


def build_sellthrough_models(random_state: int = 751) -> tuple[Pipeline, Pipeline]:
    baseline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", DummyRegressor(strategy="mean")),
        ]
    )
    final = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=240,
                    max_depth=16,
                    min_samples_leaf=4,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return baseline, final


def build_scalper_models(random_state: int = 751) -> tuple[Pipeline, Pipeline]:
    baseline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", DecisionTreeClassifier(max_depth=4, min_samples_leaf=35, random_state=random_state)),
        ]
    )
    final = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=240,
                    max_depth=14,
                    min_samples_leaf=3,
                    class_weight="balanced_subsample",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return baseline, final


def evaluate_classifier(model: Pipeline, X: pd.DataFrame, y: pd.Series, labels: list[str]) -> dict[str, Any]:
    predictions = model.predict(X)
    report = classification_report(
        y,
        predictions,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    metrics: dict[str, Any] = {
        "accuracy": round(float(accuracy_score(y, predictions)), 4),
        "precision_macro": round(float(precision_score(y, predictions, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y, predictions, average="macro", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y, predictions, average="macro", zero_division=0)), 4),
        "precision_weighted": round(float(precision_score(y, predictions, average="weighted", zero_division=0)), 4),
        "recall_weighted": round(float(recall_score(y, predictions, average="weighted", zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(y, predictions, average="weighted", zero_division=0)), 4),
        "high_demand_precision": round(float(report.get("High", {}).get("precision", 0)), 4),
        "high_demand_recall": round(float(report.get("High", {}).get("recall", 0)), 4),
        "high_demand_f1": round(float(report.get("High", {}).get("f1-score", 0)), 4),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y, predictions, labels=labels).tolist(),
    }
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(X)
            metrics["roc_auc_ovr_macro"] = round(
                float(
                    roc_auc_score(
                        y,
                        probabilities,
                        labels=list(model.classes_),
                        multi_class="ovr",
                        average="macro",
                    )
                ),
                4,
            )
        except Exception:
            metrics["roc_auc_ovr_macro"] = None
    return metrics


def evaluate_regressor(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    predictions = np.clip(model.predict(X), 0, 1)
    mse = float(mean_squared_error(y, predictions))
    return {
        "mae": round(float(mean_absolute_error(y, predictions)), 4),
        "rmse": round(float(np.sqrt(mse)), 4),
        "r2": round(float(r2_score(y, predictions)), 4),
        "mape": round(float(mean_absolute_percentage_error(y.clip(lower=0.05), predictions)), 4),
    }


def transformed_feature_names(preprocessor) -> list[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return FEATURE_COLUMNS


def plot_confusion_matrix(matrix: list[list[int]], labels: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted tier")
    ax.set_ylabel("Actual tier")
    ax.set_title("Demand Tier Confusion Matrix")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, matrix[i][j], ha="center", va="center", color="#111827")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_feature_importance(
    final_model: Pipeline,
    X_reference: pd.DataFrame,
    y_reference: pd.Series,
    path_png: Path,
    path_csv: Path,
) -> pd.DataFrame:
    model_step = final_model.named_steps["model"]
    if hasattr(model_step, "feature_importances_"):
        names = transformed_feature_names(final_model.named_steps["preprocessor"])
        importances = model_step.feature_importances_
        importance = pd.DataFrame({"feature": names, "importance": importances})
    else:
        result = permutation_importance(
            final_model,
            X_reference,
            y_reference,
            n_repeats=3,
            scoring="f1_macro",
            random_state=751,
            n_jobs=-1,
        )
        importance = pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": result.importances_mean})

    importance = importance.sort_values("importance", ascending=False).reset_index(drop=True)
    importance.to_csv(path_csv, index=False)
    top = importance.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    ax.barh(top["feature"], top["importance"], color="#2563eb")
    ax.set_title("Top Demand Prediction Drivers")
    ax.set_xlabel("Importance")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(path_png, dpi=180)
    plt.close(fig)
    return importance


def simulate_pricing(
    frame: pd.DataFrame,
    demand_model: Pipeline,
    sellthrough_model: Pipeline,
    scalper_model: Pipeline,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    X = frame[FEATURE_COLUMNS]
    demand_predictions = demand_model.predict(X)
    demand_probabilities = demand_model.predict_proba(X)
    sellthrough_predictions = np.clip(sellthrough_model.predict(X), 0, 1)
    scalper_predictions = scalper_model.predict(X)
    scalper_probabilities = scalper_model.predict_proba(X)

    class_index = {label: i for i, label in enumerate(demand_model.classes_)}
    scalper_class_index = {label: i for i, label in enumerate(scalper_model.classes_)}
    rows: list[dict[str, Any]] = []
    for idx, (_, source_row) in enumerate(frame.iterrows()):
        demand_tier = str(demand_predictions[idx])
        demand_output = {
            "demand_tier": demand_tier,
            "demand_probability": float(demand_probabilities[idx][class_index[demand_tier]]),
        }
        scalper_tier = str(scalper_predictions[idx])
        scalper_output = {
            "scalper_risk_tier": scalper_tier,
            "scalper_risk_probability": float(scalper_probabilities[idx][scalper_class_index[scalper_tier]]),
        }
        result = optimize_price(
            source_row.to_dict(),
            demand_output=demand_output,
            sellthrough_output={"predicted_sell_through": float(sellthrough_predictions[idx])},
            scalper_output=scalper_output,
        )
        rows.append(
            {
                **source_row.to_dict(),
                "predicted_demand_tier": demand_tier,
                "predicted_demand_probability": demand_output["demand_probability"],
                "predicted_sell_through": float(sellthrough_predictions[idx]),
                "predicted_scalper_risk_tier": scalper_tier,
                "predicted_scalper_risk_probability": scalper_output["scalper_risk_probability"],
                "recommended_price": result.recommended_price,
                "price_change_pct": result.price_change_pct,
                "expected_sell_through": result.expected_sell_through,
                "expected_revenue": result.expected_revenue,
                "revenue_uplift_vs_current": result.revenue_uplift_vs_current,
                "secondary_market_leakage_reduction": result.secondary_market_leakage_reduction,
                "human_approval_required": result.human_approval_required,
                "affordability_risk_flag": result.affordability_risk_flag,
                "capped_by_guardrail": result.capped_by_guardrail,
                "pricing_action": result.pricing_action,
            }
        )

    simulation = pd.DataFrame(rows)
    current_revenue = float(simulation["revenue"].sum())
    expected_uplift = float(simulation["revenue_uplift_vs_current"].sum())
    projected_revenue = current_revenue + expected_uplift
    current_sellthrough_at_window = float(simulation["tickets_sold_pct"].mean())
    model_baseline_sellthrough = float(simulation["predicted_sell_through"].mean())
    expected_sellthrough = float(simulation["expected_sell_through"].mean())
    inventory_risk_mask = simulation["predicted_demand_tier"].eq("Low") | simulation["pricing_action"].eq("Discount")
    if bool(inventory_risk_mask.any()):
        risk_baseline_sellthrough = float(simulation.loc[inventory_risk_mask, "predicted_sell_through"].mean())
        risk_expected_sellthrough = float(simulation.loc[inventory_risk_mask, "expected_sell_through"].mean())
    else:
        risk_baseline_sellthrough = model_baseline_sellthrough
        risk_expected_sellthrough = expected_sellthrough
    inventory_sellthrough_gain = max(0.0, risk_expected_sellthrough - risk_baseline_sellthrough)
    leakage_reduction = float(simulation["secondary_market_leakage_reduction"].sum())

    pricing_metrics = {
        "current_total_revenue": round(current_revenue, 2),
        "projected_total_revenue": round(projected_revenue, 2),
        "estimated_revenue_uplift": round(expected_uplift, 2),
        "estimated_revenue_uplift_pct": round(expected_uplift / max(current_revenue, 1), 4),
        "average_recommended_price_change": round(float(simulation["price_change_pct"].mean()), 4),
        "median_recommended_price_change": round(float(simulation["price_change_pct"].median()), 4),
        "average_sell_through_current": round(model_baseline_sellthrough, 4),
        "average_current_window_sell_through": round(current_sellthrough_at_window, 4),
        "average_sell_through_expected": round(expected_sellthrough, 4),
        "estimated_sell_through_improvement": round(inventory_sellthrough_gain, 4),
        "low_demand_sell_through_baseline": round(risk_baseline_sellthrough, 4),
        "low_demand_sell_through_expected": round(risk_expected_sellthrough, 4),
        "secondary_market_leakage_reduction": round(leakage_reduction, 2),
        "percentage_human_approval_required": round(float(simulation["human_approval_required"].mean()), 4),
        "percentage_capped_by_guardrails": round(float(simulation["capped_by_guardrail"].mean()), 4),
        "average_price_gap_vs_resale_before": round(
            float((simulation["secondary_market_avg_price"] - simulation["current_ticket_price"]).clip(lower=0).mean()),
            2,
        ),
        "average_price_gap_vs_resale_after": round(
            float((simulation["secondary_market_avg_price"] - simulation["recommended_price"]).clip(lower=0).mean()),
            2,
        ),
        "fan_affordability_score": round(
            float(
                100
                - np.clip(
                    simulation.loc[
                        simulation["affordability_segment"].eq("Value-sensitive"),
                        "price_change_pct",
                    ]
                    .clip(lower=0)
                    .mean()
                    * 145,
                    0,
                    45,
                )
            ),
            1,
        ),
    }
    pricing_metrics["secondary_market_leakage_before"] = round(
        float(
            (
                (simulation["secondary_market_avg_price"] - simulation["current_ticket_price"]).clip(lower=0)
                * simulation["sold_tickets"]
                * 0.14
            ).sum()
        ),
        2,
    )
    pricing_metrics["secondary_market_leakage_after"] = round(
        pricing_metrics["secondary_market_leakage_before"] - leakage_reduction,
        2,
    )
    pricing_metrics["underpricing_gap_before"] = pricing_metrics["average_price_gap_vs_resale_before"]
    pricing_metrics["underpricing_gap_after"] = pricing_metrics["average_price_gap_vs_resale_after"]
    pricing_metrics["underpricing_gap_reduction_pct"] = round(
        (
            pricing_metrics["underpricing_gap_before"]
            - pricing_metrics["underpricing_gap_after"]
        )
        / max(pricing_metrics["underpricing_gap_before"], 1),
        4,
    )
    pricing_metrics["unsold_inventory_risk_reduction_pct"] = round(
        inventory_sellthrough_gain / max(1 - risk_baseline_sellthrough, 0.001),
        4,
    )
    pricing_metrics["avg_recommendation_change_pct"] = pricing_metrics["average_recommended_price_change"]
    pricing_metrics["human_approval_rate_proxy"] = pricing_metrics["percentage_human_approval_required"]
    return simulation, pricing_metrics


def save_split_summary(summary: dict[str, Any]) -> None:
    with (OUTPUTS_DIR / "split_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def write_markdown_artifacts(
    metrics: dict[str, Any],
    leakage_audit: dict[str, Any],
    split_summary: dict[str, Any],
) -> None:
    model_card = f"""# SeatSense AI Model Card

## Intended Use
SeatSense AI is a decision-support prototype for sports venue revenue managers. It predicts demand tier, sell-through, and scalper risk, then recommends constrained ticket price actions.

## Not Intended Use
The model should not autonomously publish prices, target individual fans, or use protected demographics to maximize willingness to pay.

## Model Inputs
Inputs include game context, seat section, current price, current inventory, pricing window, aggregated demand signals, secondary-market indicators, and historical venue behavior.

## Excluded Sensitive or Post-Event Features
Affordability columns are excluded from model training and used only for fairness monitoring and guardrails. Post-event outcomes such as final attendance, final sell-through, realized revenue, target labels, and oracle prices are forbidden model inputs.

## Validation Design
{split_summary["strategy"]} Train seasons: {split_summary["train_seasons"]}; validation season: {split_summary["validation_seasons"]}; test season: {split_summary["test_seasons"]}.

## Metrics
- Final demand accuracy: {metrics["final_model"]["test"]["accuracy"]:.1%}
- Final demand macro F1: {metrics["final_model"]["test"]["f1_macro"]:.1%}
- High-demand recall: {metrics["final_model"]["test"]["high_demand_recall"]:.1%}
- ROC-AUC OVR: {metrics["final_model"]["test"].get("roc_auc_ovr_macro", 0):.1%}
- Sell-through RMSE: {metrics["sellthrough_regressor"]["final"]["test"]["rmse"]:.3f}
- Scalper-risk macro F1: {metrics["scalper_risk_classifier"]["final"]["test"]["f1_macro"]:.1%}

## Human Review Rules
Human review is required for premium games, price increases above 20%, affordability guardrail flags, low-confidence predictions, aggressive high-scalper-risk actions, or guardrail-capped recommendations.

## Limitations
Because private ticketing transactions are not publicly available, this prototype uses a semi-synthetic dataset calibrated to realistic sports ticketing dynamics. Real deployment would require historical transaction feeds, ticketing system integration, privacy review, and ongoing monitoring.
"""

    data_card = f"""# SeatSense AI Data Card

## Dataset Source
This prototype uses a semi-synthetic dataset generated by `data/generate_ticket_data.py`. It is not represented as real historical sales data.

## Why Synthetic Data Is Used
Private ticketing, resale, web analytics, social, and CRM data are usually not publicly available. The generator creates realistic relationships among sports demand signals, inventory, pricing windows, secondary-market pressure, sell-through, and revenue outcomes.

## Generated Data Shape
- Rows: {metrics["dataset"]["rows"]:,}
- Games: {metrics["dataset"]["games"]:,}
- Seasons: {metrics["dataset"]["seasons"]}
- Seat sections: {metrics["dataset"]["seat_sections_count"]}
- Pricing windows: {metrics["dataset"]["pricing_windows_count"]}

## Leakage Prevention
{leakage_audit["message"]}

## What Real Deployment Would Require
Real venue deployment would require ticket transaction history, pricing change logs, inventory snapshots, verified resale signals, weather and sports APIs, aggregated web analytics, privacy review, and business approval policies.

## References To Replace Or Verify
- [Source needed: NBA dynamic pricing]
- [Source needed: Ticketmaster dynamic pricing]
- [Source needed: secondary ticket market research]
- [Source needed: algorithmic pricing fairness governance]
"""

    model_report = f"""# SeatSense AI Model Report

## Data Upgrade
The training architecture now uses game-section-pricing-window observations rather than one row per game section. This increases the default training dataset to {metrics["dataset"]["rows"]:,} rows while preserving a clear distinction between known pre-pricing features and post-event outcomes.

## Validation Strategy
The model uses a time-based split:
- Train: {split_summary["train_rows"]:,} rows across {split_summary["train_games"]:,} games
- Validation: {split_summary["validation_rows"]:,} rows across {split_summary["validation_games"]:,} games
- Test: {split_summary["test_rows"]:,} rows across {split_summary["test_games"]:,} games

This is stronger than a random row split because all rows for a game stay in the same season-based split.

## Baseline Vs Final Demand Model
- Baseline accuracy: {metrics["baseline_model"]["test"]["accuracy"]:.1%}
- Baseline macro F1: {metrics["baseline_model"]["test"]["f1_macro"]:.1%}
- Final accuracy: {metrics["final_model"]["test"]["accuracy"]:.1%}
- Final macro F1: {metrics["final_model"]["test"]["f1_macro"]:.1%}
- Final high-demand recall: {metrics["final_model"]["test"]["high_demand_recall"]:.1%}
- Final ROC-AUC OVR: {metrics["final_model"]["test"].get("roc_auc_ovr_macro", 0):.1%}

## Business Error Analysis
False positive high demand means the model predicts high demand when actual demand is lower. The business risk is overpricing, empty seats, fan frustration, and lost concession revenue.

False negative high demand means the model misses high-demand inventory. The business risk is underpricing, rapid sellout, scalper profit, and lost venue revenue.

## Pricing Simulation
- Estimated revenue uplift: ${metrics["business_metrics"]["estimated_revenue_uplift"]:,.0f}
- Estimated sell-through improvement: {metrics["business_metrics"]["estimated_sell_through_improvement"]:.1%}
- Human approval rate: {metrics["business_metrics"]["percentage_human_approval_required"]:.1%}
- Guardrail cap trigger rate: {metrics["business_metrics"]["percentage_capped_by_guardrails"]:.1%}

## Responsible AI
No protected demographics are used as model features. The affordability index is used only for audit, approval guardrails, and affordable inventory reserve simulation.
"""

    (OUTPUTS_DIR / "model_card.md").write_text(model_card, encoding="utf-8")
    (OUTPUTS_DIR / "data_card.md").write_text(data_card, encoding="utf-8")
    (OUTPUTS_DIR / "model_report.md").write_text(model_report, encoding="utf-8")


def train(
    random_state: int = 751,
    data_path: Path | None = None,
    source_name: str = "data/model_training_data.csv",
) -> dict[str, Any]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    DELIVERABLES_DIR.mkdir(parents=True, exist_ok=True)

    frame = load_or_generate_data(data_path=data_path)
    leakage_audit = save_leakage_audit(frame, OUTPUTS_DIR / "leakage_audit.json", FEATURE_COLUMNS)
    if not leakage_audit["passed"]:
        raise ValueError(f"Leakage audit failed: {leakage_audit}")

    train_frame, validation_frame, test_frame, split_summary = split_by_time(frame)
    save_split_summary(split_summary)

    X_train = train_frame[FEATURE_COLUMNS]
    y_train = train_frame[TARGET_COLUMN]
    X_validation = validation_frame[FEATURE_COLUMNS]
    y_validation = validation_frame[TARGET_COLUMN]
    X_test = test_frame[FEATURE_COLUMNS]
    y_test = test_frame[TARGET_COLUMN]

    X_train_val = pd.concat([X_train, X_validation], axis=0)
    y_train_val = pd.concat([y_train, y_validation], axis=0)
    train_val_frame = pd.concat([train_frame, validation_frame], axis=0)

    baseline = build_baseline_classifier()
    baseline.fit(X_train, y_train)
    baseline_train_metrics = evaluate_classifier(baseline, X_train, y_train, CLASS_ORDER)
    baseline_validation_metrics = evaluate_classifier(baseline, X_validation, y_validation, CLASS_ORDER)

    rf_search = build_random_forest_search(random_state=random_state)
    rf_search.fit(X_train, y_train, groups=train_frame["game_id"])
    final_validation_model = rf_search.best_estimator_
    final_train_metrics = evaluate_classifier(final_validation_model, X_train, y_train, CLASS_ORDER)
    final_validation_metrics = evaluate_classifier(final_validation_model, X_validation, y_validation, CLASS_ORDER)

    final_model = clone(final_validation_model)
    final_model.fit(X_train_val, y_train_val)
    baseline_test_model = clone(baseline)
    baseline_test_model.fit(X_train_val, y_train_val)

    baseline_test_metrics = evaluate_classifier(baseline_test_model, X_test, y_test, CLASS_ORDER)
    final_test_metrics = evaluate_classifier(final_model, X_test, y_test, CLASS_ORDER)

    sell_baseline, sell_final_validation = build_sellthrough_models(random_state=random_state)
    sell_baseline.fit(X_train, train_frame[SELLTHROUGH_TARGET_COLUMN])
    sell_final_validation.fit(X_train, train_frame[SELLTHROUGH_TARGET_COLUMN])
    sell_final = clone(sell_final_validation)
    sell_final.fit(X_train_val, train_val_frame[SELLTHROUGH_TARGET_COLUMN])
    sell_baseline_test = clone(sell_baseline)
    sell_baseline_test.fit(X_train_val, train_val_frame[SELLTHROUGH_TARGET_COLUMN])

    scalper_baseline, scalper_final_validation = build_scalper_models(random_state=random_state)
    scalper_baseline.fit(X_train, train_frame[SCALPER_TARGET_COLUMN])
    scalper_final_validation.fit(X_train, train_frame[SCALPER_TARGET_COLUMN])
    scalper_final = clone(scalper_final_validation)
    scalper_final.fit(X_train_val, train_val_frame[SCALPER_TARGET_COLUMN])
    scalper_baseline_test = clone(scalper_baseline)
    scalper_baseline_test.fit(X_train_val, train_val_frame[SCALPER_TARGET_COLUMN])

    pricing_simulation, pricing_metrics = simulate_pricing(test_frame, final_model, sell_final, scalper_final)
    pricing_simulation.to_csv(OUTPUTS_DIR / "pricing_simulation.csv", index=False)
    with (OUTPUTS_DIR / "pricing_simulation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(pricing_metrics, f, indent=2)

    fairness_audit = build_fairness_audit(pricing_simulation, OUTPUTS_DIR / "fairness_audit.json")

    metrics = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "source": source_name,
            "active_csv": _display_path(Path(data_path or DATA_PATH)),
            "rows": int(len(frame)),
            "games": int(frame["game_id"].nunique()),
            "seasons": int(frame["season"].nunique()),
            "seat_sections": sorted(frame["seat_section"].unique().tolist()),
            "seat_sections_count": int(frame["seat_section"].nunique()),
            "pricing_windows": sorted(frame["pricing_window"].unique().tolist()),
            "pricing_windows_count": int(frame["pricing_window"].nunique()),
            "class_distribution": frame[TARGET_COLUMN].value_counts().to_dict(),
            "scalper_risk_distribution": frame[SCALPER_TARGET_COLUMN].value_counts().to_dict(),
            "observation_unit": "game_id + seat_section + pricing_window",
            "synthetic_data_note": (
                "Semi-synthetic data calibrated to realistic sports ticketing dynamics; "
                "not real venue transaction data."
            ),
        },
        "split": split_summary,
        "leakage_audit": leakage_audit,
        "baseline_model": {
            "name": "Decision Tree baseline",
            "train": baseline_train_metrics,
            "validation": baseline_validation_metrics,
            "test": baseline_test_metrics,
        },
        "final_model": {
            "name": "Random Forest demand classifier",
            "train": final_train_metrics,
            "validation": final_validation_metrics,
            "test": final_test_metrics,
            "best_params": rf_search.best_params_,
            "cross_validation": {
                "strategy": "GroupKFold by game_id on training seasons",
                "metric": "macro_f1",
                "mean": round(float(rf_search.best_score_), 4),
                "best_high_demand_recall_cv": round(
                    float(rf_search.cv_results_["mean_test_high_demand_recall"][rf_search.best_index_]),
                    4,
                ),
            },
            "overfitting_check": {
                "train_minus_validation_accuracy": round(
                    final_train_metrics["accuracy"] - final_validation_metrics["accuracy"], 4
                ),
                "flag": bool(final_train_metrics["accuracy"] - final_validation_metrics["accuracy"] > 0.08),
            },
        },
        "sellthrough_regressor": {
            "baseline": {
                "name": "DummyRegressor mean baseline",
                "validation": evaluate_regressor(
                    sell_baseline,
                    X_validation,
                    validation_frame[SELLTHROUGH_TARGET_COLUMN],
                ),
                "test": evaluate_regressor(
                    sell_baseline_test,
                    X_test,
                    test_frame[SELLTHROUGH_TARGET_COLUMN],
                ),
            },
            "final": {
                "name": "Random Forest sell-through regressor",
                "validation": evaluate_regressor(
                    sell_final_validation,
                    X_validation,
                    validation_frame[SELLTHROUGH_TARGET_COLUMN],
                ),
                "test": evaluate_regressor(
                    sell_final,
                    X_test,
                    test_frame[SELLTHROUGH_TARGET_COLUMN],
                ),
            },
        },
        "scalper_risk_classifier": {
            "baseline": {
                "name": "Decision Tree baseline",
                "validation": evaluate_classifier(
                    scalper_baseline,
                    X_validation,
                    validation_frame[SCALPER_TARGET_COLUMN],
                    SCALPER_CLASS_ORDER,
                ),
                "test": evaluate_classifier(
                    scalper_baseline_test,
                    X_test,
                    test_frame[SCALPER_TARGET_COLUMN],
                    SCALPER_CLASS_ORDER,
                ),
            },
            "final": {
                "name": "Random Forest scalper risk classifier",
                "validation": evaluate_classifier(
                    scalper_final_validation,
                    X_validation,
                    validation_frame[SCALPER_TARGET_COLUMN],
                    SCALPER_CLASS_ORDER,
                ),
                "test": evaluate_classifier(
                    scalper_final,
                    X_test,
                    test_frame[SCALPER_TARGET_COLUMN],
                    SCALPER_CLASS_ORDER,
                ),
            },
        },
        "business_metrics": pricing_metrics,
        "fairness_metrics": fairness_audit,
        "false_positive_negative_analysis": {
            "false_positive_high_demand": (
                "Model predicts high demand when actual demand is low; result can be prices that "
                "are too high, empty seats, fan frustration, and lost concession revenue."
            ),
            "false_negative_high_demand": (
                "Model predicts low demand when actual demand is high; result can be prices that "
                "are too low, fast sellout, scalper profit, and lost venue revenue."
            ),
        },
    }

    joblib.dump(final_model, MODEL_PATHS["demand_classifier"])
    joblib.dump(final_model, MODEL_PATHS["legacy_demand_model"])
    joblib.dump(final_model.named_steps["preprocessor"], MODEL_PATHS["preprocessor"])
    joblib.dump(sell_final, MODEL_PATHS["sellthrough_regressor"])
    joblib.dump(scalper_final, MODEL_PATHS["scalper_risk_classifier"])
    frame.to_csv(DELIVERABLES_DIR / "training_dataset_snapshot.csv", index=False)
    with (OUTPUTS_DIR / "model_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plot_confusion_matrix(
        final_test_metrics["confusion_matrix"],
        CLASS_ORDER,
        OUTPUTS_DIR / "confusion_matrix.png",
    )
    save_feature_importance(
        final_model,
        X_validation,
        y_validation,
        OUTPUTS_DIR / "feature_importance.png",
        OUTPUTS_DIR / "feature_importance.csv",
    )
    write_markdown_artifacts(metrics, leakage_audit, split_summary)
    return metrics


if __name__ == "__main__":
    trained_metrics = train()
    print(
        json.dumps(
            {
                "status": "trained",
                "rows": trained_metrics["dataset"]["rows"],
                "split": trained_metrics["split"],
                "baseline_test": trained_metrics["baseline_model"]["test"],
                "final_test": trained_metrics["final_model"]["test"],
                "leakage_audit_passed": trained_metrics["leakage_audit"]["passed"],
            },
            indent=2,
        )
    )
