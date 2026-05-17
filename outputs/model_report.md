# SeatSense AI Model Report

## Data Upgrade
The training architecture now uses game-section-pricing-window observations rather than one row per game section. This increases the default training dataset to 20,500 rows while preserving a clear distinction between known pre-pricing features and post-event outcomes.

## Validation Strategy
The model uses a time-based split:
- Train: 12,300 rows across 246 games
- Validation: 4,100 rows across 82 games
- Test: 4,100 rows across 82 games

This is stronger than a random row split because all rows for a game stay in the same season-based split.

## Baseline Vs Final Demand Model
- Baseline accuracy: 73.7%
- Baseline macro F1: 54.0%
- Final accuracy: 86.7%
- Final macro F1: 86.1%
- Final high-demand recall: 89.8%
- Final ROC-AUC OVR: 96.9%

## Business Error Analysis
False positive high demand means the model predicts high demand when actual demand is lower. The business risk is overpricing, empty seats, fan frustration, and lost concession revenue.

False negative high demand means the model misses high-demand inventory. The business risk is underpricing, rapid sellout, scalper profit, and lost venue revenue.

## Pricing Simulation
- Estimated revenue uplift: $18,931,567
- Estimated sell-through improvement: 1.5%
- Human approval rate: 52.8%
- Guardrail cap trigger rate: 0.2%

## Responsible AI
No protected demographics are used as model features. The affordability index is used only for audit, approval guardrails, and affordable inventory reserve simulation.
