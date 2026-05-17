# SeatSense AI Model Card

## Intended Use
SeatSense AI is a decision-support prototype for sports venue revenue managers. It predicts demand tier, sell-through, and scalper risk, then recommends constrained ticket price actions.

## Not Intended Use
The model should not autonomously publish prices, target individual fans, or use protected demographics to maximize willingness to pay.

## Model Inputs
Inputs include game context, seat section, current price, current inventory, pricing window, aggregated demand signals, secondary-market indicators, and historical venue behavior.

## Excluded Sensitive or Post-Event Features
Affordability columns are excluded from model training and used only for fairness monitoring and guardrails. Post-event outcomes such as final attendance, final sell-through, realized revenue, target labels, and oracle prices are forbidden model inputs.

## Validation Design
Time-based split by season; all rows for the same game_id remain in one split. Train seasons: [2021, 2022, 2023]; validation season: [2024]; test season: [2025].

## Metrics
- Final demand accuracy: 86.7%
- Final demand macro F1: 86.1%
- High-demand recall: 89.8%
- ROC-AUC OVR: 96.9%
- Sell-through RMSE: 0.059
- Scalper-risk macro F1: 88.2%

## Human Review Rules
Human review is required for premium games, price increases above 20%, affordability guardrail flags, low-confidence predictions, aggressive high-scalper-risk actions, or guardrail-capped recommendations.

## Limitations
Because private ticketing transactions are not publicly available, this prototype uses a semi-synthetic dataset calibrated to realistic sports ticketing dynamics. Real deployment would require historical transaction feeds, ticketing system integration, privacy review, and ongoing monitoring.
