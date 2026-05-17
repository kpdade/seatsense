# SeatSense AI User Guide

## 1. Purpose Of The App

SeatSense AI is a dynamic ticket-pricing intelligence app for sports venues. It helps a revenue manager forecast demand, compare current pricing against AI-recommended pricing, estimate revenue impact, and decide whether a pricing action needs human approval.

The app is designed for:

- Venue revenue managers
- Ticketing operations teams
- Sports executives
- Marketing and fan experience teams
- Business stakeholders evaluating pricing operations and revenue impact

The product combines three layers:

1. Predictive models that estimate demand, sell-through, and scalper risk.
2. A pricing optimizer that applies business rules and responsible AI guardrails.
3. An explanation layer that turns model outputs into a revenue-manager briefing.

## 2. First-Time App Walkthrough

Use this page order for the clearest first experience:

1. Executive Overview
2. Model Performance
3. Pricing Workbench
4. Demand Intelligence
5. Revenue Impact
6. Responsible AI

This flow starts with the business problem, proves that a model exists, then shows the product workflow and governance controls.

## 3. Executive Overview

### 3.1 What To Look At

Start here to understand the business case. The page highlights:

- Revenue uplift opportunity
- Sell-through improvement
- Secondary-market leakage reduction
- Human-reviewed pricing guardrails

### 3.2 How To Interpret It

**Revenue uplift opportunity** estimates how much additional ticket revenue could be retained through smarter price recommendations.

**Sell-through improvement** shows whether recommendations may reduce unsold inventory risk.

**Secondary-market leakage reduction** estimates how much value could stay with the venue instead of shifting to resale markets.

**Human-reviewed guardrails** shows that the system is decision support, not fully autonomous pricing.

## 4. Model Performance

The Model Performance page explains what the system was trained on, how it was evaluated, and whether the model is credible.

### 4.1 Default Training Data

The default dataset uses game-section-pricing-window observations.

Current default structure:

- 20,500 training rows
- 5 seasons
- 410 games
- 10 seat sections
- 5 pricing windows per game-section

Each row represents:

```text
game_id + seat_section + pricing_window
```

This is stronger than one row per game because it lets the model learn how demand changes by seat location and timing.

### 4.2 Train / Validation / Test Split

The app currently uses:

```text
Train:      2021, 2022, 2023
Validation: 2024
Test:       2025
```

This is a time-based split. It is appropriate because the real business use case is training on past seasons and predicting future games.

Why this is useful:

- It avoids randomly mixing the same game patterns across train and test.
- It better simulates a real deployment.
- It reveals whether the model generalizes to future seasons.

What to watch:

- If future fan behavior changes, performance may drift.
- The model should be monitored and retrained as new seasons arrive.

### 4.3 Key Metrics

**Accuracy** measures overall correct demand-tier predictions.

**Macro F1** balances performance across Low, Medium, and High demand classes.

**High-demand recall** measures how well the model catches high-demand games.

**ROC-AUC** measures how well the model separates demand classes.

Business interpretation:

- False positive high demand: the model predicts high demand when demand is actually lower. Risk: overpricing, empty seats, fan frustration, and lost concession revenue.
- False negative high demand: the model misses a high-demand game. Risk: underpricing, fast sellout, scalper profit, and lost venue revenue.

### 4.4 Dataset Drill-Down Charts

Use the filters for seat section, demand tier, and pricing window.

Key widgets:

- Demand mix donut: shows the share of Low, Medium, and High demand rows.
- Price gap scatter: compares current primary price with secondary-market price.
- Signal bubble chart: shows relationships among traffic, sell-through, demand, and scalper risk.
- Section drill-down: compares sections by resale gap, demand share, and pricing behavior.

How to interpret the charts:

- A large resale gap may indicate underpricing.
- Low sell-through close to game day may indicate discount or promotion risk.
- High-demand concentration in premium sections often requires human review.
- Large increases in value-sensitive sections should trigger fairness review.

## 5. Importing A Client CSV And Retraining

Use the CSV import area on the Model Performance page when a venue has a better dataset.

### 5.1 When To Import Data

Import a CSV if you have historical ticketing data such as:

- Game or event records
- Seat section records
- Pricing history
- Capacity and inventory snapshots
- Tickets sold or sell-through
- Resale-market prices
- Demand labels or final outcomes

### 5.2 Minimum Useful Fields

The model needs enough information to learn from historical outcomes.

Most helpful fields:

- Event ID or game ID
- Game date or season
- Seat section
- Current ticket price
- Base ticket price
- Section capacity
- Tickets sold
- Tickets sold percentage
- Final sell-through rate
- Demand tier
- Secondary-market average price
- Opponent or event type
- Day of week
- Days before game
- Website traffic or demand index
- Social sentiment or buzz score

The CSV does not need to use the exact SeatSense column names. The import workflow lets the user map their own columns to SeatSense concepts.

### 5.3 Step-By-Step Import Process

1. Open the Model Performance page.
2. Find the client CSV import and retraining section.
3. Upload a CSV file.
4. Review the detected feed type. Ticketing-agency or marketplace feeds are recognized separately from venue training exports.
5. Review the auto-detected column mappings.
6. Manually correct mappings where needed.
7. Preview the adapted training dataset.
8. Review validation warnings and the Pydantic schema result.
9. If the data passes validation, run retraining.
10. Review the updated metrics, leakage audit, and split summary.

### 5.4 What Happens With A Large Dataset

For a large file, such as 100,000 rows:

- If date or season fields exist, the app uses a chronological split.
- Older data trains the model.
- The next period validates model selection.
- The newest period tests final performance.

If dates are missing but event IDs exist, the app uses a group-aware split so rows from the same event do not appear in both training and testing.

### 5.5 Important Import Notes

Do not use future or post-event outcomes as model inputs. Final attendance, realized revenue, final sell-through, oracle price, and demand labels are targets or evaluation fields, not model features.

Affordability-related fields should be used only for monitoring and guardrails. They should not be used to maximize price.

Ticketing-agency feeds such as Ticketmaster, SeatGeek, or StubHub-style exports may include event IDs, event dates, market prices, listing counts, popularity scores, and provider IDs. SeatSense can use those as event and market signals, but it will not treat them as complete supervised training data unless historical sales outcomes are also available.

The app uses a Pydantic schema gate after mapping. This means the uploaded file can have agency-specific column names, but the converted SeatSense training rows must pass type, range, label, and required-field validation before retraining starts.

CRM and demographic-style fields such as customer ID, age group, income band, ZIP/geography, loyalty tier, membership status, and historical spend are blocked from model-training concepts. They may be used only for fairness review, segmentation monitoring, or governance analysis.

## 6. Pricing Workbench

The Pricing Workbench is the main interactive page.

### 6.1 Choose A Scenario

Start with:

```text
High-demand rivalry game
```

Then compare it against:

```text
Low-demand weekday game
```

Other useful scenarios:

- Star player injured
- Bad weather game
- Premium playoff-style game

### 6.2 Adjust Scenario Inputs

Try changing:

- Day of week
- Rivalry flag
- Star player availability
- Premium game flag
- Days before game
- Opponent strength
- Website traffic
- Social sentiment
- Current ticket price
- Secondary-market price
- Seat section

### 6.3 Read The Outputs

The main outputs are:

- Demand tier
- Confidence score
- Scalper risk
- Current price
- Recommended price
- Price change percentage
- Revenue impact
- Sell-through estimate
- Leakage reduction
- Human approval flag

### 6.4 How To Interpret Pricing Actions

**Increase price** means demand and market signals support a higher price, subject to caps and approval rules.

**Hold price** means signals are balanced or uncertainty is too high for a large move.

**Discount or promotion** means sell-through risk is high and lower prices or marketing support may be needed.

**Human approval required** means the recommendation has higher business, fairness, or fan-trust impact.

Common reasons for human approval:

- Premium game
- Large price increase
- Affordability risk
- Lower model confidence
- Aggressive action under high scalper risk

## 7. OpenAI Analyst Explanation

The Pricing Workbench includes two explanation modes:

- Fast template
- OpenAI analyst

### 7.1 Fast Template

Use Fast template when you want instant output or when no OpenAI key is configured.

It still uses the actual model prediction and pricing result.

### 7.2 OpenAI Analyst

Use OpenAI analyst when Streamlit Secrets include:

```toml
OPENAI_API_KEY = "your_real_key_here"
OPENAI_MODEL = "gpt-4.1-mini"
OPENAI_TIMEOUT_SECONDS = 30
```

The OpenAI explanation is generated after the model and pricing engine finish. It does not decide the price.

### 7.3 What The Explanation Covers

The briefing explains:

- Executive summary
- Recommended pricing action
- Key demand drivers
- Expected business impact
- Risks and guardrails
- Human review decision

If the OpenAI call fails, the deterministic fallback remains active so the demo still works.

## 8. Demand Intelligence

Use this page to understand why the model made its demand prediction.

### 8.1 Key Signals

Important demand signals include:

- Opponent strength
- Website traffic
- Search interest
- Social sentiment
- Secondary-market price gap
- Star player availability
- Rivalry flag
- Weather severity
- Days before game

### 8.2 How To Read The Page

Look at the demand probability chart first. A high probability for one class means the model is more confident.

Then review top drivers. These show the business signals most associated with the prediction.

Example interpretations:

- Strong opponent plus high traffic may push demand higher.
- Bad weather or star injury may reduce demand.
- A large resale gap may indicate underpricing.
- Low current sell-through close to game day may indicate unsold inventory risk.

## 9. Revenue Impact

This page translates model outputs into business value.

### 9.1 What To Review

Look for:

- Current revenue vs optimized revenue
- Conservative, expected, and aggressive scenarios
- Revenue uplift
- Secondary-market leakage reduction
- Sell-through impact
- Human approval rate

### 9.2 How To Interpret It

Executives care less about model accuracy by itself and more about business outcomes.

Use this page to answer:

- How much revenue could the venue retain?
- How much leakage could be reduced?
- Would attendance remain healthy?
- How often would humans need to approve recommendations?

## 10. Responsible AI

Dynamic pricing can create fairness and trust risks. This page explains how SeatSense manages those risks.

### 10.1 Risks Covered

The page covers:

- Affordability risk
- Geographic or income-proxy bias
- Fan trust risk
- Privacy risk
- Overpricing risk
- Overreaction to noisy signals

### 10.2 Guardrails

SeatSense uses:

- Price increase caps
- Affordable ticket reserve logic
- Human approval for premium or high-impact moves
- Manual override
- Monitoring by affordability segment
- No direct protected attributes in model training
- Aggregated traffic and sentiment signals

### 10.3 How To Explain It

SeatSense is not designed to maximize every seat price at any cost. It is designed to support revenue managers with explainable recommendations, approval gates, and fairness monitoring.

## 11. Recommended First-Time Practice Flow

Use this 10-minute practice flow:

1. Open Executive Overview.
2. Explain the static-pricing problem.
3. Go to Model Performance.
4. Show the 20,500-row dataset, split strategy, leakage audit, and metrics.
5. Open Pricing Workbench.
6. Run the high-demand rivalry scenario.
7. Review demand, recommended price, revenue impact, and approval flag.
8. Switch to low-demand weekday game.
9. Compare how the recommended action changes.
10. Open Responsible AI.
11. Explain price caps, human approval, and affordability monitoring.

## 12. Quick Interpretation Cheat Sheet

| Signal Or Output | What It Usually Means |
|---|---|
| High demand | Price increase may be justified, but check approval and affordability flags. |
| Low demand | Discount, promotion, or marketing action may be needed. |
| High resale price gap | Venue may be underpricing relative to secondary market. |
| Low sell-through close to game day | Unsold inventory risk is high. |
| High scalper risk | Underpricing leakage is likely. |
| Human approval required | Recommendation is high-impact or needs governance review. |
| Affordability risk flag | Preserve value inventory and review price movement manually. |
| Low model confidence | Treat recommendation as decision support, not automatic approval. |

## 13. Troubleshooting

### 13.1 The App Shows Template Fallback

This means no OpenAI key is visible to the running app, or the OpenAI API call failed.

For Streamlit Cloud, add secrets under app settings:

```toml
OPENAI_API_KEY = "your_real_key_here"
OPENAI_MODEL = "gpt-4.1-mini"
OPENAI_TIMEOUT_SECONDS = 30
```

Then reboot the app.

### 13.2 Uploaded CSV Does Not Validate

Common reasons:

- Too few rows
- Missing historical outcome field
- Only one demand class
- No event or game identifier
- Price or capacity columns are not numeric

Fix the column mapping or add the missing historical outcome fields.

### 13.3 Metrics Look Too Good Or Too Weak

Very high metrics can indicate leakage. Very weak metrics can indicate missing demand signals or poor target quality.

Check:

- Leakage audit
- Train/validation/test split
- Demand tier distribution
- Feature importance
- Dataset preview

## 14. Best Demo Message

Use this short summary when explaining the app:

SeatSense AI helps sports venues avoid two pricing failures: underpricing high-demand games and overpricing low-demand games. It predicts demand, recommends guarded price actions, estimates revenue impact, explains the recommendation, and keeps humans in the loop for high-impact or fairness-sensitive decisions.
