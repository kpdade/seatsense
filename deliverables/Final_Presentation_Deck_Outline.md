# SeatSense AI Executive Deck Outline

## Slide 1: Title Slide
- SeatSense AI
- Dynamic Ticket Pricing & Demand Intelligence for Sports Venues
- Enterprise revenue intelligence for ticketing teams and venue operators
- Product strategy and operating model
Speaker notes: Introduce the product as a practical AI revenue-management system for teams and venues.
Suggested visual: Clean product title with ticket, arena, and price-signal motif.
Demo: None.

## Slide 2: Business Problem
- Static pricing misses fast-changing demand signals.
- Low-demand games leave seats unsold.
- High-demand games are underpriced.
- Secondary markets capture value that should stay with the venue.
Speaker notes: Frame tickets as perishable inventory with both revenue and fan-experience consequences.
Suggested visual: Primary price, resale price, and unsold-seat gap diagram.
Demo: None.

## Slide 3: Market Context and Existing Customers
- Customers: professional teams, arenas, and ticketing departments.
- Users: revenue managers, operations, marketing, fan experience, executives.
- Existing market already values dynamic pricing, but trust and governance are gaps.
- SeatSense focuses on explainable decision support.
Speaker notes: Make clear that this is not a made-up market; sports ticketing departments already pay for revenue tools.
Suggested visual: Stakeholder map.
Demo: None.

## Slide 4: Who, Why, How Framework
- Who: teams, venues, fans, executives, ticketing staff.
- Why: demand is multivariate, volatile, and financially material.
- How: prediction model plus governed pricing engine plus explanation layer.
- Human approval stays in the workflow.
Speaker notes: Use this slide to connect business strategy to AI implementation.
Suggested visual: Three-column Who/Why/How framework.
Demo: None.

## Slide 5: Why AI Is the Right Solution
- Rules miss interaction effects among opponent, timing, weather, sentiment, and resale gap.
- Classification predicts Low, Medium, or High demand.
- Probabilities create confidence and approval thresholds.
- Generative explanation turns model output into revenue-manager language using OpenAI when a key is available and a deterministic fallback when it is not.
Speaker notes: Emphasize that the product is not just an API wrapper; it includes trained supervised models.
Suggested visual: Signal-to-decision flow.
Demo: None.

## Slide 6: Product Overview: SeatSense AI
- Upload sample data or map a client CSV export.
- Predict demand tier.
- Recommend guarded price change.
- Estimate revenue uplift and resale leakage reduction.
- Explain human approval requirement.
Speaker notes: Introduce the app pages and workflow.
Suggested visual: Product navigation screenshot placeholder.
Demo: Open landing page.

## Slide 7: Data Pipeline and Feature Design
- Semi-synthetic 20,500-row game-section-pricing-window dataset.
- Five seasons, 410 games, 10 seat sections, five pricing windows.
- Features include current inventory, sell-through, traffic, search, social, weather, price, resale, section, and historical behavior.
- Leakage audit excludes post-event outcomes and fairness-only affordability columns.
- Ticketing-agency adapter maps vendor feeds, then Pydantic validates the SeatSense schema before retraining.
Speaker notes: Explain why synthetic data is used and why the generator is transparent.
Suggested visual: Data pipeline diagram.
Demo: Show dataset overview on Model Performance page.

## Slide 8: Model Architecture
- Baseline: simple decision tree.
- Final demand model: tuned random forest classifier.
- Sell-through model: random forest regression.
- Scalper-risk model: random forest classifier.
- Time-based split: first 3 seasons train, 4th validation, 5th test.
- Pricing optimizer simulates candidate prices and applies guardrails.
Speaker notes: Distinguish predictive model from pricing rules and guardrails.
Suggested visual: Model architecture stack.
Demo: None.

## Slide 9: Model Performance
- Final demand accuracy about 86.7%, macro F1 about 86.1%, ROC-AUC about 96.9%.
- High-demand recall about 89.8%, which protects against underpricing/scalper leakage.
- Sell-through RMSE about 0.059; scalper-risk macro F1 about 88.2%.
- Confusion matrix shows where pricing mistakes occur.
- Feature importance and leakage audit support explainability.
- Drill-down widgets include demand mix, price gap scatter, signal bubble, and section analysis.
Speaker notes: Discuss false positives and false negatives in business terms.
Suggested visual: Confusion matrix and feature importance.
Demo: Show Model Performance page.

## Slide 10: Pricing Recommendation Workflow
- Input current price, demand tier, probability, resale gap, days before game, section, sell-through.
- Optional live APIs enrich resale, weather, sports form, social buzz, and traffic signals.
- Apply price caps and affordability guardrails.
- Estimate revenue uplift and leakage reduction.
- Flag human approval for high-impact decisions.
Speaker notes: Explain how model output becomes a controlled business action.
Suggested visual: Pricing decision flow.
Demo: Show Pricing Workbench page.

## Slide 11: Streamlit Product Demo Screens
- Executive Overview frames the problem and KPIs.
- Pricing Workbench page shows live API signal mode, probabilities, recommended price, and approval logic.
- Model Performance page shows flexible CSV import, agency-feed detection, schema validation, and data quality charts.
- Demand Intelligence page explains drivers behind the forecast.
Speaker notes: Walk the audience through the app as a real product, not a notebook.
Suggested visual: Four-screen montage placeholder.
Demo: High-demand rivalry game.

## Slide 12: Business Impact and KPIs
- Revenue per game.
- Total ticket revenue uplift.
- Sell-through and attendance rate.
- Secondary-market leakage reduction.
- Fan affordability score and approval rate.
Speaker notes: Connect the model to CFO and revenue-manager outcomes.
Suggested visual: KPI dashboard and scenario bars.
Demo: Show Revenue Impact page.

## Slide 13: Responsible AI and Fairness Guardrails
- Lower-income fans may be priced out without controls.
- No protected attributes used directly.
- Affordable ticket reserve and value-section caps.
- Human review for premium games and large price moves.
- Bias audit by affordability segment.
Speaker notes: Show that the product addresses trust and fairness as first-class requirements.
Suggested visual: Guardrail checklist and audit table.
Demo: Show Responsible AI page.

## Slide 14: Implementation Roadmap
- Phase 1: historical data integration and offline validation.
- Phase 2: pilot with revenue-manager review.
- Phase 3: controlled A/B testing by section and game type.
- Phase 4: production monitoring for drift, fairness, and fan sentiment.
Speaker notes: Explain how the concept would become an enterprise deployment.
Suggested visual: Four-phase roadmap.
Demo: None.

## Slide 15: Conclusion and Ask
- SeatSense captures revenue that static pricing misses.
- It reduces resale leakage while protecting attendance.
- It combines trained predictive AI with explainable generative recommendations and safe fallback behavior.
- Ask: approve a pilot using real historical venue data.
Speaker notes: End with a crisp business ask and responsible implementation stance.
Suggested visual: Closing product promise.
Demo: None.
