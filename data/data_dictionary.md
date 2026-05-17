# SeatSense AI Data Dictionary

SeatSense AI uses a semi-synthetic dataset because private venue ticketing,
resale, web analytics, and CRM data are not publicly available. The generator is
transparent and designed to simulate realistic sports ticketing dynamics without
claiming to represent a real team.

## Observation Unit

Each modeling row represents:

`game_id + seat_section + pricing_window`

Default dataset size:

- 5 seasons
- 82 games per season
- 10 seat sections
- 5 pricing windows
- 20,500 model training rows

## Reference Logic Used For Simulation

The generator is calibrated from public business concepts rather than a copied
dataset:

- Dynamic pricing in sports and live events, including demand-based price moves.
- Primary vs secondary ticket-market economics and resale leakage.
- Venue revenue management concepts such as sell-through, inventory scarcity,
  price elasticity, premium games, and promotion timing.
- Publicly discussed demand signals: opponent quality, team form, day of week,
  star-player availability, weather, search interest, social buzz, and website
  traffic.
- Responsible AI guidance for algorithmic pricing, transparency, price caps,
  human review, and fairness monitoring.

Replace these placeholders with verified citations in the final paper:

- [Source needed: NBA dynamic pricing]
- [Source needed: Ticketmaster dynamic pricing]
- [Source needed: secondary ticket market research]
- [Source needed: algorithmic pricing fairness governance]

## Model Feature Columns

These columns are known before or at the pricing decision and are eligible model
inputs.

| Field | Type | Description |
|---|---:|---|
| `season` | Numeric | Season year used for time-based validation. |
| `game_id` | Text | Unique generated game identifier. Not used as a model feature. |
| `game_date` | Date | Scheduled event date. Not used as a model feature. |
| `day_of_week` | Text | Game day. |
| `weekend_flag` | Binary | 1 for Friday, Saturday, or Sunday. |
| `holiday_flag` | Binary | 1 for major holiday-style dates. |
| `rivalry_flag` | Binary | 1 for high-interest rivalry matchups. |
| `division_matchup_flag` | Binary | 1 for divisional matchups. |
| `opponent` | Text | Visiting team. |
| `opponent_strength` | Numeric | 0-100 opponent quality and brand-draw proxy. |
| `home_team_win_rate` | Numeric | Home team win rate at pricing time. |
| `away_team_win_rate` | Numeric | Opponent win rate at pricing time. |
| `home_team_recent_form` | Numeric | Recent home team form. |
| `away_team_recent_form` | Numeric | Recent opponent form. |
| `playoff_implication_score` | Numeric | 0-1 importance of the game to standings or playoff push. |
| `premium_game_flag` | Binary | 1 for rivalry, elite opponent, holiday, or playoff-style games. |
| `seat_section` | Text | One of 10 generated venue seat sections. |
| `section_capacity` | Numeric | Seat count for the section. |
| `inventory_remaining` | Numeric | Seats still available at the pricing window. |
| `inventory_remaining_pct` | Numeric | Inventory remaining divided by section capacity. |
| `current_sell_through_rate` | Numeric | Current sold share at the pricing window. |
| `seat_quality_score` | Numeric | 0-1 section quality score. |
| `base_ticket_price` | Numeric | Base price before dynamic adjustments. |
| `current_ticket_price` | Numeric | Current primary-market price. |
| `price_change_from_base_pct` | Numeric | Current price relative to base price. |
| `days_before_game` | Numeric | Days remaining at the pricing window. |
| `pricing_window` | Text | 30 days, 14 days, 7 days, 2 days, or day of game. |
| `days_until_game_bucket` | Text | Binned days-until-game category. |
| `month` | Numeric | Calendar month. |
| `season_stage` | Text | Early, mid, late, or playoff-push season stage. |
| `website_traffic_index` | Numeric | Aggregated ticketing-site traffic index. |
| `search_interest_index` | Numeric | Search demand signal. |
| `social_sentiment_score` | Numeric | -1 to 1 social sentiment signal. |
| `social_volume_index` | Numeric | Social conversation volume index. |
| `email_campaign_active` | Binary | Whether marketing email is active. |
| `promotion_flag` | Binary | Whether a promotion is active. |
| `marketing_spend_index` | Numeric | Relative marketing spend signal. |
| `star_player_available` | Binary | Whether marquee player is expected to play. |
| `injury_news_severity` | Numeric | 0-1 severity of injury-related news. |
| `weather_severity` | Numeric | 0-1 friction from weather. |
| `temperature_score` | Numeric | 0-1 event-day comfort proxy. |
| `demand_signal_score` | Numeric | Composite pre-decision demand signal from traffic, search, social, resale, inventory, and game context. |
| `secondary_market_avg_price` | Numeric | Current secondary-market average listing price. |
| `secondary_market_listing_count` | Numeric | Current secondary listings count. |
| `resale_price_gap_pct` | Numeric | Secondary-market average price relative to primary price. |
| `resale_velocity_index` | Numeric | Current secondary-market sales/listing velocity proxy. |
| `scalper_pressure_score` | Numeric | Composite resale-pressure score. |
| `rolling_3_game_attendance_rate` | Numeric | Historical attendance rate before pricing decision. |
| `rolling_5_game_sell_through_rate` | Numeric | Historical sell-through before pricing decision. |
| `previous_similar_game_sell_through` | Numeric | Historical sell-through for similar games. |
| `same_opponent_last_season_attendance` | Numeric | Historical attendance for same opponent. |
| `section_historical_fill_rate` | Numeric | Historical fill rate by section. |
| `historical_price_elasticity_section` | Numeric | Section-level elasticity estimate. |
| `historical_attendance_rate` | Numeric | Compatibility alias for historical attendance behavior. |

## Fairness-Only Columns

These columns are not used for demand, sell-through, or scalper-risk model
training. They are used only for responsible AI monitoring and guardrails.

| Field | Type | Description |
|---|---:|---|
| `affordability_segment` | Text | Ticket-product affordability segment. |
| `affordability_index` | Numeric | Synthetic affordability proxy for audit and guardrails only. |
| `local_income_proxy` | Numeric | Backward-compatible alias; excluded from model features. |

## Target And Post-Event Outcome Columns

These columns must never be model input features.

| Field | Type | Description |
|---|---:|---|
| `final_sell_through_rate` | Numeric | Final sold share after event. Regression target. |
| `final_attendance_rate` | Numeric | Final attendance rate. |
| `realized_revenue` | Numeric | Post-event realized ticket revenue. |
| `revenue_per_available_seat` | Numeric | Post-event revenue efficiency. |
| `unsold_inventory_pct` | Numeric | Final unsold share. |
| `actual_secondary_market_gap` | Numeric | Post-event secondary-market gap. |
| `realized_demand_score` | Numeric | Latent demand score used to create labels. |
| `demand_tier` | Text | Classification target: Low, Medium, or High. |
| `scalper_risk_tier` | Text | Scalper-risk target: Low, Medium, or High. |
| `optimal_price_oracle` | Numeric | Post-event oracle benchmark used only for evaluation context. |
| `recommended_price_target` | Numeric | Legacy benchmark target; not used as a model feature. |

## Leakage Rule

`src/leakage_audit.py` checks every training run and fails if any forbidden
post-event, target, recommended-price, or fairness-only column appears in
`FEATURE_COLUMNS`.
