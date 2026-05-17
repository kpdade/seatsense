# SeatSense AI: Dynamic Ticket Pricing & Demand Intelligence

SeatSense AI is a Streamlit revenue-intelligence product for sports teams and venues. It helps operators predict game demand, estimate pricing risk, recommend guarded ticket price changes, and explain decisions in business language.

## Business Problem

Sports venues lose revenue because static ticket prices do not adapt to real-time demand signals. Low-demand games leave seats unsold, while high-demand games are often underpriced and allow scalpers or secondary marketplaces to capture value that should have gone to the team or venue.

Primary customers include professional sports teams, arenas, ticketing departments, and venue revenue managers. End users include revenue operations, ticketing, marketing, fan experience, and executive teams.

## AI Approach

SeatSense intentionally combines predictive AI and generative AI:

- Supervised classification predicts demand tier: Low, Medium, or High.
- A simple decision-tree baseline is compared with a tuned random forest demand classifier.
- A sell-through regressor estimates final inventory conversion.
- A scalper-risk classifier estimates resale leakage pressure.
- The pricing optimizer simulates candidate prices, estimates expected revenue, and applies business constraints before recommending an action.
- The generative explanation layer creates executive-ready pricing explanations using the actual prediction, confidence, recommended price, revenue impact, leakage estimate, feature drivers, and approval flag. The app renders an instant template-based briefing for demo speed and can generate an OpenAI Responses API analyst briefing when `OPENAI_API_KEY` is available. Without a key, it stays fully functional with the deterministic fallback.

## Data and API Use

The app includes an optional live API signal mode. When provider credentials are configured, SeatSense can call Ticketmaster, SeatGeek, StubHub-compatible listing endpoints, Open-Meteo weather, NBA scoreboard data, X/Reddit social search, and Plausible/custom website analytics to enrich the scenario inputs before pricing. These signals can update values such as secondary-market price, weather severity, opponent strength, social sentiment, and website traffic.

The final recommendation still comes from the trained local demand model and deterministic pricing engine. OpenAI is used only after the model/pricing result exists to generate executive explanation text; it does not fetch market data, train the model, or decide the price. The app remains functional when live providers or OpenAI are not configured.

The semi-synthetic generator is rule-based and transparent. It does not copy a real team or ticketing dataset. Its assumptions are informed by public market context: sports dynamic pricing descriptions, ticketing platform documentation, secondary-market/scalping research, venue revenue management concepts, and algorithmic pricing ethics research.

## Data and Model Quality

The default training data now uses a game-section-pricing-window design:

- 20,500 training rows
- 5 seasons
- 410 unique games
- 10 seat sections
- 5 pricing windows per game-section
- Time-based split: train on the first 3 seasons, validate on the 4th season, test on the 5th season
- Group-safe split: all rows for a `game_id` stay in one split

The trained demand model is intentionally strong but not perfect. Current metrics are approximately:

- Baseline accuracy: 73.7%
- Final demand accuracy: 86.7%
- Final macro F1: 86.1%
- High-demand recall: 89.8%
- ROC-AUC OVR: 96.9%
- Sell-through RMSE: 0.059
- Scalper-risk macro F1: 88.2%

The leakage audit passes and confirms that target/post-event columns are not used as model features.

## Install

```bash
cd SeatSenseAI
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Optional OpenAI API Key

SeatSense AI works without an OpenAI API key. In that mode, the explanation panel uses a polished deterministic template. With a key configured, the Pricing Workbench exposes an OpenAI analyst mode that uses the official OpenAI Responses API without exposing the key in the UI.

### Local `.env`

Create a local `.env` file from the example:

```bash
cp .env.example .env
```

Edit `.env`:

```bash
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4.1-mini

TICKETMASTER_API_KEY=your_ticketmaster_api_key_here
SEATGEEK_CLIENT_ID=your_seatgeek_client_id_here
SEATGEEK_CLIENT_SECRET=your_seatgeek_client_secret_here
STUBHUB_ACCESS_TOKEN=your_stubhub_access_token_here
STUBHUB_API_URL=https://api.stubhub.net/example/event-or-listings-endpoint

X_BEARER_TOKEN=your_x_bearer_token_here
PLAUSIBLE_API_TOKEN=your_plausible_api_token_here
PLAUSIBLE_SITE_ID=yourdomain.com
WEBSITE_ANALYTICS_API_URL=https://your-analytics-endpoint.example.com/traffic.json
```

Never commit `.env`; it is ignored by `.gitignore`.

### Streamlit Secrets

For Streamlit deployment, create `.streamlit/secrets.toml` from the example:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Edit `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "your_api_key_here"
OPENAI_MODEL = "gpt-4.1-mini"

TICKETMASTER_API_KEY = "your_ticketmaster_api_key_here"
SEATGEEK_CLIENT_ID = "your_seatgeek_client_id_here"
SEATGEEK_CLIENT_SECRET = "your_seatgeek_client_secret_here"
STUBHUB_ACCESS_TOKEN = "your_stubhub_access_token_here"
STUBHUB_API_URL = "https://api.stubhub.net/example/event-or-listings-endpoint"

X_BEARER_TOKEN = "your_x_bearer_token_here"
PLAUSIBLE_API_TOKEN = "your_plausible_api_token_here"
PLAUSIBLE_SITE_ID = "yourdomain.com"
WEBSITE_ANALYTICS_API_URL = "https://your-analytics-endpoint.example.com/traffic.json"
```

Never commit `.streamlit/secrets.toml`; it is ignored by `.gitignore`.

### Credential Priority

The app checks credentials in this order:

1. `OPENAI_API_KEY` environment variable or `.env`
2. `st.secrets["OPENAI_API_KEY"]`
3. Template fallback explanation

## Run the App

```bash
streamlit run app.py
```

The app automatically generates data and trains the model if required artifacts are missing.

## Regenerate Data

```bash
python data/generate_ticket_data.py
```

This creates:

- `data/games.csv`
- `data/ticket_sales.csv`
- `data/secondary_market.csv`
- `data/model_training_data.csv`

## Train Models

```bash
python -m src.train_model
```

This creates:

- `models/demand_model.pkl`
- `models/demand_classifier.pkl`
- `models/sellthrough_regressor.pkl`
- `models/scalper_risk_classifier.pkl`
- `models/preprocessor.pkl`
- `outputs/model_metrics.json`
- `outputs/model_report.md`
- `outputs/model_card.md`
- `outputs/data_card.md`
- `outputs/leakage_audit.json`
- `outputs/split_summary.json`
- `outputs/pricing_simulation_summary.json`
- `outputs/confusion_matrix.png`
- `outputs/feature_importance.png`
- `outputs/fairness_audit.csv`
- `outputs/fairness_audit.json`
- `deliverables/training_dataset_snapshot.csv`

The current trained model uses `data/model_training_data.csv`, which contains game-section-pricing-window rows. The Model Performance page shows row counts, class distribution, split strategy, leakage audit, and an active dataset preview. It also includes an "Import client CSV and retrain model" mapping wizard. Client exports do not need to perfectly match the SeatSense schema; users can map common columns such as event ID, date, section, price, capacity, tickets sold, final sell-through, demand tier, resale price, traffic, and opponent context. Optional missing fields are derived or defaulted, while historical outcomes remain required for honest supervised training.

## Ticketing Agency CSV Adapter

SeatSense keeps a stable internal model schema and uses adapters for external ticketing feeds. This means Ticketmaster, SeatGeek, StubHub, CRM, or venue exports can use their own column names without forcing the model to train directly on vendor-specific schemas.

Import flow:

1. Upload a client or agency CSV in Model Performance.
2. SeatSense detects common agency fields such as `eventId`, `datetimeUtc`, `averagePrice`, `medianPrice`, `listingCount`, `eventScore`, `popularityScore`, `ticketmasterId`, `stubhubId`, and `integratedProvider`.
3. The UI auto-maps recognizable fields and lets the user review or adjust mappings.
4. The adapter converts the file into the canonical SeatSense training schema.
5. A Pydantic schema gate validates the converted rows before training.
6. Leakage and fairness-only fields are excluded from model features.
7. The model retrains only if historical outcomes are available.

Agency feeds without sales outcomes are treated as market-signal enrichment, not complete supervised training data. To retrain, the feed must be joined with historical outcomes such as final sell-through, demand tier, tickets sold by section, or realized sales.

CRM and demographic-style fields such as customer ID, age group, income band, ZIP/geography, loyalty tier, membership status, and historical spend are blocked from model-training concepts. They may be used only for governance, fairness monitoring, segmentation review, or guardrail analysis.

For large client datasets, SeatSense uses a chronological split when season or event-date fields are available: roughly the oldest 60% of seasons train, the next 20% validate, and the newest 20% test. If dates are unavailable but event IDs exist, it falls back to a group-aware game split so the same game does not appear in both train and test.

## Generate PPTX and DOCX Deliverables

```bash
python -m src.create_deliverables
```

This creates:

- `deliverables/SeatSense_AI_Final_Presentation.pptx`
- `deliverables/Project_Report.docx`

## App Pages

- Executive Overview: premium business overview, value proposition, workflow, and KPI cards
- Pricing Workbench: live scenario workspace with inputs, recommendation, revenue impact, and AI explanation
- Demand Intelligence: demand probabilities, top drivers, and market signal cards
- Revenue Impact: financial simulator, current vs optimized revenue, and leakage analysis
- Model Performance: baseline vs final model metrics, leakage audit, flexible client CSV mapping/retraining, confusion matrix, feature importance, donut/scatter/bubble charts, and drill-down widgets
- Responsible AI: governance dashboard, fairness audit, controls, and human-in-the-loop policy

## File Structure

```text
SeatSenseAI/
  README.md
  requirements.txt
  app.py
  data/
    generate_ticket_data.py
    data_dictionary.md
    model_training_data.csv
  models/
    demand_classifier.pkl
    demand_model.pkl
    sellthrough_regressor.pkl
    scalper_risk_classifier.pkl
    preprocessor.pkl
  src/
    preprocess.py
    train_model.py
    predict.py
    business_logic.py
    pricing_optimizer.py
    leakage_audit.py
    model_registry.py
    explainability.py
    fairness.py
    generative_explainer.py
    ui_components.py
    create_deliverables.py
  pages/
    1_Executive_Overview.py
    2_Live_Pricing_Demo.py
    3_Demand_Intelligence.py
    4_Revenue_Impact.py
    5_Model_Performance.py
    6_Responsible_AI.py
  outputs/
    model_metrics.json
    model_report.md
    model_card.md
    data_card.md
    leakage_audit.json
    split_summary.json
    pricing_simulation_summary.json
    fairness_audit.json
    feature_importance.png
    confusion_matrix.png
  deliverables/
    Project_Report.md
    Final_Presentation_Deck_Outline.md
    Presentation_Script_10_Minutes.md
    Demo_Video_Script.md
    training_dataset_snapshot.csv
  tests/
    test_business_logic.py
    test_prediction.py
```

## Screenshot Placeholders

Add screenshots after running the app:

- Executive Overview: `screenshots/executive_overview.png`
- Pricing Workbench: `screenshots/pricing_workbench.png`
- Demand Intelligence: `screenshots/demand_intelligence.png`
- Responsible AI: `screenshots/responsible_ai.png`

## Responsible AI Statement

SeatSense AI is designed as a decision-support system, not a fully autonomous price setter. It avoids direct protected attributes, uses aggregated demand signals, caps price increases, reserves value-priced inventory, audits affordability segments, and requires human approval for premium games, large price changes, and affordability-risk recommendations.

## Why No Direct Demographics?

Real ticketing systems may have customer-level or geography-level data, but using protected or sensitive demographics directly for price optimization can create fairness, legal, and trust risks. SeatSense AI excludes protected attributes and affordability proxies from model training. The `affordability_index` appears only in fairness audits, approval routing, price caps, and affordable ticket reserve logic.

## Known Limitations

- The dataset is synthetic because real ticketing and resale data is proprietary.
- A real deployment would require integration with historical ticket sales, CRM, ticketing inventory, and resale-market data.
- Social sentiment and website traffic signals must be aggregated and privacy-reviewed.
- Pricing decisions should remain human-reviewed for premium or high-impact games.
- The model should be continuously monitored after deployment for drift, fairness, fan sentiment, and business KPI performance.
