# Demo Video Script

## Setup

1. Open a terminal.
2. Run:

```bash
cd SeatSenseAI
pip install -r requirements.txt
streamlit run app.py
```

3. Open the local Streamlit URL shown in the terminal.

## Demo Flow

### 1. Executive Overview

Narration: “This is SeatSense AI, a dynamic pricing intelligence product for sports venues. The landing page frames the business problem: static prices fail when demand changes quickly, causing unsold seats or underpriced inventory that scalpers monetize.”

Highlight: synthetic training rows, final model F1, revenue uplift, leakage reduction.

### 2. Model Performance Page

Click: “Model Performance.”

Narration: “The project uses a 20,500-row game-section-pricing-window dataset. The model portfolio includes a simple baseline, a tuned random forest demand classifier, a sell-through regressor, and a scalper-risk classifier. The dashboard shows the time-based train/validation/test split, leakage audit, accuracy, precision, recall, F1, ROC-AUC, confusion matrix, and feature importance.”

Highlight: high-demand precision and recall.

Highlight: the dataset drill-down widgets. Show the demand mix donut, current-vs-resale price scatter, signal bubble chart, and section-level drill-down. Explain that this is how a revenue manager inspects whether the training data and model behavior make business sense.

Optional client-data moment: open the “Import client CSV and retrain model” panel. Explain that a client does not need to rename every column to the SeatSense schema. They can upload a CSV, map common fields like event ID, date, section, price, capacity, tickets sold, sell-through, resale price, traffic, and opponent context, then validate before retraining. Ticketmaster, SeatGeek, or StubHub-style files are treated as agency market-signal feeds unless they are joined with historical venue outcomes. After mapping, a Pydantic schema gate validates the converted SeatSense rows before training.

Explain: “A false positive high-demand prediction can overprice seats and hurt attendance. A false negative can underprice a popular game and shift profit to secondary markets.”

### 3. Demand Intelligence Page

Click: “Demand Intelligence.”

Inputs to show:
- Seat section: Lower
- Day: Saturday
- Opponent strength: 94
- Rivalry game: on
- Star available: on
- Traffic index: high
- Secondary market average: above current price

Narration: “The model predicts demand tier and provides a probability distribution, so the revenue manager can see confidence rather than just a hard label.”

Highlight: key demand drivers table.

### 4. Pricing Workbench

Click: “Pricing Workbench.”

Narration: “The recommendation engine converts model output into a business action. It compares current primary price, resale-market price, and AI-recommended price. It estimates revenue uplift and secondary-market leakage reduction.”

Highlight:
- Current price
- AI recommended price
- Revenue uplift
- Approval required flag
- Generated business explanation

Explain: “The explanation is grounded in the prediction, probability, resale gap, and guardrails. The Pricing Workbench can enrich scenario inputs with live APIs such as ticket marketplace, weather, sports, social, and analytics signals when those providers are configured. OpenAI is used only after the model and pricing engine finish, and only to write the executive explanation. If OpenAI is unavailable, the deterministic fallback still works.”

### 5. Responsible AI Page

Click: “Responsible AI.”

Narration: “Dynamic pricing can reduce trust if it is opaque or unfair. SeatSense includes price caps, affordable ticket reserves, no direct protected attributes, human approval for premium games or large changes, and fairness monitoring by affordability segment.”

Highlight:
- Fairness audit table
- Value bucket cap
- Failure modes and mitigations

### 6. Revenue Impact Page

Click: “Revenue Impact.”

Narration: “Finally, the business impact page translates the model into executive KPIs: revenue per game, total ticket revenue uplift, sell-through rate, resale gap, leakage reduction, fan affordability score, and approval rate.”

Closing: “SeatSense AI is designed as decision support. It helps teams retain more value from high-demand games, protect attendance for low-demand games, and keep humans accountable for high-impact pricing decisions.”
