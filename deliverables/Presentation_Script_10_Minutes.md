# 10-Minute Team Presentation Script

## 0:00-0:45 · Speaker 1 · Opening

“Today we are presenting SeatSense AI, a dynamic ticket pricing and demand intelligence product for sports venues. The business problem is simple: tickets are perishable inventory. If a seat is empty at tipoff, that revenue is gone. If a high-demand game is underpriced, resale marketplaces and scalpers capture value that should have gone to the team.”

Transition: “We will show the business problem, the AI system, the Streamlit product, responsible AI guardrails, and the financial impact.”

## 0:45-1:45 · Speaker 1 · Business Problem

“Sports teams often rely on fixed prices or simple rules. But demand changes with opponent strength, team form, day of week, star player availability, weather, social sentiment, website traffic, and resale prices. Static pricing creates two failure modes: low-demand games have unsold seats, and high-demand games are underpriced.”

“The paying customer is a team, arena, or ticketing department. The daily users are revenue managers, ticketing operations, marketing, fan experience, and executives.”

Transition: “That is why this is a business AI problem, not just a technical classification problem.”

## 1:45-2:45 · Speaker 2 · Who, Why, How

“Our Who, Why, How framework clarifies the opportunity. Who is affected? Teams, venues, fans, executives, and secondary-market actors. Why AI? Because demand is multivariate and time-sensitive. Rules can detect obvious rivalry games, but they struggle with interactions, like a strong opponent on a bad-weather weekday or a star injury that changes sentiment.”

“How does SeatSense create value? It predicts demand tier, recommends a controlled price action, estimates revenue impact, and requires human approval when the recommendation is high risk.”

Transition: “Next, we will explain the data and model.”

## 2:45-4:15 · Speaker 2 · Data and Model

“Because real ticketing data is private, we generated a transparent semi-synthetic dataset with 20,500 game-section-pricing-window observations. It covers five seasons, 410 games, 10 seat sections, and five pricing windows. The data includes opponent strength, team form, rivalry and premium-game flags, current inventory, current sell-through, weather, social sentiment, search interest, website traffic, current price, secondary-market gap, days before game, section history, demand tier, sell-through, and scalper risk.”

“We now train a small model portfolio. The baseline is a simple decision tree. The final demand classifier is a tuned random forest. We also train a sell-through regressor and a scalper-risk classifier. Validation is time-based: the first three seasons train, the fourth validates, and the fifth tests. All rows for a game stay in the same split, so the test is more defensible than a random row split.”

“The main prediction target is demand tier: Low, Medium, or High. The final demand model reaches about 86.7 percent test accuracy, 86.1 percent macro F1, and 89.8 percent high-demand recall. The leakage audit confirms that post-event outcomes and affordability-only columns are not used as model inputs.”

“If a client wants to train on its own historical data, the Model Performance page includes a CSV mapping workflow. A revenue team can upload an export and map columns like event ID, date, section, price, capacity, tickets sold, sell-through, resale price, traffic, sentiment, and opponent context. Ticketmaster, SeatGeek, or StubHub-style feeds can enrich market signals, but SeatSense will not retrain unless historical outcomes are present. After mapping, a Pydantic schema gate validates the converted SeatSense rows before training. For a 100,000-row client file, SeatSense uses chronological validation when dates or seasons are available, or a group-aware split by game ID when they are not.”

Transition: “Now let’s move from model output to product workflow.”

## 4:15-6:15 · Speaker 3 · Product Demo

“I’m opening the Streamlit app. The Executive Overview shows the product value proposition and KPIs. Now I’ll go to the Pricing Workbench and select the high-demand rivalry game.”

“Before the live pricing page, I can briefly show Model Performance. This is where a client sees the active 20,500-row training file, the train/validation/test split, the leakage audit, and drill-down visuals like demand mix, price gap scatter, signal bubble, and section analysis. This makes the model story more transparent than just showing one accuracy number.”

“The model predicts demand tier and confidence. The app also shows class probabilities, so the revenue manager can see whether the model is confident or uncertain.”

“Next, the recommendation page compares current primary price, secondary-market average price, and AI-recommended price. The pricing engine estimates revenue uplift and secondary-market leakage reduction. It also explains the business action: increase price within the approved cap, hold steady, or discount/promotion for low demand.”

“The live API signal mode can enrich the scenario with configured ticket marketplace, weather, sports, social, and analytics providers before the model and pricing engine run. The explanation is generated after that model and pricing output exists. If an OpenAI API key is configured, the app uses the Responses API to write an executive-ready recommendation. If no key is present, it falls back to a polished deterministic template, so the demo still works.”

Transition: “A pricing system needs governance, so the next section covers responsible AI.”

## 6:15-7:45 · Speaker 4 · Responsible AI

“Dynamic pricing can harm trust if fans feel prices are arbitrary or unfair. SeatSense includes guardrails. It does not use protected attributes directly. It uses price increase caps, affordable ticket reserves, tighter caps for value-sensitive sections, and human approval for premium games, large price changes, or affordability flags.”

“The responsible AI page shows a fairness audit by affordability segment. It also lists failure modes. A false positive high-demand prediction could lead to overpriced seats and poor attendance. A false negative high-demand prediction could leave tickets too cheap and shift profit to scalpers.”

“Privacy also matters. In a real deployment, traffic and sentiment signals should be aggregated, minimized, disclosed, and reviewed under privacy rules.”

Transition: “Finally, we connect the system to measurable business impact.”

## 7:45-9:15 · Speaker 1 · Business Impact

“The business impact page tracks revenue per game, total ticket revenue uplift, sell-through rate, attendance rate, average price gap versus resale market, secondary-market leakage reduction, fan affordability score, and approval rate.”

“We include conservative, expected, and aggressive revenue-lift scenarios. This matters because executives do not buy model accuracy; they buy retained revenue, better attendance, and lower leakage.”

“SeatSense also supports implementation planning. A real venue would start with historical data integration, then offline validation, then a revenue-manager pilot, then controlled A/B testing, and finally production monitoring.”

Transition: “We’ll close with the main takeaway.”

## 9:15-10:00 · Speaker 4 · Closing

“SeatSense AI shows how to build a complete AI business product: a real market problem, a trained and evaluated model, a working UI, pricing logic, business KPIs, and responsible AI guardrails.”

“Our recommendation is not to replace revenue managers. It is to give them better demand intelligence, stronger pricing support, and a governed workflow. The ask is to pilot SeatSense with real historical venue data and compare revenue, sell-through, resale leakage, fan sentiment, and fairness metrics against the current pricing process.”
