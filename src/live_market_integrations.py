"""Optional live market signal integrations for SeatSense AI.

These connectors are intentionally isolated from the model and pricing engine.
When live providers are configured, they enrich scenario inputs before the
saved demand model and deterministic pricing logic run. If a provider is
missing, unavailable, rate-limited, or malformed, the app keeps using the
scenario values already on screen.
"""

from __future__ import annotations

import os
import statistics
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import requests


REQUEST_TIMEOUT_SECONDS = 6
USER_AGENT = "SeatSenseAI/1.0 revenue-intelligence"


@dataclass
class ProviderStatus:
    provider: str
    status: str
    message: str
    value: str = ""


@dataclass
class LiveMarketSignals:
    adjustments: dict[str, Any] = field(default_factory=dict)
    statuses: list[ProviderStatus] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _streamlit_secret(name: str) -> str | None:
    try:
        import streamlit as st

        value = st.secrets.get(name)
        return str(value) if value else None
    except Exception:
        return None


def _secret(name: str) -> str | None:
    value = os.getenv(name) or _streamlit_secret(name)
    if not value:
        return None
    lowered = value.lower()
    if lowered.startswith("your_") or "example" in lowered:
        return None
    return value


def _request_json(url: str, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> dict[str, Any]:
    request_headers = {"User-Agent": USER_AGENT, **(headers or {})}
    response = requests.get(
        url,
        params={key: value for key, value in (params or {}).items() if value not in (None, "")},
        headers=request_headers,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _average(values: list[float]) -> float | None:
    clean = [float(value) for value in values if value and value > 0]
    if not clean:
        return None
    return float(statistics.mean(clean))


def _sentiment_score(texts: list[str]) -> float:
    positive = {
        "win",
        "great",
        "elite",
        "healthy",
        "excited",
        "rivalry",
        "playoff",
        "hot",
        "sold",
        "star",
        "must",
    }
    negative = {
        "injury",
        "injured",
        "bad",
        "cold",
        "rain",
        "loss",
        "expensive",
        "empty",
        "bench",
        "out",
        "snow",
    }
    score = 0
    tokens = 0
    for text in texts:
        for token in text.lower().replace("/", " ").replace("-", " ").split():
            tokens += 1
            if token.strip(".,!?;:()[]") in positive:
                score += 1
            elif token.strip(".,!?;:()[]") in negative:
                score -= 1
    if tokens == 0:
        return 0.0
    return max(-1.0, min(1.0, score / max(5, len(texts))))


def _extract_prices(payload: Any, limit: int = 40) -> list[float]:
    prices: list[float] = []
    price_keys = {
        "averageprice",
        "lowestprice",
        "highestprice",
        "min",
        "max",
        "price",
        "amount",
        "listingprice",
        "currentprice",
        "facevalue",
    }

    def walk(value: Any) -> None:
        if len(prices) >= limit:
            return
        if isinstance(value, dict):
            for key, nested in value.items():
                normalized_key = str(key).replace("_", "").lower()
                if normalized_key in price_keys:
                    parsed = _safe_float(nested)
                    if parsed and 5 <= parsed <= 10000:
                        prices.append(parsed)
                walk(nested)
        elif isinstance(value, list):
            for nested in value:
                walk(nested)

    walk(payload)
    return prices


def _fetch_ticketmaster_prices(keyword: str, city: str) -> tuple[list[float], ProviderStatus]:
    api_key = _secret("TICKETMASTER_API_KEY")
    if not api_key:
        return [], ProviderStatus("Ticketmaster", "Not configured", "Set TICKETMASTER_API_KEY to use Discovery API prices.")
    try:
        payload = _request_json(
            "https://app.ticketmaster.com/discovery/v2/events.json",
            params={
                "apikey": api_key,
                "keyword": keyword,
                "city": city,
                "classificationName": "sports",
                "countryCode": "US",
                "size": 8,
                "sort": "date,asc",
            },
        )
        events = payload.get("_embedded", {}).get("events", [])
        prices = []
        for event in events:
            for price_range in event.get("priceRanges", []) or []:
                low = _safe_float(price_range.get("min"))
                high = _safe_float(price_range.get("max"))
                if low and high:
                    prices.append((low + high) / 2)
                elif low:
                    prices.append(low)
                elif high:
                    prices.append(high)
        avg = _average(prices)
        if avg:
            return prices, ProviderStatus("Ticketmaster", "Live", "Discovery API returned event price ranges.", f"${avg:,.0f}")
        return [], ProviderStatus("Ticketmaster", "No price", "Events were found but no price ranges were returned.")
    except Exception as exc:
        return [], ProviderStatus("Ticketmaster", "Unavailable", f"Live call failed: {exc.__class__.__name__}.")


def _fetch_seatgeek_prices(keyword: str, city: str) -> tuple[list[float], ProviderStatus]:
    client_id = _secret("SEATGEEK_CLIENT_ID")
    if not client_id:
        return [], ProviderStatus("SeatGeek", "Not configured", "Set SEATGEEK_CLIENT_ID to use SeatGeek event price signals.")
    try:
        params = {
            "client_id": client_id,
            "client_secret": _secret("SEATGEEK_CLIENT_SECRET"),
            "q": keyword,
            "venue.city": city,
            "type": "sports",
            "per_page": 8,
        }
        payload = _request_json("https://api.seatgeek.com/2/events", params=params)
        prices = []
        for event in payload.get("events", []) or []:
            stats = event.get("stats", {}) or {}
            for key in ("average_price", "lowest_price", "highest_price"):
                parsed = _safe_float(stats.get(key))
                if parsed:
                    prices.append(parsed)
        avg = _average(prices)
        if avg:
            return prices, ProviderStatus("SeatGeek", "Live", "SeatGeek API returned resale price statistics.", f"${avg:,.0f}")
        return [], ProviderStatus("SeatGeek", "No price", "SeatGeek events were found but no price statistics were returned.")
    except Exception as exc:
        return [], ProviderStatus("SeatGeek", "Unavailable", f"Live call failed: {exc.__class__.__name__}.")


def _fetch_stubhub_prices() -> tuple[list[float], ProviderStatus]:
    token = _secret("STUBHUB_ACCESS_TOKEN")
    api_url = _secret("STUBHUB_API_URL")
    if not token or not api_url:
        return [], ProviderStatus("StubHub", "Not configured", "Set STUBHUB_ACCESS_TOKEN and STUBHUB_API_URL for StubHub listing signals.")
    try:
        payload = _request_json(api_url, headers={"Authorization": f"Bearer {token}"})
        prices = _extract_prices(payload)
        avg = _average(prices)
        if avg:
            return prices, ProviderStatus("StubHub", "Live", "StubHub-compatible endpoint returned listing prices.", f"${avg:,.0f}")
        return [], ProviderStatus("StubHub", "No price", "StubHub endpoint responded but no parseable prices were found.")
    except Exception as exc:
        return [], ProviderStatus("StubHub", "Unavailable", f"Live call failed: {exc.__class__.__name__}.")


def _fetch_weather(latitude: float, longitude: float) -> tuple[dict[str, float], ProviderStatus]:
    try:
        payload = _request_json(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "daily": "precipitation_probability_max,wind_speed_10m_max,temperature_2m_max,temperature_2m_min",
                "forecast_days": 7,
                "timezone": "auto",
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
            },
        )
        daily = payload.get("daily", {}) or {}
        precip = _safe_float((daily.get("precipitation_probability_max") or [None])[0]) or 0
        wind = _safe_float((daily.get("wind_speed_10m_max") or [None])[0]) or 0
        high = _safe_float((daily.get("temperature_2m_max") or [None])[0])
        low = _safe_float((daily.get("temperature_2m_min") or [None])[0])
        temp_penalty = 0.0
        if high is not None and (high >= 92 or high <= 35):
            temp_penalty += 0.18
        if low is not None and low <= 28:
            temp_penalty += 0.16
        severity = max(0.0, min(1.0, (precip / 100) * 0.45 + min(wind / 45, 1) * 0.35 + temp_penalty))
        return {"weather_severity": severity}, ProviderStatus("Open-Meteo", "Live", "Forecast API returned precipitation, wind, and temperature.", f"{severity:.2f}")
    except Exception as exc:
        return {}, ProviderStatus("Open-Meteo", "Unavailable", f"Weather call failed: {exc.__class__.__name__}.")


def _team_matches(team: dict[str, Any], token: str) -> bool:
    token = token.lower().strip()
    if not token:
        return False
    haystack = " ".join(str(value).lower() for value in team.values() if isinstance(value, (str, int, float)))
    return token in haystack


def _fetch_sports_strength(team_token: str) -> tuple[dict[str, float], ProviderStatus]:
    if not team_token:
        return {}, ProviderStatus("NBA stats", "Skipped", "No team token was provided.")
    try:
        payload = _request_json("https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json")
        for game in payload.get("scoreboard", {}).get("games", []) or []:
            for side in ("homeTeam", "awayTeam"):
                team = game.get(side, {}) or {}
                if _team_matches(team, team_token):
                    wins = _safe_float(team.get("wins"))
                    losses = _safe_float(team.get("losses"))
                    if wins is not None and losses is not None and wins + losses > 0:
                        win_rate = wins / (wins + losses)
                        strength = max(35, min(98, 100 * win_rate))
                        return {
                            "opponent_strength": strength,
                            "away_team_win_rate": win_rate,
                        }, ProviderStatus("NBA stats", "Live", "NBA scoreboard returned team record.", f"{strength:.0f}/100")
                    return {}, ProviderStatus("NBA stats", "No record", "NBA scoreboard found the team but no record was available.")
        return {}, ProviderStatus("NBA stats", "No match", "NBA scoreboard responded, but no matching team was found today.")
    except Exception as exc:
        return {}, ProviderStatus("NBA stats", "Unavailable", f"Sports stats call failed: {exc.__class__.__name__}.")


def _fetch_social_signal(keyword: str) -> tuple[dict[str, float], ProviderStatus]:
    bearer = _secret("X_BEARER_TOKEN")
    try:
        if bearer:
            payload = _request_json(
                "https://api.x.com/2/tweets/search/recent",
                params={"query": keyword, "max_results": 10, "tweet.fields": "created_at"},
                headers={"Authorization": f"Bearer {bearer}"},
            )
            texts = [item.get("text", "") for item in payload.get("data", []) or []]
            score = _sentiment_score(texts)
            traffic = min(230, 80 + len(texts) * 7 + max(score, 0) * 35)
            return {
                "social_sentiment_score": score,
                "website_traffic_index": traffic,
            }, ProviderStatus("X social", "Live", "X recent-search API returned social posts.", f"{score:+.2f}")

        payload = _request_json(
            "https://www.reddit.com/search.json",
            params={"q": keyword, "sort": "new", "limit": 20},
        )
        children = payload.get("data", {}).get("children", []) or []
        texts = [
            f"{child.get('data', {}).get('title', '')} {child.get('data', {}).get('selftext', '')}"
            for child in children
        ]
        score = _sentiment_score(texts)
        traffic = min(230, 70 + len(texts) * 4 + max(score, 0) * 30)
        return {
            "social_sentiment_score": score,
            "website_traffic_index": traffic,
        }, ProviderStatus("Reddit social", "Live", "Reddit search JSON returned public social posts.", f"{score:+.2f}")
    except Exception as exc:
        return {}, ProviderStatus("Social", "Unavailable", f"Social signal call failed: {exc.__class__.__name__}.")


def _fetch_website_analytics() -> tuple[dict[str, float], ProviderStatus]:
    custom_url = _secret("WEBSITE_ANALYTICS_API_URL")
    token = _secret("PLAUSIBLE_API_TOKEN")
    site_id = _secret("PLAUSIBLE_SITE_ID")
    try:
        if custom_url:
            payload = _request_json(custom_url)
            value = (
                _safe_float(payload.get("traffic_index"))
                or _safe_float(payload.get("visitors"))
                or _safe_float(payload.get("pageviews"))
            )
            if value:
                traffic = max(35, min(230, value if value <= 230 else 60 + value / 20))
                return {"website_traffic_index": traffic}, ProviderStatus("Website analytics", "Live", "Custom analytics endpoint returned traffic.", f"{traffic:.0f}")
            return {}, ProviderStatus("Website analytics", "No metric", "Custom endpoint responded but no traffic metric was found.")

        if token and site_id:
            payload = _request_json(
                "https://plausible.io/api/v1/stats/aggregate",
                params={"site_id": site_id, "period": "7d", "metrics": "visitors,pageviews"},
                headers={"Authorization": f"Bearer {token}"},
            )
            results = payload.get("results", {}) or {}
            visitors = _safe_float(results.get("visitors", {}).get("value") if isinstance(results.get("visitors"), dict) else results.get("visitors"))
            pageviews = _safe_float(results.get("pageviews", {}).get("value") if isinstance(results.get("pageviews"), dict) else results.get("pageviews"))
            value = visitors or pageviews
            if value:
                traffic = max(35, min(230, 60 + value / 25))
                return {"website_traffic_index": traffic}, ProviderStatus("Plausible analytics", "Live", "Plausible API returned 7-day traffic.", f"{traffic:.0f}")
            return {}, ProviderStatus("Plausible analytics", "No metric", "Plausible responded but no visitors/pageviews value was found.")

        return {}, ProviderStatus("Website analytics", "Not configured", "Set PLAUSIBLE_* or WEBSITE_ANALYTICS_API_URL for traffic signals.")
    except Exception as exc:
        return {}, ProviderStatus("Website analytics", "Unavailable", f"Analytics call failed: {exc.__class__.__name__}.")


def _blend_adjustments(base: dict[str, Any], adjustments: dict[str, Any]) -> dict[str, Any]:
    merged = base.copy()
    for key, value in adjustments.items():
        if value is None:
            continue
        if key in {"weather_severity", "social_sentiment_score", "away_team_win_rate"}:
            merged[key] = round(float(value), 3)
        elif key in {"website_traffic_index", "opponent_strength"}:
            merged[key] = round(float(value), 2)
        elif key == "secondary_market_avg_price":
            merged[key] = round(float(value), 2)
        else:
            merged[key] = value
    return merged


def apply_live_signals(base_input: dict[str, Any], signals: LiveMarketSignals | None) -> dict[str, Any]:
    if not signals:
        return base_input
    return _blend_adjustments(base_input, signals.adjustments)


def provider_status_frame(signals: LiveMarketSignals | None):
    import pandas as pd

    rows = [
        {
            "Provider": status.provider,
            "Status": status.status,
            "Signal": status.value,
            "Message": status.message,
        }
        for status in (signals.statuses if signals else [])
    ]
    return pd.DataFrame(rows)


def collect_live_market_signals(
    base_input: dict[str, Any],
    *,
    market_query: str,
    city: str = "New York",
    latitude: float = 40.7505,
    longitude: float = -73.9934,
    sports_team: str = "",
    use_ticketmaster: bool = True,
    use_seatgeek: bool = True,
    use_stubhub: bool = True,
    use_weather: bool = True,
    use_sports_stats: bool = True,
    use_social: bool = True,
    use_analytics: bool = True,
) -> LiveMarketSignals:
    signals = LiveMarketSignals()
    market_prices: list[float] = []

    if use_ticketmaster:
        prices, status = _fetch_ticketmaster_prices(market_query, city)
        market_prices.extend(prices)
        signals.statuses.append(status)
    if use_seatgeek:
        prices, status = _fetch_seatgeek_prices(market_query, city)
        market_prices.extend(prices)
        signals.statuses.append(status)
    if use_stubhub:
        prices, status = _fetch_stubhub_prices()
        market_prices.extend(prices)
        signals.statuses.append(status)

    avg_market_price = _average(market_prices)
    if avg_market_price:
        current = float(base_input.get("current_ticket_price", 0) or 0)
        existing_secondary = float(base_input.get("secondary_market_avg_price", current) or current)
        blended = (avg_market_price * 0.70) + (existing_secondary * 0.30)
        signals.adjustments["secondary_market_avg_price"] = max(15, blended)
        signals.notes.append(f"Live ticket marketplaces adjusted secondary price toward ${avg_market_price:,.0f}.")

    if use_weather:
        adjustments, status = _fetch_weather(latitude, longitude)
        signals.adjustments.update(adjustments)
        signals.statuses.append(status)

    if use_sports_stats:
        adjustments, status = _fetch_sports_strength(sports_team)
        signals.adjustments.update(adjustments)
        signals.statuses.append(status)

    if use_social:
        adjustments, status = _fetch_social_signal(market_query)
        signals.adjustments.update(adjustments)
        signals.statuses.append(status)

    if use_analytics:
        adjustments, status = _fetch_website_analytics()
        signals.adjustments.update(adjustments)
        signals.statuses.append(status)

    if signals.adjustments:
        signals.notes.append("Live signals were applied before the demand model and pricing engine ran.")
    else:
        signals.notes.append("No live provider returned an adjustable signal; the scenario inputs remain unchanged.")
    return signals
