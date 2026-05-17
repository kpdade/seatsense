"""Generate semi-synthetic ticketing data for SeatSense AI.

Private ticket transaction data, live resale feeds, venue web analytics, and
fan-level CRM records are not generally publicly available. This generator
creates a transparent, semi-synthetic dataset calibrated to realistic
sports-ticketing dynamics so the ML workflow can be inspected without claiming
the observations are real venue transactions.

The unit of observation is:
    game_id + seat_section + pricing_window

Default size:
    5 seasons x 82 games x 10 seat sections x 5 pricing windows = 20,500 rows

The generator intentionally separates pre-decision features from post-event
business outcomes. Affordability signals are produced for monitoring and
guardrails only; they are not model training features.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


OPPONENTS = {
    "Atlanta Hawks": 58,
    "Boston Celtics": 92,
    "Brooklyn Nets": 70,
    "Chicago Bulls": 68,
    "Cleveland Cavaliers": 82,
    "Dallas Mavericks": 86,
    "Denver Nuggets": 90,
    "Detroit Pistons": 42,
    "Golden State Warriors": 88,
    "Houston Rockets": 61,
    "Indiana Pacers": 74,
    "LA Clippers": 80,
    "Los Angeles Lakers": 91,
    "Memphis Grizzlies": 73,
    "Miami Heat": 79,
    "Milwaukee Bucks": 87,
    "Minnesota Timberwolves": 83,
    "New Orleans Pelicans": 64,
    "New York Knicks": 85,
    "Oklahoma City Thunder": 89,
    "Orlando Magic": 66,
    "Philadelphia 76ers": 84,
    "Phoenix Suns": 81,
    "Portland Trail Blazers": 50,
    "Sacramento Kings": 71,
    "San Antonio Spurs": 62,
    "Toronto Raptors": 63,
    "Utah Jazz": 55,
    "Washington Wizards": 45,
}

RIVALRY_OPPONENTS = {
    "Boston Celtics",
    "Los Angeles Lakers",
    "Miami Heat",
    "New York Knicks",
    "Philadelphia 76ers",
}

DIVISION_OPPONENTS = {
    "Boston Celtics",
    "Brooklyn Nets",
    "New York Knicks",
    "Philadelphia 76ers",
    "Toronto Raptors",
}

HOLIDAY_DATES = {
    (10, 31),
    (11, 24),
    (12, 25),
    (12, 31),
    (1, 1),
    (2, 14),
}


@dataclass(frozen=True)
class SectionConfig:
    seat_quality_score: float
    base_price: float
    capacity: int
    elasticity: float
    affordability_segment: str
    affordability_index: float


SECTIONS: dict[str, SectionConfig] = {
    "Upper Baseline": SectionConfig(0.38, 42, 4200, -1.42, "Value-sensitive", 42),
    "Upper Sideline": SectionConfig(0.46, 55, 3600, -1.30, "Value-sensitive", 48),
    "Upper Center": SectionConfig(0.52, 66, 2600, -1.18, "Value-sensitive", 54),
    "Lower Corner": SectionConfig(0.63, 96, 2100, -0.96, "Mainstream", 66),
    "Lower Sideline": SectionConfig(0.72, 128, 1900, -0.82, "Mainstream", 72),
    "Lower Center": SectionConfig(0.79, 168, 1200, -0.74, "Mainstream", 78),
    "Club Level": SectionConfig(0.83, 235, 950, -0.58, "Affluent", 96),
    "Suite": SectionConfig(0.91, 385, 350, -0.42, "Affluent", 112),
    "Courtside/Premium": SectionConfig(0.98, 575, 180, -0.34, "Premium", 130),
    "Standing Room": SectionConfig(0.30, 34, 1400, -1.55, "Value-sensitive", 38),
}

PRICING_WINDOWS = [
    ("30_days_out", 30, 0.24),
    ("14_days_out", 14, 0.46),
    ("7_days_out", 7, 0.63),
    ("2_days_out", 2, 0.80),
    ("day_of_game", 0, 0.92),
]


def sigmoid(value: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def season_stage_from_game_number(game_number: int, games_per_season: int) -> str:
    pct = game_number / games_per_season
    if pct < 0.25:
        return "early"
    if pct < 0.68:
        return "mid"
    if pct < 0.90:
        return "late"
    return "playoff_push"


def days_bucket(days_before_game: int) -> str:
    if days_before_game >= 21:
        return "21_plus_days"
    if days_before_game >= 8:
        return "8_to_20_days"
    if days_before_game >= 3:
        return "3_to_7_days"
    if days_before_game >= 1:
        return "1_to_2_days"
    return "day_of_game"


def label_demand(final_sell_through: float, realized_score: float, rng: np.random.Generator) -> str:
    """Create a realistic target label with edge-case noise."""

    blended = 0.70 * final_sell_through + 0.30 * realized_score + rng.normal(0, 0.025)
    if blended >= 0.83:
        return "High"
    if blended >= 0.64:
        return "Medium"
    return "Low"


def label_scalper_risk(
    resale_price_gap_pct: float,
    listing_count: float,
    resale_velocity: float,
    demand_tier: str,
    rng: np.random.Generator,
) -> str:
    risk_score = (
        0.42 * np.clip(resale_price_gap_pct / 0.55, 0, 1.5)
        + 0.22 * np.clip(listing_count / 520, 0, 1.4)
        + 0.25 * np.clip(resale_velocity / 100, 0, 1.4)
        + {"Low": 0.02, "Medium": 0.16, "High": 0.30}[demand_tier]
        + rng.normal(0, 0.045)
    )
    if risk_score >= 0.72:
        return "High"
    if risk_score >= 0.42:
        return "Medium"
    return "Low"


def build_game_rows(
    seasons: int = 5,
    games_per_season: int = 82,
    seed: int = 751,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    opponents = list(OPPONENTS.keys())
    rows: list[dict] = []

    for season_offset in range(seasons):
        season = 2021 + season_offset
        season_start = pd.Timestamp(year=season, month=10, day=15)
        possible_offsets = np.sort(
            rng.choice(np.arange(0, 176), size=games_per_season, replace=False)
        )
        home_strength = float(np.clip(rng.normal(0.56 + season_offset * 0.012, 0.075), 0.34, 0.78))

        for game_number, offset in enumerate(possible_offsets, start=1):
            game_date = season_start + pd.Timedelta(days=int(offset))
            opponent = str(rng.choice(opponents))
            opponent_strength = float(np.clip(rng.normal(OPPONENTS[opponent], 6.5), 32, 98))
            day_of_week = game_date.day_name()
            weekend_flag = int(day_of_week in {"Friday", "Saturday", "Sunday"})
            holiday_flag = int((game_date.month, game_date.day) in HOLIDAY_DATES)
            rivalry_flag = int(opponent in RIVALRY_OPPONENTS and rng.random() < 0.80)
            division_matchup_flag = int(opponent in DIVISION_OPPONENTS)
            season_stage = season_stage_from_game_number(game_number, games_per_season)

            home_team_win_rate = float(np.clip(home_strength + rng.normal(0, 0.055), 0.26, 0.84))
            away_team_win_rate = float(np.clip((opponent_strength / 100) + rng.normal(0, 0.075), 0.24, 0.84))
            home_team_recent_form = float(np.clip(home_team_win_rate + rng.normal(0, 0.10), 0.18, 0.90))
            away_team_recent_form = float(np.clip(away_team_win_rate + rng.normal(0, 0.10), 0.18, 0.90))
            playoff_implication_score = float(
                np.clip(
                    0.10
                    + (0.18 if season_stage in {"late", "playoff_push"} else 0)
                    + 0.25 * max(home_team_win_rate - 0.50, 0)
                    + 0.20 * max(away_team_win_rate - 0.50, 0)
                    + rng.normal(0, 0.09),
                    0,
                    1,
                )
            )
            premium_game_flag = int(
                rivalry_flag
                or holiday_flag
                or opponent_strength >= 88
                or (weekend_flag and opponent_strength >= 82)
                or playoff_implication_score >= 0.55
            )

            late_season_injury_risk = 0.09 if season_stage in {"late", "playoff_push"} else 0.04
            injury_news_severity = float(
                np.clip(rng.beta(1.4, 7.5) + late_season_injury_risk * rng.random(), 0, 1)
            )
            star_player_available = int(rng.random() > (0.055 + 0.38 * injury_news_severity))
            month_weather_penalty = 0.16 if game_date.month in {12, 1, 2} else 0.05
            weather_severity = float(
                np.clip(rng.beta(1.8, 6.2) + month_weather_penalty + rng.normal(0, 0.06), 0, 1)
            )
            temperature_score = float(
                np.clip(1.0 - weather_severity * 0.55 + rng.normal(0, 0.08), 0, 1)
            )

            base_signal = (
                0.36 * (opponent_strength / 100)
                + 0.18 * weekend_flag
                + 0.19 * rivalry_flag
                + 0.07 * holiday_flag
                + 0.18 * playoff_implication_score
                + 0.13 * star_player_available
                - 0.16 * weather_severity
                - 0.18 * injury_news_severity
                + 0.18 * (home_team_recent_form - 0.50)
                + rng.normal(0, 0.055)
            )
            social_sentiment_score = float(
                np.clip(-0.25 + 1.65 * base_signal + rng.normal(0, 0.20), -1, 1)
            )
            social_volume_index = float(
                np.clip(
                    38
                    + 110 * base_signal
                    + 24 * premium_game_flag
                    + 18 * rivalry_flag
                    + rng.normal(0, 14),
                    5,
                    210,
                )
            )
            search_interest_index = float(
                np.clip(
                    35
                    + 100 * base_signal
                    + 20 * weekend_flag
                    + 22 * premium_game_flag
                    + rng.normal(0, 12),
                    8,
                    205,
                )
            )
            website_traffic_index = float(
                np.clip(
                    42
                    + 112 * base_signal
                    + 0.40 * social_volume_index
                    + 0.34 * search_interest_index
                    + rng.normal(0, 12),
                    25,
                    240,
                )
            )
            marketing_spend_index = float(
                np.clip(
                    rng.normal(58, 17)
                    + (20 if not premium_game_flag else 5)
                    + (14 if weather_severity > 0.55 else 0),
                    10,
                    130,
                )
            )
            promotion_flag = int(rng.random() < (0.24 if base_signal < 0.38 else 0.09))
            email_campaign_active = int(rng.random() < (0.36 if promotion_flag else 0.19))

            rows.append(
                {
                    "season": season,
                    "game_id": f"S{season}_G{game_number:03d}",
                    "game_date": game_date.date().isoformat(),
                    "game_number": game_number,
                    "day_of_week": day_of_week,
                    "weekend_flag": weekend_flag,
                    "holiday_flag": holiday_flag,
                    "rivalry_flag": rivalry_flag,
                    "division_matchup_flag": division_matchup_flag,
                    "opponent": opponent,
                    "opponent_strength": round(opponent_strength, 2),
                    "home_team_win_rate": round(home_team_win_rate, 3),
                    "away_team_win_rate": round(away_team_win_rate, 3),
                    "home_team_recent_form": round(home_team_recent_form, 3),
                    "away_team_recent_form": round(away_team_recent_form, 3),
                    "playoff_implication_score": round(playoff_implication_score, 3),
                    "premium_game_flag": premium_game_flag,
                    "month": int(game_date.month),
                    "season_stage": season_stage,
                    "website_traffic_index_base": website_traffic_index,
                    "search_interest_index_base": search_interest_index,
                    "social_sentiment_score_base": social_sentiment_score,
                    "social_volume_index_base": social_volume_index,
                    "email_campaign_active": email_campaign_active,
                    "promotion_flag": promotion_flag,
                    "marketing_spend_index": round(marketing_spend_index, 2),
                    "star_player_available": star_player_available,
                    "injury_news_severity": round(injury_news_severity, 3),
                    "weather_severity": round(weather_severity, 3),
                    "temperature_score": round(temperature_score, 3),
                    "base_signal": base_signal,
                }
            )

    return pd.DataFrame(rows)


def _section_final_outcome(
    game: pd.Series,
    section: str,
    config: SectionConfig,
    season_index: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    section_quality = config.seat_quality_score
    season_inflation = 1 + 0.030 * season_index
    event_markup = (
        0.08 * game["weekend_flag"]
        + 0.12 * game["holiday_flag"]
        + 0.18 * game["rivalry_flag"]
        + 0.17 * game["premium_game_flag"]
        + 0.10 * game["playoff_implication_score"]
        + 0.08 * (game["opponent_strength"] / 100)
    )
    base_ticket_price = config.base_price * season_inflation * (1 + event_markup)
    base_ticket_price *= float(rng.normal(1.0, 0.035))

    historical_price_elasticity = float(np.clip(rng.normal(config.elasticity, 0.10), -1.80, -0.25))
    section_historical_fill_rate = float(
        np.clip(
            0.55
            + 0.24 * game["base_signal"]
            + 0.18 * (1 - section_quality)
            + 0.10 * game["weekend_flag"]
            + rng.normal(0, 0.055),
            0.34,
            0.98,
        )
    )
    rolling_3_game_attendance_rate = float(
        np.clip(section_historical_fill_rate + rng.normal(0, 0.055), 0.30, 0.99)
    )
    rolling_5_game_sell_through_rate = float(
        np.clip(section_historical_fill_rate + rng.normal(0, 0.050), 0.30, 0.99)
    )
    previous_similar_game_sell_through = float(
        np.clip(
            section_historical_fill_rate
            + 0.08 * game["rivalry_flag"]
            + 0.06 * game["premium_game_flag"]
            + rng.normal(0, 0.065),
            0.28,
            0.99,
        )
    )
    same_opponent_last_season_attendance = float(
        np.clip(
            section_historical_fill_rate
            + 0.06 * (game["opponent_strength"] / 100)
            + rng.normal(0, 0.070),
            0.28,
            0.99,
        )
    )

    realized_demand_score = float(
        np.clip(
            0.32
            + 0.30 * game["base_signal"]
            + 0.16 * section_historical_fill_rate
            + 0.12 * previous_similar_game_sell_through
            + 0.08 * (1 - section_quality)
            + rng.normal(0, 0.065),
            0.08,
            1.08,
        )
    )
    final_logit = (
        -1.30
        + 3.05 * realized_demand_score
        + 0.42 * (1 - section_quality)
        + 0.35 * game["premium_game_flag"]
        + 0.18 * game["weekend_flag"]
        - 0.18 * max(section_quality - 0.80, 0)
        + rng.normal(0, 0.38)
    )
    final_sell_through_rate = float(np.clip(sigmoid(final_logit), 0.28, 0.995))
    final_attendance_rate = float(
        np.clip(final_sell_through_rate - rng.normal(0.025, 0.025), 0.22, 0.995)
    )

    return {
        "base_ticket_price": base_ticket_price,
        "historical_price_elasticity_section": historical_price_elasticity,
        "section_historical_fill_rate": section_historical_fill_rate,
        "rolling_3_game_attendance_rate": rolling_3_game_attendance_rate,
        "rolling_5_game_sell_through_rate": rolling_5_game_sell_through_rate,
        "previous_similar_game_sell_through": previous_similar_game_sell_through,
        "same_opponent_last_season_attendance": same_opponent_last_season_attendance,
        "realized_demand_score": realized_demand_score,
        "final_sell_through_rate": final_sell_through_rate,
        "final_attendance_rate": final_attendance_rate,
    }


def build_observation_rows(
    games: pd.DataFrame,
    seed: int = 751,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed + 19)
    model_rows: list[dict] = []
    ticket_rows: list[dict] = []
    resale_rows: list[dict] = []

    min_season = int(games["season"].min())
    for _, game in games.iterrows():
        season_index = int(game["season"]) - min_season
        for section, config in SECTIONS.items():
            outcome = _section_final_outcome(game, section, config, season_index, rng)
            base_ticket_price = outcome["base_ticket_price"]
            final_sell_through = outcome["final_sell_through_rate"]
            base_demand_tier = label_demand(final_sell_through, outcome["realized_demand_score"], rng)

            prior_price_change = (
                0.03 * game["weekend_flag"]
                + 0.06 * game["premium_game_flag"]
                + 0.05 * game["rivalry_flag"]
                - 0.04 * game["promotion_flag"]
                + rng.normal(0, 0.025)
            )

            for pricing_window, days_before_game, sales_progress in PRICING_WINDOWS:
                urgency = (30 - days_before_game) / 30
                signal_lift = 0.08 * urgency * (final_sell_through - 0.70)
                current_price_change = prior_price_change + signal_lift + rng.normal(0, 0.025)
                if game["promotion_flag"] and final_sell_through < 0.68 and days_before_game <= 14:
                    current_price_change -= 0.06
                current_price_change = float(np.clip(current_price_change, -0.18, 0.32))
                current_ticket_price = float(max(12, base_ticket_price * (1 + current_price_change)))
                price_change_from_base_pct = current_ticket_price / max(base_ticket_price, 1) - 1

                sales_so_far = float(
                    np.clip(
                        final_sell_through * sales_progress
                        + 0.03 * game["promotion_flag"]
                        + 0.02 * game["email_campaign_active"]
                        + rng.normal(0, 0.035),
                        0.03,
                        0.985,
                    )
                )
                inventory_remaining = int(round(config.capacity * max(0, 1 - sales_so_far)))
                inventory_remaining_pct = inventory_remaining / max(config.capacity, 1)

                website_traffic_index = float(
                    np.clip(
                        game["website_traffic_index_base"]
                        * (0.82 + 0.45 * urgency + 0.20 * max(final_sell_through - 0.75, 0))
                        + rng.normal(0, 9),
                        25,
                        260,
                    )
                )
                search_interest_index = float(
                    np.clip(
                        game["search_interest_index_base"] * (0.86 + 0.38 * urgency)
                        + rng.normal(0, 8),
                        8,
                        220,
                    )
                )
                social_volume_index = float(
                    np.clip(
                        game["social_volume_index_base"] * (0.88 + 0.46 * urgency)
                        + rng.normal(0, 9),
                        5,
                        235,
                    )
                )
                social_sentiment_score = float(
                    np.clip(
                        game["social_sentiment_score_base"]
                        - 0.30 * game["injury_news_severity"] * (1 - game["star_player_available"])
                        - 0.18 * game["weather_severity"]
                        + rng.normal(0, 0.10),
                        -1,
                        1,
                    )
                )

                scarcity = max(0, sales_so_far - 0.62)
                section_premium = max(0, config.seat_quality_score - 0.70)
                secondary_markup = (
                    0.05
                    + 0.18 * game["premium_game_flag"]
                    + 0.16 * game["rivalry_flag"]
                    + 0.34 * scarcity
                    + 0.14 * max(social_sentiment_score, 0)
                    + 0.10 * section_premium
                    - 0.10 * (final_sell_through < 0.62)
                    + rng.normal(0, 0.065)
                )
                secondary_market_avg_price = float(
                    max(10, current_ticket_price * (1 + secondary_markup))
                )
                resale_price_gap_pct = secondary_market_avg_price / max(current_ticket_price, 1) - 1
                secondary_market_listing_count = float(
                    np.clip(
                        config.capacity
                        * (0.020 + 0.065 * game["premium_game_flag"] + 0.080 * max(resale_price_gap_pct, 0))
                        * (0.65 + 0.80 * urgency)
                        + rng.normal(0, 18),
                        2,
                        950,
                    )
                )
                resale_velocity_index = float(
                    np.clip(
                        22
                        + 75 * max(resale_price_gap_pct, 0)
                        + 30 * final_sell_through
                        + 18 * game["premium_game_flag"]
                        + rng.normal(0, 11),
                        5,
                        145,
                    )
                )
                scalper_pressure_score = float(
                    np.clip(
                        0.42 * np.clip(resale_price_gap_pct / 0.45, 0, 1.5)
                        + 0.22 * np.clip(secondary_market_listing_count / 520, 0, 1.5)
                        + 0.28 * np.clip(resale_velocity_index / 120, 0, 1.5)
                        + rng.normal(0, 0.035),
                        0,
                        1,
                    )
                )
                price_elasticity = outcome["historical_price_elasticity_section"]
                demand_signal_score = float(
                    np.clip(
                        0.20 * np.clip(website_traffic_index / 220, 0, 1.2)
                        + 0.16 * np.clip(search_interest_index / 190, 0, 1.2)
                        + 0.12 * np.clip(social_volume_index / 210, 0, 1.2)
                        + 0.11 * ((social_sentiment_score + 1) / 2)
                        + 0.16 * sales_so_far
                        + 0.10 * np.clip(resale_price_gap_pct / 0.50, 0, 1.2)
                        + 0.06 * game["premium_game_flag"]
                        + 0.05 * game["rivalry_flag"]
                        + 0.04 * (1 - game["weather_severity"]),
                        0,
                        1.15,
                    )
                )
                window_demand_score = float(
                    np.clip(
                        0.48 * demand_signal_score
                        + 0.25 * final_sell_through
                        + 0.17 * outcome["realized_demand_score"]
                        + 0.06 * sales_so_far
                        + 0.04 * (base_demand_tier == "High")
                        + rng.normal(0, 0.018),
                        0,
                        1.15,
                    )
                )
                if window_demand_score >= 0.77:
                    demand_tier = "High"
                elif window_demand_score >= 0.58:
                    demand_tier = "Medium"
                else:
                    demand_tier = "Low"
                scalper_risk_tier = label_scalper_risk(
                    resale_price_gap_pct,
                    secondary_market_listing_count,
                    resale_velocity_index,
                    demand_tier,
                    rng,
                )
                realized_units = config.capacity * final_sell_through
                realized_revenue = float(realized_units * current_ticket_price)
                revenue_per_available_seat = realized_revenue / config.capacity
                unsold_inventory_pct = float(1 - final_sell_through)
                actual_secondary_market_gap = float(
                    max(0, secondary_market_avg_price - current_ticket_price)
                )
                price_pressure = current_ticket_price / max(base_ticket_price, 1) - 1
                optimal_change = float(
                    np.clip(
                        0.12 * (demand_tier == "High")
                        + 0.05 * (demand_tier == "Medium")
                        - 0.08 * (demand_tier == "Low")
                        + 0.12 * max(resale_price_gap_pct, 0)
                        - 0.04 * max(price_pressure, 0)
                        - 0.05 * (config.affordability_segment == "Value-sensitive"),
                        -0.20,
                        0.35 if game["premium_game_flag"] else 0.25,
                    )
                )
                optimal_price_oracle = float(base_ticket_price * (1 + optimal_change))
                recommended_price_target = optimal_price_oracle

                row = {
                    "season": int(game["season"]),
                    "game_id": str(game["game_id"]),
                    "game_date": game["game_date"],
                    "day_of_week": game["day_of_week"],
                    "weekend_flag": int(game["weekend_flag"]),
                    "holiday_flag": int(game["holiday_flag"]),
                    "rivalry_flag": int(game["rivalry_flag"]),
                    "division_matchup_flag": int(game["division_matchup_flag"]),
                    "opponent": game["opponent"],
                    "opponent_strength": round(float(game["opponent_strength"]), 2),
                    "home_team_win_rate": round(float(game["home_team_win_rate"]), 3),
                    "away_team_win_rate": round(float(game["away_team_win_rate"]), 3),
                    "home_team_recent_form": round(float(game["home_team_recent_form"]), 3),
                    "away_team_recent_form": round(float(game["away_team_recent_form"]), 3),
                    "playoff_implication_score": round(float(game["playoff_implication_score"]), 3),
                    "premium_game_flag": int(game["premium_game_flag"]),
                    "seat_section": section,
                    "section_capacity": int(config.capacity),
                    "inventory_remaining": int(inventory_remaining),
                    "inventory_remaining_pct": round(float(inventory_remaining_pct), 4),
                    "current_sell_through_rate": round(float(sales_so_far), 4),
                    "seat_quality_score": round(float(config.seat_quality_score), 3),
                    "base_ticket_price": round(float(base_ticket_price), 2),
                    "current_ticket_price": round(float(current_ticket_price), 2),
                    "price_change_from_base_pct": round(float(price_change_from_base_pct), 4),
                    "days_before_game": int(days_before_game),
                    "pricing_window": pricing_window,
                    "days_until_game_bucket": days_bucket(days_before_game),
                    "month": int(game["month"]),
                    "season_stage": game["season_stage"],
                    "website_traffic_index": round(website_traffic_index, 2),
                    "search_interest_index": round(search_interest_index, 2),
                    "social_sentiment_score": round(social_sentiment_score, 3),
                    "social_volume_index": round(social_volume_index, 2),
                    "email_campaign_active": int(game["email_campaign_active"]),
                    "promotion_flag": int(game["promotion_flag"]),
                    "marketing_spend_index": round(float(game["marketing_spend_index"]), 2),
                    "star_player_available": int(game["star_player_available"]),
                    "injury_news_severity": round(float(game["injury_news_severity"]), 3),
                    "weather_severity": round(float(game["weather_severity"]), 3),
                    "temperature_score": round(float(game["temperature_score"]), 3),
                    "demand_signal_score": round(float(demand_signal_score), 4),
                    "secondary_market_avg_price": round(float(secondary_market_avg_price), 2),
                    "secondary_market_listing_count": round(float(secondary_market_listing_count), 2),
                    "resale_price_gap_pct": round(float(resale_price_gap_pct), 4),
                    "resale_velocity_index": round(float(resale_velocity_index), 2),
                    "scalper_pressure_score": round(float(scalper_pressure_score), 4),
                    "rolling_3_game_attendance_rate": round(outcome["rolling_3_game_attendance_rate"], 3),
                    "rolling_5_game_sell_through_rate": round(outcome["rolling_5_game_sell_through_rate"], 3),
                    "previous_similar_game_sell_through": round(outcome["previous_similar_game_sell_through"], 3),
                    "same_opponent_last_season_attendance": round(
                        outcome["same_opponent_last_season_attendance"], 3
                    ),
                    "section_historical_fill_rate": round(outcome["section_historical_fill_rate"], 3),
                    "historical_price_elasticity_section": round(float(price_elasticity), 3),
                    "historical_attendance_rate": round(outcome["rolling_3_game_attendance_rate"], 3),
                    "affordability_segment": config.affordability_segment,
                    "affordability_index": round(float(config.affordability_index + rng.normal(0, 3)), 2),
                    "local_income_proxy": round(float(config.affordability_index + rng.normal(0, 3)), 2),
                    "final_sell_through_rate": round(float(final_sell_through), 3),
                    "final_attendance_rate": round(float(outcome["final_attendance_rate"]), 3),
                    "realized_revenue": round(float(realized_revenue), 2),
                    "revenue_per_available_seat": round(float(revenue_per_available_seat), 2),
                    "unsold_inventory_pct": round(float(unsold_inventory_pct), 3),
                    "actual_secondary_market_gap": round(float(actual_secondary_market_gap), 2),
                    "realized_demand_score": round(float(outcome["realized_demand_score"]), 3),
                    "demand_index": round(float(outcome["realized_demand_score"]), 3),
                    "demand_tier": demand_tier,
                    "scalper_risk_tier": scalper_risk_tier,
                    "scalper_risk_label": scalper_risk_tier,
                    "optimal_price_oracle": round(float(optimal_price_oracle), 2),
                    "recommended_price_target": round(float(recommended_price_target), 2),
                    "tickets_sold_pct": round(float(sales_so_far), 3),
                    "sold_tickets": int(round(config.capacity * sales_so_far)),
                    "attendance_rate": round(float(outcome["final_attendance_rate"]), 3),
                    "revenue": round(float(current_ticket_price * config.capacity * sales_so_far), 2),
                }
                model_rows.append(row)

                ticket_rows.append(
                    {
                        "game_id": row["game_id"],
                        "season": row["season"],
                        "seat_section": row["seat_section"],
                        "pricing_window": row["pricing_window"],
                        "section_capacity": row["section_capacity"],
                        "inventory_remaining": row["inventory_remaining"],
                        "inventory_remaining_pct": row["inventory_remaining_pct"],
                        "current_sell_through_rate": row["current_sell_through_rate"],
                        "base_ticket_price": row["base_ticket_price"],
                        "current_ticket_price": row["current_ticket_price"],
                        "tickets_sold_pct": row["tickets_sold_pct"],
                        "sold_tickets": row["sold_tickets"],
                        "final_sell_through_rate": row["final_sell_through_rate"],
                        "realized_revenue": row["realized_revenue"],
                        "revenue": row["revenue"],
                        "promotion_flag": row["promotion_flag"],
                    }
                )
                resale_rows.append(
                    {
                        "game_id": row["game_id"],
                        "season": row["season"],
                        "seat_section": row["seat_section"],
                        "pricing_window": row["pricing_window"],
                        "secondary_market_avg_price": row["secondary_market_avg_price"],
                        "secondary_market_listing_count": row["secondary_market_listing_count"],
                        "resale_price_gap_pct": row["resale_price_gap_pct"],
                        "resale_velocity_index": row["resale_velocity_index"],
                        "scalper_pressure_score": row["scalper_pressure_score"],
                        "scalper_risk_tier": row["scalper_risk_tier"],
                    }
                )

    return pd.DataFrame(ticket_rows), pd.DataFrame(resale_rows), pd.DataFrame(model_rows)


def generate_all(
    output_dir: Path | None = None,
    seasons: int = 5,
    games_per_season: int = 82,
    seed: int = 751,
) -> dict[str, Path]:
    output_dir = Path(output_dir) if output_dir else DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    games = build_game_rows(seasons=seasons, games_per_season=games_per_season, seed=seed)
    ticket_sales, secondary_market, model_training = build_observation_rows(games, seed=seed)

    game_public = games.drop(
        columns=[
            "website_traffic_index_base",
            "search_interest_index_base",
            "social_sentiment_score_base",
            "social_volume_index_base",
            "base_signal",
        ]
    )
    paths = {
        "games": output_dir / "games.csv",
        "ticket_sales": output_dir / "ticket_sales.csv",
        "secondary_market": output_dir / "secondary_market.csv",
        "model_training_data": output_dir / "model_training_data.csv",
    }
    game_public.to_csv(paths["games"], index=False)
    ticket_sales.to_csv(paths["ticket_sales"], index=False)
    secondary_market.to_csv(paths["secondary_market"], index=False)
    model_training.to_csv(paths["model_training_data"], index=False)
    return paths


if __name__ == "__main__":
    generated_paths = generate_all()
    for name, path in generated_paths.items():
        print(f"{name}: {path}")
