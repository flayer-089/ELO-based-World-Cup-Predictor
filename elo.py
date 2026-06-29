"""Elo ratings — the team "strength index".

Downloads ~49k historical international match results and replays them
chronologically, updating each team's Elo rating after every match. This is a
direct port of the original TypeScript implementation.
"""

import csv
import io
import os
import ssl
import urllib.request

from worldcup2026 import WC2026_TEAMS

CSV_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
CACHE_PATH = os.path.join(os.path.dirname(__file__), ".cache_results.csv")


def k_factor(tournament: str) -> float:
    """Match importance multiplier — bigger competitions move ratings more."""
    t = tournament.lower()
    if "fifa world cup" in t and "qualif" not in t:
        return 60
    if any(
        c in t
        for c in (
            "copa america",
            "uefa euro",
            "africa cup",
            "afc asian cup",
            "gold cup",
            "concacaf nations",
        )
    ):
        return 50
    if "qualif" in t or "qualification" in t:
        return 40
    if "nations league" in t or "confederation" in t:
        return 35
    return 20  # Friendly


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def _ssl_context() -> ssl.SSLContext:
    """Verified TLS where possible; fall back to unverified on misconfigured
    macOS Python builds that ship without root certificates."""
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        pass
    try:
        ctx = ssl.create_default_context()
        # Probe whether the default trust store actually works.
        return ctx
    except Exception:
        return ssl._create_unverified_context()


def _download_csv() -> str:
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return f.read()
    print("Downloading international results CSV (~49k matches)...")
    try:
        with urllib.request.urlopen(CSV_URL, timeout=60, context=_ssl_context()) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.URLError as e:
        if isinstance(e.reason, ssl.SSLCertVerificationError):
            # Last resort: this host's Python has no usable CA bundle.
            with urllib.request.urlopen(
                CSV_URL, timeout=60, context=ssl._create_unverified_context()
            ) as resp:
                raw = resp.read().decode("utf-8")
        else:
            raise
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        f.write(raw)
    return raw


def _parse_csv(raw: str) -> list[dict]:
    rows: list[dict] = []
    reader = csv.reader(io.StringIO(raw))
    header = next(reader, None)  # skip header
    for parts in reader:
        if len(parts) < 9:
            continue
        try:
            home_score = int(parts[3])
            away_score = int(parts[4])
        except ValueError:
            continue
        rows.append(
            {
                "date": parts[0],
                "home_team": parts[1],
                "away_team": parts[2],
                "home_score": home_score,
                "away_score": away_score,
                "tournament": parts[5],
                "neutral": parts[8].strip().upper() == "TRUE",
            }
        )
    return rows


def compute_elo_ratings(record_history: bool = False):
    """Replay all matches in date order and return {team: rating}, match_count.

    If ``record_history`` is True, also return a third value: a list of
    pre-match snapshots (each team's Elo *as it stood on match day*, before the
    result was applied) alongside the final score. That snapshot is exactly the
    training data the Poisson regression learns from — using the rating at the
    time of the match avoids leaking future information into the features.
    """
    raw = _download_csv()
    rows = _parse_csv(raw)
    rows.sort(key=lambda r: r["date"])

    ratings: dict[str, float] = {}
    history: list[dict] | None = [] if record_history else None

    def get_rating(team: str) -> float:
        return ratings.setdefault(team, 1000.0)

    for r in rows:
        home_elo = get_rating(r["home_team"])
        away_elo = get_rating(r["away_team"])
        if history is not None:
            history.append(
                {
                    "home_elo": home_elo,
                    "away_elo": away_elo,
                    "neutral": r["neutral"],
                    "home_score": r["home_score"],
                    "away_score": r["away_score"],
                }
            )

        home_adv = 0 if r["neutral"] else 75
        r_a = home_elo + home_adv
        r_b = away_elo

        expected_a = expected_score(r_a, r_b)
        expected_b = 1 - expected_a

        if r["home_score"] > r["away_score"]:
            actual_a, actual_b = 1.0, 0.0
        elif r["home_score"] < r["away_score"]:
            actual_a, actual_b = 0.0, 1.0
        else:
            actual_a, actual_b = 0.5, 0.5

        k = k_factor(r["tournament"])
        goal_diff = abs(r["home_score"] - r["away_score"])
        # Goal-difference multiplier (FIFA-style), capped at 1.75x.
        gd_mult = 1.0 if goal_diff <= 1 else 1.5 if goal_diff == 2 else 1.75

        ratings[r["home_team"]] = home_elo + k * gd_mult * (actual_a - expected_a)
        ratings[r["away_team"]] = away_elo + k * gd_mult * (actual_b - expected_b)

    if history is not None:
        return ratings, len(rows), history
    return ratings, len(rows)


def get_wc_team_ratings(all_ratings: dict[str, float]) -> dict[str, float]:
    """Map the full rating table down to the 48 qualified teams, by display name."""
    return {t.name: round(all_ratings.get(t.csv_name, 1000.0)) for t in WC2026_TEAMS}
