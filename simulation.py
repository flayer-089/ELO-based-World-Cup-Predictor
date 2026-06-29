"""Poisson match model + Monte Carlo tournament simulation.

Ported from the original TypeScript. Elo ratings drive an expected-goals (xG)
model; goals are drawn from a Poisson distribution; the full 48-team tournament
is simulated thousands of times to estimate each team's odds at every stage.
"""

import math
import random

from worldcup2026 import WC2026_TEAMS, GROUPS, WCTeam

NUM_SIMULATIONS = 10_000
BASE_XG = 1.25  # average goals per team per game (fallback formula only)
ELO_SCALE = 400  # Elo scale factor (fallback formula only)

# The trained Poisson-regression goal model. When set (see set_goal_model),
# expected goals come from the machine-learning model instead of the hand-coded
# fallback formula below. Injected at runtime so this module has no hard
# dependency on scikit-learn when used standalone.
_GOAL_MODEL = None


def set_goal_model(model) -> None:
    """Install the trained Poisson regression as the expected-goals source."""
    global _GOAL_MODEL
    _GOAL_MODEL = model


# ---------- Math helpers ----------

def poisson_sample(lam: float) -> int:
    """Knuth's algorithm for sampling a Poisson-distributed integer."""
    if lam <= 0:
        return 0
    target = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= random.random()
        if p <= target:
            break
    return k - 1


def expected_goals(elo_a: float, elo_b: float) -> tuple[float, float]:
    """Convert an Elo gap into expected goals for each side.

    Uses the trained Poisson-regression model when one has been installed via
    set_goal_model(); otherwise falls back to the original hand-coded formula.
    """
    if _GOAL_MODEL is not None:
        return _GOAL_MODEL.expected_goals(elo_a, elo_b)

    diff = (elo_a - elo_b) / ELO_SCALE
    mult = 10 ** diff
    ratio = min(max(math.sqrt(mult), 0.33), 3)  # cap dominance
    total = BASE_XG * 2  # share ~2.5 goals between the two teams
    xg_a = (total * ratio) / (1 + ratio)
    xg_b = total - xg_a
    return xg_a, xg_b


def simulate_match(elo_a: float, elo_b: float) -> tuple[int, int]:
    xg_a, xg_b = expected_goals(elo_a, elo_b)
    return poisson_sample(xg_a), poisson_sample(xg_b)


def match_probabilities(elo_a: float, elo_b: float, trials: int = 50_000) -> dict:
    """Monte Carlo win/draw/loss probabilities + xG + most likely scoreline."""
    xg_a, xg_b = expected_goals(elo_a, elo_b)
    win_a = draw = win_b = 0
    # Track scorelines per outcome so the "most likely score" can reflect the
    # favourite. The unconditional modal score is ~1-1 across a wide band of
    # realistic xG (both 1-2 goals), since that's the joint mode of two
    # independent Poissons — it hides which side is actually ahead.
    score_freq: dict[str, dict[str, int]] = {"a": {}, "draw": {}, "b": {}}

    for _ in range(trials):
        ga = poisson_sample(xg_a)
        gb = poisson_sample(xg_b)
        key = f"{ga}-{gb}"
        if ga > gb:
            win_a += 1
            bucket = score_freq["a"]
        elif ga < gb:
            win_b += 1
            bucket = score_freq["b"]
        else:
            draw += 1
            bucket = score_freq["draw"]
        bucket[key] = bucket.get(key, 0) + 1

    # Most likely scoreline *within the most likely outcome*.
    outcome = max(("a", win_a), ("draw", draw), ("b", win_b), key=lambda kv: kv[1])[0]
    bucket = score_freq[outcome]
    most_likely = max(bucket.items(), key=lambda kv: kv[1])[0] if bucket else "1-1"

    return {
        "p_win_a": win_a / trials,
        "p_draw": draw / trials,
        "p_win_b": win_b / trials,
        "xg_a": round(xg_a, 2),
        "xg_b": round(xg_b, 2),
        "most_likely_score": most_likely,
    }


# ---------- Tournament ----------

# Fixed Round-of-32 bracket template (faithful to the real World Cup format,
# without FIFA's full third-place lookup table). Each entry is one R32 match as
# (slot_a, slot_b); slots are filled in after the group stage. Codes:
#   ("W", group) -> winner of that group
#   ("R", group) -> runner-up of that group
#   ("T", i)     -> the i-th best third-place team (0 = best)
# Matches are listed in bracket order, so consecutive matches feed the same
# Round-of-16 tie, those feed the same quarter-final, and so on.
#
# Properties that mirror the real draw:
#   * No group winner can meet another group winner in the Round of 32.
#   * Each group's winner and runner-up sit in opposite halves of the bracket,
#     so two teams from the same group can only meet again in the final.
R32_BRACKET: list[tuple[tuple, tuple]] = [
    # --- Top half ---
    (("W", "A"), ("T", 0)),
    (("R", "C"), ("R", "D")),
    (("W", "E"), ("T", 1)),
    (("W", "G"), ("R", "H")),
    (("W", "B"), ("T", 2)),
    (("R", "F"), ("R", "L")),
    (("W", "I"), ("T", 3)),
    (("W", "K"), ("R", "J")),
    # --- Bottom half ---
    (("W", "C"), ("T", 4)),
    (("R", "A"), ("R", "B")),
    (("W", "F"), ("T", 5)),
    (("W", "H"), ("R", "G")),
    (("W", "D"), ("T", 6)),
    (("R", "E"), ("R", "I")),
    (("W", "J"), ("T", 7)),
    (("W", "L"), ("R", "K")),
]

# Group whose winner occupies the other side of each ("T", i) slot above, used
# to keep a third-place team from being drawn against its own group's winner.
_TSLOT_WINNER_GROUP = {0: "A", 1: "E", 2: "B", 3: "I", 4: "C", 5: "F", 6: "D", 7: "L"}


def _assign_thirds(thirds: list[WCTeam]) -> list[WCTeam]:
    """Place the 8 best third-place teams into the 8 ("T", i) slots, avoiding
    (where possible) a third-place team facing its own group's winner."""
    remaining = list(range(len(thirds)))
    assigned: list[WCTeam] = []
    for i in range(len(thirds)):
        winner_group = _TSLOT_WINNER_GROUP[i]
        pick = next(
            (ti for ti in remaining if thirds[ti].group != winner_group),
            remaining[0],
        )
        assigned.append(thirds[pick])
        remaining.remove(pick)
    return assigned


class _Standing:
    __slots__ = ("team", "elo", "points", "gf", "ga")

    def __init__(self, team: WCTeam, elo: float):
        self.team = team
        self.elo = elo
        self.points = 0
        self.gf = 0
        self.ga = 0

    @property
    def gd(self) -> int:
        return self.gf - self.ga


def _simulate_group(group_teams: list[WCTeam], ratings: dict[str, float]) -> list[_Standing]:
    standings = [_Standing(t, ratings.get(t.name, 1000)) for t in group_teams]
    n = len(standings)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = standings[i], standings[j]
            ga, gb = simulate_match(a.elo, b.elo)
            a.gf += ga
            a.ga += gb
            b.gf += gb
            b.ga += ga
            if ga > gb:
                a.points += 3
            elif gb > ga:
                b.points += 3
            else:
                a.points += 1
                b.points += 1
    standings.sort(key=lambda s: (s.points, s.gd, s.gf), reverse=True)
    return standings


def _simulate_knockout(elo_a: float, elo_b: float) -> bool:
    """Return True if A advances. Ties go to a slightly-weighted penalty shootout."""
    ga, gb = simulate_match(elo_a, elo_b)
    if ga > gb:
        return True
    if gb > ga:
        return False
    pen_edge = min(0.6, 0.5 + (elo_a - elo_b) / 2000)
    return random.random() < pen_edge


def run_simulations(ratings: dict[str, float], num_simulations: int = NUM_SIMULATIONS) -> dict:
    names = [t.name for t in WC2026_TEAMS]
    result = {
        stage: {n: 0 for n in names}
        for stage in (
            "titles",
            "finals",
            "semi_finals",
            "quarter_finals",
            "round_of_16",
            "group_wins",
            "group_advances",
        )
    }

    groups_index = {g: [t for t in WC2026_TEAMS if t.group == g] for g in GROUPS}

    for _ in range(num_simulations):
        group_results = [_simulate_group(groups_index[g], ratings) for g in GROUPS]

        winners_by_group: dict[str, WCTeam] = {}
        runners_by_group: dict[str, WCTeam] = {}
        third_placers: list[_Standing] = []

        for g, standings in zip(GROUPS, group_results):
            winner, second, third = standings[0], standings[1], standings[2]
            result["group_wins"][winner.team.name] += 1
            result["group_advances"][winner.team.name] += 1
            result["group_advances"][second.team.name] += 1
            winners_by_group[g] = winner.team
            runners_by_group[g] = second.team
            third_placers.append(third)

        # Best 8 third-place teams complete the 32-team knockout bracket.
        third_placers.sort(key=lambda s: (s.points, s.gd, s.gf), reverse=True)
        best_thirds = third_placers[:8]
        for s in best_thirds:
            result["group_advances"][s.team.name] += 1
        thirds = _assign_thirds([s.team for s in best_thirds])

        # Fill the fixed bracket template instead of a random draw, so that
        # finishing position and seeding protection actually shape the path.
        def resolve(slot: tuple) -> WCTeam:
            kind, key = slot
            if kind == "W":
                return winners_by_group[key]
            if kind == "R":
                return runners_by_group[key]
            return thirds[key]  # ("T", i)

        pool: list[WCTeam] = []
        for slot_a, slot_b in R32_BRACKET:
            pool.append(resolve(slot_a))
            pool.append(resolve(slot_b))

        def knockout_round(teams: list[WCTeam]) -> list[WCTeam]:
            winners: list[WCTeam] = []
            for i in range(0, len(teams), 2):
                a, b = teams[i], teams[i + 1]
                a_wins = _simulate_knockout(ratings.get(a.name, 1000), ratings.get(b.name, 1000))
                winners.append(a if a_wins else b)
            return winners

        r16 = knockout_round(pool)
        for t in r16:
            result["round_of_16"][t.name] += 1

        qf = knockout_round(r16)
        for t in qf:
            result["quarter_finals"][t.name] += 1

        sf = knockout_round(qf)
        for t in sf:
            result["semi_finals"][t.name] += 1

        finalists = knockout_round(sf)
        for t in finalists:
            result["finals"][t.name] += 1

        champion = knockout_round(finalists)[0]
        result["titles"][champion.name] += 1

    return result
