"""Poisson regression goal model — the machine-learning core.

This is the supervised-learning step. It turns two teams' Elo ratings into
expected goals using a **Poisson regression** (a generalized linear model)
whose coefficients are *learned* from ~49k historical international matches by
maximum likelihood — `model.fit(X, y)` — instead of a hand-coded formula.

Pipeline (matches the on-camera description):
  1. Elo ratings are the team "strength index" feature (built in elo.py).
  2. Every past match becomes training rows: features = (Elo difference,
     home-field flag), target = goals that side actually scored.
  3. A Poisson GLM is fit to those rows. The fitted coefficients are the
     learned model.
  4. For a fixture we feed the two teams' Elo into the model to get expected
     goals for each side, then sample scorelines / run the Monte Carlo.
"""

import math
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import PoissonRegressor

from elo import compute_elo_ratings

# Elo differences span hundreds of points; scaling keeps the learned
# coefficient at a sane magnitude and the optimiser well-conditioned. It does
# not change predictions — it is undone consistently at fit and predict time.
ELO_DIFF_SCALE = 100.0


@dataclass
class PoissonGoalModel:
    """The trained model: log(xG) is linear in Elo difference and home field."""

    intercept: float
    elo_coef: float
    home_coef: float
    n_matches: int
    n_rows: int

    def expected_goals(
        self, elo_a: float, elo_b: float, a_home: bool = False, b_home: bool = False
    ) -> tuple[float, float]:
        """Predict expected goals for each side from their Elo ratings.

        World Cup fixtures are played on neutral ground, so by default neither
        side gets the home-field term.
        """
        log_xg_a = (
            self.intercept
            + self.elo_coef * (elo_a - elo_b) / ELO_DIFF_SCALE
            + self.home_coef * (1.0 if a_home else 0.0)
        )
        log_xg_b = (
            self.intercept
            + self.elo_coef * (elo_b - elo_a) / ELO_DIFF_SCALE
            + self.home_coef * (1.0 if b_home else 0.0)
        )
        return math.exp(log_xg_a), math.exp(log_xg_b)


def build_training_table(history: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Turn pre-match Elo snapshots into a supervised (X, y) table.

    Each match contributes two rows — one per team's scoring perspective:
        features = [Elo difference (own - opponent) / scale, had home field?]
        target   = goals that team scored
    Stacking both perspectives lets a single Poisson GLM learn one consistent
    "Elo gap -> goals" relationship for favourite and underdog alike.
    """
    X: list[tuple[float, float]] = []
    y: list[int] = []
    for m in history:
        host_flag = 0.0 if m["neutral"] else 1.0  # only the home side, non-neutral
        # Home team's attack
        X.append(((m["home_elo"] - m["away_elo"]) / ELO_DIFF_SCALE, host_flag))
        y.append(m["home_score"])
        # Away team's attack (the visitor never has home field)
        X.append(((m["away_elo"] - m["home_elo"]) / ELO_DIFF_SCALE, 0.0))
        y.append(m["away_score"])
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)


def train_poisson_model(history: list[dict] | None = None, verbose: bool = True) -> PoissonGoalModel:
    """Fit the Poisson regression on the historical match data and return it.

    If ``history`` is None, the Elo replay is run to produce it.
    """
    if history is None:
        _, _, history = compute_elo_ratings(record_history=True)

    X, y = build_training_table(history)

    # Poisson GLM with a log link. alpha is a tiny L2 term purely for numerical
    # stability; this is effectively maximum-likelihood estimation.
    model = PoissonRegressor(alpha=1e-8, max_iter=2000)
    model.fit(X, y)  # <-- the machine learning: coefficients learned from data

    intercept = float(model.intercept_)
    elo_coef, home_coef = (float(c) for c in model.coef_)

    if verbose:
        print(
            f"Trained Poisson regression on {len(y):,} team-match goal records "
            f"from {len(history):,} matches."
        )
        print(
            f"  learned: log(xG) = {intercept:.4f} "
            f"+ {elo_coef:.4f}*(EloDiff/{int(ELO_DIFF_SCALE)}) "
            f"+ {home_coef:.4f}*home"
        )
        print(
            f"  => neutral & evenly matched: {math.exp(intercept):.2f} xG per side; "
            f"home field is worth x{math.exp(home_coef):.2f} on goals; "
            f"each +{int(ELO_DIFF_SCALE)} Elo is x{math.exp(elo_coef):.2f}."
        )

    return PoissonGoalModel(intercept, elo_coef, home_coef, len(history), len(y))


if __name__ == "__main__":
    # Standalone: train and show the learned model. Good for the screen capture.
    train_poisson_model()
