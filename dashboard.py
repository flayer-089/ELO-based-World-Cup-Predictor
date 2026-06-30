"""Streamlit dashboard for the 2026 World Cup Oracle.

Reuses the same Elo -> Poisson -> Monte Carlo pipeline as oracle.py. Pick any
two qualified teams in the sidebar and the dashboard shows win/draw/loss
probabilities, expected goals, the expected score (rounded xG), the most
likely scoreline, a top-20 score distribution, and each side's stage-by-stage
tournament odds.

Run with:
    streamlit run dashboard.py
"""

import math

import pandas as pd
import streamlit as st

from elo import compute_elo_ratings, get_wc_team_ratings
from poisson_model import train_poisson_model
from simulation import (
    expected_goals,
    match_probabilities,
    poisson_sample,
    run_simulations,
    set_goal_model,
)
from worldcup2026 import WC2026_TEAMS

NUM_SIMS = 5_000
H2H_TRIALS = 50_000
SCORE_TRIALS = 50_000
TOP_N_SCORES = 20

st.set_page_config(
    page_title="World Cup Oracle 2026",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading Elo ratings, training Poisson model, running 5,000 tournament sims…")
def load_all():
    """Heavy work runs once and is cached across reruns / users."""
    all_ratings, match_count, history = compute_elo_ratings(record_history=True)
    ratings = get_wc_team_ratings(all_ratings)
    model = train_poisson_model(history, verbose=False)
    set_goal_model(model)
    sim = run_simulations(ratings, NUM_SIMS)
    return ratings, sim, match_count, model


ratings, sim, match_count, model = load_all()

team_by_name = {t.name: t for t in WC2026_TEAMS}
team_names = sorted(team_by_name.keys(), key=lambda n: (team_by_name[n].group, n))


def team_label(name: str) -> str:
    t = team_by_name[name]
    return f"{t.flag}  {t.name}  ·  Group {t.group}"


# ---------- Sidebar ----------

with st.sidebar:
    st.title("⚽ Match Setup")
    st.caption("Pick any two qualified teams.")
    st.divider()

    team_a = st.selectbox(
        "Team A",
        team_names,
        index=team_names.index("Brazil"),
        format_func=team_label,
    )
    team_b = st.selectbox(
        "Team B",
        team_names,
        index=team_names.index("Argentina"),
        format_func=team_label,
    )

    if team_a == team_b:
        st.warning("Pick two different teams to see a prediction.")

    st.divider()
    st.markdown("##### 🧠 Model details")
    st.caption(f"Trained on **{match_count:,}** historical international matches.")
    st.caption(
        f"xG = exp({model.intercept:.3f} "
        f"+ {model.elo_coef:.3f}·ΔElo/100 "
        f"+ {model.home_coef:.3f}·home)"
    )
    st.caption(f"Each **+100 Elo** → **×{math.exp(model.elo_coef):.2f}** goals")
    st.caption(f"Home field → **×{math.exp(model.home_coef):.2f}** goals")
    st.caption(f"Neutral & evenly matched: **{math.exp(model.intercept):.2f} xG** per side")
    st.divider()
    st.caption(
        f"Tournament: **{NUM_SIMS:,}** simulated brackets  ·  "
        f"H2H: **{H2H_TRIALS:,}** trials per query"
    )


# ---------- Header ----------

st.title("🏆 2026 FIFA World Cup — Match Oracle")
st.markdown(
    "Probabilistic predictions powered by **Elo ratings** (replayed from ~49k "
    "historical matches), a **Poisson-regression** goal model, and "
    "**Monte Carlo** tournament simulation."
)
st.divider()


@st.cache_data(show_spinner=False)
def h2h_block(elo_a: float, elo_b: float) -> dict:
    return match_probabilities(elo_a, elo_b, trials=H2H_TRIALS)


@st.cache_data(show_spinner=False)
def score_distribution(elo_a: float, elo_b: float) -> pd.DataFrame:
    xg_a, xg_b = expected_goals(elo_a, elo_b)
    counts: dict[str, int] = {}
    for _ in range(SCORE_TRIALS):
        key = f"{poisson_sample(xg_a)}-{poisson_sample(xg_b)}"
        counts[key] = counts.get(key, 0) + 1
    rows = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:TOP_N_SCORES]
    df = pd.DataFrame(rows, columns=["Score", "Frequency"])
    df["Probability"] = df["Frequency"] / df["Frequency"].sum()
    return df


def outcome_for(score: str, ta: str, tb: str, t_a_obj, t_b_obj) -> str:
    ga_s, gb_s = score.split("-")
    ga, gb = int(ga_s), int(gb_s)
    if ga > gb:
        return f"{t_a_obj.flag} {ta} win"
    if ga < gb:
        return f"{tb} {t_b_obj.flag} win"
    return "Draw"


elo_a = ratings[team_a]
elo_b = ratings[team_b]
t_a = team_by_name[team_a]
t_b = team_by_name[team_b]

if team_a == team_b:
    st.info("👈 Choose two different teams in the sidebar to see the prediction.")
    st.stop()

res = h2h_block(elo_a, elo_b)

# Compute the expected (rounded xG) score and the modal score's winner label.
exp_ga = round(res["xg_a"])
exp_gb = round(res["xg_b"])
if exp_ga > exp_gb:
    exp_winner = f"{t_a.flag} {team_a} win"
elif exp_ga < exp_gb:
    exp_winner = f"{team_b} {t_b.flag} win"
else:
    exp_winner = "Draw"

_sa, _sb = res["most_likely_score"].split("-")
if int(_sa) > int(_sb):
    modal_winner = f"{t_a.flag} {team_a} win"
elif int(_sa) < int(_sb):
    modal_winner = f"{team_b} {t_b.flag} win"
else:
    modal_winner = "Draw"


# ---------- Matchup header ----------

left, mid, right = st.columns([5, 1, 5])
with left:
    st.markdown(
        f"<h1 style='text-align:center; margin-bottom:0;'>{t_a.flag} {team_a}</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<h4 style='text-align:center; color:gray; margin-top:0;'>Elo {elo_a}</h4>",
        unsafe_allow_html=True,
    )
with mid:
    st.markdown(
        "<h1 style='text-align:center; padding-top:24px;'>⚔️</h1>",
        unsafe_allow_html=True,
    )
with right:
    st.markdown(
        f"<h1 style='text-align:center; margin-bottom:0;'>{team_b} {t_b.flag}</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<h4 style='text-align:center; color:gray; margin-top:0;'>Elo {elo_b}</h4>",
        unsafe_allow_html=True,
    )

st.divider()


# ---------- Win / Draw / Loss ----------

st.subheader("📊 Match Outcome")
m1, m2, m3 = st.columns(3)
m1.metric(f"{t_a.flag} {team_a} win", f"{res['p_win_a'] * 100:.1f}%")
m2.metric("Draw", f"{res['p_draw'] * 100:.1f}%")
m3.metric(f"{team_b} {t_b.flag} win", f"{res['p_win_b'] * 100:.1f}%")

prob_df = pd.DataFrame(
    {
        "Outcome": [f"{team_a} win", "Draw", f"{team_b} win"],
        "Probability": [res["p_win_a"], res["p_draw"], res["p_win_b"]],
    }
).set_index("Outcome")
st.bar_chart(prob_df, height=240, color="#1f77b4")

st.divider()


# ---------- Expected goals + score predictions ----------

st.subheader("⚽ Expected Goals & Score Predictions")
g1, g2, g3 = st.columns(3)
g1.metric(f"{team_a} xG", f"{res['xg_a']:.2f}")
g2.metric("Total xG", f"{res['xg_a'] + res['xg_b']:.2f}")
g3.metric(f"{team_b} xG", f"{res['xg_b']:.2f}")

# Big "Expected Score" banner: the deterministic point estimate (rounded xG).
# Unlike the modal sample, this naturally produces 2-0, 3-1, 4-0 etc. for
# lopsided matchups instead of always collapsing to 1-0 or 1-1.
st.markdown(
    f"""
    <div style='text-align:center; padding:24px 18px; border-radius:14px;
                background:linear-gradient(90deg, #1f4e79 50%, #7a1f1f 50%);'>
        <p style='color:#d0d0d0; margin:0; font-size:0.85em; letter-spacing:2px;'>
            EXPECTED SCORE &nbsp;·&nbsp; rounded from xG
        </p>
        <h1 style='color:white; margin:8px 0 6px 0; font-size:3em; letter-spacing:10px;'>
            {exp_ga} – {exp_gb}
        </h1>
        <p style='color:white; margin:0; font-size:1.1em; font-weight:500;'>{exp_winner}</p>
        <p style='color:#c8c8c8; margin:4px 0 0 0; font-size:0.85em;'>format: {team_a} – {team_b}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Top 5 most likely scorelines, shown as a grid so 3-1, 2-2, 4-0 etc. are
# always visible (not just the modal one).
sdf = score_distribution(elo_a, elo_b)
top5 = sdf.head(5).reset_index(drop=True)

st.markdown("##### 🥇 Top 5 Most Likely Scorelines")
grid = st.columns(5, gap="small")
medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
for i, (_, row) in enumerate(top5.iterrows()):
    score_str = row["Score"].replace("-", "–")
    label = outcome_for(row["Score"], team_a, team_b, t_a, t_b)
    with grid[i]:
        st.markdown(
            f"""
            <div style='text-align:center; padding:12px 6px; border-radius:10px;
                        background:#1e1e1e; border:1px solid #333;'>
                <div style='font-size:1.4em;'>{medals[i]}</div>
                <div style='color:white; font-size:1.6em; font-weight:600;
                            letter-spacing:3px; margin:4px 0;'>{score_str}</div>
                <div style='color:#c8c8c8; font-size:0.78em; min-height:2.4em;'>
                    {label}
                </div>
                <div style='color:#2ca02c; font-size:0.95em; font-weight:500;'>
                    {row["Probability"] * 100:.1f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()


# ---------- Score distribution ----------

st.subheader(f"🎯 Top {TOP_N_SCORES} Most Likely Scorelines")
score_df = sdf.set_index("Score")[["Probability"]]
st.bar_chart(score_df, height=420, color="#2ca02c")

# Detailed table with outcome labels and counts.
with st.expander(f"📋 Show top {TOP_N_SCORES} as a table"):
    table_df = sdf.copy()
    table_df["Outcome"] = table_df["Score"].apply(
        lambda s: outcome_for(s, team_a, team_b, t_a, t_b)
    )
    table_df["Probability"] = table_df["Probability"].map(lambda p: f"{p * 100:.2f}%")
    table_df["Frequency"] = table_df["Frequency"].map(lambda f: f"{f:,} of {SCORE_TRIALS:,}")
    st.dataframe(
        table_df[["Score", "Outcome", "Probability", "Frequency"]],
        hide_index=True,
        width="stretch",
    )

st.divider()


# ---------- Tournament outlook ----------

st.subheader(f"🏆 Tournament Outlook ({NUM_SIMS:,} simulated brackets)")

stages = [
    ("Group advance", "group_advances"),
    ("Round of 16", "round_of_16"),
    ("Quarter-finals", "quarter_finals"),
    ("Semi-finals", "semi_finals"),
    ("Reach Final", "finals"),
    ("Champion", "titles"),
]
tdata = {
    "Stage": [label for label, _ in stages],
    team_a: [sim[key][team_a] / NUM_SIMS for _, key in stages],
    team_b: [sim[key][team_b] / NUM_SIMS for _, key in stages],
}
tdf = pd.DataFrame(tdata)

left_t, right_t = st.columns([3, 2])
with left_t:
    st.dataframe(
        tdf.style.format({team_a: "{:.1%}", team_b: "{:.1%}"}),
        hide_index=True,
        width="stretch",
    )
with right_t:
    st.bar_chart(
        tdf.set_index("Stage"),
        height=300,
        horizontal=True,
        color=["#1f77b4", "#d62728"],
    )

st.divider()


# ---------- Title odds (top 10) ----------

st.subheader("🌍 Tournament Title Odds — Top 10")
title_rows = sorted(sim["titles"].items(), key=lambda kv: kv[1], reverse=True)[:10]
title_df = pd.DataFrame(
    {
        "Team": [n for n, _ in title_rows],
        "Title %": [v / NUM_SIMS * 100 for _, v in title_rows],
    }
).set_index("Team")
st.bar_chart(title_df, height=320)

st.caption(
    "Probabilistic model for entertainment. Predictions depend on the "
    "training data and the learned Poisson coefficients, not on form, injuries, or lineups."
)
