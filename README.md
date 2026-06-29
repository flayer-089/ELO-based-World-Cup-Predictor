# World Cup Oracle — Terminal Edition

A dependency-free Python CLI that predicts the 2026 FIFA World Cup.

It computes **Elo ratings** from ~49,000 historical international matches, feeds
them into a **Poisson** scoring model, and runs a **Monte Carlo** simulation of
the full 48-team tournament (10,000 runs by default) to estimate every team's
odds of winning the title, reaching the final, and reaching the semis.

Then it drops you into an interactive prompt where you can type two teams and
get the head-to-head prediction: win/draw/loss probabilities, expected goals,
and the most likely scoreline.

## Requirements

Python 3.10+ — **no third-party packages** (standard library only).

## Run

```bash
python oracle.py
```

Options:

```bash
python oracle.py --sims 2000     # fewer simulations = faster, a bit noisier
```

On first run it downloads the historical results dataset and caches it locally
(`.cache_results.csv`); later runs are offline and instant to load. Delete that
file to refresh the data. The dataset is live — now that the tournament is under
way it already includes played 2026 World Cup matches, which feed straight into
the Elo ratings (fixtures not yet played carry no score and are ignored).

## Teams & groups

The 48 teams and their groups follow the **official FIFA final draw** held on
5 December 2025 in Washington, D.C.

| | | | |
|---|---|---|---|
| **A** Mexico · South Africa · South Korea · Czech Republic | **B** Canada · Bosnia & Herz. · Qatar · Switzerland | **C** Brazil · Morocco · Haiti · Scotland | **D** USA · Paraguay · Australia · Turkey |
| **E** Germany · Curaçao · Ivory Coast · Ecuador | **F** Netherlands · Japan · Sweden · Tunisia | **G** Belgium · Egypt · Iran · New Zealand | **H** Spain · Cape Verde · Saudi Arabia · Uruguay |
| **I** France · Senegal · Iraq · Norway | **J** Argentina · Algeria · Austria · Jordan | **K** Portugal · DR Congo · Uzbekistan · Colombia | **L** England · Croatia · Ghana · Panama |

## Using the prompt

```
> Brazil vs France      # head-to-head match prediction
> titles                # reprint the full title-odds table
> teams                 # list all 48 qualified teams + groups
> quit
```

Team names are matched loosely — `Brazil`, `BRA`, or `bra` all work.

## How it works

| File | Responsibility |
|------|----------------|
| `elo.py` | Downloads results, replays them chronologically, computes Elo strength ratings. |
| `simulation.py` | Poisson expected-goals model + Monte Carlo group stage and knockout bracket. |
| `worldcup2026.py` | The 48 qualified teams, group assignments, and dataset name mapping. |
| `oracle.py` | The CLI — wires it together and renders the tables / prompt. |

### The model in brief

- **Elo:** every historical match nudges each team's rating toward its result,
  weighted by match importance (World Cup > continental > qualifier > friendly),
  goal margin, and home advantage. Teams start at 1000.
- **Expected goals:** the Elo gap between two teams is converted into a share of
  ~2.5 total expected goals.
- **Match outcome:** each side's goals are drawn from a Poisson distribution.
- **Tournament:** 12 groups of 4 play round-robin; the top two of each group
  plus the eight best third-place teams advance to a 32-team knockout bracket.
  The bracket uses a **fixed template** that follows the real format — group
  winners are protected from each other in the Round of 32, and a group's winner
  and runner-up sit in opposite halves so they can only meet again in the final
  — rather than a random draw, so finishing position actually shapes a team's
  path. (FIFA's exact third-place lookup table is approximated.) Knockout ties
  are decided by a lightly Elo-weighted penalty shootout. Repeat 10,000 times
  and count how often each team reaches each stage.

> Predictions are a probabilistic model for entertainment, not betting advice.
