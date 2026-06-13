# Brazil WC 2026 Match Predictor 🇧🇷🏆

A machine learning pipeline that predicts Brazil's 2026 FIFA World Cup match outcomes and scorelines using historical international football data going back to 1872.

Built in Python with XGBoost and Poisson regression.

---

## How it works

Two models run together on every prediction:

**Model 1 — XGBoost Classifier** answers *who wins?*
Trained on match outcomes (win/draw/loss) using ELO ratings, recent form, head-to-head history, and tournament weight as features.

**Model 2 — Poisson Regressor** answers *what's the score?*
Predicts expected goals for each team, then simulates 50,000 matches to produce a full scoreline probability table.

Data is pulled live from [`martj42/international_results`](https://github.com/martj42/international_results) — a community-maintained dataset of 47,000+ international matches — and cached locally after the first run.

---

## Quickstart

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/wc2026-predictor.git
cd wc2026-predictor

# Set up environment
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run — pass Brazil's opponent as the second argument
python prediction.py "Brazil" "Morocco"
```

---

## Example output

```
Brazil vs Morocco — Group Stage, 2026 FIFA World Cup

P(Brazil win)  = 56.1%
P(Draw)        = 21.2%
P(Morocco win) = 22.8%
Predicted outcome: Brazil (56.1%)

Expected goals: Brazil 1.65 — Morocco 1.11

Scoreline    Result          Probability
1-1          Draw            11.5%
1-0          Brazil win      10.6%
2-1          Brazil win       9.5%
2-0          Brazil win       8.7%
0-1          Morocco win      7.0%
```

---

## Running predictions for each Brazil game

Pass Brazil's opponent as the second argument — no need to edit any code:

```bash
python prediction.py "Brazil" "Morocco"
python prediction.py "Brazil" "Croatia"
python prediction.py "Brazil" "Germany"
```

To refresh the data before a match (pulls the latest results from GitHub):

```bash
rm -rf data_cache/
python prediction.py "Brazil" "Morocco"
```

---

## Features used

| Feature | Description |
|---|---|
| `elo_diff` | ELO rating difference between teams |
| `home_elo` / `away_elo` | Individual team ELO ratings |
| `neutral` | Whether the match is on neutral ground |
| `tournament_weight` | How competitive the tournament is (1–4) |
| `home_win5` / `away_win5` | Win rate over last 5 games |
| `home_gd5` / `away_gd5` | Avg goal difference over last 5 games |
| `h2h_n` | Number of historical head-to-head matches |
| `h2h_home_winrate` | Home team win rate in H2H history |
| `h2h_home_gd` | Avg goal difference in H2H history |

---

## Project structure

```
wc2026-predictor/
├── prediction.py      # Full pipeline — data, features, models, predictions
├── requirements.txt   # Dependencies
├── MODEL_CARD.md      # Model performance, limitations, intended use
├── .gitignore
└── data_cache/        # Auto-created — cached CSV from GitHub (git-ignored)
```

---

## Dependencies

- `pandas` / `numpy` — data processing
- `xgboost` — outcome classifier
- `scikit-learn` — Poisson regressor, metrics
- `requests` — data fetching

---

## Limitations

See [`MODEL_CARD.md`](MODEL_CARD.md) for full details on performance, known weaknesses, and intended use.