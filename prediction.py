"""
World Cup 2026 Match Predictor
Based on ELO ratings + XGBoost classifier
Data: github.com/martj42/international_results
"""

import os
import warnings
import numpy as np
import pandas as pd
import requests
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss, classification_report
from sklearn.linear_model import PoissonRegressor

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────
CACHE_DIR = "data_cache"
RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results"
    "/master/results.csv"
)

# ELO parameters
ELO_BASE       = 1500.0   # Every team starts here
ELO_K          = 32       # How fast ratings update after each game
ELO_HOME_BONUS = 60       # Extra ELO points for playing at home

# Country name normalisation (dataset uses some old/alternate names)
NAME_MAP = {
    "USA":                "United States",
    "Korea Republic":     "South Korea",
    "Republic of Ireland":"Ireland",
    "Türkiye":            "Turkey",
    "Cape Verde":         "Cabo Verde",
    "Côte d'Ivoire":      "Ivory Coast",
    "Czechia":            "Czech Republic",
    "Curaçao":            "Curacao",
    "Congo DR":           "DR Congo",
    "Congo":              "Republic of the Congo",
}

# Features the model will train on
FEATURES = [
    "neutral",
    "tournament_weight",
    "home_elo",
    "away_elo",
    "elo_diff",
    "home_win5",
    "away_win5",
    "home_gd5",
    "away_gd5",
    "h2h_n",
    "h2h_home_winrate",
    "h2h_home_gd",
]


# ──────────────────────────────────────────────
# 1. DATA FETCHING
# ──────────────────────────────────────────────

def fetch_results():
    """Download the CSV once, cache it locally for speed."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, "results.csv")
    if not os.path.exists(path):
        print("📥  Downloading match data…")
        response = requests.get(RESULTS_URL, timeout=120)
        response.raise_for_status()
        with open(path, "wb") as handle:
            handle.write(response.content)
    return pd.read_csv(path)


def normalize_country(name):
    """Map alternate country names to a single canonical name."""
    if not isinstance(name, str):
        return name
    return NAME_MAP.get(name, name)


def load_results():
    """Fetch + clean the raw results dataframe."""
    results = fetch_results()
    results["home_team"]  = results["home_team"].map(normalize_country)
    results["away_team"]  = results["away_team"].map(normalize_country)
    results["date"]       = pd.to_datetime(results["date"])
    results = results.dropna(subset=["home_score", "away_score"]).copy()
    results["home_score"] = results["home_score"].astype(int)
    results["away_score"] = results["away_score"].astype(int)
    results["neutral"]    = results["neutral"].astype(str).str.upper().eq("TRUE").astype(int)
    return results.sort_values("date").reset_index(drop=True)


# ──────────────────────────────────────────────
# 2. TOURNAMENT WEIGHT
#    (how important / competitive is the match?)
# ──────────────────────────────────────────────

def tournament_weight(name):
    """
    Return a numeric weight for how meaningful a match is.
    World Cup proper = 4, qualifiers = 3, major cups = 3, friendlies = 1.
    The model uses this as a feature so it learns that WC games
    are higher-stakes than a January friendly.
    """
    text = str(name).lower()
    if "fifa world cup" in text and "qualif" not in text:
        return 4
    if "qualif" in text:
        return 3
    big = [
        "uefa nations", "copa america", "afc asian cup",
        "africa cup", "concacaf", "uefa euro", "confederations",
    ]
    if any(token in text for token in big):
        return 3
    if "friendly" in text:
        return 1
    return 2


# ──────────────────────────────────────────────
# 3. ELO RATING SYSTEM
#    (the "turf" = neutral ground logic is here too)
# ──────────────────────────────────────────────

def compute_elo(results):
    """
    Walk through every match in chronological order and compute
    a running ELO rating for every team.

    ELO logic:
    - expected_home  = probability the home team wins according to current ratings
    - score_home     = 1 if home won, 0 if lost, 0.5 if draw
    - margin_multiplier = scales the update by goal difference
    - new rating     = old rating + K * multiplier * (actual − expected)

    The "neutral" column from the dataset tells us if the match was
    on neutral turf (tournament host country, etc.). If NOT neutral,
    the home team gets ELO_HOME_BONUS added before computing expected.
    """
    rating   = {}           # team → current ELO
    n        = len(results)
    home_pre = np.zeros(n)
    away_pre = np.zeros(n)

    for i, r in results.iterrows():
        rh = rating.get(r.home_team, ELO_BASE)
        ra = rating.get(r.away_team, ELO_BASE)

        home_pre[i] = rh
        away_pre[i] = ra

        # Home advantage only if NOT on neutral ground
        bonus = 0 if r.neutral == 1 else ELO_HOME_BONUS

        expected_home = 1 / (1 + 10 ** (-((rh + bonus) - ra) / 400))
        score_home    = 1.0 if r.label == 0 else (0.5 if r.label == 1 else 0.0)
        margin        = abs(int(r.home_score) - int(r.away_score))
        multiplier    = np.log(max(margin, 1) + 1) * (2.2 / (abs(rh - ra) * 0.001 + 2.2))

        rating[r.home_team] = rh + ELO_K * multiplier * (score_home - expected_home)
        rating[r.away_team] = ra + ELO_K * multiplier * ((1 - score_home) - (1 - expected_home))

    results["home_elo"]  = home_pre
    results["away_elo"]  = away_pre
    results["elo_diff"]  = home_pre - away_pre
    return results, rating


# ──────────────────────────────────────────────
# 4. FORM FEATURES (last 5 games per team)
# ──────────────────────────────────────────────

def per_team_long(results):
    """
    Reshape the match table from wide (one row per game)
    to long (two rows per game — one per team).
    This makes rolling per-team stats trivial.
    """
    home = pd.DataFrame({
        "date": results["date"].values,
        "team": results["home_team"].values,
        "opp":  results["away_team"].values,
        "gf":   results["home_score"].values,
        "ga":   results["away_score"].values,
    })
    away = pd.DataFrame({
        "date": results["date"].values,
        "team": results["away_team"].values,
        "opp":  results["home_team"].values,
        "gf":   results["away_score"].values,
        "ga":   results["home_score"].values,
    })
    long = pd.concat([home, away], ignore_index=True)
    long["result"] = np.where(long["gf"] > long["ga"], 1.0,
                     np.where(long["gf"] == long["ga"], 0.5, 0.0))
    long["gd"] = long["gf"] - long["ga"]
    return long


def add_form_features(results):
    """
    For each match add rolling-5-game win rate and goal difference
    for both teams (lagged — we use stats BEFORE the match).
    """
    long = per_team_long(results).sort_values(["team", "date"]).reset_index(drop=True)
    long["prev_date"]   = long.groupby("team")["date"].shift(1)
    long["result_lag"]  = long.groupby("team")["result"].shift(1)
    long["gd_lag"]      = long.groupby("team")["gd"].shift(1)
    long["win5"]  = long.groupby("team")["result_lag"].transform(
                        lambda s: s.rolling(5, min_periods=1).mean())
    long["gd5"]   = long.groupby("team")["gd_lag"].transform(
                        lambda s: s.rolling(5, min_periods=1).mean())

    home_form = long.rename(columns={"team": "home_team", "win5": "home_win5", "gd5": "home_gd5"})
    away_form = long.rename(columns={"team": "away_team", "win5": "away_win5", "gd5": "away_gd5"})

    results = results.merge(
        home_form[["date", "home_team", "home_win5", "home_gd5"]],
        on=["date", "home_team"], how="left"
    )
    results = results.merge(
        away_form[["date", "away_team", "away_win5", "away_gd5"]],
        on=["date", "away_team"], how="left"
    )
    return results


# ──────────────────────────────────────────────
# 5. HEAD-TO-HEAD FEATURES
# ──────────────────────────────────────────────

def add_h2h_features(results):
    """
    For each match, look at historical head-to-head record
    between the two teams (from the home team's perspective).
    """
    h2h_n        = np.zeros(len(results))
    h2h_winrate  = np.full(len(results), 0.5)
    h2h_gd       = np.zeros(len(results))

    for i, row in results.iterrows():
        past = results[
            (results["date"] < row["date"]) &
            (
                ((results["home_team"] == row["home_team"]) & (results["away_team"] == row["away_team"])) |
                ((results["home_team"] == row["away_team"]) & (results["away_team"] == row["home_team"]))
            )
        ]
        if len(past) == 0:
            continue
        wins = 0
        gd   = 0
        for _, p in past.iterrows():
            if p["home_team"] == row["home_team"]:
                wins += (p["label"] == 0)
                gd   += p["home_score"] - p["away_score"]
            else:
                wins += (p["label"] == 2)
                gd   += p["away_score"] - p["home_score"]
        h2h_n[i]       = len(past)
        h2h_winrate[i] = wins / len(past)
        h2h_gd[i]      = gd / len(past)

    results["h2h_n"]           = h2h_n
    results["h2h_home_winrate"] = h2h_winrate
    results["h2h_home_gd"]      = h2h_gd
    return results


# ──────────────────────────────────────────────
# 6. BUILD THE FULL FEATURE SET
# ──────────────────────────────────────────────

def build_dataset(results):
    """
    Combine everything: labels, ELO, tournament weight, form, h2h.
    Returns a clean dataframe ready for ML training.
    """
    # Label: 0=home win, 1=draw, 2=away win
    results["label"] = np.where(
        results["home_score"] > results["away_score"], 0,
        np.where(results["home_score"] == results["away_score"], 1, 2)
    )

    results["tournament_weight"] = results["tournament"].apply(tournament_weight)

    # ELO (needs label already set)
    results, final_ratings = compute_elo(results)

    # Form
    results = add_form_features(results)

    # H2H (slow for large datasets — uses recent matches only for speed)
    results_recent = results[results["date"] >= "2000-01-01"].copy().reset_index(drop=True)
    results_recent = add_h2h_features(results_recent)

    results_recent = results_recent.dropna(subset=FEATURES)
    return results_recent, final_ratings


# ──────────────────────────────────────────────
# 7. MODEL TRAINING
# ──────────────────────────────────────────────

def train_model(results_recent):
    """
    Train an XGBoost classifier on 80% of data, validate on 20%.
    XGBoost works great here because:
    - handles mixed numerical features
    - gives feature importances
    - outputs calibrated probabilities
    """
    split = int(len(results_recent) * 0.8)
    train = results_recent.iloc[:split]
    val   = results_recent.iloc[split:]

    X_train, y_train = train[FEATURES], train["label"]
    X_val,   y_val   = val[FEATURES],   val["label"]

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    preds = model.predict(X_val)
    probs = model.predict_proba(X_val)

    print(f"\nValidation accuracy: {accuracy_score(y_val, preds):.3f}")
    print(f"Validation log-loss: {log_loss(y_val, probs):.3f}")
    print(classification_report(y_val, preds, target_names=["home_win", "draw", "away_win"]))

    importances = dict(zip(FEATURES, model.feature_importances_))
    print("Top features by importance:")
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        print(f"  {feat:<25} {imp:.3f}")

    return model


# ──────────────────────────────────────────────
# 8. PREDICTION
# ──────────────────────────────────────────────

def predict_match(
    model,
    results_recent,
    final_ratings,
    home_team,
    away_team,
    neutral=True,
    tournament="FIFA World Cup",
    match_label="",
):
    """
    Predict probabilities for a single match.
    We build a one-row feature vector the same way we built training data.
    """
    rh = final_ratings.get(home_team, ELO_BASE)
    ra = final_ratings.get(away_team, ELO_BASE)

    bonus = 0 if neutral else ELO_HOME_BONUS
    elo_diff = (rh + bonus) - ra

    # Last 5 form for each team
    def team_form(team):
        played = results_recent[
            (results_recent["home_team"] == team) | (results_recent["away_team"] == team)
        ].tail(5)
        wins, gds = [], []
        for _, r in played.iterrows():
            if r["home_team"] == team:
                gds.append(r["home_score"] - r["away_score"])
                wins.append(1.0 if r["label"] == 0 else (0.5 if r["label"] == 1 else 0.0))
            else:
                gds.append(r["away_score"] - r["home_score"])
                wins.append(1.0 if r["label"] == 2 else (0.5 if r["label"] == 1 else 0.0))
        return (np.mean(wins) if wins else 0.5), (np.mean(gds) if gds else 0.0)

    home_win5, home_gd5 = team_form(home_team)
    away_win5, away_gd5 = team_form(away_team)

    # H2H
    h2h = results_recent[
        ((results_recent["home_team"] == home_team) & (results_recent["away_team"] == away_team)) |
        ((results_recent["home_team"] == away_team) & (results_recent["away_team"] == home_team))
    ]
    h2h_n, h2h_wr, h2h_gd = 0, 0.5, 0.0
    if len(h2h) > 0:
        wins, gds = 0, 0
        for _, r in h2h.iterrows():
            if r["home_team"] == home_team:
                wins += (r["label"] == 0)
                gds  += r["home_score"] - r["away_score"]
            else:
                wins += (r["label"] == 2)
                gds  += r["away_score"] - r["home_score"]
        h2h_n  = len(h2h)
        h2h_wr = wins / h2h_n
        h2h_gd = gds  / h2h_n

    row = pd.DataFrame([{
        "neutral":            int(neutral),
        "tournament_weight":  tournament_weight(tournament),
        "home_elo":           rh,
        "away_elo":           ra,
        "elo_diff":           elo_diff,
        "home_win5":          home_win5,
        "away_win5":          away_win5,
        "home_gd5":           home_gd5,
        "away_gd5":           away_gd5,
        "h2h_n":              h2h_n,
        "h2h_home_winrate":   h2h_wr,
        "h2h_home_gd":        h2h_gd,
    }])

    probs = model.predict_proba(row[FEATURES])[0]
    p_home_win, p_draw, p_away_win = probs

    label_str = f"\n{match_label}" if match_label else ""
    print(label_str)
    print(f"{home_team} vs {away_team}")
    print(f"  P({home_team} win) = {p_home_win:.3f}")
    print(f"  P(Draw)            = {p_draw:.3f}")
    print(f"  P({away_team} win) = {p_away_win:.3f}")
    best = np.argmax(probs)
    outcomes = [f"{home_team}", "Draw", f"{away_team}"]
    print(f"  Predicted outcome: {outcomes[best]} ({max(probs)*100:.1f}%)")

    return {"home_win": p_home_win, "draw": p_draw, "away_win": p_away_win}


# ──────────────────────────────────────────────
# 9. POISSON SCORE PREDICTOR
#    Predicts HOW MANY goals each team scores
# ──────────────────────────────────────────────

SCORE_FEATURES = [
    "elo_diff",
    "tournament_weight",
    "neutral",
    "home_win5",
    "home_gd5",
    "away_win5",
    "away_gd5",
]

def train_score_model(results_recent):
    """
    Train two Poisson regressors:
      - one to predict goals scored by the 'home' team
      - one to predict goals scored by the 'away' team

    Poisson regression is the standard tool for count data like goals.
    It learns: given ELO diff, form, tournament importance →
    what is the expected number of goals?
    """
    df = results_recent.dropna(subset=SCORE_FEATURES + ["home_score", "away_score"]).copy()
    split = int(len(df) * 0.8)
    train = df.iloc[:split]

    X_train = train[SCORE_FEATURES]

    home_model = PoissonRegressor(alpha=0.1, max_iter=300)
    away_model = PoissonRegressor(alpha=0.1, max_iter=300)

    home_model.fit(X_train, train["home_score"])
    away_model.fit(X_train, train["away_score"])

    return home_model, away_model


def predict_scoreline(
    home_model,
    away_model,
    results_recent,
    final_ratings,
    home_team,
    away_team,
    neutral=True,
    tournament="FIFA World Cup",
    simulations=50_000,
    top_n=8,
):
    """
    Use the Poisson models to get expected goals, then simulate
    `simulations` matches by drawing from Poisson distributions.
    This gives us a full probability table for every scoreline.

    Why simulate instead of just using the expected value?
    Because goals are discrete (you can't score 1.87 goals) and
    the distribution is asymmetric — we want to capture the full
    range of possible outcomes, not just the average.
    """
    rh = final_ratings.get(home_team, ELO_BASE)
    ra = final_ratings.get(away_team, ELO_BASE)
    elo_diff = rh - ra

    def team_form(team):
        played = results_recent[
            (results_recent["home_team"] == team) | (results_recent["away_team"] == team)
        ].tail(5)
        wins, gds = [], []
        for _, r in played.iterrows():
            if r["home_team"] == team:
                gds.append(r["home_score"] - r["away_score"])
                wins.append(1.0 if r["label"] == 0 else (0.5 if r["label"] == 1 else 0.0))
            else:
                gds.append(r["away_score"] - r["home_score"])
                wins.append(1.0 if r["label"] == 2 else (0.5 if r["label"] == 1 else 0.0))
        return (np.mean(wins) if wins else 0.5), (np.mean(gds) if gds else 0.0)

    home_win5, home_gd5 = team_form(home_team)
    away_win5, away_gd5 = team_form(away_team)

    row = pd.DataFrame([{
        "elo_diff":           elo_diff,
        "tournament_weight":  tournament_weight(tournament),
        "neutral":            int(neutral),
        "home_win5":          home_win5,
        "home_gd5":           home_gd5,
        "away_win5":          away_win5,
        "away_gd5":           away_gd5,
    }])

    # Expected goals from Poisson models
    lambda_home = float(home_model.predict(row[SCORE_FEATURES])[0])
    lambda_away = float(away_model.predict(row[SCORE_FEATURES])[0])

    # Clamp to reasonable range (models can sometimes go wild)
    lambda_home = np.clip(lambda_home, 0.3, 5.0)
    lambda_away = np.clip(lambda_away, 0.3, 5.0)

    # Monte Carlo simulation — draw 50k matches from the Poisson distributions
    np.random.seed(42)
    home_goals = np.random.poisson(lambda_home, simulations)
    away_goals = np.random.poisson(lambda_away, simulations)

    # Count how often each scoreline appeared
    from collections import Counter
    scorelines = Counter(zip(home_goals, away_goals))
    total = simulations

    # Sort by frequency
    sorted_scores = sorted(scorelines.items(), key=lambda x: -x[1])

    # Aggregate win/draw/loss probabilities from simulation
    home_wins = sum(v for (h, a), v in scorelines.items() if h > a) / total
    draws     = sum(v for (h, a), v in scorelines.items() if h == a) / total
    away_wins = sum(v for (h, a), v in scorelines.items() if h < a) / total

    print(f"\n{'='*50}")
    print(f"  SCORELINE PREDICTION")
    print(f"  {home_team} vs {away_team}")
    print(f"{'='*50}")
    print(f"  Expected goals: {home_team} {lambda_home:.2f} — {away_team} {lambda_away:.2f}")
    print(f"\n  {'Scoreline':<12} {'Result':<12} {'Probability'}")
    print(f"  {'-'*38}")

    for (h, a), count in sorted_scores[:top_n]:
        prob = count / total * 100
        if h > a:
            result = f"{home_team} win"
        elif h == a:
            result = "Draw"
        else:
            result = f"{away_team} win"
        print(f"  {h}-{a:<11} {result:<16} {prob:.1f}%")

    print(f"\n  Simulated match odds ({simulations:,} games):")
    print(f"    {home_team} win : {home_wins*100:.1f}%")
    print(f"    Draw           : {draws*100:.1f}%")
    print(f"    {away_team} win : {away_wins*100:.1f}%")

    return {
        "lambda_home": lambda_home,
        "lambda_away": lambda_away,
        "home_win": home_wins,
        "draw": draws,
        "away_win": away_wins,
        "scorelines": sorted_scores[:top_n],
    }


# ──────────────────────────────────────────────
# 10. RUN EVERYTHING
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=== World Cup 2026 Predictor ===\n")

    print("Loading & cleaning data…")
    results = load_results()

    print("Building features…")
    results_recent, final_ratings = build_dataset(results)

    print("Training classifier…")
    model = train_model(results_recent)

    print("Training score predictor…")
    home_score_model, away_score_model = train_score_model(results_recent)

    print("\n" + "="*45)
    print("BRAZIL vs MOROCCO — June 13, 2026")
    print("="*45)

    predict_match(
        model,
        results_recent,
        final_ratings,
        home_team="Brazil",
        away_team="Morocco",
        neutral=True,
        tournament="FIFA World Cup",
        match_label="Group Stage, 2026 FIFA World Cup",
    )

    predict_scoreline(
        home_score_model,
        away_score_model,
        results_recent,
        final_ratings,
        home_team="Brazil",
        away_team="Morocco",
        neutral=True,
        tournament="FIFA World Cup",
    )