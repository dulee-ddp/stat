from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
import re


# -------------------------
# CDN fetch
def _cdn_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
            "Origin": "https://www.nba.com",
            "Referer": "https://www.nba.com/",
        }
    )
    return s


def fetch_pbp_from_cdn(game_id: str, timeout: int = 10) -> dict:
    url = f"https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{game_id}.json"
    s = _cdn_session()
    r = s.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


# -------------------------
# Clock parsing: "PT11M32.00S" -> seconds left in period
_CLOCK_RE = re.compile(r"PT(\d+)M(\d+(?:\.\d+)?)S")


def _clock_to_sec_left(clock: Any) -> Optional[int]:
    if not isinstance(clock, str):
        return None
    m = _CLOCK_RE.match(clock)
    if not m:
        return None
    mm = int(m.group(1))
    ss = int(float(m.group(2)))
    return mm * 60 + ss


def _period_length_seconds(period: int) -> int:
    return 12 * 60 if period <= 4 else 5 * 60


def _compute_seconds_left_game(period: int, sec_left_period: int) -> int:
    # Regulation: include future quarters remaining
    if period <= 4:
        future_periods = 4 - period
        return sec_left_period + future_periods * 12 * 60
    # OT: just return remaining in current OT
    return sec_left_period


# -------------------------
# Build “snapshot rows” from CDN actions (must match training)
def _actions_to_scored_rows(actions: List[dict]) -> pd.DataFrame:
    rows = []
    for a in actions:
        period = a.get("period")
        hscore = a.get("scoreHome")
        ascore = a.get("scoreAway")
        clock = a.get("clock")

        if period is None or hscore is None or ascore is None:
            continue

        sec_left_period = _clock_to_sec_left(clock)
        if sec_left_period is None:
            continue

        period = int(period)
        hscore = int(hscore)
        ascore = int(ascore)

        rows.append(
            {
                "period": period,
                "sec_left_period": sec_left_period,
                "sec_left_game": _compute_seconds_left_game(period, sec_left_period),
                "home_score": hscore,
                "away_score": ascore,
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # IMPORTANT: actions can repeat identical score states.
    # Keep only changes in score OR time to reduce noise.
    df = df.sort_values(["period", "sec_left_period"], ascending=[True, False]).reset_index(drop=True)

    # Drop exact duplicates
    df = df.drop_duplicates(subset=["period", "sec_left_period", "home_score", "away_score"])

    return df.reset_index(drop=True)


def _add_training_features(df: pd.DataFrame) -> pd.DataFrame:
    # These match File #2 training
    df = df.copy()

    df["score_diff"] = df["home_score"] - df["away_score"]
    df["abs_diff"] = df["score_diff"].abs()

    # fraction remaining in regulation (OT -> 0)
    df["reg_sec_left"] = np.where(df["period"] <= 4, df["sec_left_game"], 0)
    df["reg_frac_left"] = (df["reg_sec_left"] / (4 * 12 * 60)).clip(0.0, 1.0)

    # trend features
    df["diff_delta"] = df["score_diff"].diff().fillna(0.0)
    df["run_5"] = df["diff_delta"].rolling(window=5, min_periods=1).sum()
    df["run_15"] = df["diff_delta"].rolling(window=15, min_periods=1).sum()

    # period indicators
    df["is_q1"] = (df["period"] == 1).astype(int)
    df["is_q2"] = (df["period"] == 2).astype(int)
    df["is_q3"] = (df["period"] == 3).astype(int)
    df["is_q4"] = (df["period"] == 4).astype(int)
    df["is_ot"] = (df["period"] >= 5).astype(int)

    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return df


# -------------------------
# Load bundle + predict from latest snapshot
def load_cdn_bundle(path: str = "models/xgb_pbp_live.pkl") -> Dict[str, Any]:
    return joblib.load(path)


def predict_live_from_cdn(
    game_id: str,
    bundle_path: str = "models/xgb_pbp_live.pkl",
) -> Dict[str, Any]:
    bundle = load_cdn_bundle(bundle_path)
    model = bundle["model"]
    feat_cols: List[str] = bundle["feature_names"]

    try:
        js = fetch_pbp_from_cdn(game_id)
        game = js.get("game", {}) or {}
        actions = game.get("actions", []) or []
    except Exception as e:
        return {"gameId": game_id, "ok": False, "error": f"CDN fetch failed: {e}"}

    if not actions:
        return {"gameId": game_id, "ok": False, "error": "No actions yet (game may not have started)"}

    df = _actions_to_scored_rows(actions)
    if df.empty:
        return {"gameId": game_id, "ok": False, "error": "No scored rows yet (no score updates)"}

    df = _add_training_features(df)
    if df.empty:
        return {"gameId": game_id, "ok": False, "error": "No valid rows after feature build/cleaning"}

    last = df.iloc[-1]
    x = pd.DataFrame([{c: float(last[c]) for c in feat_cols}])
    p_home = float(model.predict_proba(x)[0, 1])

    return {
        "gameId": game_id,
        "ok": True,
        "period": int(last["period"]),
        "secLeftPeriod": int(last["sec_left_period"]),
        "homeScore": int(last["home_score"]),
        "awayScore": int(last["away_score"]),
        "homeWinProb": p_home,
        "awayWinProb": 1.0 - p_home,
        "features": {c: float(last[c]) for c in feat_cols},  # super useful for debugging
    }

def build_live_curve_from_cdn(
    game_id: str,
    bundle_path: str = "models/xgb_pbp_live.pkl",
    max_points: int = 500,
) -> Dict[str, Any]:
    bundle = load_cdn_bundle(bundle_path)
    model = bundle["model"]
    feat_cols: List[str] = bundle["feature_names"]

    js = fetch_pbp_from_cdn(game_id)
    game = js.get("game", {}) or {}
    actions = game.get("actions", []) or []

    if not actions:
        return {"gameId": game_id, "ok": False, "error": "No actions yet"}

    df = _actions_to_scored_rows(actions)
    if df.empty:
        return {"gameId": game_id, "ok": False, "error": "No scored rows yet"}

    df = _add_training_features(df)
    if df.empty:
        return {"gameId": game_id, "ok": False, "error": "No valid rows after feature build"}

    # Predict prob for every row (vectorized)
    X = df[feat_cols].astype(float)
    p = model.predict_proba(X)[:, 1]

    # Build points (from start->now). Keep order by game time (earliest -> latest)
    # Your df currently sorts within _actions_to_scored_rows; ensure final order:
    df = df.reset_index(drop=True)
    points = []
    for i in range(len(df)):
        row = df.iloc[i]
        points.append(
            {
                "period": int(row["period"]),
                "secLeftPeriod": int(row["sec_left_period"]),
                "secLeftGame": int(row["sec_left_game"]),
                "homeScore": int(row["home_score"]),
                "awayScore": int(row["away_score"]),
                "homeWinProb": float(p[i]),
            }
        )

    # Downsample if huge (keeps shape)
    if len(points) > max_points:
        step = max(1, len(points) // max_points)
        points = points[::step]
        if points[-1] != points[-1]:
            pass

    return {"gameId": game_id, "ok": True, "points": points}
