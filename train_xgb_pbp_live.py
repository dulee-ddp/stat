from __future__ import annotations

import argparse
import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from nba_api.stats.library.http import NBAStatsHTTP
from nba_api.stats.endpoints import leaguegamelog, playbyplayv2
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score

from xgboost import XGBClassifier


# -------------------------
# Logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


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


def _cdn_get_json(url: str, timeout: int = 10) -> dict:
    s = _cdn_session()
    r = s.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


def fetch_pbp_from_cdn(game_id: str) -> dict:
    url = f"https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{game_id}.json"
    return _cdn_get_json(url, timeout=10)



# -------------------------
# Resilient stats.nba.com session (matches your app.py style)
def configure_stats_session(timeout: int = 20) -> None:
    s = Session()
    retry = Retry(
        total=8,
        read=8,
        connect=8,
        backoff_factor=0.7,
        status_forcelist=[429, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=50, pool_maxsize=50)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update(
        {
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://www.nba.com/",
            "Origin": "https://www.nba.com",
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
        }
    )
    NBAStatsHTTP.timeout = timeout
    NBAStatsHTTP._session = s


def safe_call(fn, *args, max_retries: int = 6, base_pause: float = 0.9, **kwargs):
    for i in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except (requests.exceptions.RequestException, KeyError, ValueError) as e:
            sleep_s = base_pause * (1.6 ** i) + random.uniform(0.05, 0.35)
            logging.warning(
                "%s failed %s/%s: %s -> sleep %.1fs",
                getattr(fn, "__name__", str(fn)),
                i + 1,
                max_retries,
                e,
                sleep_s,
            )
            time.sleep(sleep_s)
    return fn(*args, **kwargs)


# -------------------------
# Helpers: time/score parsing
def parse_pbp_clock_to_sec_left_in_period(clock_str: str, period: int) -> Optional[int]:
    """
    PCTIMESTRING format is usually 'MM:SS' remaining in the period.
    Returns seconds remaining in current period.
    """
    if not clock_str or not isinstance(clock_str, str):
        return None
    m = clock_str.strip().split(":")
    if len(m) != 2:
        return None
    try:
        mm = int(m[0])
        ss = int(m[1])
        return mm * 60 + ss
    except Exception:
        return None


def period_length_seconds(period: int) -> int:
    # NBA: 12 min quarters (1-4), 5 min OT (5+)
    return 12 * 60 if period <= 4 else 5 * 60


def compute_seconds_left_game(period: int, sec_left_period: int) -> int:
    """
    Convert (period, sec_left_period) to seconds left in game.
    Regulation game length = 4*12*60 = 2880 seconds.
    For OT, this returns seconds left incl OT segments remaining in that OT period only.
    We still include OT by adding remaining regulation (0) + remaining OT time in current OT.
    """
    if period <= 4:
        # seconds remaining in current period + full future periods
        future_periods = 4 - period
        return sec_left_period + future_periods * 12 * 60
    else:
        # Overtime: only track remaining in current OT (game can end within this)
        return sec_left_period


def parse_score(score_str: str) -> Optional[Tuple[int, int]]:
    """
    Score string is usually like '102 - 98' in PBP.
    Return (home_score, away_score).
    """
    if not score_str or not isinstance(score_str, str):
        return None
    s = score_str.replace(" ", "")
    if "-" not in s:
        return None
    parts = s.split("-")
    if len(parts) != 2:
        return None
    try:
        a = int(parts[0])
        b = int(parts[1])
        # nba_api PBP SCORE is usually "VISITOR - HOME" OR "HOME - VISITOR" depending on feed.
        # To avoid guessing wrong, we will infer mapping using FINAL row logic later.
        # For now return raw pair (left, right).
        return (a, b)
    except Exception:
        return None


# -------------------------
# Get unique games for a season (regular season)
def get_season_games(season_label: str) -> pd.DataFrame:
    """
    Uses LeagueGameLog to get team-game rows, then collapses to unique GAME_ID list.
    """
    logging.info("Fetching season game logs for %s ...", season_label)
    resp = safe_call(
        leaguegamelog.LeagueGameLog,
        season=season_label,
        season_type_all_star="Regular Season",
        league_id="00",
        sorter="DATE",
        direction="ASC",
        counter=0,
    )
    df = resp.get_data_frames()[0].copy()
    # Unique game ids
    g = (
        df[["GAME_ID", "GAME_DATE"]]
        .drop_duplicates()
        .sort_values(["GAME_DATE", "GAME_ID"])
        .reset_index(drop=True)
    )
    g["SEASON"] = season_label
    return g


# -------------------------
# Pull PBP and build snapshots
@dataclass
class GamePBPResult:
    game_id: str
    season: str
    ok: bool
    error: Optional[str]
    snapshots: Optional[pd.DataFrame]


def build_game_snapshots_from_pbp(game_id: str, season: str) -> GamePBPResult:
    """
    Build in-game snapshots from NBA CDN play-by-play.
    Label = final HOME win (1/0).
    """
    try:
        js = fetch_pbp_from_cdn(game_id)
        game = js.get("game", {}) or {}
        actions = game.get("actions", []) or []
    except Exception as e:
        return GamePBPResult(game_id, season, False, f"CDN PBP fetch failed: {e}", None)

    if not actions:
        return GamePBPResult(game_id, season, False, "empty actions", None)

    # Build rows from actions with known scores
    rows = []
    for a in actions:
        # These keys exist in CDN actions most of the time
        period = a.get("period")
        clock = a.get("clock")  # "PT11M32.00S" format often
        hscore = a.get("scoreHome")
        ascore = a.get("scoreAway")

        if period is None or hscore is None or ascore is None:
            continue

        # Parse clock like "PT11M32.00S" -> seconds left in period
        sec_left_period = None
        if isinstance(clock, str):
            import re
            m = re.match(r"PT(\d+)M(\d+(?:\.\d+)?)S", clock)
            if m:
                mm = int(m.group(1))
                ss = int(float(m.group(2)))
                sec_left_period = mm * 60 + ss

        if sec_left_period is None:
            continue

        period = int(period)
        hscore = int(hscore)
        ascore = int(ascore)

        sec_left_game = compute_seconds_left_game(period, sec_left_period)

        rows.append(
            {
                "period": period,
                "sec_left_period": sec_left_period,
                "sec_left_game": sec_left_game,
                "home_score": hscore,
                "away_score": ascore,
            }
        )

    if not rows:
        return GamePBPResult(game_id, season, False, "no valid scored rows", None)

    df = pd.DataFrame(rows).dropna().reset_index(drop=True)

    # Final label
    final_home = int(df["home_score"].iloc[-1])
    final_away = int(df["away_score"].iloc[-1])
    y_home_win = int(final_home > final_away)

    df["score_diff"] = df["home_score"] - df["away_score"]
    df["abs_diff"] = df["score_diff"].abs()

    # Time fraction remaining in regulation
    df["reg_sec_left"] = np.where(df["period"] <= 4, df["sec_left_game"], 0)
    df["reg_frac_left"] = (df["reg_sec_left"] / (4 * 12 * 60)).clip(0.0, 1.0)

    # Trend features
    df["diff_delta"] = df["score_diff"].diff().fillna(0.0)
    df["run_5"] = df["diff_delta"].rolling(window=5, min_periods=1).sum()
    df["run_15"] = df["diff_delta"].rolling(window=15, min_periods=1).sum()

    # Period indicators
    df["is_q1"] = (df["period"] == 1).astype(int)
    df["is_q2"] = (df["period"] == 2).astype(int)
    df["is_q3"] = (df["period"] == 3).astype(int)
    df["is_q4"] = (df["period"] == 4).astype(int)
    df["is_ot"] = (df["period"] >= 5).astype(int)

    df["game_id"] = str(game_id)
    df["season"] = str(season)
    df["y_home_win"] = y_home_win

    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    if df.empty:
        return GamePBPResult(game_id, season, False, "no valid snapshots after cleaning", None)

    # Keep only the columns your trainer expects
    snaps = df[
        [
            "period",
            "sec_left_period",
            "sec_left_game",
            "reg_frac_left",
            "home_score",
            "away_score",
            "score_diff",
            "abs_diff",
            "diff_delta",
            "run_5",
            "run_15",
            "is_q1",
            "is_q2",
            "is_q3",
            "is_q4",
            "is_ot",
            "game_id",
            "season",
            "y_home_win",
        ]
    ].copy()

    return GamePBPResult(game_id, season, True, None, snaps)


# -------------------------
# Training
def train_xgb_pbp_live(
    last_season: str = "2024-25",
    current_season: str = "2025-26",
    w_last: float = 0.15,
    w_cur: float = 0.85,
    max_games: Optional[int] = None,
    out_path: str = "models/xgb_pbp_live.pkl",
) -> None:
    configure_stats_session()

    g_last = get_season_games(last_season)
    g_cur = get_season_games(current_season)

    # Combine game ids (unique)
    games = pd.concat([g_last, g_cur], ignore_index=True)
    games = games.drop_duplicates(subset=["GAME_ID"]).reset_index(drop=True)

    if max_games is not None and max_games > 0:
        games = games.head(max_games)

    logging.info("Total games to process: %d", len(games))

    all_snaps: List[pd.DataFrame] = []
    failures = 0

    for i, row in games.iterrows():
        gid = str(row["GAME_ID"])
        season = str(row["SEASON"])
        if (i + 1) % 25 == 0:
            logging.info("Progress: %d/%d games ...", i + 1, len(games))

        res = build_game_snapshots_from_pbp(gid, season)
        if not res.ok or res.snapshots is None:
            failures += 1
            continue
        all_snaps.append(res.snapshots)

        # light pacing (stats api is sensitive)
        time.sleep(0.25 + random.uniform(0.0, 0.25))

    if not all_snaps:
        raise RuntimeError("No snapshots collected. Try increasing retries or using --max-games small for debugging.")

    df = pd.concat(all_snaps, ignore_index=True)
    logging.info("Snapshots collected: %d (failed games: %d)", len(df), failures)

    # Sample weights by season
    df["sample_weight"] = np.where(df["season"] == current_season, w_cur, w_last).astype(float)

    # Features + label + groups
    y = df["y_home_win"].astype(int)
    groups = df["game_id"].astype(str)

    feature_cols = [
        "period",
        "sec_left_period",
        "sec_left_game",
        "reg_frac_left",
        "home_score",
        "away_score",
        "score_diff",
        "abs_diff",
        "diff_delta",
        "run_5",
        "run_15",
        "is_q1",
        "is_q2",
        "is_q3",
        "is_q4",
        "is_ot",
    ]
    X = df[feature_cols].copy()

    # Train/val split by GAME (prevents leakage from same game)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, val_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    sw_train = df["sample_weight"].iloc[train_idx]
    sw_val = df["sample_weight"].iloc[val_idx]

    # XGBoost model (tuned to be stable, not overkill)
    model = XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        min_child_weight=2,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=8,
        random_state=42,
    )

    logging.info("Training XGBoost live PBP model ...")
    model.fit(
        X_train,
        y_train,
        sample_weight=sw_train,
        eval_set=[(X_val, y_val)],
        sample_weight_eval_set=[sw_val],
        verbose=False,
    )

    # Eval
    proba = model.predict_proba(X_val)[:, 1]
    pred = (proba >= 0.5).astype(int)

    ll = log_loss(y_val, proba, sample_weight=sw_val)
    auc = roc_auc_score(y_val, proba, sample_weight=sw_val)
    acc = accuracy_score(y_val, pred, sample_weight=sw_val)

    logging.info("Validation logloss: %.4f", ll)
    logging.info("Validation AUC:     %.4f", auc)
    logging.info("Validation ACC:     %.4f", acc)

    # Save bundle
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model": model,
        "feature_names": feature_cols,
        "last_season": last_season,
        "current_season": current_season,
        "weights": {"last": w_last, "current": w_cur},
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        # This tells your live predictor how we inferred home/away mapping.
        # You may not need it, but it's useful if you later enforce mapping.
        "notes": "Trained from PlayByPlayV2 score states; label=final home win. Features are in-game only.",
    }

    joblib.dump(bundle, out)
    logging.info("Saved model bundle -> %s", str(out.resolve()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--last-season", default="2024-25")
    ap.add_argument("--current-season", default="2025-26")
    ap.add_argument("--w-last", type=float, default=0.15)
    ap.add_argument("--w-cur", type=float, default=0.85)
    ap.add_argument("--max-games", type=int, default=None, help="Use for quick testing (e.g. 50).")
    ap.add_argument("--out", default="models/xgb_pbp_live.pkl")
    args = ap.parse_args()

    train_xgb_pbp_live(
        last_season=args.last_season,
        current_season=args.current_season,
        w_last=args.w_last,
        w_cur=args.w_cur,
        max_games=args.max_games,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
