import os  # Standard library for environment variables and file paths
import sqlite3  # SQLite library
import time  # Time utilities (used for sleeping/backoff in other helpers)
import random  # Random utilities (used for jitter in retry logic)
import logging
import boto3
import io
import json
from predict_today_svm import get_predictions_with_features_for_date
from pathlib import Path  # Path helper for building filesystem paths
from datetime import datetime, date  # datetime and date classes
from zoneinfo import ZoneInfo  # Like (America/New_York)
from typing import Any, Dict, List  # Type hints for dictionaries and lists
from predict_today_svm import get_predictions_for_date, get_prediction_for_game
from explainability_from_app import explain_predictions, shap_rows_for_date

import pandas as pd
import requests  # Requests for HTTP calls (to NBA CDN and others)
from flask import Flask, jsonify, request  # Flask core app object and helpers for JSON + query params
from flask_cors import CORS  # CORS support so React frontend can call our API
from flask_caching import Cache  # Simple in-memory caching for expensive endpoints
from nba_api.stats.endpoints import (  # nba_api endpoints we use in various routes
    boxscoretraditionalv2,
    leaguegamefinder,
    leaguestandingsv3,
    scoreboardv2,
)
from nba_api.stats.library.http import NBAStatsHTTP
from nba_api.stats.static import teams as static_teams
from nba_api.stats.static import teams
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry  # Retry strategy helper
from dotenv import load_dotenv
from openai import OpenAI
from prediction_store import get_predictions_cached_for_date
from explainability_from_app import explain_predictions
from pbp_live_xgb import (
    predict_live_from_cdn, 
    fetch_pbp_from_cdn, 
    load_cdn_bundle, 
    _actions_to_scored_rows, 
    _add_training_features
)
from train_xgb_pbp_live import train_xgb_pbp_live


# openai api
load_dotenv()
# verifying ig our key is safely stored and backend can access it


# Configuration

# Use same DB path (override with NBA_DB_PATH if set)
DB_PATH = (
    os.environ.get("NBA_DB_PATH")  # If an environment variable is set, use that as the DB path
    or str((Path(__file__).parent / "nba_live.sqlite").resolve())  # Otherwise, default to local nba_live.sqlite
)

app = Flask(__name__)  # Create the Flask application instance
# Allow React (different origin) to call our Flask API
CORS(app, resources={r"/api/*": {"origins": "*"}})  # Enable CORS for all routes starting with /api/

# Register picks/leagues/auth blueprint
from picks_blueprint import picks_bp, init_picks_db
app.register_blueprint(picks_bp)
init_picks_db(app)

SWISH_MODEL = os.getenv("SWISH_MODEL", "gpt-5.2")


def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=api_key)


logging.basicConfig(level=logging.INFO)

# Static team lookup cache: TEAM_ID -> {"tri": "...", "name": "..."}
TEAMS_BY_ID: Dict[int, Dict[str, str]] = {}

# ── S3 helpers ────────────────────────────────────────────────────────────────
S3_BUCKET = os.environ.get("S3_BUCKET", "statline-nba-data")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

def _get_s3():
    return boto3.client("s3", region_name=AWS_REGION)

def _read_csv_from_s3(key: str) -> pd.DataFrame:
    try:
        s3 = _get_s3()
        obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
        df = pd.read_csv(io.StringIO(obj["Body"].read().decode("utf-8")))
        return df.where(pd.notnull(df), None)
    except Exception as e:
        logging.error("Failed to read s3://%s/%s: %s", S3_BUCKET, key, e)
        return pd.DataFrame()

GLOSSARY = {
    "BLEND_FG_PCT_DIFF": ("Field goal %", "pp"),
    "BLEND_FG3_PCT_DIFF": ("3-point %", "pp"),
    "BLEND_FT_PCT_DIFF": ("Free throw %", "pp"),
    "BLEND_PTS_DIFF": ("Points per game", "pg"),
    "BLEND_REB_DIFF": ("Rebounds per game", "pg"),
    "BLEND_AST_DIFF": ("Assists per game", "pg"),
    "BLEND_TOV_DIFF": ("Turnovers per game", "pg"),
    "BLEND_PLUS_MINUS_DIFF": ("Plus/minus", "pg"),
    "BLEND_NET_MARGIN_DIFF": ("Net margin", "pg"),
    "RECENT_WIN_PCT_DIFF": ("Recent win rate", "pp"),
    "ELO_PRE_DIFF": ("Elo rating", "elo"),
}

def _fmt_diff(feature: str, val: float) -> str:
    name, unit = GLOSSARY.get(feature, (feature, "raw"))
    if unit == "pp":
        return f"{val * 100:+.1f} pp"
    if unit == "elo":
        return f"{val:+.0f}"
    if unit == "pg":
        return f"{val:+.1f}"
    return f"{val:+.3f}"

def _pick_top_diffs(feature_row: dict, k: int = 5) -> list[dict]:
    # Always include Elo + recent if present, then biggest absolute diffs
    must = [f for f in ["ELO_PRE_DIFF", "RECENT_WIN_PCT_DIFF"] if f in feature_row]
    remaining = [f for f in feature_row.keys() if f not in set(must)]
    remaining = sorted(remaining, key=lambda f: abs(float(feature_row.get(f, 0.0))), reverse=True)

    chosen = must + [f for f in remaining if f not in must]
    chosen = chosen[:k]

    out = []
    for f in chosen:
        v = float(feature_row.get(f, 0.0))
        friendly, _ = GLOSSARY.get(f, (f, "raw"))
        out.append({
            "feature": f,
            "friendly": friendly,
            "diff": v,  # home - away
            "diff_fmt": _fmt_diff(f, v),
            "edge": "home" if v > 0 else ("away" if v < 0 else "even"),
        })
    return out

def _pick_top_by_shap(feature_row: dict, shap_row: dict | None, k: int = 5) -> list[dict]:
    """
    Uses SHAP to choose which features to show, but uses your existing diff formatting
    so the bullets stay in human units.

    shap_row values:
      + -> pushes model toward HOME win
      - -> pushes model toward AWAY win
    """
    if not shap_row:
        return _pick_top_diffs(feature_row, k=k)

    must = [f for f in ["ELO_PRE_DIFF", "RECENT_WIN_PCT_DIFF"] if f in feature_row]

    # Sort by absolute shap magnitude
    ranked = sorted(shap_row.items(), key=lambda kv: abs(float(kv[1])), reverse=True)
    ranked_feats = [f for (f, _sv) in ranked if f in feature_row]

    # Prepend must-haves, then take first k unique
    chosen = []
    for f in must + ranked_feats:
        if f not in chosen:
            chosen.append(f)
        if len(chosen) >= k:
            break

    out = []
    for f in chosen:
        v = float(feature_row.get(f, 0.0))
        sv = float(shap_row.get(f, 0.0))  # SHAP for this feature (home class)
        friendly, _ = GLOSSARY.get(f, (f, "raw"))
        out.append({
            "feature": f,
            "friendly": friendly,
            "diff": v,                 # home - away (your original)
            "diff_fmt": _fmt_diff(f, v),
            "edge": "home" if v > 0 else ("away" if v < 0 else "even"),
            "shap": sv,                # add for debugging (optional, but useful)
        })
    return out


# Helpers: teams, dates, sessions

def _ensure_team_lookup() -> None:
    """Populate TEAMS_BY_ID once from nba_api static teams."""
    # If the cache is already populated, do nothing
    if TEAMS_BY_ID:
        return

    # Otherwise, fetch all teams from the static teams endpoint
    for t in static_teams.get_teams():
        tid = int(t["id"])  # Convert team id to int
        # Store abbreviation and full name in the global cache
        TEAMS_BY_ID[tid] = {
            "tri": str(t.get("abbreviation", "")).upper(),  # Uppercase
            "name": t.get("full_name", ""),  # Full team name
        }


def today_et_iso() -> str:
    """Return today's date string in America/New_York as YYYY-MM-DD."""
    # Get current datetime in Eastern Time
    return datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")


def _season_window_for_label(season_label: str) -> tuple[date, date]:
    """Return (start_date, end_date) for a season label like '2025-26'."""
    start_year = int(season_label.split("-")[0])  # Extract the starting year from the season label
    start = date(start_year, 10, 1)  # Define the approximate NBA regular-season start date
    end = date(start_year + 1, 6, 30)  # Define the approximate season end date
    return start, end  # Return the date window as a tuple


def _current_season_label() -> str:
    """
    Return NBA season label like '2025-26'.
    NBA seasons begin around Oct and end the following year.
    """
    now = datetime.now(ZoneInfo("America/New_York"))
    year = now.year  # Extract the current year

    # If we're in August or later, we assume the new season starts this year
    if now.month >= 8:
        start = year  # Season start year
        end = (year + 1) % 100  # end the year with e.g., 2026 -> 26
    else:
        # Before August were still in the season that started last year
        start = year - 1  # Previous year is the season start
        end = year % 100  # Current year as last two digits
    return f"{start}-{end:02d}"


# HTTP sessions (CDN + Stats)


def _cdn_session() -> requests.Session:
    # request session using given user like headers to reduce blockage
    s = requests.Session()
    s.headers.update(
        {
            # chrome
            "User-Agent": "Mozilla/5.0",
            # ask server for json file
            "Accept": "application/json",
            # where the request is coming from
            "Origin": "https://www.nba.com",
            # what page you were previously on
            "Referer": "https://www.nba.com/",
        }
    )
    return s


# get the json from the nba cdn then return it as a dict
def _cdn_get_json(url: str, timeout: int = 8) -> Dict[str, Any]:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    # start the session
    s = _cdn_session()
    # GET request
    r = s.get(url, timeout=timeout, verify=False)
    # error check
    r.raise_for_status()
    return r.json()


#
def _configure_stats_session(timeout: int = 20) -> None:
    """Configure a resilient requests.Session for stats.nba.com via nba_api."""
    s = Session()
    # retry factors if failed
    retry = Retry(
        total=8,
        # read errors
        read=8,
        # connection errors
        connect=8,
        # wait longer each time
        backoff_factor=0.7,
        # common errors
        status_forcelist=[429, 502, 503, 504],
        # only GET
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(
        max_retries=retry, pool_connections=50, pool_maxsize=50)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    # just like in cdn session
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
    # make the api calls use this session always
    NBAStatsHTTP.timeout = timeout
    NBAStatsHTTP._session = s


# call endpoint of nba_api with retires and error check
def _safe_stats_call(fn, *args, max_retries: int = 5,
                     base_pause: float = 0.9, **kwargs):

    for i in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except (requests.exceptions.RequestException,
                KeyError, ValueError) as e:
            sleep_s = base_pause * (1.6 ** i) + random.uniform(0.05, 0.35)
            logging.warning(
                "%s failed %s/%s: %s -> sleep %.1fs",
                fn.__name__, i + 1, max_retries, e, sleep_s
            )
            time.sleep(sleep_s)
    return fn(*args, **kwargs)

# ------ Scoreboard parsing (Stats API base) + CDN patchers
# fetches nba game data (scheduled, live, or completed) for a given date using Scoreboard V2 from nba api
# parses the response to extract game metadata (team, arena, tipoff time) into dictionary format one per game
# function accepts a date string


def _parse_scoreboard_day(mmddyyyy: str) -> List[Dict[str, Any]]:
    # configure a hhtp session and populate the TEAMS_BY_ID dictionary with team metadata
    _configure_stats_session()
    _ensure_team_lookup()
    # fetch the ScoreboardV2 data from nba api for the given date using a safe call wrapper
    resp = _safe_stats_call(
        scoreboardv2.ScoreboardV2,
        game_date=mmddyyyy,
        league_id="00",
        day_offset=0,
    )
    # calls try catch to extract gameheader dataframe that includes basic game info
    # if fails we use empty data frame
    try:
        gh = resp.game_header.get_data_frame()
    except Exception:
        gh = pd.DataFrame()
    # tries to extract the LineScore dataframe, which are the scores for each team, defaults to empty if fails
    try:
        ls = resp.line_score.get_data_frame()
    except Exception:
        ls = pd.DataFrame()
    # initialize a dictionary that maps game IDS to their respective line score rows
    lines_by_game: Dict[str, list] = {}
    # if LineScore is available, it organizes each row under its GAME_ID into the lines_by_game dictionary
    if ls is not None and not ls.empty:
        for _, row in ls.iterrows():
            gid = str(row.get("GAME_ID", "") or "")
            if gid:
                lines_by_game.setdefault(gid, []).append(row)
    # internal helper that will safely select the first valid (non-null) field from a row

    def pick(row, *names, default=None):
        for n in names:
            if n in row and pd.notna(row[n]):
                return row[n]
        return default
    # prepare a list to hold the parsed and cleaned-up game dictionaries
    games: List[Dict[str, Any]] = []
    # if the GameHeader is missing or empty, return an empty result list
    if gh is None or gh.empty:
        return games
    # main loop - parse each game
    seen: set[str] = set()
    # iterate each row (game) in the game headers
    for _, r in gh.iterrows():
        # get the game id and skip if not present
        gid = str(pick(r, "GAME_ID", default="") or "")
        if not gid:
            continue
        if gid in seen:
            continue
        seen.add(gid)
        # fetch the game status info (1 = scheduled, 2 = live, 3 = finished)
        status_id = int(pick(r, "GAME_STATUS_ID", default=1) or 1)
        status_txt = pick(r, "GAME_STATUS_TEXT", default="") or ""
        # extracts the home and away team IDs
        home_id = int(pick(r, "HOME_TEAM_ID") or 0) or None
        away_id = int(pick(r, "VISITOR_TEAM_ID") or 0) or None
        # initialize the team strcutres and scores
        home = {"id": home_id, "name": "", "tri": ""}
        away = {"id": away_id, "name": "", "tri": ""}
        home_score = None
        away_score = None
        # populate the scores if line score data exist
        # loop over each line score row for the game, extract team ID, abbreviation, name, and points
        for lr in lines_by_game.get(gid, []):
            tid = int(lr.get("TEAM_ID") or 0) or None
            tri = str(lr.get("TEAM_ABBREVIATION") or "").upper()
            city = lr.get("TEAM_CITY_NAME") or ""
            nick = lr.get("TEAM_NICKNAME") or ""
            name = f"{city} {nick}".strip()
            pts = lr.get("PTS")
            # checks if this ow is for the home team, if so fill in name, abbreviation, and score
            if tid and home_id and tid == home_id:
                meta = TEAMS_BY_ID.get(home_id, {})
                home["name"] = name or meta.get("name", "")
                home["tri"] = tri or meta.get("tri", "")
                home_score = int(pts) if pd.notna(pts) else None
            # do the same for the away team
            if tid and away_id and tid == away_id:
                meta = TEAMS_BY_ID.get(away_id, {})
                away["name"] = name or meta.get("name", "")
                away["tri"] = tri or meta.get("tri", "")
                away_score = int(pts) if pd.notna(pts) else None
        # fallback if team info is missing
        # lookups for if team names/abbr are missing
        if home_id and (not home["name"] or not home["tri"]):
            meta = TEAMS_BY_ID.get(home_id, {})
            home["name"] = home["name"] or meta.get("name", "Home")
            home["tri"] = home["tri"] or meta.get("tri", "")
        # away team lookups
        if away_id and (not away["name"] or not away["tri"]):
            meta = TEAMS_BY_ID.get(away_id, {})
            away["name"] = away["name"] or meta.get("name", "Away")
            away["tri"] = away["tri"] or meta.get("tri", "")
        # compute tipoff times
        # if its scheduled game (status_id == 1), use status text as tipoff time
        tipoff = status_txt if status_id == 1 else ""
        # otherwise, parse the estimated tipoff time from UTC and convert to EST
        if not tipoff:
            try:
                gdt = pd.to_datetime(pick(r, "GAME_DATE_EST"))
                if not pd.isna(gdt):
                    tipoff = (
                        gdt.tz_localize("UTC")
                        .tz_convert("America/New_York")
                        .strftime("%-I:%M %p ET")
                    )
            except Exception:
                tipoff = ""
        # get the arena name
        arena = pick(r, "ARENA_NAME", default="") or ""
        # build and store result for the game
        # construct the full game dictionary with all fields and append it to the list
        games.append(
            {
                "gameId": gid,
                "gameStatusId": status_id,
                "status": status_txt,
                "tipoffET": tipoff,
                "livePeriod": None,
                "liveClock": None,
                "home": home,
                "away": away,
                "homeScore": home_score,
                "awayScore": away_score,
                "arena": arena,
            }
        )
    # sort and return
    # sort games by tipoff time (or home name/tri if missing).
    games.sort(
        key=lambda g: (
            g.get("tipoffET") or "",
            g["home"].get("tri") or g["home"].get("name") or "",
        )
    )
    # return the final list of game dictionaries
    return games

# function updates list of NBA games with live information (score, game clock, period, status, team IDs) for today's games
# pulls fast, real-time data from the NBA CDN endpoint todaysScoreboard_00.json
# accepts a list of dictionaries for today's games and returns the updated list of game dictionaries


def _patch_from_cdn_today(games: List[Dict[str, Any]], debug: bool = False) -> List[Dict[str, Any]]:
    # try catch attempts to fetch real-time game data from the NBA CDN’s public JSON endpoint
    try:
        js = _cdn_get_json(
            "https://cdn.nba.com/static/json/liveData/scoreboard/"
            "todaysScoreboard_00.json"
        )
        # retrieves the list of games from the JSON under scoreboard.games
        arr = js.get("scoreboard", {}).get("games", [])
        # creates a lookup dictionary (by_id) mapping each game’s ID to its data for fast access
        by_id = {str(g.get("gameId")): g for g in arr}
    # if the fetch fails return original unmodified lsit
    except Exception:
        return games
    # loops through each game in the input list
    # if real-time data (j) for this game isn’t found, skip it
    for g in games:
        j = by_id.get(g["gameId"])
        if not j:
            continue
        # update game status and live clock
        # overwrite the game's status ID and text if available from real-time data
        g["gameStatusId"] = j.get("gameStatus") or g.get("gameStatusId")
        g["status"] = j.get("gameStatusText") or g.get("status")
        # update the current quarter/period
        if j.get("period") is not None:
            g["livePeriod"] = j.get("period")
        # update live game clock
        g["liveClock"] = j.get("gameClock") or g.get("liveClock")
        # update team identity fields
        # get nested dictionaries for home and away
        ht = j.get("homeTeam", {}) or {}
        at = j.get("awayTeam", {}) or {}
        # if the home or away team ID is missing, update it from the CDN
        if g["home"].get("id") is None and ht.get("teamId"):
            g["home"]["id"] = int(ht["teamId"])
        if g["away"].get("id") is None and at.get("teamId"):
            g["away"]["id"] = int(at["teamId"])
        # same check for team abbr
        if not g["home"].get("tri") and ht.get("teamTricode"):
            g["home"]["tri"] = str(ht["teamTricode"]).upper()
        if not g["away"].get("tri") and at.get("teamTricode"):
            g["away"]["tri"] = str(at["teamTricode"]).upper()

        # try except block that checks if the scores exist and are valid
        # update them (note: the score might be "-", empty, or null before the game starts)
        try:
            if ht.get("score") not in (None, "", "-"):
                g["homeScore"] = int(ht["score"])
            if at.get("score") not in (None, "", "-"):
                g["awayScore"] = int(at["score"])
        except Exception:
            # if error happens just skip
            pass
    # returns the full list of updated games with live data injected where available
    return games

# patch scores for active or missing games using cdn
# this way we have updated live scores and any other missing scores
# we take in a list of game dictionaries and find the missing scores


def _patch_from_cdn_boxscores(
    games: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:

    if not games:
        return games

    # go through each game
    for g in games:
        need = (
            # if there is no home score
            g.get("homeScore") is None
            # if there is no away score
            or g.get("awayScore") is None
            # if the game is active or finished (get the right score)
            or g.get("gameStatusId") in (2, 3)
        )
        if not need:
            continue

        # get game id
        gid = g.get("gameId")
        if not gid:
            continue

        # get box score for given game based on game id
        url = (
            "https://cdn.nba.com/static"
            "/json/liveData/boxscore/"
            f"boxscore_{gid}.json"
        )
        # get the json from nba cdn
        try:
            js = _cdn_get_json(url, timeout=8)
            game = js.get("game", {}) or {}

            # patch status and clock, quarter, clock, final/live
            if "gameStatus" in game:
                g["gameStatusId"] = game.get("gameStatus") or g.get(
                    "gameStatusId"
                )
            if "gameStatusText" in game:
                g["status"] = game.get("gameStatusText") or g.get("status")
            if "period" in game and game.get("period") is not None:
                g["livePeriod"] = game.get("period")
            if "gameClock" in game and game.get("gameClock"):
                g["liveClock"] = game.get("gameClock")

            # get home and away team info
            ht = game.get("homeTeam", {}) or {}
            at = game.get("awayTeam", {}) or {}

            # fill in any missing values
            if not g["home"].get("tri") and ht.get("teamTricode"):
                g["home"]["tri"] = str(ht["teamTricode"]).upper()
            if not g["away"].get("tri") and at.get("teamTricode"):
                g["away"]["tri"] = str(at["teamTricode"]).upper()
            if not g["home"].get("name") and ht.get("name"):
                g["home"]["name"] = ht["name"]
            if not g["away"].get("name") and at.get("name"):
                g["away"]["name"] = at["name"]
            if not g["home"].get("id") and ht.get("teamId"):
                g["home"]["id"] = int(ht["teamId"])
            if not g["away"].get("id") and at.get("teamId"):
                g["away"]["id"] = int(at["teamId"])

            # fill in the scores
            if ht.get("score") not in (None, "", "-"):
                g["homeScore"] = int(ht["score"])
            if at.get("score") not in (None, "", "-"):
                g["awayScore"] = int(at["score"])

        except Exception:
            # ignore single game failures for now
            continue

    return games

_prediction_cache = {}

def get_predictions_cached_for_date(date: str):
    if date not in _prediction_cache:
        df, X = get_predictions_for_date(date, return_features=True)
        _prediction_cache[date] = (df, X)
    return _prediction_cache[date]

# Standings

def _normalize_standings(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Normalize LeagueStandingsV3 DataFrame to a simple JSON list."""
    # Build a mapping of lowercase column names to actual column names
    # This lets us handle slightly different column naming across API versions
    cols = {c.lower(): c for c in df.columns}

    # Helper to pick the first available, non-null column value from a set of candidates
    def pick(row, *cands, default=None):
        for cand in cands:
            # Look up the real column name ignoring case
            col = cols.get(str(cand).lower())

            # If the column exists in the row and is not NaN, return its value
            if col and col in row and pd.notna(row[col]):
                return row[col]
        # If nothing matched, fall back to the default value
        return default

    # Helper to combine two numbers into a "wins-losses" string like "25-16"
    def combo(a, b, sep: str = "-"):
        if a is None or b is None:
            return None
        try:
            # Convert both to int and join with the given separator
            return f"{int(a)}{sep}{int(b)}"
        except Exception:
            # If conversion fails, return None
            return None

    # This will hold our normalized standings entries as a list of dicts
    out: List[Dict[str, Any]] = []

    # Iterate over each team row in the standings DataFrame
    for _, r in df.iterrows():
        # Tri-code (like BOS) trying several possible column names
        tri = str(
            pick(
                r,
                "TeamAbbreviation",
                "TeamTricode",
                "Tricode",
                "TEAM_ABBREVIATION",
                default="",
            ) or ""
        ).upper()

        team_id = pick(r, "TEAM_ID")  # Team ID used by the NBA
        team = f"{pick(r, 'TeamCity', '')} {pick(r, 'TeamName', '')}".strip()  # Full team name as "City Nickname" (e.g., "Boston Celtics")
        team_short = pick(r, "TeamName", default="")  # Short name / nickname (e.g., "Celtics")

        # Wins and losses, with different possible column names
        wins = pick(r, "WINS", "Wins", default=0)
        losses = pick(r, "LOSSES", "Losses", default=0)
        # Win percentage column, which may be called WinPCT or PCT
        winpct = pick(r, "WinPCT", "PCT", default=0.0)

        # Conference ("East" or "West")
        conference = str(pick(r, "Conference", default=""))
        # Division name (e.g., "Atlantic")
        division = pick(r, "Division", default="")
        # Conference rank (1 = top seed)
        conf_rank = pick(r, "ConferenceRank", "ConfRank", default=0)

        # Games back (how far behind the first place team)
        gb = pick(r, "GamesBack", "GB", default="0.0")

        # Current streak, which can be encoded in various ways
        streak = pick(r, "Streak", "CurrentStreak",
                      "CurrentStreakText", default="")
        if isinstance(streak, (int, float)) and not pd.isna(streak):
            # Numeric streak positive = winning streak, negative = losing streak
            streak = f"{'W' if streak >= 0 else 'L'}{abs(int(streak))}"
        elif not streak:
            # Some versions split streak into wins/losses across separate columns
            sw, sl = pick(r, "StreakWins"), pick(r, "StreakLosses")
            streak = (
                f"W{int(sw)}"
                if (sw and sw > 0)
                else (f"L{int(sl)}" if (sl and sl > 0) else "")
            )

        # Last-10-games record, either directly or built from separate columns
        l10 = pick(r, "L10", "Last10")
        if not l10:
            l10 = combo(pick(r, "L10Wins",
                             "L10W"), pick(r, "L10Losses", "L10L"))
        if not l10:
            # If everything fails, show an em dash
            l10 = "—"

        # Home record string, or build it from wins/losses if missing
        home = pick(r, "HomeRecord", "HOME_RECORD")
        if not home:
            home = combo(pick(r, "HOME_WINS", "HomeWins"),
                         pick(r, "HOME_LOSSES", "HomeLosses")) or "—"

        # Road (away) record string, or build it if missing
        road = pick(r, "RoadRecord", "AWAY_RECORD", "Road")
        if not road:
            road = combo(
                pick(r, "ROAD_WINS", "AwayWins", "RoadWins"),
                pick(r, "ROAD_LOSSES", "AwayLosses", "RoadLosses"),
            ) or "—"

        # Construct the normalized standings entry for this team
        item = {
            "teamId": int(team_id) if pd.notna(team_id) else None,  # Numeric team ID (or None if not available)
            "tri": tri,  # Tri-code like "BOS"
            "team": team,  # Full team display name
            "teamShort": team_short,  # Team nickname (short label)
            "conference": conference,  # Conference name ("East" / "West")
            "division": division,  # Division name
            "wins": int(wins or 0),  # Total wins
            "losses": int(losses or 0),  # Total losses
            "pct": float(winpct or 0.0),  # Win percentage as float
            "gb": str(gb) if gb is not None else "0.0",  # Games back as string to avoid rounding issues
            "streak": streak or "—",  # Current streak ("W4", "L2", or "—")
            "l10": l10,  # Last 10 games record ("7-3")
            "home": home,  # Home record string ("15-3")
            "road": road,  # Road record string ("8-7")
            "confRank": int(conf_rank or 0),  # Conference rank as integer (used for ordering)
        }
        # Add this team's normalized data to the output list
        out.append(item)

    # Sort the list by conference, then by conference rank, then by win percentage
    out.sort(
        key=lambda x: (
            x["conference"],
            x["confRank"] if x["confRank"] else 999,
            -x["pct"],
        )
    )
    # Return the final normalized standings list
    return out

# Routes

# This route fetches NBA standings for a requested season, normalizes the data,
# splits it into Eastern and Western conferences, and returns it as clean JSON for your frontend.


@app.get("/api/standings")
def standings():
    season = request.args.get("season") or _current_season_label()
    season_type = request.args.get("seasonType") or "Regular Season"

    season_safe = season.replace("-", "_")
    type_safe = season_type.replace(" ", "_")
    key = f"standings/{season_safe}_{type_safe}.csv"

    df = _read_csv_from_s3(key)
    if df.empty:
        return jsonify({"season": season, "seasonType": season_type, "east": [], "west": []})

    df = df.fillna("")
    rows = df.to_dict(orient="records")
    east = [r for r in rows if str(r.get("conference", "")).lower().startswith("east")]
    west = [r for r in rows if str(r.get("conference", "")).lower().startswith("west")]

    return jsonify({"season": season, "seasonType": season_type, "east": east, "west": west})


# This route takes a season label (or auto-detects the current season),
# calculates when that NBA season starts and ends, and returns those dates in ISO format.
@app.get("/api/season/window")
def season_window():
    """Return season boundaries as ISO dates."""
    season = request.args.get("season") or _current_season_label()  # Season param or detect automatically
    start, end = _season_window_for_label(season)  # Convert label into date range
    return jsonify({
        "season": season,
        "start": start.isoformat(),
        "end": end.isoformat()
    })


# Define GET /api/schedule/day route
@app.get("/api/schedule/day")
def schedule_day():
    d = request.args.get("date")
    if not d:
        return jsonify({"error": "missing 'date' (YYYY-MM-DD)"}), 400

    try:
        day = datetime.strptime(d, "%Y-%m-%d").date()
    except Exception:
        return jsonify({"error": "invalid 'date' format, expected YYYY-MM-DD"}), 400

    is_today = d == datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")

    try:
        if is_today:
            js = _cdn_get_json(
                "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"
            )
            cdn_games = js.get("scoreboard", {}).get("games", [])

            # Get arena names from season schedule
            try:
                schedule_js = _cdn_get_json(
                    "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2_1.json"
                )
                arena_by_game = {}
                for gd in schedule_js.get("leagueSchedule", {}).get("gameDates", []):
                    for g in gd.get("games", []):
                        arena_by_game[g.get("gameId", "")] = g.get("arenaName", "")
                for g in cdn_games:
                    gid = g.get("gameId", "")
                    if gid in arena_by_game:
                        g["arenaName"] = arena_by_game[gid]
            except Exception:
                pass
        else:
            schedule_js = _cdn_get_json(
                "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2_1.json"
            )
            cdn_games = []
            for gd in schedule_js.get("leagueSchedule", {}).get("gameDates", []):
                raw_date = gd.get("gameDate", "")
                try:
                    gd_str = datetime.strptime(raw_date, "%m/%d/%Y %H:%M:%S").strftime("%Y-%m-%d")
                except Exception:
                    continue
                if gd_str == d:
                    cdn_games = gd.get("games", [])
                    break

        _ensure_team_lookup()
        games = []
        for g in cdn_games:
            ht = g.get("homeTeam", {})
            at = g.get("awayTeam", {})
            home_id = ht.get("teamId")
            away_id = at.get("teamId")
            home_meta = TEAMS_BY_ID.get(home_id, {})
            away_meta = TEAMS_BY_ID.get(away_id, {})

            status_id = g.get("gameStatus") or g.get("gameStatusId") or 1
            home_score = ht.get("score") if status_id > 1 else None
            away_score = at.get("score") if status_id > 1 else None

            # tipoff time
            tipoff = ""
            try:
                utc_str = g.get("gameDateTimeUTC", "")
                if utc_str:
                    tipoff = (
                        datetime.fromisoformat(utc_str.replace("Z", "+00:00"))
                        .astimezone(ZoneInfo("America/New_York"))
                        .strftime("%-I:%M %p ET")
                    )
            except Exception:
                pass

            games.append({
                "gameId": g.get("gameId", ""),
                "gameStatusId": status_id,
                "status": g.get("gameStatusText", ""),
                "tipoffET": tipoff,
                "livePeriod": g.get("period") if status_id == 2 else None,
                "liveClock": g.get("gameClock", "") if status_id == 2 else None,
                "home": {
                    "id": home_id,
                    "name": ht.get("teamName", home_meta.get("name", "")),
                    "tri": ht.get("teamTricode", home_meta.get("tri", "")),
                },
                "away": {
                    "id": away_id,
                    "name": at.get("teamName", away_meta.get("name", "")),
                    "tri": at.get("teamTricode", away_meta.get("tri", "")),
                },
                "homeScore": int(home_score) if home_score is not None else None,
                "awayScore": int(away_score) if away_score is not None else None,
                "arena": g.get("arenaName", ""),
            })

    except Exception as e:
        return jsonify({"error": str(e), "dateET": d, "games": []}), 200

    return jsonify({"dateET": d, "games": games})


# Define GET /api/debug/state
@app.get("/api/debug/state")
def debug_state():
    """
    Debug route for checking if SQLite and today's rows are accessible.
    """
    try:
        con = sqlite3.connect(DB_PATH)  # Open database
        cur = con.cursor()  # Create cursor
        cur.execute(
            "SELECT COUNT(*) FROM games_today WHERE date_et = ?",
            (today_et_iso(),),
        )
        cnt = cur.fetchone()[0]  # Retrieve count
        con.close()
        ok = True  # Database access succeeded
    except sqlite3.Error as e:
        cnt = f"error: {e}"  # Capture error
        ok = False  # Failed status

    return jsonify({
        "ok": ok,
        "dbPath": DB_PATH,
        "today": today_et_iso(),
        "rows_today": cnt
    })


# Configure cache system with SimpleCache (in-memory)
cache = Cache(app, config={
    'CACHE_TYPE': 'SimpleCache',
    'CACHE_DEFAULT_TIMEOUT': 300  # Cache entries last 5 minutes
})


# Build a mapping: TEAM_ID -> abbreviation (e.g., 1610612747 -> "LAL")
TEAM_MAP = {t['id']: t['abbreviation'] for t in teams.get_teams()}


# Define root "/" route
@app.route('/')
def home():
    """Root endpoint: returns available API routes."""
    return jsonify({
        "message": "NBA API Backend is running",
        "routes": [
            "/games/date/<YYYY-MM-DD>",
            "/games/boxscore/<game_id>",
            "/games/season/<SEASON>"
        ]
    })


# Route for SwishAI, makes sure our open AI api key is safely set up and working
@app.route("/api/swish/chat", methods=["POST", "OPTIONS"])
def swish_chat():
    if request.method == "OPTIONS":
        return jsonify({"ok": True}), 200

    data = request.get_json(silent=True) or {}
    msgs = data.get("messages") or []
    intent = (data.get("intent") or "").strip()

    if intent == "explain_slate":
        date_str = (data.get("date") or "").strip()
        if not date_str:
            now_et = datetime.now(ZoneInfo("America/New_York"))
            date_str = now_et.strftime("%Y-%m-%d")

        preds_df, X_date = get_predictions_with_features_for_date(date_str)
        if preds_df is None or preds_df.empty or X_date is None or X_date.empty:
            return jsonify({"reply": f"No games found (or insufficient data) for {date_str}."}), 200

        preds_df = preds_df.reset_index(drop=True)
        X_date = X_date.reset_index(drop=True)

        # Try to get explainer-picked top features (fallback)
        try:
            explained = explain_predictions(date_str)  # list of per-game objects
        except Exception:
            explained = []

        top_feats_by_idx = [None] * len(preds_df)
        if explained and len(explained) == len(preds_df):
            for i, ex in enumerate(explained):
                feats = []
                for r in ex.get("top_reasons_home", []):
                    if r.get("feature"):
                        feats.append(r["feature"])
                for r in ex.get("top_reasons_away", []):
                    if r.get("feature"):
                        feats.append(r["feature"])
                top_feats_by_idx[i] = feats

        games_payload = []
        for i in range(len(preds_df)):
            pr = preds_df.iloc[i].to_dict()
            fr = {k: float(v) for k, v in X_date.iloc[i].to_dict().items()}

            shap_row = None

            games_payload.append({
                "home": f"{pr.get('home_name')} ({pr.get('home_tri','')})".strip(),
                "away": f"{pr.get('away_name')} ({pr.get('away_tri','')})".strip(),
                "p_home_win": float(pr["p_home_win"]),
                "key_diffs": _pick_top_by_shap(fr, shap_row, k=5),
                "definition": "All diffs are (home - away). Positive means home edge; negative means away edge.",
            })

        system_prompt = (
            "You are Swish, a basketball assistant. You must explain ONLY using the provided numbers. "
            "Do NOT invent injuries, roster changes, rest, betting lines, or stats not present. "
            "Use beginner-friendly language. Be number-centric. "
            "Output Markdown (not JSON). For each game produce:\n"
            "- A bold headline\n"
            "- 2–3 sentence summary that includes p_home_win as a percentage\n"
            "- Exactly 3 bullet reasons. Each bullet MUST include the diff_fmt exactly as given.\n"
            "- 1 caveat sentence\n"
            "Ignore the 'shap' field if present; it is not user-facing.\n"
        )

        user_payload = {
            "date": date_str,
            "model_context": (
                "Model: SVM (RBF) trained on the current season using game-level (home-away) feature diffs built from "
                "blended last-season averages (15%) + current momentum (85%, rolling 10 games with early-season shrinkage), plus Elo."
            ),
            "games": games_payload,
        }

        try:
            client = get_openai_client()
            resp = client.responses.create(
                model=SWISH_MODEL,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(user_payload)},
                ],
                temperature=0.2,
            )
            return jsonify({"reply": resp.output_text}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # ---- normal chat fallback ----
    if not isinstance(msgs, list) or len(msgs) == 0:
        return jsonify({"error": "Missing messages"}), 400

    chat_input = [{
        "role": "system",
        "content": (
            "You are Swish, a basketball-only assistant. "
            "Answer questions strictly related to basketball (NBA, teams, players, games, stats, history). "
            "If a question is not basketball-related, politely refuse with a brief message saying "
            "'I can only answer basketball-related questions.' "
            "Use Markdown formatting when appropriate (bold, lists, paragraphs). "
            "Do not mention Markdown explicitly."
        )
    }]

    for m in msgs:
        role = (m.get("role") or "").strip()
        text = (m.get("text") or "").strip()
        if not text:
            continue
        if role == "user":
            chat_input.append({"role": "user", "content": text})
        elif role == "bot":
            chat_input.append({"role": "assistant", "content": text})

    try:
        client = get_openai_client()
        response = client.responses.create(model=SWISH_MODEL, input=chat_input)
        return jsonify({"reply": response.output_text}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Cached endpoint for games by date
@cache.cached(timeout=600)
@app.route('/games/date/<date>')
def games_by_date(date):
    """Fetch NBA games for a date using ScoreboardV2 or GameFinder fallback."""
    try:
        board = scoreboardv2.ScoreboardV2(game_date=date)  # Call NBA API
        header = board.get_normalized_dict().get('GameHeader', [])  # Basic info rows
        line_scores = board.get_normalized_dict().get('LineScore', [])  # Score rows

        if header:  # Use ScoreboardV2 if it returned data
            team_scores = {
                ls['TEAM_ID']: ls['PTS']
                for ls in line_scores if ls.get('PTS') is not None
            }

            results = []

            for g in header:
                # Extract home/away IDs
                home_id = g.get('HOME_TEAM_ID')
                away_id = g.get('VISITOR_TEAM_ID')

                raw_time = g.get('GAME_DATE_EST', '')
                formatted_time = "N/A"

                # Convert timestamp to human-readable ET time
                if raw_time:
                    try:
                        dt = datetime.fromisoformat(raw_time.replace('Z', ''))
                        formatted_time = dt.strftime("%I:%M %p ET")
                    except Exception:
                        formatted_time = raw_time

                # Build result card
                results.append({
                    'gameId': g.get('GAME_ID', 'N/A'),
                    'homeTeam': TEAM_MAP.get(home_id, home_id),
                    'awayTeam': TEAM_MAP.get(away_id, away_id),
                    'homeScore': team_scores.get(home_id, 0),
                    'awayScore': team_scores.get(away_id, 0),
                    'gameStatus': g.get('GAME_STATUS_TEXT', 'Final'),
                    'gameDate': g.get('GAME_DATE_EST', date),
                    'gameTime': formatted_time,
                    'arena': g.get('ARENA_NAME', 'N/A')
                })

            return jsonify(results)

        # If ScoreboardV2 returned nothing, fall back to LeagueGameFinder
        parsed = datetime.strptime(date, "%Y-%m-%d")
        season_year = parsed.year if parsed.month >= 10 else parsed.year - 1
        season_str = f"{season_year}-{str(season_year + 1)[-2:]}"  # Convert integer to season label

        # Query full-season results
        games = leaguegamefinder.LeagueGameFinder(
            season_nullable=season_str
        ).get_data_frames()[0]

        # Filter only games matching exact date
        games = games[
            games['GAME_DATE'] == parsed.strftime("%b %d, %Y").upper()
        ]

        if games.empty:
            return jsonify({'error': f'No games found for {date}'}), 404

        results = []
        for _, row in games.iterrows():
            # Build output rows
            results.append({
                'gameId': row['GAME_ID'],
                'matchup': row['MATCHUP'],
                'team': row['TEAM_NAME'],
                'winLoss': row['WL'],
                'pts': int(row['PTS']),
                'plusMinus': float(row.get('PLUS_MINUS', 0)),
                'gameDate': row['GAME_DATE']
            })

        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Route for all games from one NBA season
@app.route('/games/season/<season>')
def games_by_season(season):
    """Return all NBA games for a given season."""
    try:
        # Query NBA API for the given season
        games = leaguegamefinder.LeagueGameFinder(
            season_nullable=season,
            league_id_nullable='00'
        ).get_data_frames()[0]

        # Select key columns and convert to dict
        result = games[[
            'GAME_ID', 'GAME_DATE', 'TEAM_NAME', 'MATCHUP', 'WL', 'PTS'
        ]].to_dict(orient='records')

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Cache boxscore endpoint for performance
@cache.cached(timeout=600)
@app.route('/games/boxscore/<game_id>')
def boxscore(game_id):
    """Return full boxscore + top performers for an NBA game."""
    try:
        # Fetch full boxscore via NBA API
        box = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
        players = box.get_data_frames()[0]  # Player-level stats
        team_stats = box.get_data_frames()[1]  # Team-level stats

        teams_data = []
        for _, t in team_stats.iterrows():
            # Build team stat card
            teams_data.append({
                'team': TEAM_MAP.get(t['TEAM_ID'], t['TEAM_ID']),
                'fg_pct': float(t['FG_PCT']),
                'rebounds': int(t['REB']),
                'assists': int(t['AST']),
                'points': int(t['PTS']),
                'turnovers': int(t['TO'])
            })

        # Identify top statistical performers
        top_points = players.loc[players['PTS'].idxmax()]
        top_assists = players.loc[players['AST'].idxmax()]
        top_rebounds = players.loc[players['REB'].idxmax()]

        # Construct final response
        result = {
            'gameId': game_id,
            'teams': teams_data,
            'topPerformers': {
                'topScorer': {
                    'name': top_points['PLAYER_NAME'],
                    'team': TEAM_MAP.get(top_points['TEAM_ID'], top_points['TEAM_ID']),
                    'points': int(top_points['PTS'])
                },
                'topAssists': {
                    'name': top_assists['PLAYER_NAME'],
                    'team': TEAM_MAP.get(top_assists['TEAM_ID'], top_assists['TEAM_ID']),
                    'assists': int(top_assists['AST'])
                },
                'topRebounds': {
                    'name': top_rebounds['PLAYER_NAME'],
                    'team': TEAM_MAP.get(top_rebounds['TEAM_ID'], top_rebounds['TEAM_ID']),
                    'rebounds': int(top_rebounds['REB'])
                }
            }
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': f"Could not fetch boxscore: {e}"}), 500


# TEAMS ROUTES
@app.get('/api/teams')
def get_all_teams():

    _ensure_team_lookup()  # Make sure TEAMS_BY_ID is loaded/ready to use

    # This will hold the final list of team objects we return
    teams_list = []
    # Loop through every team in our lookup dict
    for team_id, info in TEAMS_BY_ID.items():
        teams_list.append({
            'teamId': team_id,
            'abbreviation': info['tri'],
            'fullName': info['name'],
            'city': info['name'].rsplit(' ', 1)[0] if ' ' in info['name'] else '',
            'nickname': info['name'].rsplit(' ', 1)[1] if ' ' in info['name'] else info['name']
        })

    # Sort teams A->Z by full name
    teams_list.sort(key=lambda x: x['fullName'])
    return jsonify(teams_list)


@app.get('/api/teams/<team_abbr>/stats')
def team_season_stats(team_abbr):
    season_year = request.args.get('season', type=int)
    if not season_year:
        now = datetime.now(ZoneInfo("America/New_York"))
        season_year = now.year if now.month >= 10 else now.year - 1

    season_str = f"{season_year}-{str(season_year + 1)[-2:]}"
    team_abbr_upper = team_abbr.upper()
    season_safe = season_str.replace("-", "_")
    key = f"team_stats/{team_abbr_upper}_{season_safe}.csv"

    df = _read_csv_from_s3(key)
    if df.empty:
        return jsonify({"error": f"No stats found for {team_abbr_upper} {season_str}"}), 404

    row = df.iloc[0]
    return jsonify({
        "team": team_abbr_upper,
        "season": season_str,
        "gamesPlayed": int(row.get("games_played", 0)),
        "record": {
            "wins": int(row.get("wins", 0)),
            "losses": int(row.get("losses", 0)),
            "winPct": float(row.get("win_pct", 0))
        },
        "offensive": {
            "ppg": float(row.get("ppg", 0)),
            "fgPct": float(row.get("fg_pct", 0)),
            "fg3Pct": float(row.get("fg3_pct", 0)),
            "ftPct": float(row.get("ft_pct", 0)),
            "apg": float(row.get("apg", 0)),
            "topg": float(row.get("topg", 0))
        },
        "defensive": {
            "rpg": float(row.get("rpg", 0)),
            "orpg": float(row.get("orpg", 0)),
            "drpg": float(row.get("drpg", 0)),
            "spg": float(row.get("spg", 0)),
            "bpg": float(row.get("bpg", 0)),
            "fpg": float(row.get("fpg", 0))
        }
    })


@app.get('/api/teams/<team_abbr>/schedule')
def team_schedule(team_abbr):
    season_year = request.args.get('season', type=int)
    if not season_year:
        now = datetime.now(ZoneInfo("America/New_York"))
        season_year = now.year if now.month >= 10 else now.year - 1

    season_str = f"{season_year}-{str(season_year + 1)[-2:]}"
    team_abbr_upper = team_abbr.upper()
    season_safe = season_str.replace("-", "_")
    key = f"team_schedule/{team_abbr_upper}_{season_safe}.csv"

    df = _read_csv_from_s3(key)
    if df.empty:
        return jsonify({"team": team_abbr_upper, "season": season_str, "games": []})

    games = []
    for _, row in df.iterrows():
        games.append({
            "date": row.get("game_date", ""),
            "opponent": row.get("opponent", ""),
            "location": row.get("location", ""),
            "result": row.get("result", ""),
            "points": int(row.get("points", 0)),
            "opponentPoints": int(row.get("opponent_points", 0))
        })

    return jsonify({"team": team_abbr_upper, "season": season_str, "games": games})


@app.get('/api/teams/<team_abbr>/roster')
def team_roster(team_abbr):
    season_year = request.args.get('season', type=int)
    if not season_year:
        now = datetime.now(ZoneInfo("America/New_York"))
        season_year = now.year if now.month >= 10 else now.year - 1

    season_str = f"{season_year}-{str(season_year + 1)[-2:]}"
    team_abbr_upper = team_abbr.upper()
    season_safe = season_str.replace("-", "_")
    key = f"team_roster/{team_abbr_upper}_{season_safe}.csv"

    df = _read_csv_from_s3(key)
    if df.empty:
        return jsonify({"team": team_abbr_upper, "season": season_str, "roster": []})

    roster = []
    for _, row in df.iterrows():
        roster.append({
            "name": row.get("player_name", "Unknown"),
            "position": row.get("position", "N/A"),
            "gamesPlayed": int(row.get("games_played", 0)),
            "stats": {
                "ppg": float(row.get("ppg", 0)),
                "rpg": float(row.get("rpg", 0)),
                "apg": float(row.get("apg", 0)),
                "fgPct": float(row.get("fg_pct", 0)),
                "fg3Pct": float(row.get("fg3_pct", 0)),
                "ftPct": float(row.get("ft_pct", 0))
            }
        })

    return jsonify({"team": team_abbr_upper, "season": season_str, "roster": roster})


# NEW ROUTES FOR THE PREDICTIONS
@app.get("/api/predictions/day")
def api_predictions_day():
    date_str = request.args.get("date")
    if not date_str:
        return jsonify({"error": "missing 'date' (YYYY-MM-DD)"}), 400

    try:
        df = get_predictions_for_date(date_str, model_path="models/svm_momentum_svm.pkl")
        return jsonify({"date": date_str, "predictions": df.to_dict(orient="records")})
    except Exception as e:
        return jsonify({"date": date_str, "predictions": [], "error": str(e)}), 200


@app.get("/api/predictions/game")
def api_prediction_game():
    date_str = request.args.get("date")
    game_id = request.args.get("gameId")

    if not date_str or not game_id:
        return jsonify({"error": "missing 'date' or 'gameId'"}), 400

    try:
        pred = get_prediction_for_game(date_str, game_id, model_path="models/svm_momentum_svm.pkl")
        return jsonify({"date": date_str, "gameId": game_id, "prediction": pred})
    except Exception as e:
        return jsonify({"date": date_str, "gameId": game_id, "prediction": None, "error": str(e)}), 200
    
@app.route("/api/explain/<date>")
def explain(date):
    """
    GET /api/explain/2025-02-24
    Optional ?gameId=00XXXXXXX  -> returns only that game's explanation.
    Always returns JSON, never an HTML error page.
    """
    try:
        game_id = request.args.get("gameId")
        games   = explain_predictions(date)

        if game_id:
            match = next(
                (g for g in games if g.get("numbers", {}).get("game_id") == game_id),
                None,
            )
            if match is None:
                return jsonify({"error": f"No explanation found for gameId={game_id}"}), 404
            return jsonify({"date": date, "gameId": game_id, "explanation": match})

        return jsonify({"date": date, "games": games})

    except Exception as exc:
        logging.exception("Error in /api/explain/%s", date)
        return jsonify({"error": str(exc)}), 500

# CDN BOXSCORE ROUTE (works for live + finished games)
@app.get("/api/boxscore/<game_id>")
def api_boxscore_cdn(game_id):
    """
    GET /api/boxscore/<game_id>
    Fetches player box score from NBA CDN (works for live and completed games).
    Returns both teams' player stats + team totals.
    """
    url = (
        "https://cdn.nba.com/static/json/liveData/boxscore/"
        f"boxscore_{game_id}.json"
    )
    try:
        js = _cdn_get_json(url, timeout=10)
        game = js.get("game", {}) or {}

        status = game.get("gameStatus", 1)
        status_text = game.get("gameStatusText", "")

        def parse_team(team_obj):
            players_raw = team_obj.get("players", [])
            players = []
            for p in players_raw:
                stats = p.get("statistics", {}) or {}
                # skip players with no minutes
                mins = stats.get("minutesCalculated", "") or stats.get("minutes", "")
                if not mins or mins in ("PT00M00.00S", "PT0M", ""):
                    played = False
                else:
                    played = True

                players.append({
                    "name": (p.get("firstName", "") or "") + " " + (p.get("familyName", "") or ""),
                    "nameI": p.get("nameI", ""),
                    "jerseyNum": p.get("jerseyNum", ""),
                    "position": p.get("position", "") or "",
                    "starter": bool(p.get("starter", "") == "1"),
                    "played": played,
                    "minutes": mins,
                    "points": int(stats.get("points", 0)),
                    "rebounds": int(stats.get("reboundsTotal", 0)),
                    "assists": int(stats.get("assists", 0)),
                    "steals": int(stats.get("steals", 0)),
                    "blocks": int(stats.get("blocks", 0)),
                    "turnovers": int(stats.get("turnovers", 0)),
                    "fgm": int(stats.get("fieldGoalsMade", 0)),
                    "fga": int(stats.get("fieldGoalsAttempted", 0)),
                    "tpm": int(stats.get("threePointersMade", 0)),
                    "tpa": int(stats.get("threePointersAttempted", 0)),
                    "ftm": int(stats.get("freeThrowsMade", 0)),
                    "fta": int(stats.get("freeThrowsAttempted", 0)),
                    "plusMinus": float(stats.get("plusMinusPoints", 0)),
                    "fouls": int(stats.get("foulsPersonal", 0)),
                })

            # Team totals from statistics key
            ts = team_obj.get("statistics", {}) or {}
            totals = {
                "points": int(ts.get("points", 0)),
                "rebounds": int(ts.get("reboundsTotal", 0)),
                "assists": int(ts.get("assists", 0)),
                "steals": int(ts.get("steals", 0)),
                "blocks": int(ts.get("blocks", 0)),
                "turnovers": int(ts.get("turnovers", 0)),
                "fgm": int(ts.get("fieldGoalsMade", 0)),
                "fga": int(ts.get("fieldGoalsAttempted", 0)),
                "tpm": int(ts.get("threePointersMade", 0)),
                "tpa": int(ts.get("threePointersAttempted", 0)),
                "ftm": int(ts.get("freeThrowsMade", 0)),
                "fta": int(ts.get("freeThrowsAttempted", 0)),
            }

            return {
                "teamId": int(team_obj.get("teamId", 0)),
                "tri": str(team_obj.get("teamTricode", "")).upper(),
                "name": team_obj.get("teamName", ""),
                "city": team_obj.get("teamCity", ""),
                "score": int(team_obj.get("score", 0)),
                "players": players,
                "totals": totals,
            }

        ht = game.get("homeTeam", {}) or {}
        at = game.get("awayTeam", {}) or {}

        return jsonify({
            "ok": True,
            "gameId": game_id,
            "gameStatus": status,
            "gameStatusText": status_text,
            "home": parse_team(ht),
            "away": parse_team(at),
        }), 200

    except Exception as e:
        return jsonify({"ok": False, "gameId": game_id, "error": str(e)}), 200
        


@app.get("/api/predictions/live")
def api_prediction_live():
    """
    GET /api/predictions/live?gameId=...
    Returns live win probabilities based on NBA CDN play-by-play.
    """
    game_id = request.args.get("gameId")
    if not game_id:
        return jsonify({"error": "missing 'gameId'"}), 400

    try:
        # CDN does NOT need stats.nba.com session or team ids
        out = predict_live_from_cdn(
            game_id=game_id,
            bundle_path="models/xgb_pbp_live.pkl",
        )
        return jsonify(out), 200
    except Exception as e:
        return jsonify({"gameId": game_id, "ok": False, "error": str(e)}), 200

@app.post("/api/predictions/live/train")
def api_train_live_pbp():
    """
    POST /api/predictions/live/train
    Trains the live PBP XGB model using NBA CDN play-by-play (2024-25 + 2025-26).
    Body (optional):
      {
        "maxGames": 200,
        "wLast": 0.15,
        "wCur": 0.85
      }
    """
    body = request.get_json(silent=True) or {}

    max_games = body.get("maxGames")
    w_last = float(body.get("wLast", 0.15))
    w_cur = float(body.get("wCur", 0.85))

    try:
        # Still needed because training uses LeagueGameLog from stats.nba.com
        _configure_stats_session()

        train_xgb_pbp_live(
            last_season="2024-25",
            current_season="2025-26",
            w_last=w_last,
            w_cur=w_cur,
            max_games=max_games,
            out_path="models/xgb_pbp_live.pkl",
        )
        return jsonify(
            {
                "ok": True,
                "modelPath": "models/xgb_pbp_live.pkl",
                "maxGames": max_games,
                "weights": {"last": w_last, "current": w_cur},
            }
        ), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    
@app.get("/api/predictions/live/history")
def api_prediction_live_history():
    game_id = request.args.get("gameId")
    if not game_id:
        return jsonify({"ok": False, "error": "missing 'gameId'"}), 400

    try:
        bundle = load_cdn_bundle("models/xgb_pbp_live.pkl")
        model = bundle["model"]
        feat_cols = bundle["feature_names"]

        js = fetch_pbp_from_cdn(game_id)
        game = (js.get("game") or {})
        actions = game.get("actions") or []
        if not actions:
            return jsonify({"ok": False, "gameId": game_id, "error": "No actions found"}), 200

        df = _actions_to_scored_rows(actions)
        if df.empty:
            return jsonify({"ok": False, "gameId": game_id, "error": "No scored rows found"}), 200

        df = _add_training_features(df)
        if df.empty:
            return jsonify({"ok": False, "gameId": game_id, "error": "No valid rows after feature build"}), 200

        X = df[feat_cols].astype(float)
        probs = model.predict_proba(X)[:, 1].astype(float)

        points = []
        for i in range(len(df)):
            points.append(
                {
                    "period": int(df.loc[i, "period"]),
                    "secLeftPeriod": int(df.loc[i, "sec_left_period"]),
                    "homeScore": int(df.loc[i, "home_score"]),
                    "awayScore": int(df.loc[i, "away_score"]),
                    "homeWinProb": float(probs[i]),
                }
            )

        return jsonify({"ok": True, "gameId": game_id, "points": points}), 200

    except Exception as e:
        return jsonify({"ok": False, "gameId": game_id, "error": str(e)}), 200

    
# Entrypoint


# Run Flask app when executed directly
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)  # Start server