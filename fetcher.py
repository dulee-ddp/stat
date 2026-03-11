import argparse
import io
import logging
import os
import random
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import boto3
import pandas as pd
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from nba_api.stats.endpoints import (
    commonteamroster,
    leaguedashplayerstats,
    leaguegamefinder,
    leaguestandingsv3,
)
from nba_api.stats.library.http import NBAStatsHTTP
from nba_api.stats.static import teams as static_teams

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

S3_BUCKET = os.environ.get("S3_BUCKET", "statline-nba-data")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

TEAMS_BY_ID = {}
TEAMS_BY_TRI = {}


def _ensure_team_lookup():
    if TEAMS_BY_ID:
        return
    for t in static_teams.get_teams():
        tid = int(t["id"])
        tri = str(t.get("abbreviation", "")).upper()
        TEAMS_BY_ID[tid] = {"tri": tri, "name": t.get("full_name", "")}
        TEAMS_BY_TRI[tri] = {"id": tid, "name": t.get("full_name", "")}


def configure_nba_session(timeout: int = 20):
    s = Session()
    retry = Retry(
        total=8, read=8, connect=8, backoff_factor=0.7,
        status_forcelist=[429, 502, 503, 504],
        allowed_methods=["GET"], raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=50, pool_maxsize=50)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.nba.com/",
        "Origin": "https://www.nba.com",
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    })
    NBAStatsHTTP.timeout = timeout
    NBAStatsHTTP._session = s


def _safe_call(fn, *args, max_retries=5, base_pause=0.9, **kwargs):
    for i in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            sleep_s = base_pause * (1.6 ** i) + random.uniform(0.05, 0.35)
            logging.warning("%s failed %s/%s: %s -> sleep %.1fs",
                            fn.__name__, i + 1, max_retries, e, sleep_s)
            time.sleep(sleep_s)
    return fn(*args, **kwargs)


def get_s3():
    return boto3.client("s3", region_name=AWS_REGION)


def upload_df_to_s3(df: pd.DataFrame, key: str):
    s3 = get_s3()
    # Replace NaN with None so JSON doesn't break
    df = df.where(pd.notnull(df), None)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=csv_buffer.getvalue(),
        ContentType="text/csv"
    )
    logging.info("Uploaded s3://%s/%s", S3_BUCKET, key)


def current_season_label() -> str:
    now = datetime.now(ZoneInfo("America/New_York"))
    year = now.year
    if now.month >= 8:
        start, end = year, (year + 1) % 100
    else:
        start, end = year - 1, year % 100
    return f"{start}-{end:02d}"


def fetch_standings(season: str = None, season_type: str = "Regular Season"):
    season = season or current_season_label()
    logging.info("Fetching standings for %s %s ...", season, season_type)
    configure_nba_session()
    _ensure_team_lookup()

    resp = _safe_call(
        leaguestandingsv3.LeagueStandingsV3,
        season=season, league_id="00", season_type=season_type
    )
    df = resp.standings.get_data_frame()
    if df is None or df.empty:
        logging.warning("No standings data returned")
        return

    cols = {c.lower(): c for c in df.columns}

    def pick(row, *cands, default=None):
        for cand in cands:
            col = cols.get(str(cand).lower())
            if col and col in row and pd.notna(row[col]):
                return row[col]
        return default

    rows = []
    for _, r in df.iterrows():
        team_id = pick(r, "TeamID", default=None)
        team_id = int(team_id) if team_id and pd.notna(team_id) else None
        tri = TEAMS_BY_ID.get(team_id, {}).get("tri", "") if team_id else ""

        team = f"{pick(r, 'TeamCity', '')} {pick(r, 'TeamName', '')}".strip()
        wins = pick(r, "WINS", "Wins", default=0)
        losses = pick(r, "LOSSES", "Losses", default=0)
        winpct = pick(r, "WinPCT", "PCT", default=0.0)
        conference = str(pick(r, "Conference", default=""))
        conf_rank = pick(r, "ConferenceRank", "ConfRank", default=0)
        gb = pick(r, "GamesBack", "GB", default="0.0")

        streak = pick(r, "Streak", "CurrentStreak", "CurrentStreakText", default="")
        if isinstance(streak, (int, float)) and not pd.isna(streak):
            streak = f"{'W' if streak >= 0 else 'L'}{abs(int(streak))}"

        l10 = pick(r, "L10", "Last10")
        if not l10:
            sw = pick(r, "L10Wins", "L10W")
            sl = pick(r, "L10Losses", "L10L")
            l10 = (f"W{int(sw)}" if (sw and sw > 0) else (f"L{int(sl)}" if (sl and sl > 0) else "—"))

        rows.append({
            "season": season,
            "season_type": season_type,
            "conference": conference,
            "team_id": team_id,
            "tri": tri,
            "team": team,
            "wins": int(wins or 0),
            "losses": int(losses or 0),
            "pct": float(winpct or 0.0),
            "gb": str(gb) if gb is not None else "0.0",
            "streak": streak or "—",
            "l10": l10 or "—",
            "conf_rank": int(conf_rank or 0),
        })

    out_df = pd.DataFrame(rows)
    season_safe = season.replace("-", "_")
    type_safe = season_type.replace(" ", "_")
    upload_df_to_s3(out_df, f"standings/{season_safe}_{type_safe}.csv")
    logging.info("Standings uploaded for %s %s (%d teams)", season, season_type, len(rows))

def fetch_season_game_logs(season_label: str):
    """Fetch all team game logs for a season and upload to S3."""
    from nba_api.stats.endpoints import leaguegamelog
    configure_nba_session()
    logging.info("Fetching game logs for season %s ...", season_label)

    lg = _safe_call(
        leaguegamelog.LeagueGameLog,
        season=season_label,
        season_type_all_star="Regular Season",
        league_id="00",
        counter=0,
        sorter="DATE",
        direction="ASC",
    )
    df = lg.get_data_frames()[0].copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["SEASON"] = season_label

    season_safe = season_label.replace("-", "_")
    upload_df_to_s3(df, f"game_logs/season_{season_safe}.csv")
    logging.info("Game logs uploaded for %s (%d rows)", season_label, len(df))

def fetch_team_stats(team_abbr: str, season_year: int = None):
    configure_nba_session()
    _ensure_team_lookup()

    if not season_year:
        now = datetime.now(ZoneInfo("America/New_York"))
        season_year = now.year if now.month >= 10 else now.year - 1

    season_str = f"{season_year}-{str(season_year + 1)[-2:]}"
    team_abbr_upper = team_abbr.upper()
    team_id = TEAMS_BY_TRI.get(team_abbr_upper, {}).get("id")

    if not team_id:
        logging.warning("Unknown team: %s", team_abbr)
        return

    logging.info("Fetching stats for %s %s ...", team_abbr_upper, season_str)

    finder = _safe_call(
        leaguegamefinder.LeagueGameFinder,
        team_id_nullable=team_id,
        season_nullable=season_str,
        season_type_nullable="Regular Season"
    )
    games_df = finder.get_data_frames()[0]

    if games_df.empty:
        logging.warning("No stats for %s %s", team_abbr_upper, season_str)
        return

    total_games = len(games_df)
    total_wins = len(games_df[games_df["WL"] == "W"])

    stats = pd.DataFrame([{
        "team_abbr": team_abbr_upper,
        "season": season_str,
        "games_played": total_games,
        "wins": total_wins,
        "losses": total_games - total_wins,
        "win_pct": round(total_wins / total_games, 3) if total_games > 0 else 0,
        "ppg": round(float(games_df["PTS"].mean()), 1),
        "fg_pct": round(float(games_df["FG_PCT"].mean() * 100), 1),
        "fg3_pct": round(float(games_df["FG3_PCT"].mean() * 100), 1),
        "ft_pct": round(float(games_df["FT_PCT"].mean() * 100), 1),
        "apg": round(float(games_df["AST"].mean()), 1),
        "topg": round(float(games_df["TOV"].mean()), 1),
        "rpg": round(float(games_df["REB"].mean()), 1),
        "orpg": round(float(games_df["OREB"].mean()), 1),
        "drpg": round(float(games_df["DREB"].mean()), 1),
        "spg": round(float(games_df["STL"].mean()), 1),
        "bpg": round(float(games_df["BLK"].mean()), 1),
        "fpg": round(float(games_df["PF"].mean()), 1),
    }])

    season_safe = season_str.replace("-", "_")
    upload_df_to_s3(stats, f"team_stats/{team_abbr_upper}_{season_safe}.csv")
    logging.info("Stats uploaded for %s %s", team_abbr_upper, season_str)


def fetch_team_roster(team_abbr: str, season_year: int = None):
    configure_nba_session()
    _ensure_team_lookup()

    if not season_year:
        now = datetime.now(ZoneInfo("America/New_York"))
        season_year = now.year if now.month >= 10 else now.year - 1

    season_str = f"{season_year}-{str(season_year + 1)[-2:]}"
    team_abbr_upper = team_abbr.upper()
    team_id = TEAMS_BY_TRI.get(team_abbr_upper, {}).get("id")

    if not team_id:
        logging.warning("Unknown team: %s", team_abbr)
        return

    logging.info("Fetching roster for %s %s ...", team_abbr_upper, season_str)

    roster_resp = _safe_call(commonteamroster.CommonTeamRoster, team_id=team_id, season=season_str)
    roster_df = roster_resp.get_data_frames()[0]

    stats_resp = _safe_call(
        leaguedashplayerstats.LeagueDashPlayerStats,
        season=season_str,
        season_type_all_star="Regular Season",
        team_id_nullable=team_id,
        per_mode_detailed="PerGame"
    )
    stats_df = stats_resp.get_data_frames()[0]

    stats_dict = {}
    if not stats_df.empty:
        for _, row in stats_df.iterrows():
            pid = row.get("PLAYER_ID")
            stats_dict[pid] = {
                "gp": int(row.get("GP", 0)),
                "ppg": round(float(row.get("PTS", 0)), 1),
                "rpg": round(float(row.get("REB", 0)), 1),
                "apg": round(float(row.get("AST", 0)), 1),
                "fg_pct": round(float(row.get("FG_PCT", 0) * 100), 1),
                "fg3_pct": round(float(row.get("FG3_PCT", 0) * 100), 1),
                "ft_pct": round(float(row.get("FT_PCT", 0) * 100), 1),
            }

    roster_rows = []
    for _, player in roster_df.iterrows():
        pid = player.get("PLAYER_ID")
        s = stats_dict.get(pid, {})
        roster_rows.append({
            "team_abbr": team_abbr_upper,
            "season": season_str,
            "player_name": player.get("PLAYER", "Unknown"),
            "position": player.get("POSITION", "N/A"),
            "games_played": s.get("gp", 0),
            "ppg": s.get("ppg", 0),
            "rpg": s.get("rpg", 0),
            "apg": s.get("apg", 0),
            "fg_pct": s.get("fg_pct", 0),
            "fg3_pct": s.get("fg3_pct", 0),
            "ft_pct": s.get("ft_pct", 0),
        })

    if roster_rows:
        out_df = pd.DataFrame(roster_rows)
        out_df = out_df.sort_values("ppg", ascending=False)
        season_safe = season_str.replace("-", "_")
        upload_df_to_s3(out_df, f"team_roster/{team_abbr_upper}_{season_safe}.csv")
        logging.info("Roster uploaded for %s %s", team_abbr_upper, season_str)


def fetch_team_schedule(team_abbr: str, season_year: int = None):
    configure_nba_session()
    _ensure_team_lookup()

    if not season_year:
        now = datetime.now(ZoneInfo("America/New_York"))
        season_year = now.year if now.month >= 10 else now.year - 1

    season_str = f"{season_year}-{str(season_year + 1)[-2:]}"
    team_abbr_upper = team_abbr.upper()
    team_id = TEAMS_BY_TRI.get(team_abbr_upper, {}).get("id")

    if not team_id:
        logging.warning("Unknown team: %s", team_abbr)
        return

    logging.info("Fetching schedule for %s %s ...", team_abbr_upper, season_str)

    finder = _safe_call(
        leaguegamefinder.LeagueGameFinder,
        team_id_nullable=team_id,
        season_nullable=season_str,
        season_type_nullable="Regular Season"
    )
    games_df = finder.get_data_frames()[0]

    if games_df.empty:
        logging.warning("No schedule for %s %s", team_abbr_upper, season_str)
        return

    schedule_rows = []
    for _, game in games_df.iterrows():
        matchup = str(game.get("MATCHUP", ""))
        is_home = "vs." in matchup
        parts = matchup.split("vs." if is_home else "@")
        opponent = parts[-1].strip() if len(parts) > 1 else "Unknown"
        pts = int(game.get("PTS", 0))
        plus_minus = int(game.get("PLUS_MINUS", 0))
        opp_pts = pts - plus_minus
        result = str(game.get("WL", "")).strip().upper()
        if result not in ["W", "L"]:
            result = "W" if plus_minus > 0 else "L"

        schedule_rows.append({
            "team_abbr": team_abbr_upper,
            "season": season_str,
            "game_date": str(game.get("GAME_DATE", "")),
            "opponent": opponent,
            "location": "Home" if is_home else "Away",
            "result": result,
            "points": pts,
            "opponent_points": opp_pts,
        })

    if schedule_rows:
        out_df = pd.DataFrame(schedule_rows)
        out_df = out_df.iloc[::-1].reset_index(drop=True)
        season_safe = season_str.replace("-", "_")
        upload_df_to_s3(out_df, f"team_schedule/{team_abbr_upper}_{season_safe}.csv")
        logging.info("Schedule uploaded for %s %s", team_abbr_upper, season_str)


def fetch_all_teams(season_year: int = None):
    _ensure_team_lookup()
    all_abbrs = sorted([info["tri"] for info in TEAMS_BY_ID.values()])
    logging.info("Fetching data for all %d teams ...", len(all_abbrs))

    for abbr in all_abbrs:
        try:
            fetch_team_stats(abbr, season_year)
            time.sleep(1)
            fetch_team_roster(abbr, season_year)
            time.sleep(1)
            fetch_team_schedule(abbr, season_year)
            time.sleep(1)
        except Exception as e:
            logging.error("Failed for %s: %s", abbr, e)
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch NBA data and upload to S3")
    parser.add_argument("--standings", action="store_true", help="Fetch standings")
    parser.add_argument("--team", type=str, help="Fetch data for one team (e.g. LAL)")
    parser.add_argument("--all", action="store_true", help="Fetch everything")
    parser.add_argument("--season", type=int, help="Season start year (e.g. 2025)")
    parser.add_argument("--gamelogs", action="store_true", help="Fetch season game logs")
    args = parser.parse_args()

    if args.all:
        fetch_standings(season_type="Regular Season")
        fetch_standings(season_type="Playoffs")
        fetch_all_teams(args.season)
    elif args.standings:
        fetch_standings(season_type="Regular Season")
        fetch_standings(season_type="Playoffs")
    elif args.team:
        fetch_team_stats(args.team, args.season)
        fetch_team_roster(args.team, args.season)
        fetch_team_schedule(args.team, args.season)
    elif args.gamelogs:
        fetch_season_game_logs("2024-25")
        fetch_season_game_logs("2025-26")
    else:
        print("Usage:")
        print("  python3 fetcher.py --standings")
        print("  python3 fetcher.py --team LAL")
        print("  python3 fetcher.py --all")
        print("  python3 fetcher.py --all --season 2025")