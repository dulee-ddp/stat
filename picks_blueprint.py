"""
picks_blueprint.py — Fantasy prediction game (Enhanced Edition)
Features: Triple Captain, Double Down, Streak Bonuses, Confidence Ranking

Picks are GLOBAL: one pick per user per game, shared across all leagues.
Leagues are social groups for separate leaderboards.
"""

import os
import sqlite3
import string
import random
import logging
from datetime import datetime, timedelta
from functools import wraps

import bcrypt
import jwt
from flask import Blueprint, jsonify, request, g

picks_bp = Blueprint("picks", __name__, url_prefix="/api/picks")

JWT_SECRET = os.getenv("JWT_SECRET", "statline-dev-secret-change-me")
JWT_EXP_HOURS = 72

# ── Scoring Constants ─────────────────────────────────────────────
BASE_POINTS = 10
STREAK_BONUS = {
    3: 5,
    5: 10,
    7: 20,
    10: 50,
}
CONFIDENCE_MULTIPLIERS = [1.5, 1.25, 1.0, 0.75, 0.5]

# ── DB ────────────────────────────────────────────────────────────

def _get_db_path():
    from pathlib import Path
    return os.environ.get("NBA_DB_PATH") or str(
        (Path(__file__).parent / "nba_live.sqlite").resolve()
    )

def get_db():
    if "picks_db" not in g:
        g.picks_db = sqlite3.connect(_get_db_path())
        g.picks_db.row_factory = sqlite3.Row
        g.picks_db.execute("PRAGMA journal_mode=WAL")
        g.picks_db.execute("PRAGMA foreign_keys=ON")
    return g.picks_db

@picks_bp.teardown_app_request
def close_db(exc):
    db = g.pop("picks_db", None)
    if db:
        db.close()

def _table_exists(con, table_name):
    row = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,)
    ).fetchone()
    return row is not None

def _get_columns(con, table_name):
    if not _table_exists(con, table_name):
        return []
    return [r[1] for r in con.execute(f"PRAGMA table_info({table_name})").fetchall()]

def _migrate_col(con, table, col, typedef):
    cols = _get_columns(con, table)
    if col not in cols:
        con.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typedef}")

def init_picks_db(app):
    path = _get_db_path()
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")

    # 1) Base tables first
    con.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL COLLATE NOCASE,
            email TEXT UNIQUE NOT NULL COLLATE NOCASE,
            password_hash TEXT NOT NULL,
            display_name TEXT DEFAULT NULL,
            favorite_team TEXT DEFAULT NULL,
            total_points INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS leagues (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            code TEXT UNIQUE NOT NULL,
            created_by INTEGER NOT NULL REFERENCES users(id),
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS league_members (
            league_id INTEGER NOT NULL REFERENCES leagues(id) ON DELETE CASCADE,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            joined_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (league_id, user_id)
        );

        CREATE TABLE IF NOT EXISTS picks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id),
            game_id TEXT NOT NULL,
            game_date TEXT NOT NULL,
            picked_team TEXT NOT NULL,
            home_tri TEXT NOT NULL,
            away_tri TEXT NOT NULL,
            locked INTEGER DEFAULT 0,
            result INTEGER DEFAULT NULL,
            points_earned INTEGER DEFAULT 0,
            confidence_rank INTEGER DEFAULT NULL,
            is_double_down INTEGER DEFAULT 0,
            is_triple_captain INTEGER DEFAULT 0,
            scored_at TEXT DEFAULT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            UNIQUE(user_id, game_id)
        );

        CREATE TABLE IF NOT EXISTS power_ups (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id),
            power_type TEXT NOT NULL,
            week_number INTEGER NOT NULL,
            year INTEGER NOT NULL,
            game_id TEXT DEFAULT NULL,
            used_at TEXT DEFAULT (datetime('now')),
            UNIQUE(user_id, power_type, week_number, year)
        );

        CREATE TABLE IF NOT EXISTS achievements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id),
            achievement_type TEXT NOT NULL,
            achieved_at TEXT DEFAULT (datetime('now')),
            metadata TEXT DEFAULT NULL,
            UNIQUE(user_id, achievement_type, achieved_at)
        );
    """)

    # 2) Check if old picks table still has league_id and migrate it safely
    picks_cols = _get_columns(con, "picks")
    if "league_id" in picks_cols:
        logging.info("Migrating picks table: removing league_id dependency...")

        con.executescript("""
            CREATE TABLE IF NOT EXISTS picks_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id),
                game_id TEXT NOT NULL,
                game_date TEXT NOT NULL,
                picked_team TEXT NOT NULL,
                home_tri TEXT NOT NULL,
                away_tri TEXT NOT NULL,
                locked INTEGER DEFAULT 0,
                result INTEGER DEFAULT NULL,
                points_earned INTEGER DEFAULT 0,
                confidence_rank INTEGER DEFAULT NULL,
                is_double_down INTEGER DEFAULT 0,
                is_triple_captain INTEGER DEFAULT 0,
                scored_at TEXT DEFAULT NULL,
                created_at TEXT DEFAULT (datetime('now')),
                UNIQUE(user_id, game_id)
            );
        """)

        old_cols = set(_get_columns(con, "picks"))

        select_parts = [
            "user_id",
            "game_id",
            "game_date",
            "picked_team",
            "home_tri",
            "away_tri",
            "MAX(COALESCE(locked, 0)) AS locked",
            "MAX(result) AS result",
            "MAX(COALESCE(points_earned, 0)) AS points_earned",
            "MAX(confidence_rank) AS confidence_rank",
            "MAX(COALESCE(is_double_down, 0)) AS is_double_down",
            "MAX(COALESCE(is_triple_captain, 0)) AS is_triple_captain",
            "MAX(scored_at) AS scored_at",
            "MIN(created_at) AS created_at"
        ]

        # If some old columns never existed, replace with literals
        fixed_select_parts = []
        for part in select_parts:
            if " AS " in part:
                fixed_select_parts.append(part)
                continue

            if part in old_cols:
                fixed_select_parts.append(part)
            else:
                if part == "points_earned":
                    fixed_select_parts.append("0 AS points_earned")
                elif part == "confidence_rank":
                    fixed_select_parts.append("NULL AS confidence_rank")
                elif part == "is_double_down":
                    fixed_select_parts.append("0 AS is_double_down")
                elif part == "is_triple_captain":
                    fixed_select_parts.append("0 AS is_triple_captain")
                elif part == "locked":
                    fixed_select_parts.append("0 AS locked")
                elif part == "result":
                    fixed_select_parts.append("NULL AS result")
                elif part == "scored_at":
                    fixed_select_parts.append("NULL AS scored_at")
                else:
                    fixed_select_parts.append(part)

        insert_sql = f"""
            INSERT OR IGNORE INTO picks_new (
                user_id, game_id, game_date, picked_team, home_tri, away_tri,
                locked, result, points_earned, confidence_rank,
                is_double_down, is_triple_captain, scored_at, created_at
            )
            SELECT
                {", ".join(fixed_select_parts)}
            FROM picks
            GROUP BY user_id, game_id
        """
        con.execute(insert_sql)
        con.execute("DROP TABLE picks")
        con.execute("ALTER TABLE picks_new RENAME TO picks")
        logging.info("Migration complete.")

    # 3) Make sure all expected columns exist even for older DBs
    _migrate_col(con, "users", "display_name", "TEXT DEFAULT NULL")
    _migrate_col(con, "users", "favorite_team", "TEXT DEFAULT NULL")
    _migrate_col(con, "users", "total_points", "INTEGER DEFAULT 0")

    _migrate_col(con, "picks", "locked", "INTEGER DEFAULT 0")
    _migrate_col(con, "picks", "result", "INTEGER DEFAULT NULL")
    _migrate_col(con, "picks", "points_earned", "INTEGER DEFAULT 0")
    _migrate_col(con, "picks", "confidence_rank", "INTEGER DEFAULT NULL")
    _migrate_col(con, "picks", "is_double_down", "INTEGER DEFAULT 0")
    _migrate_col(con, "picks", "is_triple_captain", "INTEGER DEFAULT 0")
    _migrate_col(con, "picks", "scored_at", "TEXT DEFAULT NULL")
    _migrate_col(con, "picks", "created_at", "TEXT DEFAULT (datetime('now'))")

    # 4) Only create indexes AFTER all migrations are done
    con.executescript("""
        CREATE INDEX IF NOT EXISTS idx_picks_user_date ON picks(user_id, game_date);
        CREATE INDEX IF NOT EXISTS idx_picks_game ON picks(game_id);
        CREATE INDEX IF NOT EXISTS idx_picks_result ON picks(result);
        CREATE INDEX IF NOT EXISTS idx_picks_points ON picks(points_earned);
        CREATE INDEX IF NOT EXISTS idx_powerups_user_week ON power_ups(user_id, week_number, year);
        CREATE INDEX IF NOT EXISTS idx_achievements_user ON achievements(user_id);
    """)

    con.commit()
    con.close()
    logging.info(f"Picks DB ready at {path}")

# ── Auth helpers ──────────────────────────────────────────────────

def _hash_pw(pw):
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()

def _check_pw(pw, h):
    return bcrypt.checkpw(pw.encode(), h.encode())

def _make_token(uid, usr):
    return jwt.encode(
        {"uid": uid, "usr": usr, "exp": datetime.utcnow() + timedelta(hours=JWT_EXP_HOURS)},
        JWT_SECRET,
        algorithm="HS256"
    )

def _decode_token(t):
    return jwt.decode(t, JWT_SECRET, algorithms=["HS256"])

def require_auth(f):
    @wraps(f)
    def wrapper(*a, **kw):
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return jsonify({"error": "Missing token"}), 401
        try:
            d = _decode_token(auth[7:])
            g.user_id = d["uid"]
            g.username = d["usr"]
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401
        return f(*a, **kw)
    return wrapper

def _gen_code(n=6):
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=n))

def _get_week_info(date_str=None):
    if date_str:
        d = datetime.strptime(date_str, "%Y-%m-%d")
    else:
        d = datetime.now()
    iso_cal = d.isocalendar()
    return iso_cal[1], iso_cal[0]

# ── Stats helpers ─────────────────────────────────────────────────

def _calc_streak(db, user_id):
    rows = db.execute(
        "SELECT result FROM picks WHERE user_id=? AND result IS NOT NULL ORDER BY scored_at DESC",
        (user_id,)
    ).fetchall()

    cur = best = run = 0
    first_break = False
    for r in rows:
        if r["result"] == 1:
            run += 1
            best = max(best, run)
        else:
            if not first_break:
                cur = run
                first_break = True
            run = 0

    if not first_break:
        cur = run
    best = max(best, run)
    return cur, best

def _calc_streak_bonus(streak_length):
    bonus = 0
    for threshold, points in sorted(STREAK_BONUS.items(), reverse=True):
        if streak_length >= threshold:
            bonus = points
            break
    return bonus

def _calculate_points(is_correct, confidence_rank=None, is_double_down=False,
                      is_triple_captain=False, current_streak=0):
    if not is_correct:
        return 0

    points = BASE_POINTS

    if confidence_rank is not None and 0 <= confidence_rank < len(CONFIDENCE_MULTIPLIERS):
        points = int(points * CONFIDENCE_MULTIPLIERS[confidence_rank])

    if is_triple_captain:
        points *= 3
    elif is_double_down:
        points *= 2

    points += _calc_streak_bonus(current_streak + 1)
    return points

def _leaderboard_query(db, league_id=None, date_from=None, date_to=None):
    date_conds, dp = [], []
    if date_from:
        date_conds.append("p.game_date>=?")
        dp.append(date_from)
    if date_to:
        date_conds.append("p.game_date<=?")
        dp.append(date_to)

    dw = (" AND " + " AND ".join(date_conds)) if date_conds else ""

    if league_id:
        q = f"""
            SELECT u.id, u.username, u.favorite_team, u.total_points,
                   COUNT(p.id) AS total,
                   SUM(CASE WHEN p.result=1 THEN 1 ELSE 0 END) AS correct,
                   SUM(CASE WHEN p.result=0 THEN 1 ELSE 0 END) AS wrong,
                   SUM(CASE WHEN p.result IS NULL THEN 1 ELSE 0 END) AS pending,
                   SUM(COALESCE(p.points_earned, 0)) AS points
            FROM league_members lm
            JOIN users u ON u.id=lm.user_id
            LEFT JOIN picks p ON p.user_id=u.id {dw}
            WHERE lm.league_id=?
            GROUP BY u.id
            ORDER BY points DESC, correct DESC, total ASC
        """
        rows = db.execute(q, dp + [league_id]).fetchall()
    else:
        q = f"""
            SELECT u.id, u.username, u.favorite_team, u.total_points,
                   COUNT(p.id) AS total,
                   SUM(CASE WHEN p.result=1 THEN 1 ELSE 0 END) AS correct,
                   SUM(CASE WHEN p.result=0 THEN 1 ELSE 0 END) AS wrong,
                   SUM(CASE WHEN p.result IS NULL THEN 1 ELSE 0 END) AS pending,
                   SUM(COALESCE(p.points_earned, 0)) AS points
            FROM users u
            JOIN picks p ON p.user_id=u.id {dw if dw else ""}
            GROUP BY u.id
            HAVING total > 0
            ORDER BY points DESC, correct DESC, total ASC
        """
        rows = db.execute(q, dp).fetchall()

    members = []
    for i, r in enumerate(rows):
        cs, bs = _calc_streak(db, r["id"])
        members.append({
            "rank": i + 1,
            "userId": r["id"],
            "username": r["username"],
            "favoriteTeam": r["favorite_team"],
            "total": r["total"],
            "correct": r["correct"] or 0,
            "wrong": r["wrong"] or 0,
            "pending": r["pending"] or 0,
            "streak": cs,
            "bestStreak": bs,
            "points": r["points"] or 0,
            "isMe": r["id"] == getattr(g, "user_id", None),
        })
    return members

# ── Auth routes ───────────────────────────────────────────────────

def _user_json(u):
    return {
        "id": u["id"],
        "username": u["username"],
        "email": u["email"],
        "favoriteTeam": u["favorite_team"],
        "totalPoints": u["total_points"] if "total_points" in u.keys() else 0
    }

@picks_bp.post("/auth/register")
def register():
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not username or not email or len(password) < 6:
        return jsonify({"error": "Username, email, and password (6+ chars) required"}), 400
    if len(username) < 3 or len(username) > 20:
        return jsonify({"error": "Username must be 3-20 characters"}), 400
    if "@" not in email or "." not in email:
        return jsonify({"error": "Please enter a valid email address"}), 400

    db = get_db()
    try:
        db.execute(
            "INSERT INTO users (username,email,password_hash,total_points) VALUES (?,?,?,0)",
            (username, email, _hash_pw(password))
        )
        db.commit()
        user = db.execute(
            "SELECT id,username,email,favorite_team,total_points FROM users WHERE email=?",
            (email,)
        ).fetchone()
        return jsonify({
            "ok": True,
            "token": _make_token(user["id"], user["username"]),
            "user": _user_json(user)
        }), 201
    except sqlite3.IntegrityError as e:
        err = str(e).lower()
        if "username" in err:
            return jsonify({"error": "Username already taken"}), 409
        if "email" in err:
            return jsonify({"error": "Email already registered"}), 409
        return jsonify({"error": "Registration failed"}), 409

@picks_bp.post("/auth/login")
def login():
    data = request.get_json(silent=True) or {}
    ident = (data.get("username") or data.get("email") or "").strip()
    pw = data.get("password") or ""
    if not ident or not pw:
        return jsonify({"error": "Username/email and password required"}), 400

    db = get_db()
    user = db.execute(
        "SELECT id,username,email,password_hash,favorite_team,total_points FROM users WHERE username=? OR email=?",
        (ident, ident.lower())
    ).fetchone()

    if not user or not _check_pw(pw, user["password_hash"]):
        return jsonify({"error": "Invalid credentials"}), 401

    return jsonify({
        "ok": True,
        "token": _make_token(user["id"], user["username"]),
        "user": _user_json(user)
    })

@picks_bp.get("/auth/me")
@require_auth
def me():
    db = get_db()
    user = db.execute(
        "SELECT id,username,email,favorite_team,total_points,created_at FROM users WHERE id=?",
        (g.user_id,)
    ).fetchone()
    if not user:
        return jsonify({"error": "User not found"}), 404

    leagues = db.execute("""
        SELECT l.id,l.name,l.code,l.created_by
        FROM leagues l
        JOIN league_members lm ON l.id=lm.league_id
        WHERE lm.user_id=?
        ORDER BY l.name
    """, (g.user_id,)).fetchall()

    week_num, year = _get_week_info()
    power_ups = db.execute("""
        SELECT power_type, game_id
        FROM power_ups
        WHERE user_id=? AND week_number=? AND year=?
    """, (g.user_id, week_num, year)).fetchall()

    power_up_status = {
        "tripleCaptainUsed": any(p["power_type"] == "triple_captain" for p in power_ups),
        "tripleCaptainGameId": next((p["game_id"] for p in power_ups if p["power_type"] == "triple_captain"), None),
    }

    return jsonify({
        "ok": True,
        "user": {
            **_user_json(user),
            "createdAt": user["created_at"]
        },
        "leagues": [{
            "id": l["id"],
            "name": l["name"],
            "code": l["code"],
            "isOwner": l["created_by"] == g.user_id
        } for l in leagues],
        "powerUps": power_up_status,
        "currentWeek": week_num,
    })

@picks_bp.put("/auth/profile")
@require_auth
def update_profile():
    data = request.get_json(silent=True) or {}
    db = get_db()
    updates, params = [], []

    if "favoriteTeam" in data:
        ft = (data["favoriteTeam"] or "").upper().strip()
        updates.append("favorite_team=?")
        params.append(ft if ft else None)

    if "displayName" in data:
        dn = (data["displayName"] or "").strip()
        updates.append("display_name=?")
        params.append(dn if dn else None)

    if not updates:
        return jsonify({"error": "Nothing to update"}), 400

    params.append(g.user_id)
    db.execute(f"UPDATE users SET {', '.join(updates)} WHERE id=?", params)
    db.commit()
    return jsonify({"ok": True})

@picks_bp.get("/auth/stats")
@require_auth
def user_stats():
    db = get_db()
    user = db.execute(
        "SELECT favorite_team, total_points FROM users WHERE id=?",
        (g.user_id,)
    ).fetchone()

    row = db.execute("""
        SELECT COUNT(id) AS total,
               SUM(CASE WHEN result=1 THEN 1 ELSE 0 END) AS correct,
               SUM(CASE WHEN result=0 THEN 1 ELSE 0 END) AS wrong,
               SUM(CASE WHEN result IS NULL THEN 1 ELSE 0 END) AS pending,
               SUM(COALESCE(points_earned, 0)) AS total_points
        FROM picks
        WHERE user_id=?
    """, (g.user_id,)).fetchone()

    cs, bs = _calc_streak(db, g.user_id)

    leagues = db.execute("""
        SELECT l.id,l.name
        FROM league_members lm
        JOIN leagues l ON l.id=lm.league_id
        WHERE lm.user_id=?
    """, (g.user_id,)).fetchall()

    league_stats = []
    for l in leagues:
        lr = db.execute("""
            SELECT COUNT(p.id) AS total,
                   SUM(CASE WHEN p.result=1 THEN 1 ELSE 0 END) AS correct,
                   SUM(CASE WHEN p.result=0 THEN 1 ELSE 0 END) AS wrong,
                   SUM(COALESCE(p.points_earned, 0)) AS points
            FROM picks p
            WHERE p.user_id=?
        """, (g.user_id,)).fetchone()

        league_stats.append({
            "id": l["id"],
            "name": l["name"],
            "total": lr["total"],
            "correct": lr["correct"] or 0,
            "wrong": lr["wrong"] or 0,
            "points": lr["points"] or 0
        })

    achievements = db.execute("""
        SELECT achievement_type, achieved_at, metadata
        FROM achievements
        WHERE user_id=?
        ORDER BY achieved_at DESC
        LIMIT 10
    """, (g.user_id,)).fetchall()

    return jsonify({
        "ok": True,
        "total": row["total"] or 0,
        "correct": row["correct"] or 0,
        "wrong": row["wrong"] or 0,
        "pending": row["pending"] or 0,
        "points": row["total_points"] or 0,
        "streak": cs,
        "bestStreak": bs,
        "favoriteTeam": user["favorite_team"] if user else None,
        "leagues": league_stats,
        "achievements": [{"type": a["achievement_type"], "at": a["achieved_at"]} for a in achievements]
    })

@picks_bp.get("/history")
@require_auth
def pick_history():
    limit = min(int(request.args.get("limit", 50)), 200)
    offset = int(request.args.get("offset", 0))
    db = get_db()

    rows = db.execute("""
        SELECT game_id, game_date, picked_team, home_tri, away_tri, result, locked,
               points_earned, confidence_rank, is_double_down, is_triple_captain, created_at
        FROM picks
        WHERE user_id=?
        ORDER BY game_date DESC, created_at DESC
        LIMIT ? OFFSET ?
    """, (g.user_id, limit, offset)).fetchall()

    total = db.execute("SELECT COUNT(*) FROM picks WHERE user_id=?", (g.user_id,)).fetchone()[0]

    return jsonify({
        "ok": True,
        "total": total,
        "limit": limit,
        "offset": offset,
        "picks": [{
            "gameId": r["game_id"],
            "gameDate": r["game_date"],
            "pickedTeam": r["picked_team"],
            "homeTri": r["home_tri"],
            "awayTri": r["away_tri"],
            "result": r["result"],
            "locked": bool(r["locked"]),
            "pointsEarned": r["points_earned"] or 0,
            "confidenceRank": r["confidence_rank"],
            "isDoubleDown": bool(r["is_double_down"]),
            "isTripleCaptain": bool(r["is_triple_captain"])
        } for r in rows],
    })

# ── League routes ─────────────────────────────────────────────────

@picks_bp.post("/leagues/create")
@require_auth
def create_league():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    if not name or len(name) > 40:
        return jsonify({"error": "League name required (max 40 chars)"}), 400

    db = get_db()
    code = None
    for _ in range(10):
        maybe = _gen_code()
        if not db.execute("SELECT 1 FROM leagues WHERE code=?", (maybe,)).fetchone():
            code = maybe
            break

    if code is None:
        return jsonify({"error": "Could not generate league code"}), 500

    db.execute("INSERT INTO leagues (name,code,created_by) VALUES (?,?,?)", (name, code, g.user_id))
    lid = db.execute("SELECT last_insert_rowid()").fetchone()[0]
    db.execute("INSERT INTO league_members (league_id,user_id) VALUES (?,?)", (lid, g.user_id))
    db.commit()

    return jsonify({
        "ok": True,
        "league": {"id": lid, "name": name, "code": code}
    }), 201

@picks_bp.post("/leagues/join")
@require_auth
def join_league():
    data = request.get_json(silent=True) or {}
    code = (data.get("code") or "").strip().upper()
    if not code:
        return jsonify({"error": "League code required"}), 400

    db = get_db()
    league = db.execute("SELECT id,name FROM leagues WHERE code=?", (code,)).fetchone()
    if not league:
        return jsonify({"error": "League not found"}), 404

    if db.execute(
        "SELECT 1 FROM league_members WHERE league_id=? AND user_id=?",
        (league["id"], g.user_id)
    ).fetchone():
        return jsonify({"error": "Already a member"}), 409

    db.execute("INSERT INTO league_members (league_id,user_id) VALUES (?,?)", (league["id"], g.user_id))
    db.commit()

    return jsonify({
        "ok": True,
        "league": {"id": league["id"], "name": league["name"], "code": code}
    })

@picks_bp.post("/leagues/<int:league_id>/leave")
@require_auth
def leave_league(league_id):
    db = get_db()
    db.execute("DELETE FROM league_members WHERE league_id=? AND user_id=?", (league_id, g.user_id))
    db.commit()
    return jsonify({"ok": True})

@picks_bp.get("/leagues/<int:league_id>/leaderboard")
@require_auth
def leaderboard(league_id):
    db = get_db()
    if not db.execute(
        "SELECT 1 FROM league_members WHERE league_id=? AND user_id=?",
        (league_id, g.user_id)
    ).fetchone():
        return jsonify({"error": "Not a member"}), 403

    league = db.execute("SELECT id,name,code FROM leagues WHERE id=?", (league_id,)).fetchone()
    if not league:
        return jsonify({"error": "League not found"}), 404

    members = _leaderboard_query(
        db,
        league_id=league_id,
        date_from=request.args.get("from"),
        date_to=request.args.get("to")
    )

    return jsonify({
        "ok": True,
        "league": {"id": league["id"], "name": league["name"], "code": league["code"]},
        "leaderboard": members
    })

@picks_bp.get("/global-leaderboard")
@require_auth
def global_leaderboard():
    db = get_db()
    members = _leaderboard_query(
        db,
        date_from=request.args.get("from"),
        date_to=request.args.get("to")
    )
    return jsonify({"ok": True, "leaderboard": members})

# ── Pick routes ───────────────────────────────────────────────────

@picks_bp.post("/make")
@require_auth
def make_pick():
    data = request.get_json(silent=True) or {}
    game_id = data.get("gameId")
    game_date = data.get("gameDate")
    picked_team = (data.get("pickedTeam") or "").upper()
    home_tri = (data.get("homeTri") or "").upper()
    away_tri = (data.get("awayTri") or "").upper()
    game_status = int(data.get("gameStatus", 1))
    confidence_rank = data.get("confidenceRank")
    is_double_down = bool(data.get("isDoubleDown", False))
    is_triple_captain = bool(data.get("isTripleCaptain", False))

    if not all([game_id, game_date, picked_team, home_tri, away_tri]):
        return jsonify({"error": "Missing required fields"}), 400
    if picked_team not in (home_tri, away_tri):
        return jsonify({"error": "Picked team must be home or away"}), 400

    try:
        datetime.strptime(game_date, "%Y-%m-%d")
    except ValueError:
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400

    if confidence_rank is not None:
        try:
            confidence_rank = int(confidence_rank)
        except Exception:
            return jsonify({"error": "confidenceRank must be an integer"}), 400
        if confidence_rank < 0 or confidence_rank >= len(CONFIDENCE_MULTIPLIERS):
            return jsonify({"error": "confidenceRank must be between 0 and 4"}), 400

    db = get_db()

    existing = db.execute(
        "SELECT id,locked,result FROM picks WHERE user_id=? AND game_id=?",
        (g.user_id, game_id)
    ).fetchone()

    if existing:
        if existing["result"] is not None:
            return jsonify({"error": "Game already scored"}), 400
        if existing["locked"]:
            return jsonify({"error": "Pick locked — game has started"}), 400

    if game_status >= 2:
        return jsonify({"error": "Game already started — picks are locked"}), 400

    week_num, year = _get_week_info(game_date)

    if is_triple_captain:
        tc_used = db.execute("""
            SELECT 1 FROM power_ups
            WHERE user_id=? AND power_type='triple_captain' AND week_number=? AND year=?
        """, (g.user_id, week_num, year)).fetchone()

        if tc_used:
            return jsonify({"error": "Triple Captain already used this week"}), 400

    if is_double_down:
        dd_used = db.execute("""
            SELECT 1 FROM picks
            WHERE user_id=? AND game_date=? AND is_double_down=1 AND game_id!=?
        """, (g.user_id, game_date, game_id)).fetchone()

        if dd_used:
            return jsonify({"error": "Double Down already used today"}), 400

    try:
        db.execute("""
            INSERT INTO picks (
                user_id, game_id, game_date, picked_team, home_tri, away_tri,
                confidence_rank, is_double_down, is_triple_captain
            )
            VALUES (?,?,?,?,?,?,?,?,?)
            ON CONFLICT(user_id, game_id) DO UPDATE SET
                picked_team=excluded.picked_team,
                confidence_rank=excluded.confidence_rank,
                is_double_down=excluded.is_double_down,
                is_triple_captain=excluded.is_triple_captain,
                created_at=datetime('now')
            WHERE result IS NULL AND locked=0
        """, (
            g.user_id, game_id, game_date, picked_team, home_tri, away_tri,
            confidence_rank, int(is_double_down), int(is_triple_captain)
        ))

        if is_triple_captain:
            db.execute("""
                INSERT OR REPLACE INTO power_ups (user_id, power_type, week_number, year, game_id)
                VALUES (?, 'triple_captain', ?, ?, ?)
            """, (g.user_id, week_num, year, game_id))

        db.commit()

        return jsonify({
            "ok": True,
            "pick": {
                "gameId": game_id,
                "pickedTeam": picked_team,
                "confidenceRank": confidence_rank,
                "isDoubleDown": is_double_down,
                "isTripleCaptain": is_triple_captain
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@picks_bp.post("/lock")
def lock_games():
    data = request.get_json(silent=True) or {}
    game_ids = data.get("gameIds") or []
    if not game_ids:
        return jsonify({"error": "No game IDs"}), 400

    db = get_db()
    locked = 0
    for gid in game_ids:
        cur = db.execute(
            "UPDATE picks SET locked=1 WHERE game_id=? AND locked=0 AND result IS NULL",
            (gid,)
        )
        locked += cur.rowcount

    db.commit()
    return jsonify({"ok": True, "locked": locked})

@picks_bp.get("/my")
@require_auth
def my_picks():
    game_date = request.args.get("date")
    if not game_date:
        return jsonify({"error": "date param required"}), 400

    db = get_db()
    rows = db.execute("""
        SELECT * FROM picks WHERE user_id=? AND game_date=?
    """, (g.user_id, game_date)).fetchall()

    current_streak, _ = _calc_streak(db, g.user_id)
    week_num, year = _get_week_info(game_date)

    tc_used = db.execute("""
        SELECT game_id FROM power_ups
        WHERE user_id=? AND power_type='triple_captain' AND week_number=? AND year=?
    """, (g.user_id, week_num, year)).fetchone()

    dd_used = db.execute("""
        SELECT game_id FROM picks
        WHERE user_id=? AND game_date=? AND is_double_down=1
    """, (g.user_id, game_date)).fetchone()

    return jsonify({
        "ok": True,
        "picks": [{
            "gameId": r["game_id"],
            "pickedTeam": r["picked_team"],
            "homeTri": r["home_tri"],
            "awayTri": r["away_tri"],
            "result": r["result"],
            "locked": bool(r["locked"]),
            "pointsEarned": r["points_earned"] or 0,
            "confidenceRank": r["confidence_rank"],
            "isDoubleDown": bool(r["is_double_down"]),
            "isTripleCaptain": bool(r["is_triple_captain"]),
            "potentialPoints": _calculate_points(
                True,
                r["confidence_rank"],
                bool(r["is_double_down"]),
                bool(r["is_triple_captain"]),
                current_streak
            ) if r["result"] is None else None
        } for r in rows],
        "currentStreak": current_streak,
        "powerUps": {
            "tripleCaptainAvailable": tc_used is None,
            "tripleCaptainGameId": tc_used["game_id"] if tc_used else None,
            "doubleDownAvailable": dd_used is None,
            "doubleDownGameId": dd_used["game_id"] if dd_used else None
        }
    })

@picks_bp.get("/power-ups")
@require_auth
def get_power_ups():
    db = get_db()
    game_date = request.args.get("date")

    if game_date:
        week_num, year = _get_week_info(game_date)
    else:
        week_num, year = _get_week_info()

    tc_used = db.execute("""
        SELECT game_id, used_at FROM power_ups
        WHERE user_id=? AND power_type='triple_captain' AND week_number=? AND year=?
    """, (g.user_id, week_num, year)).fetchone()

    total_tc = db.execute("""
        SELECT COUNT(*) FROM power_ups WHERE user_id=? AND power_type='triple_captain'
    """, (g.user_id,)).fetchone()[0]

    return jsonify({
        "ok": True,
        "currentWeek": week_num,
        "year": year,
        "tripleCaptain": {
            "available": tc_used is None,
            "usedGameId": tc_used["game_id"] if tc_used else None,
            "usedAt": tc_used["used_at"] if tc_used else None,
            "totalUsed": total_tc
        }
    })

# ── Scoring ───────────────────────────────────────────────────────

@picks_bp.post("/score")
def score_games():
    data = request.get_json(silent=True) or {}
    results = data.get("results") or []
    if not results:
        return jsonify({"error": "No results"}), 400

    db = get_db()
    scored = 0
    points_awarded = 0

    for r in results:
        gid = r.get("gameId")
        winner = (r.get("winnerTri") or "").upper()
        if not gid or not winner:
            continue

        picks = db.execute("""
            SELECT id, user_id, game_id, picked_team, confidence_rank,
                   is_double_down, is_triple_captain
            FROM picks
            WHERE game_id=? AND result IS NULL
        """, (gid,)).fetchall()

        for pick in picks:
            is_correct = pick["picked_team"] == winner
            result = 1 if is_correct else 0

            current_streak, _ = _calc_streak(db, pick["user_id"])

            points = _calculate_points(
                is_correct,
                pick["confidence_rank"],
                bool(pick["is_double_down"]),
                bool(pick["is_triple_captain"]),
                current_streak
            )

            db.execute("""
                UPDATE picks SET
                    result=?,
                    points_earned=?,
                    scored_at=datetime('now'),
                    locked=1
                WHERE id=?
            """, (result, points, pick["id"]))

            if points > 0:
                db.execute("""
                    UPDATE users SET total_points = total_points + ? WHERE id=?
                """, (points, pick["user_id"]))
                points_awarded += points

            if is_correct:
                new_streak = current_streak + 1
                _check_achievements(db, pick["user_id"], new_streak, pick)

            scored += 1

    db.commit()
    return jsonify({
        "ok": True,
        "scored": scored,
        "pointsAwarded": points_awarded
    })

def _check_achievements(db, user_id, streak, pick):
    achievements = []

    streak_achievements = {
        3: "hot_streak_3",
        5: "hot_streak_5",
        7: "hot_streak_7",
        10: "legendary_10",
        15: "unstoppable_15",
        20: "godlike_20"
    }

    if streak in streak_achievements:
        try:
            db.execute("""
                INSERT INTO achievements (user_id, achievement_type, metadata)
                VALUES (?, ?, ?)
            """, (user_id, streak_achievements[streak], f'{{"streak":{streak}}}'))
            achievements.append(streak_achievements[streak])
        except sqlite3.IntegrityError:
            pass

    if pick["is_triple_captain"]:
        try:
            db.execute("""
                INSERT INTO achievements (user_id, achievement_type, metadata)
                VALUES (?, 'triple_captain_success', ?)
            """, (user_id, f'{{"game_id":"{pick["game_id"]}"}}'))
        except sqlite3.IntegrityError:
            pass

    return achievements