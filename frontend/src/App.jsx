import { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

// Configure API base URL via environment variable or use empty string for same-origin
const API_BASE = "http://52.70.223.60";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
  Legend,
} from "recharts";

const TEAM_COLORS = {
  ATL: ["#E03A3E", "#C1D32F"],
  BOS: ["#007A33", "#BA9653"],
  BKN: ["#FFFFFF", "#000000"],
  CHA: ["#1D1160", "#00788C"],
  CHI: ["#CE1141", "#000000"],
  CLE: ["#860038", "#FDBB30"],
  DAL: ["#00538C", "#B8C4CA"],
  DEN: ["#0E2240", "#FEC524"],
  DET: ["#C8102E", "#1D42BA"],
  GSW: ["#1D428A", "#FFC72C"],
  HOU: ["#CE1141", "#C4CED4"],
  IND: ["#002D62", "#FDBB30"],
  LAC: ["#C8102E", "#1D428A"],
  LAL: ["#552583", "#FDB927"],
  MEM: ["#5D76A9", "#12173F"],
  MIA: ["#98002E", "#F9A01B"],
  MIL: ["#00471B", "#EEE1C6"],
  MIN: ["#0C2340", "#78BE20"],
  NOP: ["#0C2340", "#C8102E"],
  NYK: ["#006BB6", "#F58426"],
  OKC: ["#007AC1", "#EF3B24"],
  ORL: ["#0077C0", "#C4CED4"],
  PHI: ["#006BB6", "#ED174C"],
  PHX: ["#1D1160", "#E56020"],
  POR: ["#E03A3E", "#000000"],
  SAC: ["#5A2D81", "#63727A"],
  SAS: ["#C4CED4", "#000000"],
  TOR: ["#CE1141", "#000000"],
  UTA: ["#002B5C", "#F9A01B"],
  WAS: ["#002B5C", "#E31837"],
};

const NBA_TEAM_IDS = {
  ATL:1610612737, BOS:1610612738, BKN:1610612751, CHA:1610612766, CHI:1610612741,
  CLE:1610612739, DAL:1610612742, DEN:1610612743, DET:1610612765, GSW:1610612744,
  HOU:1610612745, IND:1610612754, LAC:1610612746, LAL:1610612747, MEM:1610612763,
  MIA:1610612748, MIL:1610612749, MIN:1610612750, NOP:1610612740, NYK:1610612752,
  OKC:1610612760, ORL:1610612753, PHI:1610612755, PHX:1610612756, POR:1610612757,
  SAC:1610612758, SAS:1610612759, TOR:1610612761, UTA:1610612762, WAS:1610612764,
};

function teamTheme(tri) {
  const key = (tri || "").toUpperCase();
  const [a, b] = TEAM_COLORS[key] || ["#7c5cff", "#22c55e"];
  return { a: ensureVisible(a), b: ensureVisible(b) };
}

function ensureVisible(hex) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  const lum = 0.299 * r + 0.587 * g + 0.114 * b;
  if (lum >= 48) return hex; // visible enough on dark bg
  const t = Math.min(0.58, (48 - lum) / (255 - lum + 1) + 0.30);
  const mix = (c) => Math.min(255, Math.round(c + (255 - c) * t));
  return `#${mix(r).toString(16).padStart(2,"0")}${mix(g).toString(16).padStart(2,"0")}${mix(b).toString(16).padStart(2,"0")}`;
}

/* date helpers */
function ymdNow() {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(
    d.getDate()
  ).padStart(2, "0")}`;
}
function parseYMDLocal(ymd) {
  const [y, m, d] = ymd.split("-").map(Number);
  return new Date(y, m - 1, d);
}
function fmtLongLocal(ymd) {
  const d = parseYMDLocal(ymd);
  return d.toLocaleDateString(undefined, {
    weekday: "long",
    year: "numeric",
    month: "long",
    day: "numeric",
  });
}
function isTodayStr(ymd) {
  return ymd === ymdNow();
}
function shiftDate(ymd, days) {
  const d = parseYMDLocal(ymd);
  d.setDate(d.getDate() + days);
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}

/* formatting helpers */
function tipoffMinutesFromETString(tip) {
  if (!tip) return 24 * 60 + 1;
  const m = tip.match(/(\d{1,2}):(\d{2})\s*([AP])M/i);
  if (!m) return 24 * 60 + 1;
  let hh = parseInt(m[1], 10) % 12;
  const mm = parseInt(m[2], 10);
  if (m[3].toUpperCase() === "P") hh += 12;
  return hh * 60 + mm;
}

function formatClock(clock) {
  if (!clock) return "";
  const m = clock.match(/PT(\d+)M(\d+(?:\.\d+)?)S/i);
  if (m) {
    const mm = String(parseInt(m[1], 10));
    const ss = String(Math.floor(parseFloat(m[2]))).padStart(2, "0");
    return `${mm}:${ss}`;
  }
  return clock;
}
function isEndedLike(g) {
  const period = g?.livePeriod || 0;
  const raw = g?.liveClock || "";
  const zeroish = raw === "" || /^PT0?0M0?0(\.0+)?S$/i.test(raw);
  return g?.gameStatusId === 3 || (g?.gameStatusId === 2 && period >= 4 && zeroish);
}

/* small UI stuff */
function Badge({ kind, children }) {
  const cls =
    kind === "live"
      ? "badge badge-live"
      : kind === "final"
      ? "badge badge-final"
      : kind === "ended"
      ? "badge badge-ended"
      : "badge badge-scheduled";
  return <span className={cls}>{children}</span>;
}
function StatusBadge({ g }) {
  if (g?.gameStatusId === 3 || isEndedLike(g)) {
    return <Badge kind="final">Final</Badge>;
  }
  if (g?.gameStatusId === 2) {
    const p = g?.livePeriod ? `Q${g.livePeriod}` : "LIVE";
    const clock = formatClock(g?.liveClock);
    return <Badge kind="live">{clock ? `${p} • ${clock}` : p}</Badge>;
  }
  return <Badge kind="scheduled">{g?.tipoffET || g?.status || "TBD"}</Badge>;
}

/* Team-aligned score rows with logos */
function TeamRow({ team = {}, score, emphasize = false, accent = "#7c5cff" }) {
  const tri = (team.tri || "").toUpperCase();
  const name = team.name || tri || "—";
  const logoSrc = team.id
    ? `https://cdn.nba.com/logos/nba/${team.id}/global/L/logo.svg`
    : null;

  return (
    <div
      className={`team-row ${emphasize ? "lead" : ""}`}
      style={{ "--rowAccent": accent }}
    >
      <div className="team-main">
        {logoSrc ? (
          <img
            className="team-logo"
            src={logoSrc}
            alt={tri || name}
            loading="lazy"
            onError={(e) => (e.currentTarget.style.display = "none")}
          />
        ) : (
          <div className="team-bubble small">{(tri || "—").slice(0, 3)}</div>
        )}
        <div className="team-text">
          <div className="team-name">{name}</div>
          <div className="team-tri chip">{tri}</div>
        </div>
      </div>
      <div className="team-score">{score ?? "—"}</div>
    </div>
  );
}

function GameCard({ g, onClick }) {
  const hs = g.homeScore ?? null;
  const as = g.awayScore ?? null;
  const homeLeads = hs != null && as != null ? hs > as : false;
  const awayLeads = hs != null && as != null ? as > hs : false;

  const homeTri = (g.home?.tri || "HOME").toUpperCase();
  const awayTri = (g.away?.tri || "AWAY").toUpperCase();

  const homeTheme = teamTheme(homeTri);
  const awayTheme = teamTheme(awayTri);

  return (
    <div
      className="gameCard"
      onClick={() => onClick?.(g)}
      role="button"
      tabIndex={0}
      style={{
        "--teamA": homeTheme.a,
        "--teamB": awayTheme.a,
        "--teamA2": homeTheme.b,
        "--teamB2": awayTheme.b,
      }}
    >
      <div className="gameCardInner">
        <div className="gameCardTop">
          <div className="matchupTitle">
            <span className="tri">{homeTri}</span>
            <span className="vs">vs</span>
            <span className="tri">{awayTri}</span>
          </div>
          <StatusBadge g={g} />
        </div>

        <div className="rows">
          <TeamRow
            team={g.home}
            score={g.homeScore}
            emphasize={homeLeads && (g.gameStatusId === 2 || g.gameStatusId === 3)}
            accent={homeTheme.a}
          />
          <TeamRow
            team={g.away}
            score={g.awayScore}
            emphasize={awayLeads && (g.gameStatusId === 2 || g.gameStatusId === 3)}
            accent={awayTheme.a}
          />
        </div>

        <div className="meta">
          {g.arena ? <div>Arena: {g.arena}</div> : <div>&nbsp;</div>}
          <div className="game-id">Game ID: {g.gameId}</div>
        </div>
      </div>
    </div>
  );
}

/* Date navigator */
function DateNav({ value, min, max, onChange, onToday, onPrev, onNext, isToday }) {
  return (
    <div className="datebar">
      <div className="datebar-left">
        <button className="btn" onClick={onPrev} aria-label="Previous day">
          ◀
        </button>
        <input
          className="date-input"
          type="date"
          min={min}
          max={max}
          value={value}
          onChange={(e) => onChange(e.target.value)}
        />
        <button className="btn" onClick={onNext} aria-label="Next day">
          ▶
        </button>
      </div>
      <div className="datebar-right">
        <button className="btn" onClick={onToday} disabled={isToday}>
          Today
        </button>
      </div>
    </div>
  );
}

function Donut({ value = 0, label = "Win %", stroke = "var(--accent)" }) {
  const pct = Math.max(0, Math.min(100, value));
  const r = 58;
  const c = 2 * Math.PI * r;
  const dash = (pct / 100) * c;

  return (
    <div className="donut">
      <svg viewBox="0 0 140 140" className="donut-svg" aria-label={`Win probability ${pct}%`}>
        <circle className="donut-track" cx="70" cy="70" r={r} />
        <circle
          className="donut-fill"
          cx="70"
          cy="70"
          r={r}
          stroke={stroke}
          strokeDasharray={`${dash} ${c - dash}`}
          transform="rotate(-90 70 70)"
        />
      </svg>
      <div className="donut-center">
        <div className="donut-big">{pct.toFixed(0)}%</div>
        <div className="donut-sub">{label}</div>
      </div>
    </div>
  );
}


/* ── Explainability panel ── */
function ExplainPanel({ dateStr, game, prediction }) {
  const [explain, setExplain] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]   = useState(null);
  const [expanded, setExpanded] = useState(false);

  // Only fetch when user expands the panel (lazy load — SHAP is slow)
  async function fetchExplain() {
    if (explain) { setExpanded(true); return; }
    setExpanded(true);
    setLoading(true);
    setError(null);
    try {
      const gameId = game?.gameId;
      const url = new URL(`${API_BASE}/api/explain/${dateStr}`);
      if (gameId) url.searchParams.set("gameId", gameId);

      const res  = await fetch(url.toString(), { cache: "no-store" });
      const json = await res.json();

      if (!res.ok) throw new Error(json?.error || `Server error ${res.status}`);

      // If we passed a gameId the backend returns { explanation: {...} }
      // If not, it returns { games: [...] } — pick the matching one
      let match = null;
      if (json.explanation) {
        match = json.explanation;
      } else if (Array.isArray(json.games)) {
        const homeT = (game?.home?.tri || "").toUpperCase();
        const awayT = (game?.away?.tri || "").toUpperCase();
        match =
          json.games.find((g) => g.numbers?.game_id === gameId) ||
          json.games.find((g) =>
            (g.headline || "").toUpperCase().includes(homeT) &&
            (g.headline || "").toUpperCase().includes(awayT)
          ) ||
          json.games[0];
      }

      if (!match) throw new Error("No explanation found for this game.");
      setExplain(match);
    } catch (e) {
      setError(e.message || String(e));
    } finally {
      setLoading(false);
    }
  }

  // Don't show if no prediction loaded yet
  if (!prediction) return null;

  return (
    <div className="explain-wrapper">
      {!expanded ? (
        <button className="explain-toggle" onClick={fetchExplain}>
          🔍 Why did the model pick this?
        </button>
      ) : (
        <div className="explain-panel">
          <button className="explain-close" onClick={() => setExpanded(false)}>✕ Close</button>

          {loading && (
            <div className="explain-loading">
              <div className="explain-spinner" />
              <span>Generating explanation… (this may take a moment)</span>
            </div>
          )}

          {error && !loading && (
            <div className="explain-error">⚠️ {error}</div>
          )}

          {explain && !loading && (
            <>
              <div className="explain-headline">{explain.headline}</div>

              <div className="explain-narrative">{explain.narrative}</div>

              {(explain.top_reasons_home?.length > 0 || explain.top_reasons_away?.length > 0) && (
                <div className="explain-factors">
                  {explain.top_reasons_home?.length > 0 && (
                    <div className="explain-col explain-col-home">
                      <div className="explain-col-title">🏠 Home Advantages</div>
                      {explain.top_reasons_home.map((r, i) => (
                        <div key={i} className="explain-reason explain-reason-home">
                          {r.reason}
                        </div>
                      ))}
                    </div>
                  )}
                  {explain.top_reasons_away?.length > 0 && (
                    <div className="explain-col explain-col-away">
                      <div className="explain-col-title">✈️ Away Advantages</div>
                      {explain.top_reasons_away.map((r, i) => (
                        <div key={i} className="explain-reason explain-reason-away">
                          {r.reason}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {explain.caveat && (
                <div className="explain-caveat">⚠️ {explain.caveat}</div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}

function LiveProbChart({
  data = [],
  homeName = "Home",
  awayName = "Away",
  homeTri = "HOME",
  awayTri = "AWAY",
  statusText = "live",
  homeColor = "var(--accent)",
  awayColor = "#22c55e",
}) {
  const chartData = useMemo(() => {
    return (Array.isArray(data) ? data : [])
      .map((d) => {
        const homeP = Number(d.p);
        const awayP = 100 - homeP;

        const period = Number(d.period);
        const secLeft = Number(d.secLeft);

        const periodIdx = Math.min(4, Math.max(1, period)) - 1;
        const secIntoPeriod = 12 * 60 - secLeft;
        const x = periodIdx * 12 * 60 + secIntoPeriod;

        return { ...d, period, secLeft, x, homeP, awayP };
      })
      .filter((d) => Number.isFinite(d.x) && Number.isFinite(d.homeP) && Number.isFinite(d.awayP))
      .sort((a, b) => a.x - b.x);
  }, [data]);

  const last = chartData.length ? chartData[chartData.length - 1] : null;

  function fmtX(secElapsed) {
    const q = Math.floor(secElapsed / (12 * 60)) + 1;
    const inQ = secElapsed % (12 * 60);
    const left = 12 * 60 - inQ;
    const mm = Math.floor(left / 60);
    const ss = String(left % 60).padStart(2, "0");
    return `Q${q} ${mm}:${ss}`;
  }

  return (
    <div className="market-card">
      <div className="market-head">
        <div>
          <div className="market-title">Live Prediction</div>
          <div className="market-sub">Win probability over time</div>
        </div>

        {last ? (
          <div className="market-last">
            <span className="pill" style={{ background: homeColor }}>{homeTri}</span>
            <span className="market-price">{last.homeP.toFixed(1)}%</span>
            <span style={{ color: "rgba(255,255,255,0.35)", fontWeight: 800 }}>—</span>
            <span className="market-price">{last.awayP.toFixed(1)}%</span>
            <span className="pill" style={{ background: awayColor }}>{awayTri}</span>
          </div>
        ) : (
          <div className="market-sub">Waiting for updates…</div>
        )}
      </div>

      <div className="market-chart">
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={chartData} margin={{ top: 8, right: 16, left: 6, bottom: 10 }}>
            <XAxis
              dataKey="x"
              type="number"
              domain={[0, 48 * 60]}
              tickFormatter={fmtX}
              tick={{ fontSize: 11 }}
              interval={7}
            />
            <YAxis
              domain={[0, 100]}
              tickFormatter={(v) => `${v}%`}
              tick={{ fontSize: 11 }}
              width={44}
            />
            <Tooltip
              formatter={(value, name) => [`${Number(value).toFixed(1)}%`, name]}
              labelFormatter={(label) => fmtX(label)}
              contentStyle={{
                borderRadius: 12,
                border: "1px solid var(--line)",
                background: "rgba(20,22,26,0.92)",
                color: "var(--text)",
              }}
            />
            <Legend
              verticalAlign="top"
              align="left"
              iconType="circle"
              wrapperStyle={{ paddingLeft: 8, paddingBottom: 6, fontSize: 12 }}
            />
            <ReferenceLine y={50} strokeDasharray="4 4" />

            <Line
              name={homeName}
              type="monotone"
              dataKey="homeP"
              dot={false}
              strokeWidth={2.6}
              stroke={homeColor}
              isAnimationActive={false}
            />
            <Line
              name={awayName}
              type="monotone"
              dataKey="awayP"
              dot={false}
              strokeWidth={2.2}
              stroke={awayColor}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>

        {last && (
          <div className="market-foot">
            <div className="mono">
              Q{last.period} • {last.secLeft}s • {last.homeScore}-{last.awayScore}
            </div>
            <div className="market-dot">
              <span className={`pulse-dot ${statusText === "final" ? "finalDot" : ""}`} />
              {statusText}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

/* Box Score */
function formatMins(raw) {
  if (!raw) return "—";
  const m = raw.match(/PT(\d+)M(\d+(?:\.\d+)?)S/i);
  if (m) return `${m[1]}:${String(Math.floor(parseFloat(m[2]))).padStart(2, "0")}`;
  return raw;
}

function BoxScoreTable({ team, accent }) {
  if (!team || !team.players) return null;
  const starters = team.players.filter((p) => p.starter && p.played);
  const bench = team.players.filter((p) => !p.starter && p.played);
  const dnp = team.players.filter((p) => !p.played);

  const renderRow = (p, i) => (
    <div className={`box-tr ${i % 2 ? "box-odd" : ""}`} key={p.nameI || i}>
      <div className="box-player">
        <span className="box-jersey">#{p.jerseyNum}</span>
        <span className="box-name">{p.nameI || p.name}</span>
        {p.position ? <span className="box-pos">{p.position}</span> : null}
      </div>
      <div className="box-num">{formatMins(p.minutes)}</div>
      <div className="box-num box-pts">{p.points}</div>
      <div className="box-num">{p.rebounds}</div>
      <div className="box-num">{p.assists}</div>
      <div className="box-num">{p.steals}</div>
      <div className="box-num">{p.blocks}</div>
      <div className="box-num">{p.turnovers}</div>
      <div className="box-num">{p.fgm}-{p.fga}</div>
      <div className="box-num">{p.tpm}-{p.tpa}</div>
      <div className="box-num">{p.ftm}-{p.fta}</div>
      <div className="box-num">{p.plusMinus > 0 ? "+" : ""}{p.plusMinus}</div>
    </div>
  );

  return (
    <div className="box-table-wrap">
      <div className="box-table">
        <div className="box-thead">
          <div className="box-player">Player</div>
          <div className="box-num">MIN</div>
          <div className="box-num">PTS</div>
          <div className="box-num">REB</div>
          <div className="box-num">AST</div>
          <div className="box-num">STL</div>
          <div className="box-num">BLK</div>
          <div className="box-num">TO</div>
          <div className="box-num">FG</div>
          <div className="box-num">3PT</div>
          <div className="box-num">FT</div>
          <div className="box-num">+/−</div>
        </div>

        {starters.length > 0 && (
          <>
            <div className="box-divider">Starters</div>
            {starters.map(renderRow)}
          </>
        )}
        {bench.length > 0 && (
          <>
            <div className="box-divider">Bench</div>
            {bench.map(renderRow)}
          </>
        )}

        {/* Totals */}
        {team.totals && (
          <div className="box-tr box-totals" style={{ borderTop: `2px solid ${accent}` }}>
            <div className="box-player"><strong>TOTALS</strong></div>
            <div className="box-num"></div>
            <div className="box-num box-pts">{team.totals.points}</div>
            <div className="box-num">{team.totals.rebounds}</div>
            <div className="box-num">{team.totals.assists}</div>
            <div className="box-num">{team.totals.steals}</div>
            <div className="box-num">{team.totals.blocks}</div>
            <div className="box-num">{team.totals.turnovers}</div>
            <div className="box-num">{team.totals.fgm}-{team.totals.fga}</div>
            <div className="box-num">{team.totals.tpm}-{team.totals.tpa}</div>
            <div className="box-num">{team.totals.ftm}-{team.totals.fta}</div>
            <div className="box-num"></div>
          </div>
        )}

        {dnp.length > 0 && (
          <div className="box-dnp">
            <span className="box-divider-label">DNP: </span>
            {dnp.map((p) => p.nameI || p.name).join(", ")}
          </div>
        )}
      </div>
    </div>
  );
}

function BoxScore({ data, loading, error, homeTheme, awayTheme }) {
  const [activeTeam, setActiveTeam] = useState("home");

  if (!data && !loading) return null;

  const homeTri = data?.home?.tri || "HOME";
  const awayTri = data?.away?.tri || "AWAY";
  const team = activeTeam === "home" ? data?.home : data?.away;
  const accent = activeTeam === "home" ? homeTheme.a : awayTheme.a;

  return (
    <div className="pred-section">
      <div className="box-header">
        <div className="pred-section-title">Box Score {data?.gameStatus === 2 && <span className="pred-section-tag" style={{ background: "rgba(255,59,59,0.18)", borderColor: "rgba(255,59,59,0.30)" }}>LIVE</span>}</div>
        <div className="box-team-tabs">
          <button
            className={`box-team-tab ${activeTeam === "home" ? "active" : ""}`}
            style={activeTeam === "home" ? { background: homeTheme.a, borderColor: homeTheme.a } : {}}
            onClick={() => setActiveTeam("home")}
          >
            {homeTri}
          </button>
          <button
            className={`box-team-tab ${activeTeam === "away" ? "active" : ""}`}
            style={activeTeam === "away" ? { background: awayTheme.a, borderColor: awayTheme.a } : {}}
            onClick={() => setActiveTeam("away")}
          >
            {awayTri}
          </button>
        </div>
      </div>
      {loading && !data && <div className="status">Loading box score…</div>}
      {error && !data && <div className="status error">Error: {error}</div>}
      {team && <BoxScoreTable team={team} accent={accent} />}
    </div>
  );
}

function PredictionPage({
  dateStr,
  game,
  prediction,
  predLoading,
  predError,
  livePred,
  liveLoading,
  liveError,
  liveSeries,
  boxScore,
  boxLoading,
  boxError,
  authToken,
  authUser,
  onBack,
}) {
  const hs = game?.homeScore ?? null;
  const as = game?.awayScore ?? null;

  const showScores = game?.gameStatusId === 2 || game?.gameStatusId === 3 || isEndedLike(game);
  const homeLeads = hs != null && as != null ? hs > as : false;
  const awayLeads = hs != null && as != null ? as > hs : false;

  const home = game?.home || {};
  const away = game?.away || {};

  const homeTri = (home.tri || home.name || "HOME").toUpperCase();
  const awayTri = (away.tri || away.name || "AWAY").toUpperCase();

  const homeName = home.name || homeTri;
  const awayName = away.name || awayTri;

  const homeLogo = home.id ? `https://cdn.nba.com/logos/nba/${home.id}/global/L/logo.svg` : null;
  const awayLogo = away.id ? `https://cdn.nba.com/logos/nba/${away.id}/global/L/logo.svg` : null;

  const homeTheme = teamTheme(homeTri);
  const awayTheme = teamTheme(awayTri);

  /* Pregame SVM prediction */
  const pPregame = prediction?.p_home_win != null ? Number(prediction.p_home_win) : null;
  const homePctPregame = pPregame == null ? null : pPregame * 100;
  const awayPctPregame = pPregame == null ? null : (1 - pPregame) * 100;

  const pickPregame =
    pPregame == null
      ? "—"
      : pPregame >= 0.5
      ? prediction.home_tri || homeTri
      : prediction.away_tri || awayTri;
  const pickColor =
    pPregame == null
      ? "var(--muted)"
      : pPregame >= 0.5
      ? homeTheme.a
      : awayTheme.a;

  /* Live XGB prediction */
  const pLive = livePred?.ok ? Number(livePred.homeWinProb) : null;
  const homePctLive = pLive == null ? null : pLive * 100;
  const awayPctLive = pLive == null ? null : (1 - pLive) * 100;

  const ended = game?.gameStatusId === 3 || isEndedLike(game);
  const liveEnabled = game?.gameStatusId === 2 && !ended;

  const showChart = Array.isArray(liveSeries) && liveSeries.length > 0;

  /* Quick pick from prediction page */
  const [qpPick, setQpPick] = useState(null);
  const [qpDone, setQpDone] = useState(false);

  useEffect(() => {
    if (!authToken || !game?.gameId) return;
    authFetch(`/api/picks/my?date=${dateStr}`, authToken)
      .then(r => r.json()).then(j => {
        if (j.ok) { const p = j.picks.find(p => p.gameId === game.gameId); if (p) { setQpPick(p.pickedTeam); setQpDone(true); } }
      }).catch(() => {});
  }, [authToken, game?.gameId]);

  async function quickPick(tri) {
    if (!authToken || !game?.gameId || ended || game?.gameStatusId >= 2) return;
    const body = { gameId: game.gameId, gameDate: dateStr, pickedTeam: tri, homeTri, awayTri, gameStatus: game?.gameStatusId || 1 };
    try {
      const res = await authFetch("/api/picks/make", authToken, { method: "POST", body: JSON.stringify(body) });
      const j = await res.json();
      if (j.ok) { setQpPick(tri); setQpDone(true); }
    } catch {}
  }

  const qpLocked = ended || (game?.gameStatusId >= 2) || qpDone;
  const aiPick = pPregame == null ? null : pPregame >= 0.5 ? homeTri : awayTri;
  const fadePick = aiPick === homeTri ? awayTri : awayTri === aiPick ? homeTri : null;

  return (
    <div
      className="predWrap"
      style={{
        "--teamA": homeTheme.a,
        "--teamB": awayTheme.a,
        "--teamA2": homeTheme.b,
        "--teamB2": awayTheme.b,
      }}
    >
      <div className="pred-topbar">
        <button className="btn" onClick={onBack}>
          ← Back
        </button>

        <div className="pred-topbar-mid">
          <div className="pred-date">{fmtLongLocal(dateStr)}</div>
          <div className="pred-status">
            <StatusBadge g={game || {}} />
          </div>
        </div>

        <div className="pred-spacer" />
      </div>

      <div className="predShell predShell--full">
        <div className="predMain">
          {/* ---- Team header ---- */}
          <div className="predHeader">
            <div className="predTeamBlock">
              {homeLogo ? <img className="pred-logo" src={homeLogo} alt={homeTri} /> : <div className="team-bubble">{homeTri.slice(0, 3)}</div>}
              <div className="pred-team-name">{homeName}</div>
              <div className="pred-team-tri">{homeTri}</div>
            </div>

            <div className="predVS">vs</div>

            <div className="predTeamBlock">
              {awayLogo ? <img className="pred-logo" src={awayLogo} alt={awayTri} /> : <div className="team-bubble">{awayTri.slice(0, 3)}</div>}
              <div className="pred-team-name">{awayName}</div>
              <div className="pred-team-tri">{awayTri}</div>
            </div>
          </div>

          {/* ---- Scoreboard ---- */}
          {showScores && (
            <div className="pred-scoreboard">
              <TeamRow team={home} score={hs} emphasize={homeLeads && (game.gameStatusId === 2 || game.gameStatusId === 3)} accent={homeTheme.a} />
              <TeamRow team={away} score={as} emphasize={awayLeads && (game.gameStatusId === 2 || game.gameStatusId === 3)} accent={awayTheme.a} />
            </div>
          )}

          {/* ---- Pregame Prediction (SVM) ---- */}
          <div className="pred-section">
            <div className="pred-section-title">Pregame Prediction <span className="pred-section-tag">SVM</span></div>
            {predLoading && <div className="status">Loading prediction…</div>}
            {predError && <div className="status error">Error: {predError}</div>}
            {!predLoading && !predError && (
              prediction ? (
                <div className="pred-single-row">
                  <Donut
                    value={pPregame >= 0.5 ? homePctPregame : awayPctPregame}
                    label={pickPregame}
                    stroke={pickColor}
                  />
                  <div className="pred-single-info">
                    <div className="pred-pick">
                      Pick: <span className="pred-pick-strong" style={{ color: pickColor }}>{pickPregame}</span>
                    </div>
                    <div className="pred-prob-bar">
                      <div className="pred-prob-team">
                        <span className="pill" style={{ background: homeTheme.a }}>{homeTri}</span>
                        <span className="pred-prob-pct">{homePctPregame.toFixed(1)}%</span>
                      </div>
                      <div className="pred-prob-team">
                        <span className="pill" style={{ background: awayTheme.a }}>{awayTri}</span>
                        <span className="pred-prob-pct">{awayPctPregame.toFixed(1)}%</span>
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="status">No pregame prediction available.</div>
              )
            )}
          </div>

          {/* ── AI Explanation Panel ── */}
          <ExplainPanel dateStr={dateStr} game={game} prediction={prediction} />

          {/* ---- Make Your Pick ---- */}
          {authToken && pPregame != null && (
            <div className="pred-section">
              <div className="pred-section-title">
                Make Your Pick
                <span className="pred-section-tag" style={{ background: qpLocked ? "rgba(255,255,255,0.08)" : "rgba(34,197,94,0.18)", borderColor: qpLocked ? "rgba(255,255,255,0.15)" : "rgba(34,197,94,0.30)" }}>
                  {qpLocked && qpPick ? "Locked" : qpLocked ? "Locked" : "Fantasy"}
                </span>
              </div>
              <div className="qp-wrap">
                <div className="qp-ai-row">
                  <span className="qp-ai-label">AI picks</span>
                  <span className="qp-ai-team" style={{ color: aiPick === homeTri ? homeTheme.a : awayTheme.a }}>{aiPick}</span>
                  <span className="qp-ai-conf">({(Math.max(homePctPregame, awayPctPregame)).toFixed(1)}%)</span>
                </div>
                <div className="qp-buttons">
                  {[
                    { tri: homeTri, label: homeName, theme: homeTheme, logo: homeLogo },
                    { tri: awayTri, label: awayName, theme: awayTheme, logo: awayLogo },
                  ].map(({ tri, label, theme, logo }) => (
                    <button
                      key={tri}
                      className={`qp-btn ${qpPick === tri ? "qp-picked" : ""} ${tri === aiPick ? "qp-ai" : "qp-fade"}`}
                      style={{ "--qc": theme.a }}
                      onClick={() => quickPick(tri)}
                      disabled={qpLocked}
                    >
                      {logo && <img className="qp-logo" src={logo} alt={tri} />}
                      <span className="qp-btn-name">{label}</span>
                      {tri === aiPick && !qpPick && <span className="qp-tag agree">Agree with AI</span>}
                      {tri !== aiPick && !qpPick && <span className="qp-tag fade">Fade the AI</span>}
                      {qpPick === tri && <span className="qp-tag picked">Your Pick ✓</span>}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}
          {/* ---- Win Market + Live ---- */}
          <div className="pred-section">
            <div>
              {showChart ? (
                <LiveProbChart
                  data={liveSeries}
                  homeName={homeName}
                  awayName={awayName}
                  homeTri={homeTri}
                  awayTri={awayTri}
                  statusText={ended ? "final" : "live"}
                  homeColor={homeTheme.a}
                  awayColor={awayTheme.a}
                />
              ) : liveEnabled ? (
                <div className="market-card">
                  <div className="market-title">Win “Market”</div>
                  <div className="status">Waiting for play-by-play events…</div>
                </div>
              ) : (
                <div className="market-card">
                  <div className="market-title">Win “Market”</div>
                  <div className="status">
                    {ended ? "Game ended — final chart shown above if available." : "No live data available for this game."}
                  </div>
                </div>
              )}

              {/* Live snapshot (XGB) beneath chart when live */}
              {liveEnabled && (
                <div className="live-snapshot">
                  {liveLoading && <span className="status">Updating…</span>}
                  {liveError && <span className="status error">Error: {liveError}</span>}
                  {livePred?.ok && (
                    <div className="live-snapshot-row">
                      <span className="pill" style={{ background: homeTheme.a }}>{homeTri}</span>
                      <span className="live-snapshot-pct">{homePctLive.toFixed(1)}%</span>
                      <span className="live-snapshot-sep">—</span>
                      <span className="live-snapshot-pct">{awayPctLive.toFixed(1)}%</span>
                      <span className="pill" style={{ background: awayTheme.a }}>{awayTri}</span>
                      <span className="live-snapshot-meta mono">Q{livePred.period} • {livePred.secLeftPeriod}s</span>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* ---- Box Score (live + final only) ---- */}
          {(game?.gameStatusId === 2 || game?.gameStatusId === 3 || isEndedLike(game)) && (
            <BoxScore
              data={boxScore}
              loading={boxLoading}
              error={boxError}
              homeTheme={homeTheme}
              awayTheme={awayTheme}
            />
          )}

          {/* ---- Details ---- */}
          <div className="pred-details">
            <div className="pred-detail-card">
              <div className="pred-k">Arena</div>
              <div className="pred-v">{game?.arena || "—"}</div>
            </div>
            <div className="pred-detail-card">
              <div className="pred-k">Tipoff (ET)</div>
              <div className="pred-v">{game?.tipoffET || game?.status || "—"}</div>
            </div>
            <div className="pred-detail-card">
              <div className="pred-k">Game ID</div>
              <div className="pred-v mono">{game?.gameId || "—"}</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
/* Schedule View */
function ScheduleView({ authToken, authUser }) {
  const [games, setGames] = useState([]);
  const [dateStr, setDateStr] = useState(() => ymdNow());
  const [seasonStart, setSeasonStart] = useState(null);
  const [seasonEnd, setSeasonEnd] = useState(null);
  const abortRef = useRef(null);

  const [isInitialLoading, setIsInitialLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);

  const timerRef = useRef(null);
  const hasDataRef = useRef(false);

  const [selectedGame, setSelectedGame] = useState(null);

  // Pregame prediction (SVM)
  const [prediction, setPrediction] = useState(null);
  const [predLoading, setPredLoading] = useState(false);
  const [predError, setPredError] = useState(null);

  // Live prediction (XGB PBP)
  const [livePred, setLivePred] = useState(null);
  const [liveLoading, setLiveLoading] = useState(false);
  const [liveError, setLiveError] = useState(null);

  // Chart series (history + live appends)
  const [liveSeries, setLiveSeries] = useState([]);
  const liveTimerRef = useRef(null);

  // Box score
  const [boxScore, setBoxScore] = useState(null);
  const [boxLoading, setBoxLoading] = useState(false);
  const [boxError, setBoxError] = useState(null);
  const boxTimerRef = useRef(null);

  async function loadSeasonWindow() {
    const res = await fetch(`${API_BASE}/api/season/window`);
    const json = await res.json();
    setSeasonStart(json.start);
    setSeasonEnd(json.end);
  }

  function shouldPollLive(g) {
    if (!g) return false;
    return g.gameStatusId === 2 && !isEndedLike(g);
  }

  async function fetchLiveHistory(gameObj) {
    if (!gameObj?.gameId) return;
    try {
      const url = new URL(`${API_BASE}/api/predictions/live/history`);
      url.searchParams.set("gameId", gameObj.gameId);

      const res = await fetch(url.toString(), { cache: "no-store" });
      const json = await res.json();
      if (!res.ok) throw new Error(json?.error || "History request failed");

      if (json?.ok && Array.isArray(json.points)) {
        const mapped = json.points.map((pt) => ({
          key: `${pt.period}-${pt.secLeftPeriod}-${pt.homeScore}-${pt.awayScore}`,
          period: Number(pt.period),
          secLeft: Number(pt.secLeftPeriod),
          homeScore: Number(pt.homeScore),
          awayScore: Number(pt.awayScore),
          p: Number(pt.homeWinProb) * 100,
        }));
        setLiveSeries(mapped);
      }
    } catch {
      // keep quiet; live polling will still work
    }
  }

  async function fetchLivePrediction(gameObj) {
    if (!gameObj?.gameId) return;

    setLiveLoading(true);
    setLiveError(null);

    try {
      const url = new URL(`${API_BASE}/api/predictions/live`);
      url.searchParams.set("gameId", gameObj.gameId);

      const res = await fetch(url.toString(), { cache: "no-store" });
      const json = await res.json();

      if (!res.ok) throw new Error(json?.error || "Live request failed");

      setLivePred(json);

      if (json?.ok === false && json?.error) {
        setLiveError(json.error);
        return;
      }

      if (json?.ok) {
        const point = {
          key: `${json.period}-${json.secLeftPeriod}-${json.homeScore}-${json.awayScore}`,
          period: Number(json.period),
          secLeft: Number(json.secLeftPeriod),
          homeScore: Number(json.homeScore),
          awayScore: Number(json.awayScore),
          p: Number(json.homeWinProb) * 100,
        };

        setLiveSeries((prev) => {
          if (prev.length && prev[prev.length - 1].key === point.key) return prev;
          const next = [...prev, point];
          return next.length > 900 ? next.slice(next.length - 900) : next;
        });
      }
    } catch (e) {
      setLiveError(e.message || String(e));
    } finally {
      setLiveLoading(false);
    }
  }

  function startLivePolling(gameObj) {
    clearInterval(liveTimerRef.current);

    fetchLiveHistory(gameObj).finally(() => {
      if (shouldPollLive(gameObj)) {
        fetchLivePrediction(gameObj);
        liveTimerRef.current = setInterval(() => fetchLivePrediction(gameObj), 5000);
      }
    });
  }

  function stopLivePolling() {
    clearInterval(liveTimerRef.current);
    liveTimerRef.current = null;
  }

  async function fetchBoxScore(gameObj, { silent = false } = {}) {
    if (!gameObj?.gameId) return;
    if (!silent) setBoxLoading(true);
    if (!silent) setBoxError(null);
    try {
      const res = await fetch(`/api/boxscore/${gameObj.gameId}`, { cache: "no-store" });
      const json = await res.json();
      if (json?.ok) {
        setBoxScore(json);
      } else {
        if (!silent) setBoxError(json?.error || "Box score not available");
      }
    } catch (e) {
      if (!silent) setBoxError(e.message || String(e));
    } finally {
      if (!silent) setBoxLoading(false);
    }
  }

  function startBoxPolling(gameObj) {
    clearInterval(boxTimerRef.current);
    if (gameObj?.gameStatusId === 2 && !isEndedLike(gameObj)) {
      boxTimerRef.current = setInterval(() => fetchBoxScore(gameObj, { silent: true }), 10000);
    }
  }

  function stopBoxPolling() {
    clearInterval(boxTimerRef.current);
    boxTimerRef.current = null;
  }

  async function openPrediction(g) {
    setSelectedGame(g);

    setPrediction(null);
    setPredError(null);
    setPredLoading(true);

    setLivePred(null);
    setLiveError(null);
    setLiveSeries([]);

    setBoxScore(null);
    setBoxError(null);

    startLivePolling(g);

    // Fetch box score for live or finished games
    if (g.gameStatusId === 2 || g.gameStatusId === 3 || isEndedLike(g)) {
      fetchBoxScore(g);
      startBoxPolling(g);
    }

    try {
      const url = new URL(`${API_BASE}/api/predictions/game`);
      url.searchParams.set("date", dateStr);
      url.searchParams.set("gameId", g.gameId);

      const res = await fetch(url.toString(), { cache: "no-store" });
      const json = await res.json();

      if (!res.ok) throw new Error(json?.error || "Request failed");
      setPrediction(json.prediction);
    } catch (e) {
      setPredError(e.message || String(e));
    } finally {
      setPredLoading(false);
    }
  }

  async function fetchDay(d, { silent = false } = {}) {
    if (abortRef.current) abortRef.current.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      setError(null);
      if (!silent) setIsInitialLoading(true);
      if (silent) setIsRefreshing(true);

      const res = await fetch(`${API_BASE}/api/schedule/day?date=${d}`, {
        cache: "no-store",
        signal: controller.signal,
      });

      const json = await res.json();
      if (!res.ok && json.error) throw new Error(json.error);

      const list = Array.isArray(json.games) ? json.games : [];

      // Deduplicate by gameId (keep latest occurrence)
      const deduped = Array.from(new Map(list.map((g) => [g.gameId, g])).values());

      setGames(deduped);
      hasDataRef.current = hasDataRef.current || deduped.length > 0;
      setLastUpdated(new Date());
    } catch (e) {
      if (e.name !== "AbortError") {
        if (!silent) setError(e.message || String(e));
      }
    } finally {
      if (!silent) setIsInitialLoading(false);
      if (silent) setIsRefreshing(false);
    }
  }

  useEffect(() => {
    loadSeasonWindow();
    fetchDay(dateStr, { silent: false });
  }, []);

  useEffect(() => {
    fetchDay(dateStr, { silent: false });
  }, [dateStr]);

  const refreshMs = useMemo(() => (isTodayStr(dateStr) ? 5000 : 0), [dateStr]);
  useEffect(() => {
    clearInterval(timerRef.current);
    if (refreshMs > 0) {
      timerRef.current = setInterval(() => fetchDay(dateStr, { silent: true }), refreshMs);
    }
    return () => clearInterval(timerRef.current);
  }, [refreshMs, dateStr]);

  // Cleanup live polling on unmount
  useEffect(() => {
    return () => {
      stopLivePolling();
      stopBoxPolling();
    };
  }, []);

  // Keep selectedGame in sync with latest polled data
  useEffect(() => {
    if (!selectedGame) return;
    const fresh = games.find((g) => g.gameId === selectedGame.gameId);
    if (!fresh) return;
    // Only update if something meaningful changed
    if (
      fresh.homeScore !== selectedGame.homeScore ||
      fresh.awayScore !== selectedGame.awayScore ||
      fresh.gameStatusId !== selectedGame.gameStatusId ||
      fresh.status !== selectedGame.status ||
      fresh.livePeriod !== selectedGame.livePeriod ||
      fresh.liveClock !== selectedGame.liveClock
    ) {
      setSelectedGame(fresh);
    }
  }, [games]);

  const stepDay = (n) => {
    const d = parseYMDLocal(dateStr);
    d.setDate(d.getDate() + n);
    const next = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(
      d.getDate()
    ).padStart(2, "0")}`;
    setDateStr(next);
  };

  const prettyDate = fmtLongLocal(dateStr);

  const sortedGames = useMemo(() => {
    const arr = [...games];
    arr.sort((a, b) => {
      const am = tipoffMinutesFromETString(a.tipoffET);
      const bm = tipoffMinutesFromETString(b.tipoffET);
      return am - bm;
    });
    return arr;
  }, [games]);

  const liveCount = useMemo(() => sortedGames.filter((g) => g.gameStatusId === 2 && !isEndedLike(g)).length, [sortedGames]);
  const finalCount = useMemo(() => sortedGames.filter((g) => g.gameStatusId === 3 || isEndedLike(g)).length, [sortedGames]);

  if (selectedGame) {
    return (
      <PredictionPage
        dateStr={dateStr}
        game={selectedGame}
        prediction={prediction}
        predLoading={predLoading}
        predError={predError}
        livePred={livePred}
        liveLoading={liveLoading}
        liveError={liveError}
        liveSeries={liveSeries}
        boxScore={boxScore}
        boxLoading={boxLoading}
        boxError={boxError}
        authToken={authToken}
        authUser={authUser}
        onBack={() => {
          stopLivePolling();
          stopBoxPolling();
          setSelectedGame(null);
        }}
      />
    );
  }

  return (
    <>
      <div className="hero">
        <div className="heroLeft">
          <div className="heroKicker">Schedule</div>
          <div className="heroTitle">{prettyDate}</div>
          <div className="heroMeta">
            <span className="chipPill">Live: {liveCount}</span>
            <span className="chipPill">Final: {finalCount}</span>
            <span className="chipPill">Games: {sortedGames.length}</span>
          </div>
        </div>
        <div className="heroRight">
          <DateNav
            value={dateStr}
            min={seasonStart || "2025-10-01"}
            max={seasonEnd || "2026-06-30"}
            onChange={setDateStr}
            onToday={() => {
              const t = ymdNow();
              setDateStr(t);
              fetchDay(t, { silent: false });
            }}
            onPrev={() => stepDay(-1)}
            onNext={() => stepDay(1)}
            isToday={isTodayStr(dateStr)}
          />
          <div className="updatedLine">
            {lastUpdated ? (
              <>
                Updated: {lastUpdated.toLocaleTimeString()}{" "}
                {isRefreshing && <span className="dotpulse" />}
              </>
            ) : (
              ""
            )}
          </div>
        </div>
      </div>

      {isInitialLoading && !hasDataRef.current && <div className="status">Loading…</div>}
      {error && <div className="status error">Error: {error}</div>}
      {!isInitialLoading && !error && sortedGames.length === 0 && (
        <div className="status">No games for this date.</div>
      )}

      <main className={`grid ${isRefreshing ? "list-refreshing" : ""}`}>
        {sortedGames.map((g) => (
          <GameCard key={g.gameId} g={g} onClick={openPrediction} />
        ))}
      </main>
    </>
  );
}

/* Standings View */
function TeamCell({ teamId, tri, name }) {
  const src = teamId ? `https://cdn.nba.com/logos/nba/${teamId}/global/L/logo.svg` : null;
  return (
    <div className="team-cell">
      {src ? (
        <img className="team-logo" src={src} alt={tri || name} loading="lazy" />
      ) : (
        <div className="team-bubble small">{(tri || name || "—").slice(0, 3)}</div>
      )}
      <div className="team-name">{name}</div>
      <div className="team-tri chip">{tri}</div>
    </div>
  );
}
function StandingsRow({ r, index }) {
  const playoff = r.confRank && r.confRank <= 6;
  const playin = r.confRank && r.confRank >= 7 && r.confRank <= 10;
  return (
    <div className={`tr ${index % 2 ? "odd" : ""}`}>
      <div className={`rank ${playoff ? "rank-playoff" : playin ? "rank-playin" : ""}`}>
        {r.confRank || index + 1}
      </div>
      <TeamCell teamId={r.teamId} tri={r.tri} name={r.team} />
      <div className="num">{r.wins}</div>
      <div className="num">{r.losses}</div>
      <div className="num">
        {(r.pct ?? 0).toLocaleString(undefined, {
          style: "percent",
          minimumFractionDigits: 3,
          maximumFractionDigits: 3,
        })}
      </div>
      <div className="num">{r.gb ?? "0.0"}</div>
      <div className="mono">{r.streak || "—"}</div>
      <div className="mono">{r.l10 || "—"}</div>
    </div>
  );
}
function StandingsTable({ rows }) {
  return (
    <div className="standings-table">
      <div className="thead">
        <div>#</div>
        <div>Team</div>
        <div>W</div>
        <div>L</div>
        <div>PCT</div>
        <div>GB</div>
        <div>Streak</div>
        <div>L10</div>
      </div>
      <div className="tbody">{rows.map((r, i) => <StandingsRow key={r.teamId ?? i} r={r} index={i} />)}</div>
    </div>
  );
}
function seasonLabelRange(count = 12) {
  const now = new Date();
  const y = now.getFullYear();
  const m = now.getMonth() + 1;
  let startYear = m >= 8 ? y : y - 1;
  const out = [];
  for (let i = 0; i < count; i++) {
    const s = startYear - i;
    const e = (s + 1) % 100;
    out.push(`${s}-${String(e).padStart(2, "0")}`);
  }
  return out;
}
function StandingsView() {
  const [season, setSeason] = useState(seasonLabelRange(14)[0]);
  const [seasonType, setSeasonType] = useState("Regular Season");
  const [east, setEast] = useState([]);
  const [west, setWest] = useState([]);
  const [conf, setConf] = useState("East");
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState(null);
  const [lastLoaded, setLastLoaded] = useState(null);
  const seasons = seasonLabelRange(20);

  async function load() {
    try {
      setErr(null);
      setLoading(true);
      const url = new URL(`${API_BASE}/api/standings`);
      url.searchParams.set("season", season);
      url.searchParams.set("seasonType", seasonType);
      const res = await fetch(url.toString(), { cache: "no-store" });
      const json = await res.json();
      setEast(json.east || []);
      setWest(json.west || []);
      setLastLoaded(new Date());
    } catch (e) {
      setErr(e.message || String(e));
    } finally {
      setLoading(false);
    }
  }
  useEffect(() => {
    load();
  }, [season, seasonType]);

  const rows = conf === "East" ? east : west;

  return (
    <>
      <div className="toolbar">
        <div className="controls">
          <label className="lbl">Season</label>
          <select className="select" value={season} onChange={(e) => setSeason(e.target.value)}>
            {seasons.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
          <label className="lbl">Type</label>
          <select className="select" value={seasonType} onChange={(e) => setSeasonType(e.target.value)}>
            <option>Regular Season</option>
            <option>Playoffs</option>
          </select>
          <button className="btn" onClick={load}>
            Reload
          </button>
          <span className="hint">{lastLoaded ? `Loaded: ${lastLoaded.toLocaleTimeString()}` : ""}</span>
        </div>

        <div className="tabs conf-tabs">
          <button className={`tab ${conf === "East" ? "active" : ""}`} onClick={() => setConf("East")}>
            Eastern
          </button>
          <button className={`tab ${conf === "West" ? "active" : ""}`} onClick={() => setConf("West")}>
            Western
          </button>
        </div>
      </div>

      {loading && <div className="status">Loading standings…</div>}
      {err && <div className="status error">Error: {err}</div>}
      {!loading && !err && (
        <div className="standings-panel">
          <div className="standings-title">{conf === "East" ? "East Conference" : "West Conference"}</div>
          <StandingsTable rows={rows} />
          <div className="legend">
            <span className="pill playoff" /> Playoff (1–6)
            <span className="pill playin" /> Play-In (7–10)
          </div>
        </div>
      )}
    </>
  );
}

/* Teams View */
function TeamsView({ initialTeam }) {
  const [teams, setTeams] = useState([]);
  const [selectedTeam, setSelectedTeam] = useState(initialTeam || null);
  const [teamStats, setTeamStats] = useState(null);
  const [teamRoster, setTeamRoster] = useState(null);
  const [teamSchedule, setTeamSchedule] = useState(null);
  const [season, setSeason] = useState(seasonLabelRange(14)[0]);
  const [loading, setLoading] = useState(true);
  const [statsLoading, setStatsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState("stats");
  const seasons = seasonLabelRange(20);

  useEffect(() => { if (initialTeam) setSelectedTeam(initialTeam); }, [initialTeam]);

  useEffect(() => {
    async function loadTeams() {
      try {
        setLoading(true);
        const res = await fetch("/api/teams");
        const json = await res.json();
        setTeams(json);
      } catch (e) {
        setError(e.message || String(e));
      } finally {
        setLoading(false);
      }
    }
    loadTeams();
  }, []);

  useEffect(() => {
    if (!selectedTeam) return;
    async function loadTeamData() {
      try {
        setStatsLoading(true);
        setError(null);
        const seasonYear = parseInt(season.split("-")[0]);
        const [statsRes, rosterRes, scheduleRes] = await Promise.all([
          fetch(`/api/teams/${selectedTeam}/stats?season=${seasonYear}`),
          fetch(`/api/teams/${selectedTeam}/roster?season=${seasonYear}`),
          fetch(`/api/teams/${selectedTeam}/schedule?season=${seasonYear}`),
        ]);
        const statsJ = await statsRes.json();
        const rosterJ = await rosterRes.json();
        const scheduleJ = await scheduleRes.json();
        // Only set data if the response isn't an error
        setTeamStats(statsJ.error ? null : statsJ);
        setTeamRoster(rosterJ.error ? null : rosterJ);
        setTeamSchedule(scheduleJ.error ? null : scheduleJ);
        if (statsJ.error && rosterJ.error && scheduleJ.error) {
          setError("Team data unavailable — S3 credentials may not be configured.");
        }
      } catch (e) {
        setError(e.message || String(e));
      } finally {
        setStatsLoading(false);
      }
    }
    loadTeamData();
  }, [selectedTeam, season]);

  const selectedTeamInfo = teams.find((t) => t.abbreviation === selectedTeam);
  const theme = selectedTeam ? teamTheme(selectedTeam) : { a: "var(--accent)", b: "#22c55e" };

  /* ---- Team grid (no team selected) ---- */
  if (!selectedTeam) {
    return (
      <>
        <div className="hero" style={{ justifyContent: "center" }}>
          <div className="heroLeft" style={{ textAlign: "center" }}>
            <div className="heroKicker">Teams</div>
            <div className="heroTitle">NBA Teams</div>
            <div className="heroMeta" style={{ justifyContent: "center" }}>
              <span className="chipPill">{teams.length} Teams</span>
            </div>
          </div>
        </div>
        {loading && <div className="status">Loading teams…</div>}
        {error && <div className="status error">Error: {error}</div>}
        {!loading && (
          <div className="tv-grid">
            {teams.map((team) => {
              const tc = teamTheme(team.abbreviation);
              return (
                <div
                  key={team.teamId}
                  className="tv-card"
                  style={{ "--cardTeam": tc.a, "--cardTeam2": tc.b }}
                  onClick={() => { setSelectedTeam(team.abbreviation); setActiveTab("stats"); }}
                  role="button"
                  tabIndex={0}
                >
                  <img
                    className="tv-card-logo"
                    src={`https://cdn.nba.com/logos/nba/${team.teamId}/global/L/logo.svg`}
                    alt={team.abbreviation}
                    loading="lazy"
                    onError={(e) => (e.currentTarget.style.display = "none")}
                  />
                  <div className="tv-card-name">{team.fullName}</div>
                  <div className="tv-card-abbr" style={{ borderColor: tc.a, color: tc.a }}>{team.abbreviation}</div>
                </div>
              );
            })}
          </div>
        )}
      </>
    );
  }

  /* ---- Team detail ---- */
  return (
    <div
      className="tv-detail"
      style={{ "--teamA": theme.a, "--teamB": theme.b }}
    >
      {/* Header */}
      <div className="tv-header">
        <button className="btn" onClick={() => setSelectedTeam(null)}>← Back</button>
        <div className="tv-header-mid">
          <img
            className="tv-header-logo"
            src={`https://cdn.nba.com/logos/nba/${selectedTeamInfo?.teamId}/global/L/logo.svg`}
            alt={selectedTeam}
            loading="lazy"
            onError={(e) => (e.currentTarget.style.display = "none")}
          />
          <div>
            <div className="tv-header-name">{selectedTeamInfo?.fullName}</div>
            <div className="tv-header-season">{season} Season</div>
          </div>
        </div>
        <select className="select" value={season} onChange={(e) => setSeason(e.target.value)}>
          {seasons.map((s) => <option key={s} value={s}>{s}</option>)}
        </select>
      </div>

      {/* Sub-tabs */}
      <div className="tabs tv-tabs">
        {["stats", "roster", "schedule"].map((t) => (
          <button
            key={t}
            className={`tab ${activeTab === t ? "active" : ""}`}
            onClick={() => setActiveTab(t)}
          >
            {t === "stats" ? "Statistics" : t === "roster" ? "Roster" : "Schedule"}
          </button>
        ))}
      </div>

      {statsLoading && <div className="status">Loading team data…</div>}
      {error && <div className="status error">Error: {error}</div>}

      {/* ===== Statistics ===== */}
      {activeTab === "stats" && !statsLoading && !teamStats && (
        <div className="picks-empty">Statistics unavailable for this season — S3 credentials may not be configured.</div>
      )}
      {activeTab === "stats" && teamStats && !statsLoading && (
        <div className="tv-stats-grid">
          {/* Record */}
          <div className="tv-stat-card tv-stat-card--wide">
            <div className="tv-stat-card-title">Season Record</div>
            <div className="tv-stat-row">
              <div className="tv-stat-item tv-stat-big">
                <span className="tv-stat-val" style={{ color: theme.a }}>{teamStats.record.wins}-{teamStats.record.losses}</span>
                <span className="tv-stat-lbl">Record</span>
              </div>
              <div className="tv-stat-item">
                <span className="tv-stat-val">{(teamStats.record.winPct * 100).toFixed(1)}%</span>
                <span className="tv-stat-lbl">Win %</span>
              </div>
              <div className="tv-stat-item">
                <span className="tv-stat-val">{teamStats.gamesPlayed}</span>
                <span className="tv-stat-lbl">GP</span>
              </div>
            </div>
          </div>

          {/* Offensive */}
          <div className="tv-stat-card">
            <div className="tv-stat-card-title">Offense</div>
            <div className="tv-stat-row tv-stat-row--wrap">
              {[
                ["PPG", teamStats.offensive.ppg],
                ["FG%", teamStats.offensive.fgPct + "%"],
                ["3P%", teamStats.offensive.fg3Pct + "%"],
                ["FT%", teamStats.offensive.ftPct + "%"],
                ["APG", teamStats.offensive.apg],
                ["TOV", teamStats.offensive.topg],
              ].map(([l, v]) => (
                <div className="tv-stat-item" key={l}>
                  <span className="tv-stat-val">{v}</span>
                  <span className="tv-stat-lbl">{l}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Defensive */}
          <div className="tv-stat-card">
            <div className="tv-stat-card-title">Defense</div>
            <div className="tv-stat-row tv-stat-row--wrap">
              {[
                ["RPG", teamStats.defensive.rpg],
                ["ORPG", teamStats.defensive.orpg],
                ["DRPG", teamStats.defensive.drpg],
                ["SPG", teamStats.defensive.spg],
                ["BPG", teamStats.defensive.bpg],
                ["FPG", teamStats.defensive.fpg],
              ].map(([l, v]) => (
                <div className="tv-stat-item" key={l}>
                  <span className="tv-stat-val">{v}</span>
                  <span className="tv-stat-lbl">{l}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* ===== Roster ===== */}
      {activeTab === "roster" && !statsLoading && (
        teamRoster?.roster?.length > 0 ? (
        <div className="tv-table-wrap">
          <div className="tv-table">
            <div className="tv-thead">
              <div className="tv-cell tv-cell--player">Player</div>
              <div className="tv-cell">POS</div>
              <div className="tv-cell">GP</div>
              <div className="tv-cell">PPG</div>
              <div className="tv-cell">RPG</div>
              <div className="tv-cell">APG</div>
              <div className="tv-cell">FG%</div>
              <div className="tv-cell">3P%</div>
              <div className="tv-cell">FT%</div>
            </div>
            {teamRoster.roster.map((p, i) => (
              <div key={i} className={`tv-tr ${i % 2 ? "tv-tr--odd" : ""}`}>
                <div className="tv-cell tv-cell--player">{p.name}</div>
                <div className="tv-cell tv-cell--pos">{p.position}</div>
                <div className="tv-cell">{p.gamesPlayed}</div>
                <div className="tv-cell tv-cell--hl">{p.stats.ppg}</div>
                <div className="tv-cell">{p.stats.rpg}</div>
                <div className="tv-cell">{p.stats.apg}</div>
                <div className="tv-cell">{p.stats.fgPct}%</div>
                <div className="tv-cell">{p.stats.fg3Pct}%</div>
                <div className="tv-cell">{p.stats.ftPct}%</div>
              </div>
            ))}
          </div>
        </div>
        ) : <div className="picks-empty">Roster data unavailable for this season.</div>
      )}

      {/* ===== Schedule ===== */}
      {activeTab === "schedule" && !statsLoading && (
        teamSchedule?.games?.length > 0 ? (
        <div className="tv-sched">
          {teamSchedule.games.map((g, i) => {
            const oppTheme = teamTheme(g.opponent.trim());
            return (
              <div key={i} className="tv-sched-row">
                <div className="tv-sched-date">
                  {new Date(g.date).toLocaleDateString("en-US", { month: "short", day: "numeric" })}
                </div>
                <div className="tv-sched-loc">{g.location === "Home" ? "vs" : "@"}</div>
                <div className="tv-sched-opp" style={{ color: oppTheme.a }}>{g.opponent.replace(/_/g, " ")}</div>
                <div className={`tv-sched-wl ${g.result === "W" ? "tv-sched-w" : "tv-sched-l"}`}>{g.result}</div>
                <div className="tv-sched-score">{g.points}-{g.opponentPoints}</div>
              </div>
            );
          })}
        </div>
        ) : <div className="picks-empty">Schedule data unavailable for this season.</div>
      )}
    </div>
  );
}

/* =========================
   User Profile View
========================= */
function UserView({ token, user, onLogout, onGoToTeam }) {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [history, setHistory] = useState([]);
  const [histLoading, setHistLoading] = useState(false);
  const [favTeam, setFavTeam] = useState("");
  const [favSaving, setFavSaving] = useState(false);
  const [favMsg, setFavMsg] = useState(null);
  const [activeTab, setActiveTab] = useState("stats");

  useEffect(() => {
    if (!token) return;
    setLoading(true);
    authFetch("/api/picks/auth/stats", token)
      .then(r => r.json())
      .then(j => { if (j.ok) { setStats(j); setFavTeam(j.favoriteTeam || ""); } })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [token]);

  useEffect(() => {
    if (!token || activeTab !== "history") return;
    setHistLoading(true);
    authFetch("/api/picks/history?limit=50", token)
      .then(r => r.json())
      .then(j => { if (j.ok) setHistory(j.picks || []); })
      .catch(() => {})
      .finally(() => setHistLoading(false));
  }, [token, activeTab]);

  async function saveFavTeam(tri) {
    setFavSaving(true);
    setFavMsg(null);
    try {
      const res = await authFetch("/api/picks/auth/profile", token, {
        method: "PUT", body: JSON.stringify({ favoriteTeam: tri }),
      });
      const j = await res.json();
      if (j.ok) { setFavTeam(tri); setFavMsg("Saved!"); setTimeout(() => setFavMsg(null), 2000); }
    } catch {}
    setFavSaving(false);
  }

  if (!token || !user) return <div className="picks-empty">You're not logged in.</div>;
  const pct = stats && (stats.correct + stats.wrong > 0) ? ((stats.correct / (stats.correct + stats.wrong)) * 100).toFixed(1) : null;
  const favTheme = favTeam ? teamTheme(favTeam) : null;
  const TEAM_TRIS = Object.keys(TEAM_COLORS).sort();

  return (
    <>
      <div className="hero">
        <div className="heroLeft">
          <div className="heroKicker">Profile</div>
          <div className="heroTitle" style={{ display: "flex", alignItems: "center", gap: 10 }}>
            {favTheme && favTeam && (
              <img
                src={`https://cdn.nba.com/logos/nba/${NBA_TEAM_IDS[favTeam] || 0}/global/L/logo.svg`}
                alt={favTeam}
                style={{ width: 36, height: 36, objectFit: "contain" }}
                onError={e => e.currentTarget.style.display = "none"}
              />
            )}
            {user.username}
          </div>
          {user.email && <div className="heroMeta"><div className="chipPill">{user.email}</div></div>}
        </div>
        <div className="heroRight" style={{ justifyContent: "center" }}>
          <button className="btn user-logout-btn" onClick={onLogout}>Log Out</button>
        </div>
      </div>

      <div className="toolbar">
        <div className="controls" />
        <div className="tabs conf-tabs">
          <button className={`tab ${activeTab === "stats" ? "active" : ""}`} onClick={() => setActiveTab("stats")}>Stats</button>
          <button className={`tab ${activeTab === "team" ? "active" : ""}`} onClick={() => setActiveTab("team")}>My Team</button>
          <button className={`tab ${activeTab === "history" ? "active" : ""}`} onClick={() => setActiveTab("history")}>History</button>
        </div>
      </div>

      {loading && <div className="status">Loading stats…</div>}

      {activeTab === "stats" && stats && (
        <>
          <div className="user-stats-grid">
            <div className="user-stat-card user-stat-points">
              <div className="user-stat-val" style={{ color: "#a855f7" }}>{stats.points || 0}</div>
              <div className="user-stat-label">Total Points</div>
            </div>
            <div className="user-stat-card user-stat-accent">
              <div className="user-stat-val" style={{ color: "#22c55e" }}>{stats.correct}</div>
              <div className="user-stat-label">Correct</div>
            </div>
            <div className="user-stat-card">
              <div className="user-stat-val" style={{ color: "#ef4444" }}>{stats.wrong}</div>
              <div className="user-stat-label">Wrong</div>
            </div>
            <div className="user-stat-card">
              <div className="user-stat-val">{pct ? `${pct}%` : "—"}</div>
              <div className="user-stat-label">Win Rate</div>
            </div>
            <div className="user-stat-card">
              <div className="user-stat-val" style={{ color: "#f59e0b" }}>{stats.streak}</div>
              <div className="user-stat-label">Current Streak</div>
            </div>
            <div className="user-stat-card">
              <div className="user-stat-val">{stats.bestStreak}</div>
              <div className="user-stat-label">Best Streak</div>
            </div>
          </div>

          {/* Achievements Section */}
          {stats.achievements?.length > 0 && (
            <div className="achievements-panel">
              <div className="achievements-title">🏆 Achievements</div>
              <div className="achievements-grid">
                {stats.achievements.map((a, i) => (
                  <div key={i} className="achievement-card">
                    <span className="achievement-icon">{a.icon || "🎖️"}</span>
                    <span className="achievement-name">{a.name}</span>
                    <span className="achievement-date">{a.achievedAt}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {stats.leagues?.length > 0 && (
            <div className="standings-panel" style={{ marginTop: 14 }}>
              <div className="standings-title">League Breakdown</div>
              <div className="standings-scroll">
                <div className="user-league-table">
                  <div className="user-league-head">
                    <div>League</div>
                    <div className="num">Points</div>
                    <div className="num">Correct</div>
                    <div className="num">Wrong</div>
                    <div className="num">Win %</div>
                  </div>
                  {stats.leagues.map(l => {
                    const lp = l.correct + l.wrong > 0 ? ((l.correct / (l.correct + l.wrong)) * 100).toFixed(1) : "—";
                    return (
                      <div key={l.id} className="user-league-row">
                        <div style={{ fontWeight: 900 }}>{l.name}</div>
                        <div className="num" style={{ color: "#a855f7" }}>{l.points || 0}</div>
                        <div className="num" style={{ color: "#22c55e" }}>{l.correct}</div>
                        <div className="num" style={{ color: "#ef4444" }}>{l.wrong}</div>
                        <div className="num" style={{ fontWeight: 1000 }}>{lp}{lp !== "—" ? "%" : ""}</div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          )}
        </>
      )}

      {activeTab === "team" && (
        <div style={{ marginTop: 12 }}>
          <div className="picks-modal" style={{ maxWidth: 500 }}>
            <div className="picks-modal-title">Favorite Team</div>
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
              <select className="select" value={favTeam} onChange={(e) => setFavTeam(e.target.value)} style={{ flex: 1, minWidth: 140 }}>
                <option value="">None</option>
                {TEAM_TRIS.map(t => <option key={t} value={t}>{t}</option>)}
              </select>
              <button className="btn picks-submit" onClick={() => saveFavTeam(favTeam)} disabled={favSaving} style={{ padding: "10px 20px" }}>
                {favSaving ? "…" : "Save"}
              </button>
              {favMsg && <span style={{ color: "#22c55e", fontWeight: 900, fontSize: 13 }}>{favMsg}</span>}
            </div>
          </div>

          {favTeam && (
            <div className="fav-team-card" style={{ "--ft": favTheme?.a || "#7c5cff" }}>
              <img
                className="fav-team-logo"
                src={`https://cdn.nba.com/logos/nba/${NBA_TEAM_IDS[favTeam] || 0}/global/L/logo.svg`}
                alt={favTeam}
                onError={e => e.currentTarget.style.display = "none"}
              />
              <div className="fav-team-name">{favTeam}</div>
              <button className="btn fav-team-btn" style={{ "--ft": favTheme?.a || "#7c5cff" }} onClick={() => onGoToTeam && onGoToTeam(favTeam)}>
                View Stats, Roster & Schedule →
              </button>
            </div>
          )}
        </div>
      )}

      {activeTab === "history" && (
        <div className="standings-panel" style={{ marginTop: 12 }}>
          <div className="standings-title">Recent Picks</div>
          {histLoading && <div className="status">Loading…</div>}
          <div className="standings-scroll">
            <div className="user-history-table">
              <div className="user-history-head">
                <div>Date</div>
                <div>Matchup</div>
                <div>Your Pick</div>
                <div className="num">Points</div>
                <div className="num">Result</div>
              </div>
              {history.map((p, i) => {
                const pt = teamTheme(p.pickedTeam);
                const powerUpIcon = p.isTripleCaptain ? "👑" : p.isDoubleDown ? "⚡" : "";
                return (
                  <div key={i} className={`user-history-row ${p.result === 1 ? "hist-win" : p.result === 0 ? "hist-loss" : ""}`}>
                    <div style={{ fontSize: 12, color: "rgba(255,255,255,0.55)" }}>{p.gameDate}</div>
                    <div style={{ fontWeight: 900 }}>{p.homeTri} vs {p.awayTri}</div>
                    <div style={{ color: pt.a, fontWeight: 1000 }}>
                      {powerUpIcon && <span style={{ marginRight: 4 }}>{powerUpIcon}</span>}
                      {p.pickedTeam}
                    </div>
                    <div className="num" style={{ color: "#a855f7", fontWeight: 1000 }}>
                      {p.result === 1 ? `+${p.pointsEarned || 10}` : p.result === 0 ? "0" : "—"}
                    </div>
                    <div className="num" style={{ fontWeight: 1000 }}>
                      {p.result === 1 ? <span style={{ color: "#22c55e" }}>✓</span> :
                       p.result === 0 ? <span style={{ color: "#ef4444" }}>✗</span> :
                       p.locked ? <span style={{ color: "rgba(255,255,255,0.35)" }}>Pending</span> :
                       <span style={{ color: "rgba(255,255,255,0.25)" }}>Open</span>}
                    </div>
                  </div>
                );
              })}
              {history.length === 0 && !histLoading && <div className="picks-empty" style={{ padding: 20 }}>No picks yet.</div>}
            </div>
          </div>
        </div>
      )}
    </>
  );
}

/* =========================
   Picks View — Fantasy Prediction Game
========================= */
function authFetch(url, token, opts = {}) {
  return fetch(url, { ...opts, headers: { ...opts.headers, "Content-Type": "application/json", Authorization: `Bearer ${token}` } });
}

/* Power-Up Badge Component */
function PowerUpBadge({ type, active, onClick, disabled }) {
  const icon = type === "tripleCaptain" ? "👑" : "⚡";
  const label = type === "tripleCaptain" ? "3×" : "2×";
  return (
    <button
      className={`power-up-badge ${active ? "active" : ""} ${type}`}
      onClick={onClick}
      disabled={disabled}
      title={type === "tripleCaptain" ? "Triple Captain (3× points)" : "Double Down (2× points)"}
    >
      <span className="power-up-icon">{icon}</span>
      <span className="power-up-label">{label}</span>
    </button>
  );
}

/* Points Display Component */
function PointsDisplay({ points, potential, isLoss, multiplier }) {
  if (isLoss) {
    return <div className="points-earned points-loss">0 pts</div>;
  }
  if (points != null) {
    return (
      <div className="points-earned points-win">
        +{points} pts
        {multiplier > 1 && <span className="points-bonus">{multiplier}×</span>}
      </div>
    );
  }
  if (potential != null) {
    return <div className="points-earned points-potential">~{potential} pts</div>;
  }
  return null;
}

/* Streak Display Component */
function StreakDisplay({ current, best }) {
  if (!current && !best) return null;
  const flames = current >= 10 ? "🔥🔥🔥" : current >= 5 ? "🔥🔥" : current >= 3 ? "🔥" : "";
  return (
    <div className="streak-display">
      {current > 0 && <span className="chipPill chipPill-streak">{flames} {current} streak</span>}
      {best > 0 && <span className="chipPill">Best: {best}</span>}
    </div>
  );
}

function AuthForm({ onLogin }) {
  const [mode, setMode] = useState("login");
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  async function submit(e) {
    e.preventDefault();
    setError(null);
    setLoading(true);
    const endpoint = mode === "login" ? "/api/picks/auth/login" : "/api/picks/auth/register";
    const body = mode === "login" ? { username, password } : { username, email, password };
    try {
      const res = await fetch(endpoint, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      const json = await res.json();
      if (json.ok) onLogin(json.token, json.user);
      else setError(json.error || "Something went wrong");
    } catch (err) { setError(err.message); }
    finally { setLoading(false); }
  }

  return (
    <div className="picks-auth">
      <div className="picks-auth-card">
        <div className="picks-auth-title">{mode === "login" ? "Log In" : "Create Account"}</div>
        <div className="picks-auth-sub">Pick winners. Compete with friends. Climb the leaderboard.</div>
        <div className="picks-auth-features">
          <div className="auth-feature"><span className="auth-feature-icon">👑</span> Triple Captain — 3× points once per week</div>
          <div className="auth-feature"><span className="auth-feature-icon">⚡</span> Double Down — 2× points once per day</div>
          <div className="auth-feature"><span className="auth-feature-icon">🔥</span> Streak Bonuses — Extra points for win streaks</div>
        </div>
        <div className="picks-auth-fields">
          <input className="picks-input" placeholder="Username" value={username} onChange={(e) => setUsername(e.target.value)} />
          {mode === "register" && <input className="picks-input" placeholder="Email" type="email" value={email} onChange={(e) => setEmail(e.target.value)} />}
          <input className="picks-input" placeholder="Password" type="password" value={password} onChange={(e) => setPassword(e.target.value)} onKeyDown={(e) => e.key === "Enter" && submit(e)} />
        </div>
        {error && <div className="picks-err">{error}</div>}
        <button className="btn picks-submit" onClick={submit} disabled={loading}>{loading ? "…" : mode === "login" ? "Log In" : "Sign Up"}</button>
        <div className="picks-toggle">
          {mode === "login" ? (<>No account? <span onClick={() => { setMode("register"); setError(null); }}>Sign up</span></>) : (<>Have an account? <span onClick={() => { setMode("login"); setError(null); }}>Log in</span></>)}
        </div>
      </div>
    </div>
  );
}

function PicksView({ token, user, onLogin, onLogout }) {
  const [leagues, setLeagues] = useState([]);
  const [activeLeague, setActiveLeague] = useState(null);
  const [leaderboard, setLeaderboard] = useState([]);
  const [games, setGames] = useState([]);
  const [picks, setPicks] = useState({});
  const [preds, setPreds] = useState({});
  const [dateStr, setDateStr] = useState(ymdNow());
  const [view, setView] = useState("games");
  const [showCreate, setShowCreate] = useState(false);
  const [showJoin, setShowJoin] = useState(false);
  const [newName, setNewName] = useState("");
  const [joinCode, setJoinCode] = useState("");
  const [msg, setMsg] = useState(null);
  const [lbFrom, setLbFrom] = useState("");
  const [lbTo, setLbTo] = useState("");
  const pollRef = useRef(null);
  
  // Fantasy power-up state
  const [powerUps, setPowerUps] = useState({
    tripleCaptainAvailable: true,
    tripleCaptainGameId: null,
    doubleDownAvailable: true,
    doubleDownGameId: null,
  });
  const [currentStreak, setCurrentStreak] = useState(0);
  const [totalPoints, setTotalPoints] = useState(0);
  const [selectedPowerUp, setSelectedPowerUp] = useState({}); // {gameId: "tripleCaptain" | "doubleDown" | null}

  /* ── data loaders ── */
  async function loadLeagues() {
    if (!token) return;
    try {
      const res = await authFetch("/api/picks/auth/me", token);
      const j = await res.json();
      if (j.ok) {
        setLeagues(j.leagues || []);
        if (!activeLeague && j.leagues?.length) setActiveLeague(j.leagues[0]);
      }
    } catch {}
  }

  async function loadGames(d) {
    try {
      const res = await fetch(`${API_BASE}/api/schedule/day?date=${d}`, { cache: "no-store" });
      const j = await res.json();
      setGames(j.games || []);
    } catch {}
  }

  async function loadPreds(d) {
    try {
      const res = await fetch(`${API_BASE}/api/predictions/day?date=${d}`, { cache: "no-store" });
      const j = await res.json();
      if (j.predictions) {
        const map = {};
        for (const p of j.predictions) { const gid = p.game_id || p.gameId; if (gid) map[gid] = p; }
        setPreds(map);
      }
    } catch {}
  }

  async function loadPicks(d) {
    if (!token) return;
    try {
      const res = await authFetch(`/api/picks/my?date=${d}`, token);
      const j = await res.json();
      if (j.ok) {
        const map = {};
        for (const p of j.picks) map[p.gameId] = p;
        setPicks(map);
        // Update power-up availability from response
        if (j.powerUps) {
          setPowerUps(j.powerUps);
        }
        if (j.currentStreak != null) {
          setCurrentStreak(j.currentStreak);
        }
        if (j.totalPoints != null) {
          setTotalPoints(j.totalPoints);
        }
      }
    } catch {}
  }

  async function loadLeaderboard(leagueId, from, to) {
    if (!token) return;
    try {
      let url;
      if (view === "global") {
        url = "/api/picks/global-leaderboard";
      } else {
        if (!leagueId) return;
        url = `/api/picks/leagues/${leagueId}/leaderboard`;
      }
      const params = new URLSearchParams();
      if (from) params.set("from", from);
      if (to) params.set("to", to);
      const qs = params.toString();
      const res = await authFetch(`${url}${qs ? "?" + qs : ""}`, token);
      const j = await res.json();
      if (j.ok) setLeaderboard(j.leaderboard || []);
    } catch {}
  }

  async function scoreFinished(gList) {
    const finished = (gList || games).filter((g) => g.gameStatusId === 3);
    if (!finished.length) return;
    const results = finished.map((g) => {
      const hw = (g.homeScore ?? 0) > (g.awayScore ?? 0);
      return { gameId: g.gameId, winnerTri: hw ? (g.home?.tri || "") : (g.away?.tri || "") };
    }).filter((r) => r.winnerTri);
    if (!results.length) return;
    try {
      await fetch("/api/picks/score", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ results }) });
    } catch {}
  }

  async function lockLiveGames(gList) {
    const live = (gList || games).filter((g) => g.gameStatusId >= 2);
    if (!live.length) return;
    try {
      await fetch("/api/picks/lock", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ gameIds: live.map((g) => g.gameId) }) });
    } catch {}
  }

  /* ── effects (ALL hooks run before any early return) ── */
  useEffect(() => { if (token) loadLeagues(); }, [token]);
  useEffect(() => { loadGames(dateStr); loadPreds(dateStr); }, [dateStr]);
  useEffect(() => {
    if (token) loadPicks(dateStr);
    if (activeLeague && token) loadLeaderboard(activeLeague.id, lbFrom, lbTo);
  }, [activeLeague, dateStr, token]);
  useEffect(() => {
    if (games.length && token) {
      lockLiveGames(games);
      scoreFinished(games).then(() => { if (token) loadPicks(dateStr); });
    }
  }, [games]);
  useEffect(() => {
    if (view === "global" || view === "leaderboard") loadLeaderboard(activeLeague?.id, lbFrom, lbTo);
  }, [view, lbFrom, lbTo]);

  /* Live polling: refresh scores every 30s when viewing today */
  useEffect(() => {
    if (pollRef.current) clearInterval(pollRef.current);
    if (dateStr === ymdNow()) {
      pollRef.current = setInterval(() => {
        loadGames(dateStr);
        if (token) loadPicks(dateStr);
      }, 30000);
    }
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [dateStr, token]);

  /* ── actions ── */
  async function makePick(g, team) {
    if (!token) return;
    const gameId = g.gameId;
    const powerUp = selectedPowerUp[gameId];
    const isTripleCaptain = powerUp === "tripleCaptain";
    const isDoubleDown = powerUp === "doubleDown";
    
    const body = {
      gameId, gameDate: dateStr,
      pickedTeam: team, homeTri: g.home?.tri || "", awayTri: g.away?.tri || "",
      gameStatus: g.gameStatusId || 1,
      isTripleCaptain,
      isDoubleDown,
    };
    try {
      const res = await authFetch("/api/picks/make", token, { method: "POST", body: JSON.stringify(body) });
      const j = await res.json();
      if (j.ok) {
        setPicks((p) => ({ 
          ...p, 
          [gameId]: { 
            ...body, 
            pickedTeam: team, 
            result: null, 
            locked: false,
            isTripleCaptain,
            isDoubleDown,
            potentialPoints: j.potentialPoints,
          } 
        }));
        // Clear the power-up selection for this game
        setSelectedPowerUp((prev) => ({ ...prev, [gameId]: null }));
        // Update power-up availability
        if (isTripleCaptain) {
          setPowerUps((prev) => ({ ...prev, tripleCaptainAvailable: false, tripleCaptainGameId: gameId }));
        }
        if (isDoubleDown) {
          setPowerUps((prev) => ({ ...prev, doubleDownAvailable: false, doubleDownGameId: gameId }));
        }
      }
      else setMsg(j.error);
    } catch (e) { setMsg(e.message); }
  }
  
  function togglePowerUp(gameId, type) {
    setSelectedPowerUp((prev) => {
      const current = prev[gameId];
      // If already selected, toggle off
      if (current === type) return { ...prev, [gameId]: null };
      // Otherwise, set the new power-up (and clear from any other game)
      const updated = {};
      // Clear this power-up from other games
      for (const gid of Object.keys(prev)) {
        if (prev[gid] === type) updated[gid] = null;
        else updated[gid] = prev[gid];
      }
      updated[gameId] = type;
      return updated;
    });
  }

  async function createLeague() {
    if (!newName.trim() || !token) return;
    const res = await authFetch("/api/picks/leagues/create", token, { method: "POST", body: JSON.stringify({ name: newName.trim() }) });
    const j = await res.json();
    if (j.ok) { setShowCreate(false); setNewName(""); loadLeagues(); setActiveLeague(j.league); }
    else setMsg(j.error);
  }

  async function joinLeague() {
    if (!joinCode.trim() || !token) return;
    const res = await authFetch("/api/picks/leagues/join", token, { method: "POST", body: JSON.stringify({ code: joinCode.trim() }) });
    const j = await res.json();
    if (j.ok) { setShowJoin(false); setJoinCode(""); loadLeagues(); setActiveLeague(j.league); }
    else setMsg(j.error);
  }

  /* ── computed (hooks must be before early return) ── */
  const sorted = [...games].sort((a, b) => (a.gameStatusId === 2 ? -1 : b.gameStatusId === 2 ? 1 : a.gameStatusId === 1 ? -1 : b.gameStatusId === 1 ? 1 : 0));
  const record = useMemo(() => {
    let w = 0, l = 0, p = 0;
    Object.values(picks).forEach((pk) => { if (pk.result === 1) w++; else if (pk.result === 0) l++; else p++; });
    return { w, l, p };
  }, [picks]);

  /* ── early return for auth gate (AFTER all hooks) ── */
  if (!token || !user) return <AuthForm onLogin={onLogin} />;

  return (
    <>
      <div className="hero">
        <div className="heroLeft">
          <div className="heroKicker">Fantasy Picks</div>
          <div className="heroTitle">NBA Prediction Game</div>
          <div className="heroMeta">
            {activeLeague ? (
              <>
                <div className="chipPill">{activeLeague.name}</div>
                <div className="chipPill">Code: {activeLeague.code}</div>
                <div className="chipPill" style={{ color: "#22c55e" }}>{record.w}W</div>
                <div className="chipPill" style={{ color: "#ef4444" }}>{record.l}L</div>
                {totalPoints > 0 && <div className="chipPill chipPill-points">🏆 {totalPoints} pts</div>}
                {currentStreak >= 3 && <div className="chipPill chipPill-streak">🔥 {currentStreak} streak</div>}
                {record.p > 0 && <div className="chipPill">{record.p} pending</div>}
              </>
            ) : (
              <div className="chipPill">No league selected</div>
            )}
          </div>
          {/* Power-up status bar */}
          <div className="power-up-status">
            <div className={`power-up-indicator ${powerUps.tripleCaptainAvailable ? "available" : "used"}`}>
              <span className="power-up-icon">👑</span>
              <span className="power-up-text">3× {powerUps.tripleCaptainAvailable ? "Ready" : "Used"}</span>
            </div>
            <div className={`power-up-indicator ${powerUps.doubleDownAvailable ? "available" : "used"}`}>
              <span className="power-up-icon">⚡</span>
              <span className="power-up-text">2× {powerUps.doubleDownAvailable ? "Ready" : "Used"}</span>
            </div>
          </div>
        </div>
        <div className="heroRight">
          <div className="datebar">
            <div className="datebar-left">
              <button className="btn" onClick={() => setDateStr(shiftDate(dateStr, -1))}>◀</button>
              <input className="date-input" type="date" value={dateStr} onChange={(e) => setDateStr(e.target.value)} />
              <button className="btn" onClick={() => setDateStr(shiftDate(dateStr, 1))}>▶</button>
              {dateStr !== ymdNow() && <button className="btn" onClick={() => setDateStr(ymdNow())}>Today</button>}
            </div>
          </div>
        </div>
      </div>

      <div className="toolbar">
        <div className="controls">
          {leagues.length > 0 && (
            <select className="select" value={activeLeague?.id || ""} onChange={(e) => setActiveLeague(leagues.find((l) => l.id === Number(e.target.value)) || null)}>
              {leagues.map((l) => <option key={l.id} value={l.id}>{l.name}</option>)}
            </select>
          )}
          <button className="btn" onClick={() => { setShowCreate(true); setShowJoin(false); }}>+ Create</button>
          <button className="btn" onClick={() => { setShowJoin(true); setShowCreate(false); }}>Join</button>
        </div>
        <div className="tabs conf-tabs">
          <button className={`tab ${view === "games" ? "active" : ""}`} onClick={() => setView("games")}>Games</button>
          <button className={`tab ${view === "leaderboard" ? "active" : ""}`} onClick={() => setView("leaderboard")}>League</button>
          <button className={`tab ${view === "global" ? "active" : ""}`} onClick={() => setView("global")}>Global</button>
        </div>
      </div>

      {msg && <div className="status error" onClick={() => setMsg(null)}>{msg}</div>}

      {showCreate && (
        <div className="picks-modal">
          <div className="picks-modal-title">Create League</div>
          <input className="picks-input" placeholder="League name" value={newName} onChange={(e) => setNewName(e.target.value)} onKeyDown={(e) => e.key === "Enter" && createLeague()} />
          <div style={{ display: "flex", gap: 8 }}>
            <button className="btn picks-submit" onClick={createLeague}>Create</button>
            <button className="btn" onClick={() => setShowCreate(false)}>Cancel</button>
          </div>
        </div>
      )}
      {showJoin && (
        <div className="picks-modal">
          <div className="picks-modal-title">Join League</div>
          <input className="picks-input" placeholder="Enter code (e.g. ABC123)" value={joinCode} onChange={(e) => setJoinCode(e.target.value.toUpperCase())} onKeyDown={(e) => e.key === "Enter" && joinLeague()} maxLength={6} style={{ letterSpacing: "3px", textAlign: "center", fontWeight: 950 }} />
          <div style={{ display: "flex", gap: 8 }}>
            <button className="btn picks-submit" onClick={joinLeague}>Join</button>
            <button className="btn" onClick={() => setShowJoin(false)}>Cancel</button>
          </div>
        </div>
      )}

      {!activeLeague && leagues.length === 0 && (
        <div className="picks-empty" style={{ padding: "14px 16px", fontSize: 13 }}>Join a league to compete with friends on the leaderboard, or just start picking!</div>
      )}

      {view === "games" && (
        <div className="grid" style={{ marginTop: 12 }}>
          {sorted.map((g) => {
            const homeTri = (g.home?.tri || "").toUpperCase();
            const awayTri = (g.away?.tri || "").toUpperCase();
            const ht = teamTheme(homeTri);
            const at = teamTheme(awayTri);
            const pick = picks[g.gameId];
            const isFinal = g.gameStatusId === 3;
            const isLive = g.gameStatusId === 2;
            const isLocked = isLive || isFinal || pick?.locked;
            const canPick = g.gameStatusId === 1 && !pick?.locked && !pick;
            const homeWins = isFinal && (g.homeScore ?? 0) > (g.awayScore ?? 0);

            // AI prediction
            const pred = preds[g.gameId];
            const pHome = pred?.p_home_win != null ? Number(pred.p_home_win) : null;
            const aiPick = pHome == null ? null : pHome >= 0.5 ? homeTri : awayTri;
            const aiConf = pHome == null ? null : Math.max(pHome, 1 - pHome) * 100;

            const homeLogo = g.home?.id ? `https://cdn.nba.com/logos/nba/${g.home.id}/global/L/logo.svg` : null;
            const awayLogo = g.away?.id ? `https://cdn.nba.com/logos/nba/${g.away.id}/global/L/logo.svg` : null;
            
            // Power-up state for this game
            const activePowerUp = selectedPowerUp[g.gameId];
            const pickHasTC = pick?.isTripleCaptain;
            const pickHasDD = pick?.isDoubleDown;
            const multiplier = pickHasTC ? 3 : pickHasDD ? 2 : 1;
            
            // Calculate potential points based on selected power-up
            const basePoints = 10;
            const potentialMultiplier = activePowerUp === "tripleCaptain" ? 3 : activePowerUp === "doubleDown" ? 2 : 1;
            const potentialPoints = basePoints * potentialMultiplier;

            return (
              <div key={g.gameId} className="picks-card" style={{ "--teamA": ht.a, "--teamB": at.a }}>
                <div className="picks-card-inner">
                  <div className="gameCardTop">
                    <div className="matchupTitle">
                      <span className="tri">{homeTri}</span>
                      <span className="vs">vs</span>
                      <span className="tri">{awayTri}</span>
                    </div>
                    <StatusBadge g={g} />
                  </div>

                  {(isLive || isFinal) && (
                    <div className="picks-score-row">
                      <span style={{ color: ht.a, fontWeight: 1000 }}>{g.homeScore ?? 0}</span>
                      <span style={{ color: "rgba(255,255,255,0.3)" }}>–</span>
                      <span style={{ color: at.a, fontWeight: 1000 }}>{g.awayScore ?? 0}</span>
                    </div>
                  )}

                  {aiPick && !isFinal && !isLive && (
                    <div className="picks-ai-row">
                      <span className="picks-ai-label">AI picks</span>
                      <span className="picks-ai-team" style={{ color: aiPick === homeTri ? ht.a : at.a }}>{aiPick}</span>
                      <span className="picks-ai-conf">({aiConf.toFixed(0)}%)</span>
                    </div>
                  )}

                  {/* Power-up selector - show before pick is made */}
                  {canPick && (powerUps.tripleCaptainAvailable || powerUps.doubleDownAvailable) && (
                    <div className="power-up-selector">
                      <span className="power-up-selector-label">Power-ups:</span>
                      {powerUps.tripleCaptainAvailable && (
                        <PowerUpBadge 
                          type="tripleCaptain" 
                          active={activePowerUp === "tripleCaptain"}
                          onClick={() => togglePowerUp(g.gameId, "tripleCaptain")}
                        />
                      )}
                      {powerUps.doubleDownAvailable && (
                        <PowerUpBadge 
                          type="doubleDown" 
                          active={activePowerUp === "doubleDown"}
                          onClick={() => togglePowerUp(g.gameId, "doubleDown")}
                        />
                      )}
                      {activePowerUp && (
                        <span className="power-up-potential">→ {potentialPoints} pts if correct</span>
                      )}
                    </div>
                  )}

                  {/* Show active power-up on locked picks */}
                  {pick && (pickHasTC || pickHasDD) && (
                    <div className={`power-up-active-badge ${pickHasTC ? "tc" : "dd"}`}>
                      {pickHasTC ? "👑 Triple Captain (3×)" : "⚡ Double Down (2×)"}
                    </div>
                  )}

                  {isLocked && !isFinal && pick && (
                    <div className="picks-lock-bar">Locked — pick submitted before tipoff</div>
                  )}
                  {isLocked && !isFinal && !pick && (
                    <div className="picks-lock-bar picks-lock-missed">Locked — no pick submitted</div>
                  )}

                  <div className="picks-buttons">
                    {[
                      { tri: homeTri, theme: ht, label: g.home?.name || homeTri, logo: homeLogo, isWinner: isFinal && homeWins },
                      { tri: awayTri, theme: at, label: g.away?.name || awayTri, logo: awayLogo, isWinner: isFinal && !homeWins },
                    ].map(({ tri, theme, label, logo, isWinner }) => {
                      const picked = pick?.pickedTeam === tri;
                      const correct = picked && pick?.result === 1;
                      const wrong = picked && pick?.result === 0;
                      const isAiPick = tri === aiPick;
                      return (
                        <button
                          key={tri}
                          className={`picks-btn ${picked ? "picked" : ""} ${correct ? "correct" : ""} ${wrong ? "wrong" : ""}`}
                          style={{ "--btnColor": theme.a, ...(picked ? { borderColor: theme.a, background: `color-mix(in srgb, ${theme.a} 18%, transparent)` } : {}) }}
                          onClick={() => canPick && makePick(g, tri)}
                          disabled={!canPick}
                        >
                          {logo && <img className="picks-btn-logo" src={logo} alt={tri} onError={e => e.currentTarget.style.display = "none"} />}
                          <span className="picks-btn-name">{label}</span>
                          <span className="picks-btn-tri">{tri}</span>
                          {!picked && canPick && isAiPick && <span className="picks-btn-badge ai">Agree with AI</span>}
                          {!picked && canPick && !isAiPick && aiPick && <span className="picks-btn-badge fade">Fade AI</span>}
                          {picked && !isFinal && <span className="picks-btn-badge picked-badge">Your Pick</span>}
                          {correct && <span className="picks-btn-badge correct">✓ Correct</span>}
                          {wrong && <span className="picks-btn-badge wrong">✗ Wrong</span>}
                          {!picked && isFinal && isWinner && <span className="picks-btn-badge winner">Winner</span>}
                        </button>
                      );
                    })}
                  </div>

                  {/* Points result bar */}
                  {isFinal && pick && (
                    <div className={`picks-result-bar ${pick.result === 1 ? "picks-result-win" : "picks-result-loss"}`}>
                      {pick.result === 1 ? (
                        <>
                          +{pick.pointsEarned || (basePoints * multiplier)} Points
                          {multiplier > 1 && <span className="points-multiplier">{multiplier}×</span>}
                        </>
                      ) : "0 Points"}
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {(view === "leaderboard" || view === "global") && (
        <div className="standings-panel" style={{ marginTop: 12 }}>
          <div className="standings-title">
            {view === "global" ? "Global Leaderboard" : `${activeLeague?.name || "League"} — Leaderboard`}
          </div>
          <div className="lb-date-filter">
            <label>From</label>
            <input type="date" className="picks-input lb-date-input" value={lbFrom} onChange={(e) => setLbFrom(e.target.value)} />
            <label>To</label>
            <input type="date" className="picks-input lb-date-input" value={lbTo} onChange={(e) => setLbTo(e.target.value)} />
            {(lbFrom || lbTo) && <button className="btn" onClick={() => { setLbFrom(""); setLbTo(""); }}>Clear</button>}
          </div>
          <div className="standings-scroll">
            <div className="picks-lb-table">
              <div className="picks-lb-head">
                <div className="rank">#</div>
                <div>Player</div>
                <div className="num points-col">Points</div>
                <div className="num">Correct</div>
                <div className="num">Wrong</div>
                <div className="num">Streak</div>
                <div className="num">Win %</div>
              </div>
              {leaderboard.map((m) => {
                const pct = m.correct + m.wrong > 0 ? ((m.correct / (m.correct + m.wrong)) * 100).toFixed(1) : "—";
                const favTheme = m.favoriteTeam ? teamTheme(m.favoriteTeam) : null;
                const medal = m.rank === 1 ? "🥇" : m.rank === 2 ? "🥈" : m.rank === 3 ? "🥉" : "";
                const streakFire = m.streak >= 3 ? "🔥" : "";
                return (
                  <div key={m.userId} className={`picks-lb-row ${m.isMe ? "picks-lb-me" : ""} ${m.rank % 2 ? "" : "odd"}`}>
                    <div className={`rank ${m.rank <= 3 ? "rank-playoff" : ""}`}>{medal || m.rank}</div>
                    <div style={{ fontWeight: m.isMe ? 1000 : 800, display: "flex", alignItems: "center", gap: 6 }}>
                      {favTheme && <span style={{ width: 8, height: 8, borderRadius: "50%", background: favTheme.a, flexShrink: 0 }} />}
                      {m.username}{m.isMe ? " (you)" : ""}
                    </div>
                    <div className="num points-col" style={{ color: "#a855f7", fontWeight: 1000 }}>{m.points || 0}</div>
                    <div className="num" style={{ color: "#22c55e" }}>{m.correct}</div>
                    <div className="num" style={{ color: "#ef4444" }}>{m.wrong}</div>
                    <div className="num" style={{ color: "#f59e0b" }}>{streakFire}{m.streak || 0}</div>
                    <div className="num" style={{ fontWeight: 1000 }}>{pct}{pct !== "—" ? "%" : ""}</div>
                  </div>
                );
              })}
              {leaderboard.length === 0 && <div className="picks-empty" style={{ padding: 20 }}>No picks yet — start predicting!</div>}
            </div>
          </div>
        </div>
      )}
    </>
  );
}

/* Swish Bot tab */
function SwishView() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([{ role: "bot", text: "Send a message to test the OpenAI key." }]);
  const [loading, setLoading] = useState(false);

  async function sendMessage() {
    const text = input.trim();
    if (!text || loading) return;

    const nextMessages = [...messages, { role: "user", text }];
    setMessages(nextMessages);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch("/api/swish/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: nextMessages.slice(-12) }),
      });

      const raw = await res.text();
      if (!raw) throw new Error(`Empty response body (${res.status})`);

      let data;
      try {
        data = JSON.parse(raw);
      } catch {
        throw new Error(`Non-JSON response (${res.status}): ${raw.slice(0, 200)}`);
      }

      if (!res.ok) throw new Error(data?.error || `Request failed (${res.status})`);
      setMessages((m) => [...m, { role: "bot", text: data.reply }]);
    } catch (err) {
      setMessages((m) => [...m, { role: "bot", text: `Error: ${err.message}` }]);
    } finally {
      setLoading(false);
    }
  }

  function newChat() {
    setMessages([{ role: "bot", text: "New chat started. How can I help?" }]);
    setInput("");
  }

  return (
    <>
      <div className="updated">Swish</div>

      <div className="card" style={{ display: "grid", gap: 12 }}>
        <div style={{ display: "flex", gap: 10, justifyContent: "flex-end" }}>
          <button className="tab" onClick={newChat} disabled={loading}>
            New chat
          </button>
        </div>

        <div style={{ display: "grid", gap: 10 }}>
          {messages.map((msg, idx) => (
            <div
              key={idx}
              style={{
                display: "flex",
                justifyContent: msg.role === "user" ? "flex-end" : "flex-start",
              }}
            >
              <div
                style={{
                  maxWidth: 560,
                  padding: "10px 12px",
                  borderRadius: 14,
                  border: "1px solid var(--border)",
                  background:
                    msg.role === "user"
                      ? "rgba(124, 92, 255, 0.16)"
                      : "rgba(255, 255, 255, 0.06)",
                  lineHeight: 1.5,
                  overflowWrap: "anywhere",
                }}
              >
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    p: ({ children }) => (
                      <p style={{ margin: 0 }}>{children}</p>
                    ),
                    strong: ({ children }) => (
                      <strong style={{ fontWeight: 750 }}>{children}</strong>
                    ),
                    code: ({ inline, children }) => (
                      <code
                        style={{
                          padding: inline ? "2px 6px" : "10px 12px",
                          borderRadius: 10,
                          border: "1px solid var(--border)",
                          display: inline ? "inline" : "block",
                          whiteSpace: inline ? "pre-wrap" : "pre",
                          overflowX: inline ? "visible" : "auto",
                          background: "rgba(255, 255, 255, 0.05)",
                        }}
                      >
                        {children}
                      </code>
                    ),
                    ul: ({ children }) => (
                      <ul style={{ margin: "6px 0 0 18px" }}>{children}</ul>
                    ),
                    ol: ({ children }) => (
                      <ol style={{ margin: "6px 0 0 18px" }}>{children}</ol>
                    ),
                    li: ({ children }) => (
                      <li style={{ marginTop: 4 }}>{children}</li>
                    ),
                  }}
                >
                  {msg.text}
                </ReactMarkdown>
              </div>
            </div>
          ))}
        </div>

        <div style={{ display: "flex", gap: 10 }}>
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") sendMessage();
            }}
            placeholder="Type a message..."
            style={{
              flex: 1,
              padding: "10px 12px",
              borderRadius: 12,
              border: "1px solid var(--border)",
              background: "rgba(255, 255, 255, 0.04)",
              color: "var(--text)",
              outline: "none",
            }}
            disabled={loading}
          />

          <button className="tab active" onClick={sendMessage} disabled={loading}>
            {loading ? "..." : "Send"}
          </button>
        </div>
      </div>
    </>
  );
}

/* App shell */
const AUTH_KEY = "__statline_token";

export default function App() {
  const [tab, setTab] = useState("schedule");
  const [authToken, setAuthToken] = useState(() => {
    try { return sessionStorage.getItem(AUTH_KEY) || null; } catch { return null; }
  });
  const [authUser, setAuthUser] = useState(null);
  const [teamInitial, setTeamInitial] = useState(null);

  function doLogin(token, user) {
    setAuthToken(token);
    setAuthUser(user);
    try { sessionStorage.setItem(AUTH_KEY, token); } catch {}
    setTab("picks");
  }
  function doLogout() {
    setAuthToken(null);
    setAuthUser(null);
    try { sessionStorage.removeItem(AUTH_KEY); } catch {}
    setTab("schedule");
  }
  function goToTeam(tri) {
    setTeamInitial(tri);
    setTab("teams");
  }

  useEffect(() => {
    if (!authToken) return;
    fetch("/api/picks/auth/me", { headers: { Authorization: `Bearer ${authToken}` } })
      .then((r) => r.json())
      .then((j) => { if (j.ok) setAuthUser(j.user); else doLogout(); })
      .catch(() => doLogout());
  }, [authToken]);

  // Clear teamInitial when leaving teams tab
  useEffect(() => { if (tab !== "teams") setTeamInitial(null); }, [tab]);

  /* Tab icon SVGs (shown on mobile only via CSS) */
  const tabIcons = {
    schedule: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="4" width="18" height="18" rx="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/></svg>,
    standings: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M8 21h8M12 17v4M6 3h12l-1 9H7L6 3z"/><path d="M6 7H3l1 5h2M18 7h3l-1 5h-2"/></svg>,
    teams: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20"/><path d="M2 12h20"/></svg>,
    picks: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>,
    swish: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>,
    user: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>,
  };

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <div className="brandMark" />
          <div>
            <h1>StatLine</h1>
            <div className="subtle">NBA Predictor</div>
          </div>
        </div>

        <nav className="tabs nav-tabs">
          {["schedule", "standings", "teams", "picks", "swish"].map((t) => (
            <button key={t} className={`tab ${tab === t ? "active" : ""}`} onClick={() => setTab(t)}>
              <span className="tab-icon">{tabIcons[t]}</span>
              <span className="tab-label">{t.charAt(0).toUpperCase() + t.slice(1)}</span>
            </button>
          ))}
          {authUser ? (
            <button className={`tab tab-user ${tab === "user" ? "active" : ""}`} onClick={() => setTab("user")} title="My profile">
              <span className="tab-icon">{tabIcons.user}</span>
              <span className="tab-label">{authUser.username}</span>
            </button>
          ) : (
            <button className="tab tab-login" onClick={() => setTab("picks")}>
              <span className="tab-icon">{tabIcons.user}</span>
              <span className="tab-label">Log In</span>
            </button>
          )}
        </nav>
      </header>

      {tab === "schedule" ? (
        <ScheduleView authToken={authToken} authUser={authUser} />
      ) : tab === "standings" ? (
        <StandingsView />
      ) : tab === "teams" ? (
        <TeamsView initialTeam={teamInitial} />
      ) : tab === "picks" ? (
        <PicksView token={authToken} user={authUser} onLogin={doLogin} onLogout={doLogout} />
      ) : tab === "user" ? (
        <UserView token={authToken} user={authUser} onLogout={doLogout} onGoToTeam={goToTeam} />
      ) : (
        <SwishView />
      )}
    </div>
  );
}