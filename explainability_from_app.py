from __future__ import annotations

import os
import json
import logging
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from openai import OpenAI

# Single source of truth for already-computed predictions + feature matrix
from prediction_store import get_predictions_cached_for_date

logger = logging.getLogger(__name__)

_SHAP_CACHE: Dict[str, Tuple[Optional[List], List]] = {}  # date -> (shap_rows, feature_rows)

# ---------------------------------------------------------------------------
# Glossary: internal feature name -> human-readable label
# ---------------------------------------------------------------------------
GLOSSARY: Dict[str, str] = {
    "BLEND_FG_PCT_DIFF":      "field goal percentage",
    "BLEND_FG3_PCT_DIFF":     "3-point shooting percentage",
    "BLEND_FT_PCT_DIFF":      "free throw percentage",
    "BLEND_PTS_DIFF":         "points scored per game",
    "BLEND_REB_DIFF":         "rebounds per game",
    "BLEND_AST_DIFF":         "assists per game",
    "BLEND_TOV_DIFF":         "turnovers per game",
    "BLEND_PLUS_MINUS_DIFF":  "plus/minus differential",
    "BLEND_NET_MARGIN_DIFF":  "net scoring margin",
    "RECENT_WIN_PCT_DIFF":    "recent win rate (last 10 games)",
    "ELO_PRE_DIFF":           "overall team strength rating (Elo)",
}


def _friendly_name(feature: str) -> str:
    return GLOSSARY.get(feature, feature)


# ---------------------------------------------------------------------------
# SHAP computation
# ---------------------------------------------------------------------------
def _load_model_bundle(model_path: str) -> dict:
    return joblib.load(model_path)


def _compute_shap_home_class(model, X: pd.DataFrame) -> Optional[np.ndarray]:
    """
    Returns SHAP contributions for HOME-win class (class index 1),
    shaped (n_games, n_features).

    Uses KernelExplainer because SVC wrapped in a Pipeline doesn't support
    TreeExplainer. We convert everything to plain numpy arrays before
    passing to SHAP — this avoids the sklearn Pipeline read-only
    'feature_names_in_' property error that newer sklearn versions raise
    when SHAP tries to introspect the Pipeline object.
    """
    try:
        import shap  # type: ignore
    except ImportError:
        logger.warning("shap package not installed — SHAP values unavailable.")
        return None

    if X.empty:
        return None

    # Convert DataFrame -> numpy ONCE. SHAP never sees the Pipeline or DataFrame.
    X_np = X.values.astype(float)
    bg_np = X_np[: min(len(X_np), 50)]  # background dataset for KernelExplainer

    # Plain function: numpy in, numpy out — no sklearn introspection
    def _predict_fn(arr: np.ndarray) -> np.ndarray:
        return model.predict_proba(arr)

    explainer = shap.KernelExplainer(_predict_fn, bg_np)
    shap_values = explainer.shap_values(X_np, nsamples="auto")

    # shap_values is [class0_array, class1_array]
    if isinstance(shap_values, list) and len(shap_values) >= 2:
        return np.array(shap_values[1])
    return None


# ---------------------------------------------------------------------------
# OpenAI natural-language narrative
# ---------------------------------------------------------------------------

def _get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in environment.")
    return OpenAI(api_key=api_key)


def _build_shap_prompt(
    home_label: str,
    away_label: str,
    p_home_win: float,
    feature_row: Dict[str, float],
    shap_row: Optional[Dict[str, float]],
) -> str:
    """
    Builds the prompt we send to OpenAI. The prompt includes:
      - Which team is predicted to win and by how much
      - A ranked list of features with their SHAP importance and raw diff value
    """
    pick_team = home_label if p_home_win >= 0.5 else away_label
    confidence_pct = max(p_home_win, 1 - p_home_win) * 100

    # Build feature summary — ranked by |SHAP| if available, else by |diff|
    if shap_row:
        ranked = sorted(shap_row.items(), key=lambda kv: abs(kv[1]), reverse=True)
    else:
        ranked = sorted(feature_row.items(), key=lambda kv: abs(kv[1]), reverse=True)

    feature_lines: List[str] = []
    for feat, shap_val in ranked[:8]:          # top 8 drivers
        diff_val = feature_row.get(feat, 0.0)
        friendly = _friendly_name(feat)
        if diff_val > 0:
            edge = f"{home_label} leads by {diff_val:+.3f}"
        elif diff_val < 0:
            edge = f"{away_label} leads by {abs(diff_val):.3f}"
        else:
            edge = "teams are even"

        if shap_row:
            direction = "pushed toward HOME win" if shap_val > 0 else "pushed toward AWAY win"
            feature_lines.append(
                f"- {friendly}: {edge} | SHAP influence: {shap_val:+.4f} ({direction})"
            )
        else:
            feature_lines.append(f"- {friendly}: {edge}")

    features_block = "\n".join(feature_lines) if feature_lines else "No feature data available."

    prompt = f"""You are an NBA analyst writing for a casual basketball fan who knows the game but not machine learning.

A prediction model has analyzed the matchup between **{home_label}** (home) and **{away_label}** (away).

Prediction result:
- Model pick: **{pick_team}** to win
- Confidence: {confidence_pct:.1f}%
- Home win probability: {p_home_win * 100:.1f}%

The model uses SHAP values to explain *why* it made this prediction. 
Feature values are computed as (home team value − away team value), so a positive number means the home team has an advantage on that stat.
A positive SHAP value means that feature pushed the model toward predicting a home win; negative SHAP means it pushed toward an away win.

Top model drivers (sorted by importance):
{features_block}

Write a clear, conversational explanation (3–5 sentences) that:
1. States which team is favored and how confident the model is.
2. Explains the 2–3 biggest reasons for the prediction in plain English (translate stats into what they mean on the court).
3. Mentions any notable counter-factors favoring the other team if they exist.
4. Ends with a brief one-sentence caveat that models aren't perfect and upsets happen.

Do NOT use bullet points. Write flowing prose that a basketball fan would enjoy reading. Do NOT mention "SHAP" or "machine learning" — just explain the basketball reasons."""

    return prompt


def _openai_narrate(
    home_label: str,
    away_label: str,
    p_home_win: float,
    feature_row: Dict[str, float],
    shap_row: Optional[Dict[str, float]],
    model_name: str = "gpt-4o",
) -> str:
    """
    Calls OpenAI to generate a plain-English explanation.
    Falls back to a template string if the API call fails.
    """
    try:
        client = _get_openai_client()
        prompt = _build_shap_prompt(home_label, away_label, p_home_win, feature_row, shap_row)

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a friendly, knowledgeable NBA analyst. "
                        "Explain predictions clearly and conversationally for everyday basketball fans."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=350,
            temperature=0.6,
        )
        return response.choices[0].message.content.strip()

    except Exception as exc:
        logger.error("OpenAI narration failed: %s", exc)
        # Graceful fallback — template-based explanation
        return _template_narrate(home_label, away_label, p_home_win, shap_row, feature_row)


def _template_narrate(
    home_label: str,
    away_label: str,
    p_home_win: float,
    shap_row: Optional[Dict[str, float]],
    feature_row: Dict[str, float],
) -> str:
    """Fallback if OpenAI is unavailable."""
    pick = home_label if p_home_win >= 0.5 else away_label
    conf = max(p_home_win, 1 - p_home_win) * 100

    top_feat = None
    if shap_row:
        top_feat = max(shap_row.items(), key=lambda kv: abs(kv[1]), default=None)
    elif feature_row:
        top_feat = max(feature_row.items(), key=lambda kv: abs(kv[1]), default=None)

    reason = f" driven largely by a difference in {_friendly_name(top_feat[0])}" if top_feat else ""
    return (
        f"The model favors {pick} with {conf:.0f}% confidence{reason}. "
        "As always, basketball is unpredictable and upsets happen — use this as one data point among many."
    )


# ---------------------------------------------------------------------------
# Main output builder
# ---------------------------------------------------------------------------

def _build_game_json(
    home_label: str,
    away_label: str,
    p_home_win: float,
    model_name: str,
    date: str,
    feature_row: Dict[str, float],
    shap_row: Optional[Dict[str, float]],
    game_id: Optional[str] = None,
    openai_model: str = "gpt-4o",
) -> Dict:
    """
    Returns the full explainability object for one game.

    Schema:
    {
      "headline":            str,
      "narrative":           str,   ← OpenAI-generated plain-English explanation
      "top_reasons_home":    [{"reason": str, "feature": str, "shap": float}, ...],
      "top_reasons_away":    [{"reason": str, "feature": str, "shap": float}, ...],
      "numbers": {
        "p_home_win": float,
        "model": str,
        "date": str,
      },
      "caveat": str,
    }
    """
    favored_home = p_home_win >= 0.5
    close_game   = 0.45 <= p_home_win <= 0.55

    headline = (
        f"{home_label} favored at home vs {away_label}"
        if favored_home
        else f"{away_label} favored on the road at {home_label}"
    )

    # ── SHAP-ranked reasons ──────────────────────────────────────────────
    top_home: List[Dict] = []
    top_away: List[Dict] = []

    if shap_row:
        items = list(shap_row.items())
        pos = sorted([it for it in items if it[1] > 0], key=lambda x: abs(x[1]), reverse=True)[:3]
        neg = sorted([it for it in items if it[1] < 0], key=lambda x: abs(x[1]), reverse=True)[:3]

        for feat, sv in pos:
            diff_val = float(feature_row.get(feat, 0.0))
            top_home.append({
                "reason":  f"Home team has the edge in {_friendly_name(feat)}.",
                "feature": feat,
                "shap":    round(sv, 4),
            })

        for feat, sv in neg:
            diff_val = float(feature_row.get(feat, 0.0))
            top_away.append({
                "reason":  f"Away team has the edge in {_friendly_name(feat)}.",
                "feature": feat,
                "shap":    round(sv, 4),
            })

    # ── OpenAI narrative ─────────────────────────────────────────────────
    narrative = _openai_narrate(
        home_label=home_label,
        away_label=away_label,
        p_home_win=p_home_win,
        feature_row=feature_row,
        shap_row=shap_row,
        model_name=openai_model,
    )

    return {
        "headline": headline,
        "narrative": narrative,
        "top_reasons_home": top_home,
        "top_reasons_away": top_away,
        "numbers": {
            "p_home_win": round(float(p_home_win), 4),
            "game_id": game_id,
            "model": model_name,
            "date": date,
        },
        "caveat": (
            "This is a statistical model based on historical team performance trends. "
            "Single games are inherently unpredictable — even a heavy favourite can lose."
        ),
    }


# ---------------------------------------------------------------------------
# Public API used by Flask routes
# ---------------------------------------------------------------------------

def explain_predictions(
    date: str,
    model_path: str = "models/svm_momentum_svm.pkl",
    openai_model: str = "gpt-4o",
) -> List[Dict]:
    """
    Main entrypoint called by Flask's /api/explain/<date>.

    Returns a list of explainability objects — one per game scheduled on `date`.
    Each object contains:
      - headline        : one-line summary
      - narrative       : OpenAI-generated plain-English explanation (3-5 sentences)
      - top_reasons_home: top SHAP factors favoring the home team
      - top_reasons_away: top SHAP factors favoring the away team
      - numbers         : raw numeric context
      - caveat          : standard disclaimer
    """
    preds_df, X = get_predictions_cached_for_date(date)

    if preds_df is None or len(preds_df) == 0:
        return []

    # Load model bundle
    bundle       = _load_model_bundle(model_path)
    model        = bundle["model"]
    feature_names: List[str] = list(bundle["feature_names"])
    model_name   = str(bundle.get("name", "svm_momentum_svm"))

    # Align X columns to training feature order
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    X = X.reindex(columns=feature_names, fill_value=0.0).fillna(0.0)

    # Compute SHAP (cached per date)
    if date in _SHAP_CACHE:
        shap_rows, feature_rows = _SHAP_CACHE[date]
    else:
        shap_matrix  = _compute_shap_home_class(model, X)
        feature_rows = X.to_dict(orient="records")

        if shap_matrix is None:
            shap_rows = [None] * len(preds_df)
        else:
            shap_rows = [
                {feature_names[j]: float(shap_matrix[i, j]) for j in range(len(feature_names))}
                for i in range(len(preds_df))
            ]

        _SHAP_CACHE[date] = (shap_rows, feature_rows)

    out: List[Dict] = []
    for i, row in preds_df.reset_index(drop=True).iterrows():
        p_home_win = float(row["p_home_win"])

        home_label = str(row.get("home_name", "HOME"))
        away_label = str(row.get("away_name", "AWAY"))

        home_tri = str(row.get("home_tri", "")).strip()
        away_tri = str(row.get("away_tri", "")).strip()
        if home_tri:
            home_label = f"{home_label} ({home_tri})"
        if away_tri:
            away_label = f"{away_label} ({away_tri})"

        game_id_val = str(row.get("game_id", row.get("GAME_ID", ""))).strip()

        out.append(
            _build_game_json(
                home_label=home_label,
                away_label=away_label,
                p_home_win=p_home_win,
                model_name=model_name,
                date=date,
                feature_row=feature_rows[i],
                shap_row=shap_rows[i],
                game_id=game_id_val or None,
                openai_model=openai_model,
            )
        )

    return out


def shap_rows_for_date(
    date: str,
    model_path: str = "models/svm_momentum_svm.pkl",
) -> Tuple[Optional[List[Dict[str, float]]], List[Dict[str, float]]]:
    """
    Utility: returns (shap_rows, feature_rows) for a date without
    calling OpenAI. Useful for the app.py bullet-point features section.
    """
    if date in _SHAP_CACHE:
        return _SHAP_CACHE[date]

    preds_df, X = get_predictions_cached_for_date(date)
    if preds_df is None or preds_df.empty:
        return None, []

    bundle        = _load_model_bundle(model_path)
    model         = bundle["model"]
    feature_names = list(bundle["feature_names"])

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    X = X.reindex(columns=feature_names, fill_value=0.0).fillna(0.0)

    shap_matrix  = _compute_shap_home_class(model, X)
    feature_rows = X.to_dict(orient="records")

    if shap_matrix is None:
        shap_rows = [None] * len(X)
    else:
        shap_rows = [
            {feature_names[j]: float(shap_matrix[i, j]) for j in range(len(feature_names))}
            for i in range(len(X))
        ]

    _SHAP_CACHE[date] = (shap_rows, feature_rows)
    return shap_rows, feature_rows