from typing import Tuple
import pandas as pd

from predict_today_svm import get_predictions_with_features_for_date


_prediction_cache = {}

def get_predictions_cached_for_date(date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (preds_df, X_features_df).
    This is the single source of truth for predictions used by the app + explainability.
    Results are cached in memory so SHAP only runs once per date per server session.
    """
    if date not in _prediction_cache:
        preds_df, X = get_predictions_with_features_for_date(date)
        _prediction_cache[date] = (preds_df, X)
    return _prediction_cache[date]