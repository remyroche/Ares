# src/analyst/meta_label_relevance.py

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd


def compute_mutual_information(
    X: pd.DataFrame,
    y: pd.Series,
    task: str = "classification",
    random_state: int = 42,
) -> dict[str, float]:
    """Compute mutual information scores per feature against target.

    Args:
        X: Feature frame (numeric)
        y: Target series
        task: "classification" or "regression"
        random_state: seed

    Returns:
        Dict feature_name -> MI score
    """
    try:
        Xn = X.select_dtypes(include=[np.number]).copy()
        if Xn.empty:
            return {}
        if task == "classification":
            from sklearn.feature_selection import mutual_info_classif

            mi = mutual_info_classif(Xn.fillna(0.0), y.astype(int), random_state=random_state)
        else:
            from sklearn.feature_selection import mutual_info_regression

            mi = mutual_info_regression(Xn.fillna(0.0), y, random_state=random_state)
        return {c: float(v) for c, v in zip(Xn.columns, mi)}
    except Exception:
        return {}


def compute_shap_importance(
    X: pd.DataFrame,
    y: pd.Series,
    model: Any | None = None,
    task: str = "classification",
    max_samples: int = 5000,
) -> dict[str, float]:
    """Compute approximate SHAP mean(|value|) per feature.

    If model is None, fits a lightweight LightGBM model for speed.
    """
    try:
        import shap  # type: ignore
        from lightgbm import LGBMClassifier, LGBMRegressor  # type: ignore

        Xn = X.select_dtypes(include=[np.number]).fillna(0.0)
        if len(Xn) == 0:
            return {}
        if len(Xn) > max_samples:
            Xn = Xn.sample(n=max_samples, random_state=1337)
            y = y.loc[Xn.index]

        if model is None:
            if task == "classification":
                model = LGBMClassifier(n_estimators=200, max_depth=-1, learning_rate=0.05, subsample=0.8)
            else:
                model = LGBMRegressor(n_estimators=200, max_depth=-1, learning_rate=0.05, subsample=0.8)
            model.fit(Xn, y)

        try:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(Xn)
            # For classification, sv can be a list per class; take last as positive class
            if isinstance(sv, list) and len(sv) > 0:
                sv = sv[-1]
            import numpy as _np

            magnitudes = _np.abs(_np.array(sv))
            if magnitudes.ndim == 1:
                magnitudes = magnitudes.reshape(-1, 1)
            mean_abs = _np.mean(magnitudes, axis=0)
            return {c: float(v) for c, v in zip(Xn.columns, mean_abs)}
        except Exception:
            return {}
    except Exception:
        return {}


def evaluate_sharpe_lift(
    returns_series: pd.Series,
    gating_series: pd.Series,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    """Compute Sharpe of baseline and gated series, and the delta.

    Args:
        returns_series: realized per-period returns
        gating_series: boolean/int indicator where 1 means include trade/period
        risk_free_rate: per-period risk free
    """
    try:
        r = returns_series.fillna(0.0)
        g = gating_series.fillna(0).astype(int)
        base_excess = r - risk_free_rate
        gated_r = r[g == 1]
        gated_excess = gated_r - risk_free_rate
        def _sharpe(x: pd.Series) -> float:
            mu = float(x.mean())
            sd = float(x.std(ddof=1))
            return (mu / sd) if sd > 1e-12 else 0.0
        sr_base = _sharpe(base_excess)
        sr_gated = _sharpe(gated_excess) if len(gated_excess) > 1 else 0.0
        return {
            "sharpe_base": float(sr_base),
            "sharpe_gated": float(sr_gated),
            "delta_sharpe": float(sr_gated - sr_base),
            "coverage": float(g.mean()),
        }
    except Exception:
        return {"sharpe_base": 0.0, "sharpe_gated": 0.0, "delta_sharpe": 0.0, "coverage": 0.0}