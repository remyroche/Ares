# src/analyst/meta_label_relevance.py

from __future__ import annotations

from typing import Any, Iterable

import json
import os
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


def compute_mutual_information_pair(
    Xi: pd.Series,
    Xj: pd.Series,
    y: pd.Series,
    task: str = "regression",
    random_state: int = 42,
) -> float:
    """Compute MI(y; [Xi, Xj]) for complementarity checks."""
    try:
        X = pd.DataFrame({"Xi": Xi.astype(float).fillna(0.0), "Xj": Xj.astype(float).fillna(0.0)})
        return list(compute_mutual_information(X, y, task=task, random_state=random_state).values())[-1] if X.shape[1] == 1 else sum(compute_mutual_information(X, y, task=task, random_state=random_state).values())
    except Exception:
        return 0.0


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


class MetaLabelRelevanceEvaluator:
    """Evaluate meta-label relevance with complementarity checks and persist active labels.

    Removal rule: remove a label only if it's weak alone AND does not add complementary information together with any other label.
    """

    def __init__(
        self,
        artifacts_dir: str,
        mi_threshold: float = 0.01,
        sharpe_min_delta: float = 0.0,
        synergy_mi_threshold: float = 0.005,
        max_pairs: int | None = None,
    ) -> None:
        self.artifacts_dir = artifacts_dir
        self.mi_threshold = float(mi_threshold)
        self.synergy_mi_threshold = float(synergy_mi_threshold)
        self.sharpe_min_delta = float(sharpe_min_delta)
        self.max_pairs = max_pairs
        os.makedirs(self.artifacts_dir, exist_ok=True)

    def _gating_from_intensity(
        self,
        df: pd.DataFrame,
        label_names: list[str],
        thresholds: dict[str, float],
    ) -> pd.DataFrame:
        gating = {}
        for name in label_names:
            col = f"intensity_{name}"
            if col in df.columns:
                thr = float(thresholds.get(name, 0.5))
                gating[name] = (pd.to_numeric(df[col], errors="coerce").fillna(0.0) >= thr).astype(int)
        return pd.DataFrame(gating, index=df.index)

    def evaluate_from_frame(
        self,
        df: pd.DataFrame,
        label_names: list[str],
        thresholds: dict[str, float],
        returns_col: str = "close_returns",
        risk_free_rate: float = 0.0,
    ) -> dict[str, Any]:
        # Prepare target returns
        if returns_col not in df.columns:
            return {"active_labels": label_names, "inactive_labels": [], "reason": "no_returns"}
        y = pd.to_numeric(df[returns_col], errors="coerce").fillna(0.0)
        # Build binary gating per label
        G = self._gating_from_intensity(df, label_names, thresholds)
        if G.empty:
            return {"active_labels": label_names, "inactive_labels": [], "reason": "no_gating"}

        # Univariate MI and Sharpe lift
        mi_scores = compute_mutual_information(G, y, task="regression")
        sharpe_lifts = {name: evaluate_sharpe_lift(y, G[name], risk_free_rate) for name in G.columns}

        # Pairwise complementarity (limited by max_pairs if set)
        labels = list(G.columns)
        pair_results: dict[tuple[str, str], dict[str, float]] = {}
        count = 0
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                if self.max_pairs is not None and count >= self.max_pairs:
                    break
                li, lj = labels[i], labels[j]
                # MI synergy approx
                mi_pair = compute_mutual_information(pd.DataFrame({li: G[li], lj: G[lj]}), y, task="regression")
                mi_pair_sum = sum(mi_pair.values()) if mi_pair else 0.0
                synergy_mi = mi_pair_sum - float(mi_scores.get(li, 0.0)) - float(mi_scores.get(lj, 0.0))
                # Sharpe lift for pair gating intersection
                g_pair = ((G[li] == 1) & (G[lj] == 1)).astype(int)
                sr_pair = evaluate_sharpe_lift(y, g_pair, risk_free_rate)
                pair_results[(li, lj)] = {
                    "synergy_mi": float(synergy_mi),
                    "delta_sharpe_pair": float(sr_pair.get("delta_sharpe", 0.0)),
                }
                count += 1

        # Decide active/inactive with complementarity rule
        active: set[str] = set()
        inactive: set[str] = set()
        for name in labels:
            mi_ok = float(mi_scores.get(name, 0.0)) >= self.mi_threshold
            sr_ok = float(sharpe_lifts.get(name, {"delta_sharpe": 0.0}).get("delta_sharpe", 0.0)) > self.sharpe_min_delta
            if mi_ok or sr_ok:
                active.add(name)
                continue
            # Check complementarity: exists any partner with synergy or pair Sharpe lift beyond min threshold
            complementary = False
            for other in labels:
                if other == name:
                    continue
                key = (name, other) if (name, other) in pair_results else (other, name)
                res = pair_results.get(key)
                if not res:
                    continue
                if res.get("synergy_mi", 0.0) > self.synergy_mi_threshold:
                    complementary = True
                    break
                if res.get("delta_sharpe_pair", 0.0) > self.sharpe_min_delta:
                    complementary = True
                    break
            if complementary:
                active.add(name)
            else:
                inactive.add(name)

        result = {
            "mi_scores": {k: float(v) for k, v in mi_scores.items()},
            "sharpe_lifts": {k: float(v.get("delta_sharpe", 0.0)) for k, v in sharpe_lifts.items()},
            "pair_results": pair_results,
            "active_labels": sorted(active),
            "inactive_labels": sorted(inactive),
        }
        try:
            with open(os.path.join(self.artifacts_dir, "active_labels.json"), "w") as f:
                json.dump({"active_labels": result["active_labels"], "inactive_labels": result["inactive_labels"]}, f, indent=2)
        except Exception:
            pass
        return result