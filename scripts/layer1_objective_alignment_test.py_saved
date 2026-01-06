#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb

    LGBM_AVAILABLE = True
except Exception:
    LGBM_AVAILABLE = False

from src.training.steps.labeling.generate_weights_per_label import (
    compute_horizon_consistency,
    compute_uniqueness,
    generate_weights_per_label,
)


@dataclass
class ProxyMetrics:
    auc: float
    trade_rate: float
    mean_return: float
    sharpe: float


def _safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        y_true = np.asarray(y_true, dtype=float)
        y_prob = np.asarray(y_prob, dtype=float)
        if np.unique(y_true).size < 2:
            return 0.5
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return 0.5


def _oof_predict_proba_timeseries(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    sample_weight: Optional[np.ndarray] = None,
    n_splits: int = 5,
) -> np.ndarray:
    cv = TimeSeriesSplit(n_splits=n_splits)

    y_arr = y.values if hasattr(y, "values") else np.asarray(y)
    preds = np.full(len(y_arr), np.nan, dtype=float)

    w_arr = None
    if sample_weight is not None:
        w_arr = np.asarray(sample_weight, dtype=float)
        if w_arr.shape[0] != len(y_arr):
            raise ValueError(
                f"sample_weight length mismatch: {w_arr.shape[0]} vs y {len(y_arr)}"
            )

    for train_idx, test_idx in cv.split(X, y_arr):
        est = estimator
        # clone is optional; pipelines/clfs can be stateful
        try:
            from sklearn.base import clone

            est = clone(estimator)
        except Exception:
            pass

        fit_kwargs: Dict[str, Any] = {}
        if w_arr is not None:
            # Pipelines require <step_name>__sample_weight
            if isinstance(est, Pipeline) and getattr(est, "steps", None):
                last_step_name = est.steps[-1][0]
                fit_kwargs[f"{last_step_name}__sample_weight"] = w_arr[train_idx]
            else:
                fit_kwargs["sample_weight"] = w_arr[train_idx]

        try:
            est.fit(X.iloc[train_idx], y_arr[train_idx], **fit_kwargs)
        except TypeError:
            # estimator doesn't support sample_weight
            est.fit(X.iloc[train_idx], y_arr[train_idx])

        prob = est.predict_proba(X.iloc[test_idx])[:, 1]
        preds[test_idx] = prob

    # If any NaNs remain (shouldn't), fill with 0.5
    preds = np.where(np.isfinite(preds), preds, 0.5)
    return preds


def _compute_proxy_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    realized_return: np.ndarray,
    *,
    threshold: float = 0.5,
) -> ProxyMetrics:
    auc = _safe_roc_auc(y_true, y_prob)

    # simple long-only decision rule
    take = (y_prob >= threshold).astype(float)
    pnl = take * realized_return

    trade_rate = float(np.mean(take))
    mean_return = float(np.mean(pnl))
    sharpe = float(mean_return / (np.std(pnl) + 1e-9))

    return ProxyMetrics(auc=auc, trade_rate=trade_rate, mean_return=mean_return, sharpe=sharpe)


def _select_feature_columns(df: pd.DataFrame) -> List[str]:
    numeric = df.select_dtypes(include=[np.number])

    drop_cols = {
        "realized_return",
        "binary_label",
        "binary_label_long",
        "binary_label_short",
        "target_sample_weight",
        "event_duration_bars",
        "meta_probability",
        "meta_probability_calibrated_isotonic",
        "meta_probability_lgbm_bag_mean",
        "meta_probability_lgbm_bag_lower",
    }

    keep = [c for c in numeric.columns if c not in drop_cols]
    return keep


def _load_trials(trials_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(trials_csv)
    if "score" not in df.columns:
        raise ValueError(f"Trials CSV missing 'score' column: {trials_csv}")
    return df


def _load_labeled(labeled_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(labeled_csv, index_col=0)
    # robust datetime index conversion
    try:
        df.index = pd.to_datetime(df.index)
    except Exception:
        pass

    df = df.sort_index()
    return df


def _pick_percentile_rows(df_trials: pd.DataFrame, *, n_points: int = 20) -> pd.DataFrame:
    if n_points <= 0:
        raise ValueError("n_points must be > 0")

    df_sorted = df_trials.sort_values("score", ascending=True).reset_index(drop=False)
    n = len(df_sorted)
    if n == 0:
        raise ValueError("No trials rows")

    # 0%,5%,...,95% for n_points=20
    percentiles = [i / float(n_points) for i in range(n_points)]
    rows = []
    used_pos = set()
    for p in percentiles:
        pos = int(np.floor(p * (n - 1)))
        # avoid duplicates due to rounding
        while pos in used_pos and pos < n - 1:
            pos += 1
        used_pos.add(pos)
        rows.append(df_sorted.iloc[pos])

    picked = pd.DataFrame(rows)
    picked.insert(0, "percentile", [int(round(p * 100)) for p in percentiles])
    return picked


def main() -> int:
    parser = argparse.ArgumentParser(description="Layer1 objective alignment test")
    parser.add_argument(
        "--trials_csv",
        type=str,
        default=str(REPO_ROOT / "outcomes" / "hpo_layer1_trials_ETHUSDT_15m_20251212_142122.csv"),
    )
    parser.add_argument(
        "--labeled_csv",
        type=str,
        default=str(REPO_ROOT / "outcomes" / "weighted_labeled_data_ETHUSDT_15m_20251212_123203.csv"),
    )
    parser.add_argument("--n_points", type=int, default=20)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--include_consistency", action="store_true")
    parser.add_argument("--no_include_consistency", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    trials_csv = Path(args.trials_csv)
    labeled_csv = Path(args.labeled_csv)

    include_consistency = True
    if args.no_include_consistency:
        include_consistency = False
    if args.include_consistency:
        include_consistency = True

    df_trials = _load_trials(trials_csv)
    df_labeled = _load_labeled(labeled_csv)

    if "binary_label" not in df_labeled.columns:
        raise ValueError(f"Labeled CSV missing binary_label: {labeled_csv}")
    if "realized_return" not in df_labeled.columns:
        raise ValueError(f"Labeled CSV missing realized_return: {labeled_csv}")
    if "event_duration_bars" not in df_labeled.columns:
        raise ValueError(f"Labeled CSV missing event_duration_bars: {labeled_csv}")
    if "close" not in df_labeled.columns:
        raise ValueError(f"Labeled CSV missing close: {labeled_csv}")

    y_all = pd.to_numeric(df_labeled["binary_label"], errors="coerce")
    r_all = pd.to_numeric(df_labeled["realized_return"], errors="coerce")

    valid_mask = y_all.isin([0, 1]) & np.isfinite(r_all)
    df_evt = df_labeled.loc[valid_mask].copy()
    if len(df_evt) < 200:
        raise ValueError(f"Too few labeled events after filtering: {len(df_evt)}")

    y = pd.to_numeric(df_evt["binary_label"], errors="coerce").astype(float)
    realized_return = pd.to_numeric(df_evt["realized_return"], errors="coerce").astype(float).values

    if y.nunique() < 2:
        raise ValueError("Binary label has <2 classes after filtering")

    # Feature matrix
    feature_cols = _select_feature_columns(df_evt)
    if not feature_cols:
        raise ValueError("No usable numeric feature columns")

    X = df_evt[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Precompute per-event ingredients for weights
    t_events = df_evt.index

    close_series = pd.to_numeric(df_labeled["close"], errors="coerce").astype(float).fillna(method="ffill")
    close_ret = close_series.pct_change().fillna(0.0)
    full_volatility = close_ret.rolling(20).std().fillna(0.0)
    full_consistency = compute_horizon_consistency(close_series, horizon=12)

    vol_proxy = full_volatility.reindex(t_events).fillna(0.0).values
    if include_consistency:
        consistency_scores = full_consistency.reindex(t_events).fillna(0.0).values
    else:
        consistency_scores = None

    # Build t1 series for uniqueness
    dur = pd.to_numeric(df_evt["event_duration_bars"], errors="coerce").fillna(1.0).values
    dur_int = np.maximum(1, np.round(dur).astype(int))

    full_index = df_labeled.index
    t0_locs = pd.Series(np.arange(len(full_index)), index=full_index)
    start_locs = t0_locs.loc[t_events].values.astype(int)
    end_locs = np.minimum(start_locs + dur_int, len(full_index) - 1)
    t1_vals = full_index[end_locs]
    t1_series = pd.Series(t1_vals, index=t_events)

    uniqueness_scores = compute_uniqueness(t1_series, market_index=full_index).reindex(t_events).fillna(1.0).values

    # Pick trials across percentiles
    df_pick = _pick_percentile_rows(df_trials, n_points=args.n_points)

    # Models
    logreg = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "clf",
                LogisticRegression(
                    max_iter=400,
                    n_jobs=-1,
                    penalty="l2",
                    solver="lbfgs",
                ),
            ),
        ]
    )

    lgbm = None
    if LGBM_AVAILABLE:
        lgbm = lgb.LGBMClassifier(
            boosting_type="gbdt",
            objective="binary",
            n_estimators=80,
            max_depth=3,
            num_leaves=8,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=50,
            reg_alpha=0.1,
            reg_lambda=0.1,
            n_jobs=-1,
            random_state=42,
            verbose=-1,
            feature_pre_filter=False,
        )

    results: List[Dict[str, Any]] = []

    for _, row in df_pick.iterrows():
        layer1_score = float(row.get("score", np.nan))

        params = {
            "mag_compression": float(row.get("param_mag_compression")),
            "learn_slope": float(row.get("param_learn_slope")),
            "learn_center": float(row.get("param_learn_center")),
            "uniq_intensity": float(row.get("param_uniq_intensity")),
            "exp_mag": float(row.get("param_exp_mag")),
            "exp_uniq": float(row.get("param_exp_uniq")),
            "exp_learn": float(row.get("param_exp_learn")),
            "downside_multiplier": float(row.get("param_downside_multiplier")),
            "mag_clip_pct": float(row.get("param_mag_clip_pct")),
        }

        weights = generate_weights_per_label(
            returns=realized_return,
            t_events=t_events,
            close_series=None,
            consistency_scores=consistency_scores,
            uniqueness_scores=uniqueness_scores,
            vol_proxy=vol_proxy,
            mag_compression=params["mag_compression"],
            learn_slope=params["learn_slope"],
            learn_center=params["learn_center"],
            uniq_intensity=params["uniq_intensity"],
            exp_mag=params["exp_mag"],
            exp_learn=params["exp_learn"],
            exp_uniq=params["exp_uniq"],
            exp_cross=1.0,
            downside_multiplier=params["downside_multiplier"],
            mag_clip_pct=params["mag_clip_pct"],
        )

        # LOGREG
        oof_lr = _oof_predict_proba_timeseries(
            logreg, X, y, sample_weight=weights, n_splits=args.n_splits
        )
        m_lr = _compute_proxy_metrics(y.values, oof_lr, realized_return, threshold=args.threshold)

        # LGBM
        m_lgbm = ProxyMetrics(auc=float("nan"), trade_rate=float("nan"), mean_return=float("nan"), sharpe=float("nan"))
        if lgbm is not None:
            oof_lgb = _oof_predict_proba_timeseries(
                lgbm, X, y, sample_weight=weights, n_splits=args.n_splits
            )
            m_lgbm = _compute_proxy_metrics(y.values, oof_lgb, realized_return, threshold=args.threshold)

        out = {
            "percentile": int(row.get("percentile", -1)),
            "layer1_score": layer1_score,
            **{f"param_{k}": v for k, v in params.items()},
            "include_consistency": bool(include_consistency),
            "auc_logreg": m_lr.auc,
            "trade_rate_logreg": m_lr.trade_rate,
            "mean_return_logreg": m_lr.mean_return,
            "sharpe_logreg": m_lr.sharpe,
            "auc_lgbm": m_lgbm.auc,
            "trade_rate_lgbm": m_lgbm.trade_rate,
            "mean_return_lgbm": m_lgbm.mean_return,
            "sharpe_lgbm": m_lgbm.sharpe,
        }
        results.append(out)

    df_out = pd.DataFrame(results).sort_values("percentile")

    # Correlations
    def _corr(a: pd.Series, b: pd.Series, method: str) -> float:
        try:
            return float(pd.Series(a).corr(pd.Series(b), method=method))
        except Exception:
            return float("nan")

    summary = {
        "n_points": int(args.n_points),
        "n_events": int(len(df_evt)),
        "n_features": int(X.shape[1]),
        "include_consistency": bool(include_consistency),
        "corr_spearman_score_auc_logreg": _corr(df_out["layer1_score"], df_out["auc_logreg"], "spearman"),
        "corr_pearson_score_auc_logreg": _corr(df_out["layer1_score"], df_out["auc_logreg"], "pearson"),
        "corr_spearman_score_auc_lgbm": _corr(df_out["layer1_score"], df_out["auc_lgbm"], "spearman"),
        "corr_pearson_score_auc_lgbm": _corr(df_out["layer1_score"], df_out["auc_lgbm"], "pearson"),
        "corr_spearman_score_sharpe_logreg": _corr(df_out["layer1_score"], df_out["sharpe_logreg"], "spearman"),
        "corr_spearman_score_sharpe_lgbm": _corr(df_out["layer1_score"], df_out["sharpe_lgbm"], "spearman"),
    }

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = REPO_ROOT / "outcomes" / f"layer1_objective_alignment_ETHUSDT_15m_{ts}.csv"
    df_out.to_csv(out_path, index=False)

    summary_path = REPO_ROOT / "outcomes" / f"layer1_objective_alignment_summary_ETHUSDT_15m_{ts}.json"
    try:
        import json

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
    except Exception:
        pass

    print("\n=== Layer1 Objective Alignment Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"\nWrote: {out_path}")
    print(f"Wrote: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
