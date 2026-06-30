#!/usr/bin/env python3
"""Train and evaluate one global portfolio-period multiplier.

The multiplier is timestamp-level and portfolio-wide.  It does not change
strategy scores, rank references, eligibility thresholds, or auction ordering.
It only changes the new-risk budget via ``portfolio_wallet_cap_multiplier`` and,
for G6 only, the per-bar entry cap.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.portfolio_policy_replay import (  # noqa: E402
    fit_hierarchical_ev_curves,
    normalise_candidate_table,
    portfolio_policy_params_from_live_config,
    replay_candidates,
)


DEFAULT_TRAIN_BROAD = Path(
    "data_perp/artifacts/reliability_blend_native_simple_policy_replay_20260624_floor070"
    "/simple_policy_optimiser/simple_policy_candidates_broad.parquet"
)
DEFAULT_TRAIN_DEPLOYABLE = Path(
    "data_perp/artifacts/reliability_blend_native_simple_policy_replay_20260624_floor070"
    "/simple_policy_optimiser/simple_policy_candidates.parquet"
)
DEFAULT_EVAL_CANDIDATES = Path(
    "data_perp/artifacts/reliability_blend_arm_A0_anchor_only_20260625_jun15_22"
    "/simple_policy_optimiser/simple_policy_candidates_broad.parquet"
)
DEFAULT_EVAL_COMPONENTS = Path(
    "data_perp/reports/native_reliability_blend_scores_20260625_jun15_22_fullfit"
    "/native_reliability_blend_scores.parquet"
)
DEFAULT_POLICY_MANIFEST = Path(
    "data_perp/reports/reliability_blend_component_arm_portfolio_ablation_20260625"
    "/A0_anchor_only/portfolio_policy_ablation_manifest.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "data_perp/reports/global_portfolio_period_multiplier_20260625"
)

HEADS = ("long_bars", "long_dist", "short_asset", "short_boll")
OUTCOME_TOKENS = (
    "return",
    "pnl",
    "exit",
    "mtm_path",
    "holding",
    "target",
    "barrier_hit",
)
BASE_FEATURE_COLS = (
    "calibrated_score",
    "normalized_rank_score",
    "strategy_rank_pct",
    "policy_rank_pct",
    "rank_pct",
    "base_strategy_threshold",
    "deployment_rank_threshold",
    "expected_friction_bps",
    "expected_spread_bps",
    "expected_half_spread_bps",
    "spread_cost_bps",
    "fees_bps",
    "slippage_bps",
    "price_gap_bps",
    "entry_gap_bps",
    "entry_slippage_proxy_bps",
    "liquidity_capacity_weight",
    "orderbook_slippage_bps",
    "barrier_pct",
    "policy_effective_barrier_pct",
)
FEATURE_KEYWORDS = (
    "uncert",
    "entropy",
    "drift",
    "leaf",
    "support",
    "centroid",
    "similarity",
    "regime",
    "period",
    "qfail",
    "blend_",
    "rare",
    "contrib",
    "psi",
    "ks_",
    "mahalanobis",
    "frobenius",
    "fund",
    "oi_",
    "spread",
    "liquidity",
    "volume",
    "volatility",
    "breadth",
    "dispersion",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(float(value)) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _infer_head(strategy_id: Any) -> str:
    sid = str(strategy_id)
    for head in HEADS:
        if sid.startswith(head):
            return head
    return "unknown"


def _load_policy_params(path: Path, variant: str):
    payload = json.loads(path.read_text(encoding="utf-8"))
    params = payload.get("variant_params", {}).get(variant)
    if not isinstance(params, dict):
        raise KeyError(f"Missing variant_params[{variant!r}] in {path}")
    return portfolio_policy_params_from_live_config(params), payload


def _load_candidates(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if "head" not in df.columns:
        df["head"] = df["strategy_id"].map(_infer_head)
    for col in ("symbol", "side", "strategy_id", "head"):
        if col in df.columns:
            df[col] = df[col].astype(str)
    if "deployment_rank_threshold" not in df.columns:
        df["deployment_rank_threshold"] = df.get("base_strategy_threshold", np.nan)
    return normalise_candidate_table(df)


def _attach_eval_components(candidates: pd.DataFrame, component_path: Path | None) -> pd.DataFrame:
    if component_path is None or not component_path.exists():
        return candidates
    comp = pd.read_parquet(component_path)
    comp["timestamp"] = pd.to_datetime(comp["timestamp"], utc=True, errors="coerce")
    keep = [
        "timestamp",
        "symbol",
        "strategy_id",
        "period_component_score",
        "period_component_rank",
        "qfail_component_score",
        "qfail_component_rank",
        "anchor_component_rank",
    ]
    keep = [col for col in keep if col in comp.columns]
    comp = comp[keep].drop_duplicates(["timestamp", "symbol", "strategy_id"])
    return candidates.merge(
        comp,
        on=["timestamp", "symbol", "strategy_id"],
        how="left",
        validate="many_to_one",
    )


def _safe_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _feature_columns(df: pd.DataFrame, max_cols: int) -> list[str]:
    numeric = set(df.select_dtypes(include=[np.number, "bool"]).columns)
    priority: list[str] = []
    secondary: list[str] = []
    for col in df.columns:
        lower = col.lower()
        if col not in numeric:
            continue
        if any(token in lower for token in OUTCOME_TOKENS):
            continue
        if col in BASE_FEATURE_COLS or any(token in lower for token in ("period", "qfail", "blend_")):
            priority.append(col)
        elif any(token in lower for token in FEATURE_KEYWORDS):
            secondary.append(col)
    cols = priority + [col for col in secondary if col not in set(priority)]
    return cols[: max(1, int(max_cols))]


def _timestamp_features(
    df: pd.DataFrame,
    *,
    feature_cols: list[str] | None = None,
    max_cols: int = 96,
    fill_values: pd.Series | None = None,
) -> pd.DataFrame:
    work = df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    if "head" not in work.columns:
        work["head"] = work["strategy_id"].map(_infer_head)
    if feature_cols is None:
        feature_cols = _feature_columns(work, max_cols=max_cols)
    rows: list[pd.DataFrame] = []
    base = work.groupby("timestamp", sort=True).size().rename("candidate_count").to_frame()
    base["strategy_count"] = work.groupby("timestamp")["strategy_id"].nunique()
    base["symbol_count"] = work.groupby("timestamp")["symbol"].nunique()
    rank = _safe_numeric(work, "normalized_rank_score")
    threshold = _safe_numeric(work, "base_strategy_threshold")
    work["_rank_ge_threshold"] = (rank >= threshold).astype(float)
    work["_rank_ge_070"] = (rank >= 0.70).astype(float)
    for col in ("_rank_ge_threshold", "_rank_ge_070"):
        base[col + "_mean"] = work.groupby("timestamp")[col].mean()
    for col in feature_cols:
        vals = _safe_numeric(work, col)
        if vals.notna().sum() == 0:
            continue
        tmp = work[["timestamp"]].copy()
        tmp[col] = vals
        agg = tmp.groupby("timestamp")[col].agg(["mean", "std", "min", "max"])
        agg.columns = [f"{col}__{stat}" for stat in agg.columns]
        rows.append(agg)
    by_head_frames: list[pd.DataFrame] = []
    for head in HEADS:
        g = work.loc[work["head"].eq(head)].copy()
        if g.empty:
            continue
        h = g.groupby("timestamp").size().rename(f"{head}__rows").to_frame()
        h[f"{head}__frac_rank_ge_threshold"] = g.groupby("timestamp")["_rank_ge_threshold"].mean()
        h[f"{head}__rank_mean"] = _safe_numeric(g, "normalized_rank_score").groupby(g["timestamp"]).mean()
        h[f"{head}__rank_max"] = _safe_numeric(g, "normalized_rank_score").groupby(g["timestamp"]).max()
        h[f"{head}__score_mean"] = _safe_numeric(g, "calibrated_score").groupby(g["timestamp"]).mean()
        by_head_frames.append(h)
    out = pd.concat([base] + rows + by_head_frames, axis=1).sort_index()
    head_score_cols = [c for c in out.columns if c.endswith("__score_mean")]
    if head_score_cols:
        out["cross_head_score_mean_std"] = out[head_score_cols].std(axis=1)
        out["cross_head_score_mean_range"] = out[head_score_cols].max(axis=1) - out[head_score_cols].min(axis=1)
    out = out.replace([np.inf, -np.inf], np.nan)
    if fill_values is None:
        fill_values = out.median(numeric_only=True)
    out = out.fillna(fill_values).fillna(0.0)
    out.index.name = "timestamp"
    return out.reset_index()


def _timestamp_feature_fill_values(features: pd.DataFrame) -> pd.Series:
    numeric = features.drop(columns=["timestamp"], errors="ignore")
    return numeric.median(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _accepted_trades(candidates: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    accepted = decisions.loc[decisions["accepted"].astype(bool)].copy()
    if accepted.empty:
        return pd.DataFrame()
    idx = pd.to_numeric(accepted["candidate_index"], errors="coerce").astype("Int64")
    accepted = accepted.loc[idx.notna()].copy()
    idx = idx.loc[idx.notna()].astype(int)
    cand = candidates.reset_index(drop=True).iloc[idx.to_numpy()].reset_index(drop=True)
    accepted = accepted.reset_index(drop=True)
    for col in ["head", "net_return", "gross_return", "simple_policy_exit_reason", "exit_timestamp"]:
        if col in cand.columns:
            accepted[col] = cand[col].to_numpy()
    accepted["timestamp"] = pd.to_datetime(accepted["timestamp"], utc=True, errors="coerce")
    if "exit_timestamp" in accepted.columns:
        accepted["exit_timestamp"] = pd.to_datetime(accepted["exit_timestamp"], utc=True, errors="coerce")
    accepted["position_size"] = pd.to_numeric(accepted["position_size"], errors="coerce").fillna(0.0)
    accepted["net_return"] = pd.to_numeric(accepted["net_return"], errors="coerce").fillna(0.0)
    accepted["gross_return"] = pd.to_numeric(accepted["gross_return"], errors="coerce").fillna(0.0)
    accepted["net_pnl"] = accepted["position_size"] * accepted["net_return"]
    accepted["gross_pnl"] = accepted["position_size"] * accepted["gross_return"]
    accepted["cost_pnl"] = accepted["gross_pnl"] - accepted["net_pnl"]
    return accepted


def _forward_labels(timestamps: pd.Series, accepted: pd.DataFrame, horizon_hours: int) -> pd.DataFrame:
    ts_values = pd.to_datetime(timestamps, utc=True, errors="coerce").dropna().drop_duplicates().sort_values()
    rows: list[dict[str, Any]] = []
    acc = accepted.copy()
    if acc.empty:
        return pd.DataFrame({"timestamp": ts_values})
    acc["timestamp"] = pd.to_datetime(acc["timestamp"], utc=True, errors="coerce")
    full_sl = acc["simple_policy_exit_reason"].astype(str).str.lower().isin(["sl", "full_sl", "stop", "stop_loss"])
    timeout = acc["simple_policy_exit_reason"].astype(str).str.lower().str.contains("timeout", regex=False)
    acc["_full_sl"] = full_sl.astype(float)
    acc["_timeout"] = timeout.astype(float)
    for ts in ts_values:
        end = ts + pd.Timedelta(hours=int(horizon_hours))
        window = acc.loc[(acc["timestamp"] > ts) & (acc["timestamp"] <= end)]
        notional = float(window["position_size"].sum())
        if window.empty or notional <= 0.0:
            rows.append(
                {
                    "timestamp": ts,
                    "future_notional": 0.0,
                    "future_utility": np.nan,
                    "future_full_sl_rate": np.nan,
                    "future_timeout_rate": np.nan,
                    "future_cost_to_gross": np.nan,
                }
            )
            continue
        gross_abs = float(window["gross_pnl"].abs().sum())
        rows.append(
            {
                "timestamp": ts,
                "future_notional": notional,
                "future_utility": float(window["net_pnl"].sum() / max(notional, 1e-9)),
                "future_full_sl_rate": float((window["_full_sl"] * window["position_size"]).sum() / max(notional, 1e-9)),
                "future_timeout_rate": float((window["_timeout"] * window["position_size"]).sum() / max(notional, 1e-9)),
                "future_cost_to_gross": float(window["cost_pnl"].sum() / max(gross_abs, 1e-9)),
            }
        )
    labels = pd.DataFrame(rows)
    labels["future_low_opportunity"] = (
        pd.to_numeric(labels["future_timeout_rate"], errors="coerce").fillna(0.0)
        + pd.to_numeric(labels["future_cost_to_gross"], errors="coerce").clip(lower=0.0, upper=3.0).fillna(0.0)
    )
    return labels


def _add_trailing_performance(features: pd.DataFrame, accepted: pd.DataFrame) -> pd.DataFrame:
    out = features.copy().sort_values("timestamp")
    if accepted.empty:
        for col in ("trailing_net_pnl_24h", "trailing_full_sl_rate_24h", "trailing_cost_to_gross_24h"):
            out[col] = 0.0
        return out
    acc = accepted.copy().sort_values("timestamp")
    event_time = pd.to_datetime(acc.get("exit_timestamp", acc["timestamp"]), utc=True, errors="coerce")
    acc["_performance_event_time"] = event_time.fillna(pd.to_datetime(acc["timestamp"], utc=True, errors="coerce"))
    rows: list[dict[str, float]] = []
    for ts in pd.to_datetime(out["timestamp"], utc=True):
        start = ts - pd.Timedelta(hours=24)
        w = acc.loc[(acc["_performance_event_time"] < ts) & (acc["_performance_event_time"] >= start)]
        gross_abs = float(w["gross_pnl"].abs().sum())
        notional = float(w["position_size"].sum())
        reason = w["simple_policy_exit_reason"].astype(str).str.lower()
        rows.append(
            {
                "trailing_net_pnl_24h": float(w["net_pnl"].sum()) if len(w) else 0.0,
                "trailing_full_sl_rate_24h": float((reason.isin(["sl", "full_sl", "stop", "stop_loss"]).astype(float) * w["position_size"]).sum() / max(notional, 1e-9)) if len(w) else 0.0,
                "trailing_cost_to_gross_24h": float(w["cost_pnl"].sum() / max(gross_abs, 1e-9)) if len(w) else 0.0,
            }
        )
    return pd.concat([out.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def _add_portfolio_state_features(features: pd.DataFrame, equity: pd.DataFrame) -> pd.DataFrame:
    out = features.copy().sort_values("timestamp")
    state_cols = [
        "wallet",
        "mtm_equity",
        "unrealized_pnl",
        "open_notional",
        "open_capital_pct",
        "open_positions",
        "entries_this_bar",
    ]
    if equity.empty:
        for col in state_cols:
            out[f"portfolio_state_{col}"] = 0.0
        return out
    eq = equity.copy()
    eq["timestamp"] = pd.to_datetime(eq["timestamp"], utc=True, errors="coerce")
    eq = eq.dropna(subset=["timestamp"]).sort_values("timestamp")
    keep = ["timestamp"] + [col for col in state_cols if col in eq.columns]
    eq = eq[keep].drop_duplicates("timestamp", keep="last")
    left = out[["timestamp"]].copy()
    left["timestamp"] = pd.to_datetime(left["timestamp"], utc=True, errors="coerce")
    merged = pd.merge_asof(
        left.sort_values("timestamp"),
        eq.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
        allow_exact_matches=False,
    )
    merged = merged.reindex(left.sort_values("timestamp").index).sort_index()
    for col in state_cols:
        out[f"portfolio_state_{col}"] = pd.to_numeric(merged.get(col), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return out


def _add_open_position_concentration_features(features: pd.DataFrame, accepted: pd.DataFrame) -> pd.DataFrame:
    out = features.copy().sort_values("timestamp")
    cols = [
        "portfolio_open_head_hhi",
        "portfolio_open_side_hhi",
        "portfolio_open_symbol_hhi",
        "portfolio_open_max_head_share",
        "portfolio_open_max_symbol_share",
        "portfolio_open_short_share",
        "portfolio_open_long_share",
    ]
    if accepted.empty:
        for col in cols:
            out[col] = 0.0
        return out
    acc = accepted.copy()
    acc["timestamp"] = pd.to_datetime(acc["timestamp"], utc=True, errors="coerce")
    acc["exit_timestamp"] = pd.to_datetime(acc.get("exit_timestamp"), utc=True, errors="coerce")
    acc["position_size"] = pd.to_numeric(acc.get("position_size"), errors="coerce").fillna(0.0)
    if "head" not in acc.columns:
        acc["head"] = acc.get("strategy_id", "").map(_infer_head)
    rows: list[dict[str, float]] = []
    for ts in pd.to_datetime(out["timestamp"], utc=True):
        open_pos = acc.loc[
            (acc["timestamp"] < ts)
            & (acc["exit_timestamp"].isna() | (acc["exit_timestamp"] > ts))
            & (acc["position_size"] > 0.0)
        ]
        total = float(open_pos["position_size"].sum())
        if open_pos.empty or total <= 0.0:
            rows.append({col: 0.0 for col in cols})
            continue

        def _shares(group_col: str) -> pd.Series:
            if group_col not in open_pos.columns:
                return pd.Series(dtype=float)
            return open_pos.groupby(group_col)["position_size"].sum() / total

        head_share = _shares("head")
        side_share = _shares("side")
        symbol_share = _shares("symbol")
        rows.append(
            {
                "portfolio_open_head_hhi": float((head_share**2).sum()) if len(head_share) else 0.0,
                "portfolio_open_side_hhi": float((side_share**2).sum()) if len(side_share) else 0.0,
                "portfolio_open_symbol_hhi": float((symbol_share**2).sum()) if len(symbol_share) else 0.0,
                "portfolio_open_max_head_share": float(head_share.max()) if len(head_share) else 0.0,
                "portfolio_open_max_symbol_share": float(symbol_share.max()) if len(symbol_share) else 0.0,
                "portfolio_open_short_share": float(side_share.get("short", 0.0)) if len(side_share) else 0.0,
                "portfolio_open_long_share": float(side_share.get("long", 0.0)) if len(side_share) else 0.0,
            }
        )
    return pd.concat([out.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def _fit_models(train: pd.DataFrame, feature_cols: list[str]):
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    work = train.dropna(subset=["future_utility"]).copy()
    if len(work) < 30:
        raise RuntimeError(f"Not enough timestamp labels for global period model: {len(work)}")
    utility_tail = float(work["future_utility"].quantile(0.15))
    adverse_tail = float(work["future_full_sl_rate"].quantile(0.75))
    lowopp_tail = float(work["future_low_opportunity"].quantile(0.75))
    work["adverse_label"] = (
        (work["future_utility"] <= utility_tail)
        | (work["future_full_sl_rate"] >= adverse_tail)
    ).astype(int)
    work["lowopp_label"] = (work["future_low_opportunity"] >= lowopp_tail).astype(int)
    X = work[feature_cols]
    models = {
        "utility_mean": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            GradientBoostingRegressor(random_state=17, max_depth=2, n_estimators=120, learning_rate=0.04),
        ),
        "utility_q10": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            GradientBoostingRegressor(
                random_state=19,
                loss="quantile",
                alpha=0.10,
                max_depth=2,
                n_estimators=140,
                learning_rate=0.04,
            ),
        ),
        "adverse": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=180,
                max_depth=4,
                min_samples_leaf=max(3, int(len(work) * 0.05)),
                random_state=23,
                n_jobs=1,
                class_weight="balanced_subsample",
            ),
        ),
        "lowopp": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=180,
                max_depth=4,
                min_samples_leaf=max(3, int(len(work) * 0.05)),
                random_state=29,
                n_jobs=1,
                class_weight="balanced_subsample",
            ),
        ),
    }
    for model in models.values():
        model.fit(X, work[{
            "utility_mean": "future_utility",
            "utility_q10": "future_utility",
            "adverse": "adverse_label",
            "lowopp": "lowopp_label",
        }[next(k for k, v in models.items() if v is model)]])
    train_pred = _predict_models(models, work, feature_cols)
    cutoffs = {
        "risk_p50": float(train_pred["combined_risk"].quantile(0.50)),
        "risk_p75": float(train_pred["combined_risk"].quantile(0.75)),
        "risk_p90": float(train_pred["combined_risk"].quantile(0.90)),
        "period_p50": float(train["period_proxy"].quantile(0.50)) if "period_proxy" in train else 0.5,
        "period_p75": float(train["period_proxy"].quantile(0.75)) if "period_proxy" in train else 0.75,
        "period_p90": float(train["period_proxy"].quantile(0.90)) if "period_proxy" in train else 0.90,
        "utility_tail": utility_tail,
        "adverse_tail": adverse_tail,
        "lowopp_tail": lowopp_tail,
    }
    return models, cutoffs, work


def _predict_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.shape[1] == 1:
            return np.zeros(len(X), dtype=float) + float(proba[:, 0].mean())
        return proba[:, 1]
    pred = model.predict(X)
    return np.asarray(pred, dtype=float)


def _predict_models(models: dict[str, Any], frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    X = frame[feature_cols]
    out = frame[["timestamp"]].copy()
    out["pred_utility_mean"] = np.asarray(models["utility_mean"].predict(X), dtype=float)
    out["pred_utility_q10"] = np.asarray(models["utility_q10"].predict(X), dtype=float)
    out["pred_adverse_risk"] = _predict_proba(models["adverse"], X)
    out["pred_lowopp_risk"] = _predict_proba(models["lowopp"], X)
    out["combined_risk"] = np.maximum(out["pred_adverse_risk"], out["pred_lowopp_risk"])
    return out


def _map_risk_to_multiplier(risk: pd.Series, cutoffs: dict[str, float]) -> pd.Series:
    r = pd.to_numeric(risk, errors="coerce").fillna(0.0)
    out = pd.Series(1.0, index=r.index, dtype=float)
    out.loc[r >= float(cutoffs["risk_p50"])] = 0.75
    out.loc[r >= float(cutoffs["risk_p75"])] = 0.50
    out.loc[r >= float(cutoffs["risk_p90"])] = 0.25
    return out.clip(lower=0.25, upper=1.0)


def _map_period_proxy_to_multiplier(period_proxy: pd.Series, cutoffs: dict[str, float]) -> pd.Series:
    p = pd.to_numeric(period_proxy, errors="coerce").fillna(0.0)
    if (
        abs(float(cutoffs["period_p90"]) - float(cutoffs["period_p50"])) < 1e-12
        or float(p.std()) < 1e-12
    ):
        return pd.Series(1.0, index=p.index, dtype=float)
    out = pd.Series(1.0, index=p.index, dtype=float)
    out.loc[p >= float(cutoffs["period_p50"])] = 0.75
    out.loc[p >= float(cutoffs["period_p75"])] = 0.50
    out.loc[p >= float(cutoffs["period_p90"])] = 0.25
    return out.clip(lower=0.25, upper=1.0)


def _smooth_multiplier(ts: pd.Series, raw: pd.Series, restore_alpha: float = 0.33) -> pd.Series:
    order = pd.Series(pd.to_datetime(ts, utc=True)).sort_values().index
    vals = pd.to_numeric(raw, errors="coerce").fillna(1.0).clip(0.25, 1.0)
    smoothed = vals.copy()
    prev = 1.0
    for idx in order:
        value = float(vals.loc[idx])
        if value < prev:
            cur = value
        else:
            cur = prev + float(restore_alpha) * (value - prev)
        smoothed.loc[idx] = cur
        prev = cur
    return smoothed


def _period_proxy(features: pd.DataFrame) -> pd.Series:
    for col in (
        "period_component_rank__mean",
        "period_component_score__mean",
        "blend_B3_new_period_soft_qfail_rank__mean",
        "blend_B3_new_period_soft_qfail_score__mean",
    ):
        if col in features.columns:
            return pd.to_numeric(features[col], errors="coerce")
    return pd.Series(0.0, index=features.index, dtype=float)


def _apply_multiplier(candidates: pd.DataFrame, schedule: pd.DataFrame, *, scale_entries: bool, max_entries: int) -> pd.DataFrame:
    sched = schedule[["timestamp", "multiplier"]].copy()
    sched["timestamp"] = pd.to_datetime(sched["timestamp"], utc=True, errors="coerce")
    out = candidates.merge(sched, on="timestamp", how="left", validate="many_to_one")
    out["multiplier"] = pd.to_numeric(out["multiplier"], errors="coerce").fillna(1.0).clip(0.25, 1.0)
    out["portfolio_wallet_cap_multiplier"] = out["multiplier"]
    if scale_entries:
        out["portfolio_max_new_entries_per_bar"] = np.maximum(
            1,
            np.ceil(float(max_entries) * out["multiplier"]).astype(int),
        )
    return normalise_candidate_table(out.drop(columns=["multiplier"]))


def _worst_24h_net_pnl(accepted: pd.DataFrame) -> float:
    if accepted.empty:
        return 0.0
    work = accepted[["timestamp", "net_pnl"]].copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work = work.dropna(subset=["timestamp"]).sort_values("timestamp")
    if work.empty:
        return 0.0
    values: list[float] = []
    for ts in work["timestamp"].drop_duplicates().sort_values():
        start = ts - pd.Timedelta(hours=24)
        values.append(float(work.loc[(work["timestamp"] > start) & (work["timestamp"] <= ts), "net_pnl"].sum()))
    return float(min(values)) if values else 0.0


def _metrics_row(arm: str, metrics: dict[str, Any], schedule: pd.DataFrame, accepted: pd.DataFrame) -> dict[str, Any]:
    mult = pd.to_numeric(schedule["multiplier"], errors="coerce") if "multiplier" in schedule else pd.Series(dtype=float)
    gross = float(metrics.get("gross_pnl", 0.0) or 0.0)
    cost = float(gross - float(metrics.get("net_pnl", 0.0) or 0.0))
    return {
        "arm": arm,
        "trade_count": metrics.get("trade_count", 0),
        "net_pnl": metrics.get("net_pnl", 0.0),
        "gross_pnl": gross,
        "cost_pnl": cost,
        "cost_to_abs_gross": float(cost / max(abs(gross), 1e-9)),
        "notional_turnover": metrics.get("notional_turnover"),
        "mean_trade_notional": metrics.get("mean_trade_notional"),
        "compounded_return": metrics.get("compounded_return"),
        "max_drawdown": metrics.get("max_drawdown"),
        "worst_24h_net_pnl": _worst_24h_net_pnl(accepted),
        "full_sl_rate": metrics.get("full_sl_rate"),
        "timeout_rate": metrics.get("timeout_rate"),
        "avg_open_positions": metrics.get("avg_open_positions"),
        "mean_multiplier": float(mult.mean()) if len(mult) else 1.0,
        "min_multiplier": float(mult.min()) if len(mult) else 1.0,
        "p25_multiplier": float(mult.quantile(0.25)) if len(mult) else 1.0,
    }


def _render_report(summary: pd.DataFrame, manifest: dict[str, Any]) -> str:
    lines = [
        "# Global Portfolio Period Multiplier",
        "",
        f"Generated: {manifest['generated_at_utc']}",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Contract",
        "",
        "- One timestamp-level multiplier is applied portfolio-wide.",
        "- Scores, ranks, thresholds and auction ordering are unchanged.",
        "- The multiplier changes only the effective new-risk wallet cap; G6 also changes max_new_entries_per_bar.",
        "- Targets are generated from the unmodified frozen baseline replay.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-broad-candidates", type=Path, default=DEFAULT_TRAIN_BROAD)
    parser.add_argument("--train-deployable-candidates", type=Path, default=DEFAULT_TRAIN_DEPLOYABLE)
    parser.add_argument("--eval-candidates", type=Path, default=DEFAULT_EVAL_CANDIDATES)
    parser.add_argument("--eval-components", type=Path, default=DEFAULT_EVAL_COMPONENTS)
    parser.add_argument("--policy-manifest", type=Path, default=DEFAULT_POLICY_MANIFEST)
    parser.add_argument("--policy-variant", default="refit_bar4_strategy_bar2")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--horizon-hours", type=int, default=72)
    parser.add_argument("--max-feature-cols", type=int, default=96)
    parser.add_argument("--market-mode", default="perps")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    params, _ = _load_policy_params(args.policy_manifest, args.policy_variant)
    train_broad = _load_candidates(args.train_broad_candidates)
    train_deployable = _load_candidates(args.train_deployable_candidates)
    eval_candidates = _attach_eval_components(_load_candidates(args.eval_candidates), args.eval_components)

    ev_curve = fit_hierarchical_ev_curves(train_deployable)
    train_decisions, train_equity, train_metrics = replay_candidates(
        train_broad,
        params,
        mode="global_auction",
        ev_curve=ev_curve,
        market_mode=args.market_mode,
    )
    train_accepted = _accepted_trades(train_broad, train_decisions)
    eval_baseline_decisions, eval_baseline_equity, _ = replay_candidates(
        eval_candidates,
        params,
        mode="global_auction",
        ev_curve=ev_curve,
        market_mode=args.market_mode,
    )

    feature_cols_raw = _feature_columns(train_broad, max_cols=args.max_feature_cols)
    train_features = _timestamp_features(train_broad, feature_cols=feature_cols_raw, max_cols=args.max_feature_cols)
    train_fill_values = _timestamp_feature_fill_values(train_features)
    eval_features = _timestamp_features(
        eval_candidates,
        feature_cols=feature_cols_raw,
        max_cols=args.max_feature_cols,
        fill_values=train_fill_values,
    )
    train_features = _add_trailing_performance(train_features, train_accepted)
    train_features = _add_portfolio_state_features(train_features, train_equity)
    train_features = _add_open_position_concentration_features(train_features, train_accepted)
    eval_features = _add_trailing_performance(eval_features, train_accepted)
    eval_features = _add_portfolio_state_features(eval_features, eval_baseline_equity)
    eval_features = _add_open_position_concentration_features(eval_features, _accepted_trades(eval_candidates, eval_baseline_decisions))
    train_features["period_proxy"] = _period_proxy(train_features)
    eval_features["period_proxy"] = _period_proxy(eval_features)
    for col in train_features.columns:
        if col not in eval_features.columns:
            eval_features[col] = 0.0
    eval_features = eval_features[train_features.columns]
    labels = _forward_labels(train_features["timestamp"], train_accepted, int(args.horizon_hours))
    train_model_frame = train_features.merge(labels, on="timestamp", how="left")
    model_feature_cols = [
        col
        for col in train_features.columns
        if col != "timestamp" and pd.api.types.is_numeric_dtype(train_features[col])
    ]
    models, cutoffs, train_fit_frame = _fit_models(train_model_frame, model_feature_cols)
    eval_pred = _predict_models(models, eval_features, model_feature_cols)
    pred = eval_features[["timestamp", "period_proxy"]].merge(eval_pred, on="timestamp", how="left")

    schedules: dict[str, pd.DataFrame] = {}
    base_ts = eval_features[["timestamp"]].copy()
    schedules["G0_no_modifier"] = base_ts.assign(multiplier=1.0)
    schedules["G1_existing_new_period_global"] = base_ts.assign(
        multiplier=_map_period_proxy_to_multiplier(pred["period_proxy"], cutoffs).to_numpy(dtype=float)
    )
    schedules["G2_utility_lcb"] = base_ts.assign(
        multiplier=np.where(pd.to_numeric(pred["pred_utility_q10"], errors="coerce") < 0.0, 0.25, 1.0)
    )
    schedules["G3_adverse_risk"] = base_ts.assign(
        multiplier=_map_risk_to_multiplier(pred["pred_adverse_risk"], cutoffs).to_numpy(dtype=float)
    )
    g4_raw = _map_risk_to_multiplier(pred["combined_risk"], cutoffs)
    g4_raw = g4_raw.where(pd.to_numeric(pred["pred_utility_q10"], errors="coerce") >= 0.0, 0.25)
    schedules["G4_combined"] = base_ts.assign(multiplier=g4_raw.to_numpy(dtype=float))
    schedules["G5_combined_asymmetric_smoothing"] = base_ts.assign(
        multiplier=_smooth_multiplier(base_ts["timestamp"], g4_raw).to_numpy(dtype=float)
    )
    schedules["G6_G5_plus_entry_cap_scaling"] = schedules["G5_combined_asymmetric_smoothing"].copy()

    summary_rows: list[dict[str, Any]] = []
    decision_frames: list[pd.DataFrame] = []
    equity_frames: list[pd.DataFrame] = []
    accepted_frames: list[pd.DataFrame] = []
    for arm, schedule in schedules.items():
        candidate_arm = _apply_multiplier(
            eval_candidates,
            schedule,
            scale_entries=arm == "G6_G5_plus_entry_cap_scaling",
            max_entries=int(params.max_new_entries_per_bar),
        )
        decisions, equity, metrics = replay_candidates(
            candidate_arm,
            params,
            mode="global_auction",
            ev_curve=ev_curve,
            market_mode=args.market_mode,
        )
        decisions["arm"] = arm
        equity["arm"] = arm
        accepted = _accepted_trades(candidate_arm, decisions)
        accepted["arm"] = arm
        summary_rows.append(_metrics_row(arm, metrics, schedule, accepted))
        decision_frames.append(decisions)
        equity_frames.append(equity)
        accepted_frames.append(accepted)
        schedule.to_csv(args.output_dir / f"{arm}_schedule.csv", index=False)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "global_period_multiplier_summary.csv", index=False)
    pd.concat(decision_frames, ignore_index=True).to_parquet(args.output_dir / "decisions.parquet", index=False)
    pd.concat(equity_frames, ignore_index=True).to_parquet(args.output_dir / "equity_curves.parquet", index=False)
    pd.concat(accepted_frames, ignore_index=True).to_parquet(args.output_dir / "accepted_trades.parquet", index=False)
    eval_pred.to_csv(args.output_dir / "eval_period_model_predictions.csv", index=False)
    train_model_frame.to_csv(args.output_dir / "train_timestamp_targets.csv", index=False)
    pd.Series(model_feature_cols, name="feature").to_csv(args.output_dir / "model_features.csv", index=False)

    manifest = {
        "generated_by": "run_global_portfolio_period_multiplier",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_broad_candidates": str(args.train_broad_candidates),
        "train_deployable_candidates": str(args.train_deployable_candidates),
        "eval_candidates": str(args.eval_candidates),
        "eval_components": str(args.eval_components),
        "policy_manifest": str(args.policy_manifest),
        "policy_variant": str(args.policy_variant),
        "policy_params": asdict(params),
        "horizon_hours": int(args.horizon_hours),
        "train_baseline_metrics": train_metrics,
        "target_cutoffs": cutoffs,
        "train_timestamp_rows": int(len(train_model_frame)),
        "train_labeled_rows": int(train_fit_frame.shape[0]),
        "model_feature_count": int(len(model_feature_cols)),
        "contract": {
            "one_global_timestamp_multiplier": True,
            "changes_ranking_or_thresholds": False,
            "changes_new_risk_wallet_cap": True,
            "G6_also_scales_max_new_entries_per_bar": True,
        },
        "outputs": {
            "summary": str(args.output_dir / "global_period_multiplier_summary.csv"),
            "accepted_trades": str(args.output_dir / "accepted_trades.parquet"),
            "predictions": str(args.output_dir / "eval_period_model_predictions.csv"),
            "train_targets": str(args.output_dir / "train_timestamp_targets.csv"),
            "features": str(args.output_dir / "model_features.csv"),
            "report": str(args.output_dir / "global_portfolio_period_multiplier_report.md"),
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2) + "\n")
    (args.output_dir / "global_portfolio_period_multiplier_report.md").write_text(
        _render_report(summary, manifest),
        encoding="utf-8",
    )
    print(json.dumps(_json_safe(manifest), indent=2)[:6000])
    print(f"\nWrote {args.output_dir}")


if __name__ == "__main__":
    main()
