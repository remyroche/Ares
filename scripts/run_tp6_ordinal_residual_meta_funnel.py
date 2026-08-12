#!/usr/bin/env python3
"""Strict chronological M0--M4 ordinal residual-meta funnel for TP6/SL4.

The base is frozen same-side OOF.  Meta targets are material corrections to
that base in net bps, never absolute execution-success labels.  M0--M4 share
the identical candidate cohort, causal inputs, entry convention and pooled
global ranking; only the residual target changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.ordinal_residual_meta import (  # noqa: E402
    TOP_FRACTIONS, classifier_diagnostics, cumulative_to_simplex,
    fit_residual_class_map, fit_soft_binary_residual_scale, ordinal_labels, policy_training_mask,
    reconstruct_expected_residual, residual_bps, sample_weights,
    soft_binary_residual_labels,
)
from scripts.run_tp6_shared_residual_d0_d4 import (  # noqa: E402
    D1 as PREQUENTIAL_RELATIONSHIP_TRUST,
    D2 as PREQUENTIAL_OOD_TRUST,
    D3 as PREQUENTIAL_FAILURE_TRUST,
    weekly_prequential_trust,
)


INPUT = ROOT / "data_perp/artifacts/tp6_m6_shared_regime_residual_features_20260809_v3/shared_regime_residual_ledger.parquet"
PANEL = ROOT / "data_perp/artifacts/full_universe_t2_t4_panel_20260801_v3/parts"
OUT = ROOT / "data_perp/artifacts/tp6_ordinal_residual_meta_funnel_20260803_v2"
CACHE = ROOT / "data_perp/artifacts/tp6_ordinal_residual_meta_input_20260803_v2/ordinal_residual_input.parquet"
TRUST_CACHE = CACHE.with_name("prequential_trust.parquet")
TRUST_AUDIT = CACHE.with_name("prequential_trust_audit.parquet")
ERAS = ("2023-07_08", "2023-09_10", "2023-11_12", "2024-01_02", "2024-05_06", "2024-07_08", "2024-09_10", "2024-11")
CONTEXT = (
    "mkt_ret_eq_24h", "regime_liquidity_score", "mkt_rv_ratio_1h_24h",
    "mkt_oi_chg_z_24h", "mkt_funding_dispersion", "cross_asset_corr_4h",
    "mkt_systemic_deleveraging_score", "mkt_flush_exhaustion_score",
    "post_liquidation_rebound_score", "negative_breadth_pct",
    "btc_resilience_alt_weakness", "short_covering_score_market",
    "deleveraging_without_followthrough", "short_signal_recovery_conflict",
)
PRICE_LEVERAGE = ("price_x_oi_1d", "price_x_oi_3d", "price_x_oi_7d", "volume_price_corr_ts_resid", "atr_percentile", "ob_spread_z_24h")
CAUSAL_REGIME = (
    "soft_regime_prior_residual_bps", "soft_regime_prior_residual_scale_bps",
    "regime_p_calm", "regime_p_trend", "regime_p_stress", "regime_p_transition",
    "regime_entropy", "regime_transition_onset_proxy", "regime_state_duration_hours",
)
CAUSAL_TRUST = (*PREQUENTIAL_RELATIONSHIP_TRUST, *PREQUENTIAL_OOD_TRUST, *PREQUENTIAL_FAILURE_TRUST)
MANDATORY = (
    "p_adverse", "p_weak", "p_clear", "prequential_base_expected_net_bps",
    "base_entropy", "base_top2_margin", "base_side_rank", "cost_to_atr",
    *CAUSAL_REGIME, *CAUSAL_TRUST,
)
DEFAULT_ARMS = ("BASE", "M0_huber_residual", "M1_binary_residual_positive", "M2_binary_downgrade", "M3_ordinal_t100", "M4_ordinal_t50", "M4_ordinal_t150")
DELAY = pd.Timedelta(hours=13)
PARAMS: dict[str, Any] = dict(n_estimators=100, learning_rate=.04, num_leaves=20, min_child_samples=500, colsample_bytree=.8, subsample=.8, reg_lambda=15., random_state=20260803, n_jobs=1, verbosity=-1)


def _ordinal_threshold(arm: str) -> float | None:
    """Return a declared ordinal residual boundary, if this is an ordinal arm.

    The historical M3/M4 names remain supported.  The generic form makes a
    sequential threshold comparison (for example 50 then 75 bps) explicit
    without silently relabelling an existing experiment.
    """
    if arm == "M3_ordinal_t100":
        return 100.0
    if arm == "M4_ordinal_t50":
        return 50.0
    if arm == "M4_ordinal_t150":
        return 150.0
    if arm.startswith("ORDINAL_t"):
        try:
            threshold = float(arm.removeprefix("ORDINAL_t"))
        except ValueError as exc:
            raise ValueError(f"invalid ordinal arm name: {arm}") from exc
        if threshold <= 0:
            raise ValueError(f"ordinal threshold must be positive: {arm}")
        return threshold
    return None


def _soft_binary_spec(arm: str) -> tuple[str, float, float | None] | None:
    """Parse an explicit soft residual target name without hidden defaults."""
    token_arm = arm.removeprefix("SOFTLOG_") if arm.startswith("SOFTLOG_") else arm.removeprefix("SOFT_") if arm.startswith("SOFT_") else None
    if token_arm is None:
        return None
    if token_arm.startswith("Q"):
        token = token_arm.removeprefix("Q")
        try:
            lower, upper = token.split("_")
            lower_q, upper_q = float(lower), float(upper)
        except ValueError as exc:
            raise ValueError(f"invalid soft-percentile arm name: {arm}") from exc
        if not 0.0 < lower_q < upper_q < 100.0:
            raise ValueError(f"invalid soft-percentile arm name: {arm}")
        return "quantile", lower_q, upper_q
    if token_arm.startswith("E"):
        try:
            extrema = float(token_arm.removeprefix("E"))
        except ValueError as exc:
            raise ValueError(f"invalid soft-extrema arm name: {arm}") from exc
        if extrema <= 0:
            raise ValueError(f"invalid soft-extrema arm name: {arm}")
        return "extrema", extrema, None
    raise ValueError(f"invalid soft target arm name: {arm}")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b): h.update(b)
    return h.hexdigest()


def _rank(frame: pd.DataFrame) -> pd.Series:
    """Decision-time side-local rank; never a rank over future candidates."""
    out = pd.Series(index=frame.index, dtype=float)
    for _key, part in frame.groupby(["__ts__", "side_name"], sort=True):
        ordered = part.sort_values(["prequential_base_expected_net_bps", "candidate_key"], ascending=[True, True], kind="stable")
        out.loc[ordered.index] = 1. - np.arange(len(ordered), dtype=float) / max(len(ordered) - 1, 1)
    return out


def _read() -> pd.DataFrame:
    # The source ledger carries many legacy diagnostic columns.  Materialise a
    # narrow, exact-ID join once through DuckDB: it streams the join to disk,
    # avoiding pandas holding source, panel and merged copies simultaneously.
    ledger_columns = [
        "candidate_id", "__ts__", "side_name", "gross_bps", "net_bps", "era",
        "shared_regime_contract_complete", "p_adverse", "p_weak", "p_clear",
        "prequential_base_expected_net_bps", *CAUSAL_REGIME, *CONTEXT,
    ]
    if not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        tmp = CACHE.with_suffix(".partial.parquet")
        if tmp.exists():
            tmp.unlink()
        select_ledger = ", ".join(f'l."{name}"' for name in ledger_columns)
        select_panel = ", ".join(f'p."{name}"' for name in ("atr_1h", "decision_price", "assumed_round_trip_cost_bps", *PRICE_LEVERAGE))
        sql = (
            f"COPY (SELECT hash(l.candidate_id) AS candidate_key, {select_ledger}, {select_panel} "
            f"FROM read_parquet('{INPUT.as_posix()}') AS l "
            f"INNER JOIN read_parquet('{(PANEL / '*.parquet').as_posix()}') AS p USING (candidate_id)) "
            f"TO '{tmp.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        con = duckdb.connect(config={"threads": "2", "memory_limit": "512MB", "temp_directory": "/tmp"})
        try:
            con.execute(sql)
        finally:
            con.close()
        tmp.replace(CACHE)
    x = pd.read_parquet(CACHE)
    x["side_name"] = x["side_name"].astype("category")
    x["era"] = x["era"].astype("category")
    x = x[x.shared_regime_contract_complete.astype(bool)].copy()
    x["__ts__"] = pd.to_datetime(x["__ts__"], utc=True, errors="raise")
    x["label_available_ts"] = x["__ts__"] + DELAY
    if x.candidate_key.duplicated().any() or not set(x.side_name.unique()).issubset({"long", "short"}):
        raise ValueError("invalid same-side OOF ledger")
    if not np.allclose(x.gross_bps.to_numpy(float) - x.net_bps.to_numpy(float), 100., atol=.02):
        raise ValueError("TP6/SL4 cost is not exactly 100 bps")
    atr_bps = x.atr_1h.abs().to_numpy(float) / x.decision_price.abs().to_numpy(float) * 1e4
    cost = x.assumed_round_trip_cost_bps.to_numpy(float)
    if not np.isfinite(atr_bps).all() or (atr_bps <= 0).any() or not np.isfinite(cost).all():
        raise ValueError("invalid causal entry ATR/cost")
    x["cost_to_atr"] = np.clip(cost / atr_bps, 0., 100.).astype(np.float32)
    x["side_is_long"] = x.side_name.eq("long").astype(np.float32)
    p = x[["p_adverse", "p_weak", "p_clear"]].to_numpy(float)
    if not np.isfinite(p).all() or (p < 0).any(): raise ValueError("invalid base probability input")
    p /= p.sum(axis=1, keepdims=True)
    x[["p_adverse", "p_weak", "p_clear"]] = p
    x["base_entropy"] = -(p * np.log(np.maximum(p, 1e-12))).sum(axis=1)
    q = np.sort(p, axis=1); x["base_top2_margin"] = q[:, -1] - q[:, -2]
    x["residual_target_valid"] = np.isfinite(x["net_bps"].to_numpy(float)) & np.isfinite(x["prequential_base_expected_net_bps"].to_numpy(float))
    x["base_side_rank"] = np.nan
    valid_index = x.index[x["residual_target_valid"]]
    x.loc[valid_index, "base_side_rank"] = _rank(x.loc[valid_index])
    # Trust is deliberately prequential.  Relationship/OOD references use
    # earlier decision-time rows; weekly health uses only previously resolved
    # outcomes.  The helper supplies neutral warm-up values and a support
    # field rather than leaking a future estimate into early rows.
    if TRUST_CACHE.exists():
        trust = pd.read_parquet(TRUST_CACHE)
        required = {"candidate_key", *CAUSAL_TRUST}
        if not required.issubset(trust.columns) or trust.candidate_key.duplicated().any():
            raise ValueError("invalid cached causal trust contract")
        x = x.join(trust.set_index("candidate_key").loc[:, CAUSAL_TRUST], on="candidate_key", validate="one_to_one")
        trust_audit = pd.read_parquet(TRUST_AUDIT) if TRUST_AUDIT.exists() else pd.DataFrame()
    else:
        trust_inputs = (
            "side_is_long", "p_adverse", "p_weak", "p_clear",
            "prequential_base_expected_net_bps", "regime_entropy",
            "regime_p_calm", "regime_p_trend", "regime_p_stress", "regime_p_transition", "regime_transition_onset_proxy",
            *CONTEXT,
        )
        trust_ready = np.isfinite(x.loc[:, trust_inputs].to_numpy(float)).all(axis=1)
        # Rows without a complete *decision-time* trust vector remain in the
        # meta population, but receive an explicit neutral/no-support state.
        # They must not enter the reference fit with a future-derived fill.
        trust = pd.DataFrame(index=x.index, columns=CAUSAL_TRUST, dtype=float)
        trust.loc[:, [*PREQUENTIAL_RELATIONSHIP_TRUST, *PREQUENTIAL_OOD_TRUST]] = 0.0
        trust.loc[:, "trust_active_failure_probability"] = 0.5
        trust.loc[:, "trust_active_failure_support_weeks"] = 0.0
        if trust_ready.any():
            fitted_trust, trust_audit = weekly_prequential_trust(x.loc[trust_ready].copy())
            trust.loc[fitted_trust.index, CAUSAL_TRUST] = fitted_trust.loc[:, CAUSAL_TRUST]
        else:
            trust_audit = pd.DataFrame()
        materialized = x.loc[:, ["candidate_key"]].join(trust)
        TRUST_CACHE.parent.mkdir(parents=True, exist_ok=True)
        materialized.to_parquet(TRUST_CACHE, index=False, compression="zstd")
        trust_audit.to_parquet(TRUST_AUDIT, index=False, compression="zstd")
        x = x.join(trust)
    if x.loc[:, CAUSAL_TRUST].isna().any().any():
        raise ValueError("causal trust materialisation left missing values")
    value = x.loc[:, [*MANDATORY, *CONTEXT, *PRICE_LEVERAGE]].replace([np.inf, -np.inf], np.nan)
    coverage = 1. - value.isna().mean()
    if (coverage < .90).any(): raise ValueError(f"feature coverage below 90%: {coverage[coverage < .90].to_dict()}")
    # Imputation is deliberately performed inside each chronological fit below.
    # Filling here with future-fold medians would contaminate the feature contract.
    x.loc[:, value.columns] = value.astype(np.float32)
    # Missing base outputs are an incomplete residual contract, never an
    # economic loss.  They remain visible in the source cache but cannot train
    # or score a residual correction.
    x = x[x["residual_target_valid"]].copy()
    x.attrs["trust_audit"] = trust_audit
    return x.sort_values(["__ts__", "candidate_key"], kind="stable").reset_index(drop=True)


def _optional_features(train: pd.DataFrame, target: np.ndarray, cap: int = 8) -> list[str]:
    """Training-only side/head selection; mandatory base state is never dropped."""
    scores = []
    for field in (*CONTEXT, *PRICE_LEVERAGE):
        v = train[field].to_numpy(float)
        valid = np.isfinite(v) & np.isfinite(target)
        corr = spearmanr(v[valid], target[valid]).statistic if valid.sum() > 10 and np.std(v[valid]) > 1e-12 else 0.
        scores.append((abs(float(corr)) if np.isfinite(corr) else 0., field))
    return list(MANDATORY) + [field for _score, field in sorted(scores, key=lambda z: (-z[0], z[1]))[:cap]]


def _matrices(train: pd.DataFrame, score: pd.DataFrame, fields: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Training-only median imputation, retained for both calibration and scoring."""
    source = train.loc[:, fields].replace([np.inf, -np.inf], np.nan)
    median = source.median(axis=0).fillna(0.0)
    x_train = source.fillna(median).to_numpy(np.float32)
    x_score = score.loc[:, fields].replace([np.inf, -np.inf], np.nan).fillna(median).to_numpy(np.float32)
    return x_train, x_score


def _binary(train: pd.DataFrame, score: pd.DataFrame, y: np.ndarray, weight: np.ndarray, fields: Sequence[str]) -> np.ndarray:
    x_train, x_score = _matrices(train, score, fields)
    model = lgb.LGBMClassifier(objective="binary", **PARAMS).fit(x_train, y, sample_weight=weight)
    return np.clip(model.predict_proba(x_score)[:, 1], 1e-5, 1.-1e-5)


def _huber(train: pd.DataFrame, score: pd.DataFrame, y: np.ndarray, weight: np.ndarray, fields: Sequence[str]) -> np.ndarray:
    x_train, x_score = _matrices(train, score, fields)
    return lgb.LGBMRegressor(objective="huber", alpha=.9, **PARAMS).fit(x_train, y, sample_weight=weight).predict(x_score)


def _soft_binary(train: pd.DataFrame, score: pd.DataFrame, y: np.ndarray, weight: np.ndarray, fields: Sequence[str]) -> np.ndarray:
    """Regress a fractional confidence-calibration target, not a hard class."""
    x_train, x_score = _matrices(train, score, fields)
    return lgb.LGBMRegressor(objective="regression_l2", **PARAMS).fit(x_train, y, sample_weight=weight).predict(x_score)


def _fractional_logistic_objective(y_true: np.ndarray, raw_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bernoulli deviance for a fractional soft label, evaluated on logits."""
    probability = expit(np.asarray(raw_score, dtype=float))
    gradient = probability - np.asarray(y_true, dtype=float)
    hessian = np.maximum(probability * (1.0 - probability), 1e-6)
    return gradient, hessian


def _soft_binary_logistic(train: pd.DataFrame, score: pd.DataFrame, y: np.ndarray, weight: np.ndarray, fields: Sequence[str]) -> np.ndarray:
    x_train, x_score = _matrices(train, score, fields)
    model = lgb.LGBMRegressor(objective=_fractional_logistic_objective, **PARAMS).fit(x_train, y, sample_weight=weight)
    # LightGBM returns raw margins for a custom objective, so the calibrator
    # receives a genuine probability-like prediction for both fit and score.
    return expit(model.predict(x_score))


def _fit_isotonic(raw: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    return IsotonicRegression(out_of_bounds="clip").fit(raw, y)


def _label_certainty(residual: np.ndarray, arm: str, threshold: float = 100.) -> np.ndarray:
    """Training-only distance from the selected class boundary, in [0.5, 1]."""
    if arm == "M1_binary_residual_positive":
        distance = np.abs(residual)
    elif arm == "M2_binary_downgrade":
        distance = np.abs(residual + threshold)
    else:
        distance = np.minimum(np.abs(residual + threshold), np.abs(residual - threshold))
    return .5 + .5 * np.tanh(distance / 100.)


def _binary_means(train: pd.DataFrame, positive: np.ndarray) -> dict[str, tuple[float, float]]:
    r = residual_bps(train)
    result = {}
    global_mean = (float(r[~positive].mean()) if (~positive).any() else float(r.mean()), float(r[positive].mean()) if positive.any() else float(r.mean()))
    for side in ("long", "short"):
        mask = train.side_name.eq(side).to_numpy()
        values=[]
        for event, fallback in zip((False, True), global_mean):
            keep = mask & (positive == event); n=int(keep.sum())
            raw=float(r[keep].mean()) if n else fallback; shrink=n/(n+500.)
            values.append(shrink*raw+(1-shrink)*fallback)
        result[side]=tuple(values)
    return result


def _predict_arm(train: pd.DataFrame, evaluation: pd.DataFrame, arm: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Early fit -> chronological calibration -> full prior fit, per side."""
    result = np.empty(len(evaluation), dtype=np.float32); details: dict[str, Any] = {"arm": arm, "sides": {}}
    for side in ("long", "short"):
        full = train[train.side_name.eq(side)].copy(); ev = evaluation[evaluation.side_name.eq(side)].copy()
        if ev.empty: continue
        cut = full.__ts__.quantile(.80)
        cal = full[full.__ts__.ge(cut)].copy()
        # A calibration row must never be predicted by a model whose training
        # labels extend into the calibration decision period.
        early = full[full.label_available_ts.lt(cal.__ts__.min())].copy()
        if min(len(early), len(cal), len(ev)) < 1000: raise ValueError(f"insufficient {side} chronological calibration support")
        res_full, res_early = residual_bps(full), residual_bps(early)
        mask_full = policy_training_mask(full, candidate_column="candidate_key"); mask_early = policy_training_mask(early, candidate_column="candidate_key")
        full, early = full.loc[mask_full].copy(), early.loc[mask_early].copy()
        res_full, res_early = residual_bps(full), residual_bps(early)
        ordinal_threshold = _ordinal_threshold(arm)
        soft_spec = _soft_binary_spec(arm)
        threshold = (
            ordinal_threshold if ordinal_threshold is not None
            else 100.0
        )
        if arm == "M0_huber_residual":
            f_early=_optional_features(early,res_early); f_full=_optional_features(full,res_full)
            w_early=sample_weights(early, ordinal_labels(res_early, 100.), residual=res_early, certainty=_label_certainty(res_early, arm))
            w_full=sample_weights(full, ordinal_labels(res_full, 100.), residual=res_full, certainty=_label_certainty(res_full, arm))
            raw_cal=_huber(early,cal,res_early,w_early,f_early); iso=_fit_isotonic(raw_cal,residual_bps(cal)); raw=_huber(full,ev,res_full,w_full,f_full); correction=iso.predict(raw)
            diag={"features":f_full,"mode":"direct_huber"}
        elif arm in {"M1_binary_residual_positive","M2_binary_downgrade"}:
            positive = res_full > 0 if arm == "M1_binary_residual_positive" else res_full <= -threshold
            positive_early = res_early > 0 if arm == "M1_binary_residual_positive" else res_early <= -threshold
            f_early=_optional_features(early,positive_early); f_full=_optional_features(full,positive)
            w_early=sample_weights(early,positive_early.astype(int),residual=res_early,certainty=_label_certainty(res_early, arm, threshold)); w_full=sample_weights(full,positive.astype(int),residual=res_full,certainty=_label_certainty(res_full, arm, threshold))
            raw_cal=_binary(early,cal,positive_early.astype(int),w_early,f_early); iso=_fit_isotonic(raw_cal,positive.astype(int)[:0] if False else (residual_bps(cal)>0 if arm=="M1_binary_residual_positive" else residual_bps(cal)<=-threshold).astype(int)); raw=_binary(full,ev,positive.astype(int),w_full,f_full); p=iso.predict(raw)
            means=_binary_means(full,positive)
            if arm == "M1_binary_residual_positive": correction=(1-p)*means[side][0]+p*means[side][1]
            else: correction=p*means[side][1]+(1-p)*means[side][0]
            diag={"features":f_full,"mode":"binary","threshold":threshold,"calibration_brier":float(np.mean((p-(residual_bps(ev)>0 if arm=="M1_binary_residual_positive" else residual_bps(ev)<=-threshold))**2))}
        elif soft_spec is not None:
            kind, first, second = soft_spec
            if kind == "quantile":
                lower_early, upper_early = fit_soft_binary_residual_scale(
                    res_early, lower_percentile=first, upper_percentile=second
                )
                lower_full, upper_full = fit_soft_binary_residual_scale(
                    res_full, lower_percentile=first, upper_percentile=second
                )
            else:
                lower_early, upper_early = fit_soft_binary_residual_scale(res_early, extrema_bps=first)
                lower_full, upper_full = fit_soft_binary_residual_scale(res_full, extrema_bps=first)
            y_early = soft_binary_residual_labels(res_early, lower_bps=lower_early, upper_bps=upper_early)
            y_full = soft_binary_residual_labels(res_full, lower_bps=lower_full, upper_bps=upper_full)
            y_cal = soft_binary_residual_labels(
                residual_bps(cal), lower_bps=lower_early, upper_bps=upper_early
            )
            f_early = _optional_features(early, y_early)
            f_full = _optional_features(full, y_full)
            c_early = .5 + np.abs(y_early - .5)
            c_full = .5 + np.abs(y_full - .5)
            w_early = sample_weights(early, ordinal_labels(res_early, 100.), residual=res_early, certainty=c_early)
            w_full = sample_weights(full, ordinal_labels(res_full, 100.), residual=res_full, certainty=c_full)
            fit_soft = _soft_binary_logistic if arm.startswith("SOFTLOG_") else _soft_binary
            raw_cal = fit_soft(early, cal, y_early, w_early, f_early)
            iso = _fit_isotonic(raw_cal, y_cal)
            raw = fit_soft(full, ev, y_full, w_full, f_full)
            probability = np.clip(iso.predict(raw), 0., 1.)
            # This map is fit only on earlier resolved rows.  It converts the
            # soft calibration output back to common economic bps for global
            # long/short ranking, rather than adding probability directly.
            residual_map = _fit_isotonic(y_full, res_full)
            correction = residual_map.predict(probability)
            diag = {
                "features": f_full, "mode": "soft_binary_fractional_logistic" if arm.startswith("SOFTLOG_") else "soft_binary_residual_l2",
                "scale_kind": kind, "scale_early_bps": [lower_early, upper_early],
                "scale_full_bps": [lower_full, upper_full],
                "calibration_label_mae": float(np.mean(np.abs(iso.predict(raw_cal) - y_cal))),
                "test_label_mean": float(soft_binary_residual_labels(residual_bps(ev), lower_bps=lower_full, upper_bps=upper_full).mean()),
                "prediction_mean": float(probability.mean()),
            }
        else:
            label_full=ordinal_labels(res_full,threshold); label_early=ordinal_labels(res_early,threshold)
            f_low=_optional_features(early,(res_early>-threshold).astype(int)); f_high=_optional_features(early,(res_early>threshold).astype(int)); f_full=list(dict.fromkeys(f_low+f_high))
            w_early=sample_weights(early,label_early,residual=res_early,certainty=_label_certainty(res_early, arm, threshold)); w_full=sample_weights(full,label_full,residual=res_full,certainty=_label_certainty(res_full, arm, threshold))
            raw_low_cal=_binary(early,cal,(res_early>-threshold).astype(int),w_early,f_low); raw_hi_cal=_binary(early,cal,(res_early>threshold).astype(int),w_early,f_high)
            iso_low=_fit_isotonic(raw_low_cal,(residual_bps(cal)>-threshold).astype(int)); iso_hi=_fit_isotonic(raw_hi_cal,(residual_bps(cal)>threshold).astype(int))
            p=_predict_ordinal(full,ev,threshold,w_full,f_low,f_high,iso_low,iso_hi)
            mapping=fit_residual_class_map(full,threshold_bps=threshold); correction=reconstruct_expected_residual(p,ev.side_name,mapping)
            diag={"features_low":f_low,"features_high":f_high,"mode":"cumulative_ordinal","threshold":threshold,**classifier_diagnostics(ordinal_labels(residual_bps(ev),threshold),p)}
        position = evaluation.index.get_indexer(ev.index)
        if (position < 0).any():
            raise ValueError("evaluation side index is not aligned")
        result[position] = correction
        details["sides"][side]=diag
    return result, details


def _predict_ordinal(train: pd.DataFrame, ev: pd.DataFrame, threshold: float, weight: np.ndarray, low_fields: Sequence[str], high_fields: Sequence[str], iso_low: IsotonicRegression, iso_high: IsotonicRegression) -> np.ndarray:
    r=residual_bps(train)
    low=_binary(train,ev,(r>-threshold).astype(int),weight,low_fields)
    high=_binary(train,ev,(r>threshold).astype(int),weight,high_fields)
    return cumulative_to_simplex(iso_low.predict(low),iso_high.predict(high))


def _metric_rows(test: pd.DataFrame, score: np.ndarray, arm: str, split: str) -> list[dict[str, Any]]:
    z=test.copy(); z["score_bps"]=score; base=z.prequential_base_expected_net_bps.to_numpy(float); rows=[]
    for top in TOP_FRACTIONS:
        n=max(1,int(np.ceil(len(z)*top))); take=z.sort_values(["score_bps","candidate_key"],ascending=[False,True],kind="stable").head(n); base_take=z.assign(score_bps=base).sort_values(["score_bps","candidate_key"],ascending=[False,True],kind="stable").head(n)
        for view, q in (("global",take),("long",take[take.side_name.eq("long")]),("short",take[take.side_name.eq("short")])):
            if q.empty: continue
            rows.append({"arm":arm,"split":split,"scope":"pooled_global_after_common_bps","view":view,"top_fraction":top,"n":len(q),"net_bps":float(q.net_bps.mean()),"gross_bps":float(q.gross_bps.mean()),"net_rank_ic":float(spearmanr(z.score_bps,z.net_bps).statistic),"tail_overestimation_bps":float((q.score_bps-q.net_bps).mean()),"promoted_net_bps":float(take[~take.candidate_key.isin(base_take.candidate_key)].net_bps.mean()) if (~take.candidate_key.isin(base_take.candidate_key)).any() else np.nan,"demoted_net_bps":float(base_take[~base_take.candidate_key.isin(take.candidate_key)].net_bps.mean()) if (~base_take.candidate_key.isin(take.candidate_key)).any() else np.nan})
    return rows


def _run_cell(data: pd.DataFrame, train_eras: Sequence[str], test_eras: Sequence[str], split: str, arms: Sequence[str]) -> tuple[list[dict[str,Any]],list[dict[str,Any]]]:
    test=data[data.era.isin(test_eras)].copy(); start=test.__ts__.min(); train=data[data.era.isin(train_eras)&data.label_available_ts.lt(start)].copy()
    if train.empty or test.empty: raise ValueError(f"empty {split}")
    metrics=[]; diagnostics=[]
    for arm in arms:
        correction, detail = (np.zeros(len(test), dtype=np.float32), {}) if arm == "BASE" else _predict_arm(train, test, arm)
        score=test.prequential_base_expected_net_bps.to_numpy(float)+correction
        metrics.extend(_metric_rows(test,score,arm,split))
        if arm != "BASE": diagnostics.append({"split":split,"arm":arm,"train_rows":len(train),"test_rows":len(test),"details":json.dumps(detail,default=str)})
    return metrics,diagnostics


def run(out: Path=OUT, *, arms: Sequence[str]=DEFAULT_ARMS) -> Path:
    if not arms or arms[0] != "BASE":
        raise ValueError("arms must start with the BASE control")
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps({"schema":"tp6_ordinal_residual_meta_funnel_v1", "status":"RUNNING"}, indent=2) + "\n")
    data = _read()
    if TRUST_AUDIT.exists():
        pd.read_parquet(TRUST_AUDIT).to_parquet(out / "causal_trust_audit.parquet", index=False)
    metric: list[dict[str, Any]] = []
    diag: list[dict[str, Any]] = []
    # The first ledger era has no frozen base output, so the first strict outer
    # cell starts once an earlier resolved, valid-base era exists.
    cells = [(f"outer_{era}", ERAS[:i], (era,)) for i, era in enumerate(ERAS[2:],2)]
    cells += [("transport_2023q4_to_2024", ERAS[1:3], ERAS[3:5]), ("transport_2024h1_to_h2", ERAS[3:5], ERAS[5:])]
    for split, train, test in cells:
        checkpoint = out / f"{split}_checkpoint.parquet"
        diagnostic_checkpoint = out / f"{split}_diagnostics.parquet"
        if checkpoint.exists() and diagnostic_checkpoint.exists():
            metric.extend(pd.read_parquet(checkpoint).to_dict("records")); diag.extend(pd.read_parquet(diagnostic_checkpoint).to_dict("records")); continue
        a,b = _run_cell(data, train, test, split, arms)
        pd.DataFrame(a).to_parquet(checkpoint, index=False)
        pd.DataFrame(b).to_parquet(diagnostic_checkpoint, index=False)
        metric += a; diag += b
    pd.DataFrame(metric).to_parquet(out/'metrics.parquet',index=False);pd.DataFrame(diag).to_parquet(out/'diagnostics.parquet',index=False)
    m=pd.DataFrame(metric); gate=[]
    for split,q in m[(m.view=='global')&(m.top_fraction.isin((.05,.10)))].groupby('split'):
        base=q[q.arm=='BASE'].set_index('top_fraction').net_bps
        for arm,x in q[q.arm!='BASE'].groupby('arm'):
            gate.append({"split":split,"arm":arm,"top5_uplift":float(x.set_index('top_fraction').loc[.05,'net_bps']-base.loc[.05]),"top10_uplift":float(x.set_index('top_fraction').loc[.10,'net_bps']-base.loc[.10])})
    g=pd.DataFrame(gate); transport_g = g[g.split.str.startswith('transport_')].copy(); summary=transport_g.groupby('arm').agg(min_top5=('top5_uplift','min'),min_top10=('top10_uplift','min')).reset_index();summary['advances']=(summary.min_top5>0)&(summary.min_top10>0);g.to_parquet(out/'transport_gates.parquet',index=False);summary.to_parquet(out/'advancement.parquet',index=False)
    manifest={"schema":"tp6_ordinal_residual_meta_funnel_v1","status":"COMPLETED","input":str(INPUT),"geometry":"TP6/SL4/H12","cost_bps":100.,"arms":list(arms),"target":"realised net bps - same-side prequential base expected net bps","training_population":"side base top30% plus deterministic 10% lower-rank control","mapping":"side x class residual means shrunk to global prior-resolved class means","ranking":"global top-k after common-bps reconstruction","features":{"mandatory":list(MANDATORY),"context_candidates":list(CONTEXT+PRICE_LEVERAGE),"causal_regime":list(CAUSAL_REGIME),"causal_trust":list(CAUSAL_TRUST)},"trust_lineage":"relationship/OOD references are prior decision-time rows; active-failure trust uses prior-resolved weekly blocks only; warmup is neutral with explicit support","promotion":"top5 and top10 net uplift must both be positive in both transport splits"}
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n');return out


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,default=OUT)
    p.add_argument(
        '--ordinal-thresholds', type=float, nargs='+',
        help='Run only BASE plus generic ordinal residual arms at these thresholds.',
    )
    p.add_argument(
        '--soft-percentile-clips', type=float, nargs='+',
        help='Add zero-centred SOFT_Qxx_yy arms, e.g. 5 10 15 gives p05/95, p10/90 and p15/85.',
    )
    p.add_argument(
        '--soft-fixed-extrema', type=float, nargs='+',
        help='Add zero-centred SOFT_Exx residual arms with symmetric bps extrema.',
    )
    p.add_argument(
        '--soft-logistic-percentile-clips', type=float, nargs='+',
        help='Add SOFTLOG_Qxx_yy arms using fractional Bernoulli/logistic loss.',
    )
    p.add_argument(
        '--soft-logistic-fixed-extrema', type=float, nargs='+',
        help='Add SOFTLOG_Exx arms using fractional Bernoulli/logistic loss.',
    )
    a=p.parse_args()
    if all(value is None for value in (a.ordinal_thresholds, a.soft_percentile_clips, a.soft_fixed_extrema, a.soft_logistic_percentile_clips, a.soft_logistic_fixed_extrema)):
        arms = DEFAULT_ARMS
    else:
        ordinal = a.ordinal_thresholds or ()
        percentile = a.soft_percentile_clips or ()
        extrema = a.soft_fixed_extrema or ()
        logistic_percentile = a.soft_logistic_percentile_clips or ()
        logistic_extrema = a.soft_logistic_fixed_extrema or ()
        if any(t <= 0 for t in ordinal):
            p.error('--ordinal-thresholds must contain only positive bps values')
        if any(q <= 0 or q >= 50 for q in (*percentile, *logistic_percentile)):
            p.error('soft-percentile clips must contain values in (0, 50)')
        if any(t <= 0 for t in (*extrema, *logistic_extrema)):
            p.error('soft fixed extrema must contain only positive bps values')
        arms = (
            ("BASE",)
            + tuple(f"ORDINAL_t{t:g}" for t in ordinal)
            + tuple(f"SOFT_Q{q:02g}_{100. - q:02g}" for q in percentile)
            + tuple(f"SOFT_E{t:g}" for t in extrema)
            + tuple(f"SOFTLOG_Q{q:02g}_{100. - q:02g}" for q in logistic_percentile)
            + tuple(f"SOFTLOG_E{t:g}" for t in logistic_extrema)
        )
    print(run(a.out, arms=arms))
