#!/usr/bin/env python3
"""Run one strict TP6/SL4 H12 meta-target ablation arm.

The input ledger contains *same-side, chronological base OOF predictions*.
This runner joins it to a causal context panel, trains a side-local meta model,
calibrates each side from an earlier/later split of the meta-train interval,
then allocates globally only after those side-local calibrations.  It is an
intentionally one-arm runner: callers select a winner between invocations.

Resolved outcome/path fields are admitted exclusively for labels and sample
weights.  The feature matrix is the frozen context contract plus causal base
outputs; it is checked against the frozen base contracts and a strict outcome
deny-list before fitting.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extreme_price_movements.tp6_sl4_meta_target_weights import (  # noqa: E402
    MetaColumns, MetaTargetParameters, MetaWeightParameters,
    MetaTrainingStatistics, build_meta_target, build_meta_weight,
    build_tail_training_mask, fit_meta_training_statistics, meta_target_manifest,
)
from extreme_price_movements.base_oos_reliability import (  # noqa: E402
    derive_causal_base_reliability, reliability_feature_columns,
)

TOP_FRACTIONS = (.005, .01, .02, .03, .05, .10)
TARGETS = {"M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10", "M11", "M12", "M13", "MC1", "MC2", "MC3"}
WEIGHTS = {"MW0", "MW1", "MW2", "MW3", "MW4", "MW5", "MW6", "MW7", "MW8"}
OUTCOME_TOKENS = ("tp6_sl4", "t2_path_", "future_", "forward_", "realised", "realized", "label", "target", "mfe", "mae", "exit_")
TRUST_CONTEXT_FIELDS = (
    "base_uncertainty_entropy", "base_uncertainty_top_probability", "base_uncertainty_margin",
    "base_score_ood_abs_z_21d", "base_score_recent_mean_21d", "base_score_recent_std_21d",
    "fund_pre_drift_5h", "fund_pre_drift_10h",
    "base_reliability_score_ic_ev_ewm_3d", "base_reliability_score_ic_ev_ewm_7d",
    "base_reliability_score_ic_ev_ewm_14d", "base_reliability_ev_surprise_ewm_3d",
    "base_reliability_ev_surprise_ewm_7d", "base_reliability_local_support_weight",
)


def _json(value: str | None, path: Path | None, cls: type) -> Any:
    if value and path:
        raise ValueError("use either inline JSON or JSON file, not both")
    payload = json.loads(path.read_text()) if path else (json.loads(value) if value else {})
    unknown = set(payload) - set(cls.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**payload)


def _features(meta_artifact: Path) -> dict[str, list[str]]:
    root_manifest = meta_artifact / "manifest.json"
    if root_manifest.exists():
        manifest = json.loads(root_manifest.read_text())
        value = manifest.get("meta_features")
        if isinstance(value, list):
            result = {"long": list(value), "short": list(value)}
        elif isinstance(value, dict) and set(value) >= {"long", "short"}:
            result = {side: list(value[side]) for side in ("long", "short")}
        else:
            raise ValueError("meta artifact manifest requires meta_features list or {long,short}")
    else:
        # Existing HPO artifact: one manifest per side, preserving its frozen
        # side-specific context contract.
        result = {}
        for side in ("long", "short"):
            manifest = json.loads((meta_artifact / side / "residual_meta_manifest.json").read_text())
            value = manifest.get("meta_feature_contract", {}).get(side)
            if not isinstance(value, list):
                raise ValueError(f"missing frozen {side} meta context contract")
            result[side] = list(value)
    for side, fields in result.items():
        if len(fields) != 36 or len(set(fields)) != len(fields):
            raise ValueError(f"{side} meta feature contract must contain 36 unique fields")
    return result


def _base_features(base_artifact: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for side in ("long", "short"):
        doc = json.loads((base_artifact / side / "target_family_manifest.json").read_text())
        fields = doc.get("feature_contract", {}).get(f"T2_soft_barrier|tp3_sl2|{side}")
        if not isinstance(fields, list) or len(fields) != 36:
            raise ValueError(f"missing frozen 36-field {side} base contract")
        result[side] = fields
    return result


def _read_panel(panel: Path, needed: list[str]) -> pd.DataFrame:
    parts = sorted((panel / "parts").glob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"no panel parts in {panel}")
    return pd.concat([pd.read_parquet(part, columns=needed) for part in parts], ignore_index=True)


def _read_consensus(sidecar: Path, candidate_ids: pd.Series) -> pd.DataFrame:
    """Read the exact nine-contract agreement diagnostic for certainty heads.

    This is a label-only sidecar: it is joined after the causal feature matrix
    has been fixed and is never offered to the inference model.
    """
    parts = sorted((sidecar / "parts").glob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"no consensus parts in {sidecar}")
    columns = ["candidate_id", "tp6_sl4_contract_mode_fraction"]
    wanted = set(candidate_ids.astype(str))
    selected: list[pd.DataFrame] = []
    # Read and filter one partition at a time: the sidecar spans a much longer
    # history than the OOF ledger, so concatenating it before filtering creates
    # needless peak memory pressure in quantile/weight runs.
    for part in parts:
        chunk = pd.read_parquet(part, columns=columns)
        chunk = chunk[chunk["candidate_id"].isin(wanted)]
        if not chunk.empty:
            selected.append(chunk)
    return pd.concat(selected, ignore_index=True)


def _rank_per_side_timestamp(frame: pd.DataFrame) -> pd.Series:
    # Rank is derived solely from the already causal base score.  It never uses
    # an outcome and handles ties deterministically by candidate id.
    ordered = frame.sort_values(["side_name", "__ts__", "base_expected_net_bps", "candidate_id"], ascending=[True, True, True, True], kind="mergesort")
    rank = ordered.groupby(["side_name", "__ts__"], sort=False).cumcount() + 1
    count = ordered.groupby(["side_name", "__ts__"], sort=False)["candidate_id"].transform("size")
    # percentile=1 is best (the expectation is sorted ascending above).
    value = 1. - (rank - 1.) / np.maximum(count - 1., 1.)
    return pd.Series(value.to_numpy(), index=ordered.index).reindex(frame.index)


def _rank_per_side_fold(frame: pd.DataFrame) -> np.ndarray:
    """Fold-local admission rank, 1=best, without looking at outcomes.

    This differs intentionally from the timestamp-relative *feature*: M0-bis
    asks for a per-side base-prediction population gate, so it is recomputed on
    each training fold rather than inherited from a global ledger ranking.
    """
    ordered = frame.sort_values(["base_expected_net_bps", "candidate_id"], ascending=[True, True], kind="mergesort")
    n = len(ordered)
    value = 1. - np.arange(n, dtype=float) / max(n - 1, 1)
    return pd.Series(value, index=ordered.index).reindex(frame.index).to_numpy()


def _derived(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "t4_tp6_sl4_gross_bps" not in result:
        # The exact TP6/SL4 label contract has a fixed 100-bps round trip cost.
        result["t4_tp6_sl4_gross_bps"] = result["t4_tp6_sl4_net_bps"].astype(float) + 100.0
    p = result[["base_p_upper", "base_p_lower", "base_p_timeout"]].to_numpy(float)
    if not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("ledger base probabilities must be finite and non-negative")
    p /= np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
    result[["base_p_upper", "base_p_lower", "base_p_timeout"]] = p
    result["causal_base_expected_net_bps"] = result["base_expected_net_bps"].astype(float)
    result["causal_base_margin"] = p[:, 0] - p[:, 1]
    result["causal_base_entropy"] = -(p * np.log(np.maximum(p, 1e-12))).sum(axis=1)
    result["causal_base_rank_percentile"] = _rank_per_side_timestamp(result)
    # ATR-normalised targets require a numerically meaningful denominator.
    # A one-bp floor is a conservative unit guard, not a future-derived
    # transform; rows below it otherwise explode when converted back to bps.
    result["entry_atr_bps"] = (result["atr_1h"].abs() / result["decision_price"].abs() * 1e4).replace([np.inf, -np.inf], np.nan).clip(lower=1.0)
    # These are target-only certainty quantities.  Exact H12 has complete,
    # unambiguous one-minute paths; stability is nevertheless not fabricated
    # as one: it rises smoothly with distance from the cost-clear boundary.
    # Base stability is causal, from the ledger probability entropy.
    net = result["t4_tp6_sl4_net_bps"].to_numpy(float)
    derived_label_stability = 1. / (1. + np.exp(-np.clip(np.abs(net) / 100., -35., 35.)))
    derived_base_stability = 1. - result["causal_base_entropy"].to_numpy(float) / np.log(3.)
    label_stability = result["label_stability"].to_numpy(float) if "label_stability" in result else derived_label_stability
    defaults = {
        "label_stability": derived_label_stability,
        "base_target_stability": np.clip(derived_base_stability, 0., 1.),
        "path_completeness": np.ones(len(result)),
        "event_conflict": np.zeros(len(result)),
        "label_certainty": np.clip(label_stability * np.clip(derived_base_stability, 0., 1.), 0., 1.),
    }
    for field, default in defaults.items():
        if field not in result: result[field] = default
    result["meta_class_label"] = pd.cut(result["t4_tp6_sl4_net_bps"], [-np.inf, -300., -100., 0., 100., 250., np.inf], labels=False).astype(int)
    return result


def _add_trust_context(frame: pd.DataFrame) -> pd.DataFrame:
    """Materialise causal uncertainty, drift and recent base-performance inputs.

    All outcome-derived reliability fields are delayed until the next UTC day
    after their H12 label resolves.  Score-distribution OOD fields use only
    prior scored rows.  None of these columns is an outcome at inference.
    """
    out = frame.copy()
    out["__symbol__"] = out["candidate_id"].astype(str).str.split("|", n=1).str[0]
    p = out[["base_p_upper", "base_p_lower", "base_p_timeout"]].to_numpy(float)
    out["base_uncertainty_entropy"] = -(p * np.log(np.maximum(p, 1e-12))).sum(axis=1)
    out["base_uncertainty_top_probability"] = p.max(axis=1)
    out["base_uncertainty_margin"] = np.sort(p, axis=1)[:, -1] - np.sort(p, axis=1)[:, -2]
    out["__archetype_policy_key__"] = "all"
    out["__first_touch_policy_soft__"] = out["base_p_upper"].astype(float)
    out["__first_touch_hit__"] = (out["t4_tp6_sl4_net_bps"].astype(float) > 0.).astype(float)
    out["__first_touch_capture_net__"] = out["t4_tp6_sl4_net_bps"].astype(float)
    reliability = derive_causal_base_reliability(
        out,
        resolution_column="__label_available_at__",
        score_column="causal_base_expected_net_bps",
        rank_column="causal_base_rank_percentile",
        soft_column="__first_touch_policy_soft__",
        hit_column="__first_touch_hit__",
        ev_column="__first_touch_capture_net__",
    )
    out = out.merge(
        reliability,
        on=["__ts__", "__symbol__", "side_name", "__archetype_policy_key__"],
        how="left", validate="one_to_one",
    )
    out = out.sort_values(["side_name", "__ts__", "candidate_id"], kind="mergesort")
    for field in ("base_score_ood_abs_z_21d", "base_score_recent_mean_21d", "base_score_recent_std_21d"):
        out[field] = 0.
    # Snapshot-before-score distribution diagnostics.  They intentionally do
    # not inspect any realised net value, so they are available immediately.
    for side, group in out.groupby("side_name", sort=False):
        history_days: list[pd.Timestamp] = []
        history_scores: list[np.ndarray] = []
        for day, idx in group.groupby(group["__ts__"].dt.normalize(), sort=True).groups.items():
            positions = np.asarray(list(idx), dtype=np.int64)
            cutoff = pd.Timestamp(day) - pd.Timedelta(days=21)
            keep = [values for d, values in zip(history_days, history_scores) if d >= cutoff]
            history_days = [d for d in history_days if d >= cutoff]
            history_scores = keep
            ref = np.concatenate(keep) if keep else np.empty(0, dtype=float)
            values = out.loc[positions, "causal_base_expected_net_bps"].to_numpy(float)
            if len(ref) >= 100:
                median = float(np.median(ref))
                mad = max(float(np.median(np.abs(ref - median))) * 1.4826, float(np.std(ref)), 1e-6)
                out.loc[positions, "base_score_ood_abs_z_21d"] = np.abs(values - median) / mad
                out.loc[positions, "base_score_recent_mean_21d"] = median
                out.loc[positions, "base_score_recent_std_21d"] = mad
            history_days.append(pd.Timestamp(day))
            history_scores.append(values)
    # Missing reliability only occurs at the beginning of history, where a
    # neutral zero is the documented no-prior state.
    for field in reliability_feature_columns():
        out[field] = pd.to_numeric(out[field], errors="coerce").fillna(0.).astype(np.float32)
    return out.sort_index(kind="mergesort")


def _matrix(frame: pd.DataFrame, fields: list[str]) -> np.ndarray:
    value = frame.loc[:, fields].replace([np.inf, -np.inf], np.nan)
    return value.fillna(0.).to_numpy(np.float32)


def _model(*, binary: bool = False, soft_binary: bool = False, quantile: float | None = None) -> lgb.LGBMRegressor:
    params: dict[str, Any] = dict(n_estimators=100, learning_rate=.04, num_leaves=24, min_child_samples=400, colsample_bytree=.8, subsample=.8, reg_lambda=12., random_state=20260814, n_jobs=1, verbosity=-1)
    if quantile is not None:
        params.update(objective="quantile", alpha=quantile)
    elif soft_binary:
        # LightGBM's cross-entropy objective accepts memberships in [0, 1].
        # Do not use the integer-class binary objective for soft labels.
        params.update(objective="cross_entropy")
    elif binary:
        params.update(objective="binary")
    else:
        params.update(objective="huber", alpha=.90)
    return lgb.LGBMRegressor(**params)


def _fit_predict(train: pd.DataFrame, score: pd.DataFrame, fields: list[str], target: str, params: MetaTargetParameters, weights: np.ndarray) -> np.ndarray:
    stats = fit_meta_training_statistics(train)
    # M4 owns its quantile targets locally below; its base residual definition
    # is M0 and it has no single-target entry in the generic target module.
    bundle = build_meta_target(train, "M0" if target == "M4" else target, parameters=params, statistics=stats)
    xs, xv = _matrix(train, fields), _matrix(score, fields)
    def fit(y: np.ndarray, *, binary: bool = False, soft_binary: bool = False, mask: np.ndarray | None = None, q: float | None = None) -> np.ndarray:
        keep = np.ones(len(train), dtype=bool) if mask is None else mask
        if keep.sum() < 500:
            raise ValueError(f"{target} conditional head has fewer than 500 training rows")
        return _model(binary=binary, soft_binary=soft_binary, quantile=q).fit(xs[keep], y[keep], sample_weight=weights[keep]).predict(xv)
    if target == "M4":
        residual = train.t4_tp6_sl4_net_bps.to_numpy(float) - train.causal_base_expected_net_bps.to_numpy(float)
        # Median residual is the ranking correction.  Width is stored as a
        # diagnostic, rather than silently entering the score.
        q10, q25, q50, q75, q90 = (fit(residual, q=q) for q in (.10, .25, .50, .75, .90))
        score["meta_q10_residual_bps"], score["meta_q25_residual_bps"], score["meta_q50_residual_bps"] = q10, q25, q50
        score["meta_q75_residual_bps"], score["meta_q90_residual_bps"] = q75, q90
        return score.causal_base_expected_net_bps.to_numpy(float) + q50
    if bundle.task in {"regression", "binary"}:
        raw = fit(bundle.primary, binary=bundle.task == "binary" and target not in {"M7", "MC2"}, soft_binary=target in {"M7", "MC2"})
        if target == "M3": raw *= score.entry_atr_bps.to_numpy(float)
        if target in {"M0", "M1", "M2", "M3", "MC1", "MC3"}:
            return score.causal_base_expected_net_bps.to_numpy(float) + raw
        if target == "M12":
            return score.causal_base_expected_net_bps.to_numpy(float) - params.overestimate_margin_bps * raw
        # Classification/reliability outputs deliberately become an input to
        # side calibration, not a fabricated bps correction.
        return raw
    if target == "M8":
        # Cumulative ordinal model: fit P(Y > edge) at every frozen economic
        # edge, enforce monotonicity, then reconstruct expected net from
        # training-only bin means.  Class IDs are not treated as equally spaced.
        edges = np.asarray(params.ordinal_edges_bps, dtype=float)
        p_gt = np.column_stack([np.clip(fit((train.t4_tp6_sl4_net_bps.to_numpy(float) > edge).astype(float), binary=True), 0., 1.) for edge in edges])
        p_gt = np.minimum.accumulate(p_gt, axis=1)
        means = np.asarray(stats.ordinal_bin_means, dtype=float)
        probabilities = np.column_stack([1. - p_gt[:, 0], p_gt[:, :-1] - p_gt[:, 1:], p_gt[:, -1]])
        return probabilities @ means
    if target == "M9":
        p = np.clip(fit(bundle.heads["failure_probability"], binary=True), 0., 1.)
        severity = np.maximum(fit(bundle.heads["failure_severity"], mask=bundle.valid_masks["failure_severity"]), 0.)
        return score.causal_base_expected_net_bps.to_numpy(float) - p * severity
    if target == "M10":
        p = np.clip(fit(bundle.heads["success_probability"], binary=True), 0., 1.)
        upside = np.maximum(fit(bundle.heads["success_upside"], mask=bundle.valid_masks["success_upside"]), 0.)
        return p * upside
    if target == "M11":
        ps = np.clip(fit(bundle.heads["success_probability"], binary=True), 0., 1.)
        pf = np.clip(fit(bundle.heads["failure_probability"], binary=True), 0., 1.)
        upside = np.maximum(fit(bundle.heads["success_upside"], mask=bundle.valid_masks["success_upside"]), 0.)
        loss = np.maximum(fit(bundle.heads["failure_severity"], mask=bundle.valid_masks["failure_severity"]), 0.)
        return ps * upside - pf * loss
    raise AssertionError(target)


def _calibrated_side(train: pd.DataFrame, evaluation: pd.DataFrame, fields: list[str], target: str, weight_name: str, params: MetaTargetParameters, weights: MetaWeightParameters, tail: float) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    # Time split: the early fit is used exclusively to create later, unseen
    # calibration scores.  The final fit uses all pre-evaluation rows.
    cut = train.__ts__.quantile(.80)
    calibration = train[train.__ts__.ge(cut)].copy()
    # Labels must have resolved before the first calibration score.  This is
    # separate from base OOF lineage: it prevents a meta fit from observing a
    # H12 outcome whose path overlaps the calibration interval.
    calibration_start = calibration["__ts__"].min()
    early = train[train["__ts__"].lt(cut) & train["__label_available_at__"].lt(calibration_start)].copy()
    if min(len(early), len(calibration), len(evaluation)) < 1000:
        raise ValueError("need at least 1,000 early/calibration/evaluation rows per side")
    def weights_for(rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        # The M0-bis / MW1 population rank is fold-local and side-local.  The
        # independently timestamp-relative rank remains available as a causal
        # feature only in the model matrix.
        population = rows.copy()
        population["causal_base_rank_percentile"] = _rank_per_side_fold(population)
        mask = build_tail_training_mask(population, tail)
        eligible = population.loc[mask].copy()
        b = build_meta_target(eligible, target if target != "M4" else "M0", parameters=params, statistics=fit_meta_training_statistics(eligible))
        label = np.rint(b.primary).astype(int) if b.task in {"binary", "ordinal"} else eligible.meta_class_label.to_numpy(int)
        stat = fit_meta_training_statistics(eligible, class_label=label)
        return build_meta_weight(eligible, weight_name, parameters=weights, statistics=stat, class_label=label), mask
    w_early, mask_early = weights_for(early)
    cal_raw = _fit_predict(early.loc[mask_early].copy(), calibration, fields, target, params, w_early)
    # Isotonic learns a monotone score -> realised-net map entirely before the
    # evaluation window; fallback constant avoids undefined single-valued maps.
    ycal = calibration.t4_tp6_sl4_net_bps.to_numpy(float)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(cal_raw, ycal)
    w_full, mask_full = weights_for(train)
    raw = _fit_predict(train.loc[mask_full].copy(), evaluation, fields, target, params, w_full)
    output_fields = ["candidate_id", "__ts__", "__label_available_at__", "side_name", "t4_tp6_sl4_gross_bps", "t4_tp6_sl4_net_bps", "causal_base_expected_net_bps", "causal_base_rank_percentile"]
    # Preserve all five M4 quantiles as diagnostics.  Only q50 is the
    # correction score; the other four are explicitly non-ranking outputs.
    output_fields += [field for field in ("meta_q10_residual_bps", "meta_q25_residual_bps", "meta_q50_residual_bps", "meta_q75_residual_bps", "meta_q90_residual_bps") if field in evaluation]
    output = evaluation[output_fields].copy()
    output["meta_raw_score"] = raw
    output["side_calibrated_score_bps"] = iso.predict(raw)
    calibration_output = calibration[output_fields[:8]].copy()
    calibration_output["meta_raw_score"] = cal_raw
    calibration_output["side_calibrated_score_bps"] = iso.predict(cal_raw)
    return output, {"train_rows": len(train), "early_rows": len(early), "calibration_rows": len(calibration), "admitted_train_rows": int(mask_full.sum()), "tail_fraction": tail, "calibration_cut": str(cut), "calibration_score_net_spearman": float(spearmanr(cal_raw, ycal).statistic), "features": fields}, calibration_output


def _apply_recent_admission_map(scored: pd.DataFrame, calibration_history: pd.DataFrame) -> pd.DataFrame:
    """Causal robust 21-day side EV map over relative score percentiles.

    The meta model is refit over time, so its raw (and even isotonic-calibrated)
    scale is not a stable reference coordinate.  Per-side, per-snapshot score
    percentiles are causal and preserve ordering across those refits.
    """
    out = scored.copy()
    out["admission_score_bps"] = np.nan
    history = calibration_history.copy()
    history["__ts__"] = pd.to_datetime(history["__ts__"], utc=True)
    history["__label_available_at__"] = pd.to_datetime(history["__label_available_at__"], utc=True)
    for frame in (history, out):
        ordered = frame.sort_values(["side_name", "__ts__", "side_calibrated_score_bps", "candidate_id"], kind="mergesort")
        rank = ordered.groupby(["side_name", "__ts__"], sort=False).cumcount() + 1
        count = ordered.groupby(["side_name", "__ts__"], sort=False)["candidate_id"].transform("size")
        frame["__admission_percentile__"] = pd.Series(rank.to_numpy(float) / count.to_numpy(float), index=ordered.index).reindex(frame.index).to_numpy()
    ordered = out.sort_values(["__ts__", "candidate_id"], kind="mergesort")
    for day, indices in ordered.groupby(ordered["__ts__"].dt.normalize(), sort=True).groups.items():
        positions = list(indices)
        prior = ordered[ordered["__ts__"].lt(day) & ordered["__label_available_at__"].lt(day)]
        reference = pd.concat([history, prior], ignore_index=True)
        reference = reference[reference["__label_available_at__"].ge(pd.Timestamp(day) - pd.Timedelta(days=21)) & reference["__label_available_at__"].lt(day)]
        if len(reference) < 500:
            continue
        current = ordered.loc[positions]
        pooled = IsotonicRegression(out_of_bounds="clip").fit(reference.__admission_percentile__, reference.t4_tp6_sl4_net_bps)
        mapped = pooled.predict(current.__admission_percentile__)
        for side in ("long", "short"):
            local = reference[reference.side_name.eq(side)]
            mask = current.side_name.eq(side).to_numpy()
            if not mask.any() or len(local) < 500 or local.__admission_percentile__.nunique() < 2:
                continue
            local_iso = IsotonicRegression(out_of_bounds="clip").fit(local.__admission_percentile__, local.t4_tp6_sl4_net_bps)
            shrink = len(local) / (len(local) + 500.)
            mapped[mask] = shrink * local_iso.predict(current.loc[mask, "__admission_percentile__"]) + (1. - shrink) * mapped[mask]
        out.loc[positions, "admission_score_bps"] = mapped
    out["admission_score_bps"] = out["admission_score_bps"].fillna(out["side_calibrated_score_bps"])
    return out


def _metrics(scored: pd.DataFrame, *, rank_column: str = "side_calibrated_score_bps", max_new_per_row: int = 0) -> list[dict[str, Any]]:
    # Isotonic calibration is deliberately piecewise-constant.  Preserve the
    # causal within-side raw ordering inside a common calibrated plateau rather
    # than letting candidate_id decide the book.
    global_rank = scored.sort_values([rank_column, "meta_raw_score", "candidate_id"], ascending=[False, False, True], kind="mergesort")
    rows: list[dict[str, Any]] = []
    for fraction in TOP_FRACTIONS:
        selected = global_rank.head(int(np.ceil(len(global_rank) * fraction)))
        if max_new_per_row:
            selected = selected.groupby("__ts__", sort=False).head(max_new_per_row)
        for view, frame in (("global", selected), ("long", selected[selected.side_name.eq("long")]), ("short", selected[selected.side_name.eq("short")])):
            rows.append({"allocation": "global_after_side_calibration", "rank_column": rank_column, "max_new_per_row": max_new_per_row, "attribution_side": view, "top_fraction": fraction, "n": len(frame), "gross_bps": float(frame.t4_tp6_sl4_gross_bps.mean()), "net_bps": float(frame.t4_tp6_sl4_net_bps.mean())})
        # Per-side ranking is also reported, without contaminating allocation.
        for side in ("long", "short"):
            side_rows = scored[scored.side_name.eq(side)].sort_values([rank_column, "meta_raw_score", "candidate_id"], ascending=[False, False, True], kind="mergesort")
            take = side_rows.head(int(np.ceil(len(side_rows) * fraction)))
            rows.append({"allocation": "per_side", "rank_column": rank_column, "max_new_per_row": 0, "attribution_side": side, "top_fraction": fraction, "n": len(take), "gross_bps": float(take.t4_tp6_sl4_gross_bps.mean()), "net_bps": float(take.t4_tp6_sl4_net_bps.mean())})
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ledger", type=Path, required=True)
    p.add_argument("--panel", type=Path, required=True)
    p.add_argument("--meta-artifact", type=Path, required=True, help="artifact with frozen 36 context fields/side")
    p.add_argument("--base-artifact", type=Path, required=True)
    p.add_argument("--consensus-sidecar", type=Path, help="optional exact nine-contract stability labels for MC/MW certainty")
    p.add_argument("--extended-trust-context", action="store_true", help="add causal base uncertainty, OOD/drift and trailing reliability features to the meta matrix")
    p.add_argument("--recent-admission-map", action="store_true", help="rank with a causal robust 21-day side EV map after side calibration")
    p.add_argument("--max-new-per-row", type=int, default=0, help="post-rank cap on new entries at one timestamp; 0 disables the cap")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--target", choices=sorted(TARGETS), required=True)
    p.add_argument("--weight", choices=sorted(WEIGHTS), default="MW0")
    p.add_argument("--tail-fraction", type=float, default=1., choices=(.2, .3, .4, .5, .6, .8, 1.0))
    p.add_argument("--train-start", required=True); p.add_argument("--eval-start", required=True); p.add_argument("--eval-end", required=True)
    p.add_argument("--target-parameters"); p.add_argument("--target-parameters-json", type=Path)
    p.add_argument("--weight-parameters"); p.add_argument("--weight-parameters-json", type=Path)
    a = p.parse_args()
    if a.out.exists(): raise FileExistsError(a.out)
    start, estart, eend = (pd.Timestamp(x, tz="UTC") for x in (a.train_start, a.eval_start, a.eval_end))
    if not start < estart < eend: raise ValueError("require train-start < eval-start < eval-end")
    tparams = _json(a.target_parameters, a.target_parameters_json, MetaTargetParameters)
    wparams = _json(a.weight_parameters, a.weight_parameters_json, MetaWeightParameters)
    meta, base = _features(a.meta_artifact), _base_features(a.base_artifact)
    for side in ("long", "short"):
        overlap = sorted(set(meta[side]) & set(base[side]))
        unsafe = [x for x in meta[side] if any(token in x.lower() for token in OUTCOME_TOKENS)]
        if overlap or unsafe: raise ValueError(f"invalid {side} meta contract: base-overlap={overlap}, outcome-like={unsafe}")
    ledger_path = a.ledger / "base_oof_ledger.parquet" if a.ledger.is_dir() else a.ledger
    ledger_manifest = ledger_path.parent / "manifest.json"
    if not ledger_manifest.exists():
        raise FileNotFoundError("strict base-OOF ledger manifest is required beside the ledger parquet")
    ledger_lineage = json.loads(ledger_manifest.read_text())
    if "strict_oof" not in ledger_lineage or "TP6/SL4" not in str(ledger_lineage.get("geometry", "")):
        raise ValueError("ledger does not attest strict TP6/SL4 base OOF lineage")
    ledger = pd.read_parquet(ledger_path)
    required_ledger = {"candidate_id", "__ts__", "side_name", "t4_tp6_sl4_net_bps", "base_expected_net_bps", "base_p_upper", "base_p_lower", "base_p_timeout", "base_fit_resolved_before"}
    if missing := required_ledger - set(ledger.columns): raise KeyError(f"ledger lacks required fields: {sorted(missing)}")
    ledger["__ts__"] = pd.to_datetime(ledger["__ts__"], utc=True)
    ledger["base_fit_resolved_before"] = pd.to_datetime(ledger["base_fit_resolved_before"], utc=True)
    if not ledger["base_fit_resolved_before"].le(ledger["__ts__"]).all():
        raise ValueError("base OOF ledger contains a fit cutoff after its scored row")
    # TP6/SL4 is exact H12; the ledger does not duplicate this availability
    # field, so materialise its deterministic availability contract here.
    ledger["__label_available_at__"] = ledger["__ts__"] + pd.Timedelta(hours=12)
    # No later rows can contribute either to training, calibration, or a
    # causal admission map for this evaluation.  Prune before the wide panel
    # join to keep trust-context materialisation bounded.
    ledger = ledger[ledger["__ts__"].lt(eend)].copy()
    panel_trust = [field for field in TRUST_CONTEXT_FIELDS if field.startswith("fund_pre_drift_")]
    needed = list(dict.fromkeys(["candidate_id", "__ts__", "side_name", "atr_1h", "decision_price", *meta["long"], *meta["short"], *(panel_trust if a.extended_trust_context else [])]))
    context = _read_panel(a.panel, needed)
    data = ledger.merge(context, on=["candidate_id", "__ts__", "side_name"], how="inner", validate="one_to_one")
    data["__ts__"] = pd.to_datetime(data["__ts__"], utc=True)
    if len(data) != len(ledger): raise ValueError("ledger/context join lost rows")
    certainty_source = "net-boundary proxy"
    if a.consensus_sidecar:
        consensus = _read_consensus(a.consensus_sidecar, data["candidate_id"])
        data = data.merge(consensus, on="candidate_id", how="left", validate="one_to_one")
        if data["tp6_sl4_contract_mode_fraction"].isna().any():
            raise ValueError("consensus sidecar failed to cover all ledger rows")
        data["label_stability"] = data["tp6_sl4_contract_mode_fraction"].clip(0., 1.)
        data["event_conflict"] = 1. - data["label_stability"]
        certainty_source = "exact nine-contract TP6/SL4 agreement"
    data = _derived(data)
    if a.extended_trust_context:
        data = _add_trust_context(data)
    # Feature availability is an inference-contract property of the causal
    # side population, not of a deliberately tail-gated training subset.  A
    # tail can legitimately overrepresent an asset whose one meta field is
    # missing; its train-only imputation must not prevent M0-bis from running.
    active_meta = {side: meta[side] + (list(TRUST_CONTEXT_FIELDS) if a.extended_trust_context else []) for side in ("long", "short")}
    for side in ("long", "short"):
        coverage = 1. - data.loc[data.side_name.eq(side), active_meta[side]].replace([np.inf, -np.inf], np.nan).isna().mean()
        if (coverage < .90).any():
            raise ValueError(f"{side} meta feature coverage below 90% in full causal population: {coverage[coverage < .90].to_dict()}")
    train = data[data.__ts__.ge(start) & data.__ts__.lt(estart) & data["__label_available_at__"].lt(estart)].copy(); evaluation = data[data.__ts__.ge(estart) & data.__ts__.lt(eend)].copy()
    if train.empty or evaluation.empty: raise ValueError("empty chronological split")
    outputs, calibration_history, side_summary = [], [], {}
    # Base outputs are intentionally available to the meta layer directly, not
    # converted into a separate label.  Context remains side-specific/frozen.
    additions = ["causal_base_expected_net_bps", "base_p_upper", "base_p_lower", "base_p_timeout", "causal_base_margin", "causal_base_entropy", "causal_base_rank_percentile"]
    for side in ("long", "short"):
        out, diag, cal = _calibrated_side(train[train.side_name.eq(side)].copy(), evaluation[evaluation.side_name.eq(side)].copy(), active_meta[side] + additions, a.target, a.weight, tparams, wparams, a.tail_fraction)
        outputs.append(out); calibration_history.append(cal); side_summary[side] = diag
    scored = pd.concat(outputs, ignore_index=True)
    rank_column = "side_calibrated_score_bps"
    if a.recent_admission_map:
        scored = _apply_recent_admission_map(scored, pd.concat(calibration_history, ignore_index=True))
        rank_column = "admission_score_bps"
    metrics = _metrics(scored, rank_column=rank_column, max_new_per_row=a.max_new_per_row)
    a.out.mkdir(parents=True)
    scored.to_parquet(a.out / "predictions.parquet", index=False)
    # Keep the strictly earlier, out-of-fit calibration scores as a distinct
    # artifact.  Admission-map experiments must never reconstruct this
    # reference from evaluation outcomes or use the final-fit score scale as
    # though it had existed before the evaluation period.
    pd.concat(calibration_history, ignore_index=True).to_parquet(a.out / "calibration_history.parquet", index=False)
    pd.DataFrame(metrics).to_parquet(a.out / "metrics.parquet", index=False)
    manifest = {"schema": "tp6_sl4_meta_target_funnel_v1", "target_contract": meta_target_manifest(a.target if a.target != "M4" else "M0", a.weight, target_parameters=tparams, weight_parameters=wparams), "target": a.target, "weight": a.weight, "tail_fraction": a.tail_fraction, "windows": {"train_start": str(start), "eval_start": str(estart), "eval_end": str(eend), "strict_ledger": "base scores are same-side OOF"}, "ledger_manifest": str(ledger_manifest), "meta_context_features": active_meta, "base_features_rejected": base, "meta_direct_base_inputs": additions, "extended_trust_context": bool(a.extended_trust_context), "recent_admission_map": bool(a.recent_admission_map), "rank_column": rank_column, "max_new_per_row": int(a.max_new_per_row), "certainty_source": certainty_source, "side_summary": side_summary, "metrics": metrics, "global_spearman": float(spearmanr(scored[rank_column], scored.t4_tp6_sl4_net_bps).statistic)}
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"target": a.target, "weight": a.weight, "tail": a.tail_fraction, "metrics": metrics}, indent=2))


if __name__ == "__main__": main()
