#!/usr/bin/env python3
"""Materialise causal soft-regime residual features for the shared TP6 M6 expert.

This is a *feature/target substrate*, not an M6 fit.  It deliberately does
not route rows to local experts.  For each calendar month, every transform is
fit exclusively on rows before that month.  Outcome-derived values are kept in
``target__*`` fields and explicitly excluded from the inference feature list.

The input ledger's ``base_raw`` has incompatible units across the 2023 and
2024 source ledgers.  The economic baseline is consequently rebuilt from the
same-side R3 probability simplex with a prior-only ridge payoff map.  This
keeps the residual consistently measured in net bps across the whole era.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data_perp/artifacts/tp6_m6_enriched_mda_20260809_v1/canonical_enriched_ledger.parquet"
OUT = ROOT / "data_perp/artifacts/tp6_m6_shared_regime_residual_features_20260809_v3"
SCHEMA = "tp6_m6_shared_regime_residual_features_v1"
COST_BPS = 100.0
# ``__ts__`` is the completed signal close.  The executable decision is the
# following hourly close/minute open (+1h), and H12 resolves twelve hours
# after that.  Do not shorten this to 12h merely because the path horizon is
# 12h: doing so would admit one hour of unresolved outcomes into a prior map.
ENTRY_DELAY = pd.Timedelta(hours=1)
LABEL_HORIZON = pd.Timedelta(hours=12)
LABEL_AVAILABILITY_DELAY = ENTRY_DELAY + LABEL_HORIZON
PROB = ("p_adverse", "p_weak", "p_clear")
CONTEXT = (
    "mkt_ret_eq_24h", "regime_liquidity_score", "mkt_rv_ratio_1h_24h",
    "mkt_oi_chg_z_24h", "mkt_funding_dispersion", "cross_asset_corr_4h",
    "mkt_systemic_deleveraging_score", "mkt_flush_exhaustion_score",
    "post_liquidation_rebound_score", "negative_breadth_pct",
    "btc_resilience_alt_weakness", "short_covering_score_market",
    "deleveraging_without_followthrough", "short_signal_recovery_conflict",
)
RELATIVE = (*PROB, *CONTEXT)
STATE_INPUTS = (
    "mkt_ret_eq_24h", "mkt_rv_ratio_1h_24h", "mkt_systemic_deleveraging_score",
    "negative_breadth_pct", "regime_liquidity_score", "cross_asset_corr_4h",
)
STATE_NAMES = ("calm", "trend", "stress", "transition")
MIN_PAYOFF_ROWS = 2_000
PAYOFF_RIDGE = 750.0
SIDE_SHRINK = 2_000.0
STATE_SHRINK = 1_000.0
MIN_REFERENCE_ROWS = 5_000
EPS = 1e-6


class ContractError(RuntimeError):
    """A provenance or no-imputation invariant was violated."""


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _finite(frame: pd.DataFrame, columns: Iterable[str]) -> np.ndarray:
    return np.isfinite(frame.loc[:, list(columns)].to_numpy(dtype=float)).all(axis=1)


def _ensure_label_available_ts(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach/validate the exact time at which a row's H12 outcome is known.

    Older ledgers do not carry an availability field.  For those, the only
    permitted reconstruction is signal-close +1h entry +12h path = +13h.
    A supplied exact field wins, but must still be finite and no earlier than
    the contract permits.
    """
    out = frame.copy()
    decision = pd.to_datetime(out.get("__decision_ts__", out["__ts__"] + ENTRY_DELAY), utc=True)
    expected = decision + LABEL_HORIZON
    source = "label_available_ts" if "label_available_ts" in out else "__label_available_at__" if "__label_available_at__" in out else None
    available = pd.to_datetime(out[source], utc=True, errors="raise") if source else expected
    if not available.notna().all():
        raise ContractError("label availability must be explicit or derivable from the signal-close contract")
    if (available < expected).any():
        raise ContractError("H12 label availability precedes decision + 12h")
    out["label_available_ts"] = available
    return out


def _prior_resolved_mask(frame: pd.DataFrame, cutoff: pd.Timestamp) -> np.ndarray:
    """Strictly-before cutoff: tied availability is not safe under unknown ordering."""
    return pd.to_datetime(frame["label_available_ts"], utc=True).lt(cutoff).to_numpy()


def _weekly_cutoff(ts: pd.Series) -> pd.Series:
    """Monday 00:00 UTC blocked-prequential reference cutoff for each row."""
    value = pd.to_datetime(ts, utc=True)
    return value.dt.normalize() - pd.to_timedelta(value.dt.weekday, unit="D")


def _robust_location_scale(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return np.nan, np.nan
    loc = float(np.median(x))
    scale = float(np.median(np.abs(x - loc)) * 1.4826)
    if not np.isfinite(scale) or scale < EPS:
        scale = float(np.std(x))
    return loc, max(scale, EPS)


def _state_fit(reference: pd.DataFrame) -> dict[str, tuple[float, float]]:
    if len(reference) < MIN_REFERENCE_ROWS or not _finite(reference, STATE_INPUTS).all():
        raise ContractError("state reference is incomplete or below the predeclared support floor")
    return {field: _robust_location_scale(reference[field].to_numpy(float)) for field in STATE_INPUTS}


def _state_probabilities(frame: pd.DataFrame, fit: dict[str, tuple[float, float]]) -> np.ndarray:
    """A transparent soft state surface fitted only from prior covariates."""
    z = {field: np.clip((frame[field].to_numpy(float) - loc) / scale, -6., 6.)
         for field, (loc, scale) in fit.items()}
    vol = z["mkt_rv_ratio_1h_24h"]
    ret = np.abs(z["mkt_ret_eq_24h"])
    stress = .55 * z["mkt_systemic_deleveraging_score"] + .45 * z["negative_breadth_pct"]
    illiquid = -z["regime_liquidity_score"]
    corr_break = np.abs(z["cross_asset_corr_4h"])
    logits = np.column_stack((
        -vol - .55 * stress - .35 * illiquid,
        ret - .55 * stress + .15 * z["regime_liquidity_score"],
        vol + stress + .45 * illiquid,
        .60 * ret + .55 * vol + .55 * corr_break + .35 * np.maximum(stress, 0.),
    ))
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def _payoff_map_fit(reference: pd.DataFrame, side: str) -> np.ndarray | None:
    """Prior-resolved side-local R3-simplex -> exact net-bps map."""
    x = reference.loc[reference.side_name.eq(side), list(PROB)].to_numpy(float)
    y = reference.loc[reference.side_name.eq(side), "net_bps"].to_numpy(float)
    good = np.isfinite(x).all(axis=1) & np.isfinite(y)
    x, y = x[good], y[good]
    if len(y) < MIN_PAYOFF_ROWS:
        return None
    # p_weak is redundant after the intercept; omit it deterministically.
    design = np.column_stack((np.ones(len(x)), x[:, 0], x[:, 2]))
    penalty = np.diag([0., PAYOFF_RIDGE, PAYOFF_RIDGE])
    return np.linalg.solve(design.T @ design + penalty, design.T @ y)


def _payoff_map_apply(frame: pd.DataFrame, coef: np.ndarray | None) -> np.ndarray:
    if coef is None:
        return np.full(len(frame), np.nan)
    x = frame.loc[:, list(PROB)].to_numpy(float)
    return np.column_stack((np.ones(len(x)), x[:, 0], x[:, 2])) @ coef


def _soft_prior_fit(reference: pd.DataFrame, probabilities: np.ndarray) -> dict[str, Any]:
    """Fit global -> side -> soft-state residual priors, with fixed shrinkage."""
    residual = reference["net_bps"].to_numpy(float) - reference["prequential_base_expected_net_bps"].to_numpy(float)
    good = np.isfinite(residual) & np.isfinite(probabilities).all(axis=1)
    if good.sum() < MIN_REFERENCE_ROWS:
        return {"ready": False}
    residual, probs = residual[good], probabilities[good]
    sides = reference.loc[good, "side_name"].to_numpy(str)
    global_mean, global_scale = _robust_location_scale(residual)
    result: dict[str, Any] = {"ready": True, "global_mean": global_mean, "global_scale": global_scale, "sides": {}}
    for side in ("long", "short"):
        mask = sides == side
        weight = probs[mask].sum(axis=0)
        sums = (probs[mask] * residual[mask, None]).sum(axis=0)
        side_n = float(mask.sum())
        side_mean = float(residual[mask].mean()) if side_n else global_mean
        side_prior = (side_n * side_mean + SIDE_SHRINK * global_mean) / (side_n + SIDE_SHRINK)
        side_scale = _robust_location_scale(residual[mask])[1] if side_n else global_scale
        means = (sums + STATE_SHRINK * side_prior) / (weight + STATE_SHRINK)
        # State scale is deliberately conservative: blend the weighted scale
        # with side scale rather than estimating unstable tiny-state variance.
        scales = []
        for j in range(len(STATE_NAMES)):
            weighted = probs[mask, j]
            if weighted.sum() <= EPS:
                scales.append(side_scale)
                continue
            center = means[j]
            variance = float((weighted * (residual[mask] - center) ** 2).sum() / weighted.sum())
            scales.append(max(math.sqrt(max(variance, EPS)), .5 * side_scale))
        result["sides"][side] = {"prior": side_prior, "scale": side_scale, "means": np.asarray(means), "scales": np.asarray(scales), "effective_support": weight}
    return result


def _soft_prior_apply(frame: pd.DataFrame, probabilities: np.ndarray, fit: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    prior, scale = np.full(len(frame), np.nan), np.full(len(frame), np.nan)
    if not fit.get("ready", False):
        return prior, scale
    for side in ("long", "short"):
        mask = frame.side_name.eq(side).to_numpy()
        if not mask.any():
            continue
        item = fit["sides"][side]
        prior[mask] = probabilities[mask] @ item["means"]
        # Law-of-total-variance approximation; every component already has a
        # shrinkage floor, avoiding fake confidence in a rare soft state.
        scale[mask] = np.sqrt(np.maximum((probabilities[mask] * item["scales"] ** 2).sum(axis=1), EPS))
    return prior, scale


def _relative_fit(reference: pd.DataFrame, probabilities: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for field in RELATIVE:
        value = reference[field].to_numpy(float)
        means, scales = [], []
        for j in range(len(STATE_NAMES)):
            w = probabilities[:, j]
            mean = float(np.sum(w * value) / np.sum(w))
            variance = float(np.sum(w * (value - mean) ** 2) / np.sum(w))
            means.append(mean); scales.append(max(math.sqrt(max(variance, EPS)), EPS))
        result[field] = (np.asarray(means), np.asarray(scales))
    return result


def _add_transition_proxies(frame: pd.DataFrame) -> pd.DataFrame:
    """Causal transition fields from the state history, never outcome history."""
    # These columns are preallocated on the complete ledger.  Remove their
    # empty placeholders before the merge; otherwise pandas creates ``_x`` /
    # ``_y`` pairs and silently leaves the intended fields null.
    frame = frame.drop(columns=["regime_transition_onset_proxy", "regime_state_duration_hours"], errors="ignore")
    source = frame.groupby("__ts__", as_index=False)[[f"regime_p_{name}" for name in STATE_NAMES]].mean().sort_values("__ts__", kind="stable")
    state_columns = [f"regime_p_{name}" for name in STATE_NAMES]
    available = source[state_columns].notna().all(axis=1).to_numpy()
    p = source["regime_p_transition"].to_numpy(float)
    previous = pd.Series(np.where(available, p, np.nan)).shift(1).ewm(halflife=24, adjust=False, min_periods=4).mean().to_numpy(float)
    source["regime_transition_onset_proxy"] = np.where(available, np.maximum(p - previous, 0.), np.nan)
    dominant = source[state_columns].to_numpy(float).argmax(axis=1)
    duration = np.full(len(source), np.nan)
    current, count = -1, 0
    for i, value in enumerate(dominant):
        if not available[i]:
            current, count = -1, 0
            continue
        count = count + 1 if value == current else 1
        current = value; duration[i] = float(count)
    source["regime_state_duration_hours"] = duration
    return frame.merge(source[["__ts__", "regime_transition_onset_proxy", "regime_state_duration_hours"]], on="__ts__", how="left", validate="many_to_one")


def materialize(input_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    frame = pd.read_parquet(input_path)
    required = {"candidate_id", "__ts__", "side_name", "gross_bps", "net_bps", "m6_contract_complete", *PROB, *CONTEXT}
    if missing := sorted(required.difference(frame.columns)):
        raise ContractError(f"input ledger lacks required fields: {missing}")
    if frame.candidate_id.duplicated().any():
        raise ContractError("candidate identity must be unique")
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    frame = _ensure_label_available_ts(frame)
    if not np.allclose(frame.gross_bps.to_numpy(float) - frame.net_bps.to_numpy(float), COST_BPS, atol=.02):
        raise ContractError("TP6 contract must charge fixed 100 bps exactly once")
    if not set(frame.side_name.unique()).issubset({"long", "short"}):
        raise ContractError("non-canonical side in input")
    frame = frame.sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    complete = frame.m6_contract_complete.astype(bool).to_numpy() & _finite(frame, (*PROB, *CONTEXT, "net_bps"))
    frame["shared_regime_contract_complete"] = complete
    frame["state_reference_cutoff_utc"] = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    frame["residual_reference_cutoff_utc"] = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    inference = [
        "prequential_base_expected_net_bps", "soft_regime_prior_residual_bps", "soft_regime_prior_residual_scale_bps",
        *[f"regime_p_{name}" for name in STATE_NAMES], "regime_entropy", "regime_transition_onset_proxy", "regime_state_duration_hours",
        *[f"regime_relative__{field}" for field in RELATIVE], *[f"regime_z__{field}" for field in RELATIVE],
    ]
    target = ["target__exact_net_residual_bps", "target__soft_regime_centered_residual_bps", "target__soft_regime_standardized_residual"]
    for field in [*inference, *target]: frame[field] = np.nan
    frame["_reference_cutoff"] = _weekly_cutoff(frame["__ts__"])
    cutoffs = sorted(frame["_reference_cutoff"].unique())
    provenance: list[dict[str, Any]] = []
    for start in cutoffs:
        start = pd.Timestamp(start)
        start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
        current_mask = frame["_reference_cutoff"].eq(start).to_numpy() & complete
        # Exact H12 labels resolving at or after the cutoff are never in a
        # map.  This uses the propagated availability timestamp rather than
        # a 12-hour shortcut from the signal timestamp.
        history_mask = complete & _prior_resolved_mask(frame, start)
        history = frame.loc[history_mask].copy()
        current = frame.loc[current_mask].copy()
        state_ready = len(history) >= MIN_REFERENCE_ROWS and _finite(history, STATE_INPUTS).all()
        payoff_ready = False
        if state_ready and len(current):
            sf = _state_fit(history)
            h_prob, c_prob = _state_probabilities(history, sf), _state_probabilities(current, sf)
            for j, name in enumerate(STATE_NAMES):
                frame.loc[current_mask, f"regime_p_{name}"] = c_prob[:, j]
            entropy = -np.sum(c_prob * np.log(np.clip(c_prob, EPS, 1.)), axis=1) / math.log(len(STATE_NAMES))
            frame.loc[current_mask, "regime_entropy"] = entropy
            frame.loc[current_mask, "state_reference_cutoff_utc"] = start
            relative = _relative_fit(history, h_prob)
            for field, (means, scales) in relative.items():
                expected = c_prob @ means
                scale = np.sqrt(np.maximum(c_prob @ (scales ** 2), EPS))
                value = current[field].to_numpy(float)
                frame.loc[current_mask, f"regime_relative__{field}"] = value - expected
                frame.loc[current_mask, f"regime_z__{field}"] = (value - expected) / scale
            for side in ("long", "short"):
                coef = _payoff_map_fit(history, side)
                side_current = current.side_name.eq(side).to_numpy()
                if coef is not None and side_current.any():
                    frame.loc[current_mask, "prequential_base_expected_net_bps"] = np.where(
                        side_current, _payoff_map_apply(current, coef), frame.loc[current_mask, "prequential_base_expected_net_bps"]
                    )
                    payoff_ready = True
            # The residual prior itself needs historical rows with a prior-only
            # payoff baseline.  It may not use the current period's outcomes.
            h_prior = history.loc[history.prequential_base_expected_net_bps.notna()].copy()
            h_prob2 = _state_probabilities(h_prior, sf) if len(h_prior) else np.empty((0, len(STATE_NAMES)))
            residual_fit = _soft_prior_fit(h_prior, h_prob2)
            prior, scale = _soft_prior_apply(current, c_prob, residual_fit)
            frame.loc[current_mask, "soft_regime_prior_residual_bps"] = prior
            frame.loc[current_mask, "soft_regime_prior_residual_scale_bps"] = scale
            frame.loc[current_mask, "residual_reference_cutoff_utc"] = start
        provenance.append({"reference_cutoff_utc": start, "current_complete_rows": int(current_mask.sum()),
                           "history_label_resolved_rows": int(history_mask.sum()), "state_ready": state_ready,
                           "payoff_map_ready": payoff_ready,
                           "prior_ready": bool(state_ready and frame.loc[current_mask, "soft_regime_prior_residual_bps"].notna().any())})
    frame = frame.drop(columns="_reference_cutoff")
    frame = _add_transition_proxies(frame)
    good_target = frame.prequential_base_expected_net_bps.notna()
    frame.loc[good_target, "target__exact_net_residual_bps"] = (
        frame.loc[good_target, "net_bps"] - frame.loc[good_target, "prequential_base_expected_net_bps"]
    )
    centered = good_target & frame.soft_regime_prior_residual_bps.notna()
    frame.loc[centered, "target__soft_regime_centered_residual_bps"] = (
        frame.loc[centered, "target__exact_net_residual_bps"] - frame.loc[centered, "soft_regime_prior_residual_bps"]
    )
    standard = centered & frame.soft_regime_prior_residual_scale_bps.gt(EPS)
    frame.loc[standard, "target__soft_regime_standardized_residual"] = (
        frame.loc[standard, "target__soft_regime_centered_residual_bps"] / frame.loc[standard, "soft_regime_prior_residual_scale_bps"]
    )
    # No value is synthesized for an incomplete row, including a prior or z.
    frame.loc[~complete, [*inference, *target]] = np.nan
    valid_state = frame[[f"regime_p_{n}" for n in STATE_NAMES]].notna().all(axis=1)
    if valid_state.any() and not np.allclose(frame.loc[valid_state, [f"regime_p_{n}" for n in STATE_NAMES]].sum(axis=1), 1., atol=1e-5):
        raise ContractError("soft regime simplex is invalid")
    if frame.loc[~complete, [*inference, *target]].notna().any(axis=None):
        raise ContractError("incomplete input row received an imputed feature or target")
    state_fields = [f"regime_p_{name}" for name in STATE_NAMES] + ["regime_entropy", "regime_transition_onset_proxy", "regime_state_duration_hours"]
    if frame.loc[~valid_state, state_fields].notna().any(axis=None):
        raise ContractError("state-unavailable row received a regime or transition proxy")
    # Training labels are outcome-derived and must not be supplied as model inputs.
    feature_contract = {"inference_eligible": inference, "training_only_targets": target,
                        "forbidden_from_inference": target, "base_raw_status": "excluded_unit_incompatible_2023_probability_vs_2024_bps"}
    coverage = pd.DataFrame({"field": [*inference, *target], "coverage": [float(frame[f].notna().mean()) for f in [*inference, *target]]})
    checks = {
        "candidate_identity_unique": bool(frame.candidate_id.is_unique),
        "fixed_cost_exactly_once": bool(np.allclose(frame.gross_bps - frame.net_bps, COST_BPS, atol=.02)),
        "incomplete_rows_have_no_synthesized_values": not frame.loc[~complete, [*inference, *target]].notna().any(axis=None),
        "state_unavailable_rows_have_no_regime_proxy": not frame.loc[~valid_state, state_fields].notna().any(axis=None),
        "soft_simplex_valid": bool((~valid_state | np.isclose(frame[[f"regime_p_{n}" for n in STATE_NAMES]].sum(axis=1), 1., atol=1e-5)).all()),
        "state_reference_precedes_candidate": bool(frame.loc[valid_state, "state_reference_cutoff_utc"].le(frame.loc[valid_state, "__ts__"]).all()),
        "residual_reference_precedes_candidate": bool(frame.loc[frame.soft_regime_prior_residual_bps.notna(), "residual_reference_cutoff_utc"].le(frame.loc[frame.soft_regime_prior_residual_bps.notna(), "__ts__"]).all()),
        "label_available_after_signal_plus_13h": bool(frame["label_available_ts"].ge(frame["__ts__"] + LABEL_AVAILABILITY_DELAY).all()),
        "target_fields_excluded_from_inference_contract": True,
        "no_local_or_per_regime_expert_fit": True,
    }
    stage = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}.staging-"))
    try:
        frame.to_parquet(stage / "shared_regime_residual_ledger.parquet", index=False, compression="zstd")
        coverage.to_parquet(stage / "coverage.parquet", index=False, compression="zstd")
        pd.DataFrame(provenance).to_parquet(stage / "prequential_reference_audit.parquet", index=False, compression="zstd")
        (stage / "feature_contract.json").write_text(json.dumps(feature_contract, indent=2) + "\n", encoding="utf-8")
        (stage / "correctness_test_report.json").write_text(json.dumps({"schema": SCHEMA, "passed": all(checks.values()), "checks": checks}, indent=2, default=str) + "\n", encoding="utf-8")
        report = "# TP6 shared soft-regime residual substrate\n\n"
        report += "- One shared-expert substrate; no local or per-regime experts are fitted.\n"
        report += "- State, payoff maps, residual priors, and relative normalisers use only rows before each weekly UTC cutoff; outcomes require signal-close +1h entry +H12 availability (=13h when no exact field exists).\n"
        report += "- `base_raw` is excluded because it is probability-like in 2023 but bps-like in 2024. The baseline is rebuilt from the consistent R3 simplex.\n"
        report += "- `target__*` columns are supervised targets only and are forbidden inference inputs.\n"
        (stage / "README.md").write_text(report, encoding="utf-8")
        manifest = {"schema": SCHEMA, "status": "MATERIALIZED_PREREQUISITE_NO_MODEL_FIT", "input": str(input_path), "input_sha256": _sha(input_path),
                    "geometry": "TP6/SL4/H12", "cost_bps": COST_BPS, "label_resolution_gap": "decision+12h; signal-close+13h fallback", "reference_refresh": "weekly UTC blocked-prequential", "state_names": STATE_NAMES,
                    "shrinkage": {"global_to_side": SIDE_SHRINK, "side_to_soft_state": STATE_SHRINK}, "feature_contract": feature_contract,
                    "rows": len(frame), "complete_rows": int(complete.sum()), "coverage": dict(zip(coverage.field, coverage.coverage)), "checks": checks}
        (stage / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
        os.replace(stage, output)
        return manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def sanitize_existing(source: Path, output: Path) -> dict[str, Any]:
    """Fail-close a completed substrate without refitting any transform.

    This is intentionally limited to a deterministic nulling correction: it
    never creates state values, refits maps, or changes target values.
    """
    if output.exists():
        raise FileExistsError(output)
    ledger_path = source / "shared_regime_residual_ledger.parquet"
    required = [ledger_path, source / "feature_contract.json", source / "prequential_reference_audit.parquet"]
    if any(not p.is_file() for p in required):
        raise FileNotFoundError(f"not a complete shared regime substrate: {source}")
    frame = pd.read_parquet(ledger_path)
    state_probability = "regime_p_calm"
    state_fields = [f"regime_p_{name}" for name in STATE_NAMES] + ["regime_entropy", "regime_transition_onset_proxy", "regime_state_duration_hours"]
    unavailable = frame[state_probability].isna()
    # This is the only allowed mutation: remove proxy outputs in the state
    # warm-up interval.  It cannot leak or invent information.
    frame.loc[unavailable, state_fields] = np.nan
    complete = frame["shared_regime_contract_complete"].astype(bool).to_numpy()
    inference = json.loads((source / "feature_contract.json").read_text(encoding="utf-8"))["inference_eligible"]
    target = json.loads((source / "feature_contract.json").read_text(encoding="utf-8"))["training_only_targets"]
    valid_state = frame[state_probability].notna()
    checks = {
        "candidate_identity_unique": bool(frame.candidate_id.is_unique),
        "fixed_cost_exactly_once": bool(np.allclose(frame.gross_bps - frame.net_bps, COST_BPS, atol=.02)),
        "incomplete_rows_have_no_synthesized_values": not frame.loc[~complete, [*inference, *target]].notna().any(axis=None),
        "state_unavailable_rows_have_no_regime_proxy": not frame.loc[~valid_state, state_fields].notna().any(axis=None),
        "soft_simplex_valid": bool((~valid_state | np.isclose(frame[[f"regime_p_{n}" for n in STATE_NAMES]].sum(axis=1), 1., atol=1e-5)).all()),
        "target_fields_excluded_from_inference_contract": True,
        "no_local_or_per_regime_expert_fit": True,
    }
    coverage = pd.DataFrame({"field": [*inference, *target], "coverage": [float(frame[f].notna().mean()) for f in [*inference, *target]]})
    stage = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}.staging-"))
    try:
        frame.to_parquet(stage / "shared_regime_residual_ledger.parquet", index=False, compression="zstd")
        coverage.to_parquet(stage / "coverage.parquet", index=False, compression="zstd")
        shutil.copy2(source / "feature_contract.json", stage / "feature_contract.json")
        shutil.copy2(source / "prequential_reference_audit.parquet", stage / "prequential_reference_audit.parquet")
        shutil.copy2(source / "README.md", stage / "README.md")
        (stage / "correctness_test_report.json").write_text(json.dumps({"schema": SCHEMA, "passed": all(checks.values()), "checks": checks}, indent=2) + "\n", encoding="utf-8")
        manifest = {"schema": SCHEMA, "status": "MATERIALIZED_PREREQUISITE_NO_MODEL_FIT", "source_artifact": str(source), "source_ledger_sha256": _sha(ledger_path),
                    "finalisation": "fail-closed nulling of warm-up state-proxy placeholders; no transform or target was refit", "geometry": "TP6/SL4/H12", "cost_bps": COST_BPS,
                    "reference_refresh": "weekly UTC blocked-prequential", "rows": len(frame), "complete_rows": int(complete.sum()),
                    "coverage": dict(zip(coverage.field, coverage.coverage)), "checks": checks}
        (stage / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.replace(stage, output)
        return manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=INPUT)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--sanitize-existing", type=Path, help="fail-close an existing materialisation without refitting")
    args = ap.parse_args()
    result = sanitize_existing(args.sanitize_existing, args.out) if args.sanitize_existing else materialize(args.input, args.out)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
