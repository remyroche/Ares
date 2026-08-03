#!/usr/bin/env python3
"""Diagnose whether month-to-month top-decile EV changes are composition or response.

This is deliberately a *diagnostic* standardisation, not a model, calibration,
or policy.  It selects the frozen raw base score's pooled-global monthly top
decile once, then reweights the source month to the target month's *causal*
candidate context.  It therefore answers a narrow question: conditional on the
same side, asset, score rank, crowding, liquidity, volatility/trend and causal
transition state, did the executable response still change?

The two source families are kept separate.  February--April uses the canonical
base OOF panel.  May--July uses the exact all-score ledger joined one-to-one to
the outcome-free Primary100 universe.  No mapping, response label, future
feature, timestamp-local admission rule, or per-side quota is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
IDENTITY = ("candidate_id", "side_name", "__symbol__", "__ts__")
TOP_FRACTION = 0.10
MIN_ROWS = 250
MIN_COVERAGE = 0.50
MIN_EFFECTIVE_RATIO = 0.15
MIN_EFFECTIVE_ROWS = 100.0
MAX_WEIGHTED_SMD = 0.25

DEFAULT_CANONICAL = ROOT / "data_perp/artifacts/canonical_opportunity_payoff_trust_panel_20260729_v2/panel.parquet"
DEFAULT_CANONICAL_MANIFEST = DEFAULT_CANONICAL.with_name("manifest.json")
DEFAULT_CURRENT = ROOT / "data_perp/artifacts/mayjul2026_exact_allscore_ic_ev_waterfall_20260730_v1/allscore_waterfall.parquet"
DEFAULT_CURRENT_MANIFEST = DEFAULT_CURRENT.with_name("manifest.json")
DEFAULT_CURRENT_UNIVERSE = ROOT / "data_perp/artifacts/exact_policy_capture_feature_universe_20260727_v2/capture_feature_universe.parquet"
DEFAULT_CURRENT_UNIVERSE_MANIFEST = DEFAULT_CURRENT_UNIVERSE.with_name("manifest.json")
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/matched_month_pair_conversion_shift_20260730_v1"


class DiagnosticError(RuntimeError):
    """Raised when frozen inputs cannot prove the diagnostic's contract."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(_safe(dict(payload)), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _require(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise DiagnosticError(f"{name} lacks required columns: {missing}")


def _normalise_identity(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    _require(frame, IDENTITY, name)
    result = frame.copy()
    for field in ("candidate_id", "side_name", "__symbol__"):
        result[field] = result[field].astype(str)
    result["side_name"] = result["side_name"].str.lower()
    result["__ts__"] = pd.to_datetime(result["__ts__"], utc=True, errors="raise")
    if result.duplicated(list(IDENTITY)).any() or result.candidate_id.duplicated().any():
        raise DiagnosticError(f"{name} lacks unique frozen candidate identities")
    return result


def stable_top(frame: pd.DataFrame, score: str, fraction: float = TOP_FRACTION) -> pd.DataFrame:
    """Select one pooled global book with deterministic candidate-ID ties."""

    if frame.empty:
        raise DiagnosticError("cannot select a top tail from an empty frame")
    count = max(1, int(math.ceil(len(frame) * float(fraction))))
    values = pd.to_numeric(frame[score], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise DiagnosticError(f"{score} has non-finite frozen values")
    order = np.lexsort((frame.candidate_id.astype(str).to_numpy(), -values))
    return frame.iloc[order[:count]].copy()


def _global_ventile(frame: pd.DataFrame, score: str) -> pd.Series:
    """Rank-bin frozen scores globally inside every month, preserving ties deterministically."""

    result = pd.Series(index=frame.index, dtype="int8")
    for _, local in frame.groupby("candidate_month", observed=True, sort=False):
        order = np.lexsort((local.candidate_id.astype(str).to_numpy(), pd.to_numeric(local[score], errors="raise").to_numpy(float)))
        ranks = np.empty(len(local), dtype=np.int64)
        ranks[order] = np.arange(len(local), dtype=np.int64)
        result.loc[local.index] = np.minimum((ranks * 20) // len(local), 19).astype("int8")
    return result.astype("int8")


def _group_size(frame: pd.DataFrame) -> pd.Series:
    return frame.groupby("__ts__", observed=True)["candidate_id"].transform("size").astype(float)


def _quantile_code(values: pd.Series, *, bins: int = 4) -> pd.Series:
    """Pair-local covariate bucket; constant/missing values remain explicit."""

    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    result = pd.Series("missing", index=values.index, dtype="object")
    if finite.empty:
        return result
    edges = np.unique(np.quantile(finite.to_numpy(float), np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 2:
        result.loc[finite.index] = "constant"
        return result
    codes = np.searchsorted(edges[1:-1], finite.to_numpy(float), side="right")
    result.loc[finite.index] = pd.Series(codes, index=finite.index).astype(str).map(lambda item: f"q{item}")
    return result


def _exit_class(frame: pd.DataFrame) -> pd.Series:
    raw = frame.get("execution_exit_class", frame.get("execution_exit_reason", pd.Series("unknown", index=frame.index)))
    value = raw.astype(str).str.lower()
    value = value.replace({"full_sl": "full_stop", "full_stop": "full_stop"})
    return value.where(value.isin(("trailing", "timeout", "full_stop", "adverse_exit")), "other")


def _numeric(frame: pd.DataFrame, field: str, name: str) -> pd.Series:
    if field not in frame:
        raise DiagnosticError(f"{name} missing causal field {field}")
    value = pd.to_numeric(frame[field], errors="coerce")
    if value.notna().mean() < 0.95 or np.isinf(value.to_numpy(float, na_value=np.nan)).any():
        raise DiagnosticError(f"{name} causal field {field} is unavailable or infinite")
    return value


def _canonical_covariates(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str], dict[str, str]]:
    """Construct pair-local causal variables from canonical frozen pre-entry fields."""

    out = frame.copy()
    fields = {
        # The frozen canonical side-local feature matrices retain different
        # selected columns per side.  Use the all-row, pre-entry volume
        # confirmation state here rather than treating a side-specific missing
        # selected feature as zero.  No direct spread field survives in this
        # canonical panel; that limitation is recorded in the manifest.
        "liq_volume_confirmation": "__regime_source_volume_confirmation_score__",
        "vol_range": "range_24h_pct",
        "volatility": "__meta_raw__volatility_zscore",
        "trend": "trend_r2_24",
        "trend_level": "__regime_source_trend_following_score__",
        "transition_range": "preentry_transition__range_24h_pct__delta_3h",
        "transition_volatility": "preentry_transition__meta_raw__volatility_zscore__delta_3h",
        "transition_trend": "preentry_transition__trend_r2_24__delta_3h",
        "transition_jump": "preentry_transition__jump_intensity__delta_3h",
    }
    for output, source in fields.items():
        out[output] = _numeric(out, source, "canonical panel")
    return out, list(fields), list(fields), fields


def _current_covariates(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str], dict[str, str]]:
    """Construct pair-local causal variables from the outcome-free current universe."""

    out = frame.copy()
    fields = {
        "liq_spread": "capture_candidate__median_spread_bps",
        "liq_amihud": "capture_candidate__amihud_illiq",
        "liq_volume": "capture_candidate__median_volume_z",
        "vol_range": "capture_candidate__range_24h_pct",
        "volatility": "capture_candidate__volatility_zscore",
        "trend": "capture_candidate__ema20_slope_5h",
        "trend_level": "capture_candidate__zscore_price_200",
        "transition_compression": "capture_candidate__atr_compression_ratio",
        "transition_acceleration": "capture_candidate__trend_acceleration",
        "transition_leverage": "capture_candidate__leverage_build_score",
    }
    for output, source in fields.items():
        out[output] = _numeric(out, source, "current outcome-free universe")
    return out, list(fields), list(fields), fields


def _enrich(frame: pd.DataFrame, *, score: str, state_inputs: Sequence[str]) -> pd.DataFrame:
    result = frame.copy()
    result["score_ventile"] = _global_ventile(result, score).astype(str)
    result["candidate_group_rows"] = _group_size(result)
    # This is a compact *causal* state interaction, not a future transition label.
    state_fields = list(state_inputs[-3:])
    state_codes = [_quantile_code(result[field], bins=3).astype(str) for field in state_fields]
    result["transition_state"] = state_codes[0].str.cat(state_codes[1], sep="|").str.cat(state_codes[2], sep="|")
    result["candidate_group_size_bin"] = _quantile_code(result["candidate_group_rows"], bins=5)
    result["exit_class"] = _exit_class(result)
    net = pd.to_numeric(result["execution_net_ev_12h"], errors="raise")
    gross = pd.to_numeric(result["execution_gross_ev_12h"], errors="raise")
    cost = pd.to_numeric(result["execution_cost_return"], errors="raise")
    if not np.allclose(gross.to_numpy(float) - cost.to_numpy(float), net.to_numpy(float), rtol=0.0, atol=1e-7):
        raise DiagnosticError("exact economics violates gross - explicit cost = net")
    result["outcome_opportunity"] = net.gt(0.0)
    result["outcome_nonpositive"] = net.le(0.0)
    return result


def _weighted_mean(values: pd.Series, weights: np.ndarray) -> float:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
    valid = np.isfinite(numeric) & np.isfinite(weights) & (weights > 0.0)
    if not valid.any():
        return float("nan")
    return float(np.average(numeric[valid], weights=weights[valid]))


def _conditional_weighted_mean(values: pd.Series, condition: pd.Series, weights: np.ndarray) -> float:
    mask = condition.to_numpy(bool) & np.isfinite(weights) & (weights > 0.0)
    if not mask.any() or weights[mask].sum() <= 0.0:
        return float("nan")
    return _weighted_mean(values.loc[condition], weights[mask])


def response_metrics(frame: pd.DataFrame, weights: np.ndarray | None = None) -> dict[str, float]:
    """Economically oriented response metrics, in bps where applicable."""

    if weights is None:
        weights = np.ones(len(frame), dtype=float)
    if len(weights) != len(frame):
        raise DiagnosticError("response weights do not align to rows")
    net = pd.to_numeric(frame.execution_net_ev_12h, errors="raise")
    gross = pd.to_numeric(frame.execution_gross_ev_12h, errors="raise")
    cost = pd.to_numeric(frame.execution_cost_return, errors="raise")
    opportunity = net.gt(0.0)
    nonpositive = net.le(0.0)
    exits = frame.exit_class
    return {
        "opportunity_rate": _weighted_mean(opportunity.astype(float), weights),
        "favorable_gross_bps_given_opportunity": _conditional_weighted_mean(gross, opportunity, weights) * 1e4,
        "adverse_severity_bps_given_nonpositive": -_conditional_weighted_mean(net, nonpositive, weights) * 1e4,
        "full_stop_rate": _weighted_mean(exits.eq("full_stop").astype(float), weights),
        "timeout_rate": _weighted_mean(exits.eq("timeout").astype(float), weights),
        "adverse_exit_rate": _weighted_mean(exits.eq("adverse_exit").astype(float), weights),
        "cost_bps": _weighted_mean(cost, weights) * 1e4,
        "net_ev_bps": _weighted_mean(net, weights) * 1e4,
    }


def _effective_sample_size(weights: np.ndarray) -> float:
    valid = weights[np.isfinite(weights) & (weights > 0.0)]
    if len(valid) == 0:
        return 0.0
    return float(valid.sum() ** 2 / np.square(valid).sum())


def _smd(source: pd.Series, target: pd.Series, weights: np.ndarray | None = None) -> float:
    left = pd.to_numeric(source, errors="coerce").to_numpy(float)
    right = pd.to_numeric(target, errors="coerce").to_numpy(float)
    left = left[np.isfinite(left)]; right = right[np.isfinite(right)]
    if len(left) < 2 or len(right) < 2:
        return float("nan")
    if weights is None:
        mean_left = float(left.mean()); variance_left = float(left.var(ddof=1))
    else:
        # `source` and weights have already been jointly filtered by this caller.
        raw = pd.to_numeric(source, errors="coerce").to_numpy(float)
        mask = np.isfinite(raw) & np.isfinite(weights) & (weights > 0.0)
        raw, w = raw[mask], weights[mask]
        if len(raw) < 2:
            return float("nan")
        mean_left = float(np.average(raw, weights=w))
        variance_left = float(np.average(np.square(raw - mean_left), weights=w))
    mean_right = float(right.mean()); variance_right = float(right.var(ddof=1))
    denom = math.sqrt(max((variance_left + variance_right) / 2.0, 1e-12))
    return float((mean_left - mean_right) / denom)


def _make_propensity_pipeline(continuous: Sequence[str], categorical: Sequence[str]) -> Pipeline:
    return Pipeline([
        ("features", ColumnTransformer([
            ("continuous", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), list(continuous)),
            ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), list(categorical)),
        ], sparse_threshold=0.3)),
        ("model", LogisticRegression(C=0.25, max_iter=1000, solver="lbfgs", random_state=20260730)),
    ])


def fit_reweight(
    source: pd.DataFrame,
    target: pd.DataFrame,
    *,
    continuous: Sequence[str],
    categorical: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, dict[str, Any], pd.DataFrame]:
    """Fit a covariate-only odds reweighting and enforce observed common support."""

    if len(source) < MIN_ROWS or len(target) < MIN_ROWS:
        raise DiagnosticError("month tail has insufficient rows for common-support diagnosis")
    work = pd.concat([source.assign(__is_target__=0), target.assign(__is_target__=1)], ignore_index=True)
    pipe = _make_propensity_pipeline(continuous, categorical)
    pipe.fit(work.loc[:, [*continuous, *categorical]], work.__is_target__)
    p = pipe.predict_proba(work.loc[:, [*continuous, *categorical]])[:, 1]
    source_p, target_p = p[: len(source)], p[len(source):]
    low = max(float(np.min(source_p)), float(np.min(target_p)))
    high = min(float(np.max(source_p)), float(np.max(target_p)))
    source_keep = (source_p >= low) & (source_p <= high)
    target_keep = (target_p >= low) & (target_p <= high)
    kept_source, kept_target = source.loc[source_keep].copy(), target.loc[target_keep].copy()
    kept_source_p = source_p[source_keep]
    odds = (kept_source_p / np.maximum(1.0 - kept_source_p, 1e-8)) * (len(source) / len(target))
    # A predeclared cap limits a few nearly-separable covariate rows from making
    # a descriptive standardisation look precise.  Normalised means are cap-invariant.
    weights = np.clip(odds, 0.0, 20.0)
    balance_rows: list[dict[str, Any]] = []
    for field in continuous:
        balance_rows.append({"covariate": field, "kind": "continuous", "before_smd": _smd(kept_source[field], kept_target[field]), "after_smd": _smd(kept_source[field], kept_target[field], weights)})
    for field in categorical:
        values = sorted(set(kept_source[field].astype(str)).union(kept_target[field].astype(str)))
        for value in values:
            left = kept_source[field].astype(str).eq(value).astype(float)
            right = kept_target[field].astype(str).eq(value).astype(float)
            balance_rows.append({"covariate": f"{field}={value}", "kind": "categorical", "before_smd": _smd(left, right), "after_smd": _smd(left, right, weights)})
    balance = pd.DataFrame(balance_rows)
    finite_after = balance.after_smd.abs().replace([np.inf, -np.inf], np.nan).dropna()
    summary = {
        "source_rows": int(len(source)), "target_rows": int(len(target)),
        "source_supported_rows": int(len(kept_source)), "target_supported_rows": int(len(kept_target)),
        "source_support_coverage": float(len(kept_source) / len(source)), "target_support_coverage": float(len(kept_target) / len(target)),
        "propensity_common_support_low": low, "propensity_common_support_high": high,
        "weight_cap": 20.0, "weight_max": float(weights.max()) if len(weights) else np.nan,
        "weight_ess": _effective_sample_size(weights), "weight_ess_ratio": float(_effective_sample_size(weights) / max(len(kept_source), 1)),
        "max_abs_smd_before": float(balance.before_smd.abs().max()),
        "max_abs_smd_after": float(finite_after.max()) if not finite_after.empty else np.nan,
    }
    summary["common_support_pass"] = bool(
        summary["source_support_coverage"] >= MIN_COVERAGE
        and summary["target_support_coverage"] >= MIN_COVERAGE
        and summary["weight_ess"] >= MIN_EFFECTIVE_ROWS
        and summary["weight_ess_ratio"] >= MIN_EFFECTIVE_RATIO
        and np.isfinite(summary["max_abs_smd_after"])
        and summary["max_abs_smd_after"] <= MAX_WEIGHTED_SMD
    )
    return kept_source, kept_target, weights, summary, balance


def diagnose_pair(
    source: pd.DataFrame,
    target: pd.DataFrame,
    *,
    family: str,
    from_month: str,
    to_month: str,
    continuous: Sequence[str],
    categorical: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Produce a supported population, balance audit, and additive response shift."""

    try:
        supported_source, supported_target, weights, coverage, balance = fit_reweight(source, target, continuous=continuous, categorical=categorical)
    except DiagnosticError as exc:
        coverage = {"common_support_pass": False, "failure_reason": str(exc), "source_rows": len(source), "target_rows": len(target)}
        empty = pd.DataFrame()
        return empty, pd.DataFrame([{**coverage, "source_family": family, "from_month": from_month, "to_month": to_month}]), empty
    coverage.update({"source_family": family, "from_month": from_month, "to_month": to_month})
    balance = balance.assign(source_family=family, from_month=from_month, to_month=to_month, common_support_pass=coverage["common_support_pass"])
    # Preserve all rows and supported rows separately.  Support removal itself is
    # explicit, rather than silently attributing it to a conditional response.
    source_all = response_metrics(source)
    target_all = response_metrics(target)
    source_supported = response_metrics(supported_source)
    target_supported = response_metrics(supported_target)
    source_reweighted = response_metrics(supported_source, weights)
    rows: list[dict[str, Any]] = []
    for metric in source_all:
        raw_delta = target_all[metric] - source_all[metric]
        supported_delta = target_supported[metric] - source_supported[metric]
        # The change caused by restricting the raw all-row comparison to the
        # observed-overlap subset.  With this sign, raw = restriction +
        # composition + conditional response exactly.
        support_selection = raw_delta - supported_delta
        composition = source_reweighted[metric] - source_supported[metric]
        conditional = target_supported[metric] - source_reweighted[metric]
        rows.append({
            "source_family": family, "from_month": from_month, "to_month": to_month, "metric": metric,
            "source_all": source_all[metric], "target_all": target_all[metric],
            "source_supported": source_supported[metric], "target_supported": target_supported[metric],
            "source_reweighted_to_target": source_reweighted[metric],
            "raw_all_delta": raw_delta, "support_selection_delta": support_selection,
            "composition_shift": composition, "conditional_response_shift": conditional,
            "reconciliation_error": support_selection + composition + conditional - raw_delta,
            "common_support_pass": coverage["common_support_pass"],
        })
    return pd.DataFrame(rows), pd.DataFrame([coverage]), balance


def _load_canonical(path: Path) -> tuple[pd.DataFrame, list[str], list[str], dict[str, str]]:
    frame = _normalise_identity(pd.read_parquet(path), "canonical panel")
    required = ["candidate_month", "base_oof_score", "execution_gross_ev_12h", "execution_cost_return", "execution_net_ev_12h", "execution_exit_reason"]
    _require(frame, required, "canonical panel")
    frame, continuous, state_inputs, field_map = _canonical_covariates(frame)
    frame = _enrich(frame, score="base_oof_score", state_inputs=state_inputs)
    return frame, continuous, field_map, {"score": "base_oof_score", "source": "canonical_base_oof"}


def _load_current(ledger_path: Path, universe_path: Path) -> tuple[pd.DataFrame, list[str], list[str], dict[str, str]]:
    ledger = _normalise_identity(pd.read_parquet(ledger_path), "current all-score ledger")
    universe = _normalise_identity(pd.read_parquet(universe_path), "current outcome-free universe")
    _require(ledger, ["candidate_month", "score_base_alpha", "execution_gross_ev_12h", "execution_cost_return", "execution_net_ev_12h", "execution_exit_reason"], "current all-score ledger")
    _require(universe, ["base_oof_score", "execution_gross_ev_12h", "execution_cost_return", "execution_net_ev_12h"], "current outcome-free universe")
    joined = ledger.merge(universe, on=list(IDENTITY), how="left", validate="one_to_one", suffixes=("", "__universe"), indicator=True)
    if not joined._merge.eq("both").all():
        raise DiagnosticError("current all-score ledger does not fully join to outcome-free universe")
    for field in ("execution_gross_ev_12h", "execution_cost_return", "execution_net_ev_12h"):
        if not np.allclose(pd.to_numeric(joined[field], errors="raise"), pd.to_numeric(joined[f"{field}__universe"], errors="raise"), rtol=0.0, atol=1e-7):
            raise DiagnosticError(f"current ledger/universe {field} differs")
    if not np.allclose(pd.to_numeric(joined.score_base_alpha, errors="raise"), pd.to_numeric(joined.base_oof_score, errors="raise"), rtol=0.0, atol=1e-7):
        raise DiagnosticError("current ledger base score differs from outcome-free universe")
    joined, continuous, state_inputs, field_map = _current_covariates(joined)
    joined = _enrich(joined, score="score_base_alpha", state_inputs=state_inputs)
    return joined, continuous, field_map, {"score": "score_base_alpha", "source": "current_allscore_exact_joined_outcome_free_universe"}


def _checked_input(path: Path, manifest: Path, name: str) -> dict[str, str]:
    if not path.is_file() or not manifest.is_file():
        raise FileNotFoundError(f"frozen {name} input or manifest is absent")
    return {"path": str(path), "sha256": sha256(path), "manifest_path": str(manifest), "manifest_sha256": sha256(manifest)}


def run(
    *, canonical: Path, canonical_manifest: Path, current: Path, current_manifest: Path,
    current_universe: Path, current_universe_manifest: Path, output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite immutable output {output_dir}")
    inputs = {
        "canonical": _checked_input(canonical, canonical_manifest, "canonical"),
        "current": _checked_input(current, current_manifest, "current"),
        "current_universe": _checked_input(current_universe, current_universe_manifest, "current universe"),
    }
    canonical_frame, canonical_cont, canonical_fields, canonical_meta = _load_canonical(canonical)
    current_frame, current_cont, current_fields, current_meta = _load_current(current, current_universe)
    configs = [
        ("canonical_base_oof", canonical_frame, canonical_cont, canonical_meta["score"], (("2025-02", "2025-03"), ("2025-03", "2025-04"))),
        ("current_exact_base_oof", current_frame, current_cont, current_meta["score"], (("2026-05", "2026-06"), ("2026-06", "2026-07"))),
    ]
    response_parts: list[pd.DataFrame] = []
    coverage_parts: list[pd.DataFrame] = []
    balance_parts: list[pd.DataFrame] = []
    selected_parts: list[pd.DataFrame] = []
    categorical = ("side_name", "__symbol__", "score_ventile", "candidate_group_size_bin", "transition_state")
    for family, frame, continuous, score, pairs in configs:
        for first, second in pairs:
            source_all = frame.loc[frame.candidate_month.astype(str).eq(first)].copy()
            target_all = frame.loc[frame.candidate_month.astype(str).eq(second)].copy()
            if source_all.empty or target_all.empty:
                raise DiagnosticError(f"{family} is missing requested pair {first}->{second}")
            source = stable_top(source_all, score); target = stable_top(target_all, score)
            selected_parts.extend((
                source.loc[:, [*IDENTITY, "candidate_month", score, "score_ventile", "candidate_group_rows", "candidate_group_size_bin", "transition_state"]].assign(source_family=family, selection="pooled_global_top10", pair_from=first, pair_to=second, pair_role="source"),
                target.loc[:, [*IDENTITY, "candidate_month", score, "score_ventile", "candidate_group_rows", "candidate_group_size_bin", "transition_state"]].assign(source_family=family, selection="pooled_global_top10", pair_from=first, pair_to=second, pair_role="target"),
            ))
            response, coverage, balance = diagnose_pair(source, target, family=family, from_month=first, to_month=second, continuous=continuous, categorical=categorical)
            response_parts.append(response); coverage_parts.append(coverage); balance_parts.append(balance)
    response = pd.concat(response_parts, ignore_index=True)
    coverage = pd.concat(coverage_parts, ignore_index=True)
    balance = pd.concat(balance_parts, ignore_index=True)
    selected = pd.concat(selected_parts, ignore_index=True)
    if not response.empty and not np.allclose(response.reconciliation_error.fillna(0.0), 0.0, rtol=0.0, atol=1e-10):
        raise DiagnosticError("response decomposition does not reconcile")
    stage = output_dir.parent / f".{output_dir.name}.{uuid.uuid4().hex}.stage"
    stage.mkdir(parents=True, exist_ok=False)
    try:
        outputs: dict[str, dict[str, Any]] = {}
        for name, table in (("response_decomposition", response), ("coverage", coverage), ("balance", balance), ("frozen_selection", selected)):
            target = stage / f"{name}.parquet"
            table.to_parquet(target, index=False, compression="zstd")
            outputs[name] = {"path": str(output_dir / target.name), "rows": int(len(table)), "sha256": sha256(target)}
        report = {
            "schema": "matched_month_pair_conversion_shift_v1",
            "status": "DIAGNOSTIC_ONLY_NO_MAPPING_NO_PROMOTION",
            "contracts": {
                "selection": "Frozen raw base-score pooled-global monthly top 10%; score descending and candidate-ID ascending ties. No per-timestamp, side, asset, or state quota.",
                "standardisation": "Source month is propensity-odds reweighted to target covariates after observed propensity common-support trimming. The propensity uses causal fields only; no outcome, mapping, future, or policy field.",
                "matching": "Side, asset, base-score ventile, candidate-group size bin, and compact causal transition state are categorical reweighting inputs; liquidity, volatility/range, trend, and transition measurements are continuous inputs. The canonical panel has no all-row direct spread field, so its all-row volume-confirmation liquidity state is used instead; cost is intentionally not a matching covariate because it is a response component.",
                "decomposition": "raw all-row delta = common-support restriction + covariate/composition shift + conditional response shift, separately for opportunity, favourable payoff, adverse severity, stop/timeout mix, cost and exact net EV.",
                "fail_closed": {"minimum_tail_rows": MIN_ROWS, "minimum_source_or_target_support_coverage": MIN_COVERAGE, "minimum_weight_ess": MIN_EFFECTIVE_ROWS, "minimum_weight_ess_ratio": MIN_EFFECTIVE_RATIO, "maximum_abs_weighted_smd": MAX_WEIGHTED_SMD},
                "promotion": "forbidden",
            },
            "inputs": inputs,
            "families": {"canonical": {**canonical_meta, "causal_field_map": canonical_fields}, "current": {**current_meta, "causal_field_map": current_fields}},
            "outputs": outputs,
            "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
            "promotion_eligible": False,
        }
        _write_json(stage / "manifest.json", report)
        (stage / "manifest.sha256").write_text(sha256(stage / "manifest.json") + "\n", encoding="utf-8")
        os.replace(stage, output_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    result.add_argument("--canonical-manifest", type=Path, default=DEFAULT_CANONICAL_MANIFEST)
    result.add_argument("--current", type=Path, default=DEFAULT_CURRENT)
    result.add_argument("--current-manifest", type=Path, default=DEFAULT_CURRENT_MANIFEST)
    result.add_argument("--current-universe", type=Path, default=DEFAULT_CURRENT_UNIVERSE)
    result.add_argument("--current-universe-manifest", type=Path, default=DEFAULT_CURRENT_UNIVERSE_MANIFEST)
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    print(json.dumps(_safe(run(canonical=args.canonical, canonical_manifest=args.canonical_manifest, current=args.current, current_manifest=args.current_manifest, current_universe=args.current_universe, current_universe_manifest=args.current_universe_manifest, output_dir=args.output_dir)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
