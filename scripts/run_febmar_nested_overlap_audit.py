#!/usr/bin/env python3
"""Sealed February-to-March nested common-support audit.

This is a diagnostic only.  The model which estimates support sees selected,
pre-entry context only: never an execution label, an EV map, or a policy
decision.  Outcome decomposition is deliberately absent for a covariate set
unless that set clears the predeclared common-support, ESS and balance gates.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
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
COMMON = ROOT / "scripts/run_matched_month_pair_conversion_shift.py"
DEFAULT_PANEL = ROOT / "data_perp/artifacts/canonical_opportunity_payoff_trust_panel_20260729_v2/panel.parquet"
DEFAULT_MANIFEST = DEFAULT_PANEL.with_name("manifest.json")
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/febmar_nested_overlap_audit_20260730_v1"
IDENTITY = ("candidate_id", "side_name", "__symbol__", "__ts__")
MIN_ROWS = 250
MIN_COVERAGE = 0.50
MIN_EFFECTIVE_ROWS = 100.0
MIN_EFFECTIVE_RATIO = 0.15
MAX_WEIGHTED_SMD = 0.25
N_BOOTSTRAP = 250


class AuditError(RuntimeError):
    pass


def _load_common() -> Any:
    spec = importlib.util.spec_from_file_location("matched_conversion_common", COMMON)
    if spec is None or spec.loader is None:
        raise AuditError("cannot load canonical conversion helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def _effective_sample_size(weights: np.ndarray) -> float:
    weight = weights[np.isfinite(weights) & (weights > 0)]
    return 0.0 if len(weight) == 0 else float(weight.sum() ** 2 / np.square(weight).sum())


def _smd(source: pd.Series, target: pd.Series, weights: np.ndarray | None = None) -> float:
    left = pd.to_numeric(source, errors="coerce").to_numpy(float)
    right = pd.to_numeric(target, errors="coerce").to_numpy(float)
    if weights is None:
        left = left[np.isfinite(left)]
        if len(left) < 2:
            return float("nan")
        left_mean, left_var = float(left.mean()), float(left.var(ddof=1))
    else:
        keep = np.isfinite(left) & np.isfinite(weights) & (weights > 0)
        left, weight = left[keep], weights[keep]
        if len(left) < 2:
            return float("nan")
        left_mean = float(np.average(left, weights=weight))
        left_var = float(np.average(np.square(left - left_mean), weights=weight))
    right = right[np.isfinite(right)]
    if len(right) < 2:
        return float("nan")
    right_mean, right_var = float(right.mean()), float(right.var(ddof=1))
    return float((left_mean - right_mean) / math.sqrt(max((left_var + right_var) / 2.0, 1e-12)))


def _pipeline(continuous: Sequence[str], categorical: Sequence[str]) -> Pipeline:
    return Pipeline([
        ("features", ColumnTransformer([
            ("continuous", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), list(continuous)),
            ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), list(categorical)),
        ], sparse_threshold=0.3)),
        ("model", LogisticRegression(C=0.25, max_iter=1000, solver="lbfgs", random_state=20260730)),
    ])


def _balance(source: pd.DataFrame, target: pd.DataFrame, weights: np.ndarray, continuous: Sequence[str], categorical: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for field in continuous:
        rows.append({"covariate": field, "kind": "continuous", "before_smd": _smd(source[field], target[field]), "after_smd": _smd(source[field], target[field], weights)})
    for field in categorical:
        for value in sorted(set(source[field].astype(str)).union(target[field].astype(str))):
            left = source[field].astype(str).eq(value).astype(float)
            right = target[field].astype(str).eq(value).astype(float)
            rows.append({"covariate": f"{field}={value}", "kind": "categorical", "before_smd": _smd(left, right), "after_smd": _smd(left, right, weights)})
    return pd.DataFrame(rows)


def fit_support(source: pd.DataFrame, target: pd.DataFrame, *, continuous: Sequence[str], categorical: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, dict[str, Any], pd.DataFrame]:
    """Outcome-blind observed-overlap fit.  No label column enters ``X``."""

    if len(source) < MIN_ROWS or len(target) < MIN_ROWS:
        raise AuditError("a selected month has fewer than the predeclared 250 rows")
    fields = [*continuous, *categorical]
    work = pd.concat([source.loc[:, fields].assign(__is_target__=0), target.loc[:, fields].assign(__is_target__=1)], ignore_index=True)
    model = _pipeline(continuous, categorical)
    model.fit(work.loc[:, fields], work.__is_target__)
    propensity = model.predict_proba(work.loc[:, fields])[:, 1]
    source_p, target_p = propensity[:len(source)], propensity[len(source):]
    low = max(float(source_p.min()), float(target_p.min()))
    high = min(float(source_p.max()), float(target_p.max()))
    source_keep, target_keep = (source_p >= low) & (source_p <= high), (target_p >= low) & (target_p <= high)
    supported_source, supported_target = source.loc[source_keep].copy(), target.loc[target_keep].copy()
    supported_source_p, supported_target_p = source_p[source_keep], target_p[target_keep]
    # Target-standardisation is primary; the cap is fixed before outcomes are inspected.
    odds_weights = np.clip((supported_source_p / np.maximum(1.0 - supported_source_p, 1e-8)) * (len(source) / len(target)), 0.0, 20.0)
    balance = _balance(supported_source, supported_target, odds_weights, continuous, categorical)
    after = balance.after_smd.abs().replace([np.inf, -np.inf], np.nan).dropna()
    ess = _effective_sample_size(odds_weights)
    summary: dict[str, Any] = {
        "source_rows": int(len(source)), "target_rows": int(len(target)),
        "source_supported_rows": int(len(supported_source)), "target_supported_rows": int(len(supported_target)),
        "source_support_coverage": float(len(supported_source) / len(source)), "target_support_coverage": float(len(supported_target) / len(target)),
        "propensity_common_support_low": low, "propensity_common_support_high": high,
        "weight_cap": 20.0, "weight_max": float(odds_weights.max()) if len(odds_weights) else np.nan,
        "weight_ess": ess, "weight_ess_ratio": float(ess / max(len(supported_source), 1)),
        "max_abs_smd_before": float(balance.before_smd.abs().max()), "max_abs_smd_after": float(after.max()) if not after.empty else np.nan,
    }
    summary["common_support_pass"] = bool(
        summary["source_support_coverage"] >= MIN_COVERAGE and summary["target_support_coverage"] >= MIN_COVERAGE
        and ess >= MIN_EFFECTIVE_ROWS and summary["weight_ess_ratio"] >= MIN_EFFECTIVE_RATIO
        and np.isfinite(summary["max_abs_smd_after"]) and summary["max_abs_smd_after"] <= MAX_WEIGHTED_SMD
    )
    # Separate overlap weights are an explicitly secondary estimand, never a support repair.
    overlap_source = 1.0 - supported_source_p
    overlap_target = supported_target_p
    return supported_source, supported_target, odds_weights, overlap_source, overlap_target, summary, balance


def _cohorts(frame: pd.DataFrame, keep: np.ndarray, *, role: str, covariate_set: str) -> pd.DataFrame:
    out = frame.loc[:, ["side_name", "__symbol__", "score_ventile", "candidate_group_size_bin", "transition_state"]].copy()
    out["excluded_by_common_support"] = ~keep
    out["role"] = role
    out["covariate_set"] = covariate_set
    return out.groupby(["covariate_set", "role", "excluded_by_common_support", "side_name", "__symbol__", "score_ventile", "candidate_group_size_bin", "transition_state"], dropna=False, observed=True).size().rename("rows").reset_index()


def _sample_days(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    date = pd.to_datetime(frame.__ts__, utc=True).dt.date
    days = np.asarray(sorted(date.unique()))
    chosen = rng.choice(days, size=len(days), replace=True)
    return pd.concat([frame.loc[date.eq(day)] for day in chosen], ignore_index=True)


def _metrics(common: Any, frame: pd.DataFrame, weights: np.ndarray | None = None) -> dict[str, float]:
    return common.response_metrics(frame, weights)


def _decompose(common: Any, source: pd.DataFrame, target: pd.DataFrame, supported_source: pd.DataFrame, supported_target: pd.DataFrame, weights: np.ndarray) -> list[dict[str, float]]:
    all_source, all_target = _metrics(common, source), _metrics(common, target)
    cut_source, cut_target = _metrics(common, supported_source), _metrics(common, supported_target)
    weighted_source = _metrics(common, supported_source, weights)
    rows = []
    for metric in all_source:
        raw = all_target[metric] - all_source[metric]
        supported = cut_target[metric] - cut_source[metric]
        support = raw - supported
        composition = weighted_source[metric] - cut_source[metric]
        conditional = cut_target[metric] - weighted_source[metric]
        rows.append({"metric": metric, "source_all": all_source[metric], "target_all": all_target[metric], "source_supported": cut_source[metric], "target_supported": cut_target[metric], "source_reweighted_to_target": weighted_source[metric], "raw_all_delta": raw, "support_selection_delta": support, "composition_shift": composition, "conditional_response_shift": conditional, "reconciliation_error": support + composition + conditional - raw})
    return rows


def _day_block_intervals(common: Any, source: pd.DataFrame, target: pd.DataFrame, supported_source: pd.DataFrame, supported_target: pd.DataFrame, weights: np.ndarray) -> dict[str, dict[str, tuple[float, float]]]:
    """Fixed-support, fixed-weight day-block intervals; support is not outcome fitted."""
    rng = np.random.default_rng(20260730)
    metrics = list(_metrics(common, source))
    values: dict[str, dict[str, list[float]]] = {metric: {field: [] for field in ("raw_all_delta", "support_selection_delta", "composition_shift", "conditional_response_shift")} for metric in metrics}
    weighted = supported_source.copy(); weighted["__audit_weight__"] = weights
    for _ in range(N_BOOTSTRAP):
        source_b, target_b = _sample_days(source, rng), _sample_days(target, rng)
        left_b, right_b = _sample_days(weighted, rng), _sample_days(supported_target, rng)
        rows = _decompose(common, source_b, target_b, left_b.drop(columns="__audit_weight__"), right_b, left_b.__audit_weight__.to_numpy(float))
        for row in rows:
            for field in values[row["metric"]]:
                values[row["metric"]][field].append(row[field])
    return {metric: {field: (float(np.quantile(draws, .025)), float(np.quantile(draws, .975))) for field, draws in fields.items()} for metric, fields in values.items()}


def _overlap_sensitivity(common: Any, source: pd.DataFrame, target: pd.DataFrame, source_weights: np.ndarray, target_weights: np.ndarray) -> pd.DataFrame:
    left, right = _metrics(common, source, source_weights), _metrics(common, target, target_weights)
    return pd.DataFrame([{"metric": metric, "overlap_weighted_source": left[metric], "overlap_weighted_target": right[metric], "overlap_conditional_response_shift": right[metric] - left[metric]} for metric in left])


def _nested_configs() -> tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...]:
    core = ("side_name", "__symbol__", "score_ventile", "candidate_group_size_bin")
    lvt = ("liq_volume_confirmation", "vol_range", "volatility", "trend", "trend_level")
    transition = ("transition_range", "transition_volatility", "transition_trend", "transition_jump")
    return (
        ("core_score_context", (), core),
        ("core_plus_liquidity_volatility_trend", lvt, core),
        ("core_plus_lvt_plus_transition_state", (*lvt, *transition), (*core, "transition_state")),
    )


def run(*, panel: Path, manifest: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite immutable output {output_dir}")
    if not panel.is_file() or not manifest.is_file():
        raise FileNotFoundError("canonical panel or manifest is absent")
    common = _load_common()
    frame, _, _, _ = common._load_canonical(panel)
    source_all = frame.loc[frame.candidate_month.astype(str).eq("2025-02")].copy()
    target_all = frame.loc[frame.candidate_month.astype(str).eq("2025-03")].copy()
    if source_all.empty or target_all.empty:
        raise AuditError("canonical panel lacks February or March")
    source, target = common.stable_top(source_all, "base_oof_score"), common.stable_top(target_all, "base_oof_score")
    coverage, balance, cohorts, response, sensitivity = [], [], [], [], []
    for name, continuous, categorical in _nested_configs():
        supported_source, supported_target, odds, overlap_source, overlap_target, summary, balanced = fit_support(source, target, continuous=continuous, categorical=categorical)
        # Reconstruct keep masks from frozen candidate identities; no outcome access is involved.
        source_keep = source.candidate_id.isin(supported_source.candidate_id).to_numpy()
        target_keep = target.candidate_id.isin(supported_target.candidate_id).to_numpy()
        summary.update({"covariate_set": name, "continuous_covariates": list(continuous), "categorical_covariates": list(categorical), "outcome_decomposition_status": "RUN" if summary["common_support_pass"] else "NOT_RUN_FAILED_COMMON_SUPPORT"})
        coverage.append(summary)
        balance.append(balanced.assign(covariate_set=name, common_support_pass=summary["common_support_pass"]))
        cohorts.extend((_cohorts(source, source_keep, role="source_february", covariate_set=name), _cohorts(target, target_keep, role="target_march", covariate_set=name)))
        if not summary["common_support_pass"]:
            continue
        intervals = _day_block_intervals(common, source, target, supported_source, supported_target, odds)
        rows = _decompose(common, source, target, supported_source, supported_target, odds)
        for row in rows:
            row.update({"covariate_set": name, "common_support_pass": True, "interval_method": f"fixed_support_fixed_weight_day_block_bootstrap_{N_BOOTSTRAP}"})
            for field, (low, high) in intervals[row["metric"]].items():
                row[f"{field}_ci95_low"], row[f"{field}_ci95_high"] = low, high
        response.extend(rows)
        sensitivity.append(_overlap_sensitivity(common, supported_source, supported_target, overlap_source, overlap_target).assign(covariate_set=name, sensitivity="overlap_weights_not_a_support_repair"))
    coverage_df = pd.DataFrame(coverage)
    balance_df = pd.concat(balance, ignore_index=True)
    cohorts_df = pd.concat(cohorts, ignore_index=True)
    response_df = pd.DataFrame(response)
    sensitivity_df = pd.concat(sensitivity, ignore_index=True) if sensitivity else pd.DataFrame(columns=["covariate_set", "metric", "overlap_conditional_response_shift"])
    # January has an exact but incompatible historical score/label lineage.  It has no
    # canonical all-row covariate contract and uses a different score and cost setup;
    # it is intentionally recorded, rather than silently pooled with February.
    january = {
        "status": "NOT_RUN_LINEAGE_INCOMPATIBLE",
        "reason": "January's strict-OOF ledger has historical_base_soft_oof and a separate exact-1m/100bps label lineage, but lacks the canonical causal covariate panel and canonical base-score contract. It cannot enlarge February support without an unproven score/calibration bridge.",
        "candidate_artifact": str(ROOT / "data_perp/artifacts/janfeb2025_execution_ev_exact1m_two_layer_oof_20260727_v1/two_layer_direct_ev_strict_oof.parquet"),
        "lineage_handling": "excluded; no shared calibration, propensity fit, selection, or outcome pooling",
    }
    stage = output_dir.parent / f".{output_dir.name}.{uuid.uuid4().hex}.stage"
    stage.mkdir(parents=True, exist_ok=False)
    try:
        outputs: dict[str, Any] = {}
        for name, table in (("coverage", coverage_df), ("balance", balance_df), ("excluded_cohorts", cohorts_df), ("conditional_response_decomposition", response_df), ("overlap_weight_sensitivity", sensitivity_df)):
            target_path = stage / f"{name}.parquet"; table.to_parquet(target_path, index=False, compression="zstd")
            outputs[name] = {"path": str(output_dir / target_path.name), "rows": int(len(table)), "sha256": sha256(target_path)}
        report = {
            "schema": "febmar_nested_overlap_audit_v1", "status": "DIAGNOSTIC_ONLY_NO_MAPPING_NO_POLICY_ACTION", "promotion_eligible": False,
            "inputs": {"canonical_panel": {"path": str(panel), "sha256": sha256(panel)}, "canonical_manifest": {"path": str(manifest), "sha256": sha256(manifest)}},
            "selection": "Frozen exact canonical base-OOF pooled-global monthly top 10%, score descending/candidate_id ascending tie break; no timestamp, side or asset quota.",
            "covariate_sets": [{"name": n, "continuous": list(c), "categorical": list(k)} for n, c, k in _nested_configs()],
            "support_contract": {"outcome_blind": True, "gates": {"minimum_rows_per_month": MIN_ROWS, "minimum_each_support_coverage": MIN_COVERAGE, "minimum_weight_ess": MIN_EFFECTIVE_ROWS, "minimum_weight_ess_ratio": MIN_EFFECTIVE_RATIO, "maximum_abs_weighted_smd": MAX_WEIGHTED_SMD}, "failed_sets": "no conditional-response decomposition and no sensitivity estimate"},
            "outcome_contract": "When and only when common support passes, report exact net EV, positive-net opportunity, gross favourable payoff conditional on opportunity, adverse net severity conditional on nonpositive, full-stop and timeout rates. Intervals are fixed-support/fixed-weight day-block bootstrap intervals.",
            "january_sensitivity": january, "outputs": outputs, "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        }
        _write_json(stage / "manifest.json", report)
        (stage / "manifest.sha256").write_text(sha256(stage / "manifest.json") + "\n", encoding="utf-8")
        os.replace(stage, output_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True); raise
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    result.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    print(json.dumps(_safe(run(panel=args.panel, manifest=args.manifest, output_dir=args.output_dir)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
