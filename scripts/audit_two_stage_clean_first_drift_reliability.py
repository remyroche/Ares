#!/usr/bin/env python3
"""Audit historical-to-July drift and reliability for the frozen clean-first gate.

This is a research diagnostic, not a model-selection or promotion path.  It
binds the 249 causal candidate features and the frozen clean-first challenger
predictions.  Reliability gates are selected solely from historical temporal
OOF predictions resolved before July 20, then evaluated once on the exact
July-20--23 population.
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

import joblib
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "two_stage_clean_first_drift_reliability_audit_v1"
IDENTITY = ("candidate_id", "__ts__", "__symbol__", "side_name")
HISTORY_CUTOFF = pd.Timestamp("2026-07-20T00:00:00Z")
DEFAULT_CHALLENGER = ROOT / "data_perp/artifacts/historical_to_july_meaningful_mfe_gate_challenger_20260730_v2"
DEFAULT_FEATURES = ROOT / "data_perp/artifacts/exact_policy_capture_feature_universe_20260727_v2/capture_feature_universe.parquet"
DEFAULT_FEATURE_MANIFEST = DEFAULT_FEATURES.parent / "feature_universe_manifest.json"
DEFAULT_CURRENT_RAW = ROOT / "data_perp/artifacts/execution_ev_july20_23_retrospective_20260730_v2/candidates/candidate_features.parquet"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/two_stage_clean_first_drift_reliability_20260730_v1"
SIDES = ("long", "short")
MIN_CURRENT_SIDE_POPULATION_COVERAGE = 0.01
PROB_CLEAN = "catboost_hard_clean_first__probability"
PROB_ADVERSE = "catboost_adverse_1atr_gate__probability"
ADMISSION = "catboost_hard_clean_first__historical_admission"


class AuditError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(_safe(value), indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _record(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": _sha256(path)}


def _finite(values: pd.Series) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)


def _psi(reference: np.ndarray, evaluation: np.ndarray, bins: int = 10) -> float:
    ref = reference[np.isfinite(reference)]; cur = evaluation[np.isfinite(evaluation)]
    if len(ref) < 20 or len(cur) < 5 or np.nanstd(ref) == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 3:
        return float("nan")
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(ref, bins=edges)[0].astype(float) / len(ref)
    q = np.histogram(cur, bins=edges)[0].astype(float) / len(cur)
    p, q = np.clip(p, 1e-6, None), np.clip(q, 1e-6, None)
    return float(np.sum((q - p) * np.log(q / p)))


def feature_drift(history: pd.DataFrame, current: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    """Side-specific PSI and Wasserstein drift over the immutable raw features."""
    rows: list[dict[str, Any]] = []
    for side in SIDES:
        old, new = history.loc[history.side_name.eq(side)], current.loc[current.side_name.eq(side)]
        for feature in features:
            ref, cur = _finite(old[feature]), _finite(new[feature])
            ref, cur = ref[np.isfinite(ref)], cur[np.isfinite(cur)]
            scale = max(float(np.nanquantile(ref, .75) - np.nanquantile(ref, .25)), 1e-8) if len(ref) else np.nan
            rows.append({
                "side_name": side, "feature": feature, "history_rows": int(len(ref)), "current_rows": int(len(cur)),
                "history_mean": float(np.mean(ref)) if len(ref) else np.nan,
                "current_mean": float(np.mean(cur)) if len(cur) else np.nan,
                "psi_10bin": _psi(ref, cur),
                "wasserstein": float(wasserstein_distance(ref, cur)) if len(ref) and len(cur) else np.nan,
                "wasserstein_iqr_scaled": float(wasserstein_distance(ref, cur) / scale) if np.isfinite(scale) and len(cur) else np.nan,
            })
    return pd.DataFrame(rows)


def _robust_ood_fraction(reference: pd.DataFrame, evaluation: pd.DataFrame, features: Sequence[str]) -> pd.Series:
    """Causal-feature outlier fraction, using only historical medians/IQRs."""
    ref = reference.loc[:, features].apply(pd.to_numeric, errors="coerce")
    cur = evaluation.loc[:, features].apply(pd.to_numeric, errors="coerce")
    med = ref.median(axis=0); iqr = (ref.quantile(.75) - ref.quantile(.25)).clip(lower=1e-8)
    z = cur.sub(med).div(iqr).abs()
    return z.gt(4.0).mean(axis=1)


def _leaf_support(
    history: pd.DataFrame, current: pd.DataFrame, *, model_path: Path, features: Sequence[str]
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Frozen CatBoost leaf support; diagnostic only, never used to choose rules."""
    package = joblib.load(model_path)
    model = package["model"]
    x_old = history.loc[:, features].apply(pd.to_numeric, errors="raise")
    x_new = current.loc[:, features].apply(pd.to_numeric, errors="raise")
    old_leaf = np.asarray(model.calc_leaf_indexes(x_old), dtype=np.int64)
    new_leaf = np.asarray(model.calc_leaf_indexes(x_new), dtype=np.int64)
    supports = np.zeros_like(new_leaf, dtype=np.int64); js: list[float] = []
    for tree in range(old_leaf.shape[1]):
        counts = np.bincount(old_leaf[:, tree])
        leaves = new_leaf[:, tree]
        valid = leaves < len(counts)
        supports[valid, tree] = counts[leaves[valid]]
        width = max(len(counts), int(leaves.max()) + 1)
        p = np.bincount(old_leaf[:, tree], minlength=width) / len(old_leaf)
        q = np.bincount(leaves, minlength=width) / len(new_leaf)
        midpoint = (p + q) / 2.0
        p_mask, q_mask = p > 0, q > 0
        kl_p = np.sum(p[p_mask] * np.log(p[p_mask] / np.maximum(midpoint[p_mask], 1e-12)))
        kl_q = np.sum(q[q_mask] * np.log(q[q_mask] / np.maximum(midpoint[q_mask], 1e-12)))
        js.append(float(.5 * (kl_p + kl_q)))
    values = pd.DataFrame({
        "leaf_support_mean": supports.mean(axis=1),
        "leaf_support_min": supports.min(axis=1),
        "leaf_unseen_tree_fraction": (supports == 0).mean(axis=1),
        "leaf_low5_tree_fraction": (supports < 5).mean(axis=1),
    }, index=current.index)
    return values, {"mean_tree_js": float(np.mean(js)), "max_tree_js": float(np.max(js)), "unseen_tree_fraction": float((supports == 0).mean())}


def _outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["actual_meaningful"] = output.get("meaningful_mfe_reached", output.get("hard_meaningful")).astype(bool)
    output["actual_adverse"] = output.get("adverse_1atr_reached").astype(bool)
    output["actual_positive"] = pd.to_numeric(output["execution_net_ev_12h"], errors="raise").gt(0)
    return output


def outcome_shift(history: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for side in SIDES:
        for period, frame in (("historical_oof", history.loc[history.side_name.eq(side)]), ("july_exact", current.loc[current.side_name.eq(side)])):
            net = pd.to_numeric(frame.execution_net_ev_12h, errors="raise")
            pos, neg = net.gt(0), net.lt(0)
            rows.append({
                "side_name": side, "period": period, "rows": int(len(frame)),
                "meaningful_incidence": float(frame.actual_meaningful.mean()),
                "adverse_incidence": float(frame.actual_adverse.mean()),
                "positive_incidence": float(pos.mean()), "mean_net_bps": float(net.mean() * 1e4),
                "positive_payoff_bps": float(net[pos].mean() * 1e4) if pos.any() else np.nan,
                "adverse_payoff_bps": float(net[neg].mean() * 1e4) if neg.any() else np.nan,
                "clean_probability_residual": float(frame.actual_meaningful.mean() - pd.to_numeric(frame[PROB_CLEAN], errors="raise").mean()),
                "adverse_probability_residual": float(frame.actual_adverse.mean() - pd.to_numeric(frame[PROB_ADVERSE], errors="raise").mean()),
            })
    result = pd.DataFrame(rows)
    historical = result.loc[result.period.eq("historical_oof")].set_index("side_name")
    for col in ("meaningful_incidence", "adverse_incidence", "positive_incidence", "mean_net_bps", "positive_payoff_bps", "adverse_payoff_bps", "clean_probability_residual", "adverse_probability_residual"):
        result[f"shift_vs_historical_{col}"] = result.apply(lambda row: row[col] - historical.loc[row.side_name, col] if row.period == "july_exact" else 0.0, axis=1)
    return result


def select_reliability_rule(history: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    """Choose a short-only abstention rule using historical temporal OOF only."""
    short = history.loc[history.side_name.eq("short")].copy()
    short["month"] = pd.to_datetime(short.__ts__, utc=True).dt.strftime("%Y-%m")
    thresholds = {
        "adverse_q80": float(short[PROB_ADVERSE].quantile(.80)),
        "clean_q20": float(short[PROB_CLEAN].quantile(.20)),
        "ood_q80": float(short["raw_ood_fraction"].quantile(.80)),
    }
    rules = {
        "none": np.zeros(len(short), dtype=bool),
        "short_adverse_high": short[PROB_ADVERSE].gt(thresholds["adverse_q80"]).to_numpy(),
        "short_clean_low": short[PROB_CLEAN].lt(thresholds["clean_q20"]).to_numpy(),
        "short_raw_ood_high": short["raw_ood_fraction"].gt(thresholds["ood_q80"]).to_numpy(),
        "short_adverse_or_ood": (short[PROB_ADVERSE].gt(thresholds["adverse_q80"]) | short["raw_ood_fraction"].gt(thresholds["ood_q80"])).to_numpy(),
    }
    # This is a predeclared candidate population: top decile clean-first
    # probability per month, then a non-replacing abstention.  Coverage must
    # stay at least 70% in every month to be eligible.
    rank = short.groupby("month")[PROB_CLEAN].rank(method="first", ascending=False)
    base = rank.le(short.groupby("month")[PROB_CLEAN].transform(lambda x: max(1, math.ceil(.10 * len(x)))))
    rows: list[dict[str, Any]] = []
    for name, abstain in rules.items():
        retained = base.to_numpy() & ~abstain
        selected = short.loc[retained]
        monthly = selected.groupby("month").execution_net_ev_12h.mean() * 1e4
        coverage = selected.groupby("month").size() / short.loc[base].groupby("month").size()
        rows.append({"rule_id": name, "thresholds": json.dumps(thresholds, sort_keys=True), "selected_rows": int(len(selected)), "coverage": float(retained.sum() / max(base.sum(), 1)), "minimum_month_coverage": float(coverage.min()), "worst_month_net_bps": float(monthly.min()), "mean_month_net_bps": float(monthly.mean()), "pooled_net_bps": float(selected.execution_net_ev_12h.mean() * 1e4)})
    table = pd.DataFrame(rows).sort_values(["minimum_month_coverage", "worst_month_net_bps", "mean_month_net_bps", "pooled_net_bps", "rule_id"], ascending=[False, False, False, False, True], kind="mergesort").reset_index(drop=True)
    eligible = table.loc[table.minimum_month_coverage.ge(.70)]
    winner = eligible.iloc[0] if not eligible.empty else table.loc[table.rule_id.eq("none")].iloc[0]
    return {"rule_id": str(winner.rule_id), "thresholds": thresholds, "minimum_month_coverage_required": .70, "selection": "maximize worst-month net bps within minimum monthly coverage; tie mean-month then pooled net", "history_only": True}, table


def apply_reliability_rule(current: pd.DataFrame, state: Mapping[str, Any]) -> pd.DataFrame:
    work = current.copy()
    thresholds = state["thresholds"]; name = state["rule_id"]
    abstain = np.zeros(len(work), dtype=bool)
    short = work.side_name.eq("short")
    if name == "short_adverse_high": abstain = short & work[PROB_ADVERSE].gt(thresholds["adverse_q80"])
    elif name == "short_clean_low": abstain = short & work[PROB_CLEAN].lt(thresholds["clean_q20"])
    elif name == "short_raw_ood_high": abstain = short & work.raw_ood_fraction.gt(thresholds["ood_q80"])
    elif name == "short_adverse_or_ood": abstain = short & (work[PROB_ADVERSE].gt(thresholds["adverse_q80"]) | work.raw_ood_fraction.gt(thresholds["ood_q80"]))
    elif name != "none": raise AuditError(f"unknown frozen reliability rule: {name}")
    work["frozen_reliability_abstain"] = abstain
    work["frozen_clean_first_retained"] = work[ADMISSION].astype(bool) & ~abstain
    return work


def _current_rule_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for side in SIDES:
        for status, mask in (("base_clean_first_admission", frame[ADMISSION].astype(bool)), ("after_frozen_reliability", frame.frozen_clean_first_retained.astype(bool))):
            local = frame.loc[frame.side_name.eq(side) & mask]
            net = pd.to_numeric(local.execution_net_ev_12h, errors="raise")
            rows.append({"side_name": side, "status": status, "rows": int(len(local)), "population_coverage": float(len(local) / max(int(frame.side_name.eq(side).sum()), 1)), "mean_net_bps": float(net.mean() * 1e4) if len(net) else np.nan, "positive_rate": float(net.gt(0).mean()) if len(net) else np.nan, "meaningful_rate": float(local.actual_meaningful.mean()) if len(local) else np.nan, "adverse_rate": float(local.actual_adverse.mean()) if len(local) else np.nan})
    return pd.DataFrame(rows)


def run(*, challenger_root: Path, features_path: Path, feature_manifest_path: Path, current_raw_path: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists(): raise FileExistsError(output_dir)
    manifest = json.loads((challenger_root / "manifest.json").read_text())
    if manifest.get("schema") != "historical_to_july_meaningful_mfe_gate_challenger_v1" or manifest.get("promotion_eligible") is not False:
        raise AuditError("requires frozen non-promotable clean-first challenger")
    feature_contract = json.loads(feature_manifest_path.read_text())
    prefixed = list(feature_contract["eligible_full_period_feature_columns"])
    if len(prefixed) != 249 or not all(name.startswith("capture_candidate__") for name in prefixed):
        raise AuditError("requires exactly the 249 causal capture candidate features")
    raw_features = [name.removeprefix("capture_candidate__") for name in prefixed]
    history_pred_path = challenger_root / "historical_oof_predictions.parquet"; current_pred_path = challenger_root / "current_predictions.parquet"
    history_pred = pd.read_parquet(history_pred_path); current_pred = pd.read_parquet(current_pred_path)
    for frame, name in ((history_pred, "historical OOF"), (current_pred, "current")):
        if frame.duplicated(list(IDENTITY)).any() or not set([PROB_CLEAN, PROB_ADVERSE, "execution_net_ev_12h"]).issubset(frame): raise AuditError(f"{name} predictions lack identity or frozen probabilities")
        frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    if not history_pred["__ts__"].lt(HISTORY_CUTOFF).all(): raise AuditError("historical OOF prediction ledger crosses July20 cutoff")
    history_raw = pd.read_parquet(features_path, columns=[*IDENTITY, *prefixed]).rename(columns=dict(zip(prefixed, raw_features)))
    current_raw = pd.read_parquet(current_raw_path, columns=[*IDENTITY, *raw_features])
    for f in (history_raw, current_raw): f["__ts__"] = pd.to_datetime(f["__ts__"], utc=True, errors="raise")
    history = history_pred.merge(history_raw, on=list(IDENTITY), how="inner", validate="one_to_one")
    current = current_pred.merge(current_raw, on=list(IDENTITY), how="inner", validate="one_to_one")
    if len(history) != len(history_pred) or len(current) != len(current_pred): raise AuditError("raw causal features do not cover frozen prediction identities")
    # The frozen 249-field source has a small number of early historical warmup
    # nulls.  They are not meaningful drift observations: retain only the
    # complete historical rows (and prove coverage), while requiring July rows
    # to be fully materialized.
    history_values = history.loc[:, raw_features].apply(pd.to_numeric, errors="coerce")
    complete_history = np.isfinite(history_values.to_numpy(dtype=float)).all(axis=1)
    if float(complete_history.mean()) < .995:
        raise AuditError("historical 249-feature complete-row coverage is too low")
    history = history.loc[complete_history].copy()
    current_values = current.loc[:, raw_features].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(current_values.to_numpy(dtype=float)).all():
        raise AuditError("July exact candidates have non-finite causal feature values")
    history = _outcomes(history); current = _outcomes(current)
    for side in SIDES:
        hmask, cmask = history.side_name.eq(side), current.side_name.eq(side)
        history.loc[hmask, "raw_ood_fraction"] = _robust_ood_fraction(history.loc[hmask], history.loc[hmask], raw_features).to_numpy()
        current.loc[cmask, "raw_ood_fraction"] = _robust_ood_fraction(history.loc[hmask], current.loc[cmask], raw_features).to_numpy()
    drift = feature_drift(history, current, raw_features)
    leaf_records: list[pd.DataFrame] = []; leaf_summary: dict[str, Any] = {}
    for side in SIDES:
        model_path = challenger_root / "models" / f"catboost_hard_clean_first__{side}.joblib"
        package = joblib.load(model_path); model_features = list(package["features"])
        support, summary = _leaf_support(history.loc[history.side_name.eq(side)], current.loc[current.side_name.eq(side)], model_path=model_path, features=model_features)
        support.index = current.loc[current.side_name.eq(side)].index
        for column in support: current.loc[support.index, column] = support[column]
        leaf_summary[side] = {"feature_count": len(model_features), **summary}
        leaf_records.append(current.loc[support.index, [*IDENTITY, "leaf_support_mean", "leaf_support_min", "leaf_unseen_tree_fraction", "leaf_low5_tree_fraction"]])
    shifts = outcome_shift(history, current)
    state, historical_rules = select_reliability_rule(history)
    current = apply_reliability_rule(current, state)
    current_metrics = _current_rule_metrics(current)
    stage = output_dir.parent / f".{output_dir.name}.staging-{uuid.uuid4().hex}"; stage.mkdir(parents=True)
    try:
        paths = {"feature_drift": stage / "feature_drift.csv", "outcome_shifts": stage / "outcome_shifts.csv", "historical_rule_selection": stage / "historical_oof_rule_selection.csv", "current_rule_metrics": stage / "july_frozen_rule_metrics.csv", "current_rows": stage / "july_reliability_rows.parquet", "leaf_support": stage / "july_clean_first_leaf_support.parquet", "frozen_rule": stage / "frozen_reliability_rule.json"}
        drift.to_csv(paths["feature_drift"], index=False); shifts.to_csv(paths["outcome_shifts"], index=False); historical_rules.to_csv(paths["historical_rule_selection"], index=False); current_metrics.to_csv(paths["current_rule_metrics"], index=False)
        current.to_parquet(paths["current_rows"], index=False, compression="zstd"); pd.concat(leaf_records, ignore_index=True).to_parquet(paths["leaf_support"], index=False, compression="zstd"); _write_json(paths["frozen_rule"], state)
        short_base = current_metrics.loc[(current_metrics.side_name.eq("short")) & (current_metrics.status.eq("base_clean_first_admission"))].iloc[0]
        report = {"schema": SCHEMA, "status": "research_only_historical_oof_rule_then_single_july_evaluation", "promotion_eligible": False, "chronology": "all thresholds/rule selection are historical temporal OOF before July20; current exact outcomes are used once only after frozen rule persistence", "raw_causal_features": 249, "frozen_rule": state, "leaf_transfer": leaf_summary, "short_minimum_coverage": {"minimum_population_coverage": MIN_CURRENT_SIDE_POPULATION_COVERAGE, "observed_base_clean_first_coverage": float(short_base.population_coverage), "passes": bool(float(short_base.population_coverage) >= MIN_CURRENT_SIDE_POPULATION_COVERAGE), "meaning": "A side with fewer than 1% clean-first admissions is not actionable; abstention cannot repair missing base coverage."}, "recommendations": ["Use short adverse probability, raw 249-feature OOD fraction, and clean-first probability as shadow reliability inputs; do not promote a gate until forward coverage is adequate.", "Monitor clean-first CatBoost leaf support/unseen-tree fraction as a representation-transfer alarm, but do not use it as a selected rule until it has temporal-OOF support." ]}
        _write_json(stage / "report.json", report)
        manifest_out = {**report, "inputs": {"challenger_manifest": _record(challenger_root / "manifest.json"), "frozen_state": _record(challenger_root / "frozen_before_current_evaluation.json"), "historical_oof_predictions": _record(history_pred_path), "current_predictions": _record(current_pred_path), "historical_249_features": _record(features_path), "feature_contract": _record(feature_manifest_path), "current_249_features": _record(current_raw_path)}, "coverage": {"historical_oof_rows": int(len(history)), "july_rows": int(len(current)), "july_short_rows": int(current.side_name.eq("short").sum())}, "outputs": {key: {"path": str(output_dir / path.name), "sha256": _sha256(path)} for key, path in paths.items()} | {"report": {"path": str(output_dir / "report.json"), "sha256": _sha256(stage / "report.json")}}}
        _write_json(stage / "manifest.json", manifest_out); os.replace(stage, output_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True); raise
    return manifest_out


def _parser(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--challenger-root", type=Path, default=DEFAULT_CHALLENGER); p.add_argument("--features", type=Path, default=DEFAULT_FEATURES); p.add_argument("--feature-manifest", type=Path, default=DEFAULT_FEATURE_MANIFEST); p.add_argument("--current-raw", type=Path, default=DEFAULT_CURRENT_RAW); p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return p.parse_args(argv)


def main() -> None:
    args = _parser(); result = run(challenger_root=args.challenger_root, features_path=args.features, feature_manifest_path=args.feature_manifest, current_raw_path=args.current_raw, output_dir=args.output_dir); print(json.dumps(result["coverage"], indent=2))


if __name__ == "__main__": main()
