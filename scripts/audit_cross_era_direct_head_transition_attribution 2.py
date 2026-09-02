#!/usr/bin/env python3
"""Causal descriptive attribution of frozen direct-net head errors.

This is intentionally an audit rather than another challenger.  It never fits
models, changes mappings, or re-selects an actionable policy.  The sole global
selection is the frozen mapped-q25 global top 10% within each evaluation split;
state-local tails are clearly marked as descriptive diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_july_exact_preentry_heads import IDENTITY


SCHEMA = "cross_era_direct_head_transition_attribution_audit_v1"
SIDES = ("long", "short")
QUANTILES = (("q25_net_bps", 0.25), ("q50_net_bps", 0.50))
SEVERE = (("p_loss_le_100", -100.0), ("p_loss_le_200", -200.0), ("p_loss_le_400", -400.0))
RAW_STATE_COLUMNS = (
    "regime_transition_entropy_12h",
    "regime_transition_entropy_48h",
    "regime_stability_24h",
    "volatility_of_volatility_48",
    "vov_interaction",
)
BINARY_STATE_COLUMNS = ("is_high_vol_regime", "is_low_vol_regime", "is_ranging")
ACTIVE_BANDS = ("<0.25", "[0.25,0.50)", "[0.50,0.75)", ">=0.75", "missing")


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
    if isinstance(value, (Path, pd.Timestamp, np.generic)):
        return str(value) if isinstance(value, (Path, pd.Timestamp)) else value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def binding(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "sha256": sha256(path)}
    if rows is not None:
        result["rows"] = int(rows)
    return result


def _resolve(path: str | Path) -> Path:
    result = Path(path)
    return result if result.is_absolute() else ROOT / result


def verify_manifest(source_dir: Path, *, required_outputs: Iterable[str] = ()) -> dict[str, Any]:
    """Verify a source report and every declared output hash before reading it."""
    report_path, manifest_path = source_dir / "report.json", source_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if "report" in manifest and sha256(report_path) != manifest["report"]["sha256"]:
            raise ValueError(f"report hash mismatch: {source_dir}")
    else:
        # The active-transition artifact is manifest-only.  Its manifest is the
        # authoritative immutable declaration and is hash-bound in our output.
        report = manifest
    outputs = report.get("outputs", manifest.get("outputs", {}))
    for name, record in outputs.items():
        path = _resolve(record["path"])
        if not path.exists() or sha256(path) != record["sha256"]:
            raise ValueError(f"source output hash mismatch: {source_dir}/{name}")
        # Row-level coverage is checked after loading the identity-bearing
        # sources below.  Empty-column parquet reads are not portable: some
        # engines report zero rows even when the file is populated.
    missing = set(required_outputs).difference(outputs)
    if missing:
        raise ValueError(f"source output declarations missing: {sorted(missing)}")
    return {"report": report, "manifest": manifest, "outputs": outputs}


def assert_identity_equal(left: pd.DataFrame, right: pd.DataFrame, *, label: str) -> dict[str, Any]:
    """Fail closed on identity duplication or imperfect one-to-one coverage."""
    keys = list(IDENTITY)
    for name, frame in (("left", left), ("right", right)):
        if set(keys).difference(frame.columns):
            raise ValueError(f"{label}: {name} missing identity columns")
        if frame.duplicated(keys).any():
            raise ValueError(f"{label}: {name} has duplicate identities")
    lhs = left.loc[:, keys].sort_values(keys).reset_index(drop=True)
    rhs = right.loc[:, keys].sort_values(keys).reset_index(drop=True)
    if not lhs.equals(rhs):
        raise ValueError(f"{label}: identity coverage mismatch left={len(lhs)} right={len(rhs)}")
    return {"rows": int(len(lhs)), "identity_complete_one_to_one": True}


def fixed_z_band(values: pd.Series) -> pd.Series:
    """Fixed, outcome-free bands for transformed/winsorized z coordinates."""
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.Series(np.select(
        [numeric.lt(-.75), numeric.lt(0.0), numeric.lt(.75)],
        ["<-.75", "[-.75,0)", "[0,.75)"], default=">=.75",
    ), index=values.index, dtype="object").where(numeric.notna(), "missing")


def active_band(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.Series(np.select(
        [numeric.lt(.25), numeric.lt(.50), numeric.lt(.75)],
        ["<0.25", "[0.25,0.50)", "[0.50,0.75)"], default=">=0.75",
    ), index=values.index, dtype="object").where(numeric.notna(), "missing")


def _binary_band(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.Series(np.where(numeric >= .5, "on", "off"), index=values.index, dtype="object").where(numeric.notna(), "missing")


def add_causal_states(frame: pd.DataFrame, active: pd.DataFrame) -> pd.DataFrame:
    """Attach timestamp-observable state, retaining missing OOS active scores."""
    work = frame.copy()
    work["__ts__"] = pd.to_datetime(work["__ts__"], utc=True, errors="raise")
    active_work = active.loc[:, ["source_utc", "prediction"]].copy()
    active_work["source_utc"] = pd.to_datetime(active_work["source_utc"], utc=True, errors="raise")
    if active_work["source_utc"].duplicated().any():
        raise ValueError("active-transition source has duplicate timestamps")
    work = work.merge(active_work.rename(columns={"source_utc": "__ts__", "prediction": "active_transition_probability_oos"}), on="__ts__", how="left", validate="many_to_one")
    work["active_transition_band"] = active_band(work["active_transition_probability_oos"])
    for column in RAW_STATE_COLUMNS:
        if column not in work:
            raise ValueError(f"required causal raw state missing: {column}")
        work[f"state__{column}"] = fixed_z_band(work[column])
    for column in BINARY_STATE_COLUMNS:
        if column not in work:
            raise ValueError(f"required causal binary state missing: {column}")
        work[f"state__{column}"] = _binary_band(work[column])
    # These pre-declared transformed-space coordinates use only point-in-time
    # inputs.  The upstream entropy/stability/vov columns are winsorized and
    # transformed (~[-2.05, 2.05]); they are not literal 0--1 probabilities or
    # stability quantities.  Existing vov_interaction remains a lineage field,
    # while these coordinates are the fixed comparable state representation.
    entropy = .5 * pd.to_numeric(work["regime_transition_entropy_12h"], errors="coerce") + .5 * pd.to_numeric(work["regime_transition_entropy_48h"], errors="coerce")
    stability = pd.to_numeric(work["regime_stability_24h"], errors="coerce")
    vov = pd.to_numeric(work["volatility_of_volatility_48"], errors="coerce")
    entropy48 = pd.to_numeric(work["regime_transition_entropy_48h"], errors="coerce")
    entropy12 = pd.to_numeric(work["regime_transition_entropy_12h"], errors="coerce")
    work["transition_pressure_z"] = entropy48 - stability
    work["entropy_acceleration_z"] = entropy12 - entropy48
    work["entropy_x_vov_z"] = entropy * vov
    work["state__transition_pressure_z"] = fixed_z_band(work["transition_pressure_z"])
    work["state__entropy_acceleration_z"] = fixed_z_band(work["entropy_acceleration_z"])
    work["state__entropy_x_vov_z"] = fixed_z_band(work["entropy_x_vov_z"])
    return work


def _ece(truth: np.ndarray, probability: np.ndarray) -> float:
    bins = np.minimum((np.clip(probability, 0., 1.) * 10).astype(int), 9)
    return float(sum((bins == index).mean() * abs(probability[bins == index].mean() - truth[bins == index].mean()) for index in range(10) if (bins == index).any()))


def _rank_ic(prediction: pd.Series, actual: pd.Series) -> float:
    return float(prediction.corr(actual, method="spearman")) if len(prediction) >= 3 else float("nan")


def state_dimensions(frame: pd.DataFrame) -> list[str]:
    result = ["active_transition_band"]
    result.extend(sorted(column for column in frame.columns if column.startswith("state__")))
    if len(result) < 2:
        raise ValueError("no causal raw state dimensions materialized")
    return result


def head_metrics(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    """Head errors by side/month/era/every causal state; never selection."""
    work = frame.copy()
    work["month"] = pd.to_datetime(work["__ts__"], utc=True).dt.strftime("%Y-%m")
    work["net_bps"] = pd.to_numeric(work["execution_net_ev_12h"], errors="raise") * 1e4
    rows: list[dict[str, Any]] = []
    for state_column in state_dimensions(work):
        keys = ["era", "month", "side_name", state_column]
        for key, local in work.groupby(keys, dropna=False, sort=True):
            era, month, side, state = key
            base = {"split": split, "state_dimension": state_column.removeprefix("state__"), "era": era, "month": month, "side_name": side, "state": state, "rows": int(len(local)), "active_probability_coverage": float(local["active_transition_probability_oos"].notna().mean())}
            actual = local["net_bps"].to_numpy(float)
            for column, alpha in QUANTILES:
                prediction = pd.to_numeric(local[column], errors="raise").to_numpy(float)
                error = actual - prediction
                pinball = np.maximum(alpha * error, (alpha - 1.0) * error)
                rows.append({**base, "head_family": "quantile", "head": column, "rank_ic": _rank_ic(local[column], local["net_bps"]), "pinball_loss_bps": float(pinball.mean()), "bias_prediction_minus_actual_bps": float((prediction - actual).mean()), "brier": np.nan, "ece10": np.nan, "probability_gap": np.nan})
            for column, threshold in SEVERE:
                probability = np.clip(pd.to_numeric(local[column], errors="raise").to_numpy(float), 0., 1.)
                truth = (actual <= threshold).astype(float)
                rows.append({**base, "head_family": "severe_probability", "head": column, "rank_ic": np.nan, "pinball_loss_bps": np.nan, "bias_prediction_minus_actual_bps": np.nan, "brier": float(np.mean((probability - truth) ** 2)), "ece10": _ece(truth, probability), "probability_gap": float(probability.mean() - truth.mean()), "predicted_mean": float(probability.mean()), "actual_rate": float(truth.mean())})
    return pd.DataFrame(rows)


def state_support(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    """Long-form support for every required causal raw and transition state."""
    work = frame.copy()
    work["month"] = pd.to_datetime(work["__ts__"], utc=True).dt.strftime("%Y-%m")
    state_columns = state_dimensions(work)
    rows: list[dict[str, Any]] = []
    for state_column in state_columns:
        for key, local in work.groupby(["era", "month", "side_name", state_column], dropna=False, sort=True):
            era, month, side, state = key
            rows.append({"split": split, "state_dimension": state_column.removeprefix("state__"), "era": era, "month": month, "side_name": side, "state": state, "rows": int(len(local)), "active_probability_coverage": float(local["active_transition_probability_oos"].notna().mean()), "mean_net_bps": float((pd.to_numeric(local["execution_net_ev_12h"], errors="raise") * 1e4).mean())})
    return pd.DataFrame(rows)


def _tail_stats(frame: pd.DataFrame) -> dict[str, float]:
    net = pd.to_numeric(frame["execution_net_ev_12h"], errors="raise").to_numpy(float) * 1e4
    return {"rows": int(len(frame)), "net_ev_bps": float(net.mean()), "positive_precision": float((net > 0).mean()), "cvar05_bps": float(np.sort(net)[:max(1, int(math.ceil(.05 * len(net))))].mean()), "long_rows": int(frame["side_name"].astype(str).eq("long").sum()), "short_rows": int(frame["side_name"].astype(str).eq("short").sum())}


def tail_composition(frame: pd.DataFrame, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate raw/mapped global books plus non-actionable state-local tails."""
    score_columns = (("raw_q25", "q25_net_bps"), ("frozen_mapped_q25", "mapped_q25_bps"))
    if set(column for _, column in score_columns).difference(frame.columns):
        raise ValueError("frozen raw or mapped q25 score missing")
    work = frame.copy()
    composition: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for score_scope, score in score_columns:
        selected = work.sort_values([score, "candidate_id"], ascending=[False, True], kind="stable").iloc[:max(1, int(math.ceil(.10 * len(work))))].copy()
        composition.append({"split": split, "score_scope": score_scope, "score_column": score, "selection_scope": "one_global_top10", "descriptive_only": False, "state_dimension": "all", "state": "all", **_tail_stats(selected)})
        for state_column in state_dimensions(work):
            dimension = state_column.removeprefix("state__")
            for state, local in selected.groupby(state_column, dropna=False, sort=True):
                composition.append({"split": split, "score_scope": score_scope, "score_column": score, "selection_scope": "composition_of_one_global_top10", "descriptive_only": False, "state_dimension": dimension, "state": state, **_tail_stats(local)})
            for state, population in work.groupby(state_column, dropna=False, sort=True):
                local = population.sort_values([score, "candidate_id"], ascending=[False, True], kind="stable").iloc[:max(1, int(math.ceil(.10 * len(population))))]
                diagnostics.append({"split": split, "score_scope": score_scope, "score_column": score, "selection_scope": "state_local_top10_descriptive_only", "descriptive_only": True, "state_dimension": dimension, "state": state, "population_rows": int(len(population)), **_tail_stats(local)})
    return pd.DataFrame(composition), pd.DataFrame(diagnostics)


def _join_history(predictions: pd.DataFrame, raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    keys = list(IDENTITY)
    if predictions.duplicated(keys).any() or raw.duplicated(keys).any():
        raise ValueError("history source has duplicate identities")
    raw_columns = [*keys, "era", *RAW_STATE_COLUMNS, *BINARY_STATE_COLUMNS]
    merged = predictions.merge(raw.loc[:, raw_columns], on=keys, how="left", validate="one_to_one", suffixes=("", "_raw"))
    if len(merged) != len(predictions) or merged[list(RAW_STATE_COLUMNS)].isna().all(axis=1).any():
        raise ValueError("historical direct OOF has incomplete raw-feature coverage")
    return merged, {"prediction_rows": int(len(predictions)), "raw_dataset_rows": int(len(raw)), "raw_feature_coverage_rows": int(merged[list(RAW_STATE_COLUMNS)].notna().any(axis=1).sum()), "raw_feature_coverage_complete": True, "per_field_availability_evidence": "not materialized in this historical source; recorded as an evidence gap rather than asserted"}


def _join_current(predictions: pd.DataFrame, raw: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    keys = list(IDENTITY)
    coverage = assert_identity_equal(predictions, raw, label="current predictions vs raw pack")
    assert_identity_equal(predictions, labels, label="current predictions vs exact labels")
    availability_evidence: dict[str, Any] = {"feature_availability_checked": False, "reason": "feature_available_at or execution_decision_utc absent"}
    if {"feature_available_at", "execution_decision_utc"}.issubset(raw.columns):
        available = pd.to_datetime(raw["feature_available_at"], utc=True, errors="coerce")
        decision = pd.to_datetime(raw["execution_decision_utc"], utc=True, errors="raise")
        if available.isna().any() or (available > decision).any():
            raise ValueError("current feature availability exceeds decision time or is missing")
        availability_evidence = {"feature_availability_checked": True, "rows": int(len(raw)), "max_feature_available_at": available.max(), "min_execution_decision_utc": decision.min(), "all_feature_available_at_lte_execution_decision_utc": True}
    raw_columns = [*keys, *RAW_STATE_COLUMNS, *BINARY_STATE_COLUMNS]
    label_columns = [*keys, "execution_net_ev_12h"]
    merged = predictions.merge(raw.loc[:, raw_columns], on=keys, how="inner", validate="one_to_one")
    merged = merged.merge(labels.loc[:, label_columns], on=keys, how="inner", validate="one_to_one")
    if merged[list(RAW_STATE_COLUMNS)].isna().all(axis=1).any():
        raise ValueError("current raw-feature state coverage is incomplete")
    merged["era"] = "2026_jul20_23"
    return merged, {**coverage, "raw_feature_coverage_rows": int(merged[list(RAW_STATE_COLUMNS)].notna().any(axis=1).sum()), "raw_feature_coverage_complete": True, "availability": availability_evidence}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    dataset_source = verify_manifest(args.dataset_dir, required_outputs=("dataset",))
    direct_source = verify_manifest(args.direct_dir, required_outputs=("historical_oof_winner", "current_predictions_before_outcomes", "current_scored_exact"))
    active_source = verify_manifest(args.active_dir, required_outputs=("predictions",))
    if direct_source["report"].get("winner", {}).get("score_column") != "q25_net_bps":
        raise ValueError("audit requires the frozen direct q25 winner")
    raw_history = pd.read_parquet(_resolve(dataset_source["outputs"]["dataset"]["path"]))
    historical = pd.read_parquet(_resolve(direct_source["outputs"]["historical_oof_winner"]["path"]))
    current_pre = pd.read_parquet(_resolve(direct_source["outputs"]["current_predictions_before_outcomes"]["path"]))
    current_exact = pd.read_parquet(_resolve(direct_source["outputs"]["current_scored_exact"]["path"]))
    current_raw = pd.read_parquet(args.current_pack)
    current_labels = pd.read_parquet(args.current_labels)
    active = pd.read_parquet(_resolve(active_source["outputs"]["predictions"]["path"]))
    # Exact source-current parity prevents the audit from silently changing the
    # post-freeze outcomes or prediction values declared by the frozen runner.
    assert_identity_equal(current_pre, current_exact, label="frozen pre-outcome vs scored current")
    if not np.allclose(current_pre["q25_net_bps"], current_exact["q25_net_bps"], equal_nan=True):
        raise ValueError("current frozen q25 prediction changed after outcome join")
    assert_identity_equal(current_exact, current_labels, label="frozen scored current vs declared exact labels")
    frozen_outcomes = current_exact.loc[:, [*IDENTITY, "execution_net_ev_12h"]].merge(current_labels.loc[:, [*IDENTITY, "execution_net_ev_12h"]], on=list(IDENTITY), validate="one_to_one", suffixes=("_frozen", "_declared"))
    if not np.allclose(frozen_outcomes["execution_net_ev_12h_frozen"], frozen_outcomes["execution_net_ev_12h_declared"], equal_nan=True):
        raise ValueError("frozen scored current outcomes differ from declared exact labels")
    history, history_coverage = _join_history(historical, raw_history)
    current, current_coverage = _join_current(current_pre, current_raw, current_labels)
    history = add_causal_states(history, active)
    current = add_causal_states(current, active)
    outputs: dict[str, Any] = {}
    args.output_dir.mkdir(parents=True)
    tables = {
        "head_metrics": pd.concat([head_metrics(history, "historical_oof"), head_metrics(current, "current")], ignore_index=True),
        "state_support": pd.concat([state_support(history, "historical_oof"), state_support(current, "current")], ignore_index=True),
    }
    composition, diagnostics = tail_composition(history, "historical_oof")
    current_composition, current_diagnostics = tail_composition(current, "current")
    tables["global_selected_tail_composition"] = pd.concat([composition, current_composition], ignore_index=True)
    tables["state_local_tail_diagnostics"] = pd.concat([diagnostics, current_diagnostics], ignore_index=True)
    for name, table in tables.items():
        path = args.output_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        outputs[name] = binding(path, rows=len(table))
    report = {
        "schema": SCHEMA,
        "status": "completed_research_only_descriptive_no_train_no_mapping_no_promotion_no_portfolio_replay",
        "promotion_eligible": False,
        "actions": {"trained": False, "mapped": False, "promoted": False, "portfolio_replayed": False},
        "source_integrity": {
            "cross_era_dataset_manifest": binding(args.dataset_dir / "manifest.json"),
            "direct_challenger_manifest": binding(args.direct_dir / "manifest.json"),
            "active_transition_manifest": binding(args.active_dir / "manifest.json"),
            "current_pack": binding(args.current_pack), "current_labels": binding(args.current_labels),
        },
        "coverage": {"historical": history_coverage, "current": current_coverage, "transition_probability": {"historical_coverage": float(history["active_transition_probability_oos"].notna().mean()), "current_coverage": float(current["active_transition_probability_oos"].notna().mean()), "active_prediction_source_start": pd.to_datetime(active["source_utc"], utc=True).min(), "active_prediction_source_end": pd.to_datetime(active["source_utc"], utc=True).max(), "current_start": pd.to_datetime(current["__ts__"], utc=True).min(), "current_end": pd.to_datetime(current["__ts__"], utc=True).max(), "current_missing_is_retained_and_raw_states_are_required": True}},
        "state_contract": {"active_probability": "chronological OOS active-transition prediction joined exact by source timestamp; missing is an explicit band", "active_bands": list(ACTIVE_BANDS), "transformed_coordinate_bands": ["<-.75", "[-.75,0)", "[0,.75)", ">=.75", "missing"], "raw_feature_semantics": "entropy/stability/vov raw source columns are upstream transformed/winsorized coordinates, not literal 0-1 quantities; vov_interaction is retained only for lineage comparison", "raw_features": list(RAW_STATE_COLUMNS), "binary_features": list(BINARY_STATE_COLUMNS), "composites": ["transition_pressure_z", "entropy_acceleration_z", "entropy_x_vov_z"], "outcome_derived_thresholds": False},
        "selection_contract": "For each split and score scope, one pooled global top 10% is calculated: frozen mapped-q25 is the original challenger ranking and raw q25 is a separate diagnostic comparator. State-local top-10% rows are descriptive only and are not a policy, quota, mapping, or replay.",
        "outputs": outputs,
    }
    write_json(args.output_dir / "report.json", report)
    write_json(args.output_dir / "manifest.json", {"schema": SCHEMA, "status": report["status"], "promotion_eligible": False, "report": binding(args.output_dir / "report.json"), "outputs": outputs})
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset-dir", type=Path, default=Path("data_perp/artifacts/cross_era_tail_payoff_dataset_20260730_v3"))
    result.add_argument("--direct-dir", type=Path, default=Path("data_perp/artifacts/cross_era_direct_net_quantile_challenger_20260730_v1"))
    result.add_argument("--active-dir", type=Path, default=Path("data_perp/artifacts/regime_transition_active_head_chronological_oos_20260729_v2"))
    result.add_argument("--current-pack", type=Path, default=Path("data_perp/artifacts/execution_ev_july20_23_retrospective_20260730_v2/packb/packb_forward_context.parquet"))
    result.add_argument("--current-labels", type=Path, default=Path("data_perp/artifacts/execution_ev_july20_23_retrospective_20260730_v2/labels_12h/execution_ev_policy_labels.parquet"))
    result.add_argument("--output-dir", type=Path, default=Path("data_perp/artifacts/cross_era_direct_head_transition_attribution_audit_20260730_v2"))
    return result


if __name__ == "__main__":
    run(parser().parse_args())
