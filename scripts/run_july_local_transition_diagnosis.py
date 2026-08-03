#!/usr/bin/env python3
"""Diagnose July-local learnability of exact H12 pooled-global transitions.

This is not another residual/direct-score transfer test.  It uses only the
pooled panel's frozen 90-field decision-time geometry and its already-published
exact H12 pooled-global active/onset labels.  It first audits whether July
20--23 can be appended honestly; if not, it reports the last target-specific
resolved endpoint and keeps that source boundary immutable.

The local experiment is deliberately modest: purged grouped July OOF and
adjacent-week forward transfer, compared with prior and constant controls.
All top-10 measurements are one pooled July book and report cutoff-tie bounds.
No result may route, map, admit, or replay a policy.
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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "july_local_exact_h12_transition_diagnosis_v2"
CURRENT_SOURCE = "current_exact_spread_mayjul2026"
EXTENSION_SOURCE = "current_july20_23_retrospective_causal_mapping"
TARGETS = ("target__active_adverse", "target__adverse_onset_within_3h")
HORIZON_HOURS = 12
PURGE = pd.Timedelta(hours=36)
TOP_FRACTION = 0.10
MIN_LOGISTIC_ROWS = 60
MIN_LOGISTIC_POSITIVES = 10
DEFAULT_PANEL = ROOT / "data_perp/artifacts/pooled_historical_current_transition_panel_20260730_v1"
DEFAULT_RETROSPECTIVE = ROOT / "data_perp/artifacts/execution_ev_july20_23_retrospective_20260730_v2"
DEFAULT_EXTENSION = ROOT / "data_perp/artifacts/july20_23_exact_h12_transition_inputs_20260730_v1"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/july_local_exact_h12_transition_diagnosis_20260730_v2"


class JulyTransitionError(RuntimeError):
    """Raised when a frozen source cannot support a leakage-safe diagnostic."""


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
    if isinstance(value, (Path, pd.Timestamp, pd.Timedelta)):
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


def _read_manifest(root: Path, *, label: str) -> tuple[dict[str, Any], Path]:
    path = root / "manifest.json"
    seal = root / "manifest.sha256"
    if not path.is_file() or not seal.is_file():
        raise FileNotFoundError(f"{label} manifest or seal is absent")
    if seal.read_text(encoding="utf-8").split()[0] != sha256(path):
        raise JulyTransitionError(f"{label} manifest seal fails")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise JulyTransitionError(f"{label} manifest is not an object")
    return value, path


def _week(frame: pd.DataFrame) -> pd.Series:
    # Mondays in UTC remain explicit, avoiding local-time calendar semantics.
    anchor = pd.to_datetime(frame["cohort_anchor_utc"], utc=True, errors="raise")
    return (anchor - pd.to_timedelta(anchor.dt.dayofweek, unit="D")).dt.floor("D").dt.strftime("%Y-%m-%d")


def _load_bound_panel(root: Path, *, label: str) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    manifest, manifest_path = _read_manifest(root, label=label)
    panel_path = root / "transition_panel.parquet"
    if not panel_path.is_file() or manifest.get("outputs_sha256", {}).get(panel_path.name) != sha256(panel_path):
        raise JulyTransitionError(f"{label} parquet is not manifest-bound")
    features = list(manifest.get("feature_columns", []))
    if len(features) != 90 or len(features) != len(set(features)):
        raise JulyTransitionError(f"{label} is not the strict 90-field geometry")
    panel = pd.read_parquet(panel_path)
    required = {"cohort_anchor_utc", "source_family", "mapping_provenance_role", "context_available", *features, *TARGETS}
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise JulyTransitionError(f"{label} lacks: {missing}")
    panel["cohort_anchor_utc"] = pd.to_datetime(panel["cohort_anchor_utc"], utc=True, errors="raise")
    return panel, features, {
        "panel": str(panel_path), "panel_sha256": sha256(panel_path), "manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path),
    }


def load_panel(root: Path, extension: Path | None = None) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    panel, features, base_source = _load_bound_panel(root, label="pooled transition panel")
    sources: dict[str, Any] = {"base": base_source}
    if extension is not None:
        extended, extension_features, extension_source = _load_bound_panel(extension, label="July 20--23 extension panel")
        if extension_features != features:
            raise JulyTransitionError("July extension does not share the exact strict 90-field geometry")
        if not extended["source_family"].eq(EXTENSION_SOURCE).all() or not extended["mapping_provenance_role"].eq("retrospective_causal_21d_non_oof").all():
            raise JulyTransitionError("July extension provenance is not explicitly retrospective/non-OOF")
        panel = pd.concat([panel, extended], ignore_index=True, sort=False)
        if panel.duplicated(["source_family", "cohort_anchor_utc", "horizon_hours", "book_fraction"]).any():
            raise JulyTransitionError("transition panels overlap within a source")
        sources["extension"] = extension_source
    july = panel.loc[
        panel["source_family"].isin((CURRENT_SOURCE, EXTENSION_SOURCE))
        & panel["context_available"].astype(bool)
        & panel["cohort_anchor_utc"].dt.strftime("%Y-%m").eq("2026-07")
    ].copy()
    if july.empty:
        raise JulyTransitionError("no current July rows with strict common decision-time geometry")
    july["utc_week"] = _week(july)
    return july.sort_values(["cohort_anchor_utc", "source_family"], kind="stable").reset_index(drop=True), features, sources


def july23_extension_audit(extension_root: Path) -> pd.DataFrame:
    """Verify the separately materialised July 20--23 diagnostic extension."""

    manifest, manifest_path = _read_manifest(extension_root, label="July 20--23 materialised extension")
    outputs = manifest.get("outputs_sha256", {})
    checks = (
        ("causal 21d mapped-EV coordinates", "candidate_global_mapped_ev_coordinates.parquet"),
        ("one pooled-global H12 top10 before/after labels", "global_book_transition_labels.parquet"),
        ("strict 90-field decision-time geometry", "strict_common_geometry.parquet"),
        ("joined H12 extension panel", "transition_panel.parquet"),
    )
    rows: list[dict[str, Any]] = []
    for requirement, filename in checks:
        path = extension_root / filename
        available = path.is_file() and outputs.get(filename) == sha256(path)
        rows.append({"requirement": requirement, "available": available, "evidence": "immutable extension manifest output checksum", "extension_permitted": False})
    permitted = all(row["available"] for row in rows) and not bool(manifest.get("promotion_eligible", True)) and manifest.get("mapping_provenance_role") == "retrospective_causal_21d_non_oof"
    panel_path = extension_root / "transition_panel.parquet"
    panel = pd.read_parquet(panel_path) if panel_path.is_file() else pd.DataFrame()
    for row in rows:
        row.update({"extension_permitted": permitted, "retrospective_rows": int(len(panel)), "first_anchor_utc": panel.get("cohort_anchor_utc", pd.Series(dtype="datetime64[ns, UTC]")).min() if len(panel) else pd.NaT, "last_anchor_utc": panel.get("cohort_anchor_utc", pd.Series(dtype="datetime64[ns, UTC]")).max() if len(panel) else pd.NaT, "extension_manifest": str(manifest_path), "extension_manifest_sha256": sha256(manifest_path)})
    return pd.DataFrame(rows)


def valid_rows(july: pd.DataFrame, target: str, *, population: str) -> pd.DataFrame:
    availability = f"{target}_available_utc"
    if availability not in july:
        raise JulyTransitionError(f"{target} lacks target-specific availability")
    provenance = july["mapping_provenance_role"].astype(str)
    if population == "strict_oof_only":
        allowed = provenance.eq("strict_oof")
    elif population == "strict_oof_plus_frozen_forward_diagnostic":
        allowed = provenance.isin(("strict_oof", "frozen_forward_oos"))
    elif population == "strict_oof_plus_forward_plus_retro_extension_diagnostic":
        allowed = provenance.isin(("strict_oof", "frozen_forward_oos", "retrospective_causal_21d_non_oof"))
    else:
        raise ValueError(f"unknown diagnostic population {population}")
    result = july.loc[allowed & july[target].notna() & july[availability].notna()].copy()
    result[availability] = pd.to_datetime(result[availability], utc=True, errors="raise")
    result["target"] = pd.to_numeric(result[target], errors="raise").astype(int)
    return result.sort_values("cohort_anchor_utc", kind="stable").reset_index(drop=True)


def coverage_table(july: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target in TARGETS:
        availability = f"{target}_available_utc"
        for population in ("strict_oof_only", "strict_oof_plus_frozen_forward_diagnostic", "strict_oof_plus_forward_plus_retro_extension_diagnostic"):
            local = valid_rows(july, target, population=population)
            for week, part in local.groupby("utc_week", sort=True, observed=True):
                rows.append({"target": target, "population": population, "utc_week": week, "rows": int(len(part)), "positive_rows": int(part.target.sum()), "prevalence": float(part.target.mean()), "first_anchor_utc": part.cohort_anchor_utc.min(), "last_anchor_utc": part.cohort_anchor_utc.max(), "last_target_available_utc": part[availability].max(), "strict_oof_rows": int(part.mapping_provenance_role.eq("strict_oof").sum()), "frozen_forward_rows": int(part.mapping_provenance_role.eq("frozen_forward_oos").sum()), "retrospective_extension_rows": int(part.mapping_provenance_role.eq("retrospective_causal_21d_non_oof").sum())})
            if local.empty:
                rows.append({"target": target, "population": population, "utc_week": "NONE", "rows": 0, "positive_rows": 0, "prevalence": np.nan, "first_anchor_utc": pd.NaT, "last_anchor_utc": pd.NaT, "last_target_available_utc": pd.NaT, "strict_oof_rows": 0, "frozen_forward_rows": 0, "retrospective_extension_rows": 0})
    return pd.DataFrame(rows)


def _pipeline() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=0.25, max_iter=1000, solver="lbfgs", random_state=20260730)),
    ])


def _top_tie_metrics(frame: pd.DataFrame, score: np.ndarray) -> dict[str, float | int]:
    y = frame.target.to_numpy(dtype=int)
    n = len(frame); selected_rows = max(1, int(math.ceil(n * TOP_FRACTION)))
    anchor_key = pd.to_datetime(frame.cohort_anchor_utc, utc=True).astype("int64").to_numpy()
    order = np.lexsort((anchor_key, -np.asarray(score, dtype=float)))
    selected = order[:selected_rows]
    cutoff = float(np.asarray(score, dtype=float)[order[selected_rows - 1]])
    above = np.flatnonzero(np.asarray(score, dtype=float) > cutoff)
    ties = np.flatnonzero(np.asarray(score, dtype=float) == cutoff)
    slots = selected_rows - len(above)
    positives_above = int(y[above].sum())
    positives_tie = int(y[ties].sum())
    deterministic_precision = float(y[selected].mean())
    expected_precision = float((positives_above + slots * positives_tie / max(len(ties), 1)) / selected_rows)
    best_precision = float((positives_above + min(slots, positives_tie)) / selected_rows)
    worst_precision = float((positives_above + max(0, slots - (len(ties) - positives_tie))) / selected_rows)
    prevalence = float(y.mean())
    selected_days = max(1, int(pd.to_datetime(frame.iloc[selected].cohort_anchor_utc, utc=True).dt.floor("D").nunique()))
    return {"selected_rows": selected_rows, "cutoff_score": cutoff, "above_cutoff_rows": int(len(above)), "cutoff_tie_rows": int(len(ties)), "slots_from_tie": int(slots), "deterministic_precision": deterministic_precision, "expected_tie_precision": expected_precision, "best_tie_precision": best_precision, "worst_tie_precision": worst_precision, "deterministic_lift": float(deterministic_precision / prevalence) if prevalence else np.nan, "expected_tie_lift": float(expected_precision / prevalence) if prevalence else np.nan, "best_tie_lift": float(best_precision / prevalence) if prevalence else np.nan, "worst_tie_lift": float(worst_precision / prevalence) if prevalence else np.nan, "false_alerts": int((1 - y[selected]).sum()), "false_alerts_per_selected_day": float((1 - y[selected]).sum() / selected_days)}


def _ece(y: np.ndarray, score: np.ndarray, bins: int = 5) -> float:
    if not len(y):
        return np.nan
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        mask = (score >= edges[index]) & (score < edges[index + 1] if index < bins - 1 else score <= edges[index + 1])
        if mask.any():
            total += mask.mean() * abs(float(y[mask].mean()) - float(score[mask].mean()))
    return float(total)


def metric_row(prediction: pd.DataFrame, *, target: str, population: str, protocol: str, model: str, test_week: str | None = None, train_rows: int | None = None, train_positives: int | None = None) -> dict[str, Any]:
    y, score = prediction.target.to_numpy(int), prediction.prediction.to_numpy(float)
    values: dict[str, Any] = {"target": target, "population": population, "protocol": protocol, "model": model, "test_week": test_week, "rows": int(len(prediction)), "positive_rows": int(y.sum()), "prevalence": float(y.mean()) if len(y) else np.nan, "train_rows": train_rows, "train_positive_rows": train_positives, "brier": float(brier_score_loss(y, score)) if len(y) else np.nan, "ece5": _ece(y, score)}
    values["auc"] = float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else np.nan
    values["average_precision"] = float(average_precision_score(y, score)) if y.sum() else np.nan
    values.update(_top_tie_metrics(prediction, score))
    return values


def _predict_arms(train: pd.DataFrame, test: pd.DataFrame, features: Sequence[str]) -> tuple[list[pd.DataFrame], list[dict[str, Any]]]:
    y = train.target.to_numpy(int)
    outputs: list[pd.DataFrame] = []
    skipped: list[dict[str, Any]] = []
    for model in ("constant_0p5", "train_prevalence_prior", "logistic_90_common_features"):
        prediction = test.loc[:, ["cohort_anchor_utc", "utc_week", "mapping_provenance_role", "target"]].copy()
        if model == "constant_0p5":
            prediction["prediction"] = 0.5
        elif model == "train_prevalence_prior":
            prediction["prediction"] = float(y.mean()) if len(y) else 0.0
        else:
            if len(train) < MIN_LOGISTIC_ROWS or int(y.sum()) < MIN_LOGISTIC_POSITIVES or int((1 - y).sum()) < MIN_LOGISTIC_POSITIVES:
                skipped.append({"model": model, "reason": "insufficient_purged_training_support", "train_rows": int(len(train)), "train_positive_rows": int(y.sum()), "train_negative_rows": int((1 - y).sum())})
                continue
            fitted = _pipeline().fit(train.loc[:, list(features)], y)
            prediction["prediction"] = fitted.predict_proba(test.loc[:, list(features)])[:, 1]
        prediction["model"] = model
        outputs.append(prediction)
    return outputs, skipped


def grouped_purged_oof(frame: pd.DataFrame, *, features: Sequence[str], target: str, population: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions: list[pd.DataFrame] = []; metrics: list[dict[str, Any]] = []; skipped: list[dict[str, Any]] = []
    weeks = sorted(frame.utc_week.unique())
    if len(weeks) < 3:
        skipped.append({"target": target, "population": population, "protocol": "grouped_purged_oof", "reason": "fewer_than_three_utc_weeks", "weeks": len(weeks)})
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(skipped)
    for week in weeks:
        test = frame.loc[frame.utc_week.eq(week)].copy()
        start, end = test.cohort_anchor_utc.min(), test.cohort_anchor_utc.max()
        train = frame.loc[~frame.utc_week.eq(week) & ((frame.cohort_anchor_utc < start - PURGE) | (frame.cohort_anchor_utc > end + PURGE))].copy()
        arms, arm_skips = _predict_arms(train, test, features)
        for item in arm_skips:
            skipped.append({"target": target, "population": population, "protocol": "grouped_purged_oof", "test_week": week, **item})
        for prediction in arms:
            prediction["protocol"] = "grouped_purged_oof"; prediction["population"] = population; prediction["target_name"] = target; prediction["test_week"] = week
            prediction["train_rows"] = len(train); prediction["train_positive_rows"] = int(train.target.sum())
            predictions.append(prediction)
    all_predictions = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    if not all_predictions.empty:
        for model, local in all_predictions.groupby("model", sort=True):
            metrics.append(metric_row(local, target=target, population=population, protocol="grouped_purged_oof", model=model))
            for week, part in local.groupby("test_week", sort=True):
                metrics.append(metric_row(part, target=target, population=population, protocol="grouped_purged_oof", model=model, test_week=week, train_rows=int(part.train_rows.iloc[0]), train_positives=int(part.train_positive_rows.iloc[0])))
    return all_predictions, pd.DataFrame(metrics), pd.DataFrame(skipped)


def adjacent_week_transfer(frame: pd.DataFrame, *, features: Sequence[str], target: str, population: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions: list[pd.DataFrame] = []; metrics: list[dict[str, Any]] = []; skipped: list[dict[str, Any]] = []
    weeks = sorted(frame.utc_week.unique())
    for week in weeks[1:]:
        test = frame.loc[frame.utc_week.eq(week)].copy()
        start = test.cohort_anchor_utc.min()
        train = frame.loc[frame.cohort_anchor_utc < start - PURGE].copy()
        if train.empty:
            skipped.append({"target": target, "population": population, "protocol": "adjacent_week_forward_transfer", "test_week": week, "reason": "no_purged_prior_training_rows"})
            continue
        arms, arm_skips = _predict_arms(train, test, features)
        for item in arm_skips:
            skipped.append({"target": target, "population": population, "protocol": "adjacent_week_forward_transfer", "test_week": week, **item})
        for prediction in arms:
            prediction["protocol"] = "adjacent_week_forward_transfer"; prediction["population"] = population; prediction["target_name"] = target; prediction["test_week"] = week
            prediction["train_rows"] = len(train); prediction["train_positive_rows"] = int(train.target.sum())
            predictions.append(prediction)
    all_predictions = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    if not all_predictions.empty:
        for model, local in all_predictions.groupby("model", sort=True):
            metrics.append(metric_row(local, target=target, population=population, protocol="adjacent_week_forward_transfer", model=model))
            for week, part in local.groupby("test_week", sort=True):
                metrics.append(metric_row(part, target=target, population=population, protocol="adjacent_week_forward_transfer", model=model, test_week=week, train_rows=int(part.train_rows.iloc[0]), train_positives=int(part.train_positive_rows.iloc[0])))
    return all_predictions, pd.DataFrame(metrics), pd.DataFrame(skipped)


def run(*, panel: Path, retrospective: Path, extension: Path | None, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite immutable output {output_dir}")
    july, features, panel_source = load_panel(panel, extension)
    extension_audit = july23_extension_audit(extension) if extension is not None else pd.DataFrame([{"requirement": "July extension", "available": False, "evidence": "not supplied", "extension_permitted": False}])
    coverage = coverage_table(july)
    predictions: list[pd.DataFrame] = []; metrics: list[pd.DataFrame] = []; skipped: list[pd.DataFrame] = []
    for target in TARGETS:
        for population in ("strict_oof_only", "strict_oof_plus_frozen_forward_diagnostic", "strict_oof_plus_forward_plus_retro_extension_diagnostic"):
            frame = valid_rows(july, target, population=population)
            for function in (grouped_purged_oof, adjacent_week_transfer):
                local_predictions, local_metrics, local_skipped = function(frame, features=features, target=target, population=population)
                if not local_predictions.empty: predictions.append(local_predictions)
                if not local_metrics.empty: metrics.append(local_metrics)
                if not local_skipped.empty: skipped.append(local_skipped)
    prediction = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    metric = pd.concat(metrics, ignore_index=True) if metrics else pd.DataFrame()
    skipped_frame = pd.concat(skipped, ignore_index=True) if skipped else pd.DataFrame()
    stage = output_dir.parent / f".{output_dir.name}.{uuid.uuid4().hex}.stage"
    stage.mkdir(parents=True, exist_ok=False)
    try:
        outputs: dict[str, dict[str, Any]] = {}
        for name, table in (("july23_extension_readiness", extension_audit), ("coverage", coverage), ("metrics", metric), ("predictions", prediction), ("skipped_arms", skipped_frame)):
            path = stage / f"{name}.parquet"
            table.to_parquet(path, index=False, compression="zstd")
            outputs[name] = {"path": str(output_dir / path.name), "rows": int(len(table)), "sha256": sha256(path)}
        manifest = {
            "schema": SCHEMA, "status": "JULY_LOCAL_TRANSITION_DIAGNOSTIC_COMPLETE_NO_PROMOTION",
            "requested_extension": "through 2026-07-23", "extension_permitted": bool(extension_audit.available.all()) if not extension_audit.empty else False,
            "feature_count": len(features), "targets": list(TARGETS), "horizon_hours": HORIZON_HOURS,
            "sources": {"panel": panel_source, "retrospective_source": str(retrospective), "extension": str(extension) if extension is not None else None},
            "contracts": {
                "labels": "existing exact H12 one pooled-global causal-mapped top10 active/onset labels; no new labels and no per-timestamp/side/asset selection",
                "features": "exact strict 90-field decision-time common geometry only; no score, mapped EV, calendar, outcome, provenance, or future feature",
                "local_oof": "leave-one-UTC-week-out grouped OOF, with both-sided 36h purge around the held-out week; no HPO or action selection",
                "forward_transfer": "expanding prior July weeks only, ending at least 36h before the test week; controls and logistic share rows/features",
                "provenance": "strict_oof-only is reported separately. A second diagnostic-only population adds current frozen_forward_oos rows. A third adds the expressly retrospective/non-OOF July 20--23 extension; neither diagnostic population is promotable.",
                "tie_aware": "top10 is one pooled book per reported prediction set with deterministic timestamp ties plus expected/best/worst cutoff-tie precision/lift; no per-week quota",
                "promotion": "forbidden: no routing, policy replay, mapping, admission, timing or portfolio action",
            },
            "outputs": outputs, "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())}, "promotion_eligible": False,
        }
        _write_json(stage / "manifest.json", manifest)
        (stage / "manifest.sha256").write_text(sha256(stage / "manifest.json") + "\n", encoding="utf-8")
        os.replace(stage, output_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    result.add_argument("--retrospective", type=Path, default=DEFAULT_RETROSPECTIVE)
    result.add_argument("--extension", type=Path, default=DEFAULT_EXTENSION)
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    print(json.dumps(_safe(run(panel=args.panel, retrospective=args.retrospective, extension=args.extension, output_dir=args.output_dir)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
