#!/usr/bin/env python3
"""Causal pooled-common score mapping ablation for frozen joint economics.

This diagnostic deliberately consumes the immutable decomposition predictions;
it does not fit or refit any component/base model.  Each mapping is a small
score-to-realised-net transformation fitted only on *earlier resolved OOF*
rows.  April uses all resolved chronological OOF rows as a reused diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data_perp/artifacts/canonical_full_base_joint_economics_decomposition_20260729_v1"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/canonical_full_base_joint_economics_pooled_common_mapping_ablation_20260729_v1"
SCHEMA = "joint_economics_pooled_common_mapping_ablation_v1"
SOURCE_SCHEMA = "canonical_full_base_joint_economics_decomposition_v1"
SIDES = ("long", "short")
SCORES = {
    "direct": "direct_primary_score",
    "opportunity": "prediction__opportunity_score",
    "exit": "prediction__exit_mixture_score",
    "joint": "joint_score",
}
FRACTIONS = (0.01, 0.05, 0.10, 0.20)
MAPPING_ARMS = (
    "raw",
    "pooled_affine_ridge_net",
    "pooled_affine_ridge_net_side_residual",
    "side_prior_oof_z_then_pooled_affine_ridge_net",
    "blend_50_mapped_with_raw_common_anchor",
)
RIDGE_ALPHA = 25.0
SIDE_SHRINK_SUPPORT = 5000.0
SIDE_RESIDUAL_CAP_MULTIPLIER = 2.0
MIN_SIDE_SHARE_GATE = 0.05
IDENTITY = ("candidate_id", "side_name", "__symbol__", "__ts__", "arm")


@dataclass(frozen=True)
class Fold:
    fold_id: int
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _load_source(source_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], tuple[Fold, ...]]:
    paths = {name: source_root / name for name in (
        "manifest.json", "manifest.sha256", "development_strict_expanding_oof_predictions.parquet", "april_reused_diagnostic_predictions.parquet",
    )}
    if not all(path.is_file() for path in paths.values()):
        raise FileNotFoundError("immutable decomposition source is incomplete")
    manifest = json.loads(paths["manifest.json"].read_text())
    if manifest.get("schema") != SOURCE_SCHEMA:
        raise ValueError("canonical full-base joint-economics decomposition v1 is required")
    if paths["manifest.sha256"].read_text().split()[0] != sha256_file(paths["manifest.json"]):
        raise ValueError("source manifest SHA256 mismatch")
    for name in ("development_strict_expanding_oof_predictions.parquet", "april_reused_diagnostic_predictions.parquet"):
        expected = manifest.get("outputs_sha256", {}).get(name)
        if expected != sha256_file(paths[name]):
            raise ValueError(f"source SHA256 mismatch for {name}")
    folds = tuple(Fold(int(item["fold_id"]), pd.Timestamp(item["validation_start"]), pd.Timestamp(item["validation_end"])) for item in manifest["validation"]["folds"])
    oof, april = pd.read_parquet(paths["development_strict_expanding_oof_predictions.parquet"]), pd.read_parquet(paths["april_reused_diagnostic_predictions.parquet"])
    required = {*IDENTITY, "execution_label_end_utc", "execution_net_ev_12h", "fold_id", *SCORES.values()}
    if missing := sorted(required.difference(oof.columns)):
        raise ValueError("OOF source is missing columns: " + ", ".join(missing))
    if missing := sorted((required - {"fold_id"}).difference(april.columns)):
        raise ValueError("April source is missing columns: " + ", ".join(missing))
    for frame in (oof, april):
        for column in ("__ts__", "execution_label_end_utc"):
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
        if not frame["side_name"].isin(SIDES).all() or frame.duplicated(list(IDENTITY)).any():
            raise ValueError("source identity/side contract failed")
    if set(oof["fold_id"].unique()) != {fold.fold_id for fold in folds}:
        raise ValueError("OOF source folds do not match manifest")
    for fold in folds:
        held = oof["fold_id"].eq(fold.fold_id)
        if not oof.loc[held, "__ts__"].ge(fold.validation_start).all() or not oof.loc[held, "__ts__"].lt(fold.validation_end).all():
            raise ValueError("OOF row lies outside its manifest fold")
    return oof, april, manifest, folds


def robust_location_scale(values: Sequence[float]) -> tuple[float, float]:
    valid = np.asarray(values, dtype=float)
    valid = valid[np.isfinite(valid)]
    if not len(valid):
        return 0.0, 1.0
    median = float(np.median(valid))
    mad_scale = float(np.median(np.abs(valid - median)) * 1.4826)
    return median, max(mad_scale, float(np.std(valid)), 1e-12)


def fit_affine_ridge(score: Sequence[float], target: Sequence[float], *, alpha: float = RIDGE_ALPHA) -> tuple[float, float]:
    """One-dimensional ridge with an unpenalised intercept, in net units."""
    x, y = np.asarray(score, dtype=float), np.asarray(target, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2 or np.unique(x[valid]).size < 2:
        return 0.0, float(np.nanmean(y[valid])) if valid.any() else 0.0
    x, y = x[valid], y[valid]
    x_mean, y_mean = float(x.mean()), float(y.mean())
    slope = float(np.sum((x - x_mean) * (y - y_mean)) / (np.sum((x - x_mean) ** 2) + float(alpha)))
    return slope, y_mean - slope * x_mean


def affine_predict(score: Sequence[float], fit: tuple[float, float]) -> np.ndarray:
    return fit[0] * np.asarray(score, dtype=float) + fit[1]


def side_residual_corrections(target: Sequence[float], pooled_prediction: Sequence[float], side: Sequence[str], *, support: float = SIDE_SHRINK_SUPPORT, cap_multiplier: float = SIDE_RESIDUAL_CAP_MULTIPLIER) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Train-only side residual intercepts, shrunk and capped in net units."""
    y, prediction, sides = np.asarray(target, dtype=float), np.asarray(pooled_prediction, dtype=float), np.asarray(side, dtype=str)
    _, scale = robust_location_scale(y)
    cap = float(cap_multiplier) * scale
    values, audit = {}, {}
    for name in SIDES:
        residual = y[(sides == name) & np.isfinite(y) & np.isfinite(prediction)] - prediction[(sides == name) & np.isfinite(y) & np.isfinite(prediction)]
        n = int(len(residual))
        raw = float(np.mean(residual)) if n else 0.0
        shrink = n / (n + float(support))
        shrunk = raw * shrink
        values[name] = float(np.clip(shrunk, -cap, cap))
        audit[name] = {"rows": n, "raw_residual_intercept": raw, "shrink": shrink, "cap": cap, "correction": values[name]}
    return values, audit


def side_prior_z(train_score: Sequence[float], train_side: Sequence[str], values: Sequence[float], value_side: Sequence[str]) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    train_score, train_side = np.asarray(train_score, dtype=float), np.asarray(train_side, dtype=str)
    values, value_side = np.asarray(values, dtype=float), np.asarray(value_side, dtype=str)
    global_location, global_scale = robust_location_scale(train_score)
    output = np.empty(len(values), dtype=float)
    audit: dict[str, dict[str, float]] = {}
    for name in SIDES:
        loc, scale = robust_location_scale(train_score[train_side == name]) if (train_side == name).any() else (global_location, global_scale)
        mask = value_side == name
        output[mask] = np.clip((values[mask] - loc) / scale, -8.0, 8.0)
        audit[name] = {"location": loc, "scale": scale, "train_rows": int((train_side == name).sum())}
    return output, audit


def stable_global_top_mask(frame: pd.DataFrame, score: Sequence[float], fraction: float) -> np.ndarray:
    """A single global rank; sides are never quota-filtered or backfilled."""
    if fraction not in FRACTIONS:
        raise ValueError("only predeclared top-1/5/10/20% cuts are allowed")
    ranking = pd.DataFrame({"position": np.arange(len(frame)), "candidate_id": frame["candidate_id"].astype(str), "score": np.asarray(score, dtype=float)}).sort_values(["score", "candidate_id"], ascending=[False, True], kind="stable")
    mask = np.zeros(len(frame), dtype=bool)
    mask[ranking["position"].to_numpy()[:max(1, int(math.ceil(len(frame) * fraction)))]] = True
    return mask


def _fit_all_mapping_arms(train: pd.DataFrame, held: pd.DataFrame, score_column: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    raw_train, raw_held = train[score_column].to_numpy(float), held[score_column].to_numpy(float)
    target = train["execution_net_ev_12h"].to_numpy(float)
    pooled_fit = fit_affine_ridge(raw_train, target)
    pooled_train, pooled_held = affine_predict(raw_train, pooled_fit), affine_predict(raw_held, pooled_fit)
    correction, correction_audit = side_residual_corrections(target, pooled_train, train["side_name"].to_numpy(str))
    z_train, z_audit = side_prior_z(raw_train, train["side_name"].to_numpy(str), raw_train, train["side_name"].to_numpy(str))
    z_held, _ = side_prior_z(raw_train, train["side_name"].to_numpy(str), raw_held, held["side_name"].to_numpy(str))
    z_fit = fit_affine_ridge(z_train, target)
    return {
        "raw": raw_held,
        "pooled_affine_ridge_net": pooled_held,
        "pooled_affine_ridge_net_side_residual": pooled_held + np.array([correction.get(value, 0.0) for value in held["side_name"]], dtype=float),
        "side_prior_oof_z_then_pooled_affine_ridge_net": affine_predict(z_held, z_fit),
        "blend_50_mapped_with_raw_common_anchor": 0.5 * pooled_held + 0.5 * raw_held,
    }, {"pooled_affine": {"slope": pooled_fit[0], "intercept": pooled_fit[1]}, "side_residual": correction_audit, "side_prior_z": z_audit, "z_affine": {"slope": z_fit[0], "intercept": z_fit[1]}}


def causal_oof_mappings(frame: pd.DataFrame, folds: Sequence[Fold], score_column: str) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Map each fold only from earlier OOF rows resolved before its start."""
    output = {name: frame[score_column].to_numpy(float).copy() for name in MAPPING_ARMS}
    audit: list[dict[str, Any]] = []
    resolved = pd.to_datetime(frame["execution_label_end_utc"], utc=True)
    for fold in folds:
        held = frame["fold_id"].eq(fold.fold_id).to_numpy()
        prior = frame["fold_id"].lt(fold.fold_id).to_numpy() & resolved.lt(fold.validation_start).to_numpy()
        if fold.fold_id == 0 or not prior.any():
            audit.append({"fold_id": fold.fold_id, "status": "raw_fallback_no_prior_resolved_oof", "prior_rows": int(prior.sum()), "validation_start": fold.validation_start})
            continue
        mapped, detail = _fit_all_mapping_arms(frame.loc[prior], frame.loc[held], score_column)
        for name, value in mapped.items():
            output[name][held] = value
        audit.append({"fold_id": fold.fold_id, "status": "prior_resolved_oof_mapping", "prior_rows": int(prior.sum()), "validation_start": fold.validation_start, **detail})
    return output, audit


def april_mappings(oof: pd.DataFrame, april: pd.DataFrame, score_column: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """April reuses a map fitted on all chronological resolved OOF rows only."""
    april_start = pd.Timestamp("2025-04-01T00:00:00Z")
    train = oof.loc[pd.to_datetime(oof["execution_label_end_utc"], utc=True).lt(april_start)]
    if not len(train):
        raise ValueError("no resolved chronological OOF rows available for April map")
    result, detail = _fit_all_mapping_arms(train, april, score_column)
    return result, {"status": "all_resolved_chronological_oof_reused_april_diagnostic", "prior_rows": int(len(train)), **detail}


def _calibration(actual: np.ndarray, score: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(actual) & np.isfinite(score)
    if valid.sum() < 2:
        return {"calibration_slope": np.nan, "calibration_intercept": np.nan, "mae": np.nan}
    slope, intercept = fit_affine_ridge(score[valid], actual[valid], alpha=0.0)
    return {"calibration_slope": slope, "calibration_intercept": intercept, "mae": float(np.mean(np.abs(actual[valid] - score[valid])))}


def _selection_rows(frame: pd.DataFrame, score: np.ndarray, *, arm: str, score_name: str, split: str, scope: str, fraction: float, fold_id: int | None = None, week: pd.Timestamp | None = None) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    selected = stable_global_top_mask(frame, score, fraction)
    chosen, target = frame.loc[selected], frame["execution_net_ev_12h"].to_numpy(float)
    score_values = np.asarray(score, dtype=float)
    counts = chosen["side_name"].value_counts()
    selected_sum = float(chosen["execution_net_ev_12h"].sum())
    unique = int(pd.Series(score_values[np.isfinite(score_values)]).nunique())
    top_value = float(np.nanmax(score_values))
    top_plateau = int(np.isclose(score_values, top_value, rtol=0.0, atol=0.0).sum())
    record = {"split": split, "source_arm": arm, "score_name": score_name, "mapping_arm": scope, "rank_scope": "pooled_global", "period": "all" if week is None else str(week.date()), "fold_id": fold_id, "fraction": fraction, "eligible_rows": len(frame), "selected_rows": int(selected.sum()), "mean_net_bps": float(chosen["execution_net_ev_12h"].mean() * 10000.0), "sum_net": selected_sum, "positive_rate": float(chosen["execution_net_ev_12h"].gt(0).mean()), "score_unique_count": unique, "top_score_plateau_rows": top_plateau, "top_score_plateau_fraction": top_plateau / len(frame), **_calibration(target, score_values)}
    side_rows = []
    for side in SIDES:
        side_selected = chosen.loc[chosen["side_name"].eq(side)]
        side_sum = float(side_selected["execution_net_ev_12h"].sum())
        side_rows.append({"split": split, "source_arm": arm, "score_name": score_name, "mapping_arm": scope, "period": record["period"], "fold_id": fold_id, "fraction": fraction, "side_name": side, "selected_rows": int(len(side_selected)), "selected_share": len(side_selected) / len(chosen), "sum_net": side_sum, "net_contribution_share": side_sum / selected_sum if selected_sum else np.nan, "mean_net_bps": float(side_selected["execution_net_ev_12h"].mean() * 10000.0) if len(side_selected) else np.nan})
    gate = {"split": split, "source_arm": arm, "score_name": score_name, "mapping_arm": scope, "period": record["period"], "fold_id": fold_id, "fraction": fraction, "side_balance_gate_pass": bool(all(float(counts.get(side, 0)) / len(chosen) >= MIN_SIDE_SHARE_GATE for side in SIDES)), "dominant_side_share": float(counts.max() / len(chosen)), "unique_score_gate_pass": bool(unique > 1), "finite_score_gate_pass": bool(np.isfinite(score_values).all()), "selection_modified": False, "gate_role": "diagnostic_only_no_side_quota_or_replacement"}
    return record, side_rows, gate


def evaluate(frame: pd.DataFrame, mapped: dict[str, np.ndarray], *, split: str, source_arm: str, score_name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    metrics: list[dict[str, Any]] = []
    side_metrics: list[dict[str, Any]] = []
    gates: list[dict[str, Any]] = []
    periods: list[tuple[str, pd.DataFrame, int | None, pd.Timestamp | None]] = [("all", frame, None, None)]
    if "fold_id" in frame:
        periods.extend(("fold", group, int(fold), None) for fold, group in frame.groupby("fold_id", sort=True))
    week_key = frame["__ts__"].dt.to_period("W-SUN").dt.start_time.dt.tz_localize("UTC")
    weeks = list(frame.groupby(week_key, sort=True))
    periods.extend(("week", group, None, pd.Timestamp(week)) for week, group in weeks)
    if weeks:
        periods.append(("latest_week", weeks[-1][1], None, pd.Timestamp(weeks[-1][0])))
    for mapping_arm, values in mapped.items():
        full = pd.Series(values, index=frame.index)
        for period_kind, subset, fold_id, week in periods:
            if len(subset) == 0:
                continue
            local_score = full.loc[subset.index].to_numpy(float)
            for fraction in FRACTIONS:
                row, sides, gate = _selection_rows(subset, local_score, arm=source_arm, score_name=score_name, split=split, scope=mapping_arm, fraction=fraction, fold_id=fold_id, week=week)
                row["period_kind"] = period_kind
                gate["period_kind"] = period_kind
                for item in sides:
                    item["period_kind"] = period_kind
                metrics.append(row); side_metrics.extend(sides); gates.append(gate)
    return metrics, side_metrics, gates


def _write_outputs(output: Path, temporary: Path, manifest: dict[str, Any]) -> None:
    manifest["outputs_sha256"] = {str(path.relative_to(temporary)): sha256_file(path) for path in sorted(temporary.rglob("*")) if path.is_file() and path.name not in {"manifest.json", "manifest.sha256"}}
    manifest_path = temporary / "manifest.json"
    manifest_path.write_text(json.dumps(json_safe(manifest), indent=2, sort_keys=True, allow_nan=False) + "\n")
    (temporary / "manifest.sha256").write_text(f"{sha256_file(manifest_path)}  manifest.json\n")
    os.replace(temporary, output)


def run(args: argparse.Namespace) -> Path:
    oof, april, source_manifest, folds = _load_source(args.source_root)
    if args.plan_only:
        print(json.dumps({"source": str(args.source_root), "frozen_source_schema": SOURCE_SCHEMA, "scores": SCORES, "mapping_arms": MAPPING_ARMS, "folds": [asdict(fold) for fold in folds], "no_model_refits": True, "april": "reused diagnostic only"}, default=str, indent=2))
        return args.output
    if args.output.exists():
        raise FileExistsError(f"immutable output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=args.output.parent, prefix=f".{args.output.name}."))
    try:
        mapped_oof: list[pd.DataFrame] = []
        mapped_april: list[pd.DataFrame] = []
        metrics: list[dict[str, Any]] = []
        side_metrics: list[dict[str, Any]] = []
        gates: list[dict[str, Any]] = []
        audit: dict[str, Any] = {}
        for source_arm in sorted(oof["arm"].unique()):
            arm_oof, arm_april = oof.loc[oof["arm"].eq(source_arm)].copy(), april.loc[april["arm"].eq(source_arm)].copy()
            for score_name, column in SCORES.items():
                oof_maps, oof_audit = causal_oof_mappings(arm_oof, folds, column)
                april_maps, april_audit = april_mappings(arm_oof, arm_april, column)
                audit[f"{source_arm}/{score_name}"] = {"oof": oof_audit, "april": april_audit}
                for split, frame, maps, collector in (("development_strict_expanding_oof", arm_oof, oof_maps, mapped_oof), ("april_reused_diagnostic", arm_april, april_maps, mapped_april)):
                    long = frame.loc[:, [*IDENTITY, "execution_label_end_utc", "execution_net_ev_12h", *( ["fold_id"] if "fold_id" in frame else [])]].copy()
                    long["split"] = split; long["score_name"] = score_name; long["raw_score"] = frame[column].to_numpy(float)
                    for mapping_arm, values in maps.items():
                        long[f"mapped__{mapping_arm}"] = values
                    collector.append(long)
                    rows, sides, gate_rows = evaluate(frame, maps, split=split, source_arm=source_arm, score_name=score_name)
                    metrics.extend(rows); side_metrics.extend(sides); gates.extend(gate_rows)
        pd.concat(mapped_oof, ignore_index=True).to_parquet(temporary / "mapped_development_strict_expanding_oof_predictions.parquet", index=False, compression="zstd")
        pd.concat(mapped_april, ignore_index=True).to_parquet(temporary / "mapped_april_reused_diagnostic_predictions.parquet", index=False, compression="zstd")
        pd.DataFrame(metrics).to_parquet(temporary / "selection_metrics.parquet", index=False)
        pd.DataFrame(side_metrics).to_parquet(temporary / "side_share_contributions.parquet", index=False)
        pd.DataFrame(gates).to_parquet(temporary / "diagnostic_gates.parquet", index=False)
        manifest = {"schema": SCHEMA, "status": "COMPLETED_REUSED_APRIL_DIAGNOSTIC_NOT_PROMOTION_EVIDENCE", "source": {"root": str(args.source_root), "source_manifest_sha256": sha256_file(args.source_root / "manifest.json"), "source_outputs_sha256": source_manifest["outputs_sha256"], "runner_sha256": sha256_file(Path(__file__).resolve())}, "contract": {"no_model_refits": True, "mapping_arms": list(MAPPING_ARMS), "prediction_storage": "one row per source identity/score with mapping arms as columns", "scores": SCORES, "fold0": "raw fallback", "oof_map_rows": "strictly earlier fold and execution_label_end_utc < validation start", "side_residual": {"shrink": "n/(n+5000)", "cap": "2 * train-only robust net scale"}, "selection": {"scope": "one pooled global top-k per reported period", "fractions": list(FRACTIONS), "tie_break": "candidate_id ascending", "side_quotas": False, "replacement_or_backfill": False}, "april": "all resolved chronological OOF map, reused diagnostic only"}, "mapping_audit": audit, "folds": [asdict(fold) for fold in folds]}
        _write_outputs(args.output, temporary, manifest)
        return args.output
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
