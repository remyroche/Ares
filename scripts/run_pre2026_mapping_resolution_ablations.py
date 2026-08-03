#!/usr/bin/env python3
"""Strictly pre-2026 EV-mapping-resolution ablations on frozen hourly rows.

The sealed v2 stack output supplies model-OOF scores and the identical 2026
assessment population.  This script changes only the causal score-to-EV map:
it never refits an alpha/context model and never uses a 2026 label for fitting,
support selection or method selection.  All methods are preregistered and are
reported side-by-side; no forward winner is promoted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data_perp/artifacts/final_identical_row_regime_stack_gam_ablation_20260730_v3"
OUT = ROOT / "data_perp/artifacts/pre2026_mapping_resolution_ablations_20260730_v2"
SCHEMA = "pre2026_mapping_resolution_ablations_v2"
TARGET, GROSS, COST, ALPHA = "execution_net_ev_12h", "execution_gross_ev_12h", "execution_cost_return", "__first_touch_target_soft__"
IDENTITY = ("candidate_id", "__ts__", "__symbol__", "side_name")
TOP = 0.10


class MappingError(RuntimeError):
    pass


@dataclass(frozen=True)
class MappingMethod:
    name: str
    mode: str
    side_shrink_support: int = 0
    bins: int = 0
    min_bin_support: int = 0
    strict_rank: bool = False


# Fixed before looking at 2026 results.  The support constants are not HPO
# winners: they bracket a material local-versus-pooled shrinkage range.
METHODS = (
    MappingMethod("monotone_isotonic_control", "global_isotonic"),
    MappingMethod("rank_preserving_isotonic", "global_isotonic", strict_rank=True),
    MappingMethod("side_support_shrunk_5k", "side_isotonic", side_shrink_support=5_000),
    MappingMethod("side_support_shrunk_25k", "side_isotonic", side_shrink_support=25_000),
    MappingMethod("side_minbin_64x1k", "binned_side", bins=64, min_bin_support=1_000),
    MappingMethod("side_minbin_32x2p5k", "binned_side", bins=32, min_bin_support=2_500),
    MappingMethod("side_minbin_64x1k_strict_rank", "binned_side", bins=64, min_bin_support=1_000, strict_rank=True),
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(partial, path)


def _sealed_input(root: Path, *, arms: Sequence[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    manifest_path = root / "manifest.json"
    marker = root / "manifest.sha256"
    if not manifest_path.is_file() or not marker.is_file() or marker.read_text().split(maxsplit=1)[0] != sha(manifest_path):
        raise MappingError("sealed v2 stack manifest checksum is missing or invalid")
    manifest = json.loads(manifest_path.read_text())
    contract = manifest.get("contract", {})
    if manifest.get("schema") != "final_identical_row_regime_stack_gam_ablation_v3" or manifest.get("status") != "SEALED_STRICT_FORWARD_IDENTICAL_ROW_ABLATION_NON_PROMOTION":
        raise MappingError("mapping ablation requires the corrected sealed v3 identical-row stack")
    history_contract = str(contract.get("historical_fit_selection_calibration", ""))
    if contract.get("candidate_cadence") != "1h" or ("2022-2025 only" not in history_contract and "strictly OOF pre-2026" not in history_contract) or "2026 only" not in str(contract.get("forward", "")):
        raise MappingError("v3 stack does not prove the 1h pre-2026/2026 contract")
    names = ("historical_oof_scores.parquet", "frozen_2026_candidate_scores.parquet")
    paths = tuple(root / name for name in names)
    for path in paths:
        if not path.is_file() or manifest.get("outputs_sha256", {}).get(path.name) != sha(path):
            raise MappingError(f"sealed v2 input checksum mismatch: {path}")
    # The frozen forward sidecar also carries the full regime context.  Mapping
    # resolution is downstream of that model and must not materialise those
    # 40+ columns again: identity, raw score and sealed economics are enough.
    common_columns = [*IDENTITY, "arm", "raw_score", TARGET, "execution_label_end_utc"]
    filters = [("arm", "in", list(arms))] if arms else None
    historical = pd.read_parquet(paths[0], columns=common_columns, filters=filters)
    forward = pd.read_parquet(paths[1], columns=[*common_columns, GROSS, COST, ALPHA], filters=filters)
    for name, frame in (("historical", historical), ("forward", forward)):
        missing = [field for field in (*IDENTITY, "arm", "raw_score", TARGET, "execution_label_end_utc") if field not in frame]
        if missing:
            raise MappingError(f"{name} frozen score file lacks {missing}")
        frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
        frame["execution_label_end_utc"] = pd.to_datetime(frame["execution_label_end_utc"], utc=True, errors="raise")
        if not (frame["__ts__"].astype("int64") % pd.Timedelta(hours=1).value == 0).all():
            raise MappingError(f"{name} rows are not hourly")
        if frame.duplicated([*IDENTITY, "arm"]).any():
            raise MappingError(f"{name} has duplicate candidate-arm rows")
    if not historical["__ts__"].lt(pd.Timestamp("2026-01-01", tz="UTC")).all() or not forward["__ts__"].ge(pd.Timestamp("2026-01-01", tz="UTC")).all():
        raise MappingError("frozen historical/forward split is invalid")
    if not historical["execution_label_end_utc"].gt(historical["__ts__"]).all() or not forward["execution_label_end_utc"].gt(forward["__ts__"]).all():
        raise MappingError("invalid label availability endpoint")
    return historical, forward, manifest


def _increasing_iso(scores: np.ndarray, targets: np.ndarray) -> tuple[Any, dict[str, Any]]:
    valid = np.isfinite(scores) & np.isfinite(targets)
    x, y = scores[valid], targets[valid]
    if len(x) < 8 or np.unique(x).size < 2:
        value = float(np.nanmean(y)) if len(y) else 0.0
        return (lambda z: np.full(len(z), value, dtype=float)), {"fit_rows": int(len(x)), "kind": "constant"}
    model = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(x, y)
    return (lambda z: np.asarray(model.predict(np.asarray(z, dtype=float)), dtype=float)), {"fit_rows": int(len(x)), "kind": "increasing_isotonic"}


def _binned_side_map(frame: pd.DataFrame, global_map: Any, method: MappingMethod) -> tuple[Any, dict[str, Any]]:
    """Fit a side map with each bin shrunk toward global EV at that score."""
    side_models: dict[str, Any] = {}
    audits: dict[str, dict[str, Any]] = {}
    for side, local in frame.groupby("side_name", observed=True, sort=True):
        work = local.loc[np.isfinite(local.raw_score) & np.isfinite(local[TARGET]), ["raw_score", TARGET]].sort_values("raw_score", kind="stable")
        if len(work) < max(8, method.bins):
            continue
        pieces = []
        for chunk in np.array_split(work, method.bins):
            if len(chunk):
                score = float(chunk.raw_score.mean())
                support = len(chunk)
                local_ev = float(chunk[TARGET].mean())
                weight = support / (support + method.min_bin_support)
                pieces.append((score, weight * local_ev + (1.0 - weight) * float(global_map(np.asarray([score]))[0]), support))
        bins = pd.DataFrame(pieces, columns=["score", "target", "support"]).groupby("score", as_index=False).agg(target=("target", "mean"), support=("support", "sum"))
        model = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(bins.score, bins.target, sample_weight=bins.support)
        side_models[str(side)] = lambda z, model=model: np.asarray(model.predict(np.asarray(z, dtype=float)), dtype=float)
        audits[str(side)] = {"side_rows": int(len(work)), "bins": int(len(bins)), "min_bin_rows": int(bins.support.min()), "max_bin_rows": int(bins.support.max())}

    def mapper(scores: np.ndarray, sides: np.ndarray) -> np.ndarray:
        values = global_map(scores)
        for side, model in side_models.items():
            mask = np.asarray(sides, dtype=str) == side
            values[mask] = model(np.asarray(scores)[mask])
        return values

    return mapper, {"side_bin_audit": audits}


def _fit_method(history: pd.DataFrame, method: MappingMethod) -> tuple[Any, dict[str, Any]]:
    global_map, audit = _increasing_iso(history.raw_score.to_numpy(float), history[TARGET].to_numpy(float))
    if method.mode == "global_isotonic":
        return lambda scores, sides: global_map(scores), {"global": audit, "method": method.name}
    if method.mode == "side_isotonic":
        local: dict[str, tuple[Any, int]] = {}
        for side, work in history.groupby("side_name", observed=True, sort=True):
            local[str(side)] = (*_increasing_iso(work.raw_score.to_numpy(float), work[TARGET].to_numpy(float))[0:1], len(work))

        def mapper(scores: np.ndarray, sides: np.ndarray) -> np.ndarray:
            values = global_map(scores)
            for side, (model, support) in local.items():
                mask = np.asarray(sides, dtype=str) == side
                weight = support / (support + method.side_shrink_support)
                values[mask] = weight * model(np.asarray(scores)[mask]) + (1.0 - weight) * values[mask]
            return values

        return mapper, {"global": audit, "method": method.name, "side_rows": {side: support for side, (_, support) in local.items()}, "side_shrink_support": method.side_shrink_support}
    if method.mode == "binned_side":
        mapper, binned_audit = _binned_side_map(history, global_map, method)
        return mapper, {"global": audit, "method": method.name, "bins": method.bins, "min_bin_support": method.min_bin_support, **binned_audit}
    raise MappingError(f"unknown mapping mode {method.mode}")


def _strict_rank(mapped: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Resolve map plateaus with a tiny, score-order-preserving EV perturbation."""
    rank = pd.Series(scores).rank(method="first", pct=True).to_numpy(float)
    scale = max(float(np.nanmax(np.abs(mapped))) if np.isfinite(mapped).any() else 0.0, 1e-4)
    return np.asarray(mapped, dtype=float) + rank * scale * 1e-12


def _rank(left: pd.Series, right: pd.Series) -> float:
    valid = left.notna() & right.notna()
    if valid.sum() < 3:
        return float("nan")
    x = rankdata(left.loc[valid].to_numpy(float), method="average")
    y = rankdata(right.loc[valid].to_numpy(float), method="average")
    # Avoid pandas' repeated index alignment/rank allocations in every
    # calendar cell; this is the same average-tie Spearman definition.
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _select(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values(["mapped_score", "raw_score", "candidate_id"], ascending=[False, False, True], kind="stable")
    selected = ordered.head(max(1, math.ceil(len(frame) * TOP))).candidate_id
    result = frame.copy()
    result["selected_global_top10"] = result.candidate_id.isin(set(selected))
    return result


def _tie_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    counts = frame.mapped_score.value_counts(dropna=False)
    ordered = frame.sort_values(["mapped_score", "raw_score", "candidate_id"], ascending=[False, False, True], kind="stable")
    cutoff = ordered.iloc[max(0, math.ceil(len(frame) * TOP) - 1)].mapped_score
    cutoff_size = int(frame.mapped_score.eq(cutoff).sum())
    return {
        "mapped_unique_values": int(frame.mapped_score.nunique(dropna=False)),
        "mapped_tie_mass": float(counts.max() / len(frame)),
        "cutoff_tie_rows": cutoff_size,
        "cutoff_tie_share": float(cutoff_size / len(frame)),
        # Gate is diagnostic only.  It never admits, drops or reorders a row.
        "resolution_gate_pass": bool(counts.max() / len(frame) <= .10 and cutoff_size / len(frame) <= .025 and frame.mapped_score.nunique() >= 100),
    }


def _evaluate(frame: pd.DataFrame, *, arm: str, method: MappingMethod) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = _select(frame)
    debug = os.environ.get("MAPPING_ABLATION_DEBUG") == "1"
    if debug: print("evaluation:selected", flush=True)
    selected = frame.loc[frame.selected_global_top10]
    periods = []
    for period_type, key in (("week", frame.__ts__.dt.strftime("%G-W%V")), ("month", frame.__ts__.dt.strftime("%Y-%m"))):
        for period, local in frame.groupby(key, observed=True, sort=True):
            pick = local.loc[local.selected_global_top10]
            # The requested week/month tail is economic (selected-book EV),
            # not a noisy small-cell IC HPO surface.  Aggregate IC remains
            # reported on the full frozen forward universe.
            periods.append({"arm": arm, "method": method.name, "period_type": period_type, "period": period, "candidate_rows": len(local), "global_selected_rows": len(pick), "mean_mapped_ev": pick.mapped_score.mean(), "mean_net_ev": pick[TARGET].mean(), "mean_gross_ev": pick[GROSS].mean(), "mean_cost": pick[COST].mean(), "hit_rate": pick[TARGET].gt(0).mean()})
    period_frame = pd.DataFrame(periods)
    if debug: print("evaluation:periods", flush=True)
    summary: dict[str, Any] = {"arm": arm, "method": method.name, "mapping_mode": method.mode, "strict_rank": method.strict_rank, "candidate_rows": len(frame), "top10_rows": len(selected), "alpha_rank_ic": _rank(frame.mapped_score, frame[ALPHA]), "execution_rank_ic": _rank(frame.mapped_score, frame[TARGET]), "raw_score_net_rank_ic": _rank(frame.raw_score, frame[TARGET]), "top10_net_ev": selected[TARGET].mean(), "top10_gross_ev": selected[GROSS].mean(), "top10_cost": selected[COST].mean(), "top10_hit_rate": selected[TARGET].gt(0).mean(), **_tie_metrics(frame)}
    for period_type in ("week", "month"):
        local = period_frame.loc[period_frame.period_type.eq(period_type)]
        summary[f"{period_type}_net_ev_q10"] = local.mean_net_ev.quantile(.10)
        summary[f"{period_type}_net_ev_q50"] = local.mean_net_ev.quantile(.50)
        latest = local.sort_values("period").tail(1).iloc[0]
        worst = local.loc[local.mean_net_ev.idxmin()]
        summary[f"latest_{period_type}"] = latest.period
        summary[f"latest_{period_type}_net_ev"] = latest.mean_net_ev
        summary[f"worst_{period_type}"] = worst.period
        summary[f"worst_{period_type}_net_ev"] = worst.mean_net_ev
    sides = []
    for side, local in frame.groupby("side_name", observed=True, sort=True):
        pick = local.loc[local.selected_global_top10]
        sides.append({"arm": arm, "method": method.name, "side_name": side, "candidate_rows": len(local), "global_selected_rows": len(pick), "execution_rank_ic": _rank(local.mapped_score, local[TARGET]), "top10_net_ev": pick[TARGET].mean(), "top10_gross_ev": pick[GROSS].mean(), "top10_cost": pick[COST].mean(), "top10_hit_rate": pick[TARGET].gt(0).mean()})
    if debug: print("evaluation:sides", flush=True)
    calibration = frame.loc[:, ["mapped_score", TARGET]].copy()
    calibration["bin"] = pd.qcut(calibration.mapped_score.rank(method="first"), q=10, labels=False, duplicates="drop")
    calibration = calibration.groupby("bin", observed=True).agg(candidate_rows=(TARGET, "size"), mean_mapped_ev=("mapped_score", "mean"), mean_net_ev=(TARGET, "mean")).reset_index()
    calibration["arm"], calibration["method"] = arm, method.name
    calibration["signed_error"] = calibration.mean_mapped_ev - calibration.mean_net_ev
    summary["calibration_mae_decile"] = calibration.signed_error.abs().mean()
    summary["calibration_bias_decile"] = calibration.signed_error.mean()
    if debug: print("evaluation:calibration", flush=True)
    return summary, period_frame, pd.DataFrame(sides), calibration


def run(*, input_root: Path = INPUT, output: Path = OUT, arms: Sequence[str] | None = None, methods: Sequence[str] | None = None) -> Path:
    output = Path(output)
    if output.exists():
        raise MappingError(f"immutable output already exists: {output}")
    historical, forward, source_manifest = _sealed_input(Path(input_root), arms=arms)
    available_arms = sorted(set(historical.arm) & set(forward.arm))
    requested = list(arms) if arms else available_arms
    unknown = sorted(set(requested) - set(available_arms))
    if unknown:
        raise MappingError(f"unknown requested arms: {unknown}")
    arms = requested
    available_methods = {method.name: method for method in METHODS}
    unknown_methods = sorted(set(methods or ()) - set(available_methods))
    if unknown_methods:
        raise MappingError(f"unknown requested mapping methods: {unknown_methods}")
    requested_methods = [available_methods[name] for name in methods] if methods else list(METHODS)
    expected_keys: pd.DataFrame | None = None
    for arm in arms:
        key = forward.loc[forward.arm.eq(arm), list(IDENTITY)].sort_values(list(IDENTITY), kind="stable").reset_index(drop=True)
        if expected_keys is None:
            expected_keys = key
        elif not expected_keys.equals(key):
            raise MappingError("forward candidate universe differs by arm")
    rows: list[dict[str, Any]] = []
    periods: list[pd.DataFrame] = []
    sides: list[pd.DataFrame] = []
    calibration: list[pd.DataFrame] = []
    audit: list[dict[str, Any]] = []
    scores: list[pd.DataFrame] = []
    unavailable: list[dict[str, Any]] = []
    for arm in arms:
        history = historical.loc[historical.arm.eq(arm)].copy()
        current = forward.loc[forward.arm.eq(arm)].copy()
        if history.raw_score.notna().sum() < 8:
            unavailable.append({"arm": arm, "reason": "no_strict_pre2026_oof_raw_score_for_mapping"})
            continue
        for method in requested_methods:
            mapper, fit_audit = _fit_method(history, method)
            mapped = mapper(current.raw_score.to_numpy(float), current.side_name.to_numpy())
            if method.strict_rank:
                mapped = _strict_rank(mapped, current.raw_score.to_numpy(float))
            result = current.copy()
            result["method"], result["mapped_score"] = method.name, mapped
            # Persist exactly the same pooled global membership that the
            # evaluator reports; this is not a per-period or per-side cut.
            result = _select(result)
            row, per, side, cal = _evaluate(result, arm=arm, method=method)
            rows.append(row); periods.append(per); sides.append(side); calibration.append(cal)
            # Sealing every non-selected candidate for every map would retain
            # 8m+ duplicate rows.  The immutable source already owns that
            # full universe; retain the auditable selected global book only.
            scores.append(result.loc[result.selected_global_top10, [*IDENTITY, "arm", "method", "raw_score", "mapped_score", TARGET, GROSS, COST, "execution_label_end_utc", "selected_global_top10"]])
            audit.append({"arm": arm, "method": method.name, "history_rows": len(history), "history_label_end_max": history.execution_label_end_utc.max(), "forward_rows": len(current), "forward_label_min": current.execution_label_end_utc.min(), "strict_pre2026_fit": bool(history.__ts__.lt(pd.Timestamp("2026-01-01", tz="UTC")).all()), "no_2026_label_used_for_fit_or_hpo": True, **fit_audit})
    stage = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}."))
    try:
        pd.DataFrame(rows).to_csv(stage / "metrics_summary.csv", index=False)
        pd.concat(periods, ignore_index=True).to_parquet(stage / "period_metrics.parquet", index=False)
        pd.concat(sides, ignore_index=True).to_parquet(stage / "side_metrics.parquet", index=False)
        pd.concat(calibration, ignore_index=True).to_parquet(stage / "calibration_deciles.parquet", index=False)
        pd.concat(scores, ignore_index=True).to_parquet(stage / "frozen_2026_selected_mapping_scores.parquet", index=False)
        write_json(stage / "mapping_fit_audit.json", audit)
        pd.DataFrame(unavailable).to_csv(stage / "unavailable_arms.csv", index=False)
        contract = {"sample_cadence": "1h", "labels": "existing exact 12h labels with 1m nested replay only", "historical_fit": "only frozen pre-2026 OOF raw score + resolved 12h target rows", "forward_assessment": "frozen 2026 rows; labels are read only after all mapping methods are fixed", "methods": [method.__dict__ for method in requested_methods], "preregistered_full_method_grid": [method.__dict__ for method in METHODS], "evaluated_arms": arms, "selection": "one pooled global top10 after arm-local mapped EV; raw score only as exact mapped-score tie key, then candidate_id", "tie_gates": "diagnostic-only resolution gates; never a candidate filter", "no_hpo": "method grid is preregistered and no 2026 outcome selects a winner", "source_stack_manifest_sha256": sha(Path(input_root) / "manifest.json")}
        write_json(stage / "contract.json", contract)
        files = [path for path in stage.iterdir() if path.is_file()]
        manifest = {"schema": SCHEMA, "status": "SEALED_PRE2026_MAPPING_RESOLUTION_ABLATION_NON_PROMOTION", "promotion_eligible": False, "inputs": {str((Path(input_root) / "manifest.json").resolve()): sha(Path(input_root) / "manifest.json"), str((Path(input_root) / "historical_oof_scores.parquet").resolve()): sha(Path(input_root) / "historical_oof_scores.parquet"), str((Path(input_root) / "frozen_2026_candidate_scores.parquet").resolve()): sha(Path(input_root) / "frozen_2026_candidate_scores.parquet")}, "contract": contract, "outputs_sha256": {path.name: sha(path) for path in files}}
        write_json(stage / "manifest.json", manifest)
        (stage / "manifest.sha256").write_text(f"{sha(stage / 'manifest.json')}  manifest.json\n")
        os.replace(stage, output)
        return output
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def merge_chunks(*, chunks: Sequence[Path], output: Path) -> Path:
    """Seal a metric-only suite manifest from independent immutable chunks."""
    output = Path(output)
    if output.exists() or not chunks:
        raise MappingError("merge requires non-empty chunks and a new output")
    summaries: list[pd.DataFrame] = []
    periods: list[pd.DataFrame] = []
    sides: list[pd.DataFrame] = []
    calibration: list[pd.DataFrame] = []
    audits: list[Any] = []
    unavailable: list[pd.DataFrame] = []
    contracts: list[dict[str, Any]] = []
    inputs: dict[str, str] = {}
    for chunk in map(Path, chunks):
        manifest_path = chunk / "manifest.json"
        marker = chunk / "manifest.sha256"
        if not manifest_path.is_file() or not marker.is_file() or marker.read_text().split(maxsplit=1)[0] != sha(manifest_path):
            raise MappingError(f"invalid chunk manifest: {chunk}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema") != SCHEMA or manifest.get("status") != "SEALED_PRE2026_MAPPING_RESOLUTION_ABLATION_NON_PROMOTION":
            raise MappingError(f"not a sealed mapping chunk: {chunk}")
        contracts.append(manifest["contract"])
        inputs[str(manifest_path.resolve())] = sha(manifest_path)
        for name in ("metrics_summary.csv", "period_metrics.parquet", "side_metrics.parquet", "calibration_deciles.parquet", "mapping_fit_audit.json", "unavailable_arms.csv"):
            path = chunk / name
            if manifest.get("outputs_sha256", {}).get(name) != sha(path):
                raise MappingError(f"chunk checksum mismatch: {path}")
        summaries.append(pd.read_csv(chunk / "metrics_summary.csv"))
        periods.append(pd.read_parquet(chunk / "period_metrics.parquet"))
        sides.append(pd.read_parquet(chunk / "side_metrics.parquet"))
        calibration.append(pd.read_parquet(chunk / "calibration_deciles.parquet"))
        audits.extend(json.loads((chunk / "mapping_fit_audit.json").read_text()))
        unavailable.append(pd.read_csv(chunk / "unavailable_arms.csv"))
    summary = pd.concat(summaries, ignore_index=True)
    if summary.duplicated(["arm", "method"]).any():
        raise MappingError("chunk suite contains duplicate arm/method results")
    source_hashes = {contract["source_stack_manifest_sha256"] for contract in contracts}
    if len(source_hashes) != 1:
        raise MappingError("chunks do not share one corrected source stack")
    stage = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}."))
    try:
        summary.to_csv(stage / "metrics_summary.csv", index=False)
        pd.concat(periods, ignore_index=True).to_parquet(stage / "period_metrics.parquet", index=False)
        pd.concat(sides, ignore_index=True).to_parquet(stage / "side_metrics.parquet", index=False)
        pd.concat(calibration, ignore_index=True).to_parquet(stage / "calibration_deciles.parquet", index=False)
        write_json(stage / "mapping_fit_audit.json", audits)
        pd.concat(unavailable, ignore_index=True).drop_duplicates().to_csv(stage / "unavailable_arms.csv", index=False)
        contract = {"suite": "immutable chunks merged without rereading any forward labels", "sample_cadence": "1h", "selection": contracts[0]["selection"], "tie_gates": contracts[0]["tie_gates"], "no_hpo": contracts[0]["no_hpo"], "source_stack_manifest_sha256": source_hashes.pop(), "chunk_count": len(chunks), "all_methods": [method.__dict__ for method in METHODS]}
        write_json(stage / "contract.json", contract)
        files = [path for path in stage.iterdir() if path.is_file()]
        manifest = {"schema": SCHEMA, "status": "SEALED_PRE2026_MAPPING_RESOLUTION_SUITE_NON_PROMOTION", "promotion_eligible": False, "inputs": inputs, "contract": contract, "outputs_sha256": {path.name: sha(path) for path in files}}
        write_json(stage / "manifest.json", manifest)
        (stage / "manifest.sha256").write_text(f"{sha(stage / 'manifest.json')}  manifest.json\n")
        os.replace(stage, output)
        return output
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--arms", default=None, help="comma-separated frozen arm names; sealed chunks may be combined externally")
    parser.add_argument("--methods", default=None, help="comma-separated preregistered mapping methods")
    parser.add_argument("--merge-chunks", type=Path, nargs="+", default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = vars(parse_args())
    if args.pop("merge_chunks"):
        chunks = args.pop("merge_chunks")
        args.pop("input_root"); args.pop("arms"); args.pop("methods")
        print(merge_chunks(chunks=chunks, **args))
    else:
        for key in ("arms", "methods"):
            if args[key]:
                args[key] = tuple(part for part in args[key].split(",") if part)
        print(run(**args))
