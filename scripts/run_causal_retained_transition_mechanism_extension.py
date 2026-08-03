#!/usr/bin/env python3
"""Gate a causal extension of the retained transition mechanisms.

This is intentionally a *joinability first* runner.  The sparse-mechanism
screen persisted strict-OOF predictions, but not fitted model state.  A new
forward refit would therefore be a distinct causal refit, not reproduction of
the frozen classifier.  Before taking that exception, this runner proves that
the identical 90-field decision-time geometry covers every frozen candidate in
both requested windows.  A partial later-July join is fail-closed: no model is
fit, no score is emitted, and no interaction, policy or portfolio replay may
run on a changed candidate population.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FORWARD_ROOT = ROOT / "data_perp/artifacts/exact_strict_oof_hurdle_distributional_ablation_20260730_v3"
GEOMETRY_ROOT = ROOT / "data_perp/artifacts/forward_exact_transition_geometry_20260730_v1"
PANEL_ROOT = ROOT / "data_perp/artifacts/pooled_historical_current_transition_panel_20260730_v1"
SPARSE_ROOT = ROOT / "data_perp/artifacts/pooled_historical_current_sparse_transition_mechanism_ablation_20260730_v1"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/causal_retained_transition_mechanism_extension_20260730_v2"
SCHEMA = "causal_retained_transition_mechanism_extension_v1"
WINDOWS = ("may_to_june_forward_control", "later_july_forward")
CONTEXT_SHIFT_HOURS = 0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if value is pd.NaT or (not isinstance(value, (list, tuple, Mapping)) and pd.isna(value)):
        return None
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_safe(dict(payload)), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _bound(root: Path, name: str) -> tuple[Path, Path]:
    path, manifest, seal = root / name, root / "manifest.json", root / "manifest.sha256"
    if not path.is_file() or not manifest.is_file() or not seal.is_file():
        raise FileNotFoundError(f"incomplete immutable source: {root}")
    if seal.read_text(encoding="utf-8").split()[0] != sha256(manifest):
        raise ValueError(f"manifest seal fails: {root}")
    return path, manifest


def join_candidates(forward: pd.DataFrame, geometry: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach only the semantic context at the frozen candidate signal time."""

    required_forward = {"candidate_id", "__ts__", "window", "execution_decision_utc", "support_label_available_utc"}
    required_geometry = {"signal_context_utc", "common_transition_context_available"}
    missing = sorted((required_forward - set(forward)) | (required_geometry - set(geometry)))
    if missing:
        raise ValueError(f"source lacks required context-join fields: {missing}")
    work = forward.loc[forward["window"].isin(WINDOWS)].copy()
    if set(work["window"].unique()) != set(WINDOWS):
        raise ValueError("frozen forward source does not contain both required windows")
    if work["candidate_id"].duplicated().any():
        raise ValueError("frozen candidate identity is duplicate")
    for column in ("__ts__", "execution_decision_utc", "support_label_available_utc"):
        work[column] = pd.to_datetime(work[column], utc=True, errors="raise")
    if not work["execution_decision_utc"].eq(work["__ts__"] + pd.Timedelta(hours=1)).all():
        raise ValueError("forward decision timing is not candidate timestamp plus one hour")
    context = geometry.loc[:, ["signal_context_utc", "common_transition_context_available"]].copy()
    context["signal_context_utc"] = pd.to_datetime(context["signal_context_utc"], utc=True, errors="raise")
    if context["signal_context_utc"].duplicated().any():
        raise ValueError("semantic context timestamp is duplicate")
    work["required_signal_context_utc"] = work["__ts__"] - pd.Timedelta(hours=CONTEXT_SHIFT_HOURS)
    joined = work.merge(context, left_on="required_signal_context_utc", right_on="signal_context_utc", how="left", validate="many_to_one")
    joined["context_joined"] = joined["common_transition_context_available"].astype("boolean").fillna(False).astype(bool)
    joined["score_status"] = np.where(joined["context_joined"], "WITHHELD_FULL_WINDOW_GATE", "WITHHELD_CONTEXT_UNAVAILABLE")
    coverage = []
    for window, local in joined.groupby("window", sort=True):
        missing = local.loc[~local.context_joined, "__ts__"]
        coverage.append({
            "window": window, "candidate_rows": int(len(local)), "joined_rows": int(local.context_joined.sum()),
            "coverage": float(local.context_joined.mean()), "full_window_coverage": bool(local.context_joined.all()),
            "first_candidate_utc": local["__ts__"].min(), "last_candidate_utc": local["__ts__"].max(),
            "first_missing_candidate_utc": missing.min() if len(missing) else pd.NaT,
            "last_missing_candidate_utc": missing.max() if len(missing) else pd.NaT,
            "missing_rows": int((~local.context_joined).sum()),
        })
    return joined, pd.DataFrame(coverage)


def run(*, forward_root: Path, geometry_root: Path, panel_root: Path, sparse_root: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite immutable output {output_dir}")
    forward_path, forward_manifest = _bound(forward_root, "forward_predictions.parquet")
    geometry_path, geometry_manifest = _bound(geometry_root, "hourly_geometry.parquet")
    panel_path, panel_manifest = _bound(panel_root, "transition_panel.parquet")
    sparse_path, sparse_manifest = _bound(sparse_root, "predictions.parquet")
    forward = pd.read_parquet(forward_path)
    geometry = pd.read_parquet(geometry_path, columns=["signal_context_utc", "common_transition_context_available"])
    joined, coverage = join_candidates(forward, geometry)
    complete = bool(coverage["full_window_coverage"].all())
    # No refit is valid until the two-window candidate universe is identical.
    # These null columns prevent downstream code from treating join status as a
    # raw transition score while keeping a fully auditable row-level ledger.
    for name in ("compression_onset_probability", "memory_active_probability", "state_active_probability"):
        joined[name] = np.nan
    status = "CAUSAL_EXTENSION_BLOCKED_INCOMPLETE_EXACT_CONTEXT_COVERAGE" if not complete else "CAUSAL_EXTENSION_JOINABLE_REFIT_REQUIRED"
    stage = Path(tempfile.mkdtemp(dir=output_dir.parent, prefix=f".{output_dir.name}."))
    try:
        joined.to_parquet(stage / "candidate_context_join.parquet", index=False, compression="zstd")
        coverage.to_csv(stage / "coverage.csv", index=False)
        manifest = {
            "schema": SCHEMA, "status": status, "promotion_eligible": False,
            "interaction_ablation_ran": False,
            "gates": {"full_may_to_june_coverage": bool(coverage.loc[coverage.window.eq(WINDOWS[0]), "full_window_coverage"].iloc[0]), "full_later_july_coverage": bool(coverage.loc[coverage.window.eq(WINDOWS[1]), "full_window_coverage"].iloc[0]), "allow_causal_refit_and_interaction": complete},
            "contracts": {
                "join": "exact retained 90-field semantic context at the frozen candidate signal timestamp; execution decision is signal plus one hour; no as-of fill, interpolation, subset ranking, side quota or timestamp tie-break",
                "model_state": "the sparse mechanism artifact contains strict-OOF predictions only; fitted model state was not persisted. Any later fit is explicitly a causal refit under the identical logistic/features/targets recipe, never a claim of frozen-state reproduction",
                "fail_closed": "unless every candidate in both frozen forward windows has decision-time context, all transition probabilities and every score-by-hurdle or uncertainty interaction remain withheld",
                "prohibited": "no raw transition-probability ranking, broad HPO, timing/MAE/target-price/wait inputs, policy replay or portfolio replay",
            },
            "source_hashes": {"forward_predictions": {"path": str(forward_path), "sha256": sha256(forward_path), "manifest_sha256": sha256(forward_manifest)}, "semantic_context": {"path": str(geometry_path), "sha256": sha256(geometry_path), "manifest_sha256": sha256(geometry_manifest)}, "transition_panel": {"path": str(panel_path), "sha256": sha256(panel_path), "manifest_sha256": sha256(panel_manifest)}, "sparse_mechanism_predictions": {"path": str(sparse_path), "sha256": sha256(sparse_path), "manifest_sha256": sha256(sparse_manifest)}},
            "coverage": coverage.to_dict("records"),
            "outputs_sha256": {"candidate_context_join.parquet": sha256(stage / "candidate_context_join.parquet"), "coverage.csv": sha256(stage / "coverage.csv")},
            "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        }
        _write_json(stage / "manifest.json", manifest)
        (stage / "manifest.sha256").write_text(sha256(stage / "manifest.json") + "\n", encoding="utf-8")
        os.replace(stage, output_dir)
    except Exception:
        import shutil
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forward-root", type=Path, default=FORWARD_ROOT)
    parser.add_argument("--geometry-root", type=Path, default=GEOMETRY_ROOT)
    parser.add_argument("--panel-root", type=Path, default=PANEL_ROOT)
    parser.add_argument("--sparse-root", type=Path, default=SPARSE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(json.dumps(_safe(run(forward_root=arguments.forward_root, geometry_root=arguments.geometry_root, panel_root=arguments.panel_root, sparse_root=arguments.sparse_root, output_dir=arguments.output_dir)), sort_keys=True))
