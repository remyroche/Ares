#!/usr/bin/env python3
"""Score a past, fully materialised period with frozen P8U C0/C1 packages.

This is an offline compatibility audit for a package trained at a later
cutoff.  It is deliberately *not* a live-causal replay: the package may have
seen the scored period during fitting.  The script preserves the production
data-flow shape nonetheless: it constructs C0/C1 predictions and agreement
selection from target-free inputs first, then attaches exact policy outcomes
only to a separate replay panel.

It exists to make reverse-time checks reproducible and to prevent them from
being misrepresented as rolling-origin OOS evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.inference.p8u_c0_c1_agreement_tier import (  # noqa: E402
    UNPAIRED_ORDER_C0_THEN_C1,
    select_c0_c1_agreement_tiers,
)
from scripts import build_p8u_successor_c0_c1_prequential as mapper  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(path.rglob("*")):
        if child.is_file():
            digest.update(str(child.relative_to(path)).encode())
            digest.update(_sha256(child).encode())
    return digest.hexdigest()


def _utc(value: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _load_package(path: Path, *, family: str) -> mapper._Package:
    manifest = json.loads((path / "package_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("family") != family:
        raise AssertionError(f"{path}: expected {family}, found {manifest.get('family')}")
    # Historic packages were serialised while the builder was a direct CLI,
    # so their dataclass may be recorded as __main__._Package.  Bind the
    # canonical class before loading; this does not mutate the package.
    setattr(sys.modules["__main__"], "_Package", mapper._Package)
    package = joblib.load(path / "package.joblib")
    if tuple(package.fields) != tuple(manifest["feature_order"]):
        raise AssertionError(f"{path}: package feature order differs from manifest")
    if package.family != family:
        raise AssertionError(f"{path}: package payload family differs from manifest")
    return package


def _score(
    full: pd.DataFrame, *, package: mapper._Package, start: pd.Timestamp, end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    test = full.loc[
        full["__decision_ts__"].ge(start) & full["__decision_ts__"].lt(end)
    ].copy()
    if test.empty:
        raise ValueError("no target-free rows in requested compatibility window")
    if start != start.normalize() or start.day != 1:
        raise ValueError("reverse-time compatibility windows must start on a calendar-month boundary")
    state = mapper._shift(package, full, held=start, held_end=end)
    daily = state.set_index("decision_day")["recent_shift_bps"]
    output = test.loc[:, list(mapper.IDENTITY)].copy()
    output["static_expected_bps"] = package.predict(test)
    output["score_band_curve_bps"] = package.curve[mapper._score_bands(test)]
    output["recent_shift_bps"] = output["__decision_ts__"].dt.normalize().map(daily).fillna(0.0).to_numpy(float)
    output["mc1_expected_bps"] = output["static_expected_bps"] + output["recent_shift_bps"]
    output["mapper_family"] = package.family
    if not np.isfinite(output["mc1_expected_bps"]).all():
        raise AssertionError("frozen package emitted non-finite expected EV")
    return output, state


def _selector_input(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.rename(columns={"mc1_expected_bps": "bcf_mc1_expected_bps"}).copy()
    out["current_mc1_expected_bps"] = out["bcf_mc1_expected_bps"]
    out["auction_priority_bps"] = out["bcf_mc1_expected_bps"]
    return out


def run(args: argparse.Namespace) -> Path:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"immutable output already exists: {output}")
    start, end = _utc(args.start), _utc(args.end)
    if end <= start:
        raise ValueError("require start < end")
    source = args.source.resolve()
    snapshots = [path.resolve() for path in args.c1_snapshots]
    package_root = args.package_root.resolve()
    c0_path = package_root / "c0_base_geometry" / str(args.package_month)
    c1_path = package_root / "c1_lva_geometry" / str(args.package_month)
    for folder in (c0_path, c1_path):
        if not (folder / "package.joblib").is_file() or not (folder / "package_manifest.json").is_file():
            raise FileNotFoundError(f"missing frozen package {folder}")
    target_free, full = mapper._read_inputs(source, snapshots)
    policy_columns = [column for column in target_free.columns if column.startswith("policy_")]
    if policy_columns:
        raise AssertionError(f"target-free source has policy fields: {sorted(policy_columns)}")
    c0 = _load_package(c0_path, family="c0_base_geometry")
    c1 = _load_package(c1_path, family="c1_lva_geometry")
    c0_pred, c0_shift = _score(full, package=c0, start=start, end=end)
    c1_pred, c1_shift = _score(full, package=c1, start=start, end=end)
    selected = select_c0_c1_agreement_tiers(
        c0_scores=_selector_input(c0_pred.loc[:, list(mapper.IDENTITY) + ["mc1_expected_bps"]]),
        c1_scores=_selector_input(c1_pred.loc[:, list(mapper.IDENTITY) + ["mc1_expected_bps"]]),
        admission_floor_bps=float(args.admission_floor_bps),
        unpaired_order=UNPAIRED_ORDER_C0_THEN_C1,
    )
    if any(column.startswith("policy_") for column in selected.columns):
        raise AssertionError("target-free selection carries policy fields")
    labels = full.loc[:, list(mapper.IDENTITY) + [column for column in full.columns if column.startswith("policy_")]]
    replay = selected.merge(labels, on=list(mapper.IDENTITY), how="left", validate="one_to_one")
    temporary = output.with_name(f".{output.name}.build-{os.getpid()}")
    temporary.mkdir(parents=True)
    try:
        c0_pred.to_parquet(temporary / "predictions_c0_target_free.parquet", index=False, compression="zstd")
        c1_pred.to_parquet(temporary / "predictions_c1_target_free.parquet", index=False, compression="zstd")
        selected.to_parquet(temporary / "agreement_tier_target_free_predictions.parquet", index=False, compression="zstd")
        replay.to_parquet(temporary / "agreement_tier_policy_replay.parquet", index=False, compression="zstd")
        c0_shift.assign(family="c0_base_geometry").to_parquet(temporary / "c0_shift_state.parquet", index=False, compression="zstd")
        c1_shift.assign(family="c1_lva_geometry").to_parquet(temporary / "c1_shift_state.parquet", index=False, compression="zstd")
        manifest: dict[str, Any] = {
            "schema": "p8u_successor_c0_c1_reverse_time_compatibility_v1",
            "status": "complete",
            "scope": "offline no-order reverse-time package compatibility audit; not live-causal OOS evidence",
            "window": {"start": start.isoformat(), "end_exclusive": end.isoformat()},
            "reverse_time_warning": (
                "frozen package training may overlap this earlier scored window; results test package/input compatibility only "
                "and must not be used for promotion or live economic claims"
            ),
            "source": {
                "path": str(source),
                "manifest_sha256": _sha256(source / "run_manifest.json"),
                "target_free_panel_sha256": _sha256(source / "target_free_upstream_scores.parquet"),
                "policy_replay_panel_sha256": _sha256(source / "policy_attached_replay_panel.parquet"),
            },
            "c1_snapshots": [
                {"path": str(path), "sha256": _sha256(path), "manifest_sha256": _sha256(path.parent / "run_manifest.json")}
                for path in snapshots
            ],
            "packages": {
                "c0": {"path": str(c0_path), "sha256": _tree_hash(c0_path)},
                "c1": {"path": str(c1_path), "sha256": _tree_hash(c1_path)},
            },
            "admission_floor_bps": float(args.admission_floor_bps),
            "tier": "both-admitted -> C0-only -> C1-only",
            "target_free_rows": int(len(c0_pred)),
            "selected_target_free_rows": int(len(selected)),
            "causality": {
                "prediction": "target-free source and candidate-time C1 snapshots only; policy labels attached after selection",
                "shift": "prior 21d, 10% trimmed resolved-label residual shift as implemented by the frozen package",
                "authority": "no portfolio, execution, exchange, or order authority",
            },
            "outputs": {},
        }
        for name in (
            "predictions_c0_target_free.parquet", "predictions_c1_target_free.parquet",
            "agreement_tier_target_free_predictions.parquet", "agreement_tier_policy_replay.parquet",
            "c0_shift_state.parquet", "c1_shift_state.parquet",
        ):
            manifest["outputs"][name] = _sha256(temporary / name)
        (temporary / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--c1-snapshots", type=Path, action="append", required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--package-month", required=True, help="frozen package month, e.g. 2026-09")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--admission-floor-bps", type=float, default=50.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.admission_floor_bps <= 0.0:
        raise ValueError("admission floor must be positive")
    print(run(args))


if __name__ == "__main__":
    main()
