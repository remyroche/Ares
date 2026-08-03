#!/usr/bin/env python3
"""Strict score-only PIT readiness audit for the August--November 2025 bridge.

This audit intentionally never reads execution target values.  It verifies that
the exact hourly candidates which have separate 1m replay labels can be joined
to the native base-label identity and to the frozen base/residual feature
contracts.  The output is a readiness artifact, not an OOF result and not a
promotion claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.features_gmm_ae import transform_ae_gmm_features
from extreme_price_movements.packb_static_point_feature_loader import (
    iter_point_in_time_feature_batches,
)
from scripts.run_febapr2025_canonical_base_oof import IDENTITY, _load_contracts


SCHEMA = "augnov2025_pit_scoring_preflight_v1"
STORE = ROOT / "data_perp/features/20260711_070000"
PROMOTION = ROOT / "docs/pipeline_roadmap/20260724/r3/packb_side_fs_hpo_promotion_v1.json"
AE = ROOT / "data_perp/artifacts/packb_side_local_ae_20260724_v1"
RESIDUAL = ROOT / "data_perp/artifacts/packb_side_local_residual_oof_20260724_v1_31_8"
NATIVE = ROOT / "data_perp/artifacts/20260720_s59_h5_signalclose_causal_trailing_cost100bps_labels_v2/labels"
OUT = ROOT / "data_perp/artifacts/augnov2025_pit_scoring_preflight_20260730_v1"
INPUTS = (
    (
        ROOT / "data_perp/artifacts/augoct2025_execution_ev_common30_policy_inputs_20260727_v1/candidates.parquet",
        ROOT / "data_perp/artifacts/augoct2025_execution_ev_common30_labels_20260727_v1/labels.parquet",
    ),
    (
        ROOT / "data_perp/artifacts/nov2025_execution_ev_common30_policy_inputs_20260727_v1/candidates.parquet",
        ROOT / "data_perp/artifacts/nov2025_execution_ev_common30_labels_20260727_v1/labels.parquet",
    ),
)


class PreflightError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False, default=str)
        handle.write("\n")
    os.replace(temporary, path)


def _identity_digest(frame: pd.DataFrame) -> str:
    payload = frame.loc[:, list(IDENTITY)].copy()
    payload["__ts__"] = pd.to_datetime(payload["__ts__"], utc=True, errors="raise").astype(str)
    payload = payload.astype(str).sort_values(list(IDENTITY), kind="stable")
    return hashlib.sha256(pd.util.hash_pandas_object(payload, index=False).values.tobytes()).hexdigest()


def _load_candidates() -> tuple[pd.DataFrame, dict[str, Path]]:
    pieces: list[pd.DataFrame] = []
    execution_by_month: dict[str, Path] = {}
    for candidate_path, execution_path in INPUTS:
        frame = pd.read_parquet(candidate_path, columns=list(IDENTITY))
        frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
        if frame["candidate_id"].duplicated().any() or not frame["__ts__"].dt.minute.eq(0).all():
            raise PreflightError(f"invalid hourly candidate identity: {candidate_path}")
        months = tuple(sorted(frame["__ts__"].dt.strftime("%Y-%m").unique()))
        for month in months:
            execution_by_month[month] = execution_path
        pieces.append(frame)
    candidates = pd.concat(pieces, ignore_index=True).sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    if candidates["candidate_id"].duplicated().any():
        raise PreflightError("candidate identity overlaps across source ledgers")
    return candidates, execution_by_month


def _native_identity(candidates: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for side in ("long", "short"):
        for month in (8, 9, 10, 11):
            path = NATIVE / f"train_global_{side}_5_2025_{month:02d}.parquet"
            if not path.is_file():
                raise PreflightError(f"missing native base-label shard: {path}")
            frame = pd.read_parquet(path, columns=list(IDENTITY))
            frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
            pieces.append(frame)
    native = pd.concat(pieces, ignore_index=True)
    if native["candidate_id"].duplicated().any():
        raise PreflightError("native candidate IDs are not unique")
    return native


def _identity_match(candidates: pd.DataFrame, other: pd.DataFrame) -> tuple[int, int]:
    left = candidates.loc[:, list(IDENTITY)].copy()
    right = other.loc[:, list(IDENTITY)].copy().rename(
        columns={"side_name": "_side", "__symbol__": "_symbol", "__ts__": "_ts"}
    )
    joined = left.merge(right, on="candidate_id", how="left", validate="one_to_one")
    exact = joined["_side"].eq(joined["side_name"]) & joined["_symbol"].eq(joined["__symbol__"]) & joined["_ts"].eq(joined["__ts__"])
    return int(exact.sum()), int((~exact).sum())


def _execution_identity(candidates: pd.DataFrame, execution_by_month: Mapping[str, Path]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for month, path in sorted(execution_by_month.items()):
        execution = pd.read_parquet(path, columns=list(IDENTITY))
        execution["__ts__"] = pd.to_datetime(execution["__ts__"], utc=True, errors="raise")
        pieces.append(execution.loc[execution["__ts__"].dt.strftime("%Y-%m").eq(month)])
    output = pd.concat(pieces, ignore_index=True)
    if output["candidate_id"].duplicated().any():
        raise PreflightError("execution-label candidate IDs are not unique")
    return output


def _raw_pit(candidates: pd.DataFrame, raw_contract: Mapping[str, Any]) -> tuple[pd.DataFrame, np.ndarray]:
    point = candidates.loc[:, list(IDENTITY)].copy()
    point["__symbol__"] = point["candidate_id"].astype(str).str.split("|", n=1).str[0]
    pieces: list[pd.DataFrame] = []
    matches: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    for batch in iter_point_in_time_feature_batches(point, feature_store_dir=STORE, feature_contract=raw_contract,
                                                     coverage_discovery=True, verify_frozen_schema=False,
                                                     max_rows_per_batch=2048, max_columns_per_read=64):
        pieces.append(batch.features)
        matches.append(np.asarray(batch.matched_exact_keys, dtype=bool))
        positions.append(np.asarray(batch.ledger_row_positions, dtype=np.int64))
    values = pd.concat(pieces, ignore_index=True)
    matched = np.concatenate(matches)
    order = np.concatenate(positions)
    sequence = np.argsort(order, kind="stable")
    if not np.array_equal(order[sequence], np.arange(len(point), dtype=np.int64)):
        raise PreflightError("PIT loader lost candidate identity order")
    return values.iloc[sequence].reset_index(drop=True), matched[sequence]


def _feature_coverage(candidates: pd.DataFrame, *, side: str, route: Mapping[str, Any]) -> dict[str, Any]:
    raw_contract = json.loads((AE / side / "loader_evidence/frozen_feature_contract.json").read_text())
    raw, exact = _raw_pit(candidates, raw_contract)
    if len(raw) != len(candidates):
        raise PreflightError(f"{side}: PIT row count mismatch")
    selected_base = tuple(route["features"])
    raw_state = raw.loc[:, list(route["state_features"])]
    state_complete = np.isfinite(raw_state.to_numpy(dtype=np.float32, copy=False)).all(axis=1)
    generated = pd.DataFrame(np.nan, index=raw.index, columns=list(route["generated"]), dtype=np.float32)
    if route["generated"] and state_complete.any():
        transformed = transform_ae_gmm_features(raw_state.loc[state_complete], route["state"], index=raw.index[state_complete])
        generated.loc[state_complete, list(route["generated"])] = transformed.loc[:, list(route["generated"])].to_numpy(dtype=np.float32, copy=False)
    base = pd.concat([raw, generated], axis=1).reindex(columns=list(selected_base))
    residual_features = json.loads((RESIDUAL / side / "feature_contract.json").read_text())["features"]
    derived = {"base_prediction", "base_rank_pct_timestamp_side", "base_rank_timestamp_side", "hour_sin", "hour_cos"}
    residual_raw = [name for name in residual_features if name not in derived]
    base_finite = {name: float(np.isfinite(base[name].to_numpy(dtype=np.float32, copy=False)).mean()) for name in selected_base}
    residual_finite = {name: float(np.isfinite(raw[name].to_numpy(dtype=np.float32, copy=False)).mean()) for name in residual_raw}
    records: list[dict[str, Any]] = []
    work = candidates.loc[:, ["__ts__", "side_name", "__symbol__"]].copy()
    work["month"] = work["__ts__"].dt.strftime("%Y-%m")
    work["exact_pit_key"] = exact
    work["base_joint_complete"] = np.isfinite(base.to_numpy(dtype=np.float32, copy=False)).all(axis=1)
    work["residual_raw_joint_complete"] = np.isfinite(raw.loc[:, residual_raw].to_numpy(dtype=np.float32, copy=False)).all(axis=1)
    for (month, report_side, symbol), group in work.groupby(["month", "side_name", "__symbol__"], sort=True, observed=True):
        records.append({"month": month, "side_name": report_side, "symbol": symbol, "candidate_rows": int(len(group)),
                        "exact_pit_key_rows": int(group["exact_pit_key"].sum()),
                        "base_joint_complete_rows": int(group["base_joint_complete"].sum()),
                        "residual_raw_joint_complete_rows": int(group["residual_raw_joint_complete"].sum())})
    return {"raw_contract_features": int(len(raw_contract["feature_columns"])),
            "exact_pit_key_rows": int(exact.sum()), "exact_pit_key_fraction": float(exact.mean()),
            "state_joint_complete_rows": int(state_complete.sum()), "state_joint_complete_fraction": float(state_complete.mean()),
            "base_features": list(selected_base), "base_feature_finite_fraction": base_finite,
            "base_joint_complete_rows": int(work["base_joint_complete"].sum()), "base_joint_complete_fraction": float(work["base_joint_complete"].mean()),
            "residual_features": residual_features, "residual_raw_feature_finite_fraction": residual_finite,
            "residual_raw_joint_complete_rows": int(work["residual_raw_joint_complete"].sum()), "residual_raw_joint_complete_fraction": float(work["residual_raw_joint_complete"].mean()),
            "coverage_rows": records}


def run(*, output_dir: Path = OUT) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise PreflightError(f"refusing to overwrite existing sealed audit: {output_dir}")
    candidates, execution_paths = _load_candidates()
    native = _native_identity(candidates)
    execution = _execution_identity(candidates, execution_paths)
    native_exact, native_missing = _identity_match(candidates, native)
    execution_exact, execution_missing = _identity_match(candidates, execution)
    contracts = _load_contracts(PROMOTION, AE)
    coverage = {
        side: _feature_coverage(candidates.loc[candidates.side_name.eq(side)].reset_index(drop=True), side=side, route=contracts[side])
        for side in ("long", "short")
    }
    rows: list[dict[str, Any]] = []
    for side in ("long", "short"):
        subset = candidates.loc[candidates.side_name.eq(side)].copy()
        subset["month"] = subset["__ts__"].dt.strftime("%Y-%m")
        detail = pd.DataFrame(coverage[side]["coverage_rows"])
        for _, item in detail.iterrows():
            candidate_slice = subset.loc[(subset["month"].eq(item.month)) & (subset["__symbol__"].eq(item.symbol))]
            native_ok, native_bad = _identity_match(candidate_slice, native)
            execution_ok, execution_bad = _identity_match(candidate_slice, execution)
            rows.append({**item.to_dict(), "native_identity_rows": native_ok, "native_identity_missing_rows": native_bad,
                         "execution_identity_rows": execution_ok, "execution_identity_missing_rows": execution_bad})
    by_symbol = pd.DataFrame(rows).sort_values(["month", "side_name", "symbol"], kind="stable")
    all_pit = all(coverage[side]["exact_pit_key_fraction"] == 1.0 for side in coverage)
    all_base = all(coverage[side]["base_joint_complete_fraction"] == 1.0 for side in coverage)
    all_residual = all(coverage[side]["residual_raw_joint_complete_fraction"] == 1.0 for side in coverage)
    score_materialization_feasible = native_missing == 0 and execution_missing == 0 and all_pit and all_base and all_residual
    report = {
        "schema": SCHEMA,
        "status": "SEALED_SCORE_ONLY_PIT_PREFLIGHT_NON_PROMOTION",
        "scope": "August--November 2025 frozen 30-asset common universe; score-materialization readiness only, no target values read or evaluated",
        "model_sample_cadence": "1h", "exact_replay_bar_cadence": "1m_labels_only",
        "candidate_rows": int(len(candidates)), "candidate_identity_sha256": _identity_digest(candidates),
        "candidate_month_side_rows": candidates.assign(month=candidates["__ts__"].dt.strftime("%Y-%m")).groupby(["month", "side_name"], sort=True).size().rename("rows").reset_index().to_dict(orient="records"),
        "native_base_identity": {"exact_rows": native_exact, "missing_or_mismatched_rows": native_missing},
        "exact_execution_label_identity": {"exact_rows": execution_exact, "missing_or_mismatched_rows": execution_missing},
        "feature_coverage": coverage,
        "score_materialization": {
            "data_and_pit_feasible": score_materialization_feasible,
            "required_model_cutoff": "fit base and residual only from rows whose required native labels resolve before 2025-08-01T00:00:00Z; retain immutable HPO/feature contracts",
            "important_limit": "No serialized final base/residual models frozen through July are present in this audit. A separate score-only frozen-fit materializer is required before an August--November score ledger can be claimed.",
            "not_authorized": "do not evaluate or fit mapping/calibrator/policy on August--November execution labels during score materialization",
        },
        "inputs_sha256": {
            "augoct_candidates": _sha256(INPUTS[0][0]), "nov_candidates": _sha256(INPUTS[1][0]),
            "augoct_execution_labels": _sha256(INPUTS[0][1]), "nov_execution_labels": _sha256(INPUTS[1][1]),
            "promotion": _sha256(PROMOTION),
        },
    }
    output_dir.mkdir(parents=True)
    by_symbol.to_csv(output_dir / "coverage_by_month_side_symbol.csv", index=False)
    _write_json(output_dir / "readiness_report.json", report)
    manifest = {"schema": SCHEMA, "status": report["status"], "model_sample_cadence": "1h", "exact_replay_bar_cadence": "1m_labels_only",
                "outputs_sha256": {"coverage_by_month_side_symbol.csv": _sha256(output_dir / "coverage_by_month_side_symbol.csv"), "readiness_report.json": _sha256(output_dir / "readiness_report.json")}}
    _write_json(output_dir / "manifest.json", manifest)
    (output_dir / "manifest.sha256").write_text(f"{_sha256(output_dir / 'manifest.json')}  manifest.json\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    print(json.dumps(run(output_dir=args.output_dir), indent=2))


if __name__ == "__main__":
    main()
