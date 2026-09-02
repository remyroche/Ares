#!/usr/bin/env python3
"""Atomically publish one causal daily C0/C1 prior-21-day shift receipt.

This is deliberately a calibration-state publisher, not a model trainer.  It
keeps the two frozen mapper packages byte-identical and derives a fresh daily
shift only from policy outcomes whose labels resolved before the requested UTC
decision day.  Each published day is immutable; ``latest.json`` is the sole
atomic pointer that advances.

The publisher is fail-closed by design:

* an existing day can never be overwritten;
* a later publication must follow the previous day exactly;
* previously consumed resolved outcomes must still be identical in the
  supplied append-only policy ledger;
* the required 21-day support must be present and causal;
* package family, feature order, and package-tree fingerprints are pinned in
  the state-root manifest.

It has no feature, candidate, model-fitting, portfolio, account, exchange, or
order-submission authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid
from typing import Any

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_p8u_successor_c0_c1_prequential as builder


SCHEMA = "p8u-c0-c1-daily-calibration-state-v1"
WINDOW_DAYS = 21
TRIM_FRACTION = 0.10
MINIMUM_RESOLVED_ROWS = 500
REQUIRED_LEDGER_COLUMNS = (
    "candidate_id", "__decision_ts__", "__symbol__", "side_name",
    "base_rank_ts", "policy_path_valid", "policy_net_bps",
    "policy_label_available_ts",
)
SNAPSHOT_COLUMNS = REQUIRED_LEDGER_COLUMNS


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
            digest.update(str(child.relative_to(path)).encode("utf-8"))
            digest.update(_sha256(child).encode("ascii"))
    return digest.hexdigest()


def _utc_day(raw: str) -> pd.Timestamp:
    stamp = pd.Timestamp(raw)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    if stamp != stamp.normalize():
        raise ValueError("decision day must be a UTC midnight timestamp")
    return stamp


def _write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load_package(path: Path, *, expected_family: str) -> tuple[object, dict[str, Any], str]:
    path = path.resolve()
    manifest_path = path / "package_manifest.json"
    pickle_path = path / "package.joblib"
    if not manifest_path.is_file() or not pickle_path.is_file():
        raise FileNotFoundError(f"mapper package is incomplete: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(manifest.get("family")) != expected_family:
        raise ValueError(f"expected {expected_family}, got {manifest.get('family')}")
    # Historical package pickles bind their compatibility dataclass through
    # ``__main__``.  This is the same explicit binding used by the sealed
    # successor mapper loader; it does not alter numeric package contents.
    setattr(sys.modules["__main__"], "_Package", builder._Package)
    package = joblib.load(pickle_path)
    fields = tuple(map(str, getattr(package, "fields", ())))
    if fields != tuple(map(str, manifest.get("feature_order") or ())):
        raise ValueError(f"{expected_family} serialized feature order differs from manifest")
    curve = np.asarray(getattr(package, "curve", ()), dtype=float)
    if curve.shape != (10,) or not np.isfinite(curve).all():
        raise ValueError(f"{expected_family} score-band curve is invalid")
    return package, manifest, _tree_hash(path)


def _normalise_ledger(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"policy ledger is absent: {path}")
    frame = pd.read_parquet(path, columns=list(REQUIRED_LEDGER_COLUMNS)).copy()
    missing = set(REQUIRED_LEDGER_COLUMNS).difference(frame.columns)
    if missing:
        raise KeyError(f"policy ledger lacks required columns: {sorted(missing)}")
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    frame["__symbol__"] = frame["__symbol__"].astype(str)
    frame["side_name"] = frame["side_name"].astype(str).str.lower()
    if not frame["side_name"].eq("long").all():
        raise ValueError("daily calibration state is long-only")
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    frame["policy_label_available_ts"] = pd.to_datetime(
        frame["policy_label_available_ts"], utc=True, errors="coerce"
    )
    if frame["candidate_id"].duplicated().any():
        raise ValueError("policy ledger has duplicate candidate identifiers")
    return frame.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def _eligible_snapshot(ledger: pd.DataFrame, *, day: pd.Timestamp) -> pd.DataFrame:
    finite_policy = np.isfinite(pd.to_numeric(ledger["policy_net_bps"], errors="coerce"))
    eligible = ledger.loc[
        ledger["__decision_ts__"].ge(day - pd.Timedelta(days=WINDOW_DAYS))
        & ledger["__decision_ts__"].lt(day)
        & ledger["policy_path_valid"].fillna(False).astype(bool)
        & ledger["policy_label_available_ts"].lt(day)
        & finite_policy,
        list(SNAPSHOT_COLUMNS),
    ].copy()
    if eligible.empty:
        raise RuntimeError("daily calibration has no causally resolved policy outcomes")
    if eligible["policy_label_available_ts"].ge(day).any():
        raise AssertionError("daily calibration consumed an unresolved outcome")
    return eligible.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def _assert_snapshot_unchanged(previous: pd.DataFrame, ledger: pd.DataFrame) -> None:
    """Require all previously consumed resolved rows to remain immutable."""
    current = ledger.loc[:, list(SNAPSHOT_COLUMNS)].copy()
    current = current.loc[current["candidate_id"].isin(previous["candidate_id"])].copy()
    joined = previous.merge(current, on="candidate_id", how="left", suffixes=("__prior", "__current"), validate="one_to_one")
    missing = joined["__decision_ts____current"].isna()
    if missing.any():
        raise ValueError("append-only policy ledger dropped a previously consumed outcome")
    for field in ("__decision_ts__", "__symbol__", "side_name", "policy_label_available_ts"):
        left = joined[f"{field}__prior"]
        right = joined[f"{field}__current"]
        if not left.eq(right).all():
            raise ValueError(f"append-only policy ledger rewrote prior {field}")
    for field in ("base_rank_ts", "policy_net_bps"):
        left = pd.to_numeric(joined[f"{field}__prior"], errors="coerce").to_numpy(float)
        right = pd.to_numeric(joined[f"{field}__current"], errors="coerce").to_numpy(float)
        if not np.array_equal(left, right, equal_nan=True):
            raise ValueError(f"append-only policy ledger rewrote prior {field}")
    for field in ("policy_path_valid",):
        left = joined[f"{field}__prior"].fillna(False).astype(bool)
        right = joined[f"{field}__current"].fillna(False).astype(bool)
        if not left.eq(right).all():
            raise ValueError(f"append-only policy ledger rewrote prior {field}")


def _state_root_manifest(
    *, c0_path: Path, c0_manifest: dict[str, Any], c0_tree: str,
    c1_path: Path, c1_manifest: dict[str, Any], c1_tree: str,
) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "status": "ACTIVE_APPEND_ONLY_CALIBRATION_STATE",
        "calibration": {
            "window_days": WINDOW_DAYS,
            "trim_fraction": TRIM_FRACTION,
            "residual": "policy_net_bps - frozen_package_score_band_curve[base_rank_ts_timestamp_local_decile]",
            "eligibility": "policy_path_valid and finite policy_net_bps and policy_label_available_ts < decision_day",
            "publication": "one immutable UTC-day receipt; latest.json is atomically replaced only after receipt publication",
        },
        "packages": {
            "c0_base_geometry": {
                "path": str(c0_path.resolve()), "tree_sha256": c0_tree,
                "manifest_sha256": _sha256(c0_path / "package_manifest.json"),
                "family": c0_manifest.get("family"), "feature_order": c0_manifest.get("feature_order"),
            },
            "c1_lva_geometry": {
                "path": str(c1_path.resolve()), "tree_sha256": c1_tree,
                "manifest_sha256": _sha256(c1_path / "package_manifest.json"),
                "family": c1_manifest.get("family"), "feature_order": c1_manifest.get("feature_order"),
            },
        },
        "causality": {
            "outcomes": "only resolved exact rich-policy outcomes strictly before the UTC decision day",
            "models": "frozen mapper packages are read-only; publisher never refits or mutates them",
            "state": "prior consumed rows are verified immutable before later publication",
            "authority": "no feature, candidate, portfolio, exchange, or order authority",
        },
    }


def _require_manifest(root: Path, expected: dict[str, object]) -> None:
    path = root / "state_manifest.json"
    if path.exists():
        actual = json.loads(path.read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError("daily calibration root is bound to another mapper/package contract")
        return
    _write_json_atomic(path, expected)


def _day_directory(root: Path, day: pd.Timestamp) -> Path:
    return root / f"day={day:%Y-%m-%d}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--decision-day", required=True, help="UTC midnight covered by this receipt")
    parser.add_argument("--policy-ledger", type=Path, required=True)
    parser.add_argument("--c0-package", type=Path, required=True)
    parser.add_argument("--c1-package", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, help="optional immutable ledger manifest")
    parser.add_argument("--minimum-resolved-rows", type=int, default=MINIMUM_RESOLVED_ROWS)
    parser.add_argument("--bootstrap", action="store_true", help="allow the first immutable daily receipt")
    args = parser.parse_args()

    root = args.state_root.resolve()
    day = _utc_day(args.decision_day)
    if day > pd.Timestamp.now(tz="UTC").normalize():
        raise ValueError("cannot publish calibration for a future UTC decision day")
    if int(args.minimum_resolved_rows) < 1:
        raise ValueError("minimum resolved rows must be positive")

    c0_package, c0_manifest, c0_tree = _load_package(args.c0_package, expected_family="c0_base_geometry")
    c1_package, c1_manifest, c1_tree = _load_package(args.c1_package, expected_family="c1_lva_geometry")
    root.mkdir(parents=True, exist_ok=True)
    expected_root = _state_root_manifest(
        c0_path=args.c0_package, c0_manifest=c0_manifest, c0_tree=c0_tree,
        c1_path=args.c1_package, c1_manifest=c1_manifest, c1_tree=c1_tree,
    )
    manifest_path = root / "state_manifest.json"
    latest_path = root / "latest.json"
    if not manifest_path.exists() and not args.bootstrap:
        raise ValueError("first daily calibration receipt requires --bootstrap")
    _require_manifest(root, expected_root)

    latest: dict[str, Any] | None = None
    if latest_path.exists():
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        previous_day = _utc_day(str(latest.get("decision_day")))
        if day != previous_day + pd.Timedelta(days=1):
            raise ValueError("daily calibration publication must advance exactly one UTC day")
        previous_dir = root / str(latest.get("receipt_dir"))
        previous_input = previous_dir / "eligible_policy_input.parquet"
        if not previous_input.is_file():
            raise FileNotFoundError("latest daily calibration receipt lacks immutable input snapshot")
    elif any(root.iterdir()):
        # The root may contain only its just-created manifest.  Any historical
        # day directory without a latest pointer is ambiguous and is rejected.
        if any(child.name.startswith("day=") for child in root.iterdir()):
            raise ValueError("daily calibration root has receipts but no latest pointer")

    target_dir = _day_directory(root, day)
    if target_dir.exists():
        raise FileExistsError(f"daily calibration receipt already exists: {target_dir}")
    ledger = _normalise_ledger(args.policy_ledger.resolve())
    if latest is not None:
        _assert_snapshot_unchanged(pd.read_parquet(previous_input), ledger)
    eligible = _eligible_snapshot(ledger, day=day)
    if len(eligible) < int(args.minimum_resolved_rows):
        raise RuntimeError(
            f"daily calibration has {len(eligible)} resolved rows; requires {args.minimum_resolved_rows}"
        )

    # Use the original mapper's exact score-band/trim implementation rather
    # than independently reimplementing a numerically delicate calculation.
    c0_state = builder._shift(c0_package, ledger, held=day, held_end=day + pd.Timedelta(days=1))
    c1_state = builder._shift(c1_package, ledger, held=day, held_end=day + pd.Timedelta(days=1))
    for family, state in (("c0_base_geometry", c0_state), ("c1_lva_geometry", c1_state)):
        if len(state) != 1 or str(state.iloc[0]["family"]) != family:
            raise AssertionError(f"{family} daily state has unexpected shape")
        row = state.iloc[0]
        if not np.isfinite(float(row["recent_shift_bps"])):
            raise RuntimeError(f"{family} daily shift is non-finite")
        if int(row["resolved_rows"]) < int(args.minimum_resolved_rows):
            raise RuntimeError(f"{family} daily state has insufficient resolved support")
        maximum = pd.to_datetime(row["max_policy_label_available_ts"], utc=True, errors="coerce")
        if pd.isna(maximum) or maximum >= day:
            raise AssertionError(f"{family} daily state violates label availability causality")

    staging = root / f".staging-{day:%Y%m%d}-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    c0_path, c1_path, input_path = (
        staging / "c0_prior21d_shift_state.parquet",
        staging / "c1_prior21d_shift_state.parquet",
        staging / "eligible_policy_input.parquet",
    )
    c0_state.to_parquet(c0_path, index=False, compression="zstd")
    c1_state.to_parquet(c1_path, index=False, compression="zstd")
    eligible.to_parquet(input_path, index=False, compression="zstd")
    source_manifest_sha = None
    if args.source_manifest is not None:
        source_manifest = args.source_manifest.resolve()
        if not source_manifest.is_file():
            raise FileNotFoundError(f"source manifest is absent: {source_manifest}")
        source_manifest_sha = _sha256(source_manifest)
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "PASS_CAUSAL_APPEND_ONLY_DAILY_CALIBRATION",
        "decision_day": day.isoformat(),
        "state_manifest_sha256": _sha256(manifest_path),
        "source": {
            "policy_ledger": str(args.policy_ledger.resolve()),
            "policy_ledger_sha256": _sha256(args.policy_ledger.resolve()),
            "source_manifest": str(args.source_manifest.resolve()) if args.source_manifest else None,
            "source_manifest_sha256": source_manifest_sha,
            "eligible_rows": int(len(eligible)),
            "eligible_input_sha256": _sha256(input_path),
        },
        "c0": {"state_path": c0_path.name, "state_sha256": _sha256(c0_path), **c0_state.iloc[0].to_dict()},
        "c1": {"state_path": c1_path.name, "state_sha256": _sha256(c1_path), **c1_state.iloc[0].to_dict()},
        "append_only": {
            "bootstrap": latest is None,
            "previous_decision_day": None if latest is None else str(latest["decision_day"]),
            "previous_receipt": None if latest is None else str(latest["receipt_dir"]),
            "prior_consumed_inputs_verified_unchanged": latest is not None,
        },
        "causality": {
            "policy_window": f"[{(day - pd.Timedelta(days=WINDOW_DAYS)).isoformat()}, {day.isoformat()})",
            "label_cutoff": "policy_label_available_ts < decision_day",
            "future_outcomes_consumed": False,
            "model_refit": False,
            "package_mutated": False,
            "exchange_order_submission_called": False,
        },
    }
    receipt_path = staging / "run_manifest.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(staging, target_dir)

    latest_payload = {
        "schema": SCHEMA,
        "status": "ACTIVE_CURRENT_DAY_RECEIPT",
        "decision_day": day.isoformat(),
        "receipt_dir": target_dir.name,
        "receipt_manifest_sha256": _sha256(target_dir / "run_manifest.json"),
        "state_manifest_sha256": _sha256(manifest_path),
        "c0_state_sha256": _sha256(target_dir / "c0_prior21d_shift_state.parquet"),
        "c1_state_sha256": _sha256(target_dir / "c1_prior21d_shift_state.parquet"),
    }
    _write_json_atomic(latest_path, latest_payload)
    print(target_dir)


if __name__ == "__main__":
    main()
