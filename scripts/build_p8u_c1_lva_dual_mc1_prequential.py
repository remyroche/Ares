#!/usr/bin/env python3
"""Fit and seal C1-LVA augmented dual-MC1 mapper packages.

This is a no-order producer.  It accepts existing target-free BCF/current
score panels, joins only causal C1 snapshot outputs by candidate identity and
decision timestamp, then joins policy labels *after* target-free score rows
have been constructed.  Each held month is fitted on exactly six preceding
complete calendar months whose policy labels had resolved before that month.

The emitted target-free predictions, policy-attached replay panel, serialized
packages, and daily prior-21-day shift states are all persisted so historical
OOS replay and later inference use the same numeric contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.inference.p8u_c1_mc1_inference_package import (
    FEATURES,
    build_shift_state,
    fit_package,
    save_package,
)
from extreme_price_movements.inference.p8u_c1_mc1_selector import (
    CONFIG_SCHEMA,
    RUN_SCHEMA,
    P8UC1MC1PackageSelector,
)


CORE = FEATURES[:6]
SR_FEATURES = FEATURES[6:-1]
SR_AVAILABLE = FEATURES[-1]
POLICY_COLUMNS = (
    "policy_path_valid", "policy_gross_bps", "policy_net_bps",
    "policy_exit_bar_15m", "policy_entry_price", "policy_exit_price",
    "policy_exit_reason", "policy_label_available_ts", "policy_outcome_source",
    "policy_cost_bps",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(path: Path) -> str:
    return P8UC1MC1PackageSelector._tree_sha256(path)


def _utc(value: object) -> pd.Timestamp:
    value = pd.Timestamp(value)
    return value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")


def _months(start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Timestamp, ...]:
    start, end = _utc(start).normalize().replace(day=1), _utc(end).normalize().replace(day=1)
    if end <= start:
        raise ValueError("held end must be after held start")
    return tuple(pd.date_range(start, end - pd.offsets.MonthBegin(1), freq="MS", tz="UTC"))


def _read_score(path: Path, family: str) -> pd.DataFrame:
    required = {
        "candidate_id", "__decision_ts__", "__symbol__", "side_name", *CORE, *POLICY_COLUMNS,
    }
    names = set(pq.ParquetFile(path).schema_arrow.names)
    missing = required.difference(names)
    if missing:
        raise ValueError(f"{family} score source lacks {sorted(missing)}")
    frame = pd.read_parquet(path, columns=sorted(required)).copy()
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    frame["policy_label_available_ts"] = pd.to_datetime(
        frame["policy_label_available_ts"], utc=True, errors="raise"
    )
    if frame["candidate_id"].duplicated().any():
        raise ValueError(f"{family} score source duplicates candidate identity")
    if not frame["side_name"].astype(str).str.lower().eq("long").all():
        raise ValueError(f"{family} score source is not long-only")
    return frame


def _read_snapshots(path: Path) -> pd.DataFrame:
    required = {"candidate_id", "snapshot_ts", *SR_FEATURES}
    names = set(pq.ParquetFile(path).schema_arrow.names)
    missing = required.difference(names)
    if missing:
        raise ValueError(f"C1 snapshot source lacks {sorted(missing)}")
    snapshots = pd.read_parquet(path, columns=sorted(required)).copy()
    snapshots["candidate_id"] = snapshots["candidate_id"].astype(str)
    snapshots["snapshot_ts"] = pd.to_datetime(snapshots["snapshot_ts"], utc=True, errors="raise")
    if snapshots.duplicated(["candidate_id", "snapshot_ts"]).any():
        raise ValueError("C1 snapshot source duplicates candidate-time identity")
    return snapshots


def _augment(
    score: pd.DataFrame, snapshots: pd.DataFrame, *, core_only: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return separate target-free and label views without membership changes."""
    target_free = score.loc[:, ["candidate_id", "__decision_ts__", "__symbol__", "side_name", *CORE]].copy()
    if core_only:
        # Matched ablation control: preserve the augmented package geometry,
        # sampling, target, and shift state, while withholding all C1 state
        # information.  Constants plus explicit unavailable status make C1
        # semantically absent rather than turning missingness into a filter.
        for field in SR_FEATURES:
            target_free[field] = 0.0
        target_free[SR_AVAILABLE] = np.int8(0)
    else:
        target_free["snapshot_ts"] = target_free["__decision_ts__"]
        target_free = target_free.merge(snapshots, on=["candidate_id", "snapshot_ts"], how="left", validate="one_to_one")
        target_free[SR_AVAILABLE] = target_free.loc[:, list(SR_FEATURES)].notna().any(axis=1).astype(np.int8)
    if len(target_free) != len(score) or target_free["candidate_id"].duplicated().any():
        raise AssertionError("C1 snapshot merge changed target-free identity")
    labels = score.loc[:, ["candidate_id", *POLICY_COLUMNS]].copy()
    target_free = target_free.drop(columns="snapshot_ts", errors="ignore")
    return target_free, labels


def _strict_train(full: pd.DataFrame, *, start: pd.Timestamp, held: pd.Timestamp) -> pd.DataFrame:
    valid = (
        full["__decision_ts__"].ge(start)
        & full["__decision_ts__"].lt(held)
        & full["policy_path_valid"].fillna(False).astype(bool)
        & full["policy_label_available_ts"].lt(held)
        & np.isfinite(pd.to_numeric(full["policy_net_bps"], errors="coerce"))
    )
    train = full.loc[valid].copy()
    if not train.empty and not train["policy_label_available_ts"].lt(held).all():
        raise AssertionError("C1 MC1 training includes a non-pre-resolved policy label")
    return train


def _policy_contract(labels: pd.DataFrame, policy_source: Path) -> dict[str, object]:
    costs = sorted(pd.to_numeric(labels["policy_cost_bps"], errors="coerce").dropna().unique().tolist())
    return {
        "source": str(policy_source), "source_sha256": _sha256(policy_source),
        "target": "source-aligned policy_net_bps", "cost_application": "embedded exactly once",
        "observed_policy_cost_bps": costs,
    }


def _fit_family(
    full: pd.DataFrame, *, family: str, held_months: Sequence[pd.Timestamp],
    source_hashes: dict[str, str], policy_contract: dict[str, object], package_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    rows: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    packages: list[dict[str, object]] = []
    for held in held_months:
        held_end = held + pd.offsets.MonthBegin(1)
        train_start = held - pd.DateOffset(months=6)
        train = _strict_train(full, start=train_start, held=held)
        held_scores = full.loc[full["__decision_ts__"].ge(held) & full["__decision_ts__"].lt(held_end)].copy()
        if len(train) < 5_000 or held_scores.empty:
            raise RuntimeError(f"{family}/{held:%Y-%m}: insufficient strict-prequential support")
        package = fit_package(
            train, family=family, train_start=train_start, train_end_exclusive=held,
            held_start=held, held_end_exclusive=held_end, train_months=6,
            source_hashes=source_hashes, policy_contract=policy_contract,
        )
        state = build_shift_state(package, full, held_start=held, held_end_exclusive=held_end)
        if not (pd.to_datetime(state["max_policy_label_available_ts"], utc=True, errors="coerce").lt(
            pd.to_datetime(state["decision_day"], utc=True, errors="raise")
        ) | state["max_policy_label_available_ts"].isna()).all():
            raise AssertionError(f"{family}/{held:%Y-%m}: non-causal shift state")
        static = package.predict_static(held_scores)
        held_scores["static_expected_bps"] = static
        held_scores["score_band_curve_bps"] = package.curve_for(held_scores)
        lookup = state.set_index("decision_day")["recent_shift_bps"]
        held_scores["recent_shift_bps"] = held_scores["__decision_ts__"].dt.normalize().map(lookup).fillna(0.0).to_numpy(float)
        held_scores["mc1_expected_bps"] = held_scores["static_expected_bps"] + held_scores["recent_shift_bps"]
        held_scores["mc1_family"] = family
        target_free = held_scores.loc[:, [
            "candidate_id", "__decision_ts__", "__symbol__", "side_name", *CORE, *SR_FEATURES, SR_AVAILABLE,
            "static_expected_bps", "score_band_curve_bps", "recent_shift_bps", "mc1_expected_bps", "mc1_family",
        ]].copy()
        if set(POLICY_COLUMNS).intersection(target_free.columns):
            raise AssertionError("target-free C1 MC1 output leaked a policy outcome")
        rows.append(target_free)
        folder = package_root / family / f"{held:%Y-%m}"
        save_package(package, state, folder)
        packages.append({
            "family": family, "month": f"{held:%Y-%m}",
            "path": str(folder.relative_to(package_root)), "sha256": _tree_sha256(folder),
        })
        audits.append({
            "family": family, "held_month": f"{held:%Y-%m}",
            "train_start": train_start.isoformat(), "train_rows": len(train),
            "held_rows": len(held_scores), "sampled_rows": package.sampled_rows,
            "c1_snapshot_available_train": float(train[SR_AVAILABLE].mean()),
            "c1_snapshot_available_held": float(held_scores[SR_AVAILABLE].mean()),
        })
    return pd.concat(rows, ignore_index=True), pd.DataFrame(audits), packages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bcf", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--c1-snapshots", type=Path, required=True)
    parser.add_argument("--held-start", required=True, help="UTC month boundary inclusive")
    parser.add_argument("--held-end", required=True, help="UTC month boundary exclusive")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--core-only", action="store_true",
        help="matched C1-package control: hold every C1 input at neutral unavailable values",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"output must be immutable: {output}")
    held_months = _months(_utc(args.held_start), _utc(args.held_end))
    bcf_source, current_source, snapshot_source = args.bcf.resolve(), args.current.resolve(), args.c1_snapshots.resolve()
    bcf_score, current_score, snapshots = _read_score(bcf_source, "bcf"), _read_score(current_source, "current"), _read_snapshots(snapshot_source)
    bcf_tf, bcf_labels = _augment(bcf_score, snapshots, core_only=bool(args.core_only))
    current_tf, current_labels = _augment(current_score, snapshots, core_only=bool(args.core_only))
    # Policy labels must agree on common identities; they are read only after
    # both target-free source views have been constructed.
    common = bcf_labels.merge(current_labels, on="candidate_id", how="inner", suffixes=("_b", "_c"), validate="one_to_one")
    for field in POLICY_COLUMNS:
        left, right = common[f"{field}_b"], common[f"{field}_c"]
        if pd.api.types.is_numeric_dtype(left):
            equal = np.isclose(pd.to_numeric(left, errors="coerce"), pd.to_numeric(right, errors="coerce"), equal_nan=True).all()
        else:
            equal = left.fillna("__missing__").astype(str).equals(right.fillna("__missing__").astype(str))
        if not equal:
            raise AssertionError(f"BCF/current policy label mismatch: {field}")
    bcf_full = bcf_tf.merge(bcf_labels, on="candidate_id", how="left", validate="one_to_one")
    current_full = current_tf.merge(current_labels, on="candidate_id", how="left", validate="one_to_one")
    output.mkdir(parents=True, exist_ok=False)
    package_root = output / "packages"
    hashes = {
        "bcf_scores": _sha256(bcf_source), "current_scores": _sha256(current_source),
        "c1_snapshots": _sha256(snapshot_source),
        "package_runtime": _sha256(ROOT / "extreme_price_movements/inference/p8u_c1_mc1_inference_package.py"),
    }
    bcf_pred, bcf_audit, bcf_packages = _fit_family(
        bcf_full, family="bcf", held_months=held_months, source_hashes=hashes,
        policy_contract=_policy_contract(bcf_labels, bcf_source), package_root=package_root,
    )
    current_pred, current_audit, current_packages = _fit_family(
        current_full, family="current", held_months=held_months, source_hashes=hashes,
        policy_contract=_policy_contract(current_labels, current_source), package_root=package_root,
    )
    for family, frame in (("bcf", bcf_pred), ("current", current_pred)):
        frame.to_parquet(output / f"predictions_{family}_target_free.parquet", index=False, compression="zstd")
    paired = bcf_pred.merge(
        current_pred.loc[:, ["candidate_id", "mc1_expected_bps"]].rename(columns={"mc1_expected_bps": "current_mc1_expected_bps"}),
        on="candidate_id", how="inner", validate="one_to_one",
    ).rename(columns={"mc1_expected_bps": "bcf_mc1_expected_bps"})
    paired["dual_mc1_admitted"] = paired["bcf_mc1_expected_bps"].ge(50.0) & paired["current_mc1_expected_bps"].ge(50.0)
    paired["auction_priority_bps"] = paired["bcf_mc1_expected_bps"]
    paired.to_parquet(output / "dual_target_free_predictions.parquet", index=False, compression="zstd")
    outcome = paired.merge(bcf_labels, on="candidate_id", how="left", validate="one_to_one")
    outcome.to_parquet(output / "dual_outcome_replay_panel.parquet", index=False, compression="zstd")
    audits = pd.concat([bcf_audit, current_audit], ignore_index=True)
    audits.to_parquet(output / "fold_audit.parquet", index=False, compression="zstd")
    index = {"feature_order": list(FEATURES), "train_months": 6, "families": [*bcf_packages, *current_packages]}
    (package_root / "mc1_package_index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema": RUN_SCHEMA,
        "scope": "no-order C1-LVA dual-MC1 package/refit and historical OOS producer",
        "ablation": "core_only_neutral_c1" if args.core_only else "full_c1_state",
        "held_months": [f"{month:%Y-%m}" for month in held_months],
        "feature_order": list(FEATURES), "source_hashes": hashes,
        "causality": {
            "scores": "target-free source score panels are joined to C1 snapshots before policy labels",
            "c1": (
                "all C1 outputs set to neutral unavailable constants for matched core-only control"
                if args.core_only else
                "C1 outputs are causal candidate-time OOF snapshots; missingness is a feature, never a filter"
            ),
            "labels": "each fit uses only labels resolved strictly before held month",
            "shift": "each daily shift uses only prior-resolved 21-day policy residuals",
            "authority": "no exchange, portfolio, or order authority",
        },
    }
    manifest_path = output / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        package_root_for_config = str(package_root.relative_to(ROOT))
    except ValueError:
        # Isolated test/research outputs may live outside the repository.  The
        # selector will deliberately reject such a path for a production
        # bundle; preserving it here still makes the no-order receipt
        # self-describing without inventing a repository-relative path.
        package_root_for_config = str(package_root)
    config = {
        "schema": CONFIG_SCHEMA, "status": "SEALED_NO_ORDER_C1_LVA_MAPPER",
        "package_root": package_root_for_config, "feature_contract": list(FEATURES),
        "training": {"train_months": 6}, "admission": {"order_submission": False},
        "run_manifest_sha256": _sha256(manifest_path),
    }
    (output / "c1_mc1_selector_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
