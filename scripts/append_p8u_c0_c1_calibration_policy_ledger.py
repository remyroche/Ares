#!/usr/bin/env python3
"""Publish an immutable effective C0/C1 policy-ledger revision.

The source score population is target-free and append-only.  Exact rich-policy
labels arrive later, after their H12 path resolves.  Every revision therefore
contains the full effective ledger required by the daily 21-day calibration
publisher while preserving all earlier valid labels byte-for-byte.  A later
revision may add new score identities or upgrade an unresolved/invalid label;
it may never alter a valid economic outcome.

This is a local data-contract utility.  It has no feature, mapper, portfolio,
network, account, or exchange-order authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid
from typing import Any

import numpy as np
import pandas as pd


SCHEMA = "p8u-c0-c1-calibration-policy-ledger-v1"
FIELDS = (
    "candidate_id", "__decision_ts__", "__symbol__", "side_name", "base_rank_ts",
    "policy_path_valid", "policy_net_bps", "policy_label_available_ts",
)
CORE = FIELDS[:5]
POLICY = FIELDS[5:]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _normalise_ledger(frame: pd.DataFrame, *, origin: str) -> pd.DataFrame:
    missing = set(FIELDS).difference(frame.columns)
    if missing:
        raise KeyError(f"{origin} lacks {sorted(missing)}")
    out = frame.loc[:, list(FIELDS)].copy()
    out["candidate_id"] = out["candidate_id"].astype(str)
    out["__symbol__"] = out["__symbol__"].astype(str)
    out["side_name"] = out["side_name"].astype(str).str.lower()
    out["__decision_ts__"] = pd.to_datetime(out["__decision_ts__"], utc=True, errors="raise")
    out["policy_label_available_ts"] = pd.to_datetime(
        out["policy_label_available_ts"], utc=True, errors="coerce"
    )
    out["base_rank_ts"] = pd.to_numeric(out["base_rank_ts"], errors="coerce")
    out["policy_net_bps"] = pd.to_numeric(out["policy_net_bps"], errors="coerce")
    out["policy_path_valid"] = out["policy_path_valid"].astype("boolean")
    if out["candidate_id"].duplicated().any() or not out["side_name"].eq("long").all():
        raise ValueError(f"{origin} violates target-free long-only candidate identity")
    if not np.isfinite(out["base_rank_ts"].to_numpy(float)).all():
        raise ValueError(f"{origin} has non-finite Base rank coordinates")
    valid = out["policy_path_valid"].fillna(False).astype(bool)
    if valid.any():
        net = out.loc[valid, "policy_net_bps"].to_numpy(float)
        when = out.loc[valid, "policy_label_available_ts"]
        if not np.isfinite(net).all() or when.isna().any():
            raise ValueError(f"{origin} has incomplete valid policy labels")
    return out.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def _router50_scores(path: Path) -> pd.DataFrame:
    raw = pd.read_parquet(path).copy()
    required = {*CORE, "base_rank_ts"}
    if missing := required.difference(raw.columns):
        raise KeyError(f"target-free upstream score lacks {sorted(missing)}")
    forbidden = [
        column for column in raw.columns
        if any(token in column.lower() for token in ("outcome", "policy_", "label", "net_bps", "gross_bps"))
    ]
    if forbidden:
        raise ValueError(f"upstream score carries outcome-like fields: {forbidden[:4]}")
    raw["__decision_ts__"] = pd.to_datetime(raw["__decision_ts__"], utc=True, errors="raise")
    raw["candidate_id"] = raw["candidate_id"].astype(str)
    if raw.duplicated(["candidate_id"]).any():
        raise ValueError("upstream score duplicates candidate identity")
    explicit_router50 = "router50_eligible" in raw.columns
    if explicit_router50:
        router50 = raw["router50_eligible"].fillna(False).astype(bool)
    else:
        router50 = np.isfinite(pd.to_numeric(raw["base_rank_ts"], errors="coerce"))
    routed = raw.loc[router50, list(CORE)].copy()
    if explicit_router50:
        full_counts = raw.groupby("__decision_ts__", sort=False)["candidate_id"].size()
        routed_counts = routed.groupby("__decision_ts__", sort=False)["candidate_id"].size()
        expected = np.ceil(full_counts.reindex(routed_counts.index).to_numpy(float) * .50).astype(int)
        if not np.array_equal(routed_counts.to_numpy(int), expected):
            raise AssertionError("upstream score is not exact timestamp-local Router50")
    routed["policy_path_valid"] = pd.Series(pd.NA, index=routed.index, dtype="boolean")
    routed["policy_net_bps"] = np.nan
    routed["policy_label_available_ts"] = pd.NaT
    return _normalise_ledger(routed, origin=str(path))


def _exact_outcomes(path: Path, *, available_before: pd.Timestamp | None) -> pd.DataFrame:
    raw = pd.read_parquet(path).copy()
    required = {"candidate_id", "decision_timestamp", "outcome_available", "net_bps", "gross_bps"}
    if missing := required.difference(raw.columns):
        raise KeyError(f"exact rich-policy outcomes lack {sorted(missing)}")
    out = raw.loc[:, ["candidate_id", "decision_timestamp", "outcome_available", "net_bps", "gross_bps"]].copy()
    out["candidate_id"] = out["candidate_id"].astype(str)
    out["__decision_ts__"] = pd.to_datetime(out.pop("decision_timestamp"), utc=True, errors="raise")
    out["policy_path_valid"] = out.pop("outcome_available").fillna(False).astype(bool)
    out["policy_net_bps"] = pd.to_numeric(out.pop("net_bps"), errors="coerce")
    gross = pd.to_numeric(out.pop("gross_bps"), errors="coerce")
    out["policy_label_available_ts"] = out["__decision_ts__"] + pd.Timedelta(hours=12, minutes=5)
    if out["candidate_id"].duplicated().any():
        raise ValueError("exact rich-policy outcome duplicates candidate identity")
    valid = out["policy_path_valid"]
    if valid.any():
        if not np.isfinite(out.loc[valid, "policy_net_bps"].to_numpy(float)).all():
            raise ValueError("valid exact rich-policy outcome has non-finite net bps")
        if not np.allclose(
            gross.loc[valid].to_numpy(float) - out.loc[valid, "policy_net_bps"].to_numpy(float),
            100.0, rtol=0.0, atol=1e-8,
        ):
            raise ValueError("exact rich-policy cost is not applied exactly once")
    if available_before is not None and not out["policy_label_available_ts"].lt(available_before).all():
        raise ValueError("exact label overlay contains a label not resolved before the declared cutoff")
    return out.loc[:, ["candidate_id", "__decision_ts__", *POLICY]].sort_values(
        ["__decision_ts__", "candidate_id"], kind="stable"
    ).reset_index(drop=True)


def _same_numeric(left: pd.Series, right: pd.Series) -> bool:
    return np.array_equal(
        pd.to_numeric(left, errors="coerce").to_numpy(float),
        pd.to_numeric(right, errors="coerce").to_numpy(float), equal_nan=True,
    )


def _merge_revision(
    previous: pd.DataFrame,
    scores: pd.DataFrame | None,
    outcomes: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    work = previous.set_index("candidate_id", drop=False).copy()
    audit = {"new_score_rows": 0, "existing_score_rows_verified": 0, "label_rows_applied": 0, "valid_labels_preserved": 0}
    if scores is not None:
        for row in scores.to_dict(orient="records"):
            candidate_id = str(row["candidate_id"])
            if candidate_id not in work.index:
                work.loc[candidate_id, list(FIELDS)] = [row[field] for field in FIELDS]
                audit["new_score_rows"] += 1
                continue
            existing = work.loc[candidate_id]
            for field in ("__decision_ts__", "__symbol__", "side_name"):
                if existing[field] != row[field]:
                    raise ValueError(f"upstream score rewrites existing {field}")
            if not np.isclose(float(existing["base_rank_ts"]), float(row["base_rank_ts"]), rtol=0.0, atol=1e-12):
                raise ValueError("upstream score rewrites existing Base rank")
            audit["existing_score_rows_verified"] += 1
    if outcomes is not None:
        for row in outcomes.to_dict(orient="records"):
            candidate_id = str(row["candidate_id"])
            if candidate_id not in work.index:
                raise ValueError("exact policy outcome has no prior target-free Router50 score")
            existing = work.loc[candidate_id]
            if existing["__decision_ts__"] != row["__decision_ts__"]:
                raise ValueError("exact policy outcome rewrites candidate decision timestamp")
            prior_valid = bool(existing["policy_path_valid"]) if pd.notna(existing["policy_path_valid"]) else False
            current_valid = bool(row["policy_path_valid"])
            if prior_valid:
                if not current_valid:
                    raise ValueError("append-only ledger would downgrade a valid exact outcome")
                if not np.isclose(float(existing["policy_net_bps"]), float(row["policy_net_bps"]), rtol=0.0, atol=1e-8):
                    raise ValueError("append-only ledger rewrites a valid exact net outcome")
                if existing["policy_label_available_ts"] != row["policy_label_available_ts"]:
                    raise ValueError("append-only ledger rewrites valid label availability")
                audit["valid_labels_preserved"] += 1
                continue
            work.loc[candidate_id, list(POLICY)] = [
                bool(row["policy_path_valid"]), float(row["policy_net_bps"]) if current_valid else np.nan,
                row["policy_label_available_ts"],
            ]
            audit["label_rows_applied"] += 1
    merged = _normalise_ledger(work.reset_index(drop=True), origin="merged append-only effective ledger")
    return merged, audit


def _read_latest(root: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    latest_path = root / "latest.json"
    manifest_path = root / "ledger_manifest.json"
    if not latest_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("append-only calibration ledger root is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or latest.get("schema") != SCHEMA:
        raise ValueError("unknown append-only calibration ledger schema")
    if latest.get("ledger_manifest_sha256") != _sha256(manifest_path):
        raise ValueError("append-only calibration ledger manifest hash mismatch")
    relative = Path(str(latest.get("ledger_path") or ""))
    ledger_path = (root / relative).resolve()
    if root not in ledger_path.parents or not ledger_path.is_file():
        raise ValueError("append-only calibration ledger latest path escapes root")
    if latest.get("ledger_sha256") != _sha256(ledger_path):
        raise ValueError("append-only calibration ledger latest file hash mismatch")
    return manifest, _normalise_ledger(pd.read_parquet(ledger_path), origin=str(ledger_path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-root", type=Path, required=True)
    parser.add_argument("--revision", required=True, help="immutable logical revision label, e.g. 2026-09-03T0000Z")
    parser.add_argument("--bootstrap-ledger", type=Path)
    parser.add_argument("--upstream-scores", type=Path)
    parser.add_argument("--exact-outcomes", type=Path)
    parser.add_argument("--label-available-before", help="strict cutoff for any supplied exact outcomes")
    parser.add_argument("--bootstrap", action="store_true")
    args = parser.parse_args()
    root = args.ledger_root.resolve()
    revision = str(args.revision).strip()
    if not revision or "/" in revision or ".." in revision:
        raise ValueError("revision must be a simple immutable label")
    cutoff = _utc(args.label_available_before) if args.label_available_before else None
    if args.bootstrap:
        if args.bootstrap_ledger is None or root.exists():
            raise ValueError("bootstrap requires a new root and --bootstrap-ledger")
        previous = _normalise_ledger(pd.read_parquet(args.bootstrap_ledger), origin=str(args.bootstrap_ledger))
        manifest: dict[str, Any] = {
            "schema": SCHEMA,
            "status": "ACTIVE_APPEND_ONLY_EFFECTIVE_LEDGER",
            "base_ledger": {"path": str(args.bootstrap_ledger.resolve()), "sha256": _sha256(args.bootstrap_ledger.resolve())},
            "revisions": [],
            "scope": "target-free Router50 score coordinates plus delayed exact rich-policy labels; immutable full-ledger revisions",
        }
        root.mkdir(parents=True, exist_ok=False)
    else:
        if args.bootstrap_ledger is not None:
            raise ValueError("--bootstrap-ledger is valid only with --bootstrap")
        manifest, previous = _read_latest(root)
        revisions = list(manifest.get("revisions") or ())
        if not revisions or not isinstance(revisions[-1], dict):
            raise ValueError("append-only calibration ledger has no ordered prior revision")
        previous_revision = str(revisions[-1].get("revision") or "")
        if revision <= previous_revision:
            raise ValueError("append-only calibration ledger revision must advance monotonically")
    if args.upstream_scores is None and args.exact_outcomes is None and not args.bootstrap:
        raise ValueError("an append revision needs target-free scores and/or exact outcomes")
    scores = _router50_scores(args.upstream_scores.resolve()) if args.upstream_scores else None
    outcomes = _exact_outcomes(args.exact_outcomes.resolve(), available_before=cutoff) if args.exact_outcomes else None
    merged, audit = _merge_revision(previous, scores, outcomes)
    revision_dir = root / "revisions" / f"revision={revision}"
    if revision_dir.exists():
        raise FileExistsError("effective ledger revision is immutable")
    stage = root / f".staging-{uuid.uuid4().hex}"
    stage.mkdir(parents=False)
    try:
        ledger_path = stage / "effective_policy_ledger.parquet"
        merged.to_parquet(ledger_path, index=False, compression="zstd")
        receipt = {
            "schema": SCHEMA,
            "status": "PASS_APPEND_ONLY_EFFECTIVE_POLICY_LEDGER",
            "revision": revision,
            "prior_rows": int(len(previous)), "rows": int(len(merged)),
            "audit": audit,
            "label_available_before": None if cutoff is None else cutoff.isoformat(),
            "inputs": {
                "upstream_scores": None if args.upstream_scores is None else {"path": str(args.upstream_scores.resolve()), "sha256": _sha256(args.upstream_scores.resolve())},
                "exact_outcomes": None if args.exact_outcomes is None else {"path": str(args.exact_outcomes.resolve()), "sha256": _sha256(args.exact_outcomes.resolve())},
            },
            "causality": {
                "scores": "target-free full Router50 only",
                "labels": "exact one-minute rich policy after H12 resolution only",
                "valid_labels": "preserved byte-for-byte across revisions",
                "authority": "no model, map, portfolio, exchange, account, or order authority",
            },
            "output": {"effective_policy_ledger.parquet": _sha256(ledger_path)},
        }
        (stage / "run_manifest.json").write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        revision_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage, revision_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    relative_ledger = str((revision_dir / "effective_policy_ledger.parquet").relative_to(root))
    manifest["revisions"].append({
        "revision": revision,
        "receipt": str((revision_dir / "run_manifest.json").relative_to(root)),
        "receipt_sha256": _sha256(revision_dir / "run_manifest.json"),
        "ledger": relative_ledger,
        "ledger_sha256": _sha256(revision_dir / "effective_policy_ledger.parquet"),
    })
    _write_json_atomic(root / "ledger_manifest.json", manifest)
    _write_json_atomic(root / "latest.json", {
        "schema": SCHEMA, "status": "ACTIVE_CURRENT_EFFECTIVE_LEDGER", "revision": revision,
        "ledger_path": relative_ledger,
        "ledger_sha256": _sha256(revision_dir / "effective_policy_ledger.parquet"),
        "ledger_manifest_sha256": _sha256(root / "ledger_manifest.json"),
    })
    print(revision_dir)


if __name__ == "__main__":
    main()
