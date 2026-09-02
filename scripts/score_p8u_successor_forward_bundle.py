#!/usr/bin/env python3
"""Score a sealed P8U Router50/Base/Under bundle on target-free features.

This utility deliberately ends at upstream scores.  It is reusable for a
future C0/C1 package but has no label, mapper, portfolio, network, account, or
exchange dependency.  The bundle loader verifies every model/state hash before
scoring and Base/Under receive only exact Router50 identities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.inference.p8u_model_package import BASE_GEOMETRY, P8UModelBundle
from extreme_price_movements.inference.p8u_production_contract import IDENTITY_COLUMNS, sha256_file


IDENTITY = tuple(IDENTITY_COLUMNS)


def _sha(path: Path) -> str:
    return sha256_file(path)


def _identity_hash(frame: pd.DataFrame) -> str:
    work = frame.loc[:, list(IDENTITY)].copy().sort_values(["__decision_ts__", "candidate_id"], kind="stable")
    digest = hashlib.sha256()
    for row in work.itertuples(index=False, name=None):
        digest.update("|".join(map(str, row)).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _features(root: Path, fields: tuple[str, ...], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    manifest = root / "run_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    parts = sorted((root / "features").glob("part_*.parquet"))
    if not parts:
        raise FileNotFoundError(f"no immutable feature parts under {root}")
    required = [*IDENTITY, "__ts__", "__symbol__", *fields]
    pieces: list[pd.DataFrame] = []
    for part in parts:
        frame = pd.read_parquet(part, columns=required)
        frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
        frame = frame.loc[frame["__decision_ts__"].ge(start) & frame["__decision_ts__"].lt(end)]
        if not frame.empty:
            pieces.append(frame)
    if not pieces:
        raise AssertionError("no target-free feature rows in requested range")
    full = pd.concat(pieces, ignore_index=True).sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    if full.duplicated(list(IDENTITY)).any():
        raise AssertionError("feature source duplicates candidate identity")
    if not full["side_name"].astype(str).str.lower().eq("long").all():
        raise AssertionError("successor forward score source must be long-only")
    source_ts = pd.to_datetime(full["__ts__"], utc=True, errors="raise")
    if not full["__decision_ts__"].eq(source_ts + pd.Timedelta(hours=1)).all():
        raise AssertionError("feature decision timestamp differs from completed source hour plus one")
    return full


def _incremental_feature_commit(
    *,
    panel: Path,
    commit_receipt: Path,
    fields: tuple[str, ...],
    decision: pd.Timestamp,
) -> pd.DataFrame:
    """Read one parity-verified incremental feature commit without rebuilding it."""
    if not panel.is_file() or not commit_receipt.is_file():
        raise FileNotFoundError("incremental feature panel or commit receipt is absent")
    receipt = json.loads(commit_receipt.read_text(encoding="utf-8"))
    if str(receipt.get("features")) != str(panel.resolve()):
        raise ValueError("incremental feature receipt points to another panel")
    if str(receipt.get("features_sha256")) != _sha(panel):
        raise ValueError("incremental feature receipt hash mismatch")
    parity = dict(receipt.get("parity") or {})
    if parity.get("status") != "pass" or int(parity.get("rows_outside_tolerance", -1)) != 0:
        raise ValueError("incremental feature commit lacks full-causal parity")
    required = [*IDENTITY, "__ts__", "__symbol__", *fields]
    frame = pd.read_parquet(panel, columns=required)
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    if not frame["__decision_ts__"].eq(decision).all():
        raise ValueError("incremental feature panel does not cover exactly the requested decision")
    source_ts = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    if not source_ts.eq(decision - pd.Timedelta(hours=1)).all():
        raise AssertionError("incremental feature source timestamp is not the preceding completed hour")
    if frame.duplicated(list(IDENTITY)).any() or not frame["side_name"].astype(str).str.lower().eq("long").all():
        raise AssertionError("incremental feature panel violates target-free long-only identity contract")
    if int(receipt.get("candidate_rows", -1)) != len(frame):
        raise AssertionError("incremental feature receipt candidate count mismatch")
    return frame.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def run(args: argparse.Namespace) -> Path:
    package_root = args.package.resolve()
    feature_root = args.feature_root.resolve() if args.feature_root is not None else None
    output = args.out.resolve()
    if output.exists():
        raise FileExistsError("output must be immutable")
    start = pd.Timestamp(args.start) if args.start else None
    end = pd.Timestamp(args.end) if args.end else None
    if start is not None:
        start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    if end is not None:
        end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    if (start is None) != (end is None):
        raise ValueError("--start and --end must be supplied together")
    if start is not None and end <= start:
        raise ValueError("end must follow start")
    bundle = P8UModelBundle.load(package_root, verify_hashes=True)
    # Under's Base-query geometry is deterministically generated inside
    # ``bundle.score_under`` after the exact Router50/Base handoff.  Those
    # fields are deliberately absent from the raw causal feature panel.
    fields = tuple(field for field in dict.fromkeys(
        field for role in ("router_model", "base_model", "under_model")
        for field in bundle.states[role].fields
    ) if field not in BASE_GEOMETRY)
    if args.feature_panel is not None:
        decision = pd.Timestamp(args.decision_ts)
        decision = decision.tz_localize("UTC") if decision.tzinfo is None else decision.tz_convert("UTC")
        full = _incremental_feature_commit(
            panel=args.feature_panel.resolve(), commit_receipt=args.feature_commit.resolve(),
            fields=fields, decision=decision,
        )
        source_descriptor = {
            "kind": "incremental_parity_verified_commit",
            "panel": str(args.feature_panel.resolve()), "panel_sha256": _sha(args.feature_panel.resolve()),
            "commit_receipt": str(args.feature_commit.resolve()), "commit_receipt_sha256": _sha(args.feature_commit.resolve()),
            "decision": decision.isoformat(),
        }
        start, end = decision, decision + pd.Timedelta(nanoseconds=1)
    else:
        if feature_root is None:
            raise AssertionError("immutable feature-root mode requires --feature-root")
        full = _features(feature_root, fields, start, end)
        source_descriptor = {"kind": "immutable_feature_root", "path": str(feature_root), "manifest_sha256": _sha(feature_root / "run_manifest.json")}
    router = bundle.score_router(full)
    routed = bundle.route_router50(full, router)
    base = bundle.score_base(routed)
    under = bundle.score_under(routed, base)
    # ``__symbol__`` is not a model identity field, but it is immutable
    # candidate provenance needed by the causal C1 snapshot materialiser.
    all_scores = full.loc[:, [*IDENTITY, "__symbol__"]].merge(
        routed.loc[:, list(IDENTITY) + ["router_primary_rank", "router_raw_score", "router50_eligible"]],
        on=list(IDENTITY), how="left", validate="one_to_one",
    ).merge(
        base, on=list(IDENTITY), how="left", validate="one_to_one",
    ).merge(
        under, on=list(IDENTITY), how="left", validate="one_to_one",
    )
    if all_scores["router50_eligible"].isna().any():
        raise AssertionError("Router output does not cover source identities")
    routed_mask = all_scores["router50_eligible"].fillna(False).astype(bool)
    if all_scores.loc[routed_mask, ["base_score", "base_rank_ts", "under_raw_score", "under_rank_ts"]].isna().any().any():
        raise AssertionError("Base/Under did not cover exactly Router50")
    if all_scores.loc[~routed_mask, ["base_score", "base_rank_ts", "under_raw_score", "under_rank_ts"]].notna().any().any():
        raise AssertionError("Base/Under escaped the Router50 identity gate")
    direct_weight = float(bundle.manifest.get("under_direct_score_authority_weight", np.nan))
    if direct_weight != 0.0:
        raise AssertionError("forward scorer requires the frozen Base-authoritative / Under-telemetry package")
    all_scores["current_score"] = all_scores["base_rank_ts"]
    temp = output.with_name(f".{output.name}.build-{os.getpid()}")
    temp.mkdir(parents=True)
    try:
        path = temp / "target_free_upstream_scores.parquet"
        all_scores.to_parquet(path, index=False, compression="zstd")
        per_timestamp = all_scores.groupby("__decision_ts__", sort=True).agg(
            full_candidates=("candidate_id", "size"), router50=("router50_eligible", "sum"),
        ).reset_index()
        if not per_timestamp["router50"].eq(np.ceil(per_timestamp["full_candidates"] * .50)).all():
            raise AssertionError("Router50 count deviates from exact timestamp-local ceil rule")
        per_timestamp.to_parquet(temp / "router50_coverage.parquet", index=False, compression="zstd")
        (temp / "run_manifest.json").write_text(json.dumps({
            "schema": "p8u_successor_forward_upstream_scores_v1", "status": "complete",
            "scope": "target-free upstream scoring only; no labels, MC1, admission, portfolio, network, account, or exchange authority",
            "window": {"start": start.isoformat(), "end_exclusive": end.isoformat()},
            "package": {"path": str(package_root), "manifest_sha256": _sha(package_root / "manifest.json")},
            "feature_panel": source_descriptor,
            "source_rows": int(len(full)), "source_identity_sha256": _identity_hash(full),
            "router50_rows": int(routed_mask.sum()), "router_fraction": .50,
            "under_direct_score_authority_weight": direct_weight,
            "outputs": {
                "target_free_upstream_scores": _sha(path),
                "router50_coverage": _sha(temp / "router50_coverage.parquet"),
            },
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temp, output)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--feature-root", type=Path)
    source.add_argument("--feature-panel", type=Path)
    parser.add_argument("--feature-commit", type=Path)
    parser.add_argument("--decision-ts")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--out", type=Path, required=True)
    parsed = parser.parse_args()
    if parsed.feature_panel is not None and (parsed.feature_commit is None or parsed.decision_ts is None):
        parser.error("--feature-panel requires --feature-commit and --decision-ts")
    if parsed.feature_root is not None and (parsed.start is None or parsed.end is None):
        parser.error("--feature-root requires --start and --end")
    print(run(parsed))


if __name__ == "__main__":
    main()
