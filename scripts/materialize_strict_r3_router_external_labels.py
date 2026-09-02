#!/usr/bin/env python3
"""Build target-only external-label sidecars for router consensus research.

This is deliberately a bridge between *already strict-OOF* supportive-label
models and the P3 router consensus study.  It does not expose any realised
path coordinate to a scorer.  Instead it retains the causal expected-policy
bps projections emitted by prior-trained frozen-path and causal-joint models.

The resulting sidecar is consumed only while fitting a later consensus fold;
its candidate identity is checked against the router's target-free top-30%
population.  It is never a live feature or an MC1 input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "strict_r3_router_external_labels_v1"
MONTHS = tuple(pd.date_range("2025-11-01", "2026-07-01", freq="MS", tz="UTC"))
IDENTITY = ("candidate_id", "__decision_ts__", "side_name")
PATH_ARM = "S3_frozen_gmm_k8_shared_policy_residual"
CAUSAL_STATE_COLUMN = "C1_ward_k4_state_expected_ev"
CAUSAL_PATH_COLUMN = "C1_ward_k4_J2_soft_base_equal_causal120_plus_oof_stack"


def _sha256(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for item in sorted(paths):
        digest.update(str(item).encode())
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _month_source(source_root: Path, month: pd.Timestamp) -> pd.DataFrame:
    path = source_root / "target_free_monthly" / f"month={month:%Y-%m}" / "scores_features.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    names = set(pq.ParquetFile(path).schema_arrow.names)
    missing = set(IDENTITY) - names
    if missing:
        raise AssertionError(f"{path}: missing router identity {sorted(missing)}")
    result = pd.read_parquet(path, columns=list(IDENTITY)).copy()
    result["__decision_ts__"] = pd.to_datetime(result["__decision_ts__"], utc=True, errors="raise")
    if result["candidate_id"].duplicated().any():
        raise AssertionError(f"{path}: duplicate candidate IDs")
    return result


def _read_path(root: Path) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for folder in sorted((root / "oof_prediction_parts").glob("fold=*")):
        found = None
        for path in sorted(folder.glob("*.parquet")):
            probe = pd.read_parquet(path, columns=["arm"])
            if len(probe) and str(probe["arm"].iloc[0]) == PATH_ARM:
                found = pd.read_parquet(path, columns=["candidate_id", "__decision_ts__", "arm", "predicted_policy_net_bps"])
                break
        if found is None:
            raise FileNotFoundError(f"{folder}: {PATH_ARM} is absent")
        pieces.append(found)
    result = pd.concat(pieces, ignore_index=True).drop(columns="arm").rename(
        columns={"predicted_policy_net_bps": "frozen_path_expected_policy_bps"}
    )
    result["__decision_ts__"] = pd.to_datetime(result["__decision_ts__"], utc=True, errors="raise")
    if result["candidate_id"].duplicated().any():
        raise AssertionError("frozen path output has duplicate candidate IDs")
    return result


def _read_causal(root: Path) -> pd.DataFrame:
    columns = ["candidate_id", "__decision_ts__", CAUSAL_STATE_COLUMN, CAUSAL_PATH_COLUMN]
    pieces = [pd.read_parquet(path, columns=columns) for path in sorted((root / "causal_joint_oof_predictions").glob("fold=*.parquet"))]
    if not pieces:
        raise FileNotFoundError(f"no causal OOF panels under {root}")
    result = pd.concat(pieces, ignore_index=True).rename(columns={
        CAUSAL_STATE_COLUMN: "causal_regime_expected_policy_bps",
        CAUSAL_PATH_COLUMN: "causal_path_expected_policy_bps",
    })
    result["__decision_ts__"] = pd.to_datetime(result["__decision_ts__"], utc=True, errors="raise")
    if result["candidate_id"].duplicated().any():
        raise AssertionError("causal-joint output has duplicate candidate IDs")
    return result


def run(*, source_root: Path, path_root: Path, causal_root: Path, out: Path) -> Path:
    if out.exists():
        raise FileExistsError(out)
    source = pd.concat([_month_source(source_root, month) for month in MONTHS], ignore_index=True)
    if source["candidate_id"].duplicated().any():
        raise AssertionError("router source duplicates candidate IDs across months")
    path = _read_path(path_root)
    causal = _read_causal(causal_root)
    output = source.merge(path, on=["candidate_id", "__decision_ts__"], how="left", validate="one_to_one")
    output = output.merge(causal, on=["candidate_id", "__decision_ts__"], how="left", validate="one_to_one")
    values = ["frozen_path_expected_policy_bps", "causal_regime_expected_policy_bps", "causal_path_expected_policy_bps"]
    if output.loc[:, values].isna().any().any():
        missing = output.loc[output.loc[:, values].isna().any(axis=1), ["candidate_id", "__decision_ts__"]]
        raise AssertionError(f"external OOF labels do not cover the P3 router population: {len(missing)} rows")
    output["source_label_available_ts"] = output["__decision_ts__"]
    output["month"] = output["__decision_ts__"].dt.strftime("%Y-%m")
    out.mkdir(parents=True, exist_ok=False)
    result_path = out / "router_external_labels.parquet"
    output.to_parquet(result_path, index=False, compression="zstd")
    by_month = output.groupby("month", sort=True).size().rename("rows").reset_index()
    by_month.to_parquet(out / "coverage_by_month.parquet", index=False, compression="zstd")
    manifest = {
        "schema": SCHEMA,
        "scope": "target-only offline consensus supervision; never an inference feature or MC1 input",
        "identity": list(IDENTITY),
        "router_source": str(source_root.resolve()),
        "frozen_path_source": str(path_root.resolve()),
        "causal_joint_source": str(causal_root.resolve()),
        "months": [f"{month:%Y-%m}" for month in MONTHS],
        "columns": values,
        "availability": "each value is an already strict-OOF causal projection available at its decision timestamp; no realised target is retained",
        "rows": int(len(output)),
        "hashes": {
            "source": _sha256([source_root / "target_free_materialization_audit.parquet"]),
            "path": _sha256(sorted((path_root / "oof_prediction_parts").rglob("*.parquet"))),
            "causal": _sha256(sorted((causal_root / "causal_joint_oof_predictions").glob("*.parquet"))),
        },
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--path-root", type=Path, required=True)
    parser.add_argument("--causal-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({"status": "ok", "out": str(run(
        source_root=args.source_root.resolve(), path_root=args.path_root.resolve(),
        causal_root=args.causal_root.resolve(), out=args.out.resolve(),
    ))}))


if __name__ == "__main__":
    main()
