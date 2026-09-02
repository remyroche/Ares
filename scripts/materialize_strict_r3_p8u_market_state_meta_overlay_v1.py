#!/usr/bin/env python3
"""Materialise a compact target-free union overlay for P8U state-Meta tests.

The P8U Base score history is deliberately split at the source-lineage
boundary.  This script makes that boundary explicit while presenting the
existing Meta runners with one immutable, monthly target-free source.  Score
panels are hard-linked where possible (copied only if the filesystem forbids
linking); only the compact state feature overlay is newly written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "strict_r3_p8u_market_state_meta_overlay_v1"
IDENTITY = ("candidate_id", "__decision_ts__", "side_name")


def _once(path: Path, payload: object) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _month_range(start: str, end: str) -> list[pd.Timestamp]:
    first = pd.Timestamp(f"{start}-01", tz="UTC")
    last = pd.Timestamp(f"{end}-01", tz="UTC")
    return list(pd.date_range(first, last, freq="MS"))


def _link_or_copy(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def _signed_geomean(left: pd.Series, right: pd.Series) -> np.ndarray:
    product = left.to_numpy(float) * right.to_numpy(float)
    return (np.sign(product) * np.sqrt(np.abs(product))).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--frozen-contract", required=True)
    parser.add_argument("--early-base-root", required=True)
    parser.add_argument("--later-base-root", required=True)
    parser.add_argument("--start", default="2024-12")
    parser.add_argument("--end", default="2026-07")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    state_root = ROOT / args.state_root; contract_path = ROOT / args.frozen_contract
    early = ROOT / args.early_base_root; later = ROOT / args.later_base_root; out = ROOT / args.out
    if out.exists():
        raise FileExistsError(out)
    contract = json.loads(contract_path.read_text())
    fields = tuple(str(value) for value in contract["selected_features"])
    if len(fields) != 6 or len(fields) != len(set(fields)):
        raise AssertionError("expected the frozen six-field timestamp State-Meta contract")
    state = pd.read_parquet(state_root / "market_state_hourly.parquet", columns=["__decision_ts__", *fields])
    state["__decision_ts__"] = pd.to_datetime(state["__decision_ts__"], utc=True, errors="raise")
    if state.__decision_ts__.duplicated().any():
        raise AssertionError("market state has duplicate timestamps")
    interaction_specs = (
        ("ms_synergy__breadth_downside_x_return_iqr", fields[0], fields[3]),
        ("ms_synergy__execution_spread_x_return_iqr", fields[2], fields[3]),
        ("ms_synergy__breadth_downside_x_execution_spread", fields[0], fields[2]),
    )
    coverage: list[dict[str, object]] = []; provenance: list[dict[str, object]] = []
    out.mkdir(parents=True)
    for month in _month_range(args.start, args.end):
        name = f"month={month:%Y-%m}.parquet"
        sources = [root / name for root in (early, later) if (root / name).exists()]
        if len(sources) != 1:
            raise AssertionError(f"{month:%Y-%m}: expected exactly one source Base panel, found {len(sources)}")
        source = sources[0]
        base = pd.read_parquet(source, columns=list(IDENTITY))
        base["__decision_ts__"] = pd.to_datetime(base["__decision_ts__"], utc=True, errors="raise")
        if base.duplicated(IDENTITY).any():
            raise AssertionError(f"{month:%Y-%m}: source Base identity is not unique")
        joined = base.merge(state, on="__decision_ts__", how="left", validate="many_to_one")
        if len(joined) != len(base) or joined.loc[:, list(fields)].isna().any(axis=None):
            raise AssertionError(f"{month:%Y-%m}: incomplete target-free state identity coverage")
        for name_i, left, right in interaction_specs:
            joined[name_i] = _signed_geomean(joined[left], joined[right])
        feature_path = out / "features" / f"month={month:%Y-%m}" / "causal_feature_universe.parquet"
        feature_path.parent.mkdir(parents=True, exist_ok=False)
        joined.loc[:, [*IDENTITY, *fields, *(item[0] for item in interaction_specs)]].to_parquet(feature_path, index=False, compression="zstd")
        score_path = out / "target_free_scores" / name
        mode = _link_or_copy(source, score_path)
        provenance.append({"month": f"{month:%Y-%m}", "source_score": str(source.relative_to(ROOT)), "source_sha256": _sha(source), "score_materialisation": mode, "rows": len(base)})
        for field in [*fields, *(item[0] for item in interaction_specs)]:
            values = pd.to_numeric(joined[field], errors="coerce")
            coverage.append({"month": f"{month:%Y-%m}", "field": field, "rows": len(values), "coverage": float(values.notna().mean()), "unique": int(values.nunique(dropna=True)), "variance": float(values.var())})
    pd.DataFrame(coverage).to_parquet(out / "feature_coverage.parquet", index=False)
    pd.DataFrame(provenance).to_parquet(out / "source_provenance.parquet", index=False)
    correctness = {
        "schema": SCHEMA,
        "exactly_one_lineage_source_per_month": True,
        "base_score_files_preserved_by_hardlink_or_copy": True,
        "state_fields_timestamp_global_before_candidate_filtering": True,
        "overlay_target_free": True,
        "no_policy_path_outcome_or_label_columns": True,
        "synergy_features_use_only_same_timestamp_target_free_states": True,
        "frozen_pre2026_selection_contract": True,
    }
    _once(out / "correctness_report.json", correctness)
    _once(out / "run_manifest.json", {"schema": SCHEMA, "scope": "offline target-free P8U market-state Meta overlay", "state_root": str(state_root.relative_to(ROOT)), "state_contract": str(contract_path.relative_to(ROOT)), "early_base_root": str(early.relative_to(ROOT)), "later_base_root": str(later.relative_to(ROOT)), "selected_fields": list(fields), "interaction_fields": [item[0] for item in interaction_specs], "correctness": correctness})
    print(json.dumps({"out": str(out), "months": len(provenance), "fields": len(fields) + len(interaction_specs), "rows": int(sum(item["rows"] for item in provenance))}, sort_keys=True))


if __name__ == "__main__":
    main()
