#!/usr/bin/env python3
"""Prepare immutable date-bounded C0/C1 target-free panels for exact replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ARMS = ("C0_refit_core_postfeb", "C1_refit_core_plus_causal_sr")
REQUIRED = {
    "candidate_id", "__decision_ts__", "__symbol__", "side_name",
    "bcf_mc1_expected_bps", "current_mc1_expected_bps", "auction_priority_bps",
}
FORBIDDEN = {
    "policy_path_valid", "policy_net_bps", "policy_gross_bps", "outcome",
    "label", "exit", "exact_net_bps", "exact_gross_bps",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def slice_target_free_panel(values: pd.DataFrame, *, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Return a target-free interval exactly once, without outcome fields."""
    missing = sorted(REQUIRED.difference(values.columns))
    if missing:
        raise AssertionError(f"source panel lacks required fields: {missing}")
    present = sorted(FORBIDDEN.intersection(values.columns))
    if present:
        raise AssertionError(f"source panel contains outcome fields: {present}")
    result = values.copy()
    result["__decision_ts__"] = pd.to_datetime(result["__decision_ts__"], utc=True, errors="raise")
    result = result.loc[
        result["__decision_ts__"].ge(start) & result["__decision_ts__"].lt(end)
    ].copy()
    if result.empty:
        raise AssertionError("target-free date slice is empty")
    if result["candidate_id"].astype(str).duplicated().any():
        raise AssertionError("target-free date slice duplicates candidate identity")
    if not result["__decision_ts__"].ge(start).all() or not result["__decision_ts__"].lt(end).all():
        raise AssertionError("target-free date-slice boundary violation")
    return result.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    args = parser.parse_args()
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("--start and --end must be explicit UTC timestamps")
    start, end = start.tz_convert("UTC"), end.tz_convert("UTC")
    if end <= start:
        raise ValueError("--end must be after --start")
    source_root, out = args.source_root.resolve(), args.out.resolve()
    if out.exists():
        raise FileExistsError("output must be immutable")

    outputs: dict[str, dict[str, object]] = {}
    prepared: dict[str, pd.DataFrame] = {}
    for arm in ARMS:
        source = source_root / f"{arm}_target_free_admission.parquet"
        if not source.is_file():
            raise FileNotFoundError(source)
        table = slice_target_free_panel(pd.read_parquet(source), start=start, end=end)
        prepared[arm] = table
        outputs[arm] = {
            "source": str(source), "source_sha256": _sha256(source),
            "rows": int(len(table)), "dual_admitted_at_stored_50": int(table.get("dual_admitted", pd.Series(False, index=table.index)).fillna(False).astype(bool).sum()),
        }
    if set(prepared[ARMS[0]]["candidate_id"].astype(str)) != set(prepared[ARMS[1]]["candidate_id"].astype(str)):
        raise AssertionError("C0/C1 target-free source identities are not matched")
    out.mkdir(parents=True, exist_ok=False)
    for arm, table in prepared.items():
        table.to_parquet(out / f"{arm}_direct_target_free.parquet", index=False, compression="zstd")
    (out / "run_manifest.json").write_text(json.dumps({
        "schema": "causal_sr_c1_direct_target_free_slice_v1",
        "scope": "target-free score-panel slicing only; no outcome, policy, portfolio, exchange, or refit input",
        "start": start.isoformat(), "end_exclusive": end.isoformat(),
        "arms": outputs,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
