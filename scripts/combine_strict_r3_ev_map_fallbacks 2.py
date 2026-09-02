#!/usr/bin/env python3
"""Build causal timestamp-local fallbacks around exact-reserve EV admission.

The exact producer-reserve map remains primary.  At a decision timestamp where
it admits at least one candidate, no fallback may enter.  Only when the exact
map admits nobody may a predeclared cell-day/Bayesian arm supply candidates.
All decisions use the contemporaneous cross-section and already causal map
outputs; no future day-level admission count is consulted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ARMS = {
    "exact_primary_trim15_timestamp_fallback": ("cell_day_trim_15pct",),
    "exact_primary_bayes90_timestamp_fallback": ("bayes_k07_p90",),
    "exact_primary_trim15_bayes90_agree_timestamp_fallback": (
        "cell_day_trim_15pct", "bayes_k07_p90",
    ),
    "exact_primary_equalday_bayes90_agree_timestamp_fallback": (
        "cell_day_equal_weight", "bayes_k07_p90",
    ),
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_fallbacks(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"candidate_id", "__decision_ts__"}
    for source in {"exact_reserve_control", *(item for values in ARMS.values() for item in values)}:
        required.update({f"{source}__admitted", f"{source}__expected_net_bps"})
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"EV-map selection lacks: {missing}")
    if frame["candidate_id"].duplicated().any():
        raise ValueError("EV-map fallback requires unique candidate identities")
    out = frame.loc[:, ["candidate_id", "__decision_ts__"]].copy()
    exact = frame["exact_reserve_control__admitted"].fillna(False).astype(bool)
    exact_any = exact.groupby(frame["__decision_ts__"], sort=False).transform("any")
    exact_ev = pd.to_numeric(frame["exact_reserve_control__expected_net_bps"], errors="coerce")
    for arm, sources in ARMS.items():
        fallback = pd.Series(True, index=frame.index)
        fallback_ev: list[pd.Series] = []
        for source in sources:
            fallback &= frame[f"{source}__admitted"].fillna(False).astype(bool)
            fallback_ev.append(pd.to_numeric(frame[f"{source}__expected_net_bps"], errors="coerce"))
        selected_fallback = (~exact_any) & fallback
        admitted = exact | selected_fallback
        mean_fallback_ev = pd.concat(fallback_ev, axis=1).mean(axis=1)
        expected = exact_ev.where(exact, mean_fallback_ev.where(selected_fallback))
        if (admitted & ~np.isfinite(expected)).any():
            raise AssertionError("admitted fallback candidate lacks finite common-bps EV")
        out[f"{arm}__admitted"] = admitted.to_numpy(bool)
        out[f"{arm}__expected_net_bps"] = expected.to_numpy(float)
        out[f"{arm}__source"] = np.where(
            exact, "exact_reserve_primary",
            np.where(selected_fallback, "+".join(sources), "rejected"),
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable EV-map fallback output exists: {args.out_dir}")
    source = pd.read_parquet(args.selection)
    output = build_fallbacks(source)
    args.out_dir.mkdir(parents=True)
    output.to_parquet(args.out_dir / "fallback_selection.parquet", index=False, compression="zstd")
    audit = []
    for arm in ARMS:
        selected = output[f"{arm}__admitted"].fillna(False).astype(bool)
        values = output.loc[selected, f"{arm}__source"].value_counts()
        audit.append({
            "arm": arm, "selected_rows": int(selected.sum()),
            "exact_primary_rows": int(values.get("exact_reserve_primary", 0)),
            "fallback_rows": int(selected.sum() - values.get("exact_reserve_primary", 0)),
        })
    pd.DataFrame(audit).to_parquet(args.out_dir / "fallback_audit.parquet", index=False)
    manifest = {
        "schema": "strict_r3_timestamp_local_ev_map_fallback_v1",
        "promotion_status": "research_ablation_only",
        "source_selection": str(args.selection), "source_selection_sha256": _sha(args.selection),
        "rows": int(len(output)), "arms": ARMS,
        "contract": (
            "exact reserve primary at every decision timestamp; fallback candidates "
            "eligible only when the exact map admits zero candidates in that same "
            "contemporaneous cross-section; no day-end or future admission count"
        ),
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", "out_dir": str(args.out_dir), **manifest}))


if __name__ == "__main__":
    main()
