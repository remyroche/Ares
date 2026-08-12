#!/usr/bin/env python3
"""Refresh the mixed exact/proxy R3 surface with corrected 15m labels.

The historical exact rows and the causal feature contract are immutable.  This
utility replaces only the valid current-year proxy label columns by candidate
ID, preserving the already frozen feature projection and row order.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


LABEL_MAP = {
    "net_bps": "net_bps_proxy_15m",
    "gross_bps": "gross_bps_proxy_15m",
    "pre_adverse_mfe_bps": "pre_adverse_mfe_bps_proxy_15m",
    "lower_touch_minute": "lower_touch_minute_proxy_15m",
    "label_valid": "label_valid",
    "r3_class": "r3_class_proxy_15m",
    "robust_clear_event_b25": "robust_clear_event_b25_proxy_15m",
    "robust_clear_soft_b25_t50": "robust_clear_soft_b25_t50_proxy_15m",
}


def _replace(old: pa.ChunkedArray, new: pa.ChunkedArray, mask: pa.Array) -> pa.Array:
    if new.type != old.type:
        new = pc.cast(new, old.type, safe=False)
    return pc.if_else(mask, new, old)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--old", type=Path, required=True)
    p.add_argument("--proxy", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    old = pq.read_table(args.old)
    proxy = pq.read_table(
        args.proxy,
        columns=["candidate_id", "label_resolution", *sorted(set(LABEL_MAP.values()))],
    )
    # The combined surface intentionally contains valid current rows only.
    proxy = proxy.filter(pc.equal(proxy["label_valid"], True))
    old_proxy = pc.equal(old["label_resolution"], "proxy_15m")
    old_proxy_ids = old["candidate_id"].filter(old_proxy)
    positions = pc.index_in(old_proxy_ids, value_set=proxy["candidate_id"])
    if positions.null_count:
        raise ValueError(f"{positions.null_count} old proxy candidate IDs are absent from corrected labels")
    if len(old_proxy_ids) != proxy.num_rows:
        raise ValueError(f"proxy row mismatch: old={len(old_proxy_ids)} corrected={proxy.num_rows}")
    # Build a full-length index from candidate IDs; historical rows are absent
    # from the proxy set and receive a harmless zero index, masked out below.
    full_pos = pc.fill_null(
        pc.cast(pc.index_in(old["candidate_id"], value_set=proxy["candidate_id"]), pa.int64()),
        0,
    )
    for dst, src in LABEL_MAP.items():
        new_values = pc.take(proxy[src], full_pos)
        idx = old.schema.get_field_index(dst)
        if idx < 0:
            raise ValueError(f"old surface missing expected label column {dst}")
        old = old.set_column(idx, dst, _replace(old[dst], new_values, old_proxy))
    # Preserve the feature contract and row order; only labels changed.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(old, args.out, compression="zstd", version="2.6")
    manifest = {
        "schema": "r3_proxy_training_surface_v2",
        "status": "complete",
        "source_surface": str(args.old),
        "corrected_proxy_source": str(args.proxy),
        "rows": old.num_rows,
        "historical_resolution": "exact_minute",
        "current_resolution": "proxy_15m",
        "updated_proxy_rows": int(pc.sum(pc.cast(old_proxy, pa.int64())).as_py()),
        "target": "R3 robust clear b25 before adverse; timeout distinct from adverse",
        "contract": "decision +1h; first 15m bar; 48 contiguous 15m bars; TP +6 ATR / SL -4 ATR; adverse same-bar precedence; 100 bps cost once",
        "invalid_rows": "excluded from supervised combined surface; retained in proxy source coverage",
        "label_columns_replaced": sorted(LABEL_MAP),
    }
    (args.out.with_suffix(".manifest.json")).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
