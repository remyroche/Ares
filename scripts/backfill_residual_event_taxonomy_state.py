#!/usr/bin/env python3
"""Materialize a slim observable state source for block-level regime research.

Historical residual-event artifacts predate several OI/funding/recovery
features.  Their schema therefore contains columns that are structurally
present but entirely null.  This utility backfills only the observable
mechanism basket from an existing causal feature store.  It does not refit an
AE/GMM, alter base/meta scores, calculate outcomes, or change policy rows.

One Parquet part is emitted for each source artifact so memory remains bounded.
The resulting files are inputs to descriptive/event-detector research only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.residual_event_block_taxonomy import (  # noqa: E402
    MECHANISM_FAMILIES,
)
from scripts.run_residual_event_archetype_discovery import (  # noqa: E402
    _append_feature_store_basket,
)


STATE_KEYS = (
    "__ts__",
    "__symbol__",
    "side_name",
    "archetype_policy_key",
    "selected_top30",
)


def _read_projection(path: Path) -> pd.DataFrame:
    available = set(pq.ParquetFile(path).schema.names)
    missing = set(STATE_KEYS[:4]).difference(available)
    if missing:
        raise KeyError(f"State artifact {path} missing keys: {sorted(missing)}")
    columns = [name for name in STATE_KEYS if name in available]
    result = pd.read_parquet(path, columns=columns)
    result["__ts__"] = pd.to_datetime(result["__ts__"], utc=True, errors="coerce")
    result = result.loc[result["__ts__"].notna()].copy()
    for name in ("__symbol__", "side_name", "archetype_policy_key"):
        result[name] = result[name].astype(str)
    if "selected_top30" not in result:
        result["selected_top30"] = True
    return result


def run(args: argparse.Namespace) -> dict[str, object]:
    args.output.mkdir(parents=True, exist_ok=True)
    features = list(
        dict.fromkeys(
            feature for group in MECHANISM_FAMILIES.values() for feature in group
        )
    )
    records: list[dict[str, object]] = []
    for index, source in enumerate(args.state_artifact):
        state = _read_projection(source)
        state, coverage = _append_feature_store_basket(
            state,
            feature_root=args.feature_root,
            requested=features,
            batch_size=args.feature_append_batch_size,
        )
        part = args.output / f"state_part_{index:02d}.parquet"
        state.to_parquet(
            part,
            index=False,
            compression="zstd",
            row_group_size=int(args.row_group_size),
        )
        records.append(
            {
                "source": str(source),
                "part": str(part),
                "rows": int(len(state)),
                "start": str(state["__ts__"].min()),
                "end": str(state["__ts__"].max()),
                "feature_coverage": coverage,
            }
        )
        del state
    manifest = {
        "purpose": "observable-only taxonomy-state backfill; no model or policy mutation",
        "feature_root": str(args.feature_root),
        "mechanism_features": features,
        "state_key_contract": list(STATE_KEYS),
        "source_artifacts": records,
        "causal_contract": (
            "All backfilled values are read at their original feature-store "
            "timestamps. Outcomes, residual labels, and future values are not "
            "read or written."
        ),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-artifact", type=Path, action="append", required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-append-batch-size", type=int, default=8)
    parser.add_argument(
        "--row-group-size", type=int, default=100_000,
        help="Bound downstream Parquet batch memory for the slim state source.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(f"completed parts={len(result['source_artifacts'])}")
