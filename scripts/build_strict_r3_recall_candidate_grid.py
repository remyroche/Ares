#!/usr/bin/env python3
"""Build a target-free, point-in-time candidate grid for recall research.

The broad recall feature screen must start from the same full candidate
population that was available at each decision timestamp.  This small helper
turns the immutable monthly target-free source into the input convention used
by ``materialize_strict_r3_forward_features.py`` without reading any outcome
or path-validity field.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def _utc(value: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _months(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    return list(pd.date_range(start.normalize().replace(day=1), (end - pd.Timedelta(nanoseconds=1)).normalize().replace(day=1), freq="MS", tz="UTC"))


def _sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path).encode())
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path)
    parser.add_argument(
        "--score-source-root", type=Path,
        help=("optional complete target-free score-panel root.  When supplied, "
              "only candidate_id/decision_ts/side_name are read and symbol is "
              "recovered from the immutable candidate identity.  This is useful "
              "when a raw target-free shard is damaged."),
    )
    parser.add_argument("--start", required=True, help="inclusive decision UTC timestamp")
    parser.add_argument("--end-exclusive", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--universe-manifest",
        type=Path,
        help=(
            "Frozen target-free universe receipt containing a source_map. "
            "Builds the complete timestamp × symbol grid directly from its "
            "keys, without opening labels or a scored monthly source."
        ),
    )
    args = parser.parse_args()

    if args.universe_manifest is not None:
        if args.source_root is not None or args.score_source_root is not None:
            raise ValueError(
                "--universe-manifest is mutually exclusive with --source-root "
                "and --score-source-root"
            )
    elif args.source_root is None:
        raise ValueError("--source-root is required without --universe-manifest")

    start, end = _utc(args.start), _utc(args.end_exclusive)
    if not start < end:
        raise ValueError("start must precede end-exclusive")
    paths: list[Path] = []
    pieces: list[pd.DataFrame] = []
    if args.universe_manifest is not None:
        source = json.loads(args.universe_manifest.read_text(encoding="utf-8"))
        source_map = source.get("source_map") if isinstance(source, dict) else None
        if not isinstance(source_map, dict) or not source_map:
            raise ValueError("universe manifest has no non-empty source_map")
        symbols = sorted(str(symbol) for symbol in source_map)
        # ``start`` is the first signal-close timestamp.  The strict-R3
        # decision is exactly one hour later, so an append beginning at the
        # prior grid's end has no overlapping decision identity.
        signal_ts = pd.date_range(start, end, freq="1h", inclusive="left", tz="UTC")
        raw = pd.DataFrame({
            "__ts__": signal_ts.repeat(len(symbols)),
            "__symbol__": symbols * len(signal_ts),
        })
        raw["__decision_ts__"] = raw["__ts__"] + pd.Timedelta(hours=1)
        raw["side_name"] = "long"
        raw["candidate_id"] = (
            raw["__symbol__"].astype(str)
            + "|long|"
            + raw["__ts__"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        paths.append(args.universe_manifest)
        pieces.append(raw.loc[:, ["candidate_id", "__decision_ts__", "__symbol__", "side_name"]])
    else:
        for month in _months(start, end):
            if args.score_source_root is None:
                path = args.source_root / f"month={month:%Y-%m}" / "part-002.parquet"
                columns = ["candidate_id", "__decision_ts__", "__symbol__", "side_name"]
            else:
                path = args.score_source_root / f"month={month:%Y-%m}" / "scores_features.parquet"
                columns = ["candidate_id", "__decision_ts__", "side_name"]
            if not path.exists():
                raise FileNotFoundError(path)
            raw = pd.read_parquet(path, columns=columns)
            if args.score_source_root is not None:
                # Candidate IDs are immutable under the strict-R3 contract:
                # ``SYMBOL|long|signal_timestamp``.  No model score, path value,
                # label, or future availability is read from this fallback source.
                pieces_id = raw["candidate_id"].astype(str).str.rsplit("|", n=2, expand=True)
                if pieces_id.shape[1] != 3 or not pieces_id[1].eq(raw["side_name"].astype(str)).all():
                    raise ValueError("score-source candidate IDs do not match the strict-R3 identity convention")
                raw["__symbol__"] = pieces_id[0]
            raw["__decision_ts__"] = pd.to_datetime(raw["__decision_ts__"], utc=True, errors="raise")
            raw = raw.loc[raw["__decision_ts__"].ge(start) & raw["__decision_ts__"].lt(end)].copy()
            if raw.empty:
                continue
            paths.append(path)
            pieces.append(raw)
    if not pieces:
        raise ValueError("selected period has no target-free candidates")
    grid = pd.concat(pieces, ignore_index=True)
    if not grid.side_name.astype(str).str.lower().eq("long").all():
        raise AssertionError("recall research grid must be long-only")
    if grid.candidate_id.duplicated().any():
        raise AssertionError("target-free candidate source has duplicate candidate IDs")
    grid["__ts__"] = grid["__decision_ts__"] - pd.Timedelta(hours=1)
    grid = grid.loc[:, ["candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name"]]
    grid = grid.sort_values(["__decision_ts__", "__symbol__", "candidate_id"], kind="stable").reset_index(drop=True)
    counts = grid.groupby("__decision_ts__", sort=True)["__symbol__"].nunique()
    if (counts < 2).any():
        raise AssertionError("candidate source has a degenerate timestamp universe")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    grid.to_parquet(args.out, index=False, compression="zstd")
    manifest = {
        "schema": "strict_r3_recall_target_free_candidate_grid_v1",
        "source_root": str(args.source_root) if args.source_root else None,
        "score_source_root": str(args.score_source_root) if args.score_source_root else None,
        "universe_manifest": str(args.universe_manifest) if args.universe_manifest else None,
        "source_files": [str(path) for path in paths],
        "source_sha256": _sha256(paths),
        "decision_start": str(start),
        "decision_end_exclusive": str(end),
        "rows": int(len(grid)),
        "timestamps": int(grid["__decision_ts__"].nunique()),
        "symbols": int(grid["__symbol__"].nunique()),
        "minimum_symbols_per_timestamp": int(counts.min()),
        "maximum_symbols_per_timestamp": int(counts.max()),
        "outcome_fields_read": False,
        "score_fields_read": False,
    }
    args.out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    print(json.dumps(manifest, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
