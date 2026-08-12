#!/usr/bin/env python3
"""Build the schema-v2 point-in-time signal grid from causal hourly inputs.

The grid is independent of H12 path availability.  A symbol-hour exists when
the signal-hour close and the decision-hour open are both available at the
declared signal+one-hour entry time.  The frozen exact170 spread registry is
the only instrument-level admission input at this stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_r3_canonical_v2 import (  # noqa: E402
    CandidateSpec,
    SCHEMA,
    build_point_in_time_candidates,
)
from scripts.run_tp6_sl4_exact170_canonical_consensus import (  # noqa: E402
    FROZEN_INPUT_BACKFILL_ROOT,
    _make_panel,
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _signal_hour_spread_panel(
    symbols: list[str],
    signal_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Load the causal signal-hour Kraken bid/ask spread.

    The historical universe registry decides membership only.  Actionability
    uses the official order-book analytics timestamp that precedes the
    signal+one-hour entry decision.  Missing/stale analytics stay missing and
    are rejected by ``build_point_in_time_candidates``.
    """
    values: dict[str, pd.Series] = {}
    for symbol in symbols:
        base = symbol.split("/", 1)[0]
        path = FROZEN_INPUT_BACKFILL_ROOT / f"{base}_USD_USD.parquet"
        if not path.exists():
            continue
        try:
            frame = pd.read_parquet(
                path, columns=["ob_bid_bestPrice", "ob_ask_bestPrice"],
            )
        except Exception:
            continue
        frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")
        bid = pd.to_numeric(frame["ob_bid_bestPrice"], errors="coerce")
        ask = pd.to_numeric(frame["ob_ask_bestPrice"], errors="coerce")
        mid = 0.5 * (bid + ask)
        spread = (10_000.0 * (ask - bid) / mid.where(mid > 0.0)).replace(
            [np.inf, -np.inf], np.nan,
        )
        values[symbol] = spread.reindex(signal_index)
    return (
        pd.concat(values, axis=1).reindex(index=signal_index, columns=symbols)
        if values else pd.DataFrame(index=signal_index, columns=symbols, dtype=float)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--universe-csv", type=Path)
    source.add_argument(
        "--universe-manifest", type=Path,
        help="Prior immutable schema-v2 target-free manifest whose source_map keys freeze the universe",
    )
    parser.add_argument("--start", required=True)
    parser.add_argument("--end-exclusive", required=True)
    parser.add_argument("--sides", default="long")
    parser.add_argument("--spread-limit-bps", type=float, default=100.0)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output already exists: {args.out_dir}")
    if args.universe_manifest is not None:
        prior = json.loads(args.universe_manifest.read_text())
        if prior.get("schema") != f"{SCHEMA}_target_free_hourly_grid":
            raise ValueError("universe manifest is not a schema-v2 target-free grid")
        if float(prior.get("spread_limit_bps", np.nan)) != float(args.spread_limit_bps):
            raise ValueError("universe manifest uses a different spread limit")
        frozen_symbols = list((prior.get("source_map") or {}).keys())
        if not frozen_symbols:
            raise ValueError("universe manifest has no frozen source_map keys")
        # Membership, rather than an invented spread estimate, is the frozen
        # contract.  The boundary value makes every already-admitted member
        # pass the identical <= limit check without misrepresenting a new
        # measured spread.
        universe_table = pd.DataFrame({
            "symbol": frozen_symbols,
            "p90_spread_bps": float(args.spread_limit_bps),
        })
        universe_source = args.universe_manifest
        universe_source_type = "prior_schema_v2_admitted_membership"
    else:
        universe_table = pd.read_csv(args.universe_csv)
        if "p90_spread_bps" not in universe_table and "average_spread_bps" in universe_table:
            universe_table = universe_table.rename(columns={"average_spread_bps": "p90_spread_bps"})
        if "symbol" not in universe_table or "p90_spread_bps" not in universe_table:
            raise ValueError("universe CSV requires symbol and p90_spread_bps")
        universe_table = universe_table.loc[
            pd.to_numeric(universe_table["p90_spread_bps"], errors="coerce")
            .le(float(args.spread_limit_bps))
        ].copy()
        universe_source = args.universe_csv
        universe_source_type = "spread_registry"
    symbols = universe_table["symbol"].dropna().astype(str).drop_duplicates().tolist()
    start = pd.to_datetime(args.start, utc=True)
    end = pd.to_datetime(args.end_exclusive, utc=True)
    if start >= end:
        raise ValueError("start must precede end-exclusive")
    # Include the decision hour so its open is observable for the last signal.
    panel, source_map = _make_panel(symbols, start, end + pd.Timedelta(hours=1))
    close = panel["close"].reindex(columns=symbols)
    decision_open = panel["open"].shift(-1).reindex(columns=symbols)
    signal_index = close.index[(close.index >= start) & (close.index < end)]
    close = close.reindex(signal_index)
    decision_open = decision_open.reindex(signal_index)
    market = close.stack(dropna=False).rename("signal_close").reset_index()
    market.columns = ["__ts__", "__symbol__", "signal_close"]
    decision_values = decision_open.stack(dropna=False).rename("decision_open").reset_index()
    decision_values.columns = ["__ts__", "__symbol__", "decision_open"]
    market = market.merge(
        decision_values, on=["__ts__", "__symbol__"], how="left", validate="one_to_one"
    )
    current_spread = _signal_hour_spread_panel(symbols, signal_index)
    spread_values = current_spread.stack(dropna=False).rename("spread_bps").reset_index()
    spread_values.columns = ["__ts__", "__symbol__", "spread_bps"]
    market = market.merge(
        spread_values, on=["__ts__", "__symbol__"], how="left", validate="one_to_one",
    )
    market["__decision_ts__"] = market["__ts__"] + pd.Timedelta(hours=1)
    market["instrument_available"] = np.isfinite(
        pd.to_numeric(market["signal_close"], errors="coerce")
    )
    market["entry_executable"] = np.isfinite(
        pd.to_numeric(market["decision_open"], errors="coerce")
    )
    sides = tuple(value.strip() for value in args.sides.split(",") if value.strip())
    population, eligible, rejected = build_point_in_time_candidates(
        market,
        universe=symbols,
        feature_fields=(),
        cross_sectional_sources=(),
        spec=CandidateSpec(
            spread_limit_bps=float(args.spread_limit_bps),
            required_feature_fraction=1.0,
            side_names=sides,
        ),
    )
    args.out_dir.mkdir(parents=True)
    population.to_parquet(
        args.out_dir / "target_free_candidate_population.parquet",
        index=False,
        compression="zstd",
    )
    eligible.to_parquet(
        args.out_dir / "eligible_candidates.parquet", index=False, compression="zstd"
    )
    rejected.to_parquet(
        args.out_dir / "candidate_rejection_audit.parquet", index=False, compression="zstd"
    )
    summary = population.groupby(
        ["side_name", "eligibility_reason"], as_index=False, dropna=False
    ).agg(rows=("candidate_id", "size"))
    summary.to_parquet(args.out_dir / "candidate_rejection_reason_summary.parquet", index=False)
    manifest = {
        "schema": f"{SCHEMA}_target_free_hourly_grid",
        "start": start.isoformat(),
        "end_exclusive": end.isoformat(),
        "universe_rows": len(symbols),
        "population_rows": len(population),
        "eligible_rows": len(eligible),
        "rejected_rows": len(rejected),
        "entry": "signal close + one hour; decision-hour open availability only",
        "future_path_columns_consumed": [],
        "spread_limit_bps": float(args.spread_limit_bps),
        "spread_gate": "official_kraken_signal_hour_bid_ask_bps_before_signal_plus_1h_entry",
        "historical_universe_spread_used_for_membership_only": True,
        "universe_sha256": _sha(universe_source),
        "universe_source_type": universe_source_type,
        "source_map": source_map,
    }
    (args.out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str)
    )
    print(json.dumps({"event": "complete", "output": str(args.out_dir), **manifest}))


if __name__ == "__main__":
    main()
