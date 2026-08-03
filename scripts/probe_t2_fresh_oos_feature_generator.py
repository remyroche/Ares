#!/usr/bin/env python3
"""Probe whether the frozen T2 raw features have causal generator support.

This is deliberately a small representative universe probe, not a feature
materialisation or a model run.  It answers a narrow operational question:
which frozen fields can the historical causal feature generator emit from
preserved OHLCV/perp sidecars, and which require a separately frozen latent
state transform.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_POPULATION = ROOT / "data_perp/artifacts/febapr2025_canonical_exact_policy_base_population_20260727_v1/population.parquet"
DEFAULT_FEATURES = ROOT / "data_perp/artifacts/controlled_target_supportive_prepared_ledger_20260801_v5/frozen_raw_causal_features.json"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/t2_fresh_oos_feature_generator_probe_20260801_v1"
LATENT_FIELDS = {
    "AE_reconstruction_error", "mahalanobis_distance", "cluster_acceleration", "cluster_entropy",
    "cluster_entropy_accel_1", "cluster_entropy_delta_1", "cluster_entropy_norm",
    "cluster_flip_count_20", "cluster_speed", "cluster_t",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output_dir}")
    # Local imports keep a metadata/readiness invocation lightweight.
    from extreme_price_movements.config import CFG, enable_perp_feature_keys
    from extreme_price_movements.data_store import PartitionedOHLCVStore, to_panel
    from extreme_price_movements.features import add_regime_gates, compute_market_features
    from extreme_price_movements.pipeline_steps import _compute_features_hourly_runtime

    frozen = list(json.loads(args.features_json.read_text(encoding="utf-8"))["raw_feature_columns"])
    population = pd.read_parquet(args.population, columns=["__symbol__"])
    symbols = sorted(population["__symbol__"].astype(str).unique())[: int(args.symbols)]
    data_root = ROOT / "data_perp"
    cfg = enable_perp_feature_keys(dict(CFG))
    cfg.update({
        "data_root": str(data_root), "exchange_id": "krakenfutures", "market_mode": "perps",
        "use_perps": True, "exchange_scoped_data": True, "feature_portability_mode": "legacy",
    })
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    store = PartitionedOHLCVStore(root_dir=str(data_root / "exchanges/krakenfutures"), timeframe="1h")
    frames = {symbol: store.load(symbol, start_ts=start, end_ts=end) for symbol in symbols}
    frames = {symbol: frame for symbol, frame in frames.items() if not frame.empty}
    if len(frames) != len(symbols):
        raise ValueError(f"historical source is missing probe symbols: expected={len(symbols)} loaded={len(frames)}")
    panel = to_panel(frames)
    market = compute_market_features(panel, list(frames))
    gates = add_regime_gates(market, cfg["gate_vol_lookback_hours"], cfg["gate_trend_thr"])
    generated, _, _ = _compute_features_hourly_runtime(panel, gates, cfg, {}, requested_feature_keys=frozen)
    emitted = sorted(set(frozen).intersection(generated))
    missing = sorted(set(frozen).difference(generated))
    latent_missing = sorted(set(missing).intersection(LATENT_FIELDS))
    other_missing = sorted(set(missing).difference(LATENT_FIELDS))
    args.output_dir.mkdir(parents=True)
    pd.DataFrame({
        "feature": frozen,
        "emitted_by_causal_static_generator_on_probe": [name in generated for name in frozen],
        "requires_frozen_latent_state_transform": [name in LATENT_FIELDS for name in frozen],
    }).to_csv(args.output_dir / "feature_generator_coverage.csv", index=False)
    result = {
        "schema": "t2_fresh_oos_feature_generator_probe_v1",
        "status": "STATIC_CAUSAL_GENERATOR_SUPPORT_PROBED_NOT_FULL_MATERIALISATION",
        "probe": {"symbols": symbols, "start": start.isoformat(), "end": end.isoformat(), "source": "canonical Kraken Futures hourly OHLCV/perp sidecars"},
        "feature_contract": {"path": str(args.features_json), "sha256": _sha256(args.features_json), "frozen_features": len(frozen), "emitted": len(emitted), "missing": len(missing)},
        "missing": {"latent_state_fields": latent_missing, "other": other_missing},
        "interpretation": [
            "The emitted count is feasibility evidence only: full-universe feature values still need exact source, warm-up, availability, and parity audits.",
            "The latent fields must be produced by the pre-evaluation frozen AE/GMM transform; fitting a new latent state on the evaluation period would change the frozen feature contract.",
            "No model, target, execution cost, realised path, or score was read by this probe.",
        ],
    }
    _write(args.output_dir / "probe.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=Path, default=DEFAULT_POPULATION)
    parser.add_argument("--features-json", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--symbols", type=int, default=5)
    parser.add_argument("--start", default="2024-08-01")
    parser.add_argument("--end", default="2025-02-02")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
