from __future__ import annotations

import argparse
import gc

import pandas as pd

from extreme_price_movements.config import CFG
from extreme_price_movements.data_store import (
    PartitionedOHLCVStore,
    save_features,
    to_panel,
)
from extreme_price_movements.features import (
    add_regime_gates,
    compute_features_hourly,
    compute_market_features,
)
from extreme_price_movements.run_pipeline import (
    _configure_report_roots,
    _normalize_cfg_paths,
)
from extreme_price_movements.universe import get_training_universe
from extreme_price_movements.utils import tprint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ts", required=True, help="Timestamp in YYYYMMDD_HHMMSS")
    parser.add_argument("--key", action="append", required=True, help="Feature key")
    args = parser.parse_args()

    cfg = dict(CFG)
    _normalize_cfg_paths(cfg)
    _configure_report_roots(cfg)
    cfg["skip_feature_snapshot_validation"] = True
    cfg["skip_feature_postsave_checks"] = True

    ts_sig = pd.Timestamp(args.ts[:8] + " " + args.ts[9:], tz="UTC")
    feature_keys = list(dict.fromkeys(str(k) for k in args.key if str(k).strip()))
    tprint(
        f"Selected-features run start: ts={ts_sig} "
        f"keys={len(feature_keys)} data_root={cfg['data_root']}"
    )

    store = PartitionedOHLCVStore(root_dir=cfg["data_root"], timeframe=cfg["timeframe"])
    train_syms = get_training_universe(None, cfg, store, ts_sig=ts_sig)
    lookback_days = max(180, int(cfg["fetch_years"] * 365))

    dfs: dict[str, pd.DataFrame] = {}
    for sym in train_syms:
        df = store.load(sym, end_ts=ts_sig)
        if df.empty or len(df) < 24 * 60:
            continue
        if (ts_sig - df.index[-1]).days > 180:
            continue
        dfs[sym] = df.tail(24 * lookback_days)

    if not dfs:
        raise RuntimeError("No valid symbols loaded for selected-features run.")

    tprint(f"Loaded symbols={len(dfs)}")
    panel = to_panel(dfs)
    market = compute_market_features(panel, cfg["market_basket"])
    gates = add_regime_gates(
        market, cfg["gate_vol_lookback_hours"], cfg["gate_trend_thr"]
    )

    chunk_size = max(1, int(cfg.get("feature_backfill_symbol_chunk_size", 140)))
    all_syms = list(panel["close"].columns)
    total_chunks = (len(all_syms) + chunk_size - 1) // chunk_size

    for chunk_id, start in enumerate(range(0, len(all_syms), chunk_size), start=1):
        chunk_syms = all_syms[start : start + chunk_size]
        tprint(f"Selected-features chunk {chunk_id}/{total_chunks}: {len(chunk_syms)}")
        panel_chunk = {
            key: value.reindex(columns=chunk_syms).copy()
            for key, value in panel.items()
            if isinstance(value, pd.DataFrame)
        }
        feats, feat_index, feat_columns = compute_features_hourly(
            panel_chunk,
            gates.copy(),
            cfg,
            requested_feature_keys=feature_keys,
        )
        missing = [key for key in feature_keys if key not in feats]
        if missing:
            raise RuntimeError(
                "Missing selected feature keys before save: " + ", ".join(missing)
            )
        save_features(
            feats,
            ts_sig,
            cfg["data_root"],
            feat_index=feat_index,
            feat_columns=feat_columns,
            save_workers=int(cfg.get("feature_save_workers", 2)),
        )
        del panel_chunk, feats
        gc.collect()

    tprint("Selected-features run complete.")


if __name__ == "__main__":
    main()
