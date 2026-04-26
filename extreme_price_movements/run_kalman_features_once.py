from __future__ import annotations

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


KALMAN_FEATURE_KEYS = [
    "kalman_price",
    "kf_score_mean",
    "kf_score_rm24_mean",
    "kf_atr_mean",
    "kf_vol_ratio_mean",
    "kf_ret1h_mean",
    "kf_innov_var",
    "kf_state_uncertainty",
    "kf_snr_est",
    "price_state_slope_1h",
    "price_state_slope_6h",
    "price_state_slope_ratio_1h_6h",
    "price_minus_state_z",
    "price_innovation_z",
    "rolling_std(price_innovation)",
    "state_uncertainty_1h",
    "kalman_gain_1h",
    "vol_state_slope_1h",
    "realized_vol_minus_vol_state",
    "log_volume_state_1h",
    "volume_state_slope_1h",
    "price_slope_x_volume_surprise",
    "vol_state_x_volume_state",
]


def main() -> None:
    cfg = dict(CFG)
    _normalize_cfg_paths(cfg)
    _configure_report_roots(cfg)
    cfg["skip_feature_snapshot_validation"] = True
    cfg["skip_feature_postsave_checks"] = True

    ts_sig = pd.Timestamp("2026-03-21 14:00:00", tz="UTC")
    tprint(
        "Kalman-only feature run start: "
        f"ts={ts_sig} data_root={cfg['data_root']} keys={len(KALMAN_FEATURE_KEYS)}"
    )

    store = PartitionedOHLCVStore(root_dir=cfg["data_root"], timeframe=cfg["timeframe"])
    train_syms = get_training_universe(None, cfg, store, ts_sig=ts_sig)
    lookback_days = max(180, int(cfg["fetch_years"] * 365))

    dfs = {}
    skipped = []
    for sym in train_syms:
        df = store.load(sym, end_ts=ts_sig)
        if df.empty or len(df) < 24 * 60:
            skipped.append(sym)
            continue
        if (ts_sig - df.index[-1]).days > 180:
            skipped.append(sym)
            continue
        dfs[sym] = df.tail(24 * lookback_days)

    tprint(f"Loaded symbols={len(dfs)} skipped={len(skipped)}")
    if not dfs:
        raise RuntimeError("No valid symbols loaded for Kalman-only feature run.")

    panel = to_panel(dfs)
    market = compute_market_features(panel, cfg["market_basket"])
    gates = add_regime_gates(
        market, cfg["gate_vol_lookback_hours"], cfg["gate_trend_thr"]
    )

    chunk_size = max(1, int(cfg.get("feature_backfill_symbol_chunk_size", 140)))
    all_syms = list(panel["close"].columns)
    chunks = (len(all_syms) + chunk_size - 1) // chunk_size
    total_saved_keys = 0

    for chunk_id, start in enumerate(range(0, len(all_syms), chunk_size), start=1):
        chunk_syms = all_syms[start : start + chunk_size]
        tprint(f"Kalman chunk {chunk_id}/{chunks}: symbols={len(chunk_syms)}")
        panel_chunk = {
            key: value.reindex(columns=chunk_syms).copy()
            for key, value in panel.items()
            if isinstance(value, pd.DataFrame)
        }
        feats, feat_index, feat_columns = compute_features_hourly(
            panel_chunk,
            gates.copy(),
            cfg,
            requested_feature_keys=KALMAN_FEATURE_KEYS,
        )
        missing = [key for key in KALMAN_FEATURE_KEYS if key not in feats]
        if missing:
            raise RuntimeError(
                "Missing Kalman feature keys before save: " + ", ".join(missing)
            )
        save_features(
            feats,
            ts_sig,
            cfg["data_root"],
            feat_index=feat_index,
            feat_columns=feat_columns,
            save_workers=int(cfg.get("feature_save_workers", 2)),
        )
        total_saved_keys += len(feats)
        del panel_chunk, feats
        gc.collect()

    tprint(f"Kalman-only feature run complete: saved_feature_blocks={total_saved_keys}")


if __name__ == "__main__":
    main()
