#!/usr/bin/env python3
"""Verify saved live inference feature/prediction snapshots against recompute."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extreme_price_movements.inference.config import (  # noqa: E402
    get_inference_defaults,
    load_inference_config,
)
from extreme_price_movements.inference.feature_generator import (  # noqa: E402
    _latest_feature_matrix,
    _required_tail_warmup_hours,
    load_or_compute_features,
)
from extreme_price_movements.inference.model_orchestrator import (  # noqa: E402
    ModelOrchestrator,
)
from extreme_price_movements.inference.portfolio_policy import (  # noqa: E402
    load_portfolio_policy_config,
)
from extreme_price_movements.model_loader import load_full_state  # noqa: E402
from scripts.replay_live_signal_predictions import _load_panel  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a persisted live transformed-feature snapshot with a fresh "
            "training-style recompute from raw live data, then compare alpha/meta "
            "model predictions from both matrices."
        )
    )
    parser.add_argument("--data-root", default="data_perp")
    parser.add_argument(
        "--live-data-root", default="data_perp/exchanges/krakenfutures"
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--market-mode", default="perps")
    parser.add_argument("--snapshot-dir", default=None)
    parser.add_argument("--lookback-hours", type=int, default=24 * 60)
    parser.add_argument("--max-symbols", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument(
        "--output",
        default=None,
        help="JSON report path. Defaults under live_state/parity_reports.",
    )
    return parser.parse_args()


def _latest_snapshot_dir(run_id: str, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    root = Path("cache") / "inference_live_features" / str(run_id)
    candidates = []
    for meta_path in root.glob("*/meta.json"):
        try:
            meta = json.loads(meta_path.read_text())
            end_ts = pd.Timestamp(meta.get("end_ts"))
            candidates.append((end_ts, meta_path.stat().st_mtime, meta_path.parent))
        except Exception:
            continue
    if not candidates:
        raise FileNotFoundError(f"No live feature snapshot found under {root}")
    return sorted(candidates, key=lambda item: (item[0], item[1]))[-1][2]


def _read_snapshot(snapshot_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    matrix_path = snapshot_dir / "latest.parquet"
    meta_path = snapshot_dir / "meta.json"
    if not matrix_path.exists():
        raise FileNotFoundError(matrix_path)
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    matrix = pd.read_parquet(matrix_path)
    meta = json.loads(meta_path.read_text())
    matrix.index = matrix.index.astype(str)
    return matrix, meta


def _read_snapshot_raw_panel(snapshot_dir: Path, meta: dict[str, Any]) -> dict[str, pd.DataFrame]:
    raw_path = meta.get("raw_panel_path")
    if not raw_path:
        return {}
    path = snapshot_dir / str(raw_path)
    if not path.exists():
        return {}
    raw = pd.read_parquet(path)
    if raw.empty or not isinstance(raw.index, pd.MultiIndex):
        return {}
    out: dict[str, pd.DataFrame] = {}
    try:
        timestamps = pd.to_datetime(raw.index.get_level_values("timestamp"), utc=True)
        symbols = raw.index.get_level_values("symbol").astype(str)
    except Exception:
        return {}
    raw = raw.copy()
    raw.index = pd.MultiIndex.from_arrays(
        [timestamps, symbols], names=["timestamp", "symbol"]
    )
    for field in raw.columns:
        frame = raw[str(field)].unstack("symbol").sort_index()
        frame.index = pd.to_datetime(frame.index, utc=True)
        out[str(field)] = frame
    return out


def _strategy_side_kind(strategy_id: str) -> tuple[str, str]:
    raw = str(strategy_id)
    for side in ("long", "short"):
        prefix = f"{side}_"
        if raw.startswith(prefix):
            return side, raw[len(prefix) :]
    raise ValueError(f"Cannot infer side from strategy_id={strategy_id!r}")


def _compare_matrices(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    atol: float,
) -> dict[str, Any]:
    common_index = sorted(set(map(str, left.index)) & set(map(str, right.index)))
    common_cols = sorted(set(map(str, left.columns)) & set(map(str, right.columns)))
    left_aligned = left.reindex(index=common_index, columns=common_cols).astype(
        np.float64
    )
    right_aligned = right.reindex(index=common_index, columns=common_cols).astype(
        np.float64
    )
    lhs = left_aligned.to_numpy(dtype=np.float64, copy=False)
    rhs = right_aligned.to_numpy(dtype=np.float64, copy=False)
    both_finite = np.isfinite(lhs) & np.isfinite(rhs)
    finite_compared = int(both_finite.sum())
    diffs = np.abs(lhs - rhs)
    finite_diffs = diffs[both_finite]
    mismatch_mask = both_finite & (diffs > float(atol))
    only_left = int((np.isfinite(lhs) & ~np.isfinite(rhs)).sum())
    only_right = int((~np.isfinite(lhs) & np.isfinite(rhs)).sum())
    top: list[dict[str, Any]] = []
    if finite_compared:
        by_col = np.nanmax(np.where(both_finite, diffs, np.nan), axis=0)
        order = np.argsort(np.nan_to_num(by_col, nan=-1.0))[::-1][:20]
        for pos in order:
            if not np.isfinite(by_col[pos]) or by_col[pos] <= 0:
                continue
            col_mask = both_finite[:, int(pos)]
            col_diffs = diffs[:, int(pos)]
            example: dict[str, Any] = {}
            if col_mask.any():
                row_pos = int(np.nanargmax(np.where(col_mask, col_diffs, np.nan)))
                example = {
                    "symbol": common_index[row_pos],
                    "snapshot_value": float(lhs[row_pos, int(pos)]),
                    "recomputed_value": float(rhs[row_pos, int(pos)]),
                    "abs_diff": float(col_diffs[row_pos]),
                }
            top.append(
                {
                    "feature": common_cols[int(pos)],
                    "max_abs_diff": float(by_col[pos]),
                    "example": example,
                }
            )
    return {
        "common_symbols": len(common_index),
        "common_features": len(common_cols),
        "finite_values_compared": finite_compared,
        "finite_mismatches_gt_atol": int(mismatch_mask.sum()),
        "snapshot_only_finite_values": only_left,
        "recompute_only_finite_values": only_right,
        "max_abs_diff": float(np.nanmax(finite_diffs)) if finite_diffs.size else None,
        "top_feature_diffs": top,
    }


def _prediction_report(
    orchestrator: ModelOrchestrator,
    snapshot: pd.DataFrame,
    recomputed: pd.DataFrame,
    strategy_ids: list[str],
    *,
    atol: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for strategy_id in strategy_ids:
        side, kind = _strategy_side_kind(strategy_id)
        alpha_snapshot = orchestrator.predict_alpha(snapshot, side, kind)
        alpha_recompute = orchestrator.predict_alpha(recomputed, side, kind)
        meta_snapshot = pd.Series(dtype=float)
        meta_recompute = pd.Series(dtype=float)
        if not alpha_snapshot.empty:
            meta_base_snapshot = snapshot.copy()
            meta_base_snapshot[kind] = alpha_snapshot
            meta_base_snapshot[strategy_id] = alpha_snapshot
            meta_snapshot = orchestrator.predict_meta(
                meta_base_snapshot, side, strategy_id
            )
        if not alpha_recompute.empty:
            meta_base_recompute = recomputed.copy()
            meta_base_recompute[kind] = alpha_recompute
            meta_base_recompute[strategy_id] = alpha_recompute
            meta_recompute = orchestrator.predict_meta(
                meta_base_recompute, side, strategy_id
            )

        def _series_cmp(left: pd.Series, right: pd.Series) -> dict[str, Any]:
            common = sorted(set(map(str, left.index)) & set(map(str, right.index)))
            if not common:
                return {
                    "common_rows": 0,
                    "finite_rows": 0,
                    "mismatches_gt_atol": 0,
                    "max_abs_diff": None,
                }
            lvals = pd.to_numeric(left.reindex(common), errors="coerce").to_numpy(
                dtype=np.float64
            )
            rvals = pd.to_numeric(right.reindex(common), errors="coerce").to_numpy(
                dtype=np.float64
            )
            ok = np.isfinite(lvals) & np.isfinite(rvals)
            diffs = np.abs(lvals - rvals)
            return {
                "common_rows": len(common),
                "finite_rows": int(ok.sum()),
                "mismatches_gt_atol": int((ok & (diffs > float(atol))).sum()),
                "max_abs_diff": float(np.nanmax(diffs[ok])) if ok.any() else None,
            }

        out.append(
            {
                "strategy_id": strategy_id,
                "alpha": _series_cmp(alpha_snapshot, alpha_recompute),
                "meta": _series_cmp(meta_snapshot, meta_recompute),
            }
        )
    return out


def main() -> int:
    args = _parse_args()
    snapshot_dir = _latest_snapshot_dir(args.run_id, args.snapshot_dir)
    snapshot, meta = _read_snapshot(snapshot_dir)
    end_ts = pd.Timestamp(meta["end_ts"])
    end_ts = end_ts.tz_localize("UTC") if end_ts.tzinfo is None else end_ts.tz_convert("UTC")
    compute_symbols = [str(s) for s in (meta.get("symbols") or list(snapshot.index))]
    compute_symbols = [s for s in compute_symbols if s in snapshot.index]
    symbols = list(compute_symbols)
    if args.max_symbols and args.max_symbols > 0:
        symbols = symbols[: int(args.max_symbols)]

    cfg = load_inference_config(
        data_root=args.data_root, run_id=args.run_id, market_mode=args.market_mode
    )
    cfg["market_mode"] = args.market_mode
    cfg["data_root"] = str(args.live_data_root)
    cfg["artifact_data_root"] = str(args.data_root)
    cfg["live_data_root"] = str(args.live_data_root)
    runtime_cfg = dict(cfg.get("runtime_cfg") or {})
    runtime_cfg.update(
        {
            "data_root": str(args.live_data_root),
            "artifact_data_root": str(args.data_root),
            "live_data_root": str(args.live_data_root),
            "market_mode": args.market_mode,
            "live_feature_snapshot_cache_enabled": False,
            "live_feature_memory_cache_enabled": False,
            "live_feature_rolling_cache_enabled": False,
            "live_feature_snapshot_cache_write_enabled": False,
            "strict_feature_parity": True,
        }
    )
    cfg["runtime_cfg"] = runtime_cfg
    model_bundle = load_full_state(args.run_id, args.data_root)
    cfg["model_bundle"] = model_bundle
    runtime_cfg["model_bundle"] = model_bundle

    defaults = get_inference_defaults()
    lookback_hours = max(
        int(args.lookback_hours),
        _required_tail_warmup_hours(
            lookback_hours=int(args.lookback_hours),
            trend_sma_hours=int(defaults["trend_sma_hours"]),
            gate_vol_lookback_hours=int(defaults["gate_vol_lookback_hours"]),
        ),
    )
    # DataFetcher.load_panel(lookback_hours=N) returns N closed hourly bars,
    # inclusive of end_ts.  Recompute proof must use the same N-row window;
    # starting at end_ts - N hours would include one extra warmup row and can
    # legitimately alter rolling/causal transforms.
    start_ts = end_ts - pd.Timedelta(hours=max(int(lookback_hours) - 1, 0))
    saved_raw_panel = _read_snapshot_raw_panel(snapshot_dir, meta)
    if saved_raw_panel:
        panel = saved_raw_panel
        panel_source = "snapshot_raw_panel"
    else:
        panel = _load_panel(
            data_root=Path(args.live_data_root),
            symbols=compute_symbols,
            start_ts=start_ts,
            end_ts=end_ts,
        )
        panel_source = "live_data_root"
    panel = {
        key: frame.loc[
            (pd.to_datetime(frame.index, utc=True, errors="coerce") >= start_ts)
            & (pd.to_datetime(frame.index, utc=True, errors="coerce") <= end_ts)
        ]
        for key, frame in panel.items()
        if isinstance(frame, pd.DataFrame) and not frame.empty
    }
    feats = load_or_compute_features(
        panel=panel,
        basket_syms=compute_symbols,
        run_id=args.run_id,
        data_root=str(args.live_data_root),
        cfg=cfg,
        lookback_hours=lookback_hours,
        required_feature_keys=set(map(str, snapshot.columns)),
    )
    recomputed = _latest_feature_matrix(
        feats,
        compute_symbols,
        end_ts,
        set(map(str, snapshot.columns)),
    ).reindex(index=compute_symbols)

    policy = load_portfolio_policy_config(
        data_root=args.data_root,
        run_id=args.run_id,
        runtime_cfg={},
        require_artifact=True,
    )
    strategy_ids = list(policy.strategy_ids)
    orchestrator = ModelOrchestrator(model_bundle, runtime_cfg=runtime_cfg)
    report = {
        "run_id": args.run_id,
        "snapshot_dir": str(snapshot_dir),
        "snapshot_end_ts": end_ts.isoformat(),
        "symbols_compared": len(symbols),
        "symbols_computed": len(compute_symbols),
        "snapshot_features": int(snapshot.shape[1]),
        "recomputed_features": int(recomputed.shape[1]),
        "raw_panel_source": panel_source,
        "raw_panel_fields": sorted(panel.keys()),
        "snapshot_only_features": sorted(
            set(map(str, snapshot.columns)) - set(map(str, recomputed.columns))
        ),
        "recompute_only_features": sorted(
            set(map(str, recomputed.columns)) - set(map(str, snapshot.columns))
        ),
        "lookback_hours": int(lookback_hours),
        "feature_comparison": _compare_matrices(
            snapshot.reindex(index=symbols),
            recomputed,
            atol=float(args.atol),
        ),
        "prediction_comparison": _prediction_report(
            orchestrator,
            snapshot.reindex(index=symbols),
            recomputed,
            strategy_ids,
            atol=float(args.atol),
        ),
    }

    output = (
        Path(args.output)
        if args.output
        else Path(args.live_data_root)
        / "live_state"
        / "parity_reports"
        / f"live_snapshot_parity_{args.run_id}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
