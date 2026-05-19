#!/usr/bin/env python3
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

from extreme_price_movements.config import CFG  # noqa: E402
from extreme_price_movements.ebm_on_lgbm import (  # noqa: E402
    _compute_oof_bundle_tree_frame,
    _compute_soft_tree_features_ebm,
)
import extreme_price_movements.features as features_mod  # noqa: E402
from extreme_price_movements.feature_transforms import CausalFeatureTransformer  # noqa: E402
from extreme_price_movements.inference.feature_generator import (  # noqa: E402
    _feature_snapshot_dir,
    _live_feature_cache_key,
    _required_tail_warmup_hours,
    get_features_for_candidates,
    get_inference_required_feature_keys,
    load_or_compute_features,
    raw_required_feature_keys,
)
from extreme_price_movements.inference.config import load_inference_config  # noqa: E402
from extreme_price_movements.inference.model_orchestrator import (  # noqa: E402
    ModelOrchestrator,
    _alpha_prediction_frame_for_model,
    _extract_ebm_contract_model,
    _synthetic_ebm_raw_features,
)
from extreme_price_movements.inference.parity import strategy_core_id  # noqa: E402
from extreme_price_movements.model_loader import load_full_state  # noqa: E402
from extreme_price_movements.inference.run_inference import (  # noqa: E402
    _lgbm_mask_required_feature_keys,
    _load_lgbm_strategy_mask_rows,
)
from scripts.replay_live_signal_predictions import (  # noqa: E402
    _live_feature_cache_symbols_for_end,
    _load_panel,
    _load_recent_decisions,
    _local_quote_symbols,
    _normalise_symbol,
)


class _CaptureTransform(CausalFeatureTransformer):
    captured: dict[str, pd.DataFrame] = {}

    def transform_batch(self, features, skip_keys=None, chunk_size=50):  # type: ignore[override]
        out: dict[str, pd.DataFrame] = {}
        for key, val in (features or {}).items():
            if isinstance(val, pd.DataFrame):
                out[str(key)] = val.copy()
            else:
                try:
                    out[str(key)] = pd.DataFrame(val).copy()
                except Exception:
                    pass
        prev = _CaptureTransform.captured
        prev_width = sum(len(df.columns) for df in prev.values())
        out_width = sum(len(df.columns) for df in out.values())
        if not prev or out_width > prev_width:
            _CaptureTransform.captured = out
        return super().transform_batch(features, skip_keys=skip_keys, chunk_size=chunk_size)


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _row_value(feats: dict[str, pd.DataFrame], key: str, symbol: str, ts: pd.Timestamp) -> float:
    df = feats.get(str(key))
    if not isinstance(df, pd.DataFrame) or df.empty or symbol not in df.columns:
        return float("nan")
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
        df = df.copy()
        df.index = idx
    else:
        df = df.copy()
        df.index = idx.tz_convert("UTC")
    ts = pd.Timestamp(ts)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    hist = df.loc[df.index <= ts, symbol].dropna()
    if hist.empty:
        return float("nan")
    return _safe_float(hist.iloc[-1])


def _matrix_row_to_feature_dict(matrix: pd.DataFrame, symbol: str, ts: pd.Timestamp) -> dict[str, float]:
    if matrix is None or matrix.empty or symbol not in matrix.index:
        return {}
    row = matrix.loc[symbol]
    return {str(k): _safe_float(v) for k, v in row.items()}


def _load_snapshot_matrix(
    *,
    run_id: str,
    data_root: Path,
    symbols: list[str],
    required_keys: set[str],
    lookback_hours: int,
    end_ts: pd.Timestamp,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], Path]:
    cache_key = _live_feature_cache_key(
        run_id=run_id,
        symbols=symbols,
        required_feature_keys=required_keys,
        lookback_hours=lookback_hours,
        cfg=cfg,
        data_root=str(data_root),
    )
    cache_dir = _feature_snapshot_dir(cfg, run_id, cache_key)
    meta_path = cache_dir / "meta.json"
    data_path = cache_dir / "latest.parquet"
    meta: dict[str, Any] = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    if not data_path.exists():
        return pd.DataFrame(), meta, cache_dir
    matrix = pd.read_parquet(data_path)
    return matrix, meta, cache_dir


def _feature_comparison(
    *,
    symbol: str,
    ts: pd.Timestamp,
    keys: list[str],
    pre_feats: dict[str, pd.DataFrame],
    post_feats: dict[str, pd.DataFrame],
    snapshot_values: dict[str, float],
) -> pd.DataFrame:
    rows = []
    for key in keys:
        pre = _row_value(pre_feats, key, symbol, ts)
        post = _row_value(post_feats, key, symbol, ts)
        snap = snapshot_values.get(str(key), float("nan"))
        rows.append(
            {
                "feature": str(key),
                "replay_pre_transform": pre,
                "replay_post_transform": post,
                "snapshot_post_transform": snap,
                "post_minus_pre": post - pre if np.isfinite(post) and np.isfinite(pre) else np.nan,
                "snapshot_minus_replay_post": snap - post if np.isfinite(snap) and np.isfinite(post) else np.nan,
                "pre_missing": not np.isfinite(pre),
                "post_missing": not np.isfinite(post),
                "snapshot_missing": not np.isfinite(snap),
            }
        )
    return pd.DataFrame(rows)


def _tree_frame(model: Any, X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ebm = _extract_ebm_contract_model(model)
    if ebm is None:
        return pd.DataFrame(index=X.index), pd.DataFrame(index=X.index)
    raw_names = [str(c) for c in (getattr(ebm, "raw_selected_features", []) or [])]
    if not raw_names:
        return pd.DataFrame(index=X.index), pd.DataFrame(index=X.index)
    raw_df = X.reindex(columns=raw_names, fill_value=0.0)
    synthetic_raw_contract = all(name.startswith("f") and name[1:].isdigit() for name in raw_names)
    positional_mapping = (
        getattr(ebm, "positional_feature_mapping", None)
        or getattr(ebm, "meta_positional_feature_mapping_", None)
        or getattr(model, "positional_feature_mapping", None)
        or getattr(model, "meta_positional_feature_mapping_", None)
        or {}
    )
    if synthetic_raw_contract and isinstance(positional_mapping, dict) and positional_mapping:
        mapped = pd.DataFrame(index=X.index)
        for raw_name in raw_names:
            real_name = str(positional_mapping.get(raw_name, ""))
            mapped[raw_name] = X[real_name] if real_name in X.columns else 0.0
        raw_df = mapped
    elif synthetic_raw_contract and raw_df.abs().sum(axis=1).iloc[0] == 0.0 and X.shape[1] >= len(raw_names):
        raw_df = X.iloc[:, : len(raw_names)].copy()
        raw_df.columns = raw_names
    raw_df = raw_df.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    tree_names = [str(c) for c in (getattr(ebm, "tree_feature_names", []) or [])]
    if not tree_names:
        return raw_df, pd.DataFrame(index=X.index)
    tree_config = getattr(ebm, "tree_feature_config", {}) or {}
    if isinstance(tree_config, dict) and tree_config.get("oof_tree_features"):
        tree_df = _compute_oof_bundle_tree_frame(
            tree_config,
            raw_df,
            selected_tree_names=tree_names,
        )
    else:
        arr, names, _ = _compute_soft_tree_features_ebm(
            getattr(ebm, "tree_models", []) or [],
            raw_df.to_numpy(dtype=np.float32),
            getattr(ebm, "tree_feature_scales", None),
            selected_names=set(tree_names),
        )
        tree_df = pd.DataFrame(arr, columns=names, index=raw_df.index)
    for name in tree_names:
        if name not in tree_df.columns:
            tree_df[name] = 0.0
    tree_df = tree_df.reindex(columns=tree_names, fill_value=0.0).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return raw_df, tree_df


def _alpha_input(orchestrator: ModelOrchestrator, feature_row: pd.DataFrame, side: str, strategy_id: str) -> tuple[Any, pd.DataFrame, list[str]]:
    model_info = orchestrator.alpha_by_strategy.get(strategy_id) or orchestrator.alpha_by_strategy.get(f"{side}_{strategy_id}")
    if not isinstance(model_info, dict):
        return None, pd.DataFrame(), []
    model = model_info.get("model")
    feat_cols = [str(c) for c in (model_info.get("feat_cols", []) or [])]
    X = orchestrator._align_alpha_feature_contract(feature_row, feat_cols)
    X = _alpha_prediction_frame_for_model(model, X, feat_cols)
    return model, X, feat_cols


def _meta_input(orchestrator: ModelOrchestrator, feature_row: pd.DataFrame, side: str, model_strategy_id: str, base_pred: float) -> tuple[Any, pd.DataFrame, list[str], str]:
    selected = str(model_strategy_id)
    core = strategy_core_id(selected)
    candidates = [
        selected,
        f"{selected}_clf",
        f"{selected}_tbm_clf",
        f"{side}_{core}",
        f"{side}_{core}_clf",
        f"{side}_{core}_tbm_clf",
        core,
        f"{core}_clf",
        f"{core}_tbm_clf",
    ]
    key = next((cand for cand in candidates if cand in orchestrator.meta_models), candidates[0])
    model = orchestrator.meta_models.get(key)
    if model is None:
        return None, pd.DataFrame(), [], key
    feat_cols = [str(c) for c in (getattr(model, "feature_columns", []) or [])]
    meta = feature_row.copy()
    # Mirror run_full_chain's base-pred injection for the selected strategy.
    meta[model_strategy_id] = float(base_pred)
    core = strategy_core_id(model_strategy_id)
    if core:
        meta[core] = float(base_pred)
    meta = orchestrator._materialize_meta_model_derived_features(
        meta,
        model,
        side=side,
        kind=model_strategy_id,
    )
    X = meta.reindex(columns=feat_cols, fill_value=0.0).fillna(0.0)
    return model, X, feat_cols, key


def _tree_comparison(name: str, raw_df: pd.DataFrame, tree_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature, value in raw_df.iloc[0].items():
        rows.append({"layer": f"{name}_raw_contract", "feature": str(feature), "value": _safe_float(value)})
    for feature, value in tree_df.iloc[0].items():
        rows.append({"layer": f"{name}_lgbm_tree", "feature": str(feature), "value": _safe_float(value)})
    return pd.DataFrame(rows)


def _frame_long(df: pd.DataFrame, layer: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["layer", "feature", "value"])
    row = df.iloc[0]
    return pd.DataFrame(
        {
            "layer": layer,
            "feature": [str(c) for c in row.index],
            "value": [_safe_float(v) for v in row.to_numpy()],
        }
    )


def _find_live_debug_dir(
    *,
    data_root: Path,
    run_id: str,
    symbol: str,
    side: str,
    strategy_id: str,
    ts: pd.Timestamp,
) -> Path | None:
    root = data_root / "artifacts" / run_id / "live_feature_layer_debug"
    if not root.exists():
        return None
    target_ts = ts.tz_convert("UTC") if ts.tzinfo is not None else ts.tz_localize("UTC")
    matches: list[tuple[pd.Timestamp, Path]] = []
    for summary_path in root.glob("*/*/summary.json"):
        try:
            summary = json.loads(summary_path.read_text())
        except Exception:
            continue
        if _normalise_symbol(summary.get("symbol")) != symbol:
            continue
        if str(summary.get("side") or "").lower() != side:
            continue
        selected = str(summary.get("selected_strategy") or summary.get("strategy_id") or "")
        if strategy_core_id(selected) != strategy_core_id(strategy_id):
            continue
        try:
            signal_ts = pd.Timestamp(summary.get("signal_bar_ts"))
            signal_ts = signal_ts.tz_localize("UTC") if signal_ts.tzinfo is None else signal_ts.tz_convert("UTC")
        except Exception:
            continue
        if signal_ts != target_ts:
            continue
        try:
            created = pd.Timestamp(summary.get("created_at"))
            created = created.tz_localize("UTC") if created.tzinfo is None else created.tz_convert("UTC")
        except Exception:
            created = pd.Timestamp.min.tz_localize("UTC")
        matches.append((created, summary_path.parent))
    if not matches:
        return None
    return sorted(matches, key=lambda item: item[0])[-1][1]


def _read_live_debug_values(debug_dir: Path | None, filename: str) -> dict[str, float]:
    if debug_dir is None:
        return {}
    path = debug_dir / filename
    if not path.exists():
        return {}
    try:
        df = pd.read_parquet(path)
    except Exception:
        csv_path = path.with_suffix(".csv")
        if not csv_path.exists():
            return {}
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            return {}
    if df.empty:
        return {}
    if {"feature", "value"}.issubset(df.columns):
        return {str(r["feature"]): _safe_float(r["value"]) for _, r in df.iterrows()}
    row = df.iloc[0]
    return {str(c): _safe_float(v) for c, v in row.items()}


def _read_live_debug_wide(debug_dir: Path | None, filename: str) -> pd.DataFrame:
    if debug_dir is None:
        return pd.DataFrame()
    path = debug_dir / filename
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    if {"feature", "value"}.issubset(frame.columns):
        return pd.DataFrame([frame.set_index("feature")["value"].astype(float).to_dict()])
    return frame


def _live_replay_layer_comparison(
    replay_layers: pd.DataFrame,
    debug_dir: Path | None,
) -> pd.DataFrame:
    live_files = {
        "alpha_model_input": "alpha_model_input.parquet",
        "meta_model_input": "meta_model_input.parquet",
        "alpha_raw_contract": "alpha_ebm_raw_contract.parquet",
        "alpha_lgbm_tree": "alpha_lgbm_tree_features.parquet",
        "meta_raw_contract": "meta_ebm_raw_contract.parquet",
        "meta_lgbm_tree": "meta_lgbm_tree_features.parquet",
    }
    live_rows: list[dict[str, Any]] = []
    for layer, filename in live_files.items():
        for feature, value in _read_live_debug_values(debug_dir, filename).items():
            live_rows.append({"layer": layer, "feature": feature, "live_value": value})
    live = pd.DataFrame(live_rows)
    if live.empty:
        return pd.DataFrame(
            columns=["layer", "feature", "live_value", "replay_value", "delta", "abs_delta"]
        )
    replay = replay_layers.rename(columns={"value": "replay_value"}).copy()
    merged = live.merge(replay, on=["layer", "feature"], how="outer")
    merged["delta"] = pd.to_numeric(merged["live_value"], errors="coerce") - pd.to_numeric(
        merged["replay_value"], errors="coerce"
    )
    merged["abs_delta"] = merged["delta"].abs()
    merged["live_missing"] = pd.to_numeric(merged["live_value"], errors="coerce").isna()
    merged["replay_missing"] = pd.to_numeric(merged["replay_value"], errors="coerce").isna()
    return merged.sort_values(["abs_delta", "layer", "feature"], ascending=[False, True, True], na_position="last")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data", type=Path)
    parser.add_argument(
        "--live-data-root",
        type=Path,
        default=None,
        help=(
            "Root containing live exchange OHLCV/funding/orderbook data. "
            "Defaults to --data-root. Use this when artifacts and live data are split."
        ),
    )
    parser.add_argument("--run-id", default="20260321_140000")
    parser.add_argument(
        "--market-mode",
        choices=("spot", "perps"),
        default=None,
        help="Inference feature/config mode. Defaults to perps when data-root contains 'perp', otherwise spot.",
    )
    parser.add_argument("--ledger", default="data/live_state/prediction_ledger.parquet", type=Path)
    parser.add_argument("--trades", default="inference_trades.csv", type=Path)
    parser.add_argument("--decision-start", default="2026-05-15T21:20:00Z")
    parser.add_argument("--require-rank-source", default="policy_rank_reference_percentile")
    parser.add_argument(
        "--live-quote-currency",
        default=None,
        help="Live quote currency used for local symbol discovery. Defaults to USD in perps mode, otherwise USDC.",
    )
    parser.add_argument("--max-rows", type=int, default=1)
    parser.add_argument(
        "--live-debug-dir",
        type=Path,
        help=(
            "Compare replay layers against a specific live_feature_layer_debug "
            "decision directory. This is useful when the live cycle wrote layer "
            "debug artifacts but no prediction-ledger row was appended because "
            "all candidates were rejected before the ledger cutoff."
        ),
    )
    parser.add_argument(
        "--lookback-hours",
        type=int,
        default=24 * 60,
        help="Feature lookback hours. Default matches live inference.",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    live_data_root = args.live_data_root or args.data_root
    market_mode = args.market_mode or (
        "perps" if "perp" in str(args.data_root).lower() else "spot"
    )
    live_quote_currency = (
        str(args.live_quote_currency).upper()
        if args.live_quote_currency
        else ("USD" if market_mode == "perps" else "USDC")
    )

    live_debug_dir_arg = args.live_debug_dir
    row: pd.Series
    live_debug_dir: Path | None = None
    live_summary: dict[str, Any] = {}
    if live_debug_dir_arg is not None:
        live_debug_dir = live_debug_dir_arg
        summary_path = live_debug_dir / "summary.json"
        if not summary_path.exists():
            raise SystemExit(f"Missing live debug summary: {summary_path}")
        try:
            summary = json.loads(summary_path.read_text())
            live_summary = dict(summary)
        except Exception as exc:
            raise SystemExit(f"Could not read live debug summary {summary_path}: {exc}") from exc
        row = pd.Series(
            {
                "decision_ts": summary.get("decision_ts") or summary.get("created_at"),
                "signal_bar_ts": summary.get("signal_bar_ts"),
                "symbol": summary.get("symbol"),
                "side": summary.get("side"),
                "strategy_id": summary.get("selected_strategy") or summary.get("strategy_id"),
                "raw_prediction_score": summary.get("meta_pred"),
                "live_meta_pred": summary.get("meta_pred"),
                "base_pred": summary.get("base_pred"),
                "meta_pred": summary.get("meta_pred"),
                "calibrated_score": summary.get("calibrated_score"),
                "policy_rank_pct": summary.get("policy_rank_pct"),
                "rank_score_source": summary.get("rank_score_source"),
            }
        )
    else:
        decisions = _load_recent_decisions(
            ledger_path=args.ledger,
            trades_path=args.trades,
            max_rows=max(1, args.max_rows),
            decision_start=args.decision_start,
            require_rank_source=args.require_rank_source,
        )
        if decisions.empty:
            raise SystemExit("No decisions matched filters.")
        row = decisions.sort_values("decision_ts").iloc[-1]
    symbol = _normalise_symbol(row["symbol"])
    side = str(row["side"]).lower()
    strategy_id = str(row["strategy_id"])
    core = strategy_core_id(strategy_id)
    model_strategy_id = strategy_id if strategy_id.startswith(f"{side}_") else f"{side}_{core}"
    ts = pd.Timestamp(row["signal_bar_ts"])
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    panel_end_ts = ts

    state = load_full_state(args.run_id, str(args.data_root))
    required_keys = raw_required_feature_keys(
        set(get_inference_required_feature_keys(state, None))
    )
    try:
        mask_rows = _load_lgbm_strategy_mask_rows(
            str(args.data_root),
            args.run_id,
            market_mode=market_mode,
        )
        required_keys |= set(_lgbm_mask_required_feature_keys(mask_rows))
    except Exception:
        pass
    summary_symbols = live_summary.get("feature_universe_symbols")
    symbols = (
        sorted(str(s) for s in summary_symbols)
        if isinstance(summary_symbols, list) and summary_symbols
        else []
    )
    replay_feature_universe_source = "live_debug_summary" if symbols else ""
    if not symbols:
        symbols = _live_feature_cache_symbols_for_end(
            live_data_root,
            run_id=args.run_id,
            end_ts=ts,
            live_quote_currency=live_quote_currency,
        )
        if symbols:
            replay_feature_universe_source = "snapshot_cache_symbols"
    if not symbols:
        symbols = _local_quote_symbols(
            live_data_root,
            run_id=args.run_id,
            live_quote_currency=live_quote_currency,
            market_mode=market_mode,
        )
        replay_feature_universe_source = f"local_{live_quote_currency.lower()}_symbols"
    effective_lookback_hours = max(
        int(args.lookback_hours),
        _required_tail_warmup_hours(
            lookback_hours=int(args.lookback_hours),
            trend_sma_hours=24 * 14,
            gate_vol_lookback_hours=24 * 7,
        ),
    )
    start_ts = ts - pd.Timedelta(hours=effective_lookback_hours)
    panel = _load_panel(
        data_root=live_data_root,
        symbols=symbols,
        start_ts=start_ts,
        end_ts=panel_end_ts,
    )
    if not panel or "close" not in panel:
        raise SystemExit("No panel loaded.")

    try:
        cfg_snapshot = load_inference_config(
            data_root=str(args.data_root),
            run_id=args.run_id,
            market_mode=market_mode,
        )
    except Exception:
        cfg_snapshot = dict(CFG)
    cfg_snapshot["market_mode"] = market_mode
    cfg_snapshot["data_root"] = str(args.data_root)
    runtime_cfg = dict(cfg_snapshot.get("runtime_cfg") or {})
    runtime_cfg["use_perps"] = market_mode == "perps"
    runtime_cfg["market_mode"] = market_mode
    runtime_cfg["data_root"] = str(live_data_root)
    runtime_cfg["artifact_data_root"] = str(args.data_root)
    runtime_cfg["live_data_root"] = str(live_data_root)
    state_bundle = state.get("bundle", {}) if isinstance(state.get("bundle"), dict) else {}
    runtime_cfg.setdefault("bundle", state_bundle)
    for key in (
        "feature_transform_contract",
        "feature_transform_contract_hash",
        "feature_transform_manifest",
    ):
        value = state.get(key)
        if value is None:
            value = state_bundle.get(key)
        if value is not None:
            cfg_snapshot[key] = value
            runtime_cfg[key] = value
    cfg_snapshot["runtime_cfg"] = runtime_cfg
    feature_cfg_snapshot = dict(runtime_cfg)
    snapshot_matrix, snapshot_meta, snapshot_dir = _load_snapshot_matrix(
        run_id=args.run_id,
        data_root=live_data_root,
        symbols=symbols,
        required_keys=required_keys,
        lookback_hours=effective_lookback_hours,
        end_ts=panel_end_ts,
        cfg=feature_cfg_snapshot,
    )
    snapshot_values = _matrix_row_to_feature_dict(snapshot_matrix, symbol, ts)
    if live_debug_dir is None:
        live_debug_dir = _find_live_debug_dir(
            data_root=args.data_root,
            run_id=args.run_id,
            symbol=symbol,
            side=side,
            strategy_id=strategy_id,
            ts=ts,
        )
    live_post_values = _read_live_debug_values(
        live_debug_dir, "post_transform_candidate_features.parquet"
    )
    if live_post_values:
        snapshot_values = live_post_values

    old_transform = features_mod.CausalFeatureTransformer
    _CaptureTransform.captured = {}
    features_mod.CausalFeatureTransformer = _CaptureTransform
    try:
        cfg_replay = dict(cfg_snapshot)
        cfg_replay["live_feature_snapshot_cache_enabled"] = False
        cfg_replay["live_feature_memory_cache_enabled"] = False
        # Replay comparisons must not seed/append the production rolling cache.
        # The comparator only needs an in-memory recompute for one timestamp.
        cfg_replay["live_feature_rolling_cache_enabled"] = False
        if isinstance(cfg_replay.get("runtime_cfg"), dict):
            cfg_replay["runtime_cfg"] = dict(cfg_replay["runtime_cfg"])
            cfg_replay["runtime_cfg"]["live_feature_snapshot_cache_enabled"] = False
            cfg_replay["runtime_cfg"]["live_feature_memory_cache_enabled"] = False
            cfg_replay["runtime_cfg"]["live_feature_rolling_cache_enabled"] = False
        post_feats = load_or_compute_features(
            panel,
            list(panel["close"].columns),
            args.run_id,
            str(live_data_root),
            cfg_replay,
            lookback_hours=effective_lookback_hours,
            required_feature_keys=required_keys,
        )
        pre_feats = dict(_CaptureTransform.captured)
    finally:
        features_mod.CausalFeatureTransformer = old_transform

    feature_row = get_features_for_candidates(post_feats, [symbol], ts=ts)
    orchestrator = ModelOrchestrator(state, runtime_cfg={"model_bundle": state.get("bundle", {})})
    alpha_model, alpha_X, alpha_feat_cols = _alpha_input(orchestrator, feature_row, side, model_strategy_id)
    alpha_pred = float("nan")
    if alpha_model is not None and not alpha_X.empty:
        try:
            alpha_pred = float(alpha_model.predict(alpha_X)[0])
        except Exception as exc:
            print(f"WARNING: alpha replay prediction failed: {exc}", file=sys.stderr)
    live_base_pred = _safe_float(row.get("base_pred"))
    base_for_meta = alpha_pred if np.isfinite(alpha_pred) else live_base_pred
    meta_model, meta_X, meta_feat_cols, meta_key = _meta_input(
        orchestrator,
        feature_row,
        side,
        model_strategy_id,
        base_for_meta,
    )
    meta_pred = float("nan")
    if meta_model is not None and not meta_X.empty:
        try:
            meta_pred = float(meta_model.predict(meta_X)[0])
        except Exception as exc:
            print(f"WARNING: meta replay prediction failed: {exc}", file=sys.stderr)

    live_debug_alpha_input_pred = float("nan")
    live_debug_meta_input_pred = float("nan")
    if live_debug_dir is not None and live_summary:
        live_alpha_X = _read_live_debug_wide(live_debug_dir, "alpha_model_input.parquet")
        live_meta_X = _read_live_debug_wide(live_debug_dir, "meta_model_input.parquet")
        live_alpha_key = str(live_summary.get("alpha_model_key") or "")
        live_meta_key = str(live_summary.get("meta_model_key") or "")
        live_alpha_info = orchestrator.alpha_by_strategy.get(live_alpha_key)
        live_alpha_model = (
            live_alpha_info.get("model") if isinstance(live_alpha_info, dict) else None
        )
        live_alpha_feat_cols = [
            str(c)
            for c in (
                live_alpha_info.get("feat_cols", [])
                if isinstance(live_alpha_info, dict)
                else []
            )
        ]
        live_meta_model = orchestrator.meta_models.get(live_meta_key)
        if live_alpha_model is not None and not live_alpha_X.empty:
            try:
                live_alpha_X = _alpha_prediction_frame_for_model(
                    live_alpha_model,
                    live_alpha_X,
                    live_alpha_feat_cols,
                )
                live_debug_alpha_input_pred = float(live_alpha_model.predict(live_alpha_X)[0])
            except Exception as exc:
                print(f"WARNING: live debug alpha input prediction failed: {exc}", file=sys.stderr)
        if live_meta_model is not None and not live_meta_X.empty:
            try:
                live_debug_meta_input_pred = float(live_meta_model.predict(live_meta_X)[0])
            except Exception as exc:
                print(f"WARNING: live debug meta input prediction failed: {exc}", file=sys.stderr)

    feature_keys = sorted(set(alpha_feat_cols).union(meta_feat_cols).union(required_keys))
    feature_cmp = _feature_comparison(
        symbol=symbol,
        ts=ts,
        keys=feature_keys,
        pre_feats=pre_feats,
        post_feats=post_feats,
        snapshot_values=snapshot_values,
    )
    feature_cmp["abs_snapshot_post_delta"] = pd.to_numeric(
        feature_cmp["snapshot_minus_replay_post"], errors="coerce"
    ).abs()
    feature_cmp = feature_cmp.sort_values(
        ["abs_snapshot_post_delta", "feature"], ascending=[False, True], na_position="last"
    )

    tree_frames = []
    if alpha_model is not None and not alpha_X.empty:
        tree_frames.append(_frame_long(alpha_X, "alpha_model_input"))
        try:
            raw_df, tree_df = _tree_frame(alpha_model, alpha_X)
            tree_frames.append(_tree_comparison("alpha", raw_df, tree_df))
        except Exception as exc:
            print(f"WARNING: alpha tree-layer reconstruction failed: {exc}", file=sys.stderr)
    if meta_model is not None and not meta_X.empty:
        tree_frames.append(_frame_long(meta_X, "meta_model_input"))
        try:
            raw_df, tree_df = _tree_frame(meta_model, meta_X)
            tree_frames.append(_tree_comparison("meta", raw_df, tree_df))
        except Exception as exc:
            print(f"WARNING: meta tree-layer reconstruction failed: {exc}", file=sys.stderr)
    tree_cmp = pd.concat(tree_frames, ignore_index=True) if tree_frames else pd.DataFrame()
    live_replay_layers = _live_replay_layer_comparison(tree_cmp, live_debug_dir)

    out_dir = args.output_dir or (
        args.data_root / "artifacts" / args.run_id / "live_feature_layer_compare"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_cmp.to_csv(out_dir / "feature_pre_post_snapshot_comparison.csv", index=False)
    tree_cmp.to_csv(out_dir / "lgbm_leaf_value_features.csv", index=False)
    live_replay_layers.to_csv(out_dir / "live_vs_replay_model_layers.csv", index=False)

    summary = {
        "decision_ts": str(row.get("decision_ts")),
        "signal_bar_ts": str(ts),
        "feature_panel_end_ts": str(panel_end_ts),
        "effective_lookback_hours": int(effective_lookback_hours),
        "symbol": symbol,
        "side": side,
        "strategy_id": strategy_id,
        "model_strategy_id": model_strategy_id,
        "market_mode": market_mode,
        "live_data_root": str(live_data_root),
        "replay_feature_universe_n": int(len(symbols)),
        "replay_feature_universe_source": replay_feature_universe_source,
        "live_meta_pred": _safe_float(row.get("live_meta_pred", row.get("raw_prediction_score"))),
        "live_debug_alpha_input_pred": live_debug_alpha_input_pred,
        "live_debug_meta_input_pred": live_debug_meta_input_pred,
        "live_debug_alpha_input_delta": (
            live_debug_alpha_input_pred - _safe_float(row.get("base_pred"))
            if np.isfinite(live_debug_alpha_input_pred)
            and np.isfinite(_safe_float(row.get("base_pred")))
            else float("nan")
        ),
        "live_debug_meta_input_delta": (
            live_debug_meta_input_pred - _safe_float(row.get("live_meta_pred", row.get("raw_prediction_score")))
            if np.isfinite(live_debug_meta_input_pred)
            and np.isfinite(_safe_float(row.get("live_meta_pred", row.get("raw_prediction_score"))))
            else float("nan")
        ),
        "replay_alpha_pred": alpha_pred,
        "meta_replay_base_pred_source": "replay_alpha_pred" if np.isfinite(alpha_pred) else "live_base_pred",
        "replay_meta_pred": meta_pred,
        "snapshot_dir": str(snapshot_dir),
        "snapshot_meta_end_ts": snapshot_meta.get("end_ts"),
        "snapshot_matrix_rows": int(len(snapshot_matrix.index)) if not snapshot_matrix.empty else 0,
        "snapshot_matrix_features": int(len(snapshot_matrix.columns)) if not snapshot_matrix.empty else 0,
        "live_debug_dir": str(live_debug_dir) if live_debug_dir is not None else None,
        "live_debug_post_transform_features": int(len(live_post_values)),
        "pre_transform_features": int(len(pre_feats)),
        "post_transform_features": int(len(post_feats)),
        "feature_compare_rows": int(len(feature_cmp)),
        "snapshot_post_delta_nonzero_rows": int(
            pd.to_numeric(feature_cmp["abs_snapshot_post_delta"], errors="coerce")
            .fillna(0.0)
            .gt(1e-9)
            .sum()
        ),
        "alpha_lgbm_tree_feature_rows": int((tree_cmp.get("layer", pd.Series(dtype=str)).astype(str) == "alpha_lgbm_tree").sum()) if not tree_cmp.empty else 0,
        "meta_lgbm_tree_feature_rows": int((tree_cmp.get("layer", pd.Series(dtype=str)).astype(str) == "meta_lgbm_tree").sum()) if not tree_cmp.empty else 0,
        "live_replay_layer_compare_rows": int(len(live_replay_layers)),
        "live_replay_layer_delta_nonzero_rows": int(
            pd.to_numeric(live_replay_layers.get("abs_delta", pd.Series(dtype=float)), errors="coerce")
            .fillna(0.0)
            .gt(1e-9)
            .sum()
        ) if not live_replay_layers.empty else 0,
        "live_replay_layer_live_missing_rows": int(
            live_replay_layers.get("live_missing", pd.Series(dtype=bool)).fillna(False).sum()
        ) if not live_replay_layers.empty else 0,
        "live_replay_layer_replay_missing_rows": int(
            live_replay_layers.get("replay_missing", pd.Series(dtype=bool)).fillna(False).sum()
        ) if not live_replay_layers.empty else 0,
        "live_replay_layer_max_abs_delta": _safe_float(
            pd.to_numeric(live_replay_layers.get("abs_delta", pd.Series(dtype=float)), errors="coerce").max()
        ) if not live_replay_layers.empty else float("nan"),
        "meta_model_key": meta_key,
        "outputs": {
            "features": str(out_dir / "feature_pre_post_snapshot_comparison.csv"),
            "lgbm_tree": str(out_dir / "lgbm_leaf_value_features.csv"),
            "live_vs_replay_layers": str(out_dir / "live_vs_replay_model_layers.csv"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("\nTop feature snapshot-vs-replay post-transform deltas:")
    top_cols = [
        "feature",
        "replay_pre_transform",
        "replay_post_transform",
        "snapshot_post_transform",
        "snapshot_minus_replay_post",
    ]
    print(feature_cmp[top_cols].head(30).to_string(index=False))
    print("\nTop absolute LGBM tree values:")
    if not tree_cmp.empty:
        tmp = tree_cmp.copy()
        tmp["abs_value"] = pd.to_numeric(tmp["value"], errors="coerce").abs()
        print(tmp.sort_values("abs_value", ascending=False).head(30).to_string(index=False))
    print("\nTop live-vs-replay model-layer deltas:")
    if not live_replay_layers.empty:
        print(live_replay_layers.head(30).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
