from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from extreme_price_movements.config import CFG as DEFAULT_CFG
from extreme_price_movements.data_store import (
    PartitionedOHLCVStore,
    append_symbol_features,
    load_features_selected,
)
from extreme_price_movements.features import (
    add_regime_gates,
    compute_features_hourly,
    compute_market_features,
)
from extreme_price_movements.hf_data_loader import _load_existing_data
from extreme_price_movements.inference.candidate_selector import (
    _build_mask_for_mode,
    _up_down_zones,
)
from extreme_price_movements.inference.feature_generator import (
    get_inference_required_feature_keys,
    load_or_compute_features,
)
from extreme_price_movements.inference.model_orchestrator import ModelOrchestrator
from extreme_price_movements.model_loader import load_bucket_params, load_model_bundle
from extreme_price_movements.offline_optimisers.params_store import (
    load_inference_candidate_mask_params_per_bucket,
)
from extreme_price_movements.periods_symbols_management import (
    EventSchema,
    SlicePlannerConfig,
)
from extreme_price_movements.policy_optimiser import (
    EPS,
    MAX_DEPLOYMENT_STRATEGIES_PER_SIDE,
    MIN_DEPLOYMENT_AVG_NET_PNL_PER_TRADE,
    _safe_float,
    _strategy_side,
    build_replay_context,
    replay_exit_policy,
)
from extreme_price_movements.regime_adaptor import (
    apply_regime_adaptor,
    load_regime_adaptor,
    safe_strategy_slug,
)
from extreme_price_movements.ridge_position_sizer import (
    prepare_policy_params_from_tpsl_optimiser,
    run_policy_aware_labeling_step,
)
from extreme_price_movements.simple_position_sizer import (
    _select_topk_non_concurrent,
    evaluate_selection_profit_proxy,
)
from extreme_price_movements.universe import get_training_universe
from extreme_price_movements.utils import tprint


def _fill_nonfinite(values: np.ndarray, neutral: float = 0.0) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).copy()
    if arr.size == 0:
        return arr
    finite = np.isfinite(arr)
    if finite.all():
        return arr
    fill = float(np.nanmedian(arr[finite])) if finite.any() else float(neutral)
    arr[~finite] = fill
    return arr


_FEATURE_FRAME_CACHE: dict[tuple[str, str, tuple[str, ...]], pd.DataFrame] = {}
_COMPATIBLE_SYMBOLS_CACHE: dict[tuple[str, str, tuple[str, ...]], List[str]] = {}


def _freeze_path(data_root: str, run_id: str) -> Path:
    return (
        Path(data_root) / "artifacts" / run_id / "ridge_sizer" / "strategy_params.json"
    )


def _load_or_write_freeze_file(data_root: str, run_id: str) -> Dict[str, Any]:
    freeze_path = _freeze_path(data_root, run_id)
    if not freeze_path.exists():
        raise FileNotFoundError(
            f"Frozen strategy params not found at {freeze_path}. Run simple_position_sizer.py first."
        )
    return json.loads(freeze_path.read_text())


def _load_price_panel(symbol: str) -> Tuple[Dict[str, pd.DataFrame], str]:
    ohlcv = _load_existing_data(symbol)
    if ohlcv is None or ohlcv.empty:
        raise FileNotFoundError(f"No cached OHLCV found for {symbol}")

    if isinstance(ohlcv.index, pd.DatetimeIndex) and ohlcv.index.tz is None:
        ohlcv.index = ohlcv.index.tz_localize("UTC")

    panel_symbol = symbol.replace("_", "/")
    panel = {
        "open": pd.DataFrame({panel_symbol: ohlcv["open"].astype(np.float32)}),
        "high": pd.DataFrame({panel_symbol: ohlcv["high"].astype(np.float32)}),
        "low": pd.DataFrame({panel_symbol: ohlcv["low"].astype(np.float32)}),
        "close": pd.DataFrame({panel_symbol: ohlcv["close"].astype(np.float32)}),
        "volume": pd.DataFrame({panel_symbol: ohlcv["volume"].astype(np.float32)}),
    }
    for df in panel.values():
        df.index = pd.DatetimeIndex(ohlcv.index)
    return panel, panel_symbol


def _load_symbol_features(
    data_root: str,
    run_id: str,
    symbol: str,
    panel: Dict[str, pd.DataFrame],
    required_feature_keys: set[str],
    extra_required_keys: Optional[set[str]] = None,
) -> pd.DataFrame:
    effective_required_keys = set(required_feature_keys)
    if extra_required_keys:
        effective_required_keys.update(extra_required_keys)

    cache_key = (run_id, symbol, tuple(sorted(effective_required_keys)))
    cached = _FEATURE_FRAME_CACHE.get(cache_key)
    if cached is not None:
        tprint(
            f"Holdout feature cache hit: symbol={symbol} keys={len(effective_required_keys)} rows={len(cached)} cols={len(cached.columns)}"
        )
        return cached.copy()

    lookback_hours = max(24, int(np.ceil(len(panel["open"]) / 4.0)) + 24)
    feature_cfg = dict(DEFAULT_CFG)
    gated_required = any(
        isinstance(key, str)
        and key
        and (key in {"G_VOL", "G_TREND"} or "_G_VOL_" in key or "_G_TREND_" in key)
        for key in effective_required_keys
    )
    if gated_required:
        feature_cfg["enable_gated_features"] = True
    feature_map = load_or_compute_features(
        panel=panel,
        basket_syms=[symbol],
        run_id=run_id,
        data_root=data_root,
        cfg=feature_cfg,
        lookback_hours=lookback_hours,
        required_feature_keys=effective_required_keys,
    )
    cache_path = (
        Path(data_root)
        / "features"
        / run_id
        / f"symbol={symbol.replace('/', '_')}.parquet"
    )
    if feature_map:
        feat_series: Dict[str, pd.Series] = {}
        for feat_name, feat_series_df in feature_map.items():
            if not isinstance(feat_series_df, pd.DataFrame) or feat_series_df.empty:
                continue
            if symbol in feat_series_df.columns:
                series = feat_series_df[symbol]
            elif feat_series_df.shape[1] == 1:
                series = feat_series_df.iloc[:, 0]
            else:
                continue
            if not isinstance(series, pd.Series) or series.empty:
                continue
            feat_series[feat_name] = series.astype(np.float32)
        feat_df = pd.DataFrame(feat_series) if feat_series else pd.DataFrame()
        missing_required = {
            k for k in effective_required_keys if k not in feat_df.columns
        }
        tprint(
            f"Holdout feature load: symbol={symbol} loaded={len(feat_df.columns)} required={len(effective_required_keys)} missing={len(missing_required)}"
        )
        if missing_required and not feat_df.empty:
            tprint(
                f"Holdout feature backfill: symbol={symbol} missing_keys={len(missing_required)}"
            )
            feat_df = _backfill_missing_feature_columns(
                feature_df=feat_df,
                data_root=data_root,
                run_id=run_id,
                symbol=symbol,
                missing_keys=missing_required,
            )
            feat_df = _backfill_gated_interaction_columns(
                feature_df=feat_df,
                panel=panel,
                panel_symbol=symbol,
                missing_keys=missing_required,
            )
            missing_required = {
                k for k in effective_required_keys if k not in feat_df.columns
            }
        if len(feat_df.columns) >= 20 and not missing_required:
            feat_df = feat_df.sort_index()
            if not isinstance(feat_df.index, pd.DatetimeIndex):
                feat_df.index = pd.to_datetime(feat_df.index, utc=True, errors="coerce")
            if feat_df.index.tz is None:
                feat_df.index = feat_df.index.tz_localize("UTC")
            else:
                feat_df.index = feat_df.index.tz_convert("UTC")
            try:
                append_symbol_features(cache_path.as_posix(), symbol, feat_df)
            except Exception as exc:
                tprint(
                    f"Warning: failed to persist completed holdout features for {symbol}: {exc}"
                )
            tprint(
                f"Holdout feature persist complete: symbol={symbol} rows={len(feat_df)} cols={len(feat_df.columns)}"
            )
            _FEATURE_FRAME_CACHE[cache_key] = feat_df.copy()
            return feat_df
        if missing_required:
            tprint(
                "Selected feature cache incomplete for this strategy contract; "
                f"falling back to full recompute for {len(missing_required)} missing keys"
            )

    # The selected feature cache only carries the lightweight selector set on
    # some runs. Recompute the full alpha feature matrix directly when needed.
    compute_panel = {
        key: df.copy() for key, df in panel.items() if isinstance(df, pd.DataFrame)
    }
    tprint(
        f"Holdout feature full recompute start: symbol={symbol} required={len(effective_required_keys)}"
    )
    mkt_df = compute_market_features(compute_panel, [symbol.replace("_", "/")])
    mkt_gates = add_regime_gates(
        mkt_df,
        gate_vol_lookback_hours=24 * 7,
        gate_trend_thr=0.0,
    )
    full_feats, feat_index, feat_columns = compute_features_hourly(
        compute_panel,
        mkt_gates,
        feature_cfg,
        requested_feature_keys=(
            sorted(effective_required_keys) if effective_required_keys else None
        ),
    )
    feat_series = {}
    for feat_name, feat_value in full_feats.items():
        if isinstance(feat_value, pd.DataFrame):
            feat_series_df = feat_value
        else:
            arr = np.asarray(feat_value)
            if arr.size == 0:
                continue
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            feat_series_df = pd.DataFrame(arr, index=feat_index, columns=feat_columns)
        if feat_series_df.empty or symbol not in feat_series_df.columns:
            continue
        series = feat_series_df[symbol]
        if not isinstance(series, pd.Series) or series.empty:
            continue
        feat_series[feat_name] = series.astype(np.float32)
    feat_df = pd.DataFrame(feat_series) if feat_series else pd.DataFrame()
    if not feat_df.empty:
        feat_df = feat_df.sort_index()
        if not isinstance(feat_df.index, pd.DatetimeIndex):
            feat_df.index = pd.to_datetime(feat_df.index, utc=True, errors="coerce")
        if feat_df.index.tz is None:
            feat_df.index = feat_df.index.tz_localize("UTC")
        else:
            feat_df.index = feat_df.index.tz_convert("UTC")
        try:
            append_symbol_features(cache_path.as_posix(), symbol, feat_df)
        except Exception as exc:
            tprint(
                f"Warning: failed to persist recomputed holdout features for {symbol}: {exc}"
            )
        tprint(
            f"Holdout feature recompute complete: symbol={symbol} rows={len(feat_df)} cols={len(feat_df.columns)}"
        )
        _FEATURE_FRAME_CACHE[cache_key] = feat_df.copy()
        return feat_df

    raise FileNotFoundError(f"No usable features found for {symbol}")


def _infer_strategy_side(
    bundle: Dict[str, Any],
    strategy_id: str,
    explicit_side: str = "",
) -> str:
    side = str(explicit_side or "").strip().lower()
    if side in {"long", "short"}:
        return side
    inferred = _strategy_side({"strategy_id": strategy_id})
    if inferred in {"long", "short"}:
        return inferred
    alpha_models = bundle.get("alpha_models", {}) if isinstance(bundle, dict) else {}
    candidates: set[str] = set()
    for cand_side in ("long", "short"):
        if strategy_id in alpha_models or f"{cand_side}_{strategy_id}" in alpha_models:
            candidates.add(cand_side)
            continue
        base_strategy_id = _base_alpha_strategy_id(strategy_id)
        if (
            base_strategy_id != strategy_id
            and (
                base_strategy_id in alpha_models
                or f"{cand_side}_{base_strategy_id}" in alpha_models
            )
        ):
            candidates.add(cand_side)
            continue
        side_models = alpha_models.get(cand_side, {})
        if isinstance(side_models, dict) and strategy_id in side_models:
            candidates.add(cand_side)
    if len(candidates) == 1:
        return candidates[0]
    return ""


def _base_alpha_strategy_id(strategy_id: str) -> str:
    """Return the alpha strategy id backing a downstream strategy id."""
    sid = str(strategy_id or "")
    if sid.endswith("_tbm"):
        return sid[: -len("_tbm")]
    if sid.endswith("_tbm_clf"):
        return sid[: -len("_tbm_clf")]
    if sid.endswith("_clf"):
        return sid[: -len("_clf")]
    return sid


def _resolve_alpha_strategy_id(
    bundle: Dict[str, Any],
    strategy_id: str,
    side: str,
) -> str:
    """Resolve a sizer/meta strategy id to the loaded alpha model key."""
    alpha_models = bundle.get("alpha_models", {}) if isinstance(bundle, dict) else {}
    if not isinstance(alpha_models, dict):
        return str(strategy_id)

    sid = str(strategy_id)
    side_l = str(side or "").lower()
    candidates = [
        sid,
        f"{side_l}_{sid}" if side_l else sid,
        _base_alpha_strategy_id(sid),
        f"{side_l}_{_base_alpha_strategy_id(sid)}" if side_l else sid,
    ]
    for candidate in candidates:
        if candidate in alpha_models:
            return candidate
    return _base_alpha_strategy_id(sid)


def _backfill_missing_feature_columns(
    feature_df: pd.DataFrame,
    data_root: str,
    run_id: str,
    symbol: str,
    missing_keys: set[str],
) -> pd.DataFrame:
    if not missing_keys or feature_df.empty:
        return feature_df

    ts = pd.to_datetime(run_id, format="%Y%m%d_%H%M%S", utc=True)
    loaded = load_features_selected(
        ts=ts,
        root_dir=data_root,
        feature_keys=sorted(missing_keys),
        symbols=[symbol],
        start_ts=feature_df.index.min(),
        end_ts=feature_df.index.max(),
    )
    if not loaded:
        return feature_df

    updates: Dict[str, pd.Series] = {}
    for key, value in loaded.items():
        if key in feature_df.columns:
            continue
        if isinstance(value, pd.DataFrame):
            if symbol in value.columns:
                series = value[symbol]
            elif value.shape[1] == 1:
                series = value.iloc[:, 0]
            else:
                continue
        elif isinstance(value, pd.Series):
            series = value
        else:
            arr = np.asarray(value)
            if arr.ndim != 1 or len(arr) != len(feature_df.index):
                continue
            series = pd.Series(arr, index=feature_df.index)

        if not isinstance(series, pd.Series) or series.empty:
            continue
        series = series.reindex(feature_df.index)
        if series.notna().any():
            updates[key] = series.astype(np.float32)

    if not updates:
        return feature_df

    return pd.concat(
        [feature_df, pd.DataFrame(updates, index=feature_df.index)], axis=1
    )


_GATED_INTERACTION_RE = re.compile(
    r"^(?P<base>.+)_(?P<gate>G_[A-Z0-9_]+)_(?P<state>[01])$"
)


def _backfill_gated_interaction_columns(
    feature_df: pd.DataFrame,
    panel: Dict[str, pd.DataFrame],
    panel_symbol: str,
    missing_keys: set[str],
) -> pd.DataFrame:
    """Synthesize missing gate and gate-interaction columns generically.

    The alpha bundles for some strategies expect gate-conditioned interaction
    columns such as ``tail_asymmetry_q90_q10_atr_norm_G_VOL_1``. These are
    derived from a base feature multiplied by the corresponding binary gate,
    so they should be rebuilt from the shared feature contract rather than
    patched one by one.
    """
    if feature_df.empty or not missing_keys:
        return feature_df

    needs_gate_cols = any(
        key in {"G_VOL", "G_TREND"} or _GATED_INTERACTION_RE.match(key)
        for key in missing_keys
    )
    if needs_gate_cols:
        compute_panel = {
            key: df.copy() for key, df in panel.items() if isinstance(df, pd.DataFrame)
        }
        mkt_df = compute_market_features(compute_panel, [panel_symbol])
        mkt_gates = add_regime_gates(
            mkt_df,
            gate_vol_lookback_hours=24 * 7,
            gate_trend_thr=0.0,
        )
        gate_updates: Dict[str, pd.Series] = {}
        for gate_name in ("G_VOL", "G_TREND"):
            if gate_name in mkt_gates.columns and gate_name not in feature_df.columns:
                gate_updates[gate_name] = (
                    mkt_gates[gate_name].reindex(feature_df.index).astype(np.float32)
                )
        if gate_updates:
            feature_df = pd.concat(
                [feature_df, pd.DataFrame(gate_updates, index=feature_df.index)], axis=1
            )

    updates: Dict[str, pd.Series] = {}
    for key in sorted(missing_keys):
        if key in feature_df.columns:
            continue
        m = _GATED_INTERACTION_RE.match(key)
        if not m:
            continue
        base = m.group("base")
        gate = m.group("gate")
        state = m.group("state")
        if base not in feature_df.columns or gate not in feature_df.columns:
            continue
        base_vals = feature_df[base].astype(np.float32)
        gate_vals = feature_df[gate].astype(np.float32)
        if state == "1":
            updates[key] = (base_vals * gate_vals).astype(np.float32)
        else:
            updates[key] = (base_vals * (1.0 - gate_vals)).astype(np.float32)

    if not updates:
        return feature_df

    return pd.concat(
        [feature_df, pd.DataFrame(updates, index=feature_df.index)], axis=1
    )


def _build_candidates(
    feature_df: pd.DataFrame,
    panel: Dict[str, pd.DataFrame],
    panel_symbol: str,
    side: str,
    mask: Optional[pd.Series] = None,
) -> pd.DataFrame:
    opens = panel["open"][panel_symbol]
    candidate_ts = pd.DatetimeIndex(feature_df.index)
    lookup_ts = candidate_ts
    if lookup_ts.tz is None:
        lookup_ts = lookup_ts.tz_localize("UTC")
    else:
        lookup_ts = lookup_ts.tz_convert("UTC")
    entry_prices = opens.reindex(lookup_ts, method="bfill")
    valid = np.isfinite(entry_prices.to_numpy(dtype=float))

    if mask is not None:
        valid = valid & mask.to_numpy(dtype=bool)

    if not np.any(valid):
        return pd.DataFrame()
    lookup_ts = lookup_ts[valid]
    entry_prices = entry_prices.iloc[np.flatnonzero(valid)].to_numpy(dtype=np.float64)
    return pd.DataFrame(
        {
            "timestamp": lookup_ts,
            "symbol": [panel_symbol] * len(lookup_ts),
            "is_long": [side == "long"] * len(lookup_ts),
            "entry_price": entry_prices,
        }
    )


def _compute_policy_params(feature_df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
    if "atr_pct" in feature_df.columns:
        atr_val = float(
            np.nanmedian(np.asarray(feature_df["atr_pct"].values, dtype=np.float64))
        )
    elif "prior_volatility" in feature_df.columns:
        atr_val = float(
            np.nanmedian(
                np.asarray(feature_df["prior_volatility"].values, dtype=np.float64)
            )
        )
    else:
        atr_val = 0.02
    atr_val = float(
        np.clip(atr_val if np.isfinite(atr_val) and atr_val > 0 else 0.02, 1e-4, 0.2)
    )

    seed = {
        "tp_mult": 2.0,
        "sl_mult": 1.6,
        "act_n": 0.5,
        "be_act_n": 0.5,
    }
    return prepare_policy_params_from_tpsl_optimiser(seed, atr_values={symbol: atr_val})


def _load_best_policy_params(data_root: str, run_id: str) -> Optional[Dict[str, Any]]:
    for candidate in [
        Path(data_root)
        / "artifacts"
        / run_id
        / "policy_params"
        / "best_policy_params.json",
        Path(data_root) / "artifacts" / run_id / "best_policy_params.json",
    ]:
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text())
            strategies = (
                payload.get("strategies", []) if isinstance(payload, dict) else []
            )
            if strategies:
                return strategies[0]
        except Exception:
            continue
    return None


def _load_strategy_acceptation(data_root: str, run_id: str) -> Dict[str, Any]:
    for candidate in [
        Path(data_root)
        / "artifacts"
        / run_id
        / "policy_params"
        / "strategy_for_inference.json",
        Path(data_root)
        / "artifacts"
        / run_id
        / "policy_params"
        / "strategy_final_acceptation.json",
        Path(data_root) / "artifacts" / run_id / "strategy_final_acceptation.json",
    ]:
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text())
            strategies = (
                payload.get("strategies", []) if isinstance(payload, dict) else []
            )
            accepted_ids = {
                str(row.get("strategy_id", ""))
                for row in strategies
                if isinstance(row, dict) and row.get("strategy_id", "")
            }
            side_by_id = {
                str(row.get("strategy_id", "")): str(row.get("side", "")).lower()
                for row in strategies
                if isinstance(row, dict) and row.get("strategy_id", "")
            }
            return {
                "payload": payload,
                "accepted_ids": accepted_ids,
                "side_by_id": side_by_id,
                "found": True,
            }
        except Exception:
            continue
    return {"payload": None, "accepted_ids": set(), "side_by_id": {}, "found": False}


def _accepted_strategy_feature_keys(
    bundle: Dict[str, Any], accepted_ids: set[str]
) -> set[str]:
    keys: set[str] = set()
    if not accepted_ids:
        return keys
    alpha_models = bundle.get("alpha_models", {}) if isinstance(bundle, dict) else {}
    if not isinstance(alpha_models, dict):
        return keys
    accepted_cores = {_base_alpha_strategy_id(sid) for sid in accepted_ids}
    for sid, model_info in alpha_models.items():
        if not isinstance(model_info, dict):
            continue
        core = _base_alpha_strategy_id(str(sid))
        if core.startswith("long_"):
            core = core[len("long_") :]
        elif core.startswith("short_"):
            core = core[len("short_") :]
        if str(sid) not in accepted_ids and core not in accepted_cores:
            continue
        keys.update(model_info.get("feat_cols", []) or [])
    return keys


def _write_strategy_for_inference(
    data_root: str, run_id: str, holdout_results: List[Dict[str, Any]]
) -> Path:
    """Persist deployable holdout strategies as the explicit inference allowlist."""
    by_strategy: Dict[str, List[Dict[str, Any]]] = {}
    for result in holdout_results:
        for row in result.get("strategies", []) if isinstance(result, dict) else []:
            if not isinstance(row, dict) or not row.get("strategy_id"):
                continue
            by_strategy.setdefault(str(row["strategy_id"]), []).append(row)

    enriched: List[Dict[str, Any]] = []
    for strategy_id, rows in by_strategy.items():
        net = np.asarray([_safe_float(r.get("net_pnl"), 0.0) for r in rows])
        wallet = np.asarray([_safe_float(r.get("wallet_pnl"), 0.0) for r in rows])
        selected_trades = int(sum(int(r.get("selected_trades", 0) or 0) for r in rows))
        n_trades = int(sum(int(r.get("n_trades", 0) or 0) for r in rows))
        avg_net_pnl = float(np.nansum(net) / max(1, selected_trades))
        side = _strategy_side({"strategy_id": strategy_id, "side": rows[0].get("side")})
        trades_per_day = float(
            np.nanmean([_safe_float(r.get("trades_per_day"), 0.0) for r in rows])
        )
        avg_holding_time = max(
            1.0,
            float(
                np.nanmean(
                    [
                        _safe_float(
                            r.get("avg_holding_time_hours", r.get("holding_time_hours")),
                            1.0,
                        )
                        for r in rows
                    ]
                )
            ),
        )
        weekly_wallet_vol = float(np.nanstd(wallet)) if wallet.size > 1 else 0.0
        monthly_wallet_vol = float(
            np.nanmean([_safe_float(r.get("monthly_pnl_std"), 0.0) for r in rows])
        )
        effective_ops_day = math.sqrt(
            max(0.0, min(36.0 / avg_holding_time, max(0.0, trades_per_day)))
        )
        denominator = math.sqrt(avg_holding_time) * math.sqrt(
            max(0.0, weekly_wallet_vol) + max(0.0, monthly_wallet_vol) + EPS
        )
        rank = float(effective_ops_day * avg_net_pnl / max(denominator, EPS))
        best_row = max(rows, key=lambda r: _safe_float(r.get("net_pnl"), 0.0))
        reject_reasons: List[str] = []
        if selected_trades <= 0 or n_trades <= 0:
            reject_reasons.append("no_holdout_trades")
        if avg_net_pnl < MIN_DEPLOYMENT_AVG_NET_PNL_PER_TRADE:
            reject_reasons.append("avg_net_pnl_per_trade_below_0_2pct")
        if side not in {"long", "short"}:
            reject_reasons.append("unknown_side")
        if not np.isfinite(rank):
            reject_reasons.append("non_finite_rank")
        enriched.append(
            {
                "strategy_id": strategy_id,
                "strategy_for_inference": strategy_id,
                "side": side,
                "holdout_net_pnl": float(np.nansum(net)) if net.size else 0.0,
                "holdout_mean_net_pnl": float(np.nanmean(net)) if net.size else 0.0,
                "holdout_wallet_pnl": float(np.nansum(wallet)) if wallet.size else 0.0,
                "holdout_symbols": int(len(rows)),
                "n_trades": n_trades,
                "selected_trades": selected_trades,
                "avg_net_pnl_per_trade": avg_net_pnl,
                "deployment_min_avg_net_pnl_per_trade": (
                    MIN_DEPLOYMENT_AVG_NET_PNL_PER_TRADE
                ),
                "selection_rank": rank,
                "effective_ops_day": effective_ops_day,
                "opportunities_per_day": trades_per_day,
                "avg_holding_time_hours": avg_holding_time,
                "weekly_wallet_vol": weekly_wallet_vol,
                "monthly_wallet_vol": monthly_wallet_vol,
                "profit_factor": _safe_float(best_row.get("profit_factor"), 0.0),
                "threshold_pct": _safe_float(best_row.get("threshold_pct"), 0.0),
                "selection_frac": _safe_float(best_row.get("selection_frac"), 0.0),
                "policy": best_row.get("policy", {}),
                "metrics": {
                    "best_symbol_row": best_row,
                    "symbol_rows": rows,
                },
                "deployment_reject_reasons": reject_reasons,
            }
        )

    candidates = [row for row in enriched if not row["deployment_reject_reasons"]]
    selected: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = [
        row for row in enriched if row["deployment_reject_reasons"]
    ]
    for side in ("long", "short"):
        side_rows = [row for row in candidates if row.get("side") == side]
        side_rows.sort(
            key=lambda r: float(r.get("selection_rank", float("-inf"))),
            reverse=True,
        )
        selected.extend(side_rows[:MAX_DEPLOYMENT_STRATEGIES_PER_SIDE])
        for row in side_rows[MAX_DEPLOYMENT_STRATEGIES_PER_SIDE:]:
            row = dict(row)
            row["deployment_reject_reasons"] = ["outside_top_2_per_side"]
            rejected.append(row)

    selected.sort(
        key=lambda r: (
            str(r.get("side", "")),
            -float(r.get("selection_rank", float("-inf"))),
        )
    )
    payload = {
        "schema_version": "v2",
        "generated_by": "holdout_strategy_eval",
        "run_id": run_id,
        "selection_rules": {
            "min_avg_net_pnl_per_trade": MIN_DEPLOYMENT_AVG_NET_PNL_PER_TRADE,
            "max_strategies_per_side": MAX_DEPLOYMENT_STRATEGIES_PER_SIDE,
            "rank_formula": (
                "effective_ops_day * avg_net_pnl_per_trade / "
                "(sqrt(avg_holding_time_hours) * "
                "sqrt(weekly_wallet_vol + monthly_wallet_vol + eps))"
            ),
            "effective_ops_day_formula": (
                "sqrt(min(36 / avg_holding_time_hours, opportunities_per_day))"
            ),
        },
        "strategies": selected,
        "rejected_strategies": [
            {
                **row,
                "selected": False,
                "reject_reasons": row.get("deployment_reject_reasons", []),
            }
            for row in rejected
        ],
    }
    path = Path(data_root) / "artifacts" / run_id / "strategy_for_inference.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def _apply_policy_params_to_seed(
    policy: Dict[str, Any], symbol: str, atr_val: float
) -> Dict[str, Any]:
    tp_mult = float(policy.get("tp_mult", 2.0))
    sl_mult = float(policy.get("sl_mult", 1.6))
    trail_activation = float(policy.get("trail_activation_atr", 0.8))
    trail_giveback = float(policy.get("trail_giveback_atr", 0.3))
    trailing_pct = max(0.01, trail_activation - trail_giveback)

    result = {
        "tp_mult": tp_mult,
        "sl_mult": sl_mult,
        "trailing_pct": trailing_pct,
        "atr": {symbol: atr_val},
        "source": "policy_optimiser",
    }
    passthrough = [
        "strategy_id",
        "k_recent",
        "K_early",
        "theta_fail",
        "theta_path",
        "d_path",
        "progress_threshold",
        "lambda_path",
        "a1",
        "a2",
        "b1",
        "b2",
        "compression_start",
        "compression_full",
        "compression_max_fraction",
        "trail_activation_atr",
        "trail_giveback_atr",
        "continuation_conf_threshold",
        "multiplier_band_min",
        "multiplier_band_max",
        "score_weight_trend",
        "score_weight_asym",
        "score_weight_choppiness",
    ]
    for key in passthrough:
        if key in policy:
            result[key] = policy[key]
    return result


def evaluate_holdout_symbol(
    data_root: str,
    run_id: str,
    symbol: str,
    freeze: Dict[str, Any],
    *,
    extra_required_keys: Optional[set[str]] = None,
    bundle: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if bundle is None:
        bundle = load_model_bundle(run_id, data_root)
    orchestrator = ModelOrchestrator(bundle, {"disable_spike_filter": True})
    tprint(
        f"Holdout eval: symbol={symbol} run_id={run_id} strategies={len(freeze.get('strategies', []))}"
    )

    panel, panel_symbol = _load_price_panel(symbol)
    required_feature_keys = get_inference_required_feature_keys(bundle)
    feature_df = _load_symbol_features(
        data_root,
        run_id,
        symbol=panel_symbol,
        panel=panel,
        required_feature_keys=required_feature_keys,
        extra_required_keys=extra_required_keys,
    )
    alpha_required_keys = set()
    alpha_models = bundle.get("alpha_models", {})
    if isinstance(alpha_models, dict):
        for model_info in alpha_models.values():
            if isinstance(model_info, dict) and (
                "feat_cols" in model_info or "model" in model_info
            ):
                alpha_required_keys.update(model_info.get("feat_cols", []) or [])
                continue
            if isinstance(model_info, dict):
                for nested_info in model_info.values():
                    if isinstance(nested_info, dict):
                        alpha_required_keys.update(
                            nested_info.get("feat_cols", []) or []
                        )
    missing_alpha_keys = {k for k in alpha_required_keys if k not in feature_df.columns}
    if missing_alpha_keys:
        tprint(
            f"Holdout feature contract warning: {len(missing_alpha_keys)} alpha features "
            f"missing for {panel_symbol}; prediction will use model contract alignment. "
            f"Missing sample: {sorted(missing_alpha_keys)[:10]}"
        )
    tprint(
        f"Holdout eval feature contract satisfied: symbol={symbol} features={len(feature_df.columns)} alpha_keys={len(alpha_required_keys)}"
    )
    frozen_rows: List[Dict[str, Any]] = []
    portfolio_score_parts: List[np.ndarray] = []
    portfolio_return_parts: List[np.ndarray] = []
    portfolio_ts_parts: List[np.ndarray] = []

    policy_params = _compute_policy_params(feature_df, panel_symbol)

    best_policy = _load_best_policy_params(data_root, run_id)
    if best_policy is not None:
        if "atr_pct" in feature_df.columns:
            atr_val = float(
                np.nanmedian(np.asarray(feature_df["atr_pct"].values, dtype=np.float64))
            )
        else:
            atr_val = 0.02
        atr_val = float(
            np.clip(
                atr_val if np.isfinite(atr_val) and atr_val > 0 else 0.02,
                1e-4,
                0.2,
            )
        )
        policy_params = _apply_policy_params_to_seed(best_policy, panel_symbol, atr_val)
        tprint(
            f"  Loaded policy params: tp={policy_params['tp_mult']:.2f} "
            f"sl={policy_params['sl_mult']:.2f} "
            f"trailing={policy_params.get('trailing_pct', 0):.2f} "
            f"rule={policy_params.get('better_rule', 'N/A')}"
        )

    # Load per-strategy bucket params
    bucket_params = load_bucket_params(run_id, data_root)

    # 1. Load active strategy masks
    tprint("Loading inference candidate mask params...")
    dyn_strategies = load_inference_candidate_mask_params_per_bucket(top_n=99)
    mask_params_by_mode = {}
    for s in dyn_strategies:
        sid = s.get("strategy_id")
        m_params = s.get("mask_params", {})
        if sid and m_params:
            mask_params_by_mode[sid] = m_params

    # 2. Precompute up/down zones
    # Minimal feats needed for up_down_zones

    # We need to construct a mini-feats dict with panel symbols
    # Since up_down_zones expects full panel formatting, we simulate it
    sim_feats = {}
    for k in feature_df.columns:
        sim_feats[k] = pd.DataFrame({panel_symbol: feature_df[k]})

    up_zone, down_zone = _up_down_zones(sim_feats, panel, metric="ret12h")

    # 3. Create walk-forward unseen mask
    # We generate a dummy events df to use SlicePlanner
    dummy_events = pd.DataFrame(
        {
            "event_id": range(len(feature_df)),
            "symbol": panel_symbol,
            "t0": feature_df.index,
            "t1": feature_df.index + pd.Timedelta(hours=1),
        }
    )
    planner_cfg = SlicePlannerConfig.robust_defaults(schema=EventSchema())
    from extreme_price_movements.inference_backtest import _build_unseen_mask

    unseen_mask_np = _build_unseen_mask(dummy_events, planner_cfg)
    unseen_mask = pd.Series(unseen_mask_np, index=feature_df.index)
    if not bool(unseen_mask.any()):
        tprint(
            f"Holdout eval: planner returned no unseen rows for {symbol}; "
            "using all rows because this symbol was selected as cross-asset OOS."
        )
        unseen_mask = pd.Series(True, index=feature_df.index)

    strategies_list = freeze.get("strategies", [])
    total_strategies = len(strategies_list)
    tprint_interval = max(1, total_strategies // 10)

    for i, strat in enumerate(strategies_list):
        if i % tprint_interval == 0:
            tprint(
                f"Holdout eval progress: {i}/{total_strategies} strategies ({i/total_strategies*100:.1f}%) for symbol {symbol}"
            )

        strategy_id = str(strat["strategy_id"])
        side = _infer_strategy_side(bundle, strategy_id, str(strat.get("side", "")))
        if not side:
            tprint(
                f"Skipping strategy {strategy_id[:60]}: could not infer side from freeze file or alpha bundle"
            )
            continue
        tprint(
            f"Holdout eval strategy start: symbol={symbol} strategy_id={strategy_id} side={side} threshold_pct={strat.get('threshold_pct')}"
        )
        threshold_pct = float(strat["threshold_pct"])
        frac = max(1e-6, 1.0 - threshold_pct / 100.0)

        # Build specific mask for this strategy
        strat_mask = unseen_mask.copy()
        if strategy_id in mask_params_by_mode:
            mask_cfg = mask_params_by_mode[strategy_id]
            per_mode_df = _build_mask_for_mode(panel, sim_feats, mask_cfg)
            if panel_symbol in per_mode_df.columns:
                per_mode_series = per_mode_df[panel_symbol]

                # Apply up/down zone logic based on side
                if side == "long":
                    # For long, we use up_zone (if TF) or down_zone (if MR)
                    # We can infer TF/MR from strategy_id usually, but let's check
                    if "tf" in strategy_id.lower() and panel_symbol in up_zone.columns:
                        strat_mask = (
                            strat_mask & up_zone[panel_symbol] & per_mode_series
                        )
                    elif (
                        "mr" in strategy_id.lower()
                        and panel_symbol in down_zone.columns
                    ):
                        strat_mask = (
                            strat_mask & down_zone[panel_symbol] & per_mode_series
                        )
                    else:
                        strat_mask = strat_mask & per_mode_series
                else:  # short
                    if (
                        "tf" in strategy_id.lower()
                        and panel_symbol in down_zone.columns
                    ):
                        strat_mask = (
                            strat_mask & down_zone[panel_symbol] & per_mode_series
                        )
                    elif (
                        "mr" in strategy_id.lower() and panel_symbol in up_zone.columns
                    ):
                        strat_mask = (
                            strat_mask & up_zone[panel_symbol] & per_mode_series
                        )
                    else:
                        strat_mask = strat_mask & per_mode_series

        strat_policy_params = policy_params.copy()
        if strategy_id in bucket_params:
            strat_policy_params.update(bucket_params[strategy_id])
        elif best_policy is not None:
            strat_policy_params.update(best_policy)

        candidates = _build_candidates(
            feature_df, panel, panel_symbol, side=side, mask=strat_mask
        )
        if candidates.empty:
            tprint(
                f"Holdout eval strategy skipped: symbol={symbol} strategy_id={strategy_id} "
                "reason=no_candidates"
            )
            continue

        outcomes = run_policy_aware_labeling_step(
            candidates,
            panel,
            strat_policy_params,
            max_hold_hours=24,
            cost_pct=0.0,
            bars_per_hour=4,
            use_batch=True,
        )
        if outcomes.empty or "label" not in outcomes.columns:
            tprint(
                f"Holdout eval strategy skipped: symbol={symbol} strategy_id={strategy_id} "
                "reason=no_policy_outcomes"
            )
            continue

        alpha_strategy_id = _resolve_alpha_strategy_id(bundle, strategy_id, side)
        score_series = orchestrator.predict_alpha(feature_df, side, alpha_strategy_id)
        if score_series.empty:
            tprint(
                f"Holdout eval strategy skipped: symbol={symbol} strategy_id={strategy_id} "
                f"alpha_strategy_id={alpha_strategy_id} reason=no_alpha_scores"
            )
            continue

        aligned_scores = score_series.reindex(
            pd.DatetimeIndex(outcomes["timestamp"].values)
        )
        score_values = _fill_nonfinite(aligned_scores.to_numpy(dtype=np.float64))
        adaptor_path = (
            Path(data_root)
            / "artifacts"
            / run_id
            / "ridge_sizer"
            / "regime_adaptors"
            / safe_strategy_slug(strategy_id)
            / "regime_adaptor.json"
        )
        if adaptor_path.exists():
            try:
                adaptor = load_regime_adaptor(adaptor_path)
                if bool(adaptor.get("enable_regime_adaptor", False)):
                    feature_rows = feature_df.reindex(
                        pd.DatetimeIndex(outcomes["timestamp"].values)
                    )
                    applied = apply_regime_adaptor(
                        feature_rows.reset_index(drop=True),
                        score_values,
                        adaptor,
                        timestamps=np.asarray(outcomes["timestamp"].values),
                        symbols=np.repeat(panel_symbol, len(score_values)),
                    )
                    score_values = np.asarray(
                        applied["deployment_score_rank"], dtype=np.float64
                    )
                    eligible = np.asarray(applied["eligible"], dtype=bool)
                    if not bool(eligible.any()):
                        tprint(
                            f"Holdout eval strategy skipped: symbol={symbol} "
                            f"strategy_id={strategy_id} reason=all_regime_gated"
                        )
                        continue
            except Exception as exc:
                tprint(
                    f"Warning: holdout regime adaptor failed for "
                    f"{strategy_id[:60]}: {exc}"
                )
        raw_returns = np.asarray(outcomes["label"].values, dtype=np.float64)
        ts_values = np.asarray(outcomes["timestamp"].values)

        if best_policy is not None:
            mfe = np.asarray(
                outcomes.get("mfe_ret", np.abs(raw_returns)), dtype=np.float32
            )
            mae = np.asarray(
                outcomes.get("mae_ret", np.abs(np.minimum(raw_returns, 0.0))),
                dtype=np.float32,
            )
            bars = np.asarray(
                outcomes.get("exit_bar", np.full(len(raw_returns), 4)),
                dtype=np.int32,
            )
            barrier = np.maximum(mae * 2.5, 1e-4)
            context = build_replay_context(
                returns=raw_returns.astype(np.float32),
                mfe_ret=mfe,
                mae_ret=mae,
                bars_since_entry=bars,
                barrier_pct=barrier,
                confidence=score_values.astype(np.float32),
            )
            raw_returns = replay_exit_policy(
                raw_returns.astype(np.float32), context, strat_policy_params
            ).astype(np.float64)

        t_diff = pd.to_datetime(ts_values.max()) - pd.to_datetime(ts_values.min())
        n_days = float(t_diff / np.timedelta64(1, "D")) if len(ts_values) > 1 else 0.0

        metrics_df, opt_rets, opt_ts = evaluate_selection_profit_proxy(
            score_values,
            raw_returns,
            timestamps=ts_values,
            top_fracs=[frac],
            cost_pct=0.003,
            n_days=n_days,
        )
        row = metrics_df.iloc[0].to_dict() if not metrics_df.empty else {}
        row.update(
            {
                "strategy_id": strat["strategy_id"],
                "side": side,
                "threshold_pct": threshold_pct,
                "selection_frac": frac,
                "n_trades": int(len(raw_returns)),
                "selected_trades": int(len(opt_rets)),
            }
        )
        frozen_rows.append(row)
        tprint(
            f"Holdout eval strategy done: symbol={symbol} strategy_id={strategy_id} "
            f"selected={int(len(opt_rets))}/{int(len(raw_returns))} net_pnl={float(row.get('net_pnl', 0.0)):.4f}"
        )

        if float(row.get("net_pnl", 0.0)) > 0:
            k = max(1, int(len(score_values) * frac))
            idx = _select_topk_non_concurrent(
                scores=score_values,
                k=k,
                timestamps=ts_values,
                symbols=np.full(len(score_values), symbol, dtype=object),
                horizon_hours=4.0,
                max_global_concurrent=3,
            )
            portfolio_score_parts.append(score_values[idx])
            portfolio_return_parts.append(raw_returns[idx])
            portfolio_ts_parts.append(ts_values[idx])

    portfolio_row: Dict[str, Any] = {}
    if portfolio_score_parts:
        portfolio_scores = np.concatenate(portfolio_score_parts)
        portfolio_returns = np.concatenate(portfolio_return_parts)
        portfolio_ts = np.concatenate(portfolio_ts_parts)
        portfolio_days = (
            float(
                (
                    pd.to_datetime(portfolio_ts.max())
                    - pd.to_datetime(portfolio_ts.min())
                )
                / np.timedelta64(1, "D")
            )
            if len(portfolio_ts) > 1
            else 0.0
        )
        portfolio_metrics, _, _ = evaluate_selection_profit_proxy(
            portfolio_scores,
            portfolio_returns,
            timestamps=portfolio_ts,
            top_fracs=[1.0],
            cost_pct=0.003,
            n_days=portfolio_days,
        )
        if not portfolio_metrics.empty:
            portfolio_row = portfolio_metrics.iloc[0].to_dict()

    return {
        "symbol": symbol,
        "run_id": run_id,
        "fee_pct": 0.003,
        "strategies": frozen_rows,
        "portfolio": portfolio_row,
    }


def _pick_best_model_per_strategy(
    freeze: Dict[str, Any], force_model: str = ""
) -> Dict[str, Any]:
    """Auto-pick the best model (ridge vs et) per strategy_id from freeze data, or force a specific one."""
    data_root = freeze.get("_data_root_", "data")
    run_id = freeze.get("_run_id_", "")

    per_strategy_winner: Dict[str, str] = {}
    comparison_path = (
        Path(data_root)
        / "artifacts"
        / run_id
        / "ridge_sizer"
        / "head_to_head_comparison.json"
    )
    if comparison_path.exists():
        try:
            comparisons = json.loads(comparison_path.read_text())
            if isinstance(comparisons, list):
                for row in comparisons:
                    per_strategy_winner[row.get("strategy_id", "")] = row.get(
                        "winner", "ridge"
                    )
                et_count = sum(1 for w in per_strategy_winner.values() if w == "et")
                tprint(
                    f"Loaded head-to-head comparison: {len(per_strategy_winner)} strategies, {et_count} ET winners"
                )
        except Exception:
            pass

    et_freeze_path = (
        Path(data_root) / "artifacts" / run_id / "et_sizer" / "strategy_params.json"
    )
    et_strategies: Dict[str, Dict] = {}
    if et_freeze_path.exists():
        try:
            et_freeze = json.loads(et_freeze_path.read_text())
            for s in et_freeze.get("strategies", []):
                et_strategies[s.get("strategy_id", "")] = s
        except Exception:
            pass

    best_freeze = dict(freeze)
    merged_strategies = []
    for strat in best_freeze.get("strategies", []):
        sid = strat.get("strategy_id", "")

        # Determine winner based on force_model flag or head-to-head comparison
        if force_model == "ridge":
            winner = "ridge"
        elif force_model == "et":
            winner = "et"
        else:
            winner = per_strategy_winner.get(sid, "ridge")

        if winner == "et" and sid in et_strategies:
            merged = dict(strat)
            merged.update(et_strategies[sid])
            merged["model_source"] = "et"
            merged_strategies.append(merged)
        else:
            merged = dict(strat)
            merged["model_source"] = "ridge"
            merged_strategies.append(merged)

    if merged_strategies:
        best_freeze["strategies"] = merged_strategies

    best_freeze["model_source"] = (
        force_model if force_model else ("mixed" if per_strategy_winner else "ridge")
    )
    return best_freeze


def _sample_random_symbols(n: int = 5, seed: int = 42) -> List[str]:
    """Sample n random symbols from the training universe."""
    rng = np.random.RandomState(seed)
    try:
        universe = get_training_universe()
        symbols = (
            list(universe.keys()) if isinstance(universe, dict) else list(universe)
        )
    except Exception:
        symbols = ["AIXBT_USDC"]
    if len(symbols) <= n:
        return symbols
    return list(rng.choice(symbols, size=n, replace=False))


def _alpha_required_feature_keys(bundle: Dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    alpha_models = bundle.get("alpha_models", {})
    if not isinstance(alpha_models, dict):
        return keys
    for model_info in alpha_models.values():
        if not isinstance(model_info, dict):
            continue
        if "feat_cols" in model_info or "model" in model_info:
            keys.update(model_info.get("feat_cols", []) or [])
            continue
        for nested_info in model_info.values():
            if isinstance(nested_info, dict):
                keys.update(nested_info.get("feat_cols", []) or [])
    return keys


def _normalise_symbol_key(symbol: Any) -> str:
    return str(symbol or "").strip().replace("/", "_").upper()


def _load_cross_asset_oos_candidates(
    data_root: str, run_id: str, universe_symbols: List[str]
) -> List[str]:
    """Return symbols not present in train/meta/sizer artifacts for OOS holdout."""
    art_dir = Path(data_root) / "artifacts" / run_id
    basket_path = art_dir / "oos_eval_basket.json"
    if basket_path.exists():
        try:
            payload = json.loads(basket_path.read_text())
            oos_symbols = [str(s) for s in payload.get("oos", []) if str(s)]
            if oos_symbols:
                tprint(
                    f"Holdout sampler using explicit OOS basket with {len(oos_symbols)} symbols."
                )
                return oos_symbols
        except Exception:
            pass

    seen_symbols: set[str] = set()
    parquet_paths: List[Path] = []
    for rel in ("base_oof", "meta_oof"):
        folder = art_dir / rel
        if folder.exists():
            parquet_paths.extend(sorted(folder.glob("*.parquet")))
    sizer_oof = art_dir / "oof" / "simple_sizer_oof_all.parquet"
    if sizer_oof.exists():
        parquet_paths.append(sizer_oof)

    for path in parquet_paths:
        try:
            frame = pd.read_parquet(path, columns=["symbol"])
        except Exception:
            continue
        seen_symbols.update(_normalise_symbol_key(s) for s in frame["symbol"].dropna())

    if not seen_symbols:
        tprint(
            "Holdout sampler could not infer in-sample artifact symbols; "
            "using full compatible universe."
        )
        return universe_symbols

    oos = [
        symbol
        for symbol in universe_symbols
        if _normalise_symbol_key(symbol) not in seen_symbols
    ]
    tprint(
        f"Holdout sampler excluded {len(seen_symbols)} artifact-seen symbols; "
        f"{len(oos)}/{len(universe_symbols)} symbols remain cross-asset OOS."
    )
    if not oos:
        raise RuntimeError(
            "No cross-asset OOS symbols remain after excluding train/meta/sizer artifacts."
        )
    return oos


def _sample_compatible_symbols(
    data_root: str,
    run_id: str,
    n: int = 5,
    seed: int = 42,
    extra_required_keys: Optional[set[str]] = None,
    cache_key: Optional[tuple[str, str, tuple[str, ...]]] = None,
    bundle: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Sample symbols that satisfy the alpha feature contract."""
    tprint(
        f"Holdout sampler: requesting up to {n} compatible symbols for run_id={run_id}"
    )
    if cache_key is not None:
        cached = _COMPATIBLE_SYMBOLS_CACHE.get(cache_key)
        if cached is not None:
            tprint(
                f"Holdout sampler cache hit: {len(cached)} compatible symbols already cached"
            )
            return cached[:n]
        cache_path = (
            Path(data_root)
            / "artifacts"
            / run_id
            / "policy_params"
            / "holdout_compatible_symbols.json"
        )
        if cache_path.exists():
            try:
                payload = json.loads(cache_path.read_text())
                if (
                    payload.get("run_id") == run_id
                    and tuple(payload.get("feature_keys", [])) == cache_key[2]
                ):
                    symbols = [str(s) for s in payload.get("symbols", []) if str(s)]
                    if symbols:
                        _COMPATIBLE_SYMBOLS_CACHE[cache_key] = symbols
                        tprint(
                            f"Holdout sampler disk cache hit: {len(symbols)} compatible symbols"
                        )
                        return symbols[:n]
            except Exception:
                pass

    rng = np.random.RandomState(seed)
    try:
        cfg = dict(DEFAULT_CFG)
        cfg["data_root"] = data_root
        store = PartitionedOHLCVStore(
            root_dir=data_root, timeframe=str(cfg.get("timeframe", "1h"))
        )
        universe = get_training_universe(None, cfg, store, ts_sig=None)
        symbols = (
            list(universe.keys()) if isinstance(universe, dict) else list(universe)
        )
    except Exception:
        symbols = ["AIXBT_USDT"]

    if len(symbols) == 0:
        raise RuntimeError("No symbols available for holdout evaluation")

    symbols = _load_cross_asset_oos_candidates(data_root, run_id, symbols)

    rng.shuffle(symbols)
    if bundle is None:
        bundle = load_model_bundle(run_id, data_root)
    required_feature_keys = get_inference_required_feature_keys(bundle)
    alpha_required_keys = _alpha_required_feature_keys(bundle)

    compatible: List[str] = []
    rejected: List[str] = []

    for i, symbol in enumerate(symbols):
        if i > 0 and i % 10 == 0:
            tprint(
                f"Holdout sampler progress: checked {i}/{len(symbols)} symbols, found {len(compatible)} compatible"
            )

        try:
            panel, panel_symbol = _load_price_panel(symbol)
            feature_df = _load_symbol_features(
                data_root,
                run_id,
                symbol=panel_symbol,
                panel=panel,
                required_feature_keys=required_feature_keys,
                extra_required_keys=extra_required_keys,
            )
            missing_alpha_keys = {
                k for k in alpha_required_keys if k not in feature_df.columns
            }
            if missing_alpha_keys:
                rejected.append(symbol)
                continue
            compatible.append(symbol)
            if len(compatible) >= n:
                break
        except Exception as exc:
            rejected.append(symbol)
            tprint(f"Holdout sampler skipped {symbol}: {exc}")
            continue

    if not compatible:
        raise RuntimeError(
            "No holdout symbols satisfied the alpha feature contract. "
            f"Checked {len(symbols)} symbols, rejected {len(rejected)}."
        )
    if len(compatible) < n:
        tprint(
            f"Holdout sampler found only {len(compatible)} compatible symbols "
            f"out of requested {n}; proceeding with the available set."
        )
    tprint(
        f"Holdout sampler selected {len(compatible)} compatible symbols "
        f"(rejected={len(rejected)})"
    )
    if cache_key is not None:
        _COMPATIBLE_SYMBOLS_CACHE[cache_key] = compatible[:]
        cache_path = (
            Path(data_root)
            / "artifacts"
            / run_id
            / "policy_params"
            / "holdout_compatible_symbols.json"
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "feature_keys": list(cache_key[2]),
                    "symbols": compatible[:],
                },
                indent=2,
            )
        )
        tprint(
            f"Holdout sampler cache saved: {len(compatible)} symbols to {cache_path}"
        )
    return compatible[:n]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen strategy thresholds on holdout symbols."
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--run-id", default="20260321_140000")
    parser.add_argument(
        "--symbol", default="", help="Single symbol to evaluate (overrides --n-symbols)"
    )
    parser.add_argument(
        "--n-symbols", type=int, default=5, help="Number of random symbols to evaluate"
    )
    parser.add_argument("--freeze-json", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--seed", type=int, default=42)

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--ridge", action="store_true", help="Force use of Ridge model only"
    )
    group.add_argument(
        "--et", action="store_true", help="Force use of ExtraTrees model only"
    )
    group.add_argument(
        "--compare", action="store_true", help="Auto-pick best model (default)"
    )

    args = parser.parse_args()

    freeze = (
        json.loads(Path(args.freeze_json).read_text())
        if args.freeze_json
        else _load_or_write_freeze_file(args.data_root, args.run_id)
    )

    force_model = "ridge" if args.ridge else ("et" if args.et else "")
    freeze = _pick_best_model_per_strategy(freeze, force_model=force_model)
    acceptance = _load_strategy_acceptation(args.data_root, args.run_id)
    bundle = load_model_bundle(args.run_id, args.data_root)
    accepted_feature_keys: set[str] = set()
    if acceptance.get("found", False):
        accepted_ids = acceptance.get("accepted_ids", set())
        side_by_id = acceptance.get("side_by_id", {}) or {}
        accepted_feature_keys = _accepted_strategy_feature_keys(bundle, accepted_ids)
        freeze_strats = freeze.get("strategies", [])
        if isinstance(freeze_strats, list):
            filtered = []
            for strat in freeze_strats:
                sid = str(strat.get("strategy_id", ""))
                if sid not in accepted_ids:
                    continue
                enriched = dict(strat)
                policy_side = str(side_by_id.get(sid, "")).lower()
                if policy_side in {"long", "short"}:
                    enriched["side"] = policy_side
                filtered.append(enriched)
            skipped = len(freeze_strats) - len(filtered)
            freeze["strategies"] = filtered
            tprint(
                f"Acceptance filter kept {len(filtered)} strategies and skipped {skipped}"
            )
            if not filtered:
                tprint(
                    "No strategies passed the final acceptance gates; holdout evaluation will skip all strategies."
                )
    else:
        tprint(
            "No strategy acceptance file found; holdout evaluator will use all frozen strategies."
        )

    cache_key = (
        args.run_id,
        force_model or "mixed",
        tuple(sorted(accepted_feature_keys)),
    )
    if args.symbol:
        symbols = [args.symbol]
    else:
        symbols = _sample_compatible_symbols(
            args.data_root,
            args.run_id,
            n=args.n_symbols,
            seed=args.seed,
            extra_required_keys=accepted_feature_keys,
            cache_key=cache_key,
            bundle=bundle,
        )
        tprint(f"Sampled {len(symbols)} compatible holdout symbols: {symbols}")

    all_results: List[Dict[str, Any]] = []
    pass

    for i, symbol in enumerate(symbols):
        tprint(f"Starting evaluation for symbol {i+1}/{len(symbols)}: {symbol}")
        try:
            result = evaluate_holdout_symbol(
                args.data_root,
                args.run_id,
                symbol,
                freeze,
                extra_required_keys=accepted_feature_keys,
                bundle=bundle,
            )
            all_results.append(result)
            if result.get("portfolio"):
                tprint(
                    f"  {symbol}: portfolio wallet_pnl={result['portfolio'].get('wallet_pnl', 'N/A')}"
                )
        except Exception as e:
            tprint(f"  {symbol}: SKIPPED ({e})")

    output_path = (
        Path(args.output_json)
        if args.output_json
        else Path(args.data_root)
        / "artifacts"
        / args.run_id
        / "ridge_sizer"
        / "holdout_multi_metrics.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(all_results, indent=2, default=str))
    tprint(f"Wrote holdout metrics for {len(all_results)} symbols to {output_path}")
    inference_strategy_path = _write_strategy_for_inference(
        args.data_root, args.run_id, all_results
    )
    tprint(f"Wrote holdout-selected inference strategies to {inference_strategy_path}")

    print(json.dumps(all_results, indent=2, default=str))


if __name__ == "__main__":
    main()
