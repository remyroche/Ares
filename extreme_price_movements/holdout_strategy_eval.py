from __future__ import annotations

import argparse
import json
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
from extreme_price_movements.inference.feature_generator import (
    get_inference_required_feature_keys,
    load_or_compute_features,
)
from extreme_price_movements.inference.model_orchestrator import ModelOrchestrator
from extreme_price_movements.model_loader import load_model_bundle
from extreme_price_movements.ridge_position_sizer import (
    prepare_policy_params_from_tpsl_optimiser,
    run_policy_aware_labeling_step,
)
from extreme_price_movements.simple_position_sizer import (
    evaluate_selection_profit_proxy,
)
from extreme_price_movements.policy_optimiser import (
    build_replay_context,
    replay_exit_policy,
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
                feat_df=feat_df,
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
        requested_feature_keys=sorted(effective_required_keys)
        if effective_required_keys
        else None,
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
    alpha_models = bundle.get("alpha_models", {}) if isinstance(bundle, dict) else {}
    candidates: List[str] = []
    for cand_side in ("long", "short"):
        side_models = alpha_models.get(cand_side, {})
        if isinstance(side_models, dict) and strategy_id in side_models:
            candidates.append(cand_side)
    if len(candidates) == 1:
        return candidates[0]
    return ""


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
        / "strategy_final_acceptation.json",
        Path(data_root) / "artifacts" / run_id / "strategy_final_acceptation.json",
    ]:
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text())
            strategies = payload.get("strategies", []) if isinstance(payload, dict) else []
            accepted_ids = {
                str(row.get("strategy_id", ""))
                for row in strategies
                if isinstance(row, dict) and row.get("strategy_id", "")
            }
            return {
                "payload": payload,
                "accepted_ids": accepted_ids,
                "found": True,
            }
        except Exception:
            continue
    return {"payload": None, "accepted_ids": set(), "found": False}


def _accepted_strategy_feature_keys(
    bundle: Dict[str, Any], accepted_ids: set[str]
) -> set[str]:
    keys: set[str] = set()
    if not accepted_ids:
        return keys
    alpha_models = bundle.get("alpha_models", {}) if isinstance(bundle, dict) else {}
    for side_models in alpha_models.values():
        if not isinstance(side_models, dict):
            continue
        for sid, model_info in side_models.items():
            if sid not in accepted_ids or not isinstance(model_info, dict):
                continue
            keys.update(model_info.get("feat_cols", []) or [])
    return keys


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
    tprint(f"Holdout eval: symbol={symbol} run_id={run_id} strategies={len(freeze.get('strategies', []))}")

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
    for side_models in bundle.get("alpha_models", {}).values():
        if not isinstance(side_models, dict):
            continue
        for model_info in side_models.values():
            if isinstance(model_info, dict):
                alpha_required_keys.update(model_info.get("feat_cols", []) or [])
    missing_alpha_keys = {k for k in alpha_required_keys if k not in feature_df.columns}
    if missing_alpha_keys:
        raise FileNotFoundError(
            f"Feature contract violation: {len(missing_alpha_keys)} required alpha features "
            f"missing for {panel_symbol}. Missing: {sorted(missing_alpha_keys)[:10]}. "
            f"Available: {len(feature_df.columns)}. Re-run feature generation."
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

    for strat in freeze.get("strategies", []):
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

        candidates = _build_candidates(feature_df, panel, panel_symbol, side=side)
        if candidates.empty:
            continue

        outcomes = run_policy_aware_labeling_step(
            candidates,
            panel,
            policy_params,
            max_hold_hours=24,
            cost_pct=0.0,
            bars_per_hour=4,
            use_batch=True,
        )
        if outcomes.empty or "label" not in outcomes.columns:
            continue

        score_series = orchestrator.predict_alpha(feature_df, side, strategy_id)
        if score_series.empty:
            continue

        aligned_scores = score_series.reindex(
            pd.DatetimeIndex(outcomes["timestamp"].values)
        )
        score_values = _fill_nonfinite(aligned_scores.to_numpy(dtype=np.float64))
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
                raw_returns.astype(np.float32), context, best_policy
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
            idx = np.argpartition(score_values, -k)[-k:]
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
    for side_models in bundle.get("alpha_models", {}).values():
        if not isinstance(side_models, dict):
            continue
        for model_info in side_models.values():
            if not isinstance(model_info, dict):
                continue
            keys.update(model_info.get("feat_cols", []) or [])
    return keys


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
                    and tuple(payload.get("feature_keys", []))
                    == cache_key[2]
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
        symbols = list(universe.keys()) if isinstance(universe, dict) else list(universe)
    except Exception:
        symbols = ["AIXBT_USDT"]

    if len(symbols) == 0:
        raise RuntimeError("No symbols available for holdout evaluation")

    rng.shuffle(symbols)
    if bundle is None:
        bundle = load_model_bundle(run_id, data_root)
    required_feature_keys = get_inference_required_feature_keys(bundle)
    alpha_required_keys = _alpha_required_feature_keys(bundle)

    compatible: List[str] = []
    rejected: List[str] = []

    for symbol in symbols:
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
        accepted_feature_keys = _accepted_strategy_feature_keys(bundle, accepted_ids)
        freeze_strats = freeze.get("strategies", [])
        if isinstance(freeze_strats, list):
            filtered = [
                strat
                for strat in freeze_strats
                if str(strat.get("strategy_id", "")) in accepted_ids
            ]
            skipped = len(freeze_strats) - len(filtered)
            freeze["strategies"] = filtered
            tprint(
                f"Acceptance filter kept {len(filtered)} strategies and skipped {skipped}"
            )
            if not filtered:
                tprint("No strategies passed the final acceptance gates; holdout evaluation will skip all strategies.")
    else:
        tprint("No strategy acceptance file found; holdout evaluator will use all frozen strategies.")

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
    portfolio_parts: Dict[str, List] = {"scores": [], "returns": [], "ts": []}

    for symbol in symbols:
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

    print(json.dumps(all_results, indent=2, default=str))


if __name__ == "__main__":
    main()
