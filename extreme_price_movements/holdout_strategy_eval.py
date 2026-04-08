from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from extreme_price_movements.hf_data_loader import _load_existing_data
from extreme_price_movements.config import CFG as DEFAULT_CFG
from extreme_price_movements.inference.feature_generator import (
    get_inference_required_feature_keys,
    load_or_compute_features,
)
from extreme_price_movements.inference.model_orchestrator import ModelOrchestrator
from extreme_price_movements.model_loader import load_model_bundle
from extreme_price_movements.features import (
    add_regime_gates,
    compute_features_hourly,
    compute_market_features,
)
from extreme_price_movements.data_store import load_features_selected
from extreme_price_movements.ridge_position_sizer import prepare_policy_params_from_tpsl_optimiser
from extreme_price_movements.simple_position_sizer import evaluate_selection_profit_proxy
from extreme_price_movements.ridge_position_sizer import run_policy_aware_labeling_step
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


def _freeze_path(data_root: str, run_id: str) -> Path:
    return Path(data_root) / "artifacts" / run_id / "ridge_sizer" / "strategy_params.json"


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
) -> pd.DataFrame:
    lookback_hours = max(24, int(np.ceil(len(panel["open"]) / 4.0)) + 24)
    feature_cfg = dict(DEFAULT_CFG)
    gated_required = any(
        isinstance(key, str)
        and key
        and (key in {"G_VOL", "G_TREND"} or "_G_VOL_" in key or "_G_TREND_" in key)
        for key in required_feature_keys
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
        required_feature_keys=required_feature_keys,
        )
    if feature_map:
        feat_df = pd.DataFrame()
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
            if feat_df.empty:
                feat_df = pd.DataFrame(index=series.index)
            feat_df[feat_name] = series.astype(np.float32)
        missing_required = {k for k in required_feature_keys if k not in feat_df.columns}
        if len(feat_df.columns) >= 20 and not missing_required:
            feat_df = feat_df.sort_index()
            if not isinstance(feat_df.index, pd.DatetimeIndex):
                feat_df.index = pd.to_datetime(feat_df.index, utc=True, errors="coerce")
            if feat_df.index.tz is None:
                feat_df.index = feat_df.index.tz_localize("UTC")
            else:
                feat_df.index = feat_df.index.tz_convert("UTC")
            return feat_df
        if missing_required:
            tprint(
                "Selected feature cache incomplete for this strategy contract; "
                f"falling back to full recompute for {len(missing_required)} missing keys"
            )

    # The selected feature cache only carries the lightweight selector set on
    # some runs. Recompute the full alpha feature matrix directly when needed.
    compute_panel = {
        key: df.copy()
        for key, df in panel.items()
        if isinstance(df, pd.DataFrame)
    }
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
        requested_feature_keys=sorted(required_feature_keys) if required_feature_keys else None,
    )
    feat_df = pd.DataFrame()
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
        if feat_df.empty:
            feat_df = pd.DataFrame(index=series.index)
        feat_df[feat_name] = series.astype(np.float32)
    if not feat_df.empty:
        feat_df = feat_df.sort_index()
        if not isinstance(feat_df.index, pd.DatetimeIndex):
            feat_df.index = pd.to_datetime(feat_df.index, utc=True, errors="coerce")
        if feat_df.index.tz is None:
            feat_df.index = feat_df.index.tz_localize("UTC")
        else:
            feat_df.index = feat_df.index.tz_convert("UTC")
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

    return pd.concat([feature_df, pd.DataFrame(updates, index=feature_df.index)], axis=1)


_GATED_INTERACTION_RE = re.compile(r"^(?P<base>.+)_(?P<gate>G_[A-Z0-9_]+)_(?P<state>[01])$")


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
        compute_panel = {key: df.copy() for key, df in panel.items() if isinstance(df, pd.DataFrame)}
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

    return pd.concat([feature_df, pd.DataFrame(updates, index=feature_df.index)], axis=1)


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
        atr_val = float(np.nanmedian(np.asarray(feature_df["atr_pct"].values, dtype=np.float64)))
    elif "prior_volatility" in feature_df.columns:
        atr_val = float(np.nanmedian(np.asarray(feature_df["prior_volatility"].values, dtype=np.float64)))
    else:
        atr_val = 0.02
    atr_val = float(np.clip(atr_val if np.isfinite(atr_val) and atr_val > 0 else 0.02, 1e-4, 0.2))

    seed = {
        "tp_mult": 2.0,
        "sl_mult": 1.6,
        "act_n": 0.5,
        "be_act_n": 0.5,
    }
    return prepare_policy_params_from_tpsl_optimiser(seed, atr_values={symbol: atr_val})


def evaluate_holdout_symbol(
    data_root: str,
    run_id: str,
    symbol: str,
    freeze: Dict[str, Any],
) -> Dict[str, Any]:
    bundle = load_model_bundle(run_id, data_root)
    orchestrator = ModelOrchestrator(bundle, {"disable_spike_filter": True})

    panel, panel_symbol = _load_price_panel(symbol)
    required_feature_keys = get_inference_required_feature_keys(bundle)
    feature_df = _load_symbol_features(
        data_root,
        run_id,
        symbol=panel_symbol,
        panel=panel,
        required_feature_keys=required_feature_keys,
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
    frozen_rows: List[Dict[str, Any]] = []
    portfolio_score_parts: List[np.ndarray] = []
    portfolio_return_parts: List[np.ndarray] = []
    portfolio_ts_parts: List[np.ndarray] = []

    policy_params = _compute_policy_params(feature_df, panel_symbol)

    for strat in freeze.get("strategies", []):
        strategy_id = str(strat["strategy_id"])
        side = _infer_strategy_side(bundle, strategy_id, str(strat.get("side", "")))
        if not side:
            tprint(
                f"Skipping strategy {strategy_id[:60]}: could not infer side from freeze file or alpha bundle"
            )
            continue
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

        aligned_scores = score_series.reindex(pd.DatetimeIndex(outcomes["timestamp"].values))
        score_values = _fill_nonfinite(aligned_scores.to_numpy(dtype=np.float64))
        raw_returns = np.asarray(outcomes["label"].values, dtype=np.float64)
        ts_values = np.asarray(outcomes["timestamp"].values)

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
        portfolio_days = float(
            (pd.to_datetime(portfolio_ts.max()) - pd.to_datetime(portfolio_ts.min()))
            / np.timedelta64(1, "D")
        ) if len(portfolio_ts) > 1 else 0.0
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


def _pick_best_model_per_strategy(freeze: Dict[str, Any]) -> Dict[str, Any]:
    """Auto-pick the best model (ridge vs et) per strategy_id from freeze data."""
    comparison_path = Path(freeze.get("_comparison_path_", ""))
    if comparison_path.exists():
        try:
            comparison = json.loads(comparison_path.read_text())
            winner = comparison.get("winner", "ridge")
            tprint(f"Auto-picked best model from comparison: {winner}")
        except Exception:
            winner = "ridge"
    else:
        winner = "ridge"

    best_freeze = dict(freeze)
    strategies = best_freeze.get("strategies", [])
    if winner == "et":
        et_freeze_path = Path(str(_freeze_path("", "")).replace("ridge_sizer", "et_sizer"))
        for alt_path in [
            Path(best_freeze.get("_data_root_", "data")) / "artifacts" / best_freeze.get("_run_id_", "") / "et_sizer" / "strategy_params.json",
        ]:
            if alt_path.exists():
                try:
                    et_freeze = json.loads(alt_path.read_text())
                    if et_freeze.get("strategies"):
                        strategies = et_freeze["strategies"]
                        best_freeze["strategies"] = strategies
                        best_freeze["model_source"] = "et"
                        tprint(f"Using ET strategy params from {alt_path}")
                        break
                except Exception:
                    pass
    else:
        best_freeze["model_source"] = "ridge"
    return best_freeze


def _sample_random_symbols(n: int = 5, seed: int = 42) -> List[str]:
    """Sample n random symbols from the training universe."""
    rng = np.random.RandomState(seed)
    try:
        universe = get_training_universe()
        symbols = list(universe.keys()) if isinstance(universe, dict) else list(universe)
    except Exception:
        symbols = ["AIXBT_USDC"]
    if len(symbols) <= n:
        return symbols
    return list(rng.choice(symbols, size=n, replace=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen strategy thresholds on holdout symbols.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--run-id", default="20260321_140000")
    parser.add_argument("--symbol", default="", help="Single symbol to evaluate (overrides --n-symbols)")
    parser.add_argument("--n-symbols", type=int, default=5, help="Number of random symbols to evaluate")
    parser.add_argument("--freeze-json", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    freeze = (
        json.loads(Path(args.freeze_json).read_text())
        if args.freeze_json
        else _load_or_write_freeze_file(args.data_root, args.run_id)
    )

    freeze = _pick_best_model_per_strategy(freeze)

    if args.symbol:
        symbols = [args.symbol]
    else:
        symbols = _sample_random_symbols(n=args.n_symbols, seed=args.seed)
        tprint(f"Sampled {len(symbols)} holdout symbols: {symbols}")

    all_results: List[Dict[str, Any]] = []
    portfolio_parts: Dict[str, List] = {"scores": [], "returns": [], "ts": []}

    for symbol in symbols:
        try:
            result = evaluate_holdout_symbol(args.data_root, args.run_id, symbol, freeze)
            all_results.append(result)
            if result.get("portfolio"):
                tprint(f"  {symbol}: portfolio wallet_pnl={result['portfolio'].get('wallet_pnl', 'N/A')}")
        except Exception as e:
            tprint(f"  {symbol}: SKIPPED ({e})")

    output_path = (
        Path(args.output_json)
        if args.output_json
        else Path(args.data_root) / "artifacts" / args.run_id / "ridge_sizer" / "holdout_multi_metrics.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(all_results, indent=2, default=str))
    tprint(f"Wrote holdout metrics for {len(all_results)} symbols to {output_path}")

    print(json.dumps(all_results, indent=2, default=str))


if __name__ == "__main__":
    main()
