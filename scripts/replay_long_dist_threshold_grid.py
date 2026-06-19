#!/usr/bin/env python3
"""Replay the long-dist policy rank threshold grid with 15m execution paths.

This diagnostic rebuilds threshold metrics from the persisted policy-OOS rank
reference, joins strategy-specific policy params, resolves 15m paths from local
1m execution candles, and uses the per-symbol spread baseline.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from extreme_price_movements.data_store import _fetch_ohlcv_paged, make_perp_exchange
from extreme_price_movements.simple_policy_optimiser import (
    DEFAULT_FORWARD_BARS,
    DEFAULT_POLICY_PER_SIDE_COST_PCT,
    apply_deployment_concurrency_constraints,
    score_deployment_threshold_rows,
    simulate_and_score,
    _apply_delayed_entry_execution_model,
    _fetch_policy_paths,
    _json_safe,
    _path_take,
    _policy_path_finite_mask,
    _with_policy_spread_cost_columns,
)


LONG_DIST_STRATEGY_ID = (
    "long_dist_ema20_atr_-0_92271453_loc_bb_channel_pos_48_0_60767579_"
    "leverage_build_score_0_45107844_return_autocorr_48_1_18643_"
    "rolling_range_20_-0_25967735"
)


class LocalExecution15mStore:
    """Read local 1m execution candles and resample them to 15m bars."""

    timeframe = "15m"

    def __init__(
        self,
        execution_1m_root: Path,
        *,
        download_missing_15m: bool = False,
    ):
        self.root_dir = str(execution_1m_root)
        self.ohlcv_dir = str(execution_1m_root / "ohlcv")
        self._cache: Dict[Tuple[str, str, str], pd.DataFrame] = {}
        self.download_missing_15m = bool(download_missing_15m)
        self._exchange: Any = None
        self.downloaded_symbols: set[str] = set()
        self.downloaded_rows: int = 0

    @staticmethod
    def _symbol_candidates(symbol: str) -> Iterable[str]:
        raw = str(symbol or "").strip()
        yield raw
        yield raw.replace("/", "_")
        if "/" in raw:
            yield raw.replace("/", "_")
        if raw.endswith("/USD:USD"):
            yield raw.replace("/USD:USD", "_USD:USD")
        if raw.endswith("_USD:USD"):
            yield raw.replace("_USD:USD", "/USD:USD")

    def _symbol_dir(self, symbol: str) -> Optional[Path]:
        for candidate in self._symbol_candidates(symbol):
            if not candidate:
                continue
            safe = candidate.replace("/", "_")
            path = Path(self.ohlcv_dir) / f"symbol={safe}"
            if path.exists():
                return path
        return None

    def load(
        self,
        symbol: str,
        columns: Optional[Iterable[str]] = None,
        start_ts: Optional[pd.Timestamp] = None,
        end_ts: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        start = pd.Timestamp(start_ts).tz_convert("UTC") if start_ts is not None else None
        end = pd.Timestamp(end_ts).tz_convert("UTC") if end_ts is not None else None
        cache_key = (
            str(symbol),
            str(start.isoformat() if start is not None else ""),
            str(end.isoformat() if end is not None else ""),
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            return self._select_columns(cached, columns)

        sym_dir = self._symbol_dir(symbol)
        if sym_dir is None:
            out = self._maybe_fetch_15m(symbol, self._empty_frame(), start, end)
            self._cache[cache_key] = out
            return self._select_columns(out, columns)

        start_sec = int(start.timestamp()) if start is not None else 0
        end_sec = int(end.timestamp()) if end is not None else 2**63 - 1
        files = []
        for file_path in sym_dir.glob("year=*/*.parquet"):
            base = file_path.name.replace(".parquet", "")
            parts = base.split("-")
            include = True
            if len(parts) >= 3:
                try:
                    file_min = int(parts[-2])
                    file_max = int(parts[-1])
                    include = not (file_min > end_sec or file_max < start_sec)
                except ValueError:
                    include = True
            if include:
                files.append(file_path)
        if not files:
            out = self._maybe_fetch_15m(symbol, self._empty_frame(), start, end)
            self._cache[cache_key] = out
            return self._select_columns(out, columns)

        pieces = []
        read_cols = ["ts", "open", "high", "low", "close"]
        for file_path in files:
            try:
                pieces.append(pd.read_parquet(file_path, columns=read_cols))
            except Exception:
                continue
        if not pieces:
            out = self._maybe_fetch_15m(symbol, self._empty_frame(), start, end)
            self._cache[cache_key] = out
            return self._select_columns(out, columns)

        raw = pd.concat(pieces, ignore_index=True)
        raw["ts"] = pd.to_datetime(raw["ts"], utc=True, errors="coerce")
        raw = raw.dropna(subset=["ts"]).sort_values("ts")
        raw = raw.drop_duplicates("ts", keep="last").set_index("ts")
        if start is not None:
            raw = raw.loc[raw.index >= start]
        if end is not None:
            raw = raw.loc[raw.index <= end]
        if raw.empty:
            local_15m = self._empty_frame()
        else:
            for col in ("open", "high", "low", "close"):
                raw[col] = pd.to_numeric(raw[col], errors="coerce")
            local_15m = raw.resample(
                "15min",
                label="left",
                closed="left",
                origin="epoch",
            ).agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                }
            )
            local_15m = local_15m.dropna(subset=["open", "high", "low", "close"])
            local_15m = local_15m.astype(
                {
                    "open": "float32",
                    "high": "float32",
                    "low": "float32",
                    "close": "float32",
                }
            )
            local_15m.index.name = None
            local_15m["ts"] = local_15m.index

        resampled = self._maybe_fetch_15m(symbol, local_15m, start, end)
        self._cache[cache_key] = resampled
        return self._select_columns(resampled, columns)

    def _maybe_fetch_15m(
        self,
        symbol: str,
        local_15m: pd.DataFrame,
        start: Optional[pd.Timestamp],
        end: Optional[pd.Timestamp],
    ) -> pd.DataFrame:
        if not self.download_missing_15m or start is None or end is None:
            return local_15m
        expected = pd.date_range(
            start.floor("15min"),
            end.ceil("15min"),
            freq="15min",
            tz="UTC",
        )
        local_idx = (
            pd.DatetimeIndex(local_15m.index).tz_convert("UTC")
            if isinstance(local_15m.index, pd.DatetimeIndex) and len(local_15m)
            else pd.DatetimeIndex([], tz="UTC")
        )
        coverage = float(len(local_idx.intersection(expected)) / max(len(expected), 1))
        if coverage >= 0.995:
            return local_15m
        try:
            if self._exchange is None:
                self._exchange = make_perp_exchange()
            chunk_days = max(
                1,
                int(os.environ.get("EPM_SIMPLE_POLICY_15M_FETCH_CHUNK_DAYS", "14") or "14"),
            )
            chunks = []
            cursor = pd.Timestamp(start).floor("15min")
            end_ts = pd.Timestamp(end).ceil("15min")
            while cursor < end_ts:
                chunk_end = min(cursor + pd.Timedelta(days=chunk_days), end_ts)
                fetched_chunk = _fetch_ohlcv_paged(
                    self._exchange,
                    str(symbol),
                    int(cursor.value // 10**6),
                    int(chunk_end.value // 10**6),
                    timeframe="15m",
                    limit=1000,
                )
                if fetched_chunk is not None and not fetched_chunk.empty:
                    chunks.append(fetched_chunk)
                cursor = chunk_end
        except Exception:
            return local_15m
        fetched = pd.concat(chunks, axis=0) if chunks else pd.DataFrame()
        if fetched is None or fetched.empty:
            return local_15m
        fetched = fetched.copy()
        fetched.index = pd.to_datetime(fetched.index, utc=True, errors="coerce")
        fetched = fetched.dropna().sort_index()
        fetched = fetched[["open", "high", "low", "close"]].astype("float32")
        fetched.index.name = None
        fetched["ts"] = fetched.index
        self.downloaded_symbols.add(str(symbol))
        self.downloaded_rows += int(len(fetched))
        if local_15m.empty:
            return fetched
        out = pd.concat([local_15m, fetched], axis=0)
        out = out[~out.index.duplicated(keep="last")].sort_index()
        out["ts"] = out.index
        return out

    @staticmethod
    def _empty_frame() -> pd.DataFrame:
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "ts"],
            index=pd.DatetimeIndex([], tz="UTC"),
        )

    @staticmethod
    def _select_columns(df: pd.DataFrame, columns: Optional[Iterable[str]]) -> pd.DataFrame:
        if not columns:
            return df.copy()
        requested = list(dict.fromkeys([*columns, "ts"]))
        available = [col for col in requested if col in df.columns]
        return df.loc[:, available].copy()


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _iter_strategy_payloads(payload: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    keys = (
        "strategies",
        "selected_strategies",
        "rejected_strategies",
        "shadow_source_rejected_strategies",
        "top_strategies",
    )
    for key in keys:
        value = payload.get(key)
        if isinstance(value, dict):
            for item in value.values():
                if isinstance(item, dict):
                    yield item
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    yield item


def _find_strategy(artifact_root: Path, strategy_id: str) -> Dict[str, Any]:
    candidates = [
        artifact_root / "strategy_for_inference.json",
        artifact_root / "simple_policy_optimiser" / "strategy_for_inference.json",
        artifact_root / "simple_policy_optimiser" / "policy_optimisation_results.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        payload = _read_json(path)
        if str(payload.get("strategy_id") or "") == strategy_id:
            return payload
        for item in _iter_strategy_payloads(payload):
            if str(item.get("strategy_id") or item.get("id") or "") == strategy_id:
                return item
    raise FileNotFoundError(f"Could not find strategy {strategy_id} under {artifact_root}")


def _first_existing(paths: Iterable[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("None of the candidate paths exists: " + ", ".join(map(str, paths)))


def _load_rank_and_policy_rows(
    artifact_root: Path,
    strategy_id: str,
) -> pd.DataFrame:
    rank_path = _first_existing(
        [
            artifact_root / "simple_policy_optimiser" / "rank_reference" / f"{strategy_id}.parquet",
            artifact_root / "simple_policy_optimiser" / "rank_reference" / f"{strategy_id}.csv",
        ]
    )
    if rank_path.suffix == ".parquet":
        rank = pd.read_parquet(rank_path)
    else:
        rank = pd.read_csv(rank_path)
    policy_path = _first_existing(
        [
            artifact_root / "policy_oos_predictions" / f"policy_oos_{strategy_id}_clf.parquet",
            artifact_root / "policy_oos_predictions" / f"policy_oos_{strategy_id}.parquet",
        ]
    )
    policy_cols = ["timestamp", "symbol", "barrier_pct", "u_policy_net"]
    policy = pd.read_parquet(policy_path, columns=[c for c in policy_cols if c])

    rank["timestamp"] = pd.to_datetime(rank["timestamp"], utc=True, errors="coerce")
    policy["timestamp"] = pd.to_datetime(policy["timestamp"], utc=True, errors="coerce")
    rank["symbol"] = rank["symbol"].astype(str)
    policy["symbol"] = policy["symbol"].astype(str)
    policy = policy.dropna(subset=["timestamp", "symbol"]).drop_duplicates(
        ["timestamp", "symbol"],
        keep="last",
    )
    rows = rank.merge(policy, on=["timestamp", "symbol"], how="left", suffixes=("", "_policy"))
    rows["strategy_id"] = strategy_id
    rows["side"] = 1.0
    rows["side_name"] = "long"
    rows["market_mode"] = rows.get("market_mode", "perps")
    rows = rows.dropna(subset=["timestamp", "symbol", "rank_pct", "calibrated_score", "barrier_pct"])
    rows = rows.sort_values(["timestamp", "rank_pct"], ascending=[True, False]).reset_index(drop=True)
    return rows


def _strategy_sim_params(strategy: Dict[str, Any]) -> Dict[str, Any]:
    keys = (
        "size_power",
        "best_size_power",
        "sl_mult",
        "trailing_activation_mult",
        "trailing_power",
        "trailing_squash_divisor",
        "giveback_beta",
        "capital_protect_mfe_mult",
        "capital_protect_regression_frac",
        "adverse_exit_enabled",
        "adverse_exit_min_mae_atr",
        "adverse_exit_min_speed",
        "adverse_exit_theta_quantile",
        "adverse_exit_theta",
        "adverse_exit_alpha",
        "adverse_exit_beta",
        "adverse_exit_delta",
        "adverse_exit_fast_bars",
        "adverse_exit_max_mfe_atr",
        "atr_power",
        "atr_multiplier",
        "hard_tp_abs_pct",
        "exit_pressure_enabled",
        "exit_pressure_alpha",
        "exit_pressure_beta",
        "exit_pressure_delta",
        "exit_pressure_kappa",
        "exit_pressure_psi",
        "exit_pressure_omega",
        "exit_pressure_min_multiplier",
        "redeploy_scale_bps",
        "target_holding_hours",
        "churn_penalty_bps",
        "median_barrier_frac",
        "policy_median_barrier_frac",
    )
    params: Dict[str, Any] = {}
    for key in keys:
        if key in strategy:
            params[key] = strategy[key]
    if "size_power" not in params and "best_size_power" in params:
        params["size_power"] = params.pop("best_size_power")
    params.setdefault("size_power", 1.0)
    params.setdefault("sl_mult", 1.0)
    params.setdefault("trailing_activation_mult", 1.0)
    params.setdefault("trailing_power", 1.5)
    params.setdefault("trailing_squash_divisor", 2.0)
    params.setdefault("giveback_beta", 0.5)
    params.setdefault("atr_power", 1.0)
    params.setdefault("atr_multiplier", 1.0)
    params.setdefault("hard_tp_abs_pct", 0.0)
    return params


def _simulate_candidate_rows(
    rows: pd.DataFrame,
    paths: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    strategy: Dict[str, Any],
    *,
    market_mode: str,
) -> pd.DataFrame:
    sim_params = _strategy_sim_params(strategy)
    sim_params["max_concurrent_trades"] = 1_000_000
    sim_params["max_concurrent_per_asset"] = 1_000_000
    metrics = simulate_and_score(
        rows,
        *paths,
        cost_pct=DEFAULT_POLICY_PER_SIDE_COST_PCT,
        market_mode=market_mode,
        **sim_params,
    )
    selected_mask = np.asarray(metrics.get("selected_mask", []), dtype=bool)
    if len(selected_mask) == len(rows):
        out = rows.iloc[np.flatnonzero(selected_mask)].copy().reset_index(drop=True)
    else:
        out = rows.copy().reset_index(drop=True)
    arrays = {
        "net_gain": np.asarray(metrics.get("raw_gains", []), dtype=np.float64),
        "gross_gain": np.asarray(metrics.get("gross_gains", []), dtype=np.float64),
        "position_size": np.asarray(metrics.get("sizes", []), dtype=np.float64),
        "exit_bars": np.asarray(metrics.get("exit_bars", []), dtype=np.float64),
        "exit_reason": np.asarray(metrics.get("exit_reason", []), dtype=object),
        "expected_spread_bps": np.asarray(metrics.get("expected_spread_bps", []), dtype=np.float64),
        "entry_half_spread_bps": np.asarray(metrics.get("entry_half_spread_bps", []), dtype=np.float64),
        "exit_spread_cost_bps": np.asarray(metrics.get("exit_spread_cost_bps", []), dtype=np.float64),
        "entry_slippage_proxy_bps": np.asarray(metrics.get("entry_slippage_proxy_bps", []), dtype=np.float64),
        "entry_reanchor_bps": np.asarray(metrics.get("entry_reanchor_bps", []), dtype=np.float64),
    }
    for key, value in arrays.items():
        if len(value) == len(out):
            out[key] = value
    if "position_size" in out.columns:
        denom = pd.to_numeric(out["position_size"], errors="coerce").replace(0.0, np.nan)
        out["net_return_per_notional"] = pd.to_numeric(out["net_gain"], errors="coerce") / denom
        out["gross_return_per_notional"] = pd.to_numeric(out["gross_gain"], errors="coerce") / denom
    out["deployment_rank_pct"] = pd.to_numeric(out["rank_pct"], errors="coerce")
    return out


def _max_drawdown(gains: pd.Series) -> float:
    if gains.empty:
        return 0.0
    cum = gains.cumsum()
    return float((cum - cum.cummax()).min())


def _summarize_rows(rows: pd.DataFrame, prefix: str) -> Dict[str, Any]:
    if rows.empty:
        return {
            f"{prefix}_n_trades": 0,
            f"{prefix}_net_pnl": 0.0,
            f"{prefix}_mean_net_gain": 0.0,
            f"{prefix}_mean_net_return_per_notional": 0.0,
            f"{prefix}_net_hit_rate": 0.0,
            f"{prefix}_gross_hit_rate": 0.0,
            f"{prefix}_max_drawdown": 0.0,
            f"{prefix}_trades_per_day": 0.0,
        }
    net = pd.to_numeric(rows["net_gain"], errors="coerce").dropna()
    gross = pd.to_numeric(rows.get("gross_gain"), errors="coerce").reindex(net.index)
    ret = pd.to_numeric(rows.get("net_return_per_notional"), errors="coerce").reindex(net.index)
    timestamps = pd.to_datetime(rows.loc[net.index, "timestamp"], utc=True, errors="coerce")
    day_span = max(
        1.0,
        float((timestamps.max() - timestamps.min()).total_seconds() / 86400.0)
        if len(timestamps.dropna()) >= 2
        else 1.0,
    )
    return {
        f"{prefix}_n_trades": int(len(net)),
        f"{prefix}_net_pnl": float(net.sum()),
        f"{prefix}_gross_pnl": float(gross.sum()) if len(gross.dropna()) else float("nan"),
        f"{prefix}_mean_net_gain": float(net.mean()) if len(net) else 0.0,
        f"{prefix}_mean_gross_gain": float(gross.mean()) if len(gross.dropna()) else float("nan"),
        f"{prefix}_mean_net_return_per_notional": float(ret.mean()) if len(ret.dropna()) else float("nan"),
        f"{prefix}_net_hit_rate": float((net > 0).mean()) if len(net) else 0.0,
        f"{prefix}_gross_hit_rate": float((gross > 0).mean()) if len(gross.dropna()) else float("nan"),
        f"{prefix}_max_drawdown": _max_drawdown(net),
        f"{prefix}_trades_per_day": float(len(net) / day_span),
        f"{prefix}_symbols": int(rows.loc[net.index, "symbol"].nunique()) if "symbol" in rows else 0,
        f"{prefix}_mean_expected_spread_bps": float(pd.to_numeric(rows.get("expected_spread_bps"), errors="coerce").mean()),
        f"{prefix}_median_expected_spread_bps": float(pd.to_numeric(rows.get("expected_spread_bps"), errors="coerce").median()),
    }


def _threshold_grid(
    replayed: pd.DataFrame,
    strategy: Dict[str, Any],
    *,
    thresholds: Iterable[float],
) -> pd.DataFrame:
    rows = []
    total_cap = int(strategy.get("max_concurrent_trades") or 4)
    per_asset_cap = int(strategy.get("max_concurrent_per_asset") or 1)
    for threshold in thresholds:
        threshold = float(threshold)
        cumulative_raw = replayed.loc[replayed["rank_pct"] >= threshold].copy()
        cumulative = apply_deployment_concurrency_constraints(
            cumulative_raw,
            timestamp_col="timestamp",
            symbol_col="symbol",
            side_col="side_name",
            strategy_col="strategy_id",
            rank_col="deployment_rank_pct",
            holding_bars_col="exit_bars",
            bar_minutes=15,
            dynamic_threshold_enabled=False,
            max_concurrent_total=total_cap,
            max_concurrent_per_side=1_000_000,
            max_concurrent_per_asset=per_asset_cap,
            max_concurrent_per_strategy=1_000_000,
        )
        upper = threshold + 0.01
        local_raw = replayed.loc[
            (replayed["rank_pct"] >= threshold)
            & (replayed["rank_pct"] < (upper if upper < 1.0 else 1.0000001))
        ].copy()
        local = apply_deployment_concurrency_constraints(
            local_raw,
            timestamp_col="timestamp",
            symbol_col="symbol",
            side_col="side_name",
            strategy_col="strategy_id",
            rank_col="deployment_rank_pct",
            holding_bars_col="exit_bars",
            bar_minutes=15,
            dynamic_threshold_enabled=False,
            max_concurrent_total=total_cap,
            max_concurrent_per_side=1_000_000,
            max_concurrent_per_asset=per_asset_cap,
            max_concurrent_per_strategy=1_000_000,
        )
        threshold_metrics = score_deployment_threshold_rows(cumulative)
        row = {
            "threshold": threshold,
            "raw_candidates": int(len(cumulative_raw)),
            "raw_local_band_candidates": int(len(local_raw)),
            **{f"policy_{k}": v for k, v in threshold_metrics.items()},
            **_summarize_rows(cumulative, "cumulative"),
            **_summarize_rows(local, "local_band"),
            "selected_trade_indices": ",".join(map(str, cumulative.index.tolist())),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data_perp")
    parser.add_argument("--run-id", default="20260617_090000_no_mkt4_labelhpo_final_fit")
    parser.add_argument("--strategy-id", default=LONG_DIST_STRATEGY_ID)
    parser.add_argument("--market-mode", default="perps")
    parser.add_argument("--threshold-lo", type=float, default=0.70)
    parser.add_argument("--threshold-hi", type=float, default=0.99)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--spread-baseline-path",
        default="data_perp/exchanges/krakenfutures/spread_model/per_asset_spread_baseline_latest.csv",
    )
    parser.add_argument(
        "--download-missing-15m",
        action="store_true",
        help="Fill missing local 15m paths from Kraken Futures chart candles.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    artifact_root = data_root / "artifacts" / args.run_id
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else artifact_root
        / "simple_policy_optimiser"
        / "threshold_replay_15m"
        / args.strategy_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ["EPM_SIMPLE_POLICY_SPREAD_BASELINE_PATH"] = str(Path(args.spread_baseline_path))
    os.environ.setdefault("EPM_SIMPLE_POLICY_1M_DOWNLOAD", "0")

    strategy = _find_strategy(artifact_root, args.strategy_id)
    rows = _load_rank_and_policy_rows(artifact_root, args.strategy_id)
    rows = _with_policy_spread_cost_columns(rows, market_mode=args.market_mode)
    threshold_rows = rows.loc[
        pd.to_numeric(rows["rank_pct"], errors="coerce") >= float(args.threshold_lo)
    ].copy()

    store = LocalExecution15mStore(
        data_root / "exchanges" / "krakenfutures" / "execution_1m",
        download_missing_15m=bool(args.download_missing_15m),
    )
    paths = _fetch_policy_paths(threshold_rows, store, path_len=DEFAULT_FORWARD_BARS)
    finite_mask = _policy_path_finite_mask(paths)
    path_coverage = float(np.mean(finite_mask)) if len(finite_mask) else 0.0
    rows_with_paths = threshold_rows.iloc[np.flatnonzero(finite_mask)].copy().reset_index(drop=True)
    paths = _path_take(paths, np.flatnonzero(finite_mask))
    rows_with_paths, paths = _apply_delayed_entry_execution_model(
        rows_with_paths,
        paths,
        data_root=str(data_root),
        market_mode=args.market_mode,
    )
    delayed_count = int(
        (
            rows_with_paths.get("entry_execution_source", pd.Series(dtype=object))
            == "delayed_1m_intraminute_proxy"
        ).sum()
    )

    replayed = _simulate_candidate_rows(
        rows_with_paths,
        paths,
        strategy,
        market_mode=args.market_mode,
    )
    thresholds = np.round(
        np.arange(
            float(args.threshold_lo),
            float(args.threshold_hi) + float(args.threshold_step) / 2.0,
            float(args.threshold_step),
        ),
        4,
    )
    grid = _threshold_grid(replayed, strategy, thresholds=thresholds)
    best = (
        grid.sort_values(
            ["cumulative_net_pnl", "cumulative_mean_net_gain", "cumulative_n_trades"],
            ascending=[False, False, False],
        )
        .head(1)
        .to_dict(orient="records")
    )
    coverage = {
        "strategy_id": args.strategy_id,
        "run_id": args.run_id,
        "rank_reference_rows": int(len(rows)),
        "threshold_population_rows": int(len(threshold_rows)),
        "rows_with_full_15m_paths": int(len(rows_with_paths)),
        "path_coverage": path_coverage,
        "replayed_rows": int(len(replayed)),
        "timestamp_min": rows["timestamp"].min().isoformat() if not rows.empty else None,
        "timestamp_max": rows["timestamp"].max().isoformat() if not rows.empty else None,
        "delayed_1m_entries": delayed_count,
        "download_missing_15m": bool(args.download_missing_15m),
        "downloaded_15m_symbols": int(len(store.downloaded_symbols)),
        "downloaded_15m_rows": int(store.downloaded_rows),
        "spread_baseline_path": str(Path(args.spread_baseline_path)),
        "mean_expected_spread_bps": float(pd.to_numeric(rows.get("expected_spread_bps"), errors="coerce").mean()),
        "median_expected_spread_bps": float(pd.to_numeric(rows.get("expected_spread_bps"), errors="coerce").median()),
        "best_by_net_pnl": best[0] if best else {},
    }

    replayed_path = output_dir / "replayed_candidates.parquet"
    grid_csv_path = output_dir / "threshold_grid.csv"
    grid_json_path = output_dir / "threshold_grid.json"
    summary_path = output_dir / "summary.json"
    replayed.to_parquet(replayed_path, index=False)
    grid.to_csv(grid_csv_path, index=False)
    grid_json_path.write_text(
        json.dumps(_json_safe(grid.to_dict(orient="records")), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(_json_safe(coverage), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(_json_safe(coverage), indent=2, sort_keys=True))
    print(f"threshold_grid_csv={grid_csv_path}")
    print(f"replayed_candidates={replayed_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
