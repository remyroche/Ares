#!/usr/bin/env python3
"""Materialize the exact frozen F0/context fields for the later TP6/SL4 rows.

This is deliberately a source-side materializer, not a model retrain.  It
uses the native Kraken perps hourly/OI/funding/order-book stores, computes the
declared F0 and frozen context fields in bounded symbol chunks, and joins them
back by the canonical ``(__ts__, __symbol__)`` identity.  No near-name aliases
or target-derived values are filled in.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.config import CFG, enable_perp_feature_keys  # noqa: E402
from extreme_price_movements.data_store import (  # noqa: E402
    exchange_data_component,
    make_ohlcv_store,
    to_panel,
)
from extreme_price_movements.features import add_regime_gates, compute_market_features  # noqa: E402
from extreme_price_movements.features_negative_residuals import (  # noqa: E402
    _causal_robust_z,
)
from extreme_price_movements.features_residual import (  # noqa: E402
    _peer_resid,
    _rolling_beta_resid,
)
from extreme_price_movements.pipeline_steps import (  # noqa: E402
    _compute_features_hourly_runtime,
    _load_saved_microdata_for_symbols,
)
from extreme_price_movements.static_feature_store import compute_static_features  # noqa: E402
from extreme_price_movements.run_pipeline import (  # noqa: E402
    _apply_market_mode_paths,
    _configure_report_roots,
    _normalize_cfg_paths,
)
from extreme_price_movements.tp6_portability_data import FROZEN_META_CONTEXT  # noqa: E402


DEFAULT_CANDIDATES = ROOT / "data_perp/artifacts/execution_ev_july20_23_retrospective_20260730_v2/candidates/candidate_features.parquet"
DEFAULT_CONTRACT = ROOT / "data_perp/artifacts/feature_portability_f4_panel_20260804_v1/f4_representation_contracts.json"
DEFAULT_OUT = ROOT / "data_perp/artifacts/tp6_sl4_gam_untouched_later_20260815_v1/rebuilt_f0_context"
DEFAULT_END = "2026-07-24 23:00:00+00:00"


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode()
    ).hexdigest()


def _setup_cfg(keys: Iterable[str]) -> dict:
    cfg = dict(CFG)
    _normalize_cfg_paths(cfg)
    cfg["exchange_id"] = "kraken"
    cfg["exchange"] = "kraken"
    _apply_market_mode_paths(cfg, "perps")
    cfg["exchange_data_component"] = exchange_data_component("kraken", "perps")
    cfg = enable_perp_feature_keys(cfg)
    _configure_report_roots(cfg)
    cfg.update(
        skip_feature_snapshot_validation=True,
        skip_feature_postsave_checks=True,
        feature_portability_mode="native",
        feature_portability_strict=False,
        feature_portability_selected_dependency_closure=True,
        feature_portability_repair_keys=list(keys),
    )
    return cfg


def _load_raw_panel(symbols: list[str], *, end: pd.Timestamp, warmup_days: int) -> tuple[dict, dict, dict]:
    cfg = _setup_cfg(())
    store = make_ohlcv_store(cfg, timeframe="1h")
    start = end - pd.Timedelta(days=int(warmup_days))
    dfs: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        frame = store.load(symbol, end_ts=end)
        if frame.empty:
            continue
        frame = frame.loc[frame.index >= start].copy()
        # Some Kraken symbols have short exchange gaps in the 30-day window;
        # the selected F0/context fields only need the declared 168-hour
        # lookback, so do not reject a symbol merely for a few missing bars.
        if len(frame) >= 24 * 7:
            dfs[symbol] = frame
    if not dfs:
        raise RuntimeError("no later candidate symbols were readable from Kraken hourly store")
    panel = to_panel(dfs)
    micro, orderbook = _load_saved_microdata_for_symbols(
        cfg["data_root"], list(panel["close"].columns), panel["close"].index, cfg
    )
    for name, values in micro.items():
        panel[name] = values
    market = compute_market_features(panel, cfg["market_basket"])
    gates = add_regime_gates(market, cfg["gate_vol_lookback_hours"], cfg["gate_trend_thr"])
    return panel, gates, orderbook


def _long_features(features: dict, names: list[str], target_keys: set[tuple[pd.Timestamp, str]]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for name in names:
        value = features.get(name)
        if value is None:
            continue
        if isinstance(value, pd.Series):
            value = value.to_frame()
        if not isinstance(value, pd.DataFrame):
            continue
        long = (
            value.rename_axis(index="__ts__", columns="__symbol__")
            .stack(dropna=False)
            .rename(name)
            .reset_index()
        )
        if not target_keys:
            pieces.append(long)
        else:
            keys = pd.MultiIndex.from_frame(long[["__ts__", "__symbol__"]])
            keep = keys.isin(pd.MultiIndex.from_tuples(sorted(target_keys), names=["__ts__", "__symbol__"]))
            pieces.append(long.loc[keep].copy())
    if not pieces:
        return pd.DataFrame(columns=["__ts__", "__symbol__", *names])
    out = pieces[0]
    for piece in pieces[1:]:
        out = out.merge(piece, on=["__ts__", "__symbol__"], how="outer", validate="one_to_one")
    return out


def _repair_cross_sectional_context(
    *,
    symbols: list[str],
    target_keys: set[tuple[pd.Timestamp, str]],
    end: pd.Timestamp,
    warmup_days: int,
) -> pd.DataFrame:
    """Recover fields whose meaning requires a full peer universe.

    The selected runtime closure is intentionally narrow and can omit the
    parent ``trend_pct`` frame needed by the market residual.  This repair
    computes that parent causally per symbol over a long warm-up, then applies
    the reviewed cross-sectional operators once to the complete universe.
    """
    panel, gates, _ = _load_raw_panel(symbols, end=end, warmup_days=warmup_days)
    index = panel["close"].index
    trend_parts: list[pd.DataFrame] = []
    ret4_parts: list[pd.DataFrame] = []
    cfg = _setup_cfg(["trend_pct", "ret4h"])
    for number, symbol in enumerate(symbols, start=1):
        chunk = {
            name: frame.reindex(columns=[symbol]).copy()
            for name, frame in panel.items()
            if isinstance(frame, pd.DataFrame)
        }
        result = compute_static_features(
            chunk,
            gates,
            cfg,
            requested_feature_keys=["trend_pct", "ret4h"],
            feature_store_id="tp6_sl4_later_f0_context_trend_repair",
        )
        trend = result.features.get("trend_pct")
        ret4 = result.features.get("ret4h")
        if isinstance(trend, pd.DataFrame):
            trend_parts.append(trend.reindex(index=index, columns=[symbol]))
        if isinstance(ret4, pd.DataFrame):
            ret4_parts.append(ret4.reindex(index=index, columns=[symbol]))
        if number % 10 == 0 or number == len(symbols):
            print(json.dumps({"repair": "trend_parent", "symbols_done": number, "symbols_total": len(symbols)}), flush=True)
    if not trend_parts or not ret4_parts:
        raise RuntimeError("native trend/ret4 parent repair produced no frames")
    trend_panel = pd.concat(trend_parts, axis=1).reindex(columns=symbols)
    ret4_panel = pd.concat(ret4_parts, axis=1).reindex(columns=symbols)
    trend = gates["mkt_trend"].reindex(index).astype(np.float32)
    rv = gates["mkt_rv"].reindex(index).astype(np.float32)
    market_factor = (trend / (rv * np.sqrt(24.0) + 1e-12)).replace([np.inf, -np.inf], np.nan)
    trend_resid = _rolling_beta_resid(trend_panel, market_factor, 96, standardize=False)
    ret4_peer = _peer_resid(ret4_panel)

    def _base(symbol: str) -> str:
        text = str(symbol).upper()
        if ":" in text:
            text = text.split(":", 1)[0]
        return text.split("/", 1)[0].replace("_", "").replace("-", "")

    btc = next((s for s in symbols if _base(s) == "BTCUSD" or _base(s) == "BTC"), None)
    eth = next((s for s in symbols if _base(s) == "ETHUSD" or _base(s) == "ETH"), None)
    if btc is not None:
        alt = [s for s in symbols if s not in {btc, eth}]
        btc_ret = ret4_panel[btc]
        alt_ret = ret4_panel[alt].median(axis=1, skipna=True) if alt else ret4_panel.median(axis=1, skipna=True)
        btc_resilience = _causal_robust_z(btc_ret, 24 * 30, 24 * 7).clip(lower=0.0) * _causal_robust_z(alt_ret, 24 * 30, 24 * 7).mul(-1.0).clip(lower=0.0)
    else:
        btc_resilience = pd.Series(np.nan, index=index, dtype=np.float32)

    # At hourly cadence the canonical 1-hour realized-volatility window has
    # one observation and is therefore zero; preserve that declared proxy
    # rather than silently substituting the 24-hour gate ratio.
    market_ret = ret4_panel.div(4.0).median(axis=1, skipna=True)
    rv_1h = market_ret.rolling(1, min_periods=1).std(ddof=0)
    rv_24h = market_ret.rolling(24, min_periods=2).std(ddof=0)
    rv_ratio = (rv_1h / (rv_24h + 1e-12)).replace([np.inf, -np.inf], np.nan).clip(0.0, 10.0)

    def broadcast(series: pd.Series) -> pd.DataFrame:
        return pd.DataFrame(
            np.broadcast_to(series.to_numpy(dtype=np.float32)[:, None], (len(series), len(symbols))),
            index=index,
            columns=symbols,
        )

    repair = {
        "trend_pct_mkt_resid": trend_resid,
        "ret4h_peer_resid": ret4_peer,
        "btc_resilience_alt_weakness": broadcast(btc_resilience),
        "regime_liquidity_score": broadcast(gates["regime_liquidity_score"].reindex(index)),
        "mkt_rv_ratio_1h_24h": broadcast(rv_ratio),
    }
    pieces = []
    for name, value in repair.items():
        frame = value if isinstance(value, pd.DataFrame) else broadcast(value)
        pieces.append(_long_features({name: frame}, [name], target_keys))
    return pd.concat(pieces, ignore_index=True).groupby(["__ts__", "__symbol__"], as_index=False, sort=False).first()


def run(
    *,
    candidates: Path,
    contract_path: Path,
    output_dir: Path,
    end: str,
    warmup_days: int,
    chunk_symbols: int,
    all_symbol_runtime: bool = False,
    repair_cross_sectional: bool = False,
    extra_context_fields: Iterable[str] = (),
) -> Path:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    candidate = pd.read_parquet(candidates)
    required_identity = {"candidate_id", "__ts__", "__symbol__", "side_name"}
    missing = sorted(required_identity.difference(candidate.columns))
    if missing:
        raise ValueError(f"candidate source misses identity columns: {missing}")
    candidate["__ts__"] = pd.to_datetime(candidate["__ts__"], utc=True, errors="coerce")
    candidate["__symbol__"] = candidate["__symbol__"].astype(str)
    if candidate["candidate_id"].duplicated().any():
        raise ValueError("candidate IDs are not unique")
    with contract_path.open() as handle:
        contract = json.load(handle)["F0_current_frozen"]
    f0 = sorted(set(contract["long"]) | set(contract["short"]))
    # The default context closure is intentionally small.  Callers that need
    # to replay a frozen downstream contract may extend it explicitly; this
    # keeps the extension auditable and prevents accidental near-name or
    # feature-family substitutions.
    extra_context_fields = tuple(dict.fromkeys(map(str, extra_context_fields)))
    wanted = list(dict.fromkeys([*f0, *FROZEN_META_CONTEXT, *extra_context_fields]))
    absent = [name for name in wanted if name not in candidate.columns]
    target_keys = set(zip(candidate["__ts__"], candidate["__symbol__"]))
    panel, gates, orderbook = _load_raw_panel(sorted(candidate["__symbol__"].unique()), end=pd.Timestamp(end), warmup_days=warmup_days)
    cfg = _setup_cfg(absent)
    generated: list[pd.DataFrame] = []
    symbols = list(panel["close"].columns)
    if all_symbol_runtime:
        # Cross-sectional residuals and market composites must see the full
        # universe.  Running this on symbol chunks silently turns peer/market
        # residuals into constants, so this mode is explicit and auditable.
        features, _, _ = _compute_features_hourly_runtime(
            panel,
            gates.copy(),
            cfg,
            orderbook,
            requested_feature_keys=absent,
            feature_store_id="tp6_sl4_later_f0_context_all_symbols",
        )
        long = _long_features(features, absent, target_keys)
        if not long.empty:
            generated.append(long)
        print(json.dumps({"runtime_mode": "all_symbols", "chunk_symbols": len(symbols), "generated_fields": sorted(features), "rows": int(len(long))}), flush=True)
    else:
        for start in range(0, len(symbols), max(1, int(chunk_symbols))):
            chunk = symbols[start : start + max(1, int(chunk_symbols))]
            panel_chunk = {name: frame.reindex(columns=chunk).copy() for name, frame in panel.items() if isinstance(frame, pd.DataFrame)}
            orderbook_chunk = {name: orderbook[name] for name in chunk if name in orderbook}
            features, _, _ = _compute_features_hourly_runtime(
                panel_chunk,
                gates.copy(),
                cfg,
                orderbook_chunk,
                requested_feature_keys=absent,
                feature_store_id="tp6_sl4_later_f0_context",
            )
            long = _long_features(features, absent, target_keys)
            if not long.empty:
                generated.append(long)
            print(json.dumps({"runtime_mode": "symbol_chunks", "chunk_start": start, "chunk_symbols": len(chunk), "generated_fields": sorted(features), "rows": int(len(long))}), flush=True)
    if generated:
        extra = pd.concat(generated, ignore_index=True)
        extra = extra.groupby(["__ts__", "__symbol__"], as_index=False, sort=False).first()
    else:
        extra = pd.DataFrame(columns=["__ts__", "__symbol__", *absent])
    if repair_cross_sectional:
        repair = _repair_cross_sectional_context(
            symbols=symbols,
            target_keys=target_keys,
            end=pd.Timestamp(end),
            warmup_days=max(int(warmup_days), 180),
        )
        extra = extra.merge(
            repair,
            on=["__ts__", "__symbol__"],
            how="outer",
            validate="one_to_one",
            suffixes=("", "__repair"),
        )
        for name in ["trend_pct_mkt_resid", "ret4h_peer_resid", "btc_resilience_alt_weakness", "regime_liquidity_score", "mkt_rv_ratio_1h_24h"]:
            repair_name = f"{name}__repair"
            if repair_name in extra:
                repaired = pd.to_numeric(extra[repair_name], errors="coerce")
                existing = pd.to_numeric(extra[name], errors="coerce") if name in extra else pd.Series(np.nan, index=extra.index)
                # Runtime closure outputs can be structurally constant when a
                # parent was not requested (notably market-liquidity/BTC
                # composites).  Prefer the explicitly repaired causal parent
                # whenever it contains real variation; never preserve a
                # synthetic zero merely because it is non-null.
                existing_varies = existing.nunique(dropna=True) > 1 and bool(existing.abs().sum() > 1e-12)
                repaired_varies = repaired.nunique(dropna=True) > 1 and bool(repaired.abs().sum() > 1e-12)
                if repaired_varies and not existing_varies:
                    extra[name] = repaired
                else:
                    extra[name] = existing.where(existing.notna(), repaired)
                extra = extra.drop(columns=[repair_name])
    # The feature graph is symbol/time keyed while the candidate population is
    # side-expanded (one long and one short row per symbol/time).  A generated
    # row may therefore match both side rows; this is deliberately many-to-one
    # rather than a silent duplicate expansion.
    output = candidate.merge(extra, on=["__ts__", "__symbol__"], how="left", validate="many_to_one", suffixes=("", "__generated"))
    collisions = [name for name in absent if f"{name}__generated" in output]
    if collisions:
        raise ValueError(f"unexpected generated-column collisions: {collisions[:10]}")
    coverage = {name: float(pd.to_numeric(output[name], errors="coerce").notna().mean()) for name in wanted if name in output}
    missing_after = [name for name in wanted if name not in output]
    manifest = {
        "schema": "tp6_sl4_later_f0_context_v1",
        "status": "MATERIALIZED" if not missing_after else "BLOCKED_MISSING_FIELDS",
        "candidate_source": str(candidates),
        "candidate_source_sha256": hashlib.sha256(candidates.read_bytes()).hexdigest(),
        "feature_contract_source": str(contract_path),
        "f0_fields": {side: list(contract[side]) for side in ("long", "short")},
        "frozen_meta_context": list(FROZEN_META_CONTEXT),
        "extra_context_fields": list(extra_context_fields),
        "rows": int(len(output)),
        "symbols": int(output["__symbol__"].nunique()),
        "timestamp_min": output["__ts__"].min().isoformat(),
        "timestamp_max": output["__ts__"].max().isoformat(),
        "warmup_days": int(warmup_days),
        "runtime_mode": "all_symbols" if all_symbol_runtime else "symbol_chunks",
        "cross_sectional_repair": bool(repair_cross_sectional),
        "source_cadence": "1h Kraken perps; OI/funding/order-book sidecars",
        "generated_fields": absent,
        "missing_after": missing_after,
        "coverage": coverage,
        "feature_contract_hash": _sha256_json({"f0": contract, "meta": list(FROZEN_META_CONTEXT), "extra": list(extra_context_fields)}),
        "no_alias_substitution": True,
        "no_target_imputation": True,
    }
    output.to_parquet(output_dir / "later_f0_context.parquet", index=False, compression="zstd")
    (output_dir / "materialization_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--warmup-days", type=int, default=180)
    parser.add_argument("--chunk-symbols", type=int, default=5)
    parser.add_argument(
        "--all-symbol-runtime",
        action="store_true",
        help="Compute the selected closure once on the full universe so peer/market residuals retain their semantics.",
    )
    parser.add_argument(
        "--repair-cross-sectional",
        action="store_true",
        help="Run a long-warmup per-symbol parent pass and apply full-universe residual repairs.",
    )
    parser.add_argument(
        "--extra-context-field",
        action="append",
        default=[],
        help="Explicit additional frozen context field; repeat once per field.",
    )
    args = parser.parse_args()
    run(
        candidates=args.candidates,
        contract_path=args.contract,
        output_dir=args.output_dir,
        end=args.end,
        warmup_days=args.warmup_days,
        chunk_symbols=args.chunk_symbols,
        all_symbol_runtime=args.all_symbol_runtime,
        repair_cross_sectional=args.repair_cross_sectional,
        extra_context_fields=args.extra_context_field,
    )
