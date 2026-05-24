#!/usr/bin/env python3
"""Evaluate Kraken-only perps where reference tick data is available.

The evaluation universe is intentionally stricter than plain OHLCV availability:
bars are eligible only when OHLCV plus mark, index, and premiumIndex-derived
columns are all finite.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.inference.candidate_selector import _build_mask_for_mode
from extreme_price_movements.inference.config import load_inference_config
from extreme_price_movements.inference.feature_generator import (
    get_inference_required_feature_keys,
    load_or_compute_features,
    raw_required_feature_keys,
)
from extreme_price_movements.inference.model_orchestrator import ModelOrchestrator
from extreme_price_movements.inference.parity import strategy_core_id
from extreme_price_movements.inference.run_inference import (
    _lgbm_mask_required_feature_keys,
    _load_lgbm_strategy_mask_rows,
)
from extreme_price_movements.features import _is_feature_allowed_for_portability_mode
from extreme_price_movements.model_loader import load_full_state
from scripts.replay_live_signal_predictions import _load_panel


REQUIRED_RAW_BAR_FIELDS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "mark_price",
    "index_price",
    "premium_index",
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_symbol_key(symbol: str) -> str:
    return str(symbol).replace("/", "_").replace(":", "_")


def _bases_from_current_universe(root: Path) -> set[str]:
    out: set[str] = set()
    for path in (root / "ohlcv").glob("symbol=*"):
        raw = path.name.split("=", 1)[1]
        base = raw.rsplit("_", 1)[0].upper()
        if not base:
            continue
        out.add(base)
        out.add(re.sub(r"^(1000|10000|100000|1000000)", "", base))
    return out


def _kraken_only_symbols(
    *,
    kraken_manifest: Path,
    current_root: Path,
    bases_json: Path | None,
) -> list[str]:
    manifest = _load_json(kraken_manifest)
    rows = list(manifest.get("symbols") or [])
    if bases_json is not None and bases_json.exists():
        wanted = {str(v).upper() for v in (_load_json(bases_json).get("symbols") or [])}
    else:
        wanted = {str(row.get("base") or "").upper() for row in rows}
        wanted -= _bases_from_current_universe(current_root)
    out = [
        str(row.get("perp_symbol"))
        for row in rows
        if str(row.get("base") or "").upper() in wanted and row.get("perp_symbol")
    ]
    if not out and wanted:
        out = [f"{base}/USD:USD" for base in sorted(wanted)]
    return sorted(set(out))


def _dedupe_mask_rows(
    policy_rows: list[dict[str, Any]],
    mask_rows: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for policy in policy_rows:
        sid = str(policy.get("strategy_id") or policy.get("strategy_for_inference") or "")
        core = strategy_core_id(sid)
        row = mask_rows.get(sid) or mask_rows.get(core)
        if row is None:
            row = {}
        merged = dict(row)
        merged.update({k: v for k, v in policy.items() if k not in merged or v not in (None, "")})
        merged.setdefault("strategy_id", sid)
        merged.setdefault("side", policy.get("side"))
        key = str(merged.get("strategy_id") or sid)
        if key and key not in seen:
            out.append(merged)
            seen.add(key)
    return out


def _eligible_mask(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    close = panel.get("close")
    if not isinstance(close, pd.DataFrame) or close.empty:
        return pd.DataFrame()
    eligible = pd.DataFrame(True, index=close.index, columns=close.columns)
    for field in REQUIRED_RAW_BAR_FIELDS:
        frame = panel.get(field)
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return pd.DataFrame(False, index=close.index, columns=close.columns)
        eligible &= np.isfinite(frame.reindex_like(close).astype(float))
    return eligible.fillna(False).astype(bool)


def _horizon_for_strategy(strategy: dict[str, Any]) -> int:
    for key in ("horizon_bars", "horizon", "label_horizon"):
        try:
            value = int(strategy.get(key))
            if value > 0:
                return value
        except Exception:
            pass
    sid = str(strategy.get("strategy_id") or "")
    if "prior_range" in sid:
        return 5
    return 10


def _feature_row(feats: dict[str, pd.DataFrame], symbols: list[str], ts: pd.Timestamp) -> pd.DataFrame:
    from extreme_price_movements.inference.feature_generator import get_features_for_candidates

    return get_features_for_candidates(feats, symbols, ts=ts)


def _score_strategy(
    *,
    strategy: dict[str, Any],
    panel: dict[str, pd.DataFrame],
    feats: dict[str, pd.DataFrame],
    eligible: pd.DataFrame,
    orchestrator: ModelOrchestrator,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    base_gate_top_frac: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sid = str(strategy.get("strategy_id") or "")
    side = str(strategy.get("side") or strategy.get("trade_side") or "").lower()
    horizon = _horizon_for_strategy(strategy)
    close = panel["close"]
    mask = _build_mask_for_mode(panel, feats, dict(strategy))
    mask = mask.reindex(index=close.index, columns=close.columns).fillna(False).astype(bool)
    eligible_eval = eligible.loc[(eligible.index >= eval_start) & (eligible.index <= eval_end)]
    mask_eval = mask.reindex_like(eligible_eval).fillna(False).astype(bool)
    eval_index = mask_eval.index[mask_eval.index <= close.index.max() - pd.Timedelta(hours=horizon)]
    denom = int(eligible_eval.loc[eval_index].to_numpy(dtype=bool).sum()) if len(eval_index) else 0
    rows: list[dict[str, Any]] = []
    total_masked = 0
    missing_feature_rows = 0
    gate_rank = 1.0 - float(base_gate_top_frac)
    for ts in eval_index:
        candidates = sorted(
            set(mask_eval.columns[mask_eval.loc[ts]].intersection(eligible.columns[eligible.loc[ts]]))
        )
        if not candidates:
            continue
        feature_row = _feature_row(feats, candidates, ts)
        if feature_row.empty:
            missing_feature_rows += len(candidates)
            continue
        candidates = [sym for sym in candidates if sym in feature_row.index]
        if not candidates:
            continue
        total_masked += len(candidates)
        try:
            preds = orchestrator.predict_alpha(feature_row.loc[candidates], side, sid)
        except Exception:
            missing_feature_rows += len(candidates)
            continue
        if not isinstance(preds, pd.Series) or preds.empty:
            missing_feature_rows += len(candidates)
            continue
        preds = preds.reindex(candidates).replace([np.inf, -np.inf], np.nan).dropna()
        if preds.empty:
            continue
        ranks = preds.rank(method="first", pct=True, ascending=True)
        fwd = close.shift(-horizon).loc[ts, preds.index] / close.loc[ts, preds.index] - 1.0
        if side == "short":
            fwd = -fwd
        for symbol, pred in preds.items():
            ret = float(fwd.get(symbol, np.nan))
            if not np.isfinite(ret):
                continue
            rank = float(ranks.loc[symbol])
            rows.append(
                {
                    "timestamp": ts,
                    "symbol": symbol,
                    "strategy_id": sid,
                    "side": side,
                    "horizon_bars": horizon,
                    "base_pred": float(pred),
                    "base_rank_pct": rank,
                    "passes_base_gate": bool(rank >= gate_rank),
                    "raw_pnl": ret,
                    "return": ret,
                }
            )
    diag = {
        "strategy_id": sid,
        "side": side,
        "eligible_opportunities": denom,
        "mask_pass_rows": int(total_masked),
        "mask_pass_pct_of_eligible": float(total_masked / denom) if denom else None,
        "missing_feature_candidate_rows": int(missing_feature_rows),
    }
    return rows, diag


def _threshold_summary(frame: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if frame.empty:
        return out
    total = len(frame)
    gate = frame["passes_base_gate"].astype(bool)
    out["base_gate"] = {
        "rows": int(gate.sum()),
        "pct_of_masked_candidates": float(gate.mean()) if total else None,
    }
    for threshold in (0.7, 0.8, 0.9):
        sel = frame[pd.to_numeric(frame["base_rank_pct"], errors="coerce") >= threshold]
        out[f"base_rank_gte_{threshold:.1f}"] = {
            "trade_count": int(len(sel)),
            "pct_of_masked_candidates": float(len(sel) / total) if total else None,
            "avg_pnl_per_trade_raw": float(sel["raw_pnl"].mean()) if len(sel) else None,
            "mean_return_per_trade": float(sel["return"].mean()) if len(sel) else None,
        }
    return out


def _filter_required_by_portability(
    required: set[str],
    cfg: dict[str, Any],
) -> tuple[set[str], dict[str, str]]:
    mode = str(cfg.get("feature_portability_mode", "legacy")).lower()
    if mode in {"", "legacy", "off"}:
        return set(required), {}
    fixed_basket = bool(cfg.get("feature_portability_fixed_basket", False))
    allow_volume_dependent = bool(
        cfg.get("feature_portability_allow_volume_source_dependent", False)
    )
    allow_dataset_selected = bool(
        cfg.get("feature_portability_allow_dataset_selected", False)
    )
    allow_state_tuned = bool(cfg.get("feature_portability_allow_state_tuned", False))
    kept: set[str] = set()
    dropped: dict[str, str] = {}
    for name in sorted(str(v) for v in required if str(v)):
        if "_G_" in name or name.startswith("G_"):
            dropped[name] = f"gated feature not materializable in feature_portability_mode={mode}"
            continue
        if _is_feature_allowed_for_portability_mode(
            name,
            mode,
            fixed_basket=fixed_basket,
            allow_volume_source_dependent=allow_volume_dependent,
            allow_dataset_selected=allow_dataset_selected,
            allow_state_tuned=allow_state_tuned,
        ):
            kept.add(name)
        else:
            dropped[name] = f"not allowed in feature_portability_mode={mode}"
    return kept, dropped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data_perp/exchanges/krakenfutures"))
    parser.add_argument("--artifact-data-root", type=Path, default=Path("data_perp"))
    parser.add_argument("--current-root", type=Path, default=Path("data_perp"))
    parser.add_argument("--run-id", default="20260321_140000")
    parser.add_argument("--lookback-days", type=int, default=186)
    parser.add_argument("--eval-days", type=int, default=124)
    parser.add_argument("--base-gate-top-frac", type=float, default=0.25)
    parser.add_argument(
        "--kraken-manifest",
        type=Path,
        default=Path("data_perp/exchanges/krakenfutures/manifests/kraken_dual_market_universe_latest.json"),
    )
    parser.add_argument(
        "--bases-json",
        type=Path,
        default=Path("data_perp/exchanges/krakenfutures/manifests/kraken_only_vs_current_universe_20260519.json"),
    )
    parser.add_argument("--max-symbols", type=int, default=0)
    parser.add_argument(
        "--force-required-features",
        action="store_true",
        help=(
            "Force materialization of the model/mask required feature set. "
            "By default the evaluator computes the portable live feature basket "
            "and reports missing required keys instead of failing early."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_perp/exchanges/krakenfutures/artifacts/kraken_only_reference_eval"),
    )
    args = parser.parse_args()

    end_ts = pd.Timestamp.now(tz="UTC").floor("1h")
    start_ts = end_ts - pd.Timedelta(days=int(args.lookback_days))
    eval_start = max(start_ts, end_ts - pd.Timedelta(days=int(args.eval_days)))
    symbols = _kraken_only_symbols(
        kraken_manifest=args.kraken_manifest,
        current_root=args.current_root,
        bases_json=args.bases_json,
    )
    if args.max_symbols and args.max_symbols > 0:
        symbols = symbols[: args.max_symbols]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not symbols:
        summary = {"status": "no_symbols", "reason": "Kraken-only reference universe is empty"}
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        return 1

    panel = _load_panel(data_root=args.data_root, symbols=symbols, start_ts=start_ts, end_ts=end_ts)
    close = panel.get("close")
    if not isinstance(close, pd.DataFrame) or close.empty:
        summary = {"status": "no_ohlcv", "symbols": symbols}
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        return 1
    eligible = _eligible_mask(panel)
    eligible_counts = eligible.loc[eligible.index >= eval_start].sum().sort_values(ascending=False)
    usable_symbols = eligible_counts[eligible_counts > 0].index.astype(str).tolist()
    if not usable_symbols:
        summary = {
            "status": "no_reference_overlap",
            "symbols_loaded": list(close.columns.astype(str)),
            "required_fields": REQUIRED_RAW_BAR_FIELDS,
            "eval_start": eval_start.isoformat(),
            "eval_end": end_ts.isoformat(),
        }
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        return 1
    panel = {k: v.reindex(columns=usable_symbols) for k, v in panel.items() if isinstance(v, pd.DataFrame)}
    eligible = eligible.reindex(columns=usable_symbols).fillna(False).astype(bool)

    state = load_full_state(args.run_id, str(args.artifact_data_root))
    mask_rows = _load_lgbm_strategy_mask_rows(str(args.artifact_data_root), args.run_id, market_mode="perps")
    required = raw_required_feature_keys(get_inference_required_feature_keys(state, None))
    required |= set(_lgbm_mask_required_feature_keys(mask_rows))
    feature_cfg = load_inference_config(
        data_root=str(args.artifact_data_root),
        run_id=args.run_id,
        market_mode="perps",
    )
    runtime_cfg = dict(feature_cfg.get("runtime_cfg") or {})
    runtime_cfg.update(
        {
            "use_perps": True,
            "market_mode": "perps",
            "data_root": str(args.data_root),
            "artifact_data_root": str(args.artifact_data_root),
            "live_data_root": str(args.data_root),
            "feature_portability_strict": False,
        }
    )
    state_bundle = state.get("bundle", {}) if isinstance(state.get("bundle"), dict) else {}
    runtime_cfg.setdefault("bundle", state_bundle)
    for key in ("feature_transform_contract", "feature_transform_contract_hash", "feature_transform_manifest"):
        value = state.get(key)
        if value is None:
            value = state_bundle.get(key)
        if value is not None:
            feature_cfg[key] = value
            runtime_cfg[key] = value
    feature_cfg["runtime_cfg"] = runtime_cfg
    required_for_compute, dropped_required = _filter_required_by_portability(
        required,
        runtime_cfg,
    )
    forced_required_keys = required_for_compute if args.force_required_features else None
    feats = load_or_compute_features(
        panel,
        usable_symbols,
        args.run_id,
        str(args.data_root),
        feature_cfg,
        lookback_hours=int(args.lookback_days) * 24,
        required_feature_keys=forced_required_keys,
    )
    policy = _load_json(args.artifact_data_root / "artifacts" / args.run_id / "strategy_for_inference_perps.json")
    strategies = _dedupe_mask_rows(list(policy.get("strategies") or []), mask_rows)
    orchestrator = ModelOrchestrator(state, runtime_cfg={"model_bundle": state.get("bundle", {})})

    all_rows: list[dict[str, Any]] = []
    strategy_diags: list[dict[str, Any]] = []
    for strategy in strategies:
        rows, diag = _score_strategy(
            strategy=strategy,
            panel=panel,
            feats=feats,
            eligible=eligible,
            orchestrator=orchestrator,
            eval_start=eval_start,
            eval_end=end_ts,
            base_gate_top_frac=float(args.base_gate_top_frac),
        )
        all_rows.extend(rows)
        strategy_diags.append(diag)

    trades = pd.DataFrame(all_rows)
    if not trades.empty:
        trades.to_parquet(args.output_dir / "candidate_scores.parquet", index=False)
        trades.to_csv(args.output_dir / "candidate_scores.csv", index=False)

    missing_required_features = sorted(str(k) for k in required if str(k) not in feats)
    feature_issues = {
        "missing_required_features": missing_required_features,
        "dropped_required_by_portability_policy": dropped_required,
        "forced_required_features": bool(args.force_required_features),
        "required_fields": REQUIRED_RAW_BAR_FIELDS,
        "eligible_symbol_rows": {str(k): int(v) for k, v in eligible_counts.items() if int(v) > 0},
    }
    summary = {
        "status": "ok" if not trades.empty else "no_masked_candidates",
        "run_id": args.run_id,
        "symbols_requested": symbols,
        "symbols_with_reference_overlap": usable_symbols,
        "eval_start": eval_start.isoformat(),
        "eval_end": end_ts.isoformat(),
        "strategies": strategy_diags,
        "overall": _threshold_summary(trades),
        "by_strategy": {
            sid: _threshold_summary(group)
            for sid, group in trades.groupby("strategy_id")
        }
        if not trades.empty
        else {},
        "problematic_feature_computations": feature_issues,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0 if not trades.empty else 1


if __name__ == "__main__":
    raise SystemExit(main())
