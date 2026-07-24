#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extreme_price_movements.config import CFG  # noqa: E402
from extreme_price_movements.data_store import (  # noqa: E402
    PartitionedOHLCVStore,
    scoped_data_root,
)
from extreme_price_movements.features import (  # noqa: E402
    add_regime_gates,
    compute_features_hourly,
    compute_market_features,
)
from extreme_price_movements.features_gmm_ae import (  # noqa: E402
    AE_GMM_FEATURE_COLUMNS,
)
from extreme_price_movements.inference.feature_generator import (  # noqa: E402
    _is_live_synthesized_feature_key,
    get_features_for_candidates,
    get_inference_required_feature_keys,
    load_or_compute_features,
    live_model_feature_store_strict,
    raw_required_feature_keys,
)
from extreme_price_movements.inference.live_meta_feature_overlays import (  # noqa: E402
    apply_live_meta_reliability_priors,
    live_ae_gmm_input_feature_columns,
    load_live_ae_gmm_state_payload,
    load_meta_reliability_prior_payload,
    load_residual_reference_prior_payload,
    materialize_live_ae_gmm_features,
    materialize_live_source_regime_features,
)
from extreme_price_movements.inference.live_residual_event_state import (  # noqa: E402
    load_live_residual_event_state_payload,
    materialize_live_residual_event_features,
    residual_event_state_input_feature_columns,
)
from extreme_price_movements.inference.run_inference import (  # noqa: E402
    _build_residual_event_feature_runtime_cfg,
    _hydrate_optional_frozen_features,
    _lgbm_mask_required_feature_keys,
    _live_regime_calibration_raw_feature_columns,
    _load_lgbm_strategy_mask_rows,
)
from extreme_price_movements.inference.config import (  # noqa: E402
    load_inference_config,
    load_trained_symbol_universe,
)
from extreme_price_movements.inference.canonical_meta_postprocessor import (  # noqa: E402
    CanonicalMetaPostprocessor,
)
from extreme_price_movements.inference.data_fetcher import (  # noqa: E402
    MICRODATA_FRAME_FIELDS,
    PERP_OHLCV_EXTRA_FIELDS,
)
from extreme_price_movements.inference.symbol_mapping import symbol_bases  # noqa: E402
from extreme_price_movements.inference.model_orchestrator import (  # noqa: E402
    DELETED_MODEL_FEATURE_KEYS,
    FeatureParityError,
    ModelOrchestrator,
    _effective_selected_feature_contract,
    _strict_finite_model_matrix,
)

from extreme_price_movements.inference.parity import (  # noqa: E402
    calibrated_score_and_threshold,
    strategy_core_id,
)
from extreme_price_movements.inference.policy_rank_reference import (  # noqa: E402
    PolicyRankReferenceStore,
)
from extreme_price_movements.inference.side_residual_expert import (  # noqa: E402
    SideResidualExpertBundle,
)
from extreme_price_movements.raw_market_data_contract import (  # noqa: E402
    load_raw_market_panel,
)
from extreme_price_movements.inference.threshold_basis_policy import (  # noqa: E402
    apply_threshold_basis_policy_to_decisions,
    load_threshold_basis_policy,
)
from extreme_price_movements.model_loader import load_full_state  # noqa: E402
from extreme_price_movements.simple_position_sizer import (  # noqa: E402
    load_calibration_curves,
)

AE_GMM_FEATURE_SET = {str(column) for column in AE_GMM_FEATURE_COLUMNS}


def _slice_panel(
    panel: dict[str, pd.DataFrame],
    *,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    start_ts = pd.Timestamp(start_ts)
    end_ts = pd.Timestamp(end_ts)
    for name, frame in panel.items():
        if not isinstance(frame, pd.DataFrame) or not isinstance(
            frame.index, pd.DatetimeIndex
        ):
            out[name] = frame
            continue
        out[name] = frame.loc[(frame.index >= start_ts) & (frame.index <= end_ts)]
    return out


def _normalise_symbol(symbol: object) -> str:
    text = str(symbol or "").upper().strip()
    perp_suffix = ""
    for suffix in (":USDT", ":USDC", ":USD"):
        if text.endswith(suffix):
            perp_suffix = suffix
            text = text[: -len(suffix)]
            break
    text = text.replace("-", "/").replace("_", "/")
    if "/" not in text:
        if text.endswith("USDC"):
            text = f"{text[:-4]}/USDC"
        elif text.endswith("USDT"):
            text = f"{text[:-4]}/USDT"
        elif text.endswith("USD"):
            text = f"{text[:-3]}/USD"
    return f"{text}{perp_suffix}" if perp_suffix and ":" not in text else text


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _safe_int(value: Any, default: int = 0) -> int:
    numeric = _safe_float(value)
    return int(numeric) if np.isfinite(numeric) else int(default)


def _is_present(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    text = str(value).strip()
    return bool(text) and text.lower() not in {"nan", "none", "null"}


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _logged_decision_feature_keys(decisions: pd.DataFrame) -> set[str]:
    """Return exact feature keys logged by live inference for replay decisions."""
    keys: set[str] = set()
    for col in (
        "base_model_features_json",
        "meta_model_features_json",
        "meta_postprocessor_input_features_json",
    ):
        if col not in decisions.columns:
            continue
        for raw in decisions[col].dropna():
            try:
                parsed = json.loads(raw) if isinstance(raw, str) else raw
            except Exception:
                continue
            if isinstance(parsed, list):
                keys.update(str(v) for v in parsed if str(v))
            elif isinstance(parsed, dict):
                keys.update(str(v) for v in parsed.keys() if str(v))
    return keys


def _json_mapping(raw: Any) -> dict[str, Any]:
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _feature_value_delta_summary(
    *,
    logged_values_raw: Any,
    feature_row: pd.DataFrame,
    symbol: str,
    feature_filter: Any = None,
) -> dict[str, Any]:
    logged_values = _json_mapping(logged_values_raw)
    if not logged_values or feature_row.empty or symbol not in feature_row.index:
        return {
            "count": 0,
            "max_abs": float("nan"),
            "mean_abs": float("nan"),
            "worst_feature": "",
            "worst_live_value": float("nan"),
            "worst_replay_value": float("nan"),
        }
    deltas: list[tuple[str, float]] = []
    row = feature_row.loc[symbol]
    for key, live_value in logged_values.items():
        key_s = str(key)
        if callable(feature_filter) and not bool(feature_filter(key_s)):
            continue
        if key not in row.index:
            continue
        live_float = _safe_float(live_value)
        replay_float = _safe_float(row.get(key))
        if np.isfinite(live_float) and np.isfinite(replay_float):
            deltas.append((key_s, abs(replay_float - live_float)))
    if not deltas:
        return {
            "count": 0,
            "max_abs": float("nan"),
            "mean_abs": float("nan"),
            "worst_feature": "",
            "worst_live_value": float("nan"),
            "worst_replay_value": float("nan"),
        }
    worst_feature, max_abs = max(deltas, key=lambda item: item[1])
    return {
        "count": len(deltas),
        "max_abs": float(max_abs),
        "mean_abs": float(np.mean([delta for _, delta in deltas])),
        "worst_feature": worst_feature,
        "worst_live_value": _safe_float(logged_values.get(worst_feature)),
        "worst_replay_value": _safe_float(row.get(worst_feature)),
    }


def _feature_value_contract_summary(
    *,
    logged_values_raw: Any,
    feature_row: pd.DataFrame,
    symbol: str,
    categorical_values: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Report exact finite coverage as well as numerical feature drift."""
    logged_values = _json_mapping(logged_values_raw)
    if not logged_values or feature_row.empty or symbol not in feature_row.index:
        return {
            "logged_count": int(len(logged_values)),
            "common_finite_count": 0,
            "missing_count": int(len(logged_values)),
            "missing_features": sorted(str(key) for key in logged_values),
            "max_abs": float("nan"),
            "mean_abs": float("nan"),
            "worst_feature": "",
        }
    row = feature_row.loc[symbol]
    missing: list[str] = []
    deltas: list[tuple[str, float]] = []
    categorical_values = categorical_values or {}
    for key, live_value in logged_values.items():
        key_s = str(key)
        if key_s in categorical_values:
            if str(live_value) != str(categorical_values[key_s]):
                missing.append(key_s)
            continue
        live_float = _safe_float(live_value)
        if key_s not in row.index:
            missing.append(key_s)
            continue
        replay_float = _safe_float(row.get(key_s))
        if np.isfinite(live_float) != np.isfinite(replay_float):
            missing.append(key_s)
            continue
        if np.isfinite(live_float):
            deltas.append((key_s, abs(replay_float - live_float)))
    worst_feature, max_abs = (
        max(deltas, key=lambda item: item[1]) if deltas else ("", float("nan"))
    )
    worst_live_value = _safe_float(logged_values.get(worst_feature))
    worst_replay_value = (
        _safe_float(row.get(worst_feature))
        if worst_feature and worst_feature in row.index
        else float("nan")
    )
    return {
        "logged_count": int(len(logged_values)),
        "common_finite_count": int(len(deltas)),
        "missing_count": int(len(missing)),
        "missing_features": sorted(missing),
        "max_abs": float(max_abs),
        "mean_abs": (
            float(np.mean([delta for _, delta in deltas]))
            if deltas
            else float("nan")
        ),
        "worst_feature": worst_feature,
        "worst_live_value": worst_live_value,
        "worst_replay_value": worst_replay_value,
    }


def _live_synthesized_feature_delta_summary(
    *,
    logged_values_raw: Any,
    feature_row: pd.DataFrame,
    symbol: str,
) -> dict[str, Any]:
    return _feature_value_delta_summary(
        logged_values_raw=logged_values_raw,
        feature_row=feature_row,
        symbol=symbol,
        feature_filter=_is_live_synthesized_feature_key,
    )


def _logged_values_frame(raw: Any, *, symbol: str) -> pd.DataFrame:
    values = _json_mapping(raw)
    if not values:
        return pd.DataFrame(index=[symbol])
    row: dict[str, float] = {}
    for key, value in values.items():
        val = _safe_float(value)
        row[str(key)] = val if np.isfinite(val) else np.nan
    return pd.DataFrame([row], index=[symbol], dtype=np.float32)


def _resolve_meta_model(
    orchestrator: ModelOrchestrator,
    *,
    side: str,
    strategy_id: str,
) -> tuple[str, Any]:
    meta_models = getattr(orchestrator, "meta_models", {}) or {}
    core = strategy_core_id(str(strategy_id))
    candidates = [
        str(strategy_id),
        core,
        f"{side}_{strategy_id}",
        f"{side}_{core}",
        f"{strategy_id}_clf",
        f"{core}_clf",
        f"{side}_{strategy_id}_clf",
        f"{side}_{core}_clf",
        f"{strategy_id}_tbm_clf",
        f"{core}_tbm_clf",
        f"{side}_{strategy_id}_tbm_clf",
        f"{side}_{core}_tbm_clf",
    ]
    for key in candidates:
        if key in meta_models:
            return key, meta_models.get(key)
    return str(strategy_id), None


def _predict_exact_logged_meta_input(
    *,
    orchestrator: ModelOrchestrator,
    side: str,
    strategy_id: str,
    logged_meta_frame: pd.DataFrame,
    apply_score_alignment: bool = True,
    base_prediction: float = np.nan,
) -> float:
    if logged_meta_frame.empty:
        return float("nan")
    _, meta_model = _resolve_meta_model(
        orchestrator,
        side=side,
        strategy_id=strategy_id,
    )
    if meta_model is None:
        return float("nan")
    feat_cols = _effective_selected_feature_contract(meta_model)
    if not feat_cols and hasattr(meta_model, "feature_columns"):
        feat_cols = [str(c) for c in (getattr(meta_model, "feature_columns", []) or [])]
    feat_cols = [
        str(c)
        for c in (feat_cols or [])
        if str(c) not in DELETED_MODEL_FEATURE_KEYS
    ]
    if not feat_cols:
        return float("nan")
    missing_before_overlay = [c for c in feat_cols if c not in logged_meta_frame.columns]
    if missing_before_overlay:
        runtime_cfg = getattr(orchestrator, "cfg", {}) or {}
        data_root = str(
            runtime_cfg.get("artifact_data_root")
            or runtime_cfg.get("data_root")
            or "data_perp"
        )
        run_id = str(
            runtime_cfg.get("model_artifact_run_id")
            or runtime_cfg.get("run_id")
            or ""
        )
        prior_payload = (
            load_meta_reliability_prior_payload(data_root, run_id)
            if run_id
            else {}
        )
        if prior_payload and np.isfinite(base_prediction):
            symbol = str(logged_meta_frame.index[0])
            logged_meta_frame = apply_live_meta_reliability_priors(
                logged_meta_frame,
                side=side,
                base_predictions={symbol: {"base_pred": float(base_prediction)}},
                prior_payload=prior_payload,
            )
    missing = [c for c in feat_cols if c not in logged_meta_frame.columns]
    if missing:
        return float("nan")
    try:
        X = _strict_finite_model_matrix(
            logged_meta_frame.reindex(columns=feat_cols),
            model_feature_cols=feat_cols,
            model_key=str(strategy_id),
        )
    except FeatureParityError:
        return float("nan")
    pred = meta_model.predict(X)
    arr = np.asarray(pred, dtype=float).reshape(-1)
    score_alignment = getattr(meta_model, "s52_meta_score_alignment_", None)
    if (
        apply_score_alignment
        and isinstance(score_alignment, dict)
        and score_alignment.get("enabled")
    ):
        from extreme_price_movements.inference.s52_meta_score_alignment import (
            apply_s52_meta_score_alignment,
        )

        arr = np.asarray(
            apply_s52_meta_score_alignment(
                arr,
                score_alignment,
                side=side,
            ),
            dtype=float,
        ).reshape(-1)
    return _safe_float(arr[0]) if arr.size else float("nan")


def _load_recent_decisions(
    *,
    ledger_path: Path,
    trades_path: Path,
    max_rows: int,
    decision_start: str | None = None,
    require_rank_source: str | None = None,
) -> pd.DataFrame:
    ledger = _read_table(ledger_path)
    if ledger.empty:
        return pd.DataFrame()
    ledger = ledger.copy()
    for col in ("timestamp", "decision_ts", "signal_bar_ts"):
        if col in ledger.columns:
            ledger[col] = pd.to_datetime(ledger[col], utc=True, errors="coerce")
    ledger = ledger.dropna(subset=["signal_bar_ts", "symbol", "side", "strategy_id"])
    ledger["symbol"] = ledger["symbol"].map(_normalise_symbol)
    ledger["strategy_core"] = ledger["strategy_id"].map(strategy_core_id)
    if decision_start:
        start_ts = pd.Timestamp(decision_start)
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
        else:
            start_ts = start_ts.tz_convert("UTC")
        ledger = ledger[ledger["decision_ts"] >= start_ts]
    if require_rank_source:
        if "rank_score_source" not in ledger.columns:
            return pd.DataFrame()
        ledger = ledger[
            ledger["rank_score_source"].astype(str).eq(str(require_rank_source))
        ]
    ledger = ledger.sort_values("decision_ts").tail(max_rows).reset_index(drop=True)
    live_defaults = {
        "base_pred": "live_base_pred",
        "meta_pred": "live_meta_pred",
        "calibrated_score": "live_calibrated_score",
        "normalized_rank_score": "live_rank_percentile",
        "policy_rank_pct": "live_policy_rank_pct",
        "rank_score_source": "live_rank_score_source",
        "threshold_basis_rank_score": "live_threshold_basis_rank_score",
        "threshold_basis_policy_id": "live_threshold_basis_policy_id",
        "threshold_basis_dynamic_score_threshold": "live_threshold_basis_dynamic_score_threshold",
        "threshold_basis_dynamic_ev_target": "live_threshold_basis_dynamic_ev_target",
    }
    for src, dst in live_defaults.items():
        if src in ledger.columns and dst not in ledger.columns:
            ledger[dst] = ledger[src]
    rank_source = (
        ledger.get("live_rank_score_source", pd.Series("", index=ledger.index))
        .fillna("")
        .astype(str)
    )
    threshold_basis_mask = rank_source.str.startswith("threshold_basis:")
    if threshold_basis_mask.any() and "live_threshold_basis_rank_score" in ledger.columns:
        threshold_rank = pd.to_numeric(
            ledger["live_threshold_basis_rank_score"], errors="coerce"
        )
        valid_threshold_rank = threshold_basis_mask & threshold_rank.notna()
        # Threshold-basis policies intentionally override the legacy
        # policy-rank CDF.  Keep the old CDF columns for diagnostics, but make
        # the live policy rank used by this parity checker match the actual
        # deployed gate/rank value.
        ledger.loc[valid_threshold_rank, "live_policy_rank_pct"] = threshold_rank.loc[
            valid_threshold_rank
        ]
        ledger.loc[valid_threshold_rank, "live_rank_percentile"] = threshold_rank.loc[
            valid_threshold_rank
        ]

    trades = _read_table(trades_path)
    if trades.empty:
        return ledger
    trades = trades.copy()
    if "timestamp" in trades.columns:
        trades["timestamp"] = pd.to_datetime(trades["timestamp"], utc=True, errors="coerce")
    trades = trades[trades.get("lifecycle_event", "").astype(str).isin(["entry_placed", "entry_rejected"])]
    if trades.empty:
        return ledger
    trades["symbol"] = trades["symbol"].map(_normalise_symbol)
    trades["strategy_core"] = trades["strategy_id"].map(strategy_core_id)
    keep_cols = [
        c
        for c in (
            "timestamp",
            "symbol",
            "side",
            "strategy_core",
            "base_pred",
            "meta_pred",
            "calibrated_score",
            "rank_percentile",
            "policy_rank_pct",
            "rank_score_source",
        )
        if c in trades.columns
    ]
    trades = trades[keep_cols].sort_values("timestamp")

    merged_rows: list[dict[str, Any]] = []
    for _, row in ledger.iterrows():
        out = row.to_dict()
        mask = (
            trades["symbol"].eq(row["symbol"])
            & trades["side"].astype(str).str.lower().eq(str(row["side"]).lower())
            & trades["strategy_core"].eq(row["strategy_core"])
        )
        candidates = trades.loc[mask].copy()
        if not candidates.empty and pd.notna(row.get("decision_ts")):
            candidates["_dt"] = (
                candidates["timestamp"] - row["decision_ts"]
            ).abs()
            candidates = candidates[candidates["_dt"] <= pd.Timedelta("15min")]
            if not candidates.empty:
                trade_row = candidates.sort_values("_dt").iloc[0]
                for col in (
                    "base_pred",
                    "meta_pred",
                    "calibrated_score",
                    "rank_percentile",
                    "policy_rank_pct",
                    "rank_score_source",
                ):
                    if col in trade_row:
                        dst = f"live_{col}"
                        # The prediction ledger is the source of truth for
                        # model and rank values.  Trade logs may contain stale
                        # or execution-path copies; use them only to fill older
                        # ledgers that did not record the live fields.
                        if not _is_present(out.get(dst)):
                            out[dst] = trade_row[col]
        merged_rows.append(out)
    return pd.DataFrame(merged_rows)


def _margin_cache_quote_symbols(live_quote_currency: str = "USDC") -> set[str]:
    """Best-effort offline mirror of the live margin universe resolver."""
    quote = str(live_quote_currency or "USDC").upper()
    cache_path = REPO_ROOT / "extreme_price_movements" / ".margin_universe_cache.json"
    if not cache_path.exists():
        return set()
    try:
        data = json.loads(cache_path.read_text())
    except Exception:
        return set()
    out: set[str] = set()
    for row in data if isinstance(data, list) else []:
        if not isinstance(row, dict):
            continue
        if str(row.get("status") or "").upper() != "TRADING":
            continue
        if str(row.get("quoteAsset") or row.get("quote") or "").upper() != quote:
            continue
        if not bool(row.get("isMarginTradingAllowed", True)):
            continue
        base = str(row.get("baseAsset") or row.get("base") or "").upper().strip()
        if base:
            out.add(f"{base}/{quote}")
    return out


def _market_data_root(
    data_root: Path,
    *,
    market_mode: str = "spot",
    exchange_id: str = "krakenfutures",
) -> Path:
    """Resolve the exchange-scoped market-data root used by live inference."""
    cfg = {
        "data_root": str(data_root),
        "exchange_id": exchange_id,
        "exchange": exchange_id,
        "market_mode": market_mode,
        "use_perps": str(market_mode).lower() == "perps",
    }
    scoped = Path(scoped_data_root(cfg))
    if (scoped / "ohlcv").exists() or not (Path(data_root) / "ohlcv").exists():
        return scoped
    return Path(data_root)


def _local_quote_symbols(
    data_root: Path,
    *,
    run_id: str | None = None,
    live_quote_currency: str = "USDC",
    market_mode: str = "spot",
    exchange_id: str = "krakenfutures",
) -> list[str]:
    out: list[str] = []
    quote = str(live_quote_currency or "USDC").upper()
    market_root = _market_data_root(
        data_root,
        market_mode=market_mode,
        exchange_id=exchange_id,
    )
    for path in sorted((market_root / "ohlcv").glob("symbol=*")):
        name = path.name.replace("symbol=", "")
        symbol = _normalise_symbol(name)
        if symbol.endswith(f"/{quote}") or symbol.endswith(f"/{quote}:{quote}"):
            out.append(symbol)
    symbols = set(out)
    if run_id:
        trained = load_trained_symbol_universe(str(data_root), str(run_id))
        if trained:
            trained_bases = symbol_bases(trained)
            symbols = {
                sym for sym in symbols if _normalise_symbol(sym).split("/", 1)[0] in trained_bases
            }
        margin_symbols = (
            set() if str(market_mode).lower() == "perps" else _margin_cache_quote_symbols(quote)
        )
        if margin_symbols:
            symbols &= margin_symbols
    return sorted(symbols)


def _live_feature_cache_symbols_for_end(
    data_root: Path,
    *,
    run_id: str,
    end_ts: pd.Timestamp,
    live_quote_currency: str = "USDC",
) -> list[str]:
    """Return the exact symbol universe from a live feature cache for ``end_ts``."""
    target = pd.Timestamp(end_ts)
    target = target.tz_localize("UTC") if target.tzinfo is None else target.tz_convert("UTC")
    roots: list[tuple[int, Path]] = [
        (
            0,
            Path(data_root)
            / "artifacts"
            / str(run_id)
            / "live_selected_feature_latest_matrix",
        ),
        (1, Path("cache") / "inference_live_features" / str(run_id)),
    ]
    candidates: list[tuple[int, int, float, list[str]]] = []
    for priority, root in roots:
        if not root.exists():
            continue
        for meta_path in root.glob("**/meta.json"):
            try:
                meta = json.loads(meta_path.read_text())
                raw_end = (
                    meta.get("end_ts")
                    or meta.get("target_end_ts")
                    or meta.get("feature_end_ts")
                )
                if not raw_end:
                    continue
                cache_end = pd.Timestamp(raw_end)
                cache_end = (
                    cache_end.tz_localize("UTC")
                    if cache_end.tzinfo is None
                    else cache_end.tz_convert("UTC")
                )
                if cache_end != target:
                    continue
                quote = str(live_quote_currency or "USDC").upper()
                symbols = [_normalise_symbol(s) for s in (meta.get("symbols") or [])]
                symbols = sorted(
                    {
                        s
                        for s in symbols
                        if s.endswith(f"/{quote}") or s.endswith(f"/{quote}:{quote}")
                    }
                )
                if symbols:
                    candidates.append(
                        (priority, len(symbols), float(meta_path.stat().st_mtime), symbols)
                    )
            except Exception:
                continue
    if not candidates:
        return []
    # Prefer the artifact selected-feature sidecar over generic runtime caches:
    # it is the production inference universe and replay/debug runs can add
    # broader local-OHLCV caches for the same timestamp after the fact.  Within
    # the same source, use the smallest non-trivial universe so later debug
    # sidecars do not widen the batch and perturb AE/GMM/context features.
    non_trivial = [item for item in candidates if item[1] >= 25]
    return sorted(non_trivial or candidates, key=lambda item: (item[0], item[1], item[2]))[0][3]


def _source_parity_context_symbols_for_end(
    data_root: Path,
    *,
    run_id: str,
    end_ts: pd.Timestamp,
    live_quote_currency: str = "USDC",
    market_mode: str = "spot",
    exchange_id: str = "krakenfutures",
) -> list[str]:
    """Recover the exact stable compute universe recorded by live inference.

    Live inference computes market and cross-sectional features on its stable
    model universe, then applies source eligibility to masks and orders.  The
    persisted source report therefore defines the compute universe as the union
    of accepted and rejected symbols, not accepted symbols alone.
    """
    target = pd.Timestamp(end_ts)
    target = target.tz_localize("UTC") if target.tzinfo is None else target.tz_convert("UTC")
    stamp = target.strftime("%Y%m%dT%H%M%SZ")
    market_root = _market_data_root(
        Path(data_root),
        market_mode=market_mode,
        exchange_id=exchange_id,
    )
    roots = [Path(data_root)]
    if market_root != Path(data_root):
        roots.insert(0, market_root)
    quote = str(live_quote_currency or "USDC").upper()
    for root in roots:
        report_path = (
            root
            / "artifacts"
            / str(run_id)
            / "live_source_parity"
            / f"{stamp}_model_sources.json"
        )
        if not report_path.exists():
            continue
        try:
            payload = json.loads(report_path.read_text())
        except Exception:
            continue
        report_end = pd.to_datetime(payload.get("end_ts"), utc=True, errors="coerce")
        if pd.isna(report_end) or pd.Timestamp(report_end) != target:
            continue
        symbols = {
            _normalise_symbol(symbol)
            for key in ("accepted_symbols", "rejected_symbols")
            for symbol in (payload.get(key) or [])
        }
        symbols = {
            symbol
            for symbol in symbols
            if symbol.endswith(f"/{quote}") or symbol.endswith(f"/{quote}:{quote}")
        }
        if symbols:
            return sorted(symbols)
    return []


def _feature_source_symbols(
    data_root: Path,
    *,
    feature_source_run_id: str | None,
    live_quote_currency: str = "USDC",
) -> list[str]:
    """Recover the compute universe from the pinned feature-source artifact."""
    if not feature_source_run_id:
        return []
    root = Path(data_root) / "features" / str(feature_source_run_id)
    if not root.exists():
        return []
    quote = str(live_quote_currency or "USDC").upper()
    symbols = {
        _normalise_symbol(path.stem.removeprefix("symbol="))
        for path in root.glob("symbol=*.parquet")
    }
    return sorted(
        symbol
        for symbol in symbols
        if symbol.endswith(f"/{quote}") or symbol.endswith(f"/{quote}:{quote}")
    )


def _load_panel(
    *,
    data_root: Path,
    symbols: list[str],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    market_mode: str = "spot",
    exchange_id: str = "krakenfutures",
) -> dict[str, pd.DataFrame]:
    market_root = _market_data_root(
        data_root,
        market_mode=market_mode,
        exchange_id=exchange_id,
    )
    store = PartitionedOHLCVStore(str(market_root), timeframe="1h")
    panel_fields = (
        "open",
        "high",
        "low",
        "close",
        "volume",
        *PERP_OHLCV_EXTRA_FIELDS,
    )
    try:
        workers = int(os.getenv("EPM_RAW_MARKET_DATA_LOAD_WORKERS", "8") or "8")
    except (TypeError, ValueError):
        workers = 8

    def _microdata_loader(
        requested_symbols: list[str] | tuple[str, ...],
        requested_start: pd.Timestamp | None,
        requested_end: pd.Timestamp | None,
    ) -> dict[str, pd.DataFrame]:
        return _load_microdata_panel(
            data_root=market_root,
            symbols=list(requested_symbols),
            start_ts=requested_start if requested_start is not None else start_ts,
            end_ts=requested_end if requested_end is not None else end_ts,
        )

    return load_raw_market_panel(
        store=store,
        symbols=symbols,
        panel_fields=panel_fields,
        start_ts=start_ts,
        end_ts=end_ts,
        max_workers=workers,
        microdata_loader=_microdata_loader,
    )


def _symbol_file_key(symbol: str) -> str:
    return str(symbol).replace("/", "_").replace(":", "_")


def _load_microdata_panel(
    *,
    data_root: Path,
    symbols: list[str],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    orderbook_dir = data_root / "orderbook_hourly"
    funding_dir = data_root / "funding_hourly"
    open_interest_dir = data_root / "open_interest_hourly"
    orderbook_fields = (
        "mid",
        "best_bid",
        "best_ask",
        "bid_qty_1",
        "ask_qty_1",
        "cum_bid_qty_l10",
        "cum_ask_qty_l10",
        "cum_bid_qty_l20",
        "cum_ask_qty_l20",
        "snapshot_ts",
        "trade_count_1h",
        "buy_qty_1h",
        "sell_qty_1h",
        "notional_1h",
        "buy_notional_1h",
        "sell_notional_1h",
        "vwap_1h",
        "mean_trade_qty_1h",
        "signed_flow_imbalance_1h",
    )
    by_orderbook: dict[str, dict[str, pd.Series]] = {field: {} for field in orderbook_fields}
    by_microdata: dict[str, dict[str, pd.Series]] = {
        field: {} for field in MICRODATA_FRAME_FIELDS
    }
    idx_union = None
    start = pd.Timestamp(start_ts)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = pd.Timestamp(end_ts)
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    for symbol in symbols:
        key = _symbol_file_key(symbol)
        ob_path = orderbook_dir / f"{key}.parquet"
        if ob_path.exists():
            try:
                ob = pd.read_parquet(ob_path)
                ob.index = pd.to_datetime(ob.index, utc=True)
                ob = ob.loc[(ob.index >= start) & (ob.index <= end)]
            except Exception:
                ob = pd.DataFrame()
            if not ob.empty:
                idx_union = ob.index if idx_union is None else idx_union.union(ob.index)
                for field in orderbook_fields:
                    if field in ob.columns:
                        by_orderbook[field][symbol] = pd.to_numeric(
                            ob[field], errors="coerce"
                        ).astype(np.float32)
        funding_path = funding_dir / f"{key}.parquet"
        if funding_path.exists():
            try:
                fr = pd.read_parquet(funding_path)
                fr.index = pd.to_datetime(fr.index, utc=True)
                fr = fr.loc[(fr.index >= start) & (fr.index <= end)]
            except Exception:
                fr = pd.DataFrame()
            if not fr.empty:
                idx_union = fr.index if idx_union is None else idx_union.union(fr.index)
                for field in MICRODATA_FRAME_FIELDS:
                    if field in fr.columns:
                        by_microdata[field][symbol] = pd.to_numeric(
                            fr[field], errors="coerce"
                        ).astype(np.float32)
        oi_path = open_interest_dir / f"{key}.parquet"
        if oi_path.exists():
            try:
                oi = pd.read_parquet(oi_path)
                oi.index = pd.to_datetime(oi.index, utc=True)
                oi = oi.loc[(oi.index >= start) & (oi.index <= end)]
                oi = oi[~oi.index.duplicated(keep="last")].sort_index()
            except Exception:
                oi = pd.DataFrame()
            if not oi.empty:
                oi_col = next(
                    (
                        candidate
                        for candidate in (
                            "open_interest",
                            "openInterestValue",
                            "sumOpenInterestValue",
                            "openInterestAmount",
                            "openInterest",
                            "sumOpenInterest",
                        )
                        if candidate in oi.columns
                    ),
                    None,
                )
                if oi_col is not None:
                    oi_series = pd.to_numeric(
                        oi[oi_col], errors="coerce"
                    ).astype(np.float32)
                    existing = by_microdata["open_interest"].get(symbol)
                    by_microdata["open_interest"][symbol] = (
                        existing.combine_first(oi_series)
                        if existing is not None
                        else oi_series
                    )
                    idx_union = (
                        oi.index
                        if idx_union is None
                        else idx_union.union(oi.index)
                    )
    if idx_union is None:
        return {}
    idx_union = pd.DatetimeIndex(idx_union).sort_values().unique()
    out: dict[str, pd.DataFrame] = {}
    if by_orderbook["mid"]:
        out["orderbook_hourly"] = (
            pd.DataFrame(by_orderbook["mid"]).reindex(idx_union).astype(np.float32)
        )
    for field, by_symbol in by_orderbook.items():
        if by_symbol:
            out[f"orderbook_{field}"] = (
                pd.DataFrame(by_symbol).reindex(idx_union).astype(np.float32)
            )
    for field, by_symbol in by_microdata.items():
        if by_symbol:
            out[field] = (
                pd.DataFrame(by_symbol).reindex(idx_union).ffill().astype(np.float32)
            )
    return out


def _score_row(
    *,
    row: pd.Series,
    feats: dict[str, pd.DataFrame],
    orchestrator: ModelOrchestrator,
    calibration_data: dict[str, dict[str, Any]],
    rank_store: PolicyRankReferenceStore,
    overlay_required_columns: set[str] | None = None,
    live_ae_gmm_state_payload: Mapping[str, Any] | None = None,
    feature_row_override: pd.DataFrame | None = None,
    base_feature_row_override: pd.DataFrame | None = None,
    skip_full_chain_diagnostics: bool = False,
    canonical_postprocessor: CanonicalMetaPostprocessor | None = None,
    threshold_basis_policy: Mapping[str, Any] | None = None,
    canonical_result_override: pd.DataFrame | None = None,
    prefer_logged_meta_input: bool = True,
) -> dict[str, Any]:
    symbol = str(row["symbol"])
    side = str(row["side"]).lower()
    strategy_id = str(row["strategy_id"])
    core_strategy_id = strategy_core_id(strategy_id)
    model_strategy_id = strategy_id
    if side in {"long", "short"} and core_strategy_id and not str(strategy_id).startswith(
        f"{side}_"
    ):
        model_strategy_id = f"{side}_{core_strategy_id}"
    ts = pd.Timestamp(row["signal_bar_ts"])
    override_supplied = isinstance(feature_row_override, pd.DataFrame)
    feature_row = (
        feature_row_override.copy()
        if override_supplied
        else get_features_for_candidates(feats, [symbol], ts=ts)
    )
    base_feature_row = (
        base_feature_row_override.copy()
        if isinstance(base_feature_row_override, pd.DataFrame)
        else feature_row
    )
    if not feature_row.empty:
        feature_row = feature_row.copy()
        feature_row["side"] = np.float32(
            1.0 if str(side).lower().startswith("long") else -1.0
        )
        feature_row["side_name"] = str(side).lower()
        overlay_cols = set(str(c) for c in (overlay_required_columns or set()) if str(c))
        if overlay_cols and not override_supplied:
            feature_row = materialize_live_source_regime_features(
                feature_row,
                side=side,
                signal_bar_ts=ts,
                required_columns=overlay_cols,
            )
            feature_row = materialize_live_ae_gmm_features(
                feature_row,
                side=side,
                signal_bar_ts=ts,
                required_columns=overlay_cols,
                state_payload=live_ae_gmm_state_payload,
            )
    out = {
        "decision_ts": row.get("decision_ts"),
        "signal_bar_ts": ts,
        "symbol": symbol,
        "side": side,
        "strategy_id": strategy_id,
        "policy_archetype": str(
            row.get("policy_archetype")
            or row.get("archetype_policy_key")
            or ""
        ),
        "live_feature_contract_hash": row.get("feature_contract_hash"),
        "live_feature_transform_contract_hash": row.get(
            "feature_transform_contract_hash"
        ),
        "live_base_pred": _safe_float(row.get("live_base_pred")),
        "live_meta_pred": _safe_float(row.get("live_meta_pred", row.get("raw_prediction_score"))),
        "live_meta_pred_raw_refit": _safe_float(row.get("meta_pred_raw_refit")),
        "live_meta_pred_aligned": _safe_float(row.get("meta_pred_aligned")),
        "logged_side_residual_expert_active": bool(
            str(row.get("side_residual_expert_input_hash") or "").strip()
        ),
        "live_v9_parent_rank": _safe_float(row.get("v9_tail95_predecessor_rank")),
        "live_mlp_hier_ev_score": _safe_float(row.get("score_regime_calibrated")),
        "live_expected_net_ev_after_1pct": _safe_float(
            row.get("expected_net_ev_after_1pct")
        ),
        "live_expected_ev_rank_score": _safe_float(row.get("expected_ev_rank_score")),
        "live_meta_postprocessor_input_hash": row.get(
            "meta_postprocessor_input_hash"
        ),
        "live_calibrated_score": _safe_float(row.get("live_calibrated_score")),
        "live_rank_percentile": _safe_float(row.get("live_rank_percentile", row.get("normalized_rank_score"))),
        "live_policy_rank_pct": _safe_float(row.get("live_policy_rank_pct", row.get("policy_rank_pct"))),
        "live_rank_score_source": row.get("live_rank_score_source", row.get("rank_score_source")),
        "live_threshold_basis_rank_score": _safe_float(
            row.get("live_threshold_basis_rank_score", row.get("threshold_basis_rank_score"))
        ),
        "live_threshold_basis_policy_id": row.get(
            "live_threshold_basis_policy_id", row.get("threshold_basis_policy_id")
        ),
        "live_threshold_basis_dynamic_score_threshold": _safe_float(
            row.get(
                "live_threshold_basis_dynamic_score_threshold",
                row.get("threshold_basis_dynamic_score_threshold"),
            )
        ),
        "live_threshold_basis_dynamic_ev_target": _safe_float(
            row.get(
                "live_threshold_basis_dynamic_ev_target",
                row.get("threshold_basis_dynamic_ev_target"),
            )
        ),
        "live_threshold_basis_mapped_expected_ev": _safe_float(
            row.get("threshold_basis_mapped_expected_ev_side_archetype")
        ),
        "live_threshold_basis_recent_ev_correction": _safe_float(
            row.get("threshold_basis_side_archetype_recent_ev_correction")
        ),
        "live_threshold_basis_corrected_expected_ev": _safe_float(
            row.get("threshold_basis_corrected_expected_ev")
        ),
        "live_threshold_basis_corrected_expected_ev_rank": _safe_float(
            row.get("threshold_basis_corrected_expected_ev_rank")
        ),
        "live_threshold_basis_parent_rank": _safe_float(
            row.get("threshold_basis_parent_rank")
        ),
        "live_threshold_basis_blended_rank": _safe_float(
            row.get("threshold_basis_blended_rank")
        ),
        "live_threshold_basis_selected": bool(
            row.get("threshold_basis_selected", False)
        ),
        "replay_feature_cols": int(feature_row.shape[1]) if not feature_row.empty else 0,
        "replay_missing_features": bool(feature_row.empty),
    }
    threshold_basis_mode = str(out["live_rank_score_source"] or "").startswith(
        "threshold_basis:"
    )
    if threshold_basis_mode and np.isfinite(out["live_threshold_basis_rank_score"]):
        out["threshold_basis_rank_internal_delta"] = (
            out["live_rank_percentile"] - out["live_threshold_basis_rank_score"]
        )
        out["threshold_basis_policy_rank_internal_delta"] = (
            out["live_policy_rank_pct"] - out["live_threshold_basis_rank_score"]
        )
    else:
        out["threshold_basis_rank_internal_delta"] = float("nan")
        out["threshold_basis_policy_rank_internal_delta"] = float("nan")
    live_cal_threshold = float("nan")
    if not np.isfinite(out["live_calibrated_score"]) and np.isfinite(
        out["live_meta_pred"]
    ):
        live_cal, live_cal_threshold = calibrated_score_and_threshold(
            raw_score=out["live_meta_pred"],
            strategy_id=core_strategy_id,
            calibration_data=calibration_data,
            default_threshold=1.0,
        )
        out["live_calibrated_score"] = live_cal
        out["live_calibration_threshold"] = live_cal_threshold
    if feature_row.empty:
        return out
    base_delta_summary = _feature_value_delta_summary(
        logged_values_raw=row.get("base_model_feature_values_json"),
        feature_row=base_feature_row,
        symbol=symbol,
    )
    meta_delta_summary = _feature_value_delta_summary(
        logged_values_raw=row.get("meta_model_feature_values_json"),
        feature_row=feature_row,
        symbol=symbol,
    )
    residual_expert_contract_summary = _feature_value_contract_summary(
        logged_values_raw=row.get("side_residual_expert_input_values_json"),
        feature_row=feature_row,
        symbol=symbol,
        categorical_values={
            "side_name": side,
            "archetype_policy_key": str(
                row.get("policy_archetype")
                or row.get("archetype_policy_key")
                or ""
            ),
        },
    )
    residual_expert_aegmm_summary = _feature_value_delta_summary(
        logged_values_raw=row.get("side_residual_expert_input_values_json"),
        feature_row=feature_row,
        symbol=symbol,
        feature_filter=lambda key: bool(
            re.search(
                r"(?:^|_)(?:ae|aegmm|dae|gmm|mahal|latent|cluster)(?:_|$)|reconstruction",
                str(key),
                flags=re.IGNORECASE,
            )
        ),
    )
    base_synth_delta_summary = _live_synthesized_feature_delta_summary(
        logged_values_raw=row.get("base_model_feature_values_json"),
        feature_row=base_feature_row,
        symbol=symbol,
    )
    meta_synth_delta_summary = _live_synthesized_feature_delta_summary(
        logged_values_raw=row.get("meta_model_feature_values_json"),
        feature_row=feature_row,
        symbol=symbol,
    )
    out.update(
        {
            "base_feature_value_common_count": base_delta_summary["count"],
            "base_feature_value_max_abs_delta": base_delta_summary["max_abs"],
            "base_feature_value_mean_abs_delta": base_delta_summary["mean_abs"],
            "base_feature_value_worst_feature": base_delta_summary["worst_feature"],
            "base_feature_value_worst_is_live_synthesized": bool(
                _is_live_synthesized_feature_key(
                    base_delta_summary["worst_feature"]
                )
            ),
            "base_live_synth_feature_value_common_count": base_synth_delta_summary[
                "count"
            ],
            "base_live_synth_feature_value_max_abs_delta": base_synth_delta_summary[
                "max_abs"
            ],
            "base_live_synth_feature_value_mean_abs_delta": base_synth_delta_summary[
                "mean_abs"
            ],
            "base_live_synth_feature_value_worst_feature": base_synth_delta_summary[
                "worst_feature"
            ],
            "base_live_synth_feature_value_worst_live_value": base_synth_delta_summary[
                "worst_live_value"
            ],
            "base_live_synth_feature_value_worst_replay_value": base_synth_delta_summary[
                "worst_replay_value"
            ],
            "meta_feature_value_common_count": meta_delta_summary["count"],
            "meta_feature_value_max_abs_delta": meta_delta_summary["max_abs"],
            "meta_feature_value_mean_abs_delta": meta_delta_summary["mean_abs"],
            "meta_feature_value_worst_feature": meta_delta_summary["worst_feature"],
            "meta_feature_value_worst_is_live_synthesized": bool(
                _is_live_synthesized_feature_key(
                    meta_delta_summary["worst_feature"]
                )
            ),
            "residual_expert_feature_logged_count": (
                residual_expert_contract_summary["logged_count"]
            ),
            "residual_expert_feature_common_finite_count": (
                residual_expert_contract_summary["common_finite_count"]
            ),
            "residual_expert_feature_missing_count": (
                residual_expert_contract_summary["missing_count"]
            ),
            "residual_expert_feature_missing_features": json.dumps(
                residual_expert_contract_summary["missing_features"],
                separators=(",", ":"),
            ),
            "residual_expert_feature_max_abs_delta": (
                residual_expert_contract_summary["max_abs"]
            ),
            "residual_expert_feature_mean_abs_delta": (
                residual_expert_contract_summary["mean_abs"]
            ),
            "residual_expert_feature_worst_feature": (
                residual_expert_contract_summary["worst_feature"]
            ),
            "residual_expert_feature_worst_live_value": (
                residual_expert_contract_summary["worst_live_value"]
            ),
            "residual_expert_feature_worst_replay_value": (
                residual_expert_contract_summary["worst_replay_value"]
            ),
            "residual_expert_aegmm_feature_common_count": (
                residual_expert_aegmm_summary["count"]
            ),
            "residual_expert_aegmm_feature_max_abs_delta": (
                residual_expert_aegmm_summary["max_abs"]
            ),
            "residual_expert_aegmm_feature_mean_abs_delta": (
                residual_expert_aegmm_summary["mean_abs"]
            ),
            "residual_expert_aegmm_feature_worst_feature": (
                residual_expert_aegmm_summary["worst_feature"]
            ),
            "meta_live_synth_feature_value_common_count": meta_synth_delta_summary[
                "count"
            ],
            "meta_live_synth_feature_value_max_abs_delta": meta_synth_delta_summary[
                "max_abs"
            ],
            "meta_live_synth_feature_value_mean_abs_delta": meta_synth_delta_summary[
                "mean_abs"
            ],
            "meta_live_synth_feature_value_worst_feature": meta_synth_delta_summary[
                "worst_feature"
            ],
            "meta_live_synth_feature_value_worst_live_value": meta_synth_delta_summary[
                "worst_live_value"
            ],
            "meta_live_synth_feature_value_worst_replay_value": meta_synth_delta_summary[
                "worst_replay_value"
            ],
        }
    )
    logged_base_frame = _logged_values_frame(
        row.get("base_model_feature_values_json"),
        symbol=symbol,
    )
    logged_meta_frame = _logged_values_frame(
        row.get("meta_model_feature_values_json"),
        symbol=symbol,
    )
    try:
        if not logged_base_frame.empty:
            logged_base_pred = orchestrator.predict_alpha(
                logged_base_frame,
                side,
                model_strategy_id,
            )
            # Historical/live candidate scoring can preserve NaNs for
            # LightGBM's native missing-value routing.  A one-row audit must
            # use the same behavior instead of turning a valid logged input
            # into an unscorable complete-case row.
            if not isinstance(logged_base_pred, pd.Series) or logged_base_pred.empty:
                previous_native_missing = (orchestrator.cfg or {}).get(
                    "simple_policy_allow_lgbm_native_missing"
                )
                orchestrator.cfg["simple_policy_allow_lgbm_native_missing"] = True
                try:
                    logged_base_pred = orchestrator.predict_alpha(
                        logged_base_frame,
                        side,
                        model_strategy_id,
                    )
                finally:
                    if previous_native_missing is None:
                        orchestrator.cfg.pop(
                            "simple_policy_allow_lgbm_native_missing", None
                        )
                    else:
                        orchestrator.cfg[
                            "simple_policy_allow_lgbm_native_missing"
                        ] = previous_native_missing
            out["logged_base_input_pred"] = (
                _safe_float(logged_base_pred.iloc[0])
                if isinstance(logged_base_pred, pd.Series)
                and not logged_base_pred.empty
                else float("nan")
            )
        else:
            out["logged_base_input_pred"] = float("nan")
    except Exception:
        out["logged_base_input_pred"] = float("nan")
    try:
        if not logged_meta_frame.empty:
            out["logged_meta_input_pred"] = _predict_exact_logged_meta_input(
                orchestrator=orchestrator,
                side=side,
                strategy_id=model_strategy_id,
                logged_meta_frame=logged_meta_frame,
                base_prediction=out["logged_base_input_pred"],
            )
        else:
            out["logged_meta_input_pred"] = float("nan")
    except Exception:
        out["logged_meta_input_pred"] = float("nan")
    out["logged_base_input_pred_delta"] = (
        out["logged_base_input_pred"] - out["live_base_pred"]
    )
    live_direct_meta_target = out["live_meta_pred"]
    out["logged_meta_input_comparison_target"] = (
        "inactive_direct_meta_diagnostic"
        if out["logged_side_residual_expert_active"]
        else "meta_pred"
    )
    out["logged_meta_input_pred_delta"] = (
        float("nan")
        if out["logged_side_residual_expert_active"]
        else out["logged_meta_input_pred"] - live_direct_meta_target
    )
    if canonical_postprocessor is not None:
        logged_postprocessor_frame = _logged_values_frame(
            row.get("meta_postprocessor_input_values_json"), symbol=symbol
        )
        canonical_input = (
            logged_postprocessor_frame.copy()
            if not logged_postprocessor_frame.empty
            else feature_row.loc[[symbol]].copy()
        )
        for col in logged_meta_frame.columns:
            if col not in canonical_input.columns:
                canonical_input[col] = logged_meta_frame[col].to_numpy(copy=False)
        canonical_input["side_name"] = side
        canonical_input["archetype_policy_key"] = str(
            row.get("policy_archetype")
            or row.get("archetype_policy_key")
            or ""
        )
        canonical_input["score_base"] = out["logged_base_input_pred"]
        # The canonical V9/MLP chain receives the side-residual expert rank,
        # which is persisted as meta_pred.  The direct meta-model prediction
        # reconstructed above is only a diagnostic once that expert is active.
        # Keep score as the base-anchor compatibility alias used in production.
        canonical_input["score"] = out["logged_base_input_pred"]
        canonical_input["score_meta_base_soft_label"] = out["live_meta_pred"]
        canonical_input["score_meta_base_soft_label_raw_refit"] = out[
            "live_meta_pred_raw_refit"
        ]
        if (
            isinstance(canonical_result_override, pd.DataFrame)
            and not canonical_result_override.empty
        ):
            canonical = canonical_result_override
        else:
            canonical_input["__ts__"] = ts
            canonical_input["__symbol__"] = symbol
            canonical = canonical_postprocessor.transform(canonical_input, copy=False)
        canonical_row = canonical.iloc[0]
        out["replay_v9_input_meta_score"] = _safe_float(
            canonical_row.get("score_meta_base_soft_label")
        )
        out["replay_v9_input_meta_score_raw_refit"] = _safe_float(
            canonical_row.get("score_meta_base_soft_label_raw_refit")
        )
        out["replay_v9_input_base_score"] = _safe_float(
            canonical_row.get("score_base")
        )
        out["replay_v9_input_archetype"] = str(
            canonical_row.get("archetype_policy_key") or ""
        )
        out["replay_v9_parent_rank"] = _safe_float(
            canonical_row.get("historical_rank")
        )
        out["replay_v9_score_shock_adjusted"] = _safe_float(
            canonical_row.get("score_shock_adjusted")
        )
        out["replay_mlp_hier_ev_score"] = _safe_float(
            canonical_row.get("score_regime_calibrated")
        )
        out["replay_expected_net_ev_after_1pct"] = _safe_float(
            canonical_row.get("expected_net_ev_after_1pct")
        )
        out["replay_expected_net_ev_side_archetype"] = _safe_float(
            canonical_row.get(
                "expected_net_ev_after_1pct_side_archetype",
                canonical_row.get("expected_net_ev_after_1pct"),
            )
        )
        out["replay_expected_ev_rank_score"] = _safe_float(
            canonical_row.get("expected_ev_rank_score")
        )
        out["replay_meta_postprocessor_policy_id"] = str(
            canonical_row.get("meta_postprocessor_policy_id") or ""
        )
        out["postprocessor_logged_feature_count"] = _safe_int(
            canonical_row.get("__parity_postprocessor_logged_feature_count")
        )
        out["postprocessor_common_finite_count"] = _safe_int(
            canonical_row.get("__parity_postprocessor_common_finite_count")
        )
        out["postprocessor_missing_count"] = _safe_int(
            canonical_row.get("__parity_postprocessor_missing_count")
        )
        out["postprocessor_missing_features"] = str(
            canonical_row.get("__parity_postprocessor_missing_features") or "[]"
        )
        out["postprocessor_input_max_abs_delta"] = _safe_float(
            canonical_row.get("__parity_postprocessor_input_max_abs_delta")
        )
        out["postprocessor_input_mean_abs_delta"] = _safe_float(
            canonical_row.get("__parity_postprocessor_input_mean_abs_delta")
        )
        out["postprocessor_input_worst_feature"] = str(
            canonical_row.get("__parity_postprocessor_input_worst_feature") or ""
        )
        out["postprocessor_input_worst_live_value"] = _safe_float(
            canonical_row.get("__parity_postprocessor_input_worst_live_value")
        )
        out["postprocessor_input_worst_replay_value"] = _safe_float(
            canonical_row.get("__parity_postprocessor_input_worst_replay_value")
        )
        if threshold_basis_policy:
            decision = {
                "timestamp": ts,
                "symbol": symbol,
                "side_name": side,
                "policy_archetype": canonical_input[
                    "archetype_policy_key"
                ].iloc[0],
                "strategy_id": strategy_id,
                "expected_ev_rank_score": out["replay_expected_ev_rank_score"],
                "expected_net_ev_after_1pct_side_archetype": _safe_float(
                    canonical_row.get(
                        "expected_net_ev_after_1pct_side_archetype",
                        canonical_row.get("expected_net_ev_after_1pct"),
                    )
                ),
                "v9_tail95_predecessor_rank": out["replay_v9_parent_rank"],
                "policy_rank_pct": out["replay_expected_ev_rank_score"],
            }
            apply_threshold_basis_policy_to_decisions(
                [decision], policy=threshold_basis_policy
            )
            out["replay_threshold_basis_selected"] = bool(
                decision.get("threshold_basis_selected", False)
            )
            out["threshold_basis_selection_match"] = bool(
                out["replay_threshold_basis_selected"]
                == out["live_threshold_basis_selected"]
            )
            out["replay_threshold_basis_rank_score"] = _safe_float(
                decision.get("threshold_basis_rank_score")
            )
            out["replay_threshold_basis_dynamic_score_threshold"] = _safe_float(
                decision.get("threshold_basis_dynamic_score_threshold")
            )
            out["replay_threshold_basis_policy_id"] = str(
                decision.get("threshold_basis_policy_id") or ""
            )
            out["replay_threshold_basis_mapped_expected_ev"] = _safe_float(
                decision.get("threshold_basis_mapped_expected_ev_side_archetype")
            )
            out["replay_threshold_basis_recent_ev_correction"] = _safe_float(
                decision.get("threshold_basis_side_archetype_recent_ev_correction")
            )
            out["replay_threshold_basis_corrected_expected_ev"] = _safe_float(
                decision.get("threshold_basis_corrected_expected_ev")
            )
            out["replay_threshold_basis_corrected_expected_ev_rank"] = _safe_float(
                decision.get("threshold_basis_corrected_expected_ev_rank")
            )
            out["replay_threshold_basis_parent_rank"] = _safe_float(
                decision.get("threshold_basis_parent_rank")
            )
            out["replay_threshold_basis_blended_rank"] = _safe_float(
                decision.get("threshold_basis_blended_rank")
            )
        out["canonical_final_rank_delta"] = (
            out.get("replay_threshold_basis_rank_score", np.nan)
            - out["live_threshold_basis_rank_score"]
        )
        out["v9_parent_rank_delta"] = (
            out["replay_v9_parent_rank"] - out["live_v9_parent_rank"]
        )
        out["v9_input_meta_raw_refit_delta"] = (
            out["replay_v9_input_meta_score_raw_refit"]
            - out["live_meta_pred_raw_refit"]
        )
        out["v9_input_base_score_delta"] = (
            out["replay_v9_input_base_score"] - out["live_base_pred"]
        )
        out["mlp_hier_ev_score_delta"] = (
            out["replay_mlp_hier_ev_score"] - out["live_mlp_hier_ev_score"]
        )
        out["expected_net_ev_after_1pct_delta"] = (
            out["replay_expected_net_ev_after_1pct"]
            - out["live_expected_net_ev_after_1pct"]
        )
        out["expected_ev_rank_score_delta"] = (
            out["replay_expected_ev_rank_score"]
            - out["live_expected_ev_rank_score"]
        )
        for name in (
            "mapped_expected_ev",
            "recent_ev_correction",
            "corrected_expected_ev",
            "corrected_expected_ev_rank",
            "parent_rank",
            "blended_rank",
        ):
            replay_value = out.get(f"replay_threshold_basis_{name}", np.nan)
            live_value = out.get(f"live_threshold_basis_{name}", np.nan)
            # Blacklisted/unmapped rows intentionally emit no EV value. Two
            # absent values are parity, while one-sided absence must still
            # fail through the non-finite delta check below.
            if not np.isfinite(replay_value) and not np.isfinite(live_value):
                out[f"threshold_basis_{name}_delta"] = 0.0
            else:
                out[f"threshold_basis_{name}_delta"] = replay_value - live_value
    logged_meta_cal, logged_meta_cal_threshold = calibrated_score_and_threshold(
        raw_score=out["logged_meta_input_pred"],
        strategy_id=core_strategy_id,
        calibration_data=calibration_data,
        default_threshold=1.0,
    )
    logged_meta_rank = rank_store.lookup(
        strategy_id=strategy_id,
        side=side,
        calibrated_score=logged_meta_cal,
    )
    out["logged_meta_input_calibrated_score"] = logged_meta_cal
    out["logged_meta_input_calibration_threshold"] = logged_meta_cal_threshold
    out["logged_meta_input_policy_rank_pct"] = logged_meta_rank.policy_rank_pct
    if out["logged_side_residual_expert_active"]:
        # This direct-meta calibration branch is inactive once the side residual
        # expert supplies the score entering V9. Its values remain diagnostics,
        # but comparing them with the active-chain calibration is meaningless.
        out["logged_meta_input_calibrated_score_delta"] = float("nan")
        out["logged_meta_input_rank_percentile_delta"] = float("nan")
    else:
        out["logged_meta_input_calibrated_score_delta"] = (
            logged_meta_cal - out["live_calibrated_score"]
        )
        out["logged_meta_input_rank_percentile_delta"] = (
            logged_meta_rank.policy_rank_pct - out["live_policy_rank_pct"]
        )
    try:
        alpha_pred = orchestrator.predict_alpha(
            base_feature_row.loc[[symbol]],
            side,
            model_strategy_id,
        )
        if not isinstance(alpha_pred, pd.Series) or alpha_pred.empty:
            previous_native_missing = (orchestrator.cfg or {}).get(
                "simple_policy_allow_lgbm_native_missing"
            )
            orchestrator.cfg["simple_policy_allow_lgbm_native_missing"] = True
            try:
                alpha_pred = orchestrator.predict_alpha(
                    base_feature_row.loc[[symbol]],
                    side,
                    model_strategy_id,
                )
            finally:
                if previous_native_missing is None:
                    orchestrator.cfg.pop(
                        "simple_policy_allow_lgbm_native_missing", None
                    )
                else:
                    orchestrator.cfg[
                        "simple_policy_allow_lgbm_native_missing"
                    ] = previous_native_missing
        replay_base_direct = (
            _safe_float(alpha_pred.iloc[0])
            if isinstance(alpha_pred, pd.Series) and not alpha_pred.empty
            else float("nan")
        )
    except Exception:
        replay_base_direct = float("nan")
    chain: dict[str, Any] = {}
    replay_base = replay_base_direct
    canonical_batch_meta = _safe_float(out.get("replay_v9_input_meta_score"))
    replay_meta = (
        canonical_batch_meta
        if isinstance(canonical_result_override, pd.DataFrame)
        and np.isfinite(canonical_batch_meta)
        else float("nan")
    )
    full_chain_meta = float("nan")
    full_chain_cal = float("nan")
    full_chain_cal_threshold = float("nan")
    full_chain_policy_rank_pct = float("nan")
    replay_meta_input_delta_summary = {
        "count": 0,
        "max_abs": float("nan"),
        "mean_abs": float("nan"),
        "worst_feature": "",
    }
    meta_replay_source = (
        "independent_batch_meta_input"
        if np.isfinite(replay_meta)
        else "logged_batch_meta_input_skip_full_chain"
    )
    can_skip_full_chain = bool(skip_full_chain_diagnostics) and (
        np.isfinite(out.get("logged_meta_input_pred", np.nan))
        or np.isfinite(replay_meta)
    )
    if not can_skip_full_chain:
        chain = orchestrator.run_full_chain(
            symbol,
            side,
            feature_row.loc[[symbol]],
            kind=model_strategy_id,
        )
        replay_base = _safe_float(chain.get("base_pred"))
        if not np.isfinite(replay_base):
            replay_base = replay_base_direct
        replay_meta = _safe_float(chain.get("meta_pred"))
        replay_meta_model_input = getattr(orchestrator, "_last_meta_model_input", None)
        replay_meta_input_delta_summary = (
            _feature_value_delta_summary(
                logged_values_raw=row.get("meta_model_feature_values_json"),
                feature_row=replay_meta_model_input,
                symbol=symbol,
            )
            if isinstance(replay_meta_model_input, pd.DataFrame)
            else replay_meta_input_delta_summary
        )
        full_chain_meta = replay_meta
        full_chain_cal, full_chain_cal_threshold = calibrated_score_and_threshold(
            raw_score=full_chain_meta,
            strategy_id=core_strategy_id,
            calibration_data=calibration_data,
            default_threshold=1.0,
        )
        full_chain_rank = rank_store.lookup(
            strategy_id=strategy_id,
            side=side,
            calibrated_score=full_chain_cal,
        )
        full_chain_policy_rank_pct = full_chain_rank.policy_rank_pct
        meta_replay_source = "single_row_full_chain"
    if prefer_logged_meta_input and np.isfinite(
        out.get("logged_meta_input_pred", np.nan)
    ):
        # Live performs meta prediction in batch and records the exact selected
        # meta-model matrix per decision. Some diagnostic meta inputs, notably
        # feature_drift_psi_core, are batch-context dependent; using the logged
        # selected matrix is the exact training/inference adapter parity path.
        replay_meta = out["logged_meta_input_pred"]
        meta_replay_source = "logged_batch_meta_input"
    replay_cal, replay_cal_threshold = calibrated_score_and_threshold(
        raw_score=replay_meta,
        strategy_id=core_strategy_id,
        calibration_data=calibration_data,
        default_threshold=1.0,
    )
    rank_lookup = rank_store.lookup(
        strategy_id=strategy_id,
        side=side,
        calibrated_score=replay_cal,
    )
    out.update(
        {
            "replay_action": chain.get("action"),
            "replay_reason": chain.get("reason"),
            "replay_model_strategy_id": model_strategy_id,
            "replay_base_pred": replay_base,
            "replay_meta_pred": replay_meta,
            "replay_calibrated_score": replay_cal,
            "replay_calibration_threshold": replay_cal_threshold,
            "replay_policy_rank_pct": rank_lookup.policy_rank_pct,
            "replay_policy_rank_reference_n": rank_lookup.n_rows,
            "replay_policy_rank_reference_source": rank_lookup.source,
            "replay_meta_input_source": meta_replay_source,
            "full_chain_meta_pred": full_chain_meta,
            "full_chain_calibrated_score": full_chain_cal,
            "full_chain_calibration_threshold": full_chain_cal_threshold,
            "full_chain_policy_rank_pct": full_chain_policy_rank_pct,
            "full_chain_meta_pred_delta": (
                float("nan")
                if out["logged_side_residual_expert_active"]
                else full_chain_meta - live_direct_meta_target
            ),
            "full_chain_rank_percentile_delta": full_chain_policy_rank_pct
            - out["live_policy_rank_pct"],
            "replay_meta_model_input_common_count": replay_meta_input_delta_summary[
                "count"
            ],
            "replay_meta_model_input_max_abs_delta": replay_meta_input_delta_summary[
                "max_abs"
            ],
            "replay_meta_model_input_mean_abs_delta": replay_meta_input_delta_summary[
                "mean_abs"
            ],
            "replay_meta_model_input_worst_feature": replay_meta_input_delta_summary[
                "worst_feature"
            ],
            "base_pred_delta": replay_base - out["live_base_pred"],
            "meta_pred_delta": (
                float("nan")
                if out["logged_side_residual_expert_active"]
                else replay_meta - live_direct_meta_target
            ),
            "calibrated_score_delta": (
                float("nan")
                if out["logged_side_residual_expert_active"]
                else replay_cal - out["live_calibrated_score"]
            ),
            "rank_percentile_delta": (
                float("nan")
                if out["logged_side_residual_expert_active"]
                else rank_lookup.policy_rank_pct - out["live_policy_rank_pct"]
            ),
        }
            )
    return out


def _logged_input_feature_row(row: pd.Series) -> pd.DataFrame:
    """Build the exact persisted scoring row without loading market history."""
    symbol = str(row.get("symbol") or "")
    frames = [
        _logged_values_frame(row.get(column), symbol=symbol)
        for column in (
            "base_model_feature_values_json",
            "meta_model_feature_values_json",
            "meta_postprocessor_input_values_json",
            "side_residual_expert_input_values_json",
        )
    ]
    frames = [frame for frame in frames if isinstance(frame, pd.DataFrame) and not frame.empty]
    if not frames:
        return pd.DataFrame()
    combined = frames[0].copy()
    for frame in frames[1:]:
        for column in frame.columns:
            if column not in combined.columns:
                combined[column] = frame[column].to_numpy(copy=False)
    return combined


def _alpha_contract_columns_for_replay(state: Mapping[str, Any]) -> set[str]:
    """Return generated and raw columns named by deployed base contracts."""
    state_bundle = state.get("bundle", {}) if isinstance(state, Mapping) else {}
    if not isinstance(state_bundle, Mapping):
        return set()
    columns: set[str] = set()
    for model_info in (state_bundle.get("alpha_models", {}) or {}).values():
        if not isinstance(model_info, Mapping):
            continue
        columns.update(
            str(column)
            for column in (model_info.get("feat_cols", []) or [])
            if str(column)
        )
    return columns


def _batched_replay_feature_rows(
    *,
    feats: dict[str, pd.DataFrame],
    group: pd.DataFrame,
    signal_bar_ts: pd.Timestamp,
    overlay_required_columns: set[str] | None,
    live_ae_gmm_state_payload: Mapping[str, Any] | None,
    artifact_data_root: Path | None = None,
    run_id: str | None = None,
    supplement_excluded_columns: set[str] | None = None,
) -> dict[Any, pd.DataFrame]:
    """Materialize live synthetic overlays once per side/timestamp batch.

    Production inference creates the side candidate feature matrix first, then
    appends source-regime and frozen AE/GMM overlays to that batch.  Replaying
    each row independently changes row-order-dependent AE/GMM deltas such as
    cluster speed and acceleration.  This helper mirrors the production shape
    for the rows available in the live prediction ledger.
    """

    if not isinstance(group, pd.DataFrame) or group.empty:
        return {}
    overlay_cols = set(str(c) for c in (overlay_required_columns or set()) if str(c))
    supplement_excluded = set(
        str(c) for c in (supplement_excluded_columns or set()) if str(c)
    )
    if not overlay_cols:
        return {}
    out: dict[Any, pd.DataFrame] = {}
    side_series = group.get("side", pd.Series("", index=group.index)).fillna("").astype(str)
    for side, side_group in group.groupby(side_series.str.lower(), sort=False):
        if str(side) not in {"long", "short"} or side_group.empty:
            continue
        symbols = [
            _normalise_symbol(s)
            for s in side_group["symbol"].dropna().astype(str).tolist()
        ]
        symbols = list(dict.fromkeys(symbols))
        if not symbols:
            continue
        feature_batch = _load_persisted_live_candidate_feature_matrix(
            artifact_data_root=artifact_data_root,
            run_id=run_id,
            signal_bar_ts=signal_bar_ts,
            side=side,
        )
        persisted_batch = isinstance(feature_batch, pd.DataFrame) and not feature_batch.empty
        if persisted_batch:
            missing_persisted = sorted(
                overlay_cols.difference(feature_batch.columns).difference(
                    supplement_excluded
                )
            )
            if missing_persisted:
                supplement = get_features_for_candidates(
                    feats, symbols, ts=signal_bar_ts
                )
                for col in missing_persisted:
                    if col in supplement.columns:
                        feature_batch[col] = supplement.reindex(
                            feature_batch.index
                        )[col].to_numpy(copy=False)
        else:
            feature_batch = get_features_for_candidates(feats, symbols, ts=signal_bar_ts)
        if not isinstance(feature_batch, pd.DataFrame) or feature_batch.empty:
            continue
        feature_batch = feature_batch.copy()
        missing_overlay = overlay_cols.difference(feature_batch.columns)
        nonfinite_overlay = {
            col
            for col in overlay_cols.intersection(feature_batch.columns)
            if not bool(
                np.isfinite(
                    pd.to_numeric(feature_batch[col], errors="coerce").to_numpy(
                        dtype=float,
                        copy=False,
                    )
                ).all()
            )
        }
        unusable_overlay = set(missing_overlay).union(nonfinite_overlay)
        if not persisted_batch:
            feature_batch["side"] = np.float32(1.0 if side == "long" else -1.0)
            feature_batch["side_name"] = str(side)
        if unusable_overlay:
            unusable_ae = sorted(
                set(unusable_overlay).intersection(AE_GMM_FEATURE_SET)
            )
            if unusable_ae:
                print(
                    "Replay frozen AE/GMM overlay refresh: "
                    f"side={side} rows={len(feature_batch)} "
                    f"requested={len(unusable_ae)} sample={unusable_ae[:12]}"
                )
            source_unusable = set(unusable_overlay).difference(AE_GMM_FEATURE_SET)
            if source_unusable:
                feature_batch = materialize_live_source_regime_features(
                    feature_batch,
                    side=side,
                    signal_bar_ts=signal_bar_ts,
                    required_columns=source_unusable,
                )
            # Persisted full-universe latent outputs are authoritative. Missing
            # values remain missing and are rejected by the strict row guard;
            # recomputing them here mutates the frozen transform sequence and
            # changes later-bar posterior/distance values.
            if not persisted_batch and unusable_ae:
                feature_batch = materialize_live_ae_gmm_features(
                    feature_batch,
                    side=side,
                    signal_bar_ts=signal_bar_ts,
                    required_columns=unusable_ae,
                    state_payload=live_ae_gmm_state_payload,
                )
            if unusable_ae:
                refreshed = {
                    col: int(
                        np.isfinite(
                            pd.to_numeric(
                                feature_batch.get(col), errors="coerce"
                            ).to_numpy(dtype=float, copy=False)
                        ).sum()
                    )
                    for col in unusable_ae
                    if col in feature_batch.columns
                }
                print(
                    "Replay frozen AE/GMM overlay refreshed: "
                    f"side={side} finite_counts={dict(list(refreshed.items())[:12])}"
                )
        for idx, row in side_group.iterrows():
            symbol = _normalise_symbol(str(row.get("symbol", "")))
            if symbol in feature_batch.index:
                out[idx] = feature_batch.loc[[symbol]].copy()
    return out


def _materialize_replay_meta_anchor_features(
    *,
    feature_rows: Mapping[Any, pd.DataFrame],
    group: pd.DataFrame,
    orchestrator: ModelOrchestrator,
    prior_payload: Mapping[str, Any],
) -> dict[Any, pd.DataFrame]:
    """Build protected meta anchors from replayed base predictions.

    Production creates these columns after the base top-30 handoff.  The parity
    replay must do the same instead of borrowing the logged meta matrix, or a
    nominally independent replay can silently become a logged-input check.
    """

    out = {
        idx: frame.copy()
        for idx, frame in feature_rows.items()
        if isinstance(frame, pd.DataFrame) and not frame.empty
    }
    if not out or not prior_payload:
        return out

    side_values = group.get("side", pd.Series("", index=group.index)).astype(str)
    strategy_values = group.get(
        "strategy_id", pd.Series("", index=group.index)
    ).astype(str)
    grouping = pd.DataFrame(
        {"side": side_values.str.lower(), "strategy_id": strategy_values},
        index=group.index,
    )
    for (side, strategy_id), local_index in grouping.groupby(
        ["side", "strategy_id"], sort=False
    ).groups.items():
        if side not in {"long", "short"}:
            continue
        core = strategy_core_id(str(strategy_id))
        model_strategy_id = str(strategy_id)
        if core and not model_strategy_id.startswith(f"{side}_"):
            model_strategy_id = f"{side}_{core}"

        base_predictions: dict[str, dict[str, float]] = {}
        prediction_errors: list[str] = []
        local_frames: list[pd.DataFrame] = []
        frame_index: list[Any] = []
        for idx in local_index:
            frame = out.get(idx)
            if frame is None or frame.empty:
                continue
            symbol = str(frame.index[0])
            try:
                pred = orchestrator.predict_alpha(
                    frame.loc[[symbol]], side, model_strategy_id
                )
                if not isinstance(pred, pd.Series) or pred.empty:
                    previous_native_missing = (orchestrator.cfg or {}).get(
                        "simple_policy_allow_lgbm_native_missing"
                    )
                    orchestrator.cfg[
                        "simple_policy_allow_lgbm_native_missing"
                    ] = True
                    try:
                        pred = orchestrator.predict_alpha(
                            frame.loc[[symbol]], side, model_strategy_id
                        )
                    finally:
                        if previous_native_missing is None:
                            orchestrator.cfg.pop(
                                "simple_policy_allow_lgbm_native_missing", None
                            )
                        else:
                            orchestrator.cfg[
                                "simple_policy_allow_lgbm_native_missing"
                            ] = previous_native_missing
                score = (
                    _safe_float(pred.iloc[0])
                    if isinstance(pred, pd.Series) and not pred.empty
                    else float("nan")
                )
            except Exception as exc:
                prediction_errors.append(f"{symbol}: {type(exc).__name__}: {exc}")
                score = float("nan")
            if not np.isfinite(score):
                continue
            base_predictions[symbol] = {"base_pred": float(score)}
            local_frames.append(frame.loc[[symbol]])
            frame_index.append(idx)

        if not local_frames:
            raise RuntimeError(
                "Independent base replay produced no finite scores for "
                f"{side}/{model_strategy_id}; errors={prediction_errors[:5]}"
            )
        batch = pd.concat(local_frames, axis=0, copy=False)
        batch = apply_live_meta_reliability_priors(
            batch,
            side=side,
            base_predictions=base_predictions,
            prior_payload=prior_payload,
        )
        for idx, symbol in zip(frame_index, batch.index):
            out[idx] = batch.loc[[symbol]].copy()
    return out


def _hydrate_replay_residual_context_features(
    *,
    feature_rows: Mapping[Any, pd.DataFrame],
    group: pd.DataFrame,
    panel_slice: Mapping[str, pd.DataFrame],
    signal_bar_ts: pd.Timestamp,
    feature_cfg: Mapping[str, Any],
    runtime_cfg: Mapping[str, Any],
    run_id: str,
    data_root: str,
    residual_event_payload: Mapping[str, Any],
    canonical_postprocessor: CanonicalMetaPostprocessor,
) -> dict[Any, pd.DataFrame]:
    """Mirror production's second, residual-event feature namespace pass."""

    # One feature row belongs to one ledger decision. Candidate extraction can
    # return duplicate symbol-index rows when the same symbol is represented by
    # multiple side/strategy decisions in the signal-bar group. Carrying those
    # duplicates into the residual batch makes its length diverge from the
    # decision-key/symbol vectors and can mix context across decisions.
    out = {
        idx: frame.iloc[[0]].copy()
        for idx, frame in feature_rows.items()
        if isinstance(frame, pd.DataFrame) and not frame.empty
    }
    if not out:
        return out
    residual_required = residual_event_state_input_feature_columns(
        residual_event_payload
    )
    calibration_raw_required = _live_regime_calibration_raw_feature_columns(
        canonical_postprocessor.regime_ev_artifact,
        residual_event_payload,
    )
    lazy_required = sorted(
        set(residual_required)
        | set(calibration_raw_required)
        | set(canonical_postprocessor.required_input_features())
    )
    if not lazy_required:
        return out

    # A symbol normally appears once per side in the candidate stream.  Keep
    # those rows distinct: frozen AE/GMM outputs are side-conditioned, so
    # deduplicating on symbol can silently copy the long state onto the short
    # row (or vice versa) during replay hydration.
    ordered_rows = list(out.items())
    row_symbols = [str(frame.index[0]) for _idx, frame in ordered_rows]
    existing_batch = pd.concat(
        [frame.reset_index(drop=True) for _idx, frame in ordered_rows],
        axis=0,
        ignore_index=True,
        copy=False,
    )
    residual_missing = [
        column
        for column in lazy_required
        if column not in existing_batch.columns
        or not np.isfinite(
            pd.to_numeric(existing_batch.get(column), errors="coerce").to_numpy(
                dtype=float, copy=False
            )
        ).any()
    ]
    # Refresh all observable postprocessor inputs, including columns already
    # present in the first-pass model matrix. Production does the same so the
    # V9/MLP chain cannot inherit append-only transform-state values that differ
    # from a causal batch replay after historical source repairs.
    residual_refresh = list(lazy_required)
    if not residual_refresh:
        return out

    close_panel = panel_slice.get("close", pd.DataFrame())
    residual_runtime_cfg = _build_residual_event_feature_runtime_cfg(
        runtime_cfg,
        coverage_symbols=(
            list(close_panel.columns)
            if isinstance(close_panel, pd.DataFrame)
            else []
        ),
        optional_feature_keys=set(residual_refresh).difference(residual_required),
        same_cycle_memory=False,
    )
    # Residual/V9 inputs use the same frozen static feature store as historical
    # replay and production inference. A separate raw-panel recomputation here
    # creates a second transform contract and can drift even when the model
    # inputs themselves are unchanged. Missing persisted inputs remain missing
    # and are rejected by the strict complete-case boundary below.
    residual_runtime_cfg["live_model_feature_tail_recompute_enabled"] = False
    residual_runtime_cfg["live_feature_prefer_offline_cache"] = True
    residual_runtime_cfg["live_feature_offline_cache_enabled"] = True
    residual_runtime_cfg["live_feature_offline_cache_authoritative"] = True
    residual_runtime_cfg["live_model_feature_auto_sync_selected_cache"] = False
    residual_runtime_cfg["live_model_feature_auto_sync_on_low_finite"] = False
    residual_feature_cfg = dict(feature_cfg or {})
    residual_feature_cfg.update(residual_runtime_cfg)
    residual_feature_cfg["runtime_cfg"] = residual_runtime_cfg
    panel_hours = max(
        72,
        int(
            residual_runtime_cfg.get(
                "live_residual_feature_lookback_hours",
                residual_runtime_cfg.get("live_decision_panel_lookback_hours", 72),
            )
            or 72
        ),
    )
    residual_features = load_or_compute_features(
        panel=dict(panel_slice),
        basket_syms=list(close_panel.columns),
        run_id=str(run_id),
        data_root=str(data_root),
        cfg=residual_feature_cfg,
        lookback_hours=panel_hours,
        required_feature_keys=set(residual_refresh),
    )
    symbols = list(
        dict.fromkeys(
            _normalise_symbol(value)
            for value in group.get("symbol", pd.Series(dtype=object)).dropna()
        )
    )
    supplemental = get_features_for_candidates(
        residual_features, symbols, ts=signal_bar_ts
    )
    if supplemental.empty:
        return out
    combined = existing_batch.copy()
    for column in residual_refresh:
        if column in supplemental.columns:
            replacement = pd.to_numeric(
                supplemental.reindex(row_symbols)[column], errors="coerce"
            ).reset_index(drop=True)
            if column not in combined.columns:
                combined[column] = replacement.to_numpy(copy=False)
                continue
            finite = np.isfinite(
                replacement.to_numpy(dtype=float, copy=False)
            )
            if bool(finite.any()):
                values = pd.to_numeric(
                    combined[column], errors="coerce"
                ).to_numpy(dtype=np.float32, copy=True)
                # Match production's fill-only residual hydration.  Frozen
                # AE/GMM and source-regime values are materialized on the full
                # tradable universe before the base top-30 handoff; replacing
                # those finite values with a later post-meta recomputation
                # changes their cross-sectional meaning and breaks replay/live
                # parity.
                missing = ~np.isfinite(values)
                fill = missing & finite
                values[fill] = replacement.to_numpy(
                    dtype=np.float32, copy=False
                )[fill]
                combined[column] = values
    combined, still_missing, _optional_missing = _hydrate_optional_frozen_features(
        combined,
        attempted_columns=residual_refresh,
        strict_columns=residual_required,
    )
    if still_missing:
        raise RuntimeError(
            "Replay residual-event state lacks observable inputs: "
            + ", ".join(sorted(still_missing)[:20])
        )
    for row_pos, (idx, frame) in enumerate(ordered_rows):
        symbol = str(frame.index[0])
        for column in residual_refresh:
            if column in combined.columns:
                frame[column] = combined.iloc[[row_pos]][column].to_numpy(copy=False)
    return out


def _load_persisted_live_candidate_feature_matrix(
    *,
    artifact_data_root: Path | None,
    run_id: str | None,
    signal_bar_ts: pd.Timestamp,
    side: str,
) -> pd.DataFrame:
    if artifact_data_root is None or not run_id:
        return pd.DataFrame()
    try:
        ts = pd.Timestamp(signal_bar_ts)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    except Exception:
        return pd.DataFrame()
    side_s = "short" if str(side).lower().startswith("short") else "long"
    root = (
        Path(artifact_data_root)
        / "artifacts"
        / str(run_id)
        / "live_candidate_feature_matrix"
        / side_s
    )
    if not root.exists():
        return pd.DataFrame()
    candidates: list[tuple[float, Path]] = []
    for meta_path in root.glob("*/meta.json"):
        try:
            meta = json.loads(meta_path.read_text())
            raw_ts = meta.get("signal_bar_ts") or meta.get("end_ts")
            if not raw_ts:
                continue
            meta_ts = pd.Timestamp(raw_ts)
            meta_ts = (
                meta_ts.tz_localize("UTC")
                if meta_ts.tzinfo is None
                else meta_ts.tz_convert("UTC")
            )
            if meta_ts != ts:
                continue
            data_path = meta_path.with_name("data.parquet")
            if data_path.exists():
                candidates.append((float(data_path.stat().st_mtime), data_path))
        except Exception:
            continue
    if not candidates:
        return pd.DataFrame()
    data_path = sorted(candidates, key=lambda item: item[0], reverse=True)[0][1]
    try:
        frame = pd.read_parquet(data_path)
        frame.index = frame.index.astype(str)
        return frame
    except Exception:
        return pd.DataFrame()


def _batched_canonical_postprocessor_rows(
    *,
    group: pd.DataFrame,
    feature_rows: Mapping[Any, pd.DataFrame],
    orchestrator: ModelOrchestrator,
    side_residual_expert: SideResidualExpertBundle | None,
    canonical_postprocessor: CanonicalMetaPostprocessor,
    residual_event_payload: Mapping[str, Any],
    residual_reference_prior_payload: Mapping[str, Any],
    meta_reliability_prior_payload: Mapping[str, Any],
    signal_bar_ts: pd.Timestamp,
    prefer_logged_inputs: bool = True,
) -> dict[Any, pd.DataFrame]:
    """Replay the canonical postprocessor in the same side-batch shape as live."""
    output: dict[Any, pd.DataFrame] = {}
    side_values = group.get("side", pd.Series("", index=group.index)).astype(str)
    for side, side_group in group.groupby(side_values.str.lower(), sort=False):
        if side not in {"long", "short"}:
            continue
        frames: list[pd.DataFrame] = []
        source_indices: list[Any] = []
        source_symbols: list[str] = []
        archetypes: dict[str, str] = {}
        input_contract_diagnostics: dict[Any, dict[str, Any]] = {}
        model_strategy_ids: list[str] = []
        seen_symbols: set[str] = set()
        for idx, row in side_group.iterrows():
            feature_row = feature_rows.get(idx)
            if not isinstance(feature_row, pd.DataFrame) or feature_row.empty:
                continue
            symbol = _normalise_symbol(row.get("symbol"))
            # Restart/idempotency ledgers can contain the same symbol more than
            # once for one signal bar. Production scores one cross-sectional
            # row per symbol; replay that row once and map the result back to
            # every logged decision record below.
            source_indices.append(idx)
            source_symbols.append(symbol)
            if symbol in seen_symbols:
                continue
            seen_symbols.add(symbol)
            logged_postprocessor = _logged_values_frame(
                row.get("meta_postprocessor_input_values_json"), symbol=symbol
            )
            current = (
                logged_postprocessor.copy()
                if prefer_logged_inputs and not logged_postprocessor.empty
                else feature_row.loc[[symbol]].copy()
            )
            logged_meta = _logged_values_frame(
                row.get("meta_model_feature_values_json"), symbol=symbol
            )
            logged_residual_expert = _logged_values_frame(
                row.get("side_residual_expert_input_values_json"), symbol=symbol
            )
            if prefer_logged_inputs:
                for col in (*logged_meta.columns, *logged_residual_expert.columns):
                    source = (
                        logged_residual_expert
                        if col in logged_residual_expert.columns
                        else logged_meta
                    )
                    if col not in current.columns:
                        current[col] = source[col].to_numpy(copy=False)
                        continue
                    existing = pd.to_numeric(current[col], errors="coerce")
                    replacement = pd.to_numeric(source[col], errors="coerce")
                    missing = ~np.isfinite(existing.to_numpy(dtype=float, copy=False))
                    if bool(missing.any()):
                        values = existing.to_numpy(dtype=np.float32, copy=True)
                        values[missing] = replacement.to_numpy(
                            dtype=np.float32, copy=False
                        )[missing]
                        current[col] = values
            strategy_id = str(row.get("strategy_id") or "")
            model_strategy_id = strategy_id
            core = strategy_core_id(strategy_id)
            if core and not strategy_id.startswith(f"{side}_"):
                model_strategy_id = f"{side}_{core}"
            base_prediction = _safe_float(
                row.get("live_base_pred")
                if prefer_logged_inputs
                else current.get("score", pd.Series(np.nan, index=current.index)).iloc[0]
            )
            if prefer_logged_inputs:
                current["score_meta_base_soft_label"] = (
                    _predict_exact_logged_meta_input(
                        orchestrator=orchestrator,
                        side=side,
                        strategy_id=model_strategy_id,
                        logged_meta_frame=logged_meta,
                        apply_score_alignment=True,
                        base_prediction=base_prediction,
                    )
                )
                current["score_meta_base_soft_label_raw_refit"] = (
                    _predict_exact_logged_meta_input(
                        orchestrator=orchestrator,
                        side=side,
                        strategy_id=model_strategy_id,
                        logged_meta_frame=logged_meta,
                        apply_score_alignment=False,
                        base_prediction=base_prediction,
                    )
                )
            else:
                current["score_meta_base_soft_label"] = np.float32(np.nan)
                current["score_meta_base_soft_label_raw_refit"] = np.float32(np.nan)
            current["score_base"] = base_prediction
            current["side_name"] = side
            archetype = str(
                row.get("policy_archetype")
                or row.get("archetype_policy_key")
                or ""
            )
            current["archetype_policy_key"] = archetype
            current["__ts__"] = signal_bar_ts
            current["__symbol__"] = symbol
            frames.append(current)
            model_strategy_ids.append(model_strategy_id)
            archetypes[symbol] = archetype
        if not frames:
            continue
        batch = pd.concat(frames, axis=0, copy=False)
        if not prefer_logged_inputs:
            replay_base_predictions = {
                str(symbol): {
                    "base_pred": _safe_float(
                        batch.loc[symbol, "score"]
                        if "score" in batch.columns
                        else np.nan
                    )
                }
                for symbol in batch.index
            }
            batch = apply_live_meta_reliability_priors(
                batch,
                side=side,
                base_predictions=replay_base_predictions,
                prior_payload=meta_reliability_prior_payload,
            )
            distinct_model_ids = list(dict.fromkeys(model_strategy_ids))
            if len(distinct_model_ids) != 1:
                raise RuntimeError(
                    "Independent meta parity batch mixes model contracts: "
                    + ", ".join(distinct_model_ids)
                )
            model_strategy_id = distinct_model_ids[0]
            aligned_meta = orchestrator.predict_meta(
                batch,
                side,
                model_strategy_id,
            )
            model_input = getattr(orchestrator, "_last_meta_model_input", None)
            model_key = getattr(orchestrator, "_last_meta_model_key", None)
            meta_model = (orchestrator.meta_models or {}).get(model_key)
            if (
                not isinstance(aligned_meta, pd.Series)
                or not isinstance(model_input, pd.DataFrame)
                or meta_model is None
            ):
                raise RuntimeError(
                    f"Independent meta parity adapter failed for {side}/{model_strategy_id}"
                )
            raw_meta = pd.Series(
                np.asarray(meta_model.predict(model_input), dtype=np.float32).reshape(-1),
                index=model_input.index,
                dtype=np.float32,
            )
            batch["score_meta_base_soft_label"] = pd.to_numeric(
                aligned_meta.reindex(batch.index), errors="coerce"
            ).astype(np.float32)
            batch["score_meta_base_soft_label_raw_refit"] = pd.to_numeric(
                raw_meta.reindex(batch.index), errors="coerce"
            ).astype(np.float32)
        score_columns = (
            "score_base",
            "score_meta_base_soft_label",
            "score_meta_base_soft_label_raw_refit",
        )
        score_values = batch.reindex(columns=score_columns).apply(
            pd.to_numeric, errors="coerce"
        )
        valid_score_rows = np.isfinite(
            score_values.to_numpy(dtype=float, copy=False)
        ).all(axis=1)
        if not bool(valid_score_rows.all()):
            dropped = int((~valid_score_rows).sum())
            print(
                "Independent canonical replay dropped incomplete model rows "
                f"before V9: side={side} dropped={dropped}/{len(batch)}"
            )
            batch = batch.loc[valid_score_rows].copy()
            valid_symbols = set(batch.index.astype(str))
            retained_pairs = [
                (idx, symbol)
                for idx, symbol in zip(source_indices, source_symbols)
                if symbol in valid_symbols
            ]
            source_indices = [idx for idx, _symbol in retained_pairs]
            source_symbols = [symbol for _idx, symbol in retained_pairs]
            archetypes = {
                symbol: archetype
                for symbol, archetype in archetypes.items()
                if symbol in valid_symbols
            }
        if batch.empty:
            continue
        if side_residual_expert is not None:
            batch["side_name"] = str(side)
            batch["archetype_policy_key"] = pd.Series(
                archetypes, index=batch.index, dtype="object"
            ).reindex(batch.index)
            # Match production exactly: the direct side residual expert uses
            # the frozen base prediction as its backbone, then its
            # train-reference rank becomes the score entering V9.
            batch["score_base"] = pd.to_numeric(
                batch["score_base"], errors="coerce"
            ).astype(np.float32)
            batch["score"] = batch["score_base"]
            expert = side_residual_expert.transform(batch)
            expert_complete = expert[
                "meta_residual_expert_complete_case"
            ].astype(bool)
            if not bool(expert_complete.all()):
                batch = batch.loc[expert_complete].copy()
                expert = expert.loc[expert_complete]
                valid_symbols = set(batch.index.astype(str))
                retained_pairs = [
                    (idx, symbol)
                    for idx, symbol in zip(source_indices, source_symbols)
                    if symbol in valid_symbols
                ]
                source_indices = [idx for idx, _symbol in retained_pairs]
                source_symbols = [symbol for _idx, symbol in retained_pairs]
                archetypes = {
                    symbol: archetype
                    for symbol, archetype in archetypes.items()
                    if symbol in valid_symbols
                }
            if batch.empty:
                continue
            for column in expert.columns:
                batch[column] = expert[column]
            expert_rank = pd.to_numeric(
                expert["score_base_residual_ev_rank_train_reference"],
                errors="coerce",
            ).astype(np.float32)
            batch["score_meta_base_soft_label"] = expert_rank
            batch["score_meta_base_soft_label_raw_refit"] = expert_rank
        # Production applies a second, independently fitted reliability
        # contract before V9.  These residual-reference priors intentionally
        # replace the meta-model rel_* anchors; omitting this stage changes the
        # V9 rank even when base and meta predictions are identical.
        batch = apply_live_meta_reliability_priors(
            batch,
            side=side,
            base_predictions={
                str(symbol): {
                    "base_pred": float(
                        pd.to_numeric(batch.at[symbol, "score_base"], errors="coerce")
                    )
                }
                for symbol in batch.index
            },
            prior_payload=residual_reference_prior_payload,
        )
        predecessor = canonical_postprocessor.predict_predecessor(batch)
        predecessor_input = canonical_postprocessor.attach_predecessor(
            batch, predecessor
        )
        residual_features = materialize_live_residual_event_features(
            predecessor_input,
            payload=residual_event_payload,
            side=side,
            policy_archetypes=archetypes,
            meta_scores=predecessor["historical_rank"],
            signal_bar_ts=signal_bar_ts,
        )
        comparison_input = predecessor_input.copy()
        for column in residual_features.columns:
            comparison_input[column] = residual_features[column].to_numpy(copy=False)
        for idx, row in side_group.iterrows():
            symbol = _normalise_symbol(row.get("symbol"))
            input_contract_diagnostics[idx] = _feature_value_contract_summary(
                logged_values_raw=row.get("meta_postprocessor_input_values_json"),
                feature_row=comparison_input,
                symbol=symbol,
            )
        canonical = canonical_postprocessor.apply_from_components(
            predecessor_input,
            predecessor=predecessor,
            residual_state_features=residual_features,
            copy=False,
        )
        # Reliability-prior materialization may regroup/reorder a side batch.
        # Map canonical outputs back by symbol instead of zipping by position.
        for idx, symbol in zip(source_indices, source_symbols):
            if symbol not in canonical.index:
                continue
            canonical_row = canonical.loc[symbol]
            if isinstance(canonical_row, pd.DataFrame):
                canonical_row = canonical_row.iloc[-1]
            result_row = canonical_row.to_frame().T
            diagnostics = input_contract_diagnostics.get(idx, {})
            result_row["__parity_postprocessor_logged_feature_count"] = int(
                diagnostics.get("logged_count", 0)
            )
            result_row["__parity_postprocessor_common_finite_count"] = int(
                diagnostics.get("common_finite_count", 0)
            )
            result_row["__parity_postprocessor_missing_count"] = int(
                diagnostics.get("missing_count", 0)
            )
            result_row["__parity_postprocessor_missing_features"] = json.dumps(
                diagnostics.get("missing_features", []), separators=(",", ":")
            )
            result_row["__parity_postprocessor_input_max_abs_delta"] = _safe_float(
                diagnostics.get("max_abs")
            )
            result_row["__parity_postprocessor_input_mean_abs_delta"] = _safe_float(
                diagnostics.get("mean_abs")
            )
            result_row["__parity_postprocessor_input_worst_feature"] = str(
                diagnostics.get("worst_feature") or ""
            )
            result_row["__parity_postprocessor_input_worst_live_value"] = _safe_float(
                diagnostics.get("worst_live_value")
            )
            result_row["__parity_postprocessor_input_worst_replay_value"] = _safe_float(
                diagnostics.get("worst_replay_value")
            )
            output[idx] = result_row
    return output


def _summary(frame: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "rows": int(len(frame)),
        "replay_rows_with_features": int((~frame.get("replay_missing_features", True).astype(bool)).sum())
        if not frame.empty
        else 0,
        "policy_rank_reference_available_rows": int(
            pd.to_numeric(frame.get("replay_policy_rank_reference_n"), errors="coerce")
            .fillna(0)
            .gt(0)
            .sum()
        )
        if not frame.empty and "replay_policy_rank_reference_n" in frame
        else 0,
    }
    for col in (
        "base_pred_delta",
        "meta_pred_delta",
        "calibrated_score_delta",
        "rank_percentile_delta",
        "logged_base_input_pred_delta",
        "logged_meta_input_pred_delta",
        "logged_meta_input_calibrated_score_delta",
        "logged_meta_input_rank_percentile_delta",
        "threshold_basis_rank_internal_delta",
        "threshold_basis_policy_rank_internal_delta",
        "v9_parent_rank_delta",
        "v9_input_meta_raw_refit_delta",
        "v9_input_base_score_delta",
        "mlp_hier_ev_score_delta",
        "expected_net_ev_after_1pct_delta",
        "expected_ev_rank_score_delta",
        "canonical_final_rank_delta",
        "threshold_basis_mapped_expected_ev_delta",
        "threshold_basis_recent_ev_correction_delta",
        "threshold_basis_corrected_expected_ev_delta",
        "threshold_basis_corrected_expected_ev_rank_delta",
        "threshold_basis_parent_rank_delta",
        "threshold_basis_blended_rank_delta",
    ):
        if col not in frame:
            continue
        vals = pd.to_numeric(frame[col], errors="coerce")
        summary[col] = {
            "n": int(vals.notna().sum()),
            "mean_abs": float(vals.abs().mean()) if vals.notna().any() else None,
            "max_abs": float(vals.abs().max()) if vals.notna().any() else None,
            "mean": float(vals.mean()) if vals.notna().any() else None,
        }
    if not frame.empty and "live_rank_score_source" in frame:
        summary["live_rank_score_source_counts"] = {
            str(k): int(v)
            for k, v in frame["live_rank_score_source"]
            .fillna("")
            .astype(str)
            .value_counts(dropna=False)
            .to_dict()
            .items()
        }
    if not frame.empty and "threshold_basis_selection_match" in frame:
        matches = frame["threshold_basis_selection_match"].fillna(False).astype(bool)
        summary["threshold_basis_selection_parity"] = {
            "matching_rows": int(matches.sum()),
            "mismatching_rows": int((~matches).sum()),
        }
    if not frame.empty:
        synth_summary: dict[str, Any] = {}
        for scope in ("base", "meta"):
            max_col = f"{scope}_live_synth_feature_value_max_abs_delta"
            worst_col = f"{scope}_live_synth_feature_value_worst_feature"
            if max_col not in frame:
                continue
            vals = pd.to_numeric(frame[max_col], errors="coerce")
            drift_rows = int(vals.gt(1e-7).sum())
            top_features: dict[str, int] = {}
            if worst_col in frame and drift_rows:
                top_features = {
                    str(k): int(v)
                    for k, v in frame.loc[vals.gt(1e-7), worst_col]
                    .fillna("")
                    .astype(str)
                    .value_counts()
                    .head(10)
                    .to_dict()
                    .items()
                    if str(k)
                }
            synth_summary[scope] = {
                "rows_gt_1e-7": drift_rows,
                "max_abs": float(vals.abs().max()) if vals.notna().any() else None,
                "top_worst_features": top_features,
            }
        if synth_summary:
            summary["live_synthesized_feature_reconstruction_drift"] = synth_summary
    return summary


def _parity_failures(
    frame: pd.DataFrame,
    *,
    tolerance: float,
    prediction_tolerance: float | None = None,
    require_policy_rank_reference: bool = False,
    require_live_values: bool = False,
    parity_source: str = "replay",
) -> list[str]:
    failures: list[str] = []
    if frame.empty:
        failures.append("no_rows")
        return failures
    parity_source = str(parity_source or "replay")
    if require_policy_rank_reference:
        raw_ref_n = (
            frame["replay_policy_rank_reference_n"]
            if "replay_policy_rank_reference_n" in frame.columns
            else pd.Series(np.nan, index=frame.index)
        )
        ref_n = pd.to_numeric(raw_ref_n, errors="coerce").fillna(0)
        missing = int(ref_n.le(0).sum())
        if missing:
            failures.append(f"missing_policy_rank_reference_rows={missing}")
    if require_live_values:
        required_cols = (
            "live_base_pred",
            "live_meta_pred",
            "live_calibrated_score",
            "live_policy_rank_pct",
        )
        for col in required_cols:
            raw_vals = (
                frame[col]
                if col in frame.columns
                else pd.Series(np.nan, index=frame.index)
            )
            vals = pd.to_numeric(raw_vals, errors="coerce")
            missing = int(vals.isna().sum())
            if missing:
                failures.append(f"missing_{col}_rows={missing}")
        if "expected_feature_transform_contract_hash" in frame.columns:
            expected = frame["expected_feature_transform_contract_hash"].astype(str)
            live = (
                frame["live_feature_transform_contract_hash"].astype(str)
                if "live_feature_transform_contract_hash" in frame.columns
                else pd.Series("", index=frame.index)
            )
            missing_hash = int(live.isin({"", "None", "nan", "NaN"}).sum())
            if missing_hash:
                failures.append(
                    f"missing_live_feature_transform_contract_hash_rows={missing_hash}"
                )
            mismatch = int((live != expected).sum())
            if mismatch:
                failures.append(
                    f"feature_transform_contract_hash_mismatch_rows={mismatch}"
                )
    side_residual_expert_active = bool(
        frame.get(
            "side_residual_expert_active",
            pd.Series(False, index=frame.index),
        )
        .fillna(False)
        .astype(bool)
        .all()
    )
    if parity_source == "replay" and side_residual_expert_active:
        missing_contract = pd.to_numeric(
            frame.get(
                "residual_expert_feature_missing_count",
                pd.Series(np.nan, index=frame.index),
            ),
            errors="coerce",
        )
        invalid_missing = int(
            (missing_contract.isna() | missing_contract.gt(0)).sum()
        )
        if invalid_missing:
            failures.append(
                "residual_expert_feature_contract_incomplete_rows="
                f"{invalid_missing}"
            )
        residual_delta = pd.to_numeric(
            frame.get(
                "residual_expert_feature_max_abs_delta",
                pd.Series(np.nan, index=frame.index),
            ),
            errors="coerce",
        )
        invalid_delta = int(
            (residual_delta.isna() | residual_delta.gt(float(tolerance))).sum()
        )
        if invalid_delta:
            failures.append(
                "residual_expert_feature_parity_failed_rows=" f"{invalid_delta}"
            )
        aegmm_count = pd.to_numeric(
            frame.get(
                "residual_expert_aegmm_feature_common_count",
                pd.Series(0, index=frame.index),
            ),
            errors="coerce",
        ).fillna(0)
        missing_aegmm = int(aegmm_count.le(0).sum())
        if missing_aegmm:
            failures.append(
                "residual_expert_aegmm_parity_unavailable_rows="
                f"{missing_aegmm}"
            )
    if parity_source == "logged-input":
        required_replay_cols = ["logged_base_input_pred"]
        delta_cols = ["logged_base_input_pred_delta"]
        if side_residual_expert_active:
            required_replay_cols.append("replay_v9_input_meta_score_raw_refit")
            delta_cols.append("v9_input_meta_raw_refit_delta")
        else:
            required_replay_cols.append("logged_meta_input_pred")
            delta_cols.append("logged_meta_input_pred_delta")
    else:
        required_replay_cols = ["replay_base_pred"]
        delta_cols = ["base_pred_delta"]
        if side_residual_expert_active:
            # The side-residual expert replaces the direct meta model as the
            # score entering V9.  Keep direct-meta deltas in the report as a
            # diagnostic, but gate production parity on the independently
            # recomputed expert output that actually feeds the policy.
            required_replay_cols.append("replay_v9_input_meta_score_raw_refit")
            delta_cols.append("v9_input_meta_raw_refit_delta")
        else:
            required_replay_cols.append("replay_meta_pred")
            delta_cols.append("meta_pred_delta")
    rank_source = (
        frame.get("live_rank_score_source", pd.Series("", index=frame.index))
        .fillna("")
        .astype(str)
    )
    threshold_basis_mask = rank_source.str.startswith("threshold_basis:")
    if bool(threshold_basis_mask.all()):
        required_replay_cols.extend(
            [
                "live_calibrated_score",
                "live_threshold_basis_rank_score",
            ]
        )
        delta_cols.extend(
            [
                "threshold_basis_rank_internal_delta",
                "threshold_basis_policy_rank_internal_delta",
                "v9_parent_rank_delta",
                "v9_input_base_score_delta",
                "mlp_hier_ev_score_delta",
                "expected_net_ev_after_1pct_delta",
                "expected_ev_rank_score_delta",
                "canonical_final_rank_delta",
                "threshold_basis_mapped_expected_ev_delta",
                "threshold_basis_recent_ev_correction_delta",
                "threshold_basis_corrected_expected_ev_delta",
                "threshold_basis_corrected_expected_ev_rank_delta",
                "threshold_basis_parent_rank_delta",
                "threshold_basis_blended_rank_delta",
            ]
        )
        # Raw-refit meta output is an attribution diagnostic, not an input to
        # the canonical V9/MLP policy. Enforce it when materialized, but do not
        # fail an otherwise exact policy replay merely because it was omitted.
        raw_refit_delta = pd.to_numeric(
            frame.get(
                "v9_input_meta_raw_refit_delta",
                pd.Series(np.nan, index=frame.index),
            ),
            errors="coerce",
        )
        if raw_refit_delta.notna().any():
            delta_cols.append("v9_input_meta_raw_refit_delta")
        if "threshold_basis_selection_match" in frame:
            selection_match = frame["threshold_basis_selection_match"].fillna(
                False
            ).astype(bool)
            selection_mismatch = int((~selection_match).sum())
            if selection_mismatch:
                failures.append(
                    f"threshold_basis_selection_mismatch_rows={selection_mismatch}"
                )
    elif threshold_basis_mask.any():
        failures.append(
            f"mixed_threshold_basis_and_legacy_rank_rows={int(threshold_basis_mask.sum())}/{len(frame)}"
        )
        required_replay_cols.extend(
            [
                "live_calibrated_score",
                "replay_calibrated_score",
                "replay_policy_rank_pct",
            ]
        )
        delta_cols.extend(["calibrated_score_delta", "rank_percentile_delta"])
    else:
        required_replay_cols.extend(
            [
                "replay_calibrated_score",
                "replay_policy_rank_pct",
            ]
        )
        delta_cols.extend(["calibrated_score_delta", "rank_percentile_delta"])
    for col in required_replay_cols:
        raw_vals = (
            frame[col] if col in frame.columns else pd.Series(np.nan, index=frame.index)
        )
        vals = pd.to_numeric(raw_vals, errors="coerce")
        missing = int(vals.isna().sum())
        if missing:
            failures.append(f"missing_{col}_rows={missing}")
    pred_tol = float(prediction_tolerance) if prediction_tolerance is not None else max(float(tolerance), 1e-7)
    for col in delta_cols:
        if col not in frame:
            continue
        vals = pd.to_numeric(frame[col], errors="coerce").abs()
        finite = vals[np.isfinite(vals)]
        if finite.empty:
            failures.append(f"{col}_finite_rows=0")
            continue
        missing_delta = len(frame) - int(finite.size)
        if missing_delta:
            failures.append(f"{col}_missing_rows={missing_delta}")
        max_abs = float(finite.max())
        col_tol = pred_tol if "pred_delta" in str(col) else float(tolerance)
        if max_abs > col_tol:
            failures.append(f"{col}_max_abs={max_abs:.12g}>tol={col_tol:.12g}")
    return failures


def _model_runtime_cfg(
    *,
    model_bundle: Mapping[str, Any],
    feature_runtime_cfg: Mapping[str, Any],
    disable_model_diagnostics: bool = False,
    disable_model_timing: bool = False,
) -> dict[str, Any]:
    runtime_cfg = dict(feature_runtime_cfg or {})
    runtime_cfg["model_bundle"] = model_bundle
    if disable_model_diagnostics:
        runtime_cfg["inference_lgbm_internal_diagnostics_enabled"] = False
    if disable_model_timing:
        runtime_cfg["inference_model_timing_enabled"] = False
    return runtime_cfg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data", type=Path)
    parser.add_argument(
        "--artifact-data-root",
        type=Path,
        default=None,
        help="Artifact/model root. Defaults to --data-root.",
    )
    parser.add_argument(
        "--market-mode",
        choices=("spot", "perps"),
        default=None,
        help="Inference feature/config mode. Defaults to perps when roots contain 'perp', otherwise spot.",
    )
    parser.add_argument(
        "--exchange-id",
        default="krakenfutures",
        help="Exchange component used for exchange-scoped market data.",
    )
    parser.add_argument(
        "--live-quote-currency",
        default=None,
        help="Live quote currency used for local symbol discovery. Defaults to USD in perps mode, otherwise USDC.",
    )
    parser.add_argument(
        "--live-feature-source-run-id",
        default=None,
        help=(
            "Training-path selected-feature run id to use as authoritative "
            "model feature source. Defaults to EPM_LIVE_FEATURE_SOURCE_RUN_ID."
        ),
    )
    parser.add_argument(
        "--source-parity-run-id",
        default=None,
        help=(
            "Optional artifact run containing the live source-universe reports. "
            "Use when a promoted/copied model retained predecessor reports."
        ),
    )
    parser.add_argument("--run-id", default="20260321_140000")
    parser.add_argument("--ledger", default="data/live_state/prediction_ledger.parquet", type=Path)
    parser.add_argument("--trades", default="inference_trades.csv", type=Path)
    parser.add_argument("--max-rows", default=24, type=int)
    parser.add_argument(
        "--lookback-hours",
        default=24 * 60,
        type=int,
        help="Feature lookback hours. Default matches live inference.",
    )
    parser.add_argument("--max-symbols", default=0, type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--decision-start",
        default=None,
        help="Only replay decisions with decision_ts at or after this timestamp.",
    )
    parser.add_argument(
        "--require-rank-source",
        default=None,
        help="Only replay decisions whose logged rank_score_source matches exactly.",
    )
    parser.add_argument(
        "--min-rows",
        default=1,
        type=int,
        help="Minimum replay rows required before reporting success.",
    )
    parser.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="Exit non-zero if replay/live prediction deltas exceed tolerance.",
    )
    parser.add_argument(
        "--parity-source",
        choices=("replay", "logged-input"),
        default="replay",
        help=(
            "Prediction source used by --fail-on-mismatch. 'replay' checks "
            "selected-cache reconstruction; 'logged-input' checks the exact "
            "model-input values recorded in the live ledger."
        ),
    )
    parser.add_argument(
        "--tolerance",
        default=1e-9,
        type=float,
        help="Absolute tolerance used by --fail-on-mismatch for rank/calibration deltas.",
    )
    parser.add_argument(
        "--prediction-tolerance",
        default=1e-7,
        type=float,
        help=(
            "Absolute tolerance for raw model prediction parity. Kept separate "
            "from rank/calibration tolerance because native model paths can "
            "differ at float32-scale without changing decisions."
        ),
    )
    parser.add_argument(
        "--require-policy-rank-reference",
        action="store_true",
        help="Fail if any replayed row lacks a policy-rank-reference CDF.",
    )
    parser.add_argument(
        "--require-live-values",
        action="store_true",
        help="Fail if base/meta/calibrated/policy-rank live values are absent.",
    )
    parser.add_argument(
        "--disable-model-diagnostics",
        action="store_true",
        help="Disable expensive model-internal diagnostics during replay scoring.",
    )
    parser.add_argument(
        "--disable-model-timing",
        action="store_true",
        help="Disable per-model timing instrumentation during replay scoring.",
    )
    parser.add_argument(
        "--skip-full-chain-diagnostics",
        action="store_true",
        help=(
            "Skip per-row full-chain diagnostic scoring when exact logged batch "
            "meta inputs are available; base predictions are still replayed."
        ),
    )
    parser.add_argument(
        "--skip-email-baseline-diagnostics",
        action="store_true",
        help=(
            "Skip close-email archetype baseline fields during parity replay. "
            "These fields do not affect score, threshold, or admission and are "
            "expensive to recompute independently for every persisted row."
        ),
    )
    parser.add_argument(
        "--batch-by-signal-bar-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Replay each signal timestamp in the same side-batch shape as live "
            "inference. Enabled by default because per-row reconstruction changes "
            "cross-sectional and AE/GMM features."
        ),
    )
    parser.add_argument(
        "--neutral-fill-nonfinite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use the deployed training-parity neutral fill for non-finite model "
            "inputs. Enabled by default to match the live supervisor contract."
        ),
    )
    args = parser.parse_args()
    out_dir = args.output_dir or (
        args.data_root
        / "artifacts"
        / args.run_id
        / "live_signal_prediction_replay"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # A parity audit must never mutate the selected-feature source that it is
    # auditing. Keep process-local transformed caches under the report and
    # disable sidecar writes and training-path repair jobs.
    os.environ["EPM_SELECTED_FEATURE_LATEST_MATRIX_CACHE"] = "0"
    os.environ["EPM_SELECTED_FEATURE_LATEST_MATRIX_CACHE_WRITE"] = "0"
    artifact_data_root = args.artifact_data_root or args.data_root
    market_mode = args.market_mode or (
        "perps"
        if "perp" in f"{args.data_root} {artifact_data_root}".lower()
        else "spot"
    )
    live_quote_currency = (
        str(args.live_quote_currency).upper()
        if args.live_quote_currency
        else ("USD" if market_mode == "perps" else "USDC")
    )
    market_data_root = _market_data_root(
        Path(args.data_root),
        market_mode=market_mode,
        exchange_id=args.exchange_id,
    )

    decisions = _load_recent_decisions(
        ledger_path=args.ledger,
        trades_path=args.trades,
        max_rows=max(1, int(args.max_rows)),
        decision_start=args.decision_start,
        require_rank_source=args.require_rank_source,
    )
    if decisions.empty:
        raise SystemExit("No live decisions matched the replay filters.")
    if len(decisions) < int(args.min_rows):
        raise SystemExit(
            f"Only {len(decisions)} live decisions matched the replay filters; "
            f"min_rows={int(args.min_rows)}."
        )

    min_ts = pd.Timestamp(decisions["signal_bar_ts"].min())
    max_ts = pd.Timestamp(decisions["signal_bar_ts"].max())
    max_panel_ts = max_ts
    live_feature_source_run_id = (
        args.live_feature_source_run_id
        or os.getenv("EPM_LIVE_FEATURE_SOURCE_RUN_ID")
    )
    symbols = _source_parity_context_symbols_for_end(
        args.data_root,
        run_id=(args.source_parity_run_id or args.run_id),
        end_ts=max_ts,
        live_quote_currency=live_quote_currency,
        market_mode=market_mode,
        exchange_id=args.exchange_id,
    )
    symbol_source = "live_source_parity_context"
    if not symbols:
        symbols = _live_feature_cache_symbols_for_end(
            args.data_root,
            run_id=args.run_id,
            end_ts=max_ts,
            live_quote_currency=live_quote_currency,
        )
        symbol_source = "live_feature_cache"
    if not symbols:
        symbols = _feature_source_symbols(
            args.data_root,
            feature_source_run_id=live_feature_source_run_id,
            live_quote_currency=live_quote_currency,
        )
        symbol_source = "pinned_feature_source"
    if not symbols:
        symbols = _local_quote_symbols(
            args.data_root,
            run_id=args.run_id,
            live_quote_currency=live_quote_currency,
            market_mode=market_mode,
            exchange_id=args.exchange_id,
        )
        symbol_source = "local_ohlcv"
    if args.max_symbols and args.max_symbols > 0:
        decision_symbols = list(decisions["symbol"].dropna().map(_normalise_symbol).unique())
        extra = [s for s in symbols if s not in decision_symbols]
        symbols = sorted(set(decision_symbols + extra[: max(0, args.max_symbols - len(decision_symbols))]))
    start_ts = min_ts - pd.Timedelta(hours=int(args.lookback_hours))
    end_ts = max_panel_ts
    logged_input_only = str(args.parity_source) == "logged-input"
    if logged_input_only:
        print(
            f"Replaying {len(decisions)} persisted-input decisions from {min_ts} "
            f"to {max_ts}; market-panel loading is intentionally skipped."
        )
        panel: dict[str, pd.DataFrame] = {}
    else:
        print(
            f"Replaying {len(decisions)} live decisions from {min_ts} to {max_ts}; "
            f"loading {len(symbols)} {live_quote_currency} symbols from {start_ts} to {end_ts} "
            f"(symbol_source={symbol_source})."
        )
        panel = _load_panel(
            data_root=args.data_root,
            symbols=symbols,
            start_ts=start_ts,
            end_ts=end_ts,
            market_mode=market_mode,
            exchange_id=args.exchange_id,
        )
        if not panel or "close" not in panel:
            raise SystemExit("No OHLCV panel could be loaded.")

    state = load_full_state(args.run_id, str(artifact_data_root))
    policy_dir = artifact_data_root / "artifacts" / args.run_id / "policy_params"
    portfolio_policy_path = policy_dir / "optimized_portfolio_policy_config.json"
    if not portfolio_policy_path.exists():
        raise SystemExit(
            f"Canonical portfolio policy is missing: {portfolio_policy_path}"
        )
    portfolio_payload = json.loads(portfolio_policy_path.read_text())
    selection_payload = portfolio_payload.get("selection") or {}
    predecessor_path = Path(
        portfolio_payload.get("regime_ev_predecessor_bundle_path")
        or selection_payload.get("regime_ev_predecessor_bundle_path")
        or ""
    )
    residual_state_path = Path(
        portfolio_payload.get("regime_ev_residual_event_state_path")
        or selection_payload.get("regime_ev_residual_event_state_path")
        or ""
    )
    regime_path = Path(
        portfolio_payload.get("regime_ev_calibration_artifact_path")
        or selection_payload.get("regime_ev_calibration_artifact_path")
        or ""
    )
    threshold_path = Path(
        portfolio_payload.get("threshold_basis_policy_path")
        or selection_payload.get("threshold_basis_policy_path")
        or ""
    )
    canonical_postprocessor = CanonicalMetaPostprocessor.load(
        predecessor_bundle_path=predecessor_path,
        residual_event_state_path=residual_state_path,
        regime_ev_artifact_path=regime_path,
    )
    side_residual_expert = None
    if bool(
        portfolio_payload.get("side_residual_expert_enabled")
        or selection_payload.get("side_residual_expert_enabled")
    ):
        side_residual_path = Path(
            portfolio_payload.get("side_residual_expert_artifact_path")
            or selection_payload.get("side_residual_expert_artifact_path")
            or ""
        )
        if not side_residual_path.is_file():
            raise SystemExit(
                f"Enabled side-residual expert is missing: {side_residual_path}"
            )
        side_residual_expert = SideResidualExpertBundle.load(side_residual_path)
    threshold_basis_policy = load_threshold_basis_policy(threshold_path)
    if args.skip_email_baseline_diagnostics and threshold_basis_policy:
        threshold_basis_policy = dict(threshold_basis_policy)
        threshold_basis_policy["email_archetype_baseline_enabled"] = False
    if not threshold_basis_policy:
        raise SystemExit(f"Canonical threshold policy is missing: {threshold_path}")
    accepted_strategy_keys: set[str] = set()
    for _, decision in decisions.iterrows():
        side = str(decision.get("side") or "").lower()
        core = strategy_core_id(str(decision.get("strategy_id") or ""))
        if core:
            accepted_strategy_keys.add(core)
            if side in {"long", "short"}:
                accepted_strategy_keys.add(f"{side}_{core}")
                accepted_strategy_keys.add(f"{side}_{core}_clf")
    logged_feature_keys = _logged_decision_feature_keys(decisions)
    # Replay must materialize the same feature context as live inference, not
    # only the narrower set of logged model inputs. Some live-derived features
    # share rolling/orderbook primitives, so narrowing this request can change
    # the feature generation path and create false train/live parity breaks.
    selected_feature_keys = set(
        str(c)
        for c in get_inference_required_feature_keys(state, accepted_strategy_keys)
        if str(c)
    )
    overlay_required_columns = set(selected_feature_keys).union(
        str(c) for c in logged_feature_keys if str(c)
    )
    # get_inference_required_feature_keys deliberately returns observable/raw
    # dependencies and excludes generated model features. Production appends
    # frozen AE/GMM outputs again from each alpha strategy contract immediately
    # before base scoring. Mirror that contract here so an independent replay
    # does not silently omit generated features required by the base model.
    state_bundle = state.get("bundle", {}) if isinstance(state.get("bundle"), dict) else {}
    alpha_contract_columns = _alpha_contract_columns_for_replay(state)
    overlay_required_columns.update(alpha_contract_columns)
    canonical_required = set(canonical_postprocessor.required_input_features())
    overlay_required_columns.update(canonical_required)
    side_residual_required = set(
        side_residual_expert.required_input_features()
        if side_residual_expert is not None
        else []
    ).difference({"score", "score_base", "side_name", "archetype_policy_key"})
    overlay_required_columns.update(side_residual_required)
    required_keys = raw_required_feature_keys(selected_feature_keys)
    required_keys |= raw_required_feature_keys(logged_feature_keys)
    required_keys |= raw_required_feature_keys(canonical_required)
    required_keys |= raw_required_feature_keys(side_residual_required)
    try:
        mask_rows = _load_lgbm_strategy_mask_rows(
            str(artifact_data_root),
            args.run_id,
            market_mode=market_mode,
        )
        required_keys |= set(_lgbm_mask_required_feature_keys(mask_rows))
    except Exception:
        pass
    try:
        feature_cfg = load_inference_config(
            data_root=str(artifact_data_root),
            run_id=args.run_id,
            market_mode=market_mode,
        )
    except Exception:
        feature_cfg = dict(CFG)
    runtime_cfg = dict(feature_cfg.get("runtime_cfg") or {})
    runtime_cfg["use_perps"] = market_mode == "perps"
    runtime_cfg["market_mode"] = market_mode
    # Feature generation must receive the same exchange-scoped root as live
    # inference. Panel loading already scopes this internally; retaining the
    # artifact root here creates a different residual-feature cache/runtime
    # contract even when both paths read the same OHLCV files.
    runtime_cfg["data_root"] = str(market_data_root)
    runtime_cfg["artifact_data_root"] = str(artifact_data_root)
    runtime_cfg["model_artifact_run_id"] = str(args.run_id)
    runtime_cfg["run_id"] = str(args.run_id)
    runtime_cfg["live_decision_panel_lookback_hours"] = int(args.lookback_hours)
    runtime_cfg["live_residual_feature_lookback_hours"] = int(args.lookback_hours)
    runtime_cfg["offline_feature_data_root"] = str(artifact_data_root)
    runtime_cfg["live_data_root"] = str(market_data_root)
    neutral_fill_nonfinite = bool(args.neutral_fill_nonfinite)
    feature_cfg["strict_feature_parity_neutral_fill_nonfinite"] = (
        neutral_fill_nonfinite
    )
    runtime_cfg["strict_feature_parity_neutral_fill_nonfinite"] = (
        neutral_fill_nonfinite
    )
    runtime_cfg["live_feature_cache_namespace"] = "model"
    runtime_cfg["live_feature_snapshot_cache_dir"] = str(
        out_dir / "isolated_feature_cache"
    )
    # A source replay must not read or advance the live process' append-only
    # transform states. Those states can legitimately lag historical source
    # repairs, and sharing them makes the audit both non-independent and
    # destructive. The canonical residual/V9 contract uses a causal batch
    # transform in both live and replay.
    isolated_state_dir = out_dir / "isolated_transform_state"
    runtime_cfg["live_causal_transform_state_enabled"] = False
    runtime_cfg["feature_causal_transform_state_enabled"] = False
    runtime_cfg["live_causal_transform_state_path"] = str(
        isolated_state_dir / "causal_transform_state.npz"
    )
    runtime_cfg["feature_causal_transform_state_path"] = str(
        isolated_state_dir / "causal_transform_state.npz"
    )
    runtime_cfg["live_raw_rolling_state_enabled"] = False
    runtime_cfg["feature_raw_rolling_state_enabled"] = False
    runtime_cfg["feature_causal_transform_state_scope"] = "independent_source_replay"
    runtime_cfg["live_feature_memory_cache_enabled"] = False
    # Independent replay may compute missing selected columns through the same
    # canonical static endpoint, but it must never mutate the shared store it
    # is auditing. Production and normal historical materialization append via
    # append_static_features(); replay keeps the identical formulas in memory.
    runtime_cfg["static_feature_store_write_enabled"] = False
    runtime_cfg["live_feature_prefer_offline_cache"] = True
    runtime_cfg["live_feature_offline_cache_enabled"] = True
    runtime_cfg["live_model_feature_auto_sync_selected_cache"] = False
    runtime_cfg["live_model_feature_auto_sync_on_low_finite"] = False
    runtime_cfg["live_model_feature_full_union_background_sync"] = False
    runtime_cfg["live_model_feature_store_strict"] = live_model_feature_store_strict(
        feature_cfg
    )
    runtime_cfg["live_feature_return_latest_only"] = True
    if live_feature_source_run_id:
        runtime_cfg["live_feature_source_run_id"] = str(live_feature_source_run_id)
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
            feature_cfg[key] = value
            runtime_cfg[key] = value
    feature_cfg["runtime_cfg"] = runtime_cfg
    live_ae_gmm_state_payload = load_live_ae_gmm_state_payload(
        str(artifact_data_root),
        args.run_id,
    )
    live_residual_event_state_payload = load_live_residual_event_state_payload(
        str(artifact_data_root),
        args.run_id,
    )
    meta_reliability_prior_payload = load_meta_reliability_prior_payload(
        str(artifact_data_root),
        args.run_id,
    )
    residual_reference_prior_payload = load_residual_reference_prior_payload(
        str(artifact_data_root),
        args.run_id,
    )
    if not residual_reference_prior_payload:
        raise SystemExit("Frozen V9 residual-reference priors are unavailable")
    if not live_residual_event_state_payload:
        raise SystemExit("Frozen live residual-event state payload is unavailable")
    live_ae_gmm_input_columns = live_ae_gmm_input_feature_columns(
        live_ae_gmm_state_payload
    )
    if live_ae_gmm_input_columns:
        required_keys |= raw_required_feature_keys(live_ae_gmm_input_columns)
        overlay_required_columns.update(str(c) for c in live_ae_gmm_input_columns if str(c))
        print(
            "Replay frozen AE/GMM state loaded: "
            f"state={live_ae_gmm_state_payload.get('state_path', '')} "
            f"input_features={len(live_ae_gmm_input_columns)} "
            f"raw_required_features={len(required_keys)} "
            f"overlay_required_columns={len(overlay_required_columns)}"
        )
    orchestrator = ModelOrchestrator(
        state,
        runtime_cfg=_model_runtime_cfg(
            model_bundle=state.get("bundle", {}),
            feature_runtime_cfg=runtime_cfg,
            disable_model_diagnostics=bool(args.disable_model_diagnostics),
            disable_model_timing=bool(args.disable_model_timing),
        ),
    )
    calibration_data = load_calibration_curves(str(artifact_data_root), args.run_id)
    rank_store = PolicyRankReferenceStore(data_root=artifact_data_root, run_id=args.run_id)

    rows: list[dict[str, Any]] = []
    if logged_input_only:
        signal_times = pd.to_datetime(
            decisions["signal_bar_ts"], utc=True, errors="coerce"
        )
        for signal_ts, group in decisions.groupby(signal_times, sort=True):
            feature_rows = {
                idx: _logged_input_feature_row(row) for idx, row in group.iterrows()
            }
            canonical_rows = _batched_canonical_postprocessor_rows(
                group=group,
                feature_rows=feature_rows,
                orchestrator=orchestrator,
                side_residual_expert=side_residual_expert,
                canonical_postprocessor=canonical_postprocessor,
                residual_event_payload=live_residual_event_state_payload,
                residual_reference_prior_payload=residual_reference_prior_payload,
                meta_reliability_prior_payload=meta_reliability_prior_payload,
                signal_bar_ts=pd.Timestamp(signal_ts),
                prefer_logged_inputs=True,
            )
            rows.extend(
                _score_row(
                    row=row,
                    feats={},
                    orchestrator=orchestrator,
                    calibration_data=calibration_data,
                    rank_store=rank_store,
                    overlay_required_columns=overlay_required_columns,
                    live_ae_gmm_state_payload=live_ae_gmm_state_payload,
                    feature_row_override=feature_rows[idx],
                    skip_full_chain_diagnostics=True,
                    canonical_postprocessor=canonical_postprocessor,
                    threshold_basis_policy=threshold_basis_policy,
                    canonical_result_override=canonical_rows.get(idx),
                    prefer_logged_meta_input=True,
                )
                for idx, row in group.iterrows()
            )
    elif args.batch_by_signal_bar_cache:
        signal_times = pd.to_datetime(
            decisions["signal_bar_ts"], utc=True, errors="coerce"
        )
        for signal_ts, group in decisions.groupby(signal_times, sort=True):
            signal_ts = pd.Timestamp(signal_ts)
            if pd.isna(signal_ts):
                continue
            group_start_ts = signal_ts - pd.Timedelta(hours=int(args.lookback_hours))
            panel_slice = _slice_panel(
                panel,
                start_ts=group_start_ts,
                end_ts=signal_ts,
            )
            group_feature_cfg = dict(feature_cfg)
            group_feature_cfg["runtime_cfg"] = dict(runtime_cfg)
            print(
                f"Loading replay features for signal_bar_ts={signal_ts} "
                f"rows={len(group)} start={group_start_ts}"
            )
            feats = load_or_compute_features(
                panel_slice,
                list(panel_slice["close"].columns),
                args.run_id,
                str(market_data_root),
                group_feature_cfg,
                lookback_hours=int(args.lookback_hours),
                required_feature_keys=set(required_keys),
            )
            batched_feature_rows = _batched_replay_feature_rows(
                feats=feats,
                group=group,
                signal_bar_ts=signal_ts,
                overlay_required_columns=overlay_required_columns,
                live_ae_gmm_state_payload=live_ae_gmm_state_payload,
                artifact_data_root=artifact_data_root,
                run_id=args.run_id,
                supplement_excluded_columns=(
                    canonical_required.difference(overlay_required_columns)
                ),
            )
            # Preserve the exact pre-meta matrix used by the base model. The
            # later reliability/residual hydration is a different stage and
            # must not be used to audit base feature parity.
            batched_base_feature_rows = {
                idx: frame.copy()
                for idx, frame in batched_feature_rows.items()
                if isinstance(frame, pd.DataFrame) and not frame.empty
            }
            batched_feature_rows = _materialize_replay_meta_anchor_features(
                feature_rows=batched_feature_rows,
                group=group,
                orchestrator=orchestrator,
                prior_payload=meta_reliability_prior_payload,
            )
            batched_feature_rows = _hydrate_replay_residual_context_features(
                feature_rows=batched_feature_rows,
                group=group,
                panel_slice=panel_slice,
                signal_bar_ts=signal_ts,
                feature_cfg=group_feature_cfg,
                runtime_cfg=runtime_cfg,
                run_id=args.run_id,
                data_root=str(market_data_root),
                residual_event_payload=live_residual_event_state_payload,
                canonical_postprocessor=canonical_postprocessor,
            )
            batched_canonical_rows = _batched_canonical_postprocessor_rows(
                group=group,
                feature_rows=batched_feature_rows,
                orchestrator=orchestrator,
                side_residual_expert=side_residual_expert,
                canonical_postprocessor=canonical_postprocessor,
                residual_event_payload=live_residual_event_state_payload,
                residual_reference_prior_payload=residual_reference_prior_payload,
                meta_reliability_prior_payload=meta_reliability_prior_payload,
                signal_bar_ts=signal_ts,
                prefer_logged_inputs=False,
            )
            rows.extend(
                _score_row(
                    row=row,
                    feats=feats,
                    orchestrator=orchestrator,
                    calibration_data=calibration_data,
                    rank_store=rank_store,
                    overlay_required_columns=overlay_required_columns,
                    live_ae_gmm_state_payload=live_ae_gmm_state_payload,
                    # Rows rejected by the strict base/meta finite contract do
                    # not have a canonical V9 result. Keep them in the audit as
                    # unscored rows instead of invoking a one-row fallback that
                    # would violate the same fail-closed contract.
                    feature_row_override=(
                        batched_feature_rows.get(idx)
                        if idx in batched_canonical_rows
                        else pd.DataFrame()
                    ),
                    base_feature_row_override=batched_base_feature_rows.get(idx),
                    skip_full_chain_diagnostics=bool(args.skip_full_chain_diagnostics),
                    canonical_postprocessor=canonical_postprocessor,
                    threshold_basis_policy=threshold_basis_policy,
                    canonical_result_override=batched_canonical_rows.get(idx),
                    prefer_logged_meta_input=False,
                )
                for idx, row in group.iterrows()
            )
    else:
        feats = load_or_compute_features(
            panel,
            list(panel["close"].columns),
            args.run_id,
            str(market_data_root),
            feature_cfg,
            lookback_hours=int(args.lookback_hours),
            required_feature_keys=set(required_keys),
        )
        rows = [
            _score_row(
                row=row,
                feats=feats,
                orchestrator=orchestrator,
                calibration_data=calibration_data,
                rank_store=rank_store,
                overlay_required_columns=overlay_required_columns,
                live_ae_gmm_state_payload=live_ae_gmm_state_payload,
                skip_full_chain_diagnostics=bool(args.skip_full_chain_diagnostics),
                canonical_postprocessor=canonical_postprocessor,
                threshold_basis_policy=threshold_basis_policy,
                prefer_logged_meta_input=False,
            )
            for _, row in decisions.iterrows()
        ]
    result = pd.DataFrame(rows)
    result["side_residual_expert_active"] = side_residual_expert is not None
    if threshold_basis_policy and not result.empty:
        # Production calibrates/adjudicates the complete timestamp batch. A
        # one-row replay changes current-rank context and can conceal candidate
        # set mismatches, so recompute every threshold field in batch shape.
        replay_decisions: list[dict[str, Any]] = []
        replay_positions: list[Any] = []
        for idx, replay_row in result.iterrows():
            replay_decisions.append(
                {
                    "timestamp": replay_row.get("signal_bar_ts"),
                    "symbol": replay_row.get("symbol"),
                    "side_name": replay_row.get("side"),
                    "policy_archetype": replay_row.get("policy_archetype"),
                    "strategy_id": replay_row.get("strategy_id"),
                    "expected_ev_rank_score": replay_row.get(
                        "replay_expected_ev_rank_score"
                    ),
                    "expected_net_ev_after_1pct_side_archetype": replay_row.get(
                        "replay_expected_net_ev_side_archetype"
                    ),
                    "v9_tail95_predecessor_rank": replay_row.get(
                        "replay_v9_parent_rank"
                    ),
                    "policy_rank_pct": replay_row.get(
                        "replay_expected_ev_rank_score"
                    ),
                }
            )
            replay_positions.append(idx)
        apply_threshold_basis_policy_to_decisions(
            replay_decisions,
            policy=threshold_basis_policy,
            store=rank_store,
        )
        threshold_columns = {
            "threshold_basis_selected": "replay_threshold_basis_selected",
            "threshold_basis_rank_score": "replay_threshold_basis_rank_score",
            "threshold_basis_dynamic_score_threshold": (
                "replay_threshold_basis_dynamic_score_threshold"
            ),
            "threshold_basis_policy_id": "replay_threshold_basis_policy_id",
            "threshold_basis_mapped_expected_ev_side_archetype": (
                "replay_threshold_basis_mapped_expected_ev"
            ),
            "threshold_basis_side_archetype_recent_ev_correction": (
                "replay_threshold_basis_recent_ev_correction"
            ),
            "threshold_basis_corrected_expected_ev": (
                "replay_threshold_basis_corrected_expected_ev"
            ),
            "threshold_basis_corrected_expected_ev_rank": (
                "replay_threshold_basis_corrected_expected_ev_rank"
            ),
            "threshold_basis_parent_rank": "replay_threshold_basis_parent_rank",
            "threshold_basis_blended_rank": "replay_threshold_basis_blended_rank",
        }
        for idx, decision in zip(replay_positions, replay_decisions):
            for source, target in threshold_columns.items():
                result.at[idx, target] = decision.get(source)
            result.at[idx, "threshold_basis_selection_match"] = bool(
                bool(decision.get("threshold_basis_selected", False))
                == bool(result.at[idx, "live_threshold_basis_selected"])
            )
            result.at[idx, "canonical_final_rank_delta"] = (
                _safe_float(decision.get("threshold_basis_rank_score"))
                - _safe_float(result.at[idx, "live_threshold_basis_rank_score"])
            )
    result["replay_neutral_fill_nonfinite"] = bool(neutral_fill_nonfinite)
    result["replay_feature_cache_isolated"] = True
    expected_transform_hash = runtime_cfg.get("feature_transform_contract_hash")
    if expected_transform_hash is not None:
        result["expected_feature_transform_contract_hash"] = expected_transform_hash
    result.to_csv(out_dir / "prediction_replay_comparison.csv", index=False)
    summary = _summary(result)
    (out_dir / "prediction_replay_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    print(f"Wrote {out_dir}")
    if args.fail_on_mismatch:
        failures = _parity_failures(
            result,
            tolerance=float(args.tolerance),
            prediction_tolerance=float(args.prediction_tolerance),
            require_policy_rank_reference=bool(args.require_policy_rank_reference),
            require_live_values=bool(args.require_live_values),
            parity_source=str(args.parity_source),
        )
        if failures:
            print(
                "Replay parity failed: " + ", ".join(failures),
                file=sys.stderr,
            )
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
