"""Audit live decisions against the deployed policy and training feature store.

This script is intentionally diagnostic-only. It does not place orders and does
not mutate deployment artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from extreme_price_movements.data_store import load_features_selected
from extreme_price_movements.inference.config import load_inference_config
from extreme_price_movements.inference.model_orchestrator import ModelOrchestrator
from extreme_price_movements.inference.policy_rank_reference import (
    PolicyRankReferenceStore,
)


DEFAULT_RUN_ID = "20260525_010004_nopenalty"
DEFAULT_FEATURE_SOURCE_RUN_ID = "20260523_015947"
LIVE_SENSITIVE_FEATURE_KEYS = {
    "ob_pressure_mkt_resid",
    "ob_spread_mkt_resid",
    "ob_depth_mkt_resid",
    "ob_imbalance_mkt_resid",
    "xasset_ob_pressure_ts_resid",
    "xasset_ob_pressure_peer_resid",
    "xasset_ob_liquidity_ts_resid",
    "xasset_ob_liquidity_peer_resid",
}


def _json_loads(value: Any, default: Any) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except Exception:
        return default


def _strategy_with_side(side: Any, strategy_id: Any) -> str:
    sid = str(strategy_id or "")
    prefix = f"{str(side or '').lower()}_"
    return sid if sid.startswith(prefix) else prefix + sid


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _file_hash(path: Any) -> str:
    if not path:
        return ""
    try:
        p = Path(str(path))
        if p.exists():
            return hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    except Exception:
        return ""
    return ""


def _file_mtime(path: Any) -> pd.Timestamp | pd.NaT:
    if not path:
        return pd.NaT
    try:
        p = Path(str(path))
        if p.exists():
            return pd.to_datetime(p.stat().st_mtime, unit="s", utc=True)
    except Exception:
        return pd.NaT
    return pd.NaT


def _rank_reference_current_mask(rows: pd.DataFrame) -> pd.Series:
    """Rows whose rank-reference artifacts were not modified after decision time."""
    if rows.empty:
        return pd.Series(dtype=bool)
    decision_ts = pd.to_datetime(rows.get("decision_ts"), utc=True, errors="coerce")
    mask = pd.Series(True, index=rows.index)
    for col in ("policy_rank_reference_mtime", "auction_rank_reference_mtime"):
        if col not in rows.columns:
            continue
        mtime = pd.to_datetime(rows[col], utc=True, errors="coerce")
        mask &= mtime.isna() | decision_ts.isna() | (mtime <= decision_ts)
    return mask.fillna(False).astype(bool)


def _is_traded(value: Any) -> bool:
    return str(value or "").strip().lower() in {"trade", "traded", "accepted", "filled"}


def _feature_source_ts(run_id: str) -> pd.Timestamp:
    return pd.to_datetime(run_id, format="%Y%m%d_%H%M%S", utc=True)


def _compact_symbol(symbol: Any) -> str:
    return str(symbol or "").upper().strip().replace(":USDT", "").replace("/", "_")


def _read_ledger(path: Path, run_id: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    for col in ("timestamp", "decision_ts", "signal_bar_ts", "signal_bar_close_ts"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    if "policy_artifact_run_id" in df.columns:
        df = df[df["policy_artifact_run_id"].astype(str).eq(run_id)].copy()
    return df.reset_index(drop=True)


def _rank_score(row: pd.Series) -> float:
    for col in ("threshold_rank_score", "auction_rank_pct", "normalized_rank_score", "policy_rank_pct"):
        val = _safe_float(row.get(col))
        if np.isfinite(val):
            return val
    return np.nan


def _threshold(row: pd.Series) -> float:
    for col in ("effective_threshold", "final_threshold", "initial_rank_threshold"):
        val = _safe_float(row.get(col))
        if np.isfinite(val):
            return val
    return np.nan


def build_decision_replay(ledger: pd.DataFrame, *, max_new_entries_per_bar: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Replay same-signal-bar open decisions from live ledger candidate rows.

    The replay uses only fields persisted by live inference. API-dependent gates
    are taken from the ledger outcome; deterministic rank/stale/cap gates are
    recomputed.
    """
    rows = ledger.copy()
    rows["rank_score_recomputed"] = rows.apply(_rank_score, axis=1)
    rows["threshold_recomputed"] = rows.apply(_threshold, axis=1)
    rows["rank_gate_recomputed"] = (
        pd.to_numeric(rows["rank_score_recomputed"], errors="coerce")
        >= pd.to_numeric(rows["threshold_recomputed"], errors="coerce")
    )
    rows["stale_gate_recomputed"] = ~rows.get(
        "stale_signal_age_gate_exceeded", pd.Series(False, index=rows.index)
    ).fillna(False).astype(bool)
    rows["strict_replay_eligible"] = (
        rows["signal_bar_close_ts"].notna()
        & rows["threshold_recomputed"].notna()
        & rows["rank_score_recomputed"].notna()
    )

    rows["live_opened"] = rows["portfolio_decision"].map(_is_traded)
    rows["deterministic_gate_open"] = rows["rank_gate_recomputed"] & rows["stale_gate_recomputed"]
    rows["replay_open"] = False
    rows["replay_reject_reason"] = ""

    for signal_ts, grp in rows[rows["strict_replay_eligible"]].groupby("signal_bar_ts", sort=True):
        candidates = grp[grp["deterministic_gate_open"]].copy()
        if candidates.empty:
            rows.loc[grp.index, "replay_reject_reason"] = np.where(
                ~grp["rank_gate_recomputed"],
                "rank_below_threshold",
                np.where(~grp["stale_gate_recomputed"], "stale_signal_age_exceeded", "not_candidate"),
            )
            continue
        candidates["_sort_rank"] = pd.to_numeric(candidates["rank_score_recomputed"], errors="coerce").fillna(-np.inf)
        candidates["_sort_meta"] = pd.to_numeric(candidates.get("meta_pred", np.nan), errors="coerce").fillna(-np.inf)
        candidates["_sort_friction"] = pd.to_numeric(
            candidates.get("expected_total_entry_friction_bps", np.nan), errors="coerce"
        ).fillna(np.inf)
        candidates = candidates.sort_values(
            ["_sort_rank", "_sort_meta", "_sort_friction"],
            ascending=[False, False, True],
        )
        accepted = candidates.head(max(0, int(max_new_entries_per_bar))).index
        rejected_cap = candidates.iloc[max(0, int(max_new_entries_per_bar)):].index
        rows.loc[accepted, "replay_open"] = True
        rows.loc[accepted, "replay_reject_reason"] = "accepted"
        rows.loc[rejected_cap, "replay_reject_reason"] = "max_new_entries_per_bar"
        other = grp.index.difference(candidates.index)
        rows.loc[other, "replay_reject_reason"] = np.where(
            ~rows.loc[other, "rank_gate_recomputed"],
            "rank_below_threshold",
            np.where(~rows.loc[other, "stale_gate_recomputed"], "stale_signal_age_exceeded", "not_candidate"),
        )

    rows["decision_match"] = rows["live_opened"].eq(rows["replay_open"])
    rows["gap_class"] = np.select(
        [
            rows["decision_match"],
            rows["live_opened"] & ~rows["replay_open"],
            ~rows["live_opened"] & rows["replay_open"],
        ],
        ["match", "live_opened_replay_rejected", "replay_opened_live_rejected"],
        default="unknown",
    )
    strict = rows[rows["strict_replay_eligible"]]
    legacy = rows[~rows["strict_replay_eligible"]]
    legacy_traded = legacy[legacy["live_opened"]]
    legacy_signal_to_entry = pd.to_numeric(
        legacy_traded.get("signal_to_entry_seconds", pd.Series(dtype=float)),
        errors="coerce",
    )
    legacy_max_age = pd.to_numeric(
        legacy_traded.get("max_signal_close_to_entry_seconds", pd.Series(np.nan, index=legacy_traded.index)),
        errors="coerce",
    ).fillna(900.0)
    legacy_rank_pass = legacy_traded["rank_gate_recomputed"].fillna(False).astype(bool) if not legacy_traded.empty else pd.Series(dtype=bool)
    legacy_current_stale_fail = (
        legacy_signal_to_entry.notna()
        & legacy_max_age.notna()
        & (legacy_signal_to_entry > legacy_max_age)
    )
    summary = {
        "ledger_rows": int(len(rows)),
        "strict_replay_rows": int(len(strict)),
        "legacy_or_incomplete_rows": int((~rows["strict_replay_eligible"]).sum()),
        "live_opened_strict": int(strict["live_opened"].sum()),
        "replay_opened_strict": int(strict["replay_open"].sum()),
        "strict_decision_matches": int(strict["decision_match"].sum()),
        "strict_decision_mismatches": int((~strict["decision_match"]).sum()),
        "strict_gap_classes": strict["gap_class"].value_counts(dropna=False).to_dict(),
        "legacy_decision_counts": rows.loc[~rows["strict_replay_eligible"], "portfolio_decision"].value_counts(dropna=False).to_dict(),
        "legacy_traded_rows": int(len(legacy_traded)),
        "legacy_traded_rank_gate_pass": int(legacy_rank_pass.sum()) if not legacy_rank_pass.empty else 0,
        "legacy_traded_current_stale_gate_fail": int(legacy_current_stale_fail.sum()) if not legacy_current_stale_fail.empty else 0,
        "legacy_traded_max_signal_to_entry_seconds": float(legacy_signal_to_entry.max())
        if legacy_signal_to_entry.notna().any()
        else np.nan,
    }
    return rows, summary


def _training_feature_values(
    *,
    feature_source_run_id: str,
    data_root: str,
    signal_ts: pd.Timestamp,
    symbol: str,
    feature_keys: Iterable[str],
) -> dict[str, float]:
    keys = [str(k) for k in feature_keys if str(k)]
    if not keys:
        return {}
    feats = load_features_selected(
        _feature_source_ts(feature_source_run_id),
        data_root,
        feature_keys=keys,
        symbols=[symbol],
        start_ts=pd.Timestamp(signal_ts) - pd.Timedelta(minutes=1),
        end_ts=pd.Timestamp(signal_ts) + pd.Timedelta(minutes=1),
    )
    if feats is None:
        return {k: np.nan for k in keys}
    out: dict[str, float] = {}
    for key in keys:
        try:
            vals = feats.latest_values_at(key, [symbol], pd.Timestamp(signal_ts))
            out[key] = _safe_float(vals.iloc[0] if hasattr(vals, "iloc") else np.asarray(vals)[0])
        except Exception:
            out[key] = np.nan
    return out


def _feature_report_summary(report: pd.DataFrame) -> dict[str, Any]:
    live_sensitive_mismatches = (
        report.loc[
            (~report["match"]) & report["feature_source_class"].eq("live_sensitive_orderbook")
        ]
        if not report.empty
        else pd.DataFrame()
    )
    return {
        "rows": int(len(report)),
        "decisions": int(report[["decision_ts", "symbol", "side", "strategy_id"]].drop_duplicates().shape[0]) if not report.empty else 0,
        "mismatches": int((~report["match"]).sum()) if not report.empty else 0,
        "live_sensitive_orderbook_mismatches": int(len(live_sensitive_mismatches)),
        "max_abs_diff": float(report["abs_diff"].max()) if not report.empty and report["abs_diff"].notna().any() else np.nan,
        "features_with_mismatch": sorted(report.loc[~report["match"], "feature"].dropna().unique().tolist())[:50] if not report.empty else [],
    }


def compare_features(ledger: pd.DataFrame, *, data_root: str, feature_source_run_id: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    usable = ledger.dropna(subset=["base_model_features_json", "base_model_feature_values_json", "signal_bar_ts", "symbol"])
    for _, row in usable.iterrows():
        features = _json_loads(row.get("base_model_features_json"), [])
        live_values = _json_loads(row.get("base_model_feature_values_json"), {})
        if not isinstance(features, list) or not isinstance(live_values, dict):
            continue
        train_values = _training_feature_values(
            feature_source_run_id=feature_source_run_id,
            data_root=data_root,
            signal_ts=pd.Timestamp(row["signal_bar_ts"]),
            symbol=str(row["symbol"]),
            feature_keys=features,
        )
        for feature in features:
            live_val = _safe_float(live_values.get(feature))
            train_val = _safe_float(train_values.get(feature))
            abs_diff = abs(live_val - train_val) if np.isfinite(live_val) and np.isfinite(train_val) else np.nan
            rows.append(
                {
                    "decision_ts": row.get("decision_ts"),
                    "signal_bar_ts": row.get("signal_bar_ts"),
                    "symbol": row.get("symbol"),
                    "side": row.get("side"),
                    "strategy_id": row.get("strategy_id"),
                    "feature": feature,
                    "live_value": live_val,
                    "training_value": train_val,
                    "abs_diff": abs_diff,
                    "match": bool(np.isfinite(abs_diff) and abs_diff <= 1e-6),
                    "feature_source_class": (
                        "live_sensitive_orderbook"
                        if str(feature) in LIVE_SENSITIVE_FEATURE_KEYS
                        or str(feature).startswith(("ob_", "obw_", "xasset_ob_", "xasset_mkt_ob_"))
                        else "training_feature_store"
                    ),
                }
            )
    report = pd.DataFrame(rows)
    return report, _feature_report_summary(report)


def _extract_positive_prediction(output: Any) -> float:
    arr = np.asarray(output)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return _safe_float(arr[0, 1])
    if arr.size:
        return _safe_float(arr.reshape(-1)[0])
    return np.nan


def compare_predictions(
    ledger: pd.DataFrame,
    *,
    data_root: str,
    run_id: str,
    feature_source_run_id: str,
    max_rows: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    config = load_inference_config(data_root=data_root, run_id=run_id, market_mode="perps")
    orchestrator = ModelOrchestrator(
        config["full_state"],
        runtime_cfg={**config["runtime_cfg"], "model_bundle": config["model_bundle"]},
    )
    rank_store = PolicyRankReferenceStore(data_root=data_root, run_id=run_id)
    rows: list[dict[str, Any]] = []
    usable = ledger.dropna(subset=["base_model_features_json", "base_model_feature_values_json", "signal_bar_ts", "symbol"]).tail(max_rows)
    for _, row in usable.iterrows():
        side = str(row.get("side") or "")
        strategy_id = str(row.get("strategy_id") or "")
        symbol = str(row.get("symbol") or "")
        base_features = _json_loads(row.get("base_model_features_json"), [])
        if not isinstance(base_features, list):
            continue
        train_values = _training_feature_values(
            feature_source_run_id=feature_source_run_id,
            data_root=data_root,
            signal_ts=pd.Timestamp(row["signal_bar_ts"]),
            symbol=symbol,
            feature_keys=base_features,
        )
        base_x = pd.DataFrame([train_values], index=[symbol])
        try:
            base_pred = _extract_positive_prediction(
                orchestrator.predict_alpha(base_x, side, _strategy_with_side(side, strategy_id))
            )
        except Exception as exc:
            base_pred = np.nan
            base_error = f"{type(exc).__name__}: {exc}"
        else:
            base_error = ""

        meta_pred = np.nan
        meta_error = ""
        meta_key = str(row.get("meta_model_key") or "")
        meta_values = _json_loads(row.get("meta_model_feature_values_json"), {})
        if meta_key and isinstance(meta_values, dict) and meta_key in config["model_bundle"].get("meta_models", {}):
            try:
                model = config["model_bundle"]["meta_models"][meta_key]
                meta_x = pd.DataFrame([meta_values], index=[symbol])
                cols = list(getattr(model, "feature_columns", list(meta_x.columns)))
                meta_x = meta_x.reindex(columns=cols)
                meta_pred = _extract_positive_prediction(model.predict_proba(meta_x))
            except Exception as exc:
                meta_error = f"{type(exc).__name__}: {exc}"
        live_score = _safe_float(row.get("calibrated_score"))
        rank_from_live_score = rank_store.lookup(
            strategy_id=strategy_id,
            side=side,
            calibrated_score=live_score,
        )
        auction_from_live_score = rank_store.lookup_auction(calibrated_score=live_score)
        rank_from_training_score = rank_store.lookup(
            strategy_id=strategy_id,
            side=side,
            calibrated_score=meta_pred,
        )
        auction_from_training_score = rank_store.lookup_auction(calibrated_score=meta_pred)
        policy_source = row.get("policy_rank_reference_source")
        auction_source = row.get("auction_rank_reference_source")
        policy_mtime = _file_mtime(policy_source)
        auction_mtime = _file_mtime(auction_source)
        decision_ts = pd.to_datetime(row.get("decision_ts"), utc=True, errors="coerce")
        policy_reference_after_decision = bool(pd.notna(policy_mtime) and pd.notna(decision_ts) and policy_mtime > decision_ts)
        auction_reference_after_decision = bool(pd.notna(auction_mtime) and pd.notna(decision_ts) and auction_mtime > decision_ts)
        rows.append(
            {
                "decision_ts": row.get("decision_ts"),
                "signal_bar_ts": row.get("signal_bar_ts"),
                "symbol": symbol,
                "side": side,
                "strategy_id": strategy_id,
                "base_pred_live": _safe_float(row.get("base_pred")),
                "base_pred_training_features": base_pred,
                "base_abs_diff": abs(base_pred - _safe_float(row.get("base_pred"))) if np.isfinite(base_pred) else np.nan,
                "base_error": base_error,
                "meta_pred_live": _safe_float(row.get("meta_pred")),
                "meta_pred_exact_saved_input": meta_pred,
                "meta_abs_diff": abs(meta_pred - _safe_float(row.get("meta_pred"))) if np.isfinite(meta_pred) else np.nan,
                "meta_error": meta_error,
                "calibrated_score_live": live_score,
                "policy_rank_live": _safe_float(row.get("policy_rank_pct")),
                "policy_rank_recomputed_from_live_score": rank_from_live_score.policy_rank_pct,
                "policy_rank_live_score_abs_diff": (
                    abs(rank_from_live_score.policy_rank_pct - _safe_float(row.get("policy_rank_pct")))
                    if np.isfinite(rank_from_live_score.policy_rank_pct)
                    else np.nan
                ),
                "policy_rank_recomputed_from_training_score": rank_from_training_score.policy_rank_pct,
                "policy_rank_training_score_abs_diff": (
                    abs(rank_from_training_score.policy_rank_pct - _safe_float(row.get("policy_rank_pct")))
                    if np.isfinite(rank_from_training_score.policy_rank_pct)
                    else np.nan
                ),
                "policy_rank_reference_source": policy_source,
                "policy_rank_reference_current_hash": _file_hash(policy_source),
                "policy_rank_reference_current_mtime": policy_mtime,
                "policy_rank_reference_current_after_decision": policy_reference_after_decision,
                "auction_rank_live": _safe_float(row.get("auction_rank_pct")),
                "auction_rank_recomputed_from_live_score": auction_from_live_score.policy_rank_pct,
                "auction_rank_live_score_abs_diff": (
                    abs(auction_from_live_score.policy_rank_pct - _safe_float(row.get("auction_rank_pct")))
                    if np.isfinite(auction_from_live_score.policy_rank_pct)
                    else np.nan
                ),
                "auction_rank_recomputed_from_training_score": auction_from_training_score.policy_rank_pct,
                "auction_rank_training_score_abs_diff": (
                    abs(auction_from_training_score.policy_rank_pct - _safe_float(row.get("auction_rank_pct")))
                    if np.isfinite(auction_from_training_score.policy_rank_pct)
                    else np.nan
                ),
                "auction_rank_reference_source": auction_source,
                "auction_rank_reference_current_hash": _file_hash(auction_source),
                "auction_rank_reference_current_mtime": auction_mtime,
                "auction_rank_reference_current_after_decision": auction_reference_after_decision,
            }
        )
    report = pd.DataFrame(rows)
    fresh_policy = (
        ~report["policy_rank_reference_current_after_decision"].fillna(False)
        if not report.empty and "policy_rank_reference_current_after_decision" in report
        else pd.Series(dtype=bool)
    )
    fresh_auction = (
        ~report["auction_rank_reference_current_after_decision"].fillna(False)
        if not report.empty and "auction_rank_reference_current_after_decision" in report
        else pd.Series(dtype=bool)
    )
    fresh_current = fresh_policy & fresh_auction if not report.empty else pd.Series(dtype=bool)
    summary = {
        "rows": int(len(report)),
        "base_mismatches": int((report["base_abs_diff"] > 1e-9).sum()) if not report.empty else 0,
        "meta_mismatches": int((report["meta_abs_diff"] > 1e-9).sum()) if not report.empty else 0,
        "policy_rank_rows_with_current_reference_after_decision": int(report["policy_rank_reference_current_after_decision"].sum()) if not report.empty else 0,
        "auction_rank_rows_with_current_reference_after_decision": int(report["auction_rank_reference_current_after_decision"].sum()) if not report.empty else 0,
        "policy_rank_mismatches_on_fresh_reference": int(((report["policy_rank_live_score_abs_diff"] > 1e-12) & fresh_policy).sum()) if not report.empty else 0,
        "auction_rank_mismatches_on_fresh_reference": int(((report["auction_rank_live_score_abs_diff"] > 1e-12) & fresh_auction).sum()) if not report.empty else 0,
        "max_base_abs_diff": float(report["base_abs_diff"].max()) if not report.empty and report["base_abs_diff"].notna().any() else np.nan,
        "max_meta_abs_diff": float(report["meta_abs_diff"].max()) if not report.empty and report["meta_abs_diff"].notna().any() else np.nan,
        "max_policy_rank_live_score_abs_diff": float(report["policy_rank_live_score_abs_diff"].max()) if not report.empty and report["policy_rank_live_score_abs_diff"].notna().any() else np.nan,
        "max_auction_rank_live_score_abs_diff": float(report["auction_rank_live_score_abs_diff"].max()) if not report.empty and report["auction_rank_live_score_abs_diff"].notna().any() else np.nan,
        "fresh_current_rows": int(fresh_current.sum()) if not report.empty else 0,
        "fresh_current_base_mismatches": int(((report["base_abs_diff"] > 1e-9) & fresh_current).sum()) if not report.empty else 0,
        "fresh_current_meta_mismatches": int(((report["meta_abs_diff"] > 1e-9) & fresh_current).sum()) if not report.empty else 0,
        "fresh_current_policy_rank_mismatches": int(((report["policy_rank_live_score_abs_diff"] > 1e-12) & fresh_current).sum()) if not report.empty else 0,
        "fresh_current_auction_rank_mismatches": int(((report["auction_rank_live_score_abs_diff"] > 1e-12) & fresh_current).sum()) if not report.empty else 0,
    }
    return report, summary


def symbol_universe_audit(
    *,
    ledger: pd.DataFrame,
    candidates_path: Path,
    feature_root: Path,
    ohlcv_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    candidates = pd.read_parquet(candidates_path)
    policy_symbols = {_compact_symbol(s) for s in candidates["symbol"].dropna().unique()}
    live_candidate_symbols = {_compact_symbol(s) for s in ledger["symbol"].dropna().unique()}
    live_traded_symbols = {
        _compact_symbol(s)
        for s in ledger.loc[ledger["portfolio_decision"].map(_is_traded), "symbol"].dropna().unique()
    }
    feature_symbols = {
        p.name.replace("symbol=", "").replace(".parquet", "")
        for p in feature_root.glob("symbol=*.parquet")
    }
    ohlcv_symbols = {
        p.name.replace("symbol=", "")
        for p in ohlcv_root.glob("symbol=*")
        if p.is_dir()
    }
    rows = []
    for sym in sorted(policy_symbols):
        has_features = sym in feature_symbols
        has_ohlcv = sym in ohlcv_symbols
        status = "available"
        if not has_ohlcv:
            status = "missing_live_ohlcv"
        elif not has_features:
            status = "missing_training_feature_store"
        rows.append(
            {
                "symbol": sym,
                "in_policy_oos": True,
                "in_live_candidates": sym in live_candidate_symbols,
                "live_traded": sym in live_traded_symbols,
                "has_training_features": has_features,
                "has_live_ohlcv": has_ohlcv,
                "availability_status": status,
            }
        )
    report = pd.DataFrame(rows)
    summary = {
        "policy_symbols": int(len(policy_symbols)),
        "live_candidate_symbols": int(len(live_candidate_symbols)),
        "live_traded_symbols": int(len(live_traded_symbols)),
        "policy_available_symbols": int((report["availability_status"] == "available").sum()) if not report.empty else 0,
        "available_not_seen_in_live_candidates": int(((report["availability_status"] == "available") & ~report["in_live_candidates"]).sum()) if not report.empty else 0,
        "missing_live_ohlcv": int((report["availability_status"] == "missing_live_ohlcv").sum()) if not report.empty else 0,
        "missing_training_features": int((report["availability_status"] == "missing_training_feature_store").sum()) if not report.empty else 0,
    }
    return report, summary


def non_traded_symbol_oos(
    *,
    candidates_path: Path,
    ledger: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cand = pd.read_parquet(candidates_path)
    traded = {
        _compact_symbol(s)
        for s in ledger.loc[ledger["portfolio_decision"].map(_is_traded), "symbol"].dropna().unique()
    }
    cand = cand.copy()
    cand["symbol_compact"] = cand["symbol"].map(_compact_symbol)
    cand["live_traded_symbol"] = cand["symbol_compact"].isin(traded)
    cand["gross_hit"] = pd.to_numeric(cand["gross_return"], errors="coerce") > 0
    cand["net_hit"] = pd.to_numeric(cand["net_return"], errors="coerce") > 0
    score = pd.to_numeric(cand.get("auction_rank_score", cand.get("normalized_rank_score")), errors="coerce")
    cand["confidence_band"] = pd.cut(
        score,
        bins=[0.0, 0.70, 0.76, 0.82, 0.88, 0.94, 1.0],
        include_lowest=True,
    )
    rows = []
    for keys, grp in cand.groupby(["strategy_id", "confidence_band", "live_traded_symbol"], observed=True):
        strategy_id, band, is_live_traded = keys
        rows.append(
            {
                "strategy_id": strategy_id,
                "confidence_band": str(band),
                "live_traded_symbol_group": bool(is_live_traded),
                "rows": int(len(grp)),
                "symbols": int(grp["symbol_compact"].nunique()),
                "gross_hit_rate": float(grp["gross_hit"].mean()) if len(grp) else np.nan,
                "net_hit_rate": float(grp["net_hit"].mean()) if len(grp) else np.nan,
                "mean_net_return": float(pd.to_numeric(grp["net_return"], errors="coerce").mean()),
                "mean_gross_return": float(pd.to_numeric(grp["gross_return"], errors="coerce").mean()),
            }
        )
    report = pd.DataFrame(rows)
    piv = report.pivot_table(
        index=["strategy_id", "confidence_band"],
        columns="live_traded_symbol_group",
        values="gross_hit_rate",
        aggfunc="first",
    )
    diffs = []
    if not report.empty:
        report["gross_hit_delta_vs_live_traded"] = np.nan
        report["within_5pct_of_live_traded"] = False
    for idx, vals in piv.iterrows():
        traded_hr = vals.get(True, np.nan)
        non_hr = vals.get(False, np.nan)
        if np.isfinite(traded_hr) and np.isfinite(non_hr):
            delta = float(non_hr - traded_hr)
            diffs.append(delta)
            strategy_id, confidence_band = idx
            mask = (
                report["strategy_id"].astype(str).eq(str(strategy_id))
                & report["confidence_band"].astype(str).eq(str(confidence_band))
                & ~report["live_traded_symbol_group"].astype(bool)
            )
            report.loc[mask, "gross_hit_delta_vs_live_traded"] = delta
            report.loc[mask, "within_5pct_of_live_traded"] = bool(delta >= -0.05)
    summary = {
        "rows": int(len(report)),
        "live_traded_symbols": int(len(traded)),
        "comparable_confidence_bands": int(len(diffs)),
        "non_traded_bands_within_5pct": int(
            report.loc[
                ~report.get("live_traded_symbol_group", pd.Series(False, index=report.index)).astype(bool),
                "within_5pct_of_live_traded",
            ].sum()
        )
        if not report.empty and "within_5pct_of_live_traded" in report
        else 0,
        "min_non_traded_minus_traded_gross_hit_delta": float(np.nanmin(diffs)) if diffs else np.nan,
        "non_traded_within_5pct_at_similar_confidence": bool(diffs and np.nanmin(diffs) >= -0.05),
    }
    return report, summary


def cross_strategy_symbol_oos_eligibility(
    *,
    candidates_path: Path,
    ledger: pd.DataFrame,
    min_gross_hit_delta: float = -0.05,
    min_mean_net_return: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Average symbol OOS performance across strategy/confidence bands.

    The baseline is the live-traded symbol group in the same strategy and
    confidence band. A symbol is deployable when its weighted OOS gross-hit
    delta is within tolerance and its weighted mean net return is positive.
    """
    cand = pd.read_parquet(candidates_path)
    traded = {
        _compact_symbol(s)
        for s in ledger.loc[ledger["portfolio_decision"].map(_is_traded), "symbol"].dropna().unique()
    }
    cand = cand.copy()
    cand["symbol_compact"] = cand["symbol"].map(_compact_symbol)
    cand["live_traded_symbol"] = cand["symbol_compact"].isin(traded)
    cand["gross_hit"] = pd.to_numeric(cand["gross_return"], errors="coerce") > 0
    cand["net_return_num"] = pd.to_numeric(cand["net_return"], errors="coerce")
    score = pd.to_numeric(cand.get("auction_rank_score", cand.get("normalized_rank_score")), errors="coerce")
    cand["confidence_band"] = pd.cut(
        score,
        bins=[0.0, 0.70, 0.76, 0.82, 0.88, 0.94, 1.0],
        include_lowest=True,
    )

    baseline = (
        cand[cand["live_traded_symbol"]]
        .groupby(["strategy_id", "confidence_band"], observed=True)
        .agg(
            baseline_rows=("symbol", "size"),
            baseline_gross_hit_rate=("gross_hit", "mean"),
            baseline_mean_net_return=("net_return_num", "mean"),
        )
        .reset_index()
    )
    by_symbol_band = (
        cand.groupby(["symbol_compact", "strategy_id", "confidence_band"], observed=True)
        .agg(
            rows=("symbol", "size"),
            gross_hit_rate=("gross_hit", "mean"),
            net_hit_rate=("net_return_num", lambda s: float((pd.to_numeric(s, errors="coerce") > 0).mean())),
            mean_net_return=("net_return_num", "mean"),
            mean_gross_return=("gross_return", "mean"),
        )
        .reset_index()
        .merge(baseline, on=["strategy_id", "confidence_band"], how="inner")
    )
    if by_symbol_band.empty:
        empty = pd.DataFrame(
            columns=[
                "symbol",
                "rows",
                "strategy_band_count",
                "gross_hit_rate",
                "net_hit_rate",
                "mean_net_return",
                "mean_gross_return",
                "gross_hit_delta_vs_live_traded",
                "net_return_delta_vs_live_traded",
                "deployable_symbol",
                "reject_reason",
            ]
        )
        return empty, {
            "symbols_evaluated": 0,
            "deployable_symbols": 0,
            "min_gross_hit_delta": float(min_gross_hit_delta),
            "min_mean_net_return": float(min_mean_net_return),
        }

    by_symbol_band["gross_hit_delta_vs_live_traded"] = (
        by_symbol_band["gross_hit_rate"] - by_symbol_band["baseline_gross_hit_rate"]
    )
    by_symbol_band["net_return_delta_vs_live_traded"] = (
        by_symbol_band["mean_net_return"] - by_symbol_band["baseline_mean_net_return"]
    )

    rows: list[dict[str, Any]] = []
    for symbol, grp in by_symbol_band.groupby("symbol_compact", sort=True):
        weights = pd.to_numeric(grp["rows"], errors="coerce").fillna(0.0).astype(float)
        total_weight = float(weights.sum())
        if total_weight <= 0:
            continue

        def _wavg(column: str) -> float:
            vals = pd.to_numeric(grp[column], errors="coerce")
            mask = vals.notna() & weights.gt(0)
            if not mask.any():
                return np.nan
            return float(np.average(vals[mask], weights=weights[mask]))

        gross_delta = _wavg("gross_hit_delta_vs_live_traded")
        mean_net = _wavg("mean_net_return")
        gross_ok = bool(np.isfinite(gross_delta) and gross_delta >= float(min_gross_hit_delta))
        net_ok = bool(np.isfinite(mean_net) and mean_net > float(min_mean_net_return))
        reasons = []
        if not gross_ok:
            reasons.append("gross_hit_delta_below_tolerance")
        if not net_ok:
            reasons.append("mean_net_return_not_positive")
        rows.append(
            {
                "symbol": symbol,
                "rows": int(total_weight),
                "strategy_band_count": int(len(grp)),
                "gross_hit_rate": _wavg("gross_hit_rate"),
                "net_hit_rate": _wavg("net_hit_rate"),
                "mean_net_return": mean_net,
                "mean_gross_return": _wavg("mean_gross_return"),
                "gross_hit_delta_vs_live_traded": gross_delta,
                "net_return_delta_vs_live_traded": _wavg("net_return_delta_vs_live_traded"),
                "baseline_gross_hit_rate": _wavg("baseline_gross_hit_rate"),
                "baseline_mean_net_return": _wavg("baseline_mean_net_return"),
                "deployable_symbol": bool(gross_ok and net_ok),
                "reject_reason": ",".join(reasons),
            }
        )

    report = pd.DataFrame(rows).sort_values(
        ["deployable_symbol", "gross_hit_delta_vs_live_traded", "mean_net_return"],
        ascending=[False, False, False],
    )
    deployable = report[report["deployable_symbol"].astype(bool)]
    summary = {
        "symbols_evaluated": int(len(report)),
        "deployable_symbols": int(len(deployable)),
        "rejected_symbols": int((~report["deployable_symbol"].astype(bool)).sum()),
        "min_gross_hit_delta": float(min_gross_hit_delta),
        "min_mean_net_return": float(min_mean_net_return),
        "deployable_symbol_list": deployable["symbol"].astype(str).tolist(),
    }
    return report, summary


def write_report(path: Path, summaries: dict[str, Any]) -> None:
    decision = summaries.get("decision_replay", {})
    feature = summaries.get("feature_parity", {})
    pred = summaries.get("prediction_and_rank_parity", {})
    universe = summaries.get("symbol_universe", {})
    non_traded = summaries.get("non_traded_symbol_oos", {})
    strict_rows = int(decision.get("strict_replay_rows") or 0)
    strict_mismatches = int(decision.get("strict_decision_mismatches") or 0)
    current_feature = feature.get("fresh_current", {}) if isinstance(feature.get("fresh_current"), dict) else {}
    current_pred = pred.get("fresh_current", {}) if isinstance(pred.get("fresh_current"), dict) else {}
    feature_mismatches = int(current_feature.get("mismatches", feature.get("mismatches") or 0))
    base_mismatches = int(current_pred.get("base_mismatches", pred.get("base_mismatches") or 0))
    meta_mismatches = int(current_pred.get("meta_mismatches", pred.get("meta_mismatches") or 0))
    fresh_policy_rank_mismatches = int(
        current_pred.get(
            "policy_rank_mismatches",
            pred.get("policy_rank_mismatches_on_fresh_reference") or 0,
        )
    )
    fresh_auction_rank_mismatches = int(
        current_pred.get(
            "auction_rank_mismatches",
            pred.get("auction_rank_mismatches_on_fresh_reference") or 0,
        )
    )
    rank_refs_after_decision = int(pred.get("policy_rank_rows_with_current_reference_after_decision") or 0) + int(
        pred.get("auction_rank_rows_with_current_reference_after_decision") or 0
    )
    missing_symbols = int(universe.get("missing_live_ohlcv") or 0) + int(universe.get("missing_training_features") or 0)
    strict_decision_verdict = (
        "PASS" if strict_rows > 0 and strict_mismatches == 0 else "UNPROVEN" if strict_rows == 0 else "FAIL"
    )
    feature_verdict = "PASS" if feature_mismatches == 0 else "FAIL"
    prediction_verdict = "PASS" if base_mismatches == 0 and meta_mismatches == 0 else "FAIL"
    if int(current_pred.get("rows") or 0) > 0:
        rank_verdict = (
            "PASS"
            if fresh_policy_rank_mismatches == 0 and fresh_auction_rank_mismatches == 0
            else "FAIL"
        )
    else:
        rank_verdict = (
            "PASS"
            if fresh_policy_rank_mismatches == 0 and fresh_auction_rank_mismatches == 0 and rank_refs_after_decision == 0
            else "UNPROVEN"
            if fresh_policy_rank_mismatches == 0 and fresh_auction_rank_mismatches == 0
            else "FAIL"
        )
    universe_verdict = "PASS" if missing_symbols == 0 else "FAIL"
    non_traded_verdict = (
        "PASS" if bool(non_traded.get("non_traded_within_5pct_at_similar_confidence")) else "PARTIAL"
    )
    lines = [
        "# Live Backtest Decision Reconciliation",
        "",
        "This audit uses current live ledger candidates, deployed rank-reference artifacts, and the training feature store.",
        "",
        "## Verdicts",
        "",
        f"- Same-signal-bar would-open/actually-open replay: {strict_decision_verdict}",
        f"- Feature parity: {feature_verdict}",
        f"- Prediction parity: {prediction_verdict}",
        f"- Rank-normalization parity: {rank_verdict}",
        f"- Policy symbol availability at inference: {universe_verdict}",
        f"- Available non-traded symbol OOS performance: {non_traded_verdict}",
        "",
        "## Interpretation",
        "",
        "- A strict replay row is one with persisted signal timestamp, rank threshold, rank score, and stale-signal gate state.",
        "- Legacy rows are not used to prove bidirectional equivalence because they predate the full diagnostic fields.",
        "- Legacy traded rows are separately classified against the current rank and stale-entry gates to show whether they would still be admissible today.",
        "- Rank parity is marked UNPROVEN, not FAIL, when the current rank-reference file was modified after the live decision.",
        "- Feature/prediction verdicts are based on fresh/current strict rows when available; legacy rows remain in the JSON for diagnosis.",
        "- Feature mismatches in live-sensitive orderbook-derived model inputs are failures for strict parity unless the training-selected cache value is used.",
        "- Symbol availability PASS means policy-OOS symbols have live OHLCV and training feature files; it does not mean they passed masks on the sampled live bars.",
        "",
        "## Summary",
    ]
    for section, payload in summaries.items():
        lines.append(f"### {section}")
        lines.append("```json")
        lines.append(json.dumps(payload, indent=2, default=str))
        lines.append("```")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data_perp")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--feature-source-run-id", default=DEFAULT_FEATURE_SOURCE_RUN_ID)
    parser.add_argument("--ledger", default="data_perp/exchanges/krakenfutures/live_state/prediction_ledgers/20260525_010004_nopenalty/prediction_ledger.parquet")
    parser.add_argument("--out-dir", default="extreme_price_movements/reports/inference_mismatch_investigation/live_backtest_decision_proof_20260605")
    parser.add_argument("--max-prediction-rows", type=int, default=40)
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger = _read_ledger(Path(args.ledger), args.run_id)
    policy = json.loads(
        (Path(args.data_root) / "artifacts" / args.run_id / "policy_params" / "optimized_portfolio_policy_config.json").read_text()
    )
    max_entries = int(((policy.get("concurrency") or {}).get("max_new_entries_per_bar")) or 2)

    decision_report, decision_summary = build_decision_replay(
        ledger,
        max_new_entries_per_bar=max_entries,
    )
    decision_report.to_csv(out_dir / "same_signal_bar_decision_replay.csv", index=False)

    strict_decisions = decision_report[decision_report["strict_replay_eligible"]].copy()
    feature_report, feature_summary = compare_features(
        strict_decisions,
        data_root=args.data_root,
        feature_source_run_id=args.feature_source_run_id,
    )
    feature_report.to_csv(out_dir / "live_vs_training_feature_values.csv", index=False)

    prediction_report, prediction_summary = compare_predictions(
        strict_decisions,
        data_root=args.data_root,
        run_id=args.run_id,
        feature_source_run_id=args.feature_source_run_id,
        max_rows=args.max_prediction_rows,
    )
    if not prediction_report.empty:
        fresh_current_pred = prediction_report[
            ~prediction_report["policy_rank_reference_current_after_decision"].fillna(False)
            & ~prediction_report["auction_rank_reference_current_after_decision"].fillna(False)
        ]
        if not feature_report.empty:
            keys = {
                (
                    pd.Timestamp(row["decision_ts"]).isoformat(),
                    str(row["symbol"]),
                    str(row["side"]),
                    str(row["strategy_id"]),
                )
                for _, row in fresh_current_pred.iterrows()
            }
            current_feature_report = feature_report[
                feature_report.apply(
                    lambda row: (
                        pd.Timestamp(row["decision_ts"]).isoformat(),
                        str(row["symbol"]),
                        str(row["side"]),
                        str(row["strategy_id"]),
                    )
                    in keys,
                    axis=1,
                )
            ]
            feature_summary["fresh_current"] = _feature_report_summary(current_feature_report)
        prediction_summary["fresh_current"] = {
            "rows": int(len(fresh_current_pred)),
            "base_mismatches": int((fresh_current_pred["base_abs_diff"] > 1e-9).sum()),
            "meta_mismatches": int((fresh_current_pred["meta_abs_diff"] > 1e-9).sum()),
            "policy_rank_mismatches": int((fresh_current_pred["policy_rank_live_score_abs_diff"] > 1e-12).sum()),
            "auction_rank_mismatches": int((fresh_current_pred["auction_rank_live_score_abs_diff"] > 1e-12).sum()),
            "max_base_abs_diff": float(fresh_current_pred["base_abs_diff"].max()) if fresh_current_pred["base_abs_diff"].notna().any() else np.nan,
            "max_meta_abs_diff": float(fresh_current_pred["meta_abs_diff"].max()) if fresh_current_pred["meta_abs_diff"].notna().any() else np.nan,
            "max_policy_rank_live_score_abs_diff": float(fresh_current_pred["policy_rank_live_score_abs_diff"].max()) if fresh_current_pred["policy_rank_live_score_abs_diff"].notna().any() else np.nan,
            "max_auction_rank_live_score_abs_diff": float(fresh_current_pred["auction_rank_live_score_abs_diff"].max()) if fresh_current_pred["auction_rank_live_score_abs_diff"].notna().any() else np.nan,
        }
    prediction_report.to_csv(out_dir / "live_vs_training_predictions_and_ranks.csv", index=False)

    candidate_path = Path(args.data_root) / "artifacts" / args.run_id / "simple_policy_optimiser" / "simple_policy_candidates_deployable.parquet"
    universe_report, universe_summary = symbol_universe_audit(
        ledger=ledger,
        candidates_path=candidate_path,
        feature_root=Path(args.data_root) / "features" / args.feature_source_run_id,
        ohlcv_root=Path(args.data_root) / "exchanges" / "krakenfutures" / "ohlcv",
    )
    universe_report.to_csv(out_dir / "symbol_universe_audit.csv", index=False)

    non_traded_report, non_traded_summary = non_traded_symbol_oos(
        candidates_path=candidate_path,
        ledger=ledger,
    )
    non_traded_report.to_csv(out_dir / "non_traded_available_symbols_oos_metrics.csv", index=False)

    symbol_eligibility_report, symbol_eligibility_summary = cross_strategy_symbol_oos_eligibility(
        candidates_path=candidate_path,
        ledger=ledger,
    )
    symbol_eligibility_report.to_csv(out_dir / "cross_strategy_symbol_oos_eligibility.csv", index=False)
    eligibility_payload = {
        "schema_version": "cross_strategy_oos_symbol_eligibility_v1",
        "generated_by": "live_backtest_decision_reconciliation",
        "run_id": args.run_id,
        "feature_source_run_id": args.feature_source_run_id,
        "source_candidates_path": str(candidate_path),
        "rule": {
            "weighted_gross_hit_delta_vs_live_traded_min": -0.05,
            "weighted_mean_net_return_min_exclusive": 0.0,
            "weight": "policy_oos_rows_per_symbol_strategy_confidence_band",
        },
        **symbol_eligibility_summary,
    }
    (out_dir / "cross_strategy_symbol_oos_eligibility.json").write_text(
        json.dumps(eligibility_payload, indent=2, default=str),
        encoding="utf-8",
    )
    policy_params_dir = Path(args.data_root) / "artifacts" / args.run_id / "policy_params"
    policy_params_dir.mkdir(parents=True, exist_ok=True)
    (policy_params_dir / "cross_strategy_symbol_oos_eligibility.json").write_text(
        json.dumps(eligibility_payload, indent=2, default=str),
        encoding="utf-8",
    )

    summaries = {
        "decision_replay": decision_summary,
        "feature_parity": feature_summary,
        "prediction_and_rank_parity": prediction_summary,
        "symbol_universe": universe_summary,
        "non_traded_symbol_oos": non_traded_summary,
        "cross_strategy_symbol_oos_eligibility": {
            k: v for k, v in symbol_eligibility_summary.items() if k != "deployable_symbol_list"
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2, default=str), encoding="utf-8")
    write_report(out_dir / "live_backtest_decision_reconciliation.md", summaries)
    print(json.dumps(summaries, indent=2, default=str))
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
