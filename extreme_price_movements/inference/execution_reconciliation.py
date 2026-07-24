"""Reconcile policy execution assumptions against live inference observations.

This module intentionally separates two related checks:

1. Spread/slippage reconciliation: compare the simple-policy optimiser's
   expected friction proxy with what live inference observed around
   decision/order/fill time.
2. Decision replay reconciliation: run deployed portfolio-policy replay over
   live ledger candidates to detect rows that a backtest-style policy would
   accept while live inference did not open.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from extreme_price_movements.inference.model_orchestrator import (
    DELETED_MODEL_FEATURE_KEYS,
    ModelOrchestrator,
    _effective_selected_feature_contract,
)
from extreme_price_movements.inference.canonical_meta_postprocessor import (
    CanonicalMetaPostprocessor,
)
from extreme_price_movements.inference.side_residual_expert import (
    SideResidualExpertBundle,
)
from extreme_price_movements.inference.parity import (
    calibrated_score_and_threshold,
    strategy_core_id,
)
from extreme_price_movements.inference.policy_rank_reference import (
    PolicyRankReferenceStore,
)
from extreme_price_movements.inference.threshold_basis_policy import (
    apply_threshold_basis_policy_to_decisions,
    load_threshold_basis_policy,
)
from extreme_price_movements.model_loader import load_full_state
from extreme_price_movements.portfolio_policy_replay import (
    dynamic_threshold_for_count,
    load_portfolio_policy_params,
    replay_candidates,
)
from extreme_price_movements.simple_position_sizer import load_calibration_curves


JOIN_KEYS = ["timestamp", "symbol", "side", "strategy_id"]

ACTIVE_CHAIN_FIELDS = (
    "base_pred",
    "meta_pred",
    "v9_tail95_predecessor_rank",
    "score_regime_calibrated",
    "expected_net_ev_after_1pct",
    "expected_ev_rank_score",
)

ACTIVE_ADMISSION_FIELDS = (
    "threshold_basis_selected",
    "threshold_basis_rank_score",
    "threshold_basis_corrected_expected_ev",
    "threshold_basis_corrected_expected_ev_rank",
    "threshold_basis_side_archetype_recent_ev_correction",
    "threshold_basis_ev_target_local_support",
    "threshold_basis_reference_asof",
    "threshold_basis_reason",
)


REPLAY_FIELD_GROUPS: tuple[dict[str, Any], ...] = (
    {
        "group": "decision_identity",
        "scope": "all",
        "required": (
            ("signal_bar_ts",),
            ("symbol",),
            ("side",),
            ("strategy_id",),
            ("portfolio_decision",),
        ),
    },
    {
        "group": "feature_prediction_replay",
        "scope": "all",
        "required": (
            ("base_model_features_json",),
            ("base_model_feature_values_json",),
            ("base_pred",),
            ("meta_pred",),
            ("calibrated_score", "raw_prediction_score"),
        ),
    },
    {
        "group": "rank_threshold_replay",
        "scope": "all",
        "required": (
            ("policy_rank_pct", "normalized_rank_score", "threshold_rank_score"),
            ("auction_rank_pct", "threshold_rank_score"),
            ("threshold_rank_score", "effective_threshold", "final_threshold", "initial_rank_threshold"),
            ("threshold_rank_score_source", "rank_score_source"),
            ("passed_rank_gate",),
        ),
    },
    {
        "group": "entry_timing_attribution",
        "scope": "traded",
        "required": (
            ("signal_bar_close_ts",),
            ("decision_ts",),
            ("theoretical_entry_price", "policy_entry_price", "signal_price"),
            ("expected_fill_price", "expected_entry_price"),
            ("realized_entry_price", "entry_price_actual"),
            ("signal_to_entry_seconds",),
            ("decision_to_entry_seconds",),
            ("hourly_close_to_latest_decision_price_bps", "signal_gap_bps"),
            ("decision_price_to_fill_bps", "actual_fill_vs_expected_bps"),
        ),
    },
    {
        "group": "spread_slippage_cost_attribution",
        "scope": "traded",
        "required": (
            ("ticker_spread_bps", "spread_bps", "spread_proxy_bps"),
            ("expected_fill_slippage_bps", "orderbook_live_slippage_bps", "orderbook_slippage_bps", "slippage_bps"),
            ("expected_total_entry_friction_bps", "max_entry_friction_bps"),
            ("fee_bps", "entry_fee_bps", "realized_fee_bps"),
            ("ev_haircut_bps", "expected_friction_drag_bps"),
        ),
    },
    {
        "group": "order_fill_identity",
        "scope": "traded",
        "required": (
            ("position_id",),
            ("order_id",),
            ("was_traded",),
            ("outcome_status", "portfolio_decision"),
        ),
    },
    {
        "group": "exact_portfolio_state_replay",
        "scope": "all",
        "required": (
            ("portfolio_state_snapshot_json", "open_positions_before_json", "active_positions_before_json"),
            ("portfolio_state_snapshot_hash", "portfolio_state_hash"),
            ("wallet_before", "wallet_value"),
            ("open_positions_before", "open_positions_before_count"),
            ("cooldowns_before_json", "recent_losing_trade_cooldown_state_json"),
            ("portfolio_priority",),
        ),
    },
)


def _read_table(path: str | Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    if p.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(p)
    raise ValueError(f"Unsupported table type: {p}")


def _filter_table_since(frame: pd.DataFrame, since: str | pd.Timestamp | None) -> pd.DataFrame:
    """Restrict live audit inputs to the current supervised inference session."""
    if frame.empty or since is None:
        return frame
    cutoff = pd.to_datetime(since, utc=True, errors="coerce")
    if pd.isna(cutoff):
        raise ValueError(f"Invalid reconciliation --since timestamp: {since!r}")

    # Lifecycle audit rows historically stored a host-local, timezone-naive
    # display timestamp even though entry_time/exit_time were canonical UTC.
    # Prefer the event-specific contract and only accept the generic timestamp
    # as a fallback when it carries an explicit timezone. This prevents an old
    # CEST row from being interpreted as UTC and leaking into the current live
    # session audit.
    if "lifecycle_event" in frame.columns:
        event = frame["lifecycle_event"].fillna("").astype(str).str.lower()
        action = (
            frame["action"].fillna("").astype(str).str.lower()
            if "action" in frame.columns
            else pd.Series("", index=frame.index, dtype=object)
        )
        is_exit = event.str.contains("exit|close", regex=True) | action.eq("exit")
        is_entry = event.str.contains("entry", regex=False) | action.eq("enter")
        event_ts = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
        if "exit_time" in frame.columns:
            exit_ts = pd.to_datetime(frame["exit_time"], utc=True, errors="coerce")
            event_ts.loc[is_exit] = exit_ts.loc[is_exit]
        if "entry_time" in frame.columns:
            entry_ts = pd.to_datetime(frame["entry_time"], utc=True, errors="coerce")
            event_ts.loc[is_entry] = entry_ts.loc[is_entry]
        if "timestamp" in frame.columns:
            raw_timestamp = frame["timestamp"]

            def _has_explicit_timezone(value: Any) -> bool:
                if value is None or pd.isna(value):
                    return False
                try:
                    return pd.Timestamp(value).tzinfo is not None
                except (TypeError, ValueError):
                    return False

            def _parse_explicit_utc(value: Any) -> pd.Timestamp:
                if not _has_explicit_timezone(value):
                    return pd.NaT
                return pd.Timestamp(value).tz_convert("UTC")

            timezone_aware = raw_timestamp.map(_has_explicit_timezone)
            fallback = event_ts.isna() & timezone_aware
            if bool(fallback.any()):
                generic_ts = raw_timestamp.map(_parse_explicit_utc)
                event_ts.loc[fallback] = generic_ts.loc[fallback]
        return frame.loc[event_ts >= cutoff].copy()

    for column in (
        "decision_ts",
        "timestamp",
        "entry_time",
        "exit_time",
        "created_at",
        "updated_at",
    ):
        if column not in frame.columns:
            continue
        values = pd.to_datetime(frame[column], utc=True, errors="coerce")
        return frame.loc[values >= cutoff].copy()
    return frame.iloc[0:0].copy()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(float(value)) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value) if not isinstance(value, (list, tuple, dict, str)) else False:
        return None
    return value


def _is_present(values: pd.Series) -> pd.Series:
    if values.empty:
        return pd.Series(dtype=bool)
    if pd.api.types.is_bool_dtype(values):
        return values.notna()
    if pd.api.types.is_numeric_dtype(values):
        return values.notna() & np.isfinite(pd.to_numeric(values, errors="coerce"))
    text = values.astype("string")
    return values.notna() & text.str.strip().ne("") & text.str.lower().ne("nan") & text.str.lower().ne("none")


def _alternative_present(df: pd.DataFrame, alternatives: Sequence[str]) -> pd.Series:
    present = pd.Series(False, index=df.index, dtype=bool)
    for col in alternatives:
        if col not in df.columns:
            continue
        present |= _is_present(df[col]).reindex(df.index, fill_value=False)
    return present


def _num(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def _first_numeric(df: pd.DataFrame, cols: Sequence[str], default: float = np.nan) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    for col in cols:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        out = out.where(out.notna(), vals)
    if not pd.isna(default):
        out = out.fillna(float(default))
    return out


def _abs_delta(left: pd.Series, right: pd.Series) -> pd.Series:
    return (pd.to_numeric(left, errors="coerce") - pd.to_numeric(right, errors="coerce")).abs()


def _json_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return {}
    text = str(value).strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except Exception:
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _feature_frame_from_json(value: Any, *, symbol: str) -> pd.DataFrame:
    mapping = _json_mapping(value)
    if not mapping:
        return pd.DataFrame()
    numeric: dict[str, float] = {}
    for key, raw in mapping.items():
        try:
            val = float(raw)
        except (TypeError, ValueError):
            val = np.nan
        numeric[str(key)] = val
    return pd.DataFrame([numeric], index=[str(symbol)])


def _feature_frame_from_snapshot(
    value: Any,
    *,
    symbol: str,
    categorical: Sequence[str] = (),
) -> pd.DataFrame:
    """Decode a logged model snapshot without destroying categorical keys."""
    mapping = _json_mapping(value)
    if not mapping:
        return pd.DataFrame()
    categorical_set = {str(name) for name in categorical}
    decoded: dict[str, Any] = {}
    for key, raw in mapping.items():
        name = str(key)
        if name in categorical_set:
            decoded[name] = str(raw)
            continue
        try:
            decoded[name] = float(raw)
        except (TypeError, ValueError):
            decoded[name] = np.nan
    return pd.DataFrame([decoded], index=[str(symbol)])


def _normalise_side(value: Any, strategy_id: Any = "") -> str:
    raw = str(value or "").lower()
    if raw in {"1", "1.0", "long", "buy"}:
        return "long"
    if raw in {"-1", "-1.0", "short", "sell"}:
        return "short"
    sid = str(strategy_id or "").lower()
    if sid.startswith("short"):
        return "short"
    return "long"


def _resolve_meta_model_for_prediction_replay(
    orchestrator: ModelOrchestrator,
    *,
    side: str,
    strategy_id: str,
    meta_model_key: Any = None,
) -> tuple[str, Any]:
    """Resolve the deployed meta model used by a ledger row."""
    meta_models = getattr(orchestrator, "meta_models", {}) or {}
    explicit = str(meta_model_key or "").strip()
    if explicit and explicit in meta_models:
        return explicit, meta_models.get(explicit)
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
    return explicit or str(strategy_id), None


def _logged_meta_prediction(
    orchestrator: ModelOrchestrator,
    meta_features: pd.DataFrame,
    *,
    side: str,
    strategy_id: str,
    meta_model_key: Any = None,
) -> tuple[float, str]:
    """Predict directly from the logged final meta input matrix when complete.

    Ledger rows store ``_last_meta_model_input`` from live inference. Replaying
    those rows through ``predict_meta`` can re-materialize artifact-backed drift
    columns and mutate the exact matrix being audited, so use the logged final
    input directly whenever it covers the meta model contract.
    """
    if not isinstance(meta_features, pd.DataFrame) or meta_features.empty:
        return np.nan, "missing_logged_meta_features"
    key, meta_model = _resolve_meta_model_for_prediction_replay(
        orchestrator,
        side=side,
        strategy_id=strategy_id,
        meta_model_key=meta_model_key,
    )
    if meta_model is None:
        return np.nan, f"missing_meta_model:{key}"
    feat_cols = _effective_selected_feature_contract(meta_model)
    if not feat_cols and hasattr(meta_model, "feature_columns"):
        feat_cols = [str(c) for c in (getattr(meta_model, "feature_columns", []) or [])]
    feat_cols = [
        str(c)
        for c in (feat_cols or [])
        if str(c) not in DELETED_MODEL_FEATURE_KEYS
    ]
    if not feat_cols:
        return np.nan, "missing_meta_feature_contract"
    missing = [c for c in feat_cols if c not in meta_features.columns]
    if missing:
        return np.nan, f"incomplete_logged_meta_features:{len(missing)}"
    X = meta_features.reindex(columns=feat_cols)
    # Training uses lgbm_pipeline._frame(), which maps every non-finite model
    # input to the neutral value zero.  Reconciliation must use the same
    # adapter; relying on LightGBM's native missing-value branch here can
    # produce a different score from the live/training contract.
    X = (
        X.apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], 0.0)
        .fillna(0.0)
        .astype(np.float32)
    )
    pred = meta_model.predict(X)
    score_alignment = getattr(meta_model, "s52_meta_score_alignment_", None)
    if isinstance(score_alignment, dict) and score_alignment.get("enabled"):
        from extreme_price_movements.inference.s52_meta_score_alignment import (
            apply_s52_meta_score_alignment,
        )

        pred = apply_s52_meta_score_alignment(pred, score_alignment, side=side)
    if len(pred) <= 0:
        return np.nan, "empty_meta_prediction"
    source = (
        "logged_final_meta_input_with_score_alignment"
        if isinstance(score_alignment, dict) and score_alignment.get("enabled")
        else "logged_final_meta_input"
    )
    return float(pred[0]), source


def _load_active_post_meta_chain(
    data_root: str | Path,
    run_id: str,
) -> tuple[SideResidualExpertBundle | None, CanonicalMetaPostprocessor | None]:
    """Load the active side expert and V9/MLP chain when packaged.

    Older bundles do not have these layers, so reconciliation retains the
    shared-meta fallback for them. Current bundles must replay the exact active
    chain rather than incorrectly comparing their expert rank with the shared
    meta backbone score.
    """
    policy_root = Path(data_root) / "artifacts" / str(run_id) / "policy_params"
    expert_path = policy_root / "side_residual_expert.joblib"
    post_paths = (
        policy_root / "v9_tail95_predecessor_bundle.joblib",
        policy_root / "residual_event_state.joblib",
        policy_root / "composite_policy_regime_ev_calibration.json",
    )
    expert = SideResidualExpertBundle.load(expert_path) if expert_path.is_file() else None
    postprocessor = (
        CanonicalMetaPostprocessor.load(
            predecessor_bundle_path=post_paths[0],
            residual_event_state_path=post_paths[1],
            regime_ev_artifact_path=post_paths[2],
        )
        if all(path.is_file() for path in post_paths)
        else None
    )
    return expert, postprocessor


def _replay_active_post_meta_row(
    row: pd.Series,
    *,
    side: str,
    symbol: str,
    replay_base: float,
    side_expert: SideResidualExpertBundle | None,
    postprocessor: CanonicalMetaPostprocessor | None,
) -> dict[str, Any]:
    """Replay active post-meta stages from exact ledger input snapshots."""
    output: dict[str, Any] = {}
    expert_rank = np.nan
    if side_expert is not None:
        expert_input = _feature_frame_from_snapshot(
            row.get("side_residual_expert_input_values_json"),
            symbol=symbol,
            categorical=("side_name", "archetype_policy_key"),
        )
        if not expert_input.empty:
            expert_output = side_expert.transform(expert_input)
            complete = bool(
                expert_output["meta_residual_expert_complete_case"].iloc[0]
            )
            if not complete:
                raise RuntimeError("logged side-residual expert row is incomplete")
            expert_rank = _safe_float_series_value(
                expert_output["score_base_residual_ev_rank_train_reference"].iloc[0]
            )
            output["replay_meta_pred"] = expert_rank
            output["replay_meta_source"] = "logged_side_residual_expert_input"

    if postprocessor is None:
        return output
    post_input = _feature_frame_from_json(
        row.get("meta_postprocessor_input_values_json"), symbol=symbol
    )
    if post_input.empty:
        return output
    stored_meta = _safe_float_series_value(row.get("meta_pred"))
    meta_score = expert_rank if np.isfinite(expert_rank) else stored_meta
    post_input["score"] = float(replay_base)
    post_input["score_base"] = float(replay_base)
    post_input["score_meta_base_soft_label"] = float(meta_score)
    post_input["score_meta_base_soft_label_raw_refit"] = float(meta_score)
    post_input["side_name"] = str(side)
    post_input["archetype_policy_key"] = str(
        row.get("policy_archetype") or "missing"
    )
    post_input["__symbol__"] = str(symbol)
    post_input["__ts__"] = pd.to_datetime(
        row.get("signal_bar_ts") or row.get("timestamp"),
        utc=True,
        errors="coerce",
    )
    transformed = postprocessor.transform(post_input)
    source_map = {
        "v9_tail95_predecessor_rank": "historical_rank",
        "score_regime_calibrated": "score_regime_calibrated",
        "expected_net_ev_after_1pct": "expected_net_ev_after_1pct",
        "expected_ev_rank_score": "expected_ev_rank_score",
    }
    for target, source in source_map.items():
        if source in transformed:
            output[f"replay_{target}"] = _safe_float_series_value(
                transformed[source].iloc[0]
            )
    output["replay_post_meta_source"] = "logged_canonical_postprocessor_input"
    return output


def _normalise_ledger(ledger: pd.DataFrame) -> pd.DataFrame:
    out = ledger.copy()
    decision_ts = (
        pd.to_datetime(out["decision_ts"], utc=True, errors="coerce")
        if "decision_ts" in out.columns
        else pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
    )
    if "signal_bar_ts" in out.columns:
        signal_ts = pd.to_datetime(out["signal_bar_ts"], utc=True, errors="coerce")
    else:
        signal_ts = pd.to_datetime(out.get("timestamp"), utc=True, errors="coerce")
    # Decision replay must distinguish multiple live attempts on the same
    # signal bar.  Use decision time as the replay timestamp when available,
    # while retaining signal_bar_ts as a separate diagnostic field.
    out["timestamp"] = decision_ts.where(decision_ts.notna(), signal_ts)
    out["symbol"] = out.get("symbol", pd.Series("", index=out.index)).astype(str)
    out["strategy_id"] = out.get("strategy_id", pd.Series("", index=out.index)).astype(str)
    side = out.get("side", pd.Series("", index=out.index))
    out["side"] = [
        _normalise_side(side_val, sid)
        for side_val, sid in zip(side, out["strategy_id"])
    ]
    out["_ledger_row_id"] = np.arange(len(out), dtype=np.int64)
    out["_join_seq"] = out.groupby(JOIN_KEYS, dropna=False).cumcount()
    return out


def _active_threshold_policy_path(
    *, data_root: str | Path, run_id: str
) -> Path | None:
    artifact_root = Path(data_root) / "artifacts" / str(run_id)
    config_path = artifact_root / "policy_params" / "optimized_portfolio_policy_config.json"
    if not config_path.is_file():
        return None
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    raw = str(config.get("threshold_basis_policy_path") or "").strip()
    if not raw:
        return None
    path = Path(raw)
    if path.is_file():
        return path
    local = artifact_root / "policy_params" / path.name
    return local if local.is_file() else None


def _replay_active_threshold_policy(
    report: pd.DataFrame,
    ledger: pd.DataFrame,
    *,
    policy_path: str | Path | None,
    tolerance: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Replay the deployed threshold policy from reconstructed active outputs.

    Logged threshold fields are comparators only. The replay inputs are the
    independently reproduced V9/MLP/hierarchical-EV outputs plus observable
    side/archetype identity from the live decision.
    """
    out = report.copy()
    if policy_path is None:
        return out, {"pass": False, "reason": "missing_active_policy_path"}
    policy = load_threshold_basis_policy(policy_path)
    if not policy:
        return out, {"pass": False, "reason": "unreadable_active_policy"}
    if "_ledger_row_id" not in out.columns or "_ledger_row_id" not in ledger.columns:
        return out, {"pass": False, "reason": "missing_ledger_row_id"}

    source = ledger.set_index("_ledger_row_id", drop=False)
    decisions: list[dict[str, Any]] = []
    report_indices: list[Any] = []
    for report_idx, row in out.iterrows():
        if str(row.get("replay_status") or "") != "ok":
            continue
        ledger_id = row.get("_ledger_row_id")
        if ledger_id not in source.index:
            continue
        live = source.loc[ledger_id]
        if isinstance(live, pd.DataFrame):
            live = live.iloc[-1]
        archetype = "missing"
        for field in (
            "archetype_policy_key",
            "policy_archetype",
            "local_side_archetype",
        ):
            value = live.get(field)
            if value is None or pd.isna(value):
                continue
            text = str(value).strip()
            if text and text.lower() not in {"nan", "none", "null", "<na>"}:
                archetype = text
                break
        decisions.append(
            {
                "signal_bar_ts": live.get("signal_bar_ts"),
                "decision_ts": live.get("decision_ts", live.get("timestamp")),
                "symbol": str(live.get("symbol") or ""),
                "side": str(live.get("side") or ""),
                "side_name": str(live.get("side") or ""),
                "strategy_id": str(live.get("strategy_id") or ""),
                "archetype_policy_key": str(archetype),
                "policy_archetype": str(archetype),
                "policy_rank_pct": _safe_float_series_value(
                    live.get("policy_rank_pct")
                ),
                "expected_ev_rank_score": _safe_float_series_value(
                    row.get("replay_expected_ev_rank_score")
                ),
                "expected_net_ev_after_1pct_side_archetype": (
                    _safe_float_series_value(
                        row.get("replay_expected_net_ev_after_1pct")
                    )
                ),
                "expected_net_ev_after_1pct": _safe_float_series_value(
                    row.get("replay_expected_net_ev_after_1pct")
                ),
                "v9_tail95_predecessor_rank": _safe_float_series_value(
                    row.get("replay_v9_tail95_predecessor_rank")
                ),
            }
        )
        report_indices.append(report_idx)

    if not decisions:
        return out, {"pass": False, "reason": "no_replayable_active_rows"}
    apply_threshold_basis_policy_to_decisions(decisions, policy=policy, store=None)

    for report_idx, decision in zip(report_indices, decisions):
        for field in ACTIVE_ADMISSION_FIELDS:
            out.at[report_idx, f"replay_{field}"] = decision.get(field)

    mismatch_total = 0
    field_summary: dict[str, Any] = {}
    scoped = out.loc[report_indices]
    for field in ACTIVE_ADMISSION_FIELDS:
        stored_col = f"stored_{field}"
        replay_col = f"replay_{field}"
        if stored_col not in scoped.columns or replay_col not in scoped.columns:
            field_summary[field] = {"n": 0, "mismatch_rows": len(scoped)}
            mismatch_total += len(scoped)
            continue
        stored = scoped[stored_col]
        replay = scoped[replay_col]
        if field == "threshold_basis_selected":
            matches = stored.astype("boolean").eq(replay.astype("boolean")).fillna(False)
            max_delta = 0.0 if matches.all() else np.nan
        elif field in {
            "threshold_basis_reference_asof",
            "threshold_basis_reason",
        }:
            matches = stored.fillna("<NA>").astype(str).eq(
                replay.fillna("<NA>").astype(str)
            )
            max_delta = 0.0 if matches.all() else np.nan
        else:
            delta = _abs_delta(stored, replay)
            both_missing = stored.isna() & replay.isna()
            matches = both_missing | (
                delta.le(float(tolerance)) & stored.notna() & replay.notna()
            )
            max_delta = float(delta.max()) if delta.notna().any() else np.nan
        out.loc[report_indices, f"{field}_matches"] = matches.to_numpy(dtype=bool)
        mismatches = int((~matches).sum())
        mismatch_total += mismatches
        field_summary[field] = {
            "n": int(len(matches)),
            "mismatch_rows": mismatches,
            "max_abs_delta": max_delta,
        }
    return out, {
        "pass": mismatch_total == 0,
        "policy_id": str(policy.get("policy_id") or ""),
        "policy_path": str(policy_path),
        "rows": int(len(scoped)),
        "selected_rows": int(
            pd.Series([d.get("threshold_basis_selected", False) for d in decisions])
            .fillna(False)
            .astype(bool)
            .sum()
        ),
        "mismatch_rows": int(mismatch_total),
        "fields": field_summary,
    }


def _dedupe_latest_decision_rows(ledger: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty:
        return ledger
    order_ts = pd.to_datetime(
        ledger.get("decision_ts", ledger.get("timestamp")),
        utc=True,
        errors="coerce",
    )
    work = ledger.copy()
    work["_decision_order_ts"] = order_ts
    work = work.sort_values("_decision_order_ts", na_position="first")
    work = work.drop_duplicates(JOIN_KEYS, keep="last")
    work = work.drop(columns=["_decision_order_ts"])
    work["_join_seq"] = work.groupby(JOIN_KEYS, dropna=False).cumcount()
    return work


def _candidate_join_frame(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates is None or candidates.empty:
        return pd.DataFrame(columns=JOIN_KEYS)
    out = candidates.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["symbol"] = out["symbol"].astype(str)
    out["strategy_id"] = out["strategy_id"].astype(str)
    out["side"] = [
        _normalise_side(side, sid)
        for side, sid in zip(out.get("side", ""), out["strategy_id"])
    ]
    out["_join_seq"] = out.groupby(JOIN_KEYS, dropna=False).cumcount()
    join_cols = JOIN_KEYS + ["_join_seq"]
    cols = join_cols + [
        col
        for col in [
            "slippage_bps",
            "orderbook_slippage_bps",
            "expected_friction_bps",
            "entry_slippage_proxy_bps",
            "fees_bps",
            "entry_gap_bps",
            "price_gap_bps",
        ]
        if col in out.columns
    ]
    return out[cols].drop_duplicates(join_cols, keep="last")


def build_spread_slippage_reconciliation(
    prediction_ledger: pd.DataFrame,
    *,
    candidates: Optional[pd.DataFrame] = None,
    dedupe_latest: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ledger = _normalise_ledger(prediction_ledger)
    if dedupe_latest:
        ledger = _dedupe_latest_decision_rows(ledger)
    joined = ledger
    candidate_frame = _candidate_join_frame(
        candidates if candidates is not None else pd.DataFrame()
    )
    if not candidate_frame.empty:
        joined = ledger.merge(
            candidate_frame,
            on=JOIN_KEYS + ["_join_seq"],
            how="left",
            suffixes=("", "_policy_candidate"),
        )

    side_sign = np.where(joined["side"].astype(str).eq("short"), -1.0, 1.0)
    expected_policy_slippage = _first_numeric(
        joined,
        [
            "slippage_bps_policy_candidate",
            "entry_slippage_proxy_bps_policy_candidate",
            "entry_slippage_proxy_bps",
        ],
    )
    expected_policy_orderbook_slippage = _first_numeric(
        joined,
        [
            "orderbook_slippage_bps_policy_candidate",
            "slippage_bps_policy_candidate",
            "entry_slippage_proxy_bps_policy_candidate",
            "entry_slippage_proxy_bps",
        ],
    )
    expected_policy_friction = _first_numeric(
        joined,
        [
            "expected_friction_bps_policy_candidate",
            "expected_friction_bps",
            "expected_total_entry_friction_bps",
        ],
    )
    live_spread = _first_numeric(joined, ["ticker_spread_bps", "spread_bps", "spread_proxy_bps"])
    live_expected_slippage = _first_numeric(
        joined,
        [
            "expected_fill_slippage_bps",
            "orderbook_live_slippage_bps",
            "orderbook_slippage_bps",
            "slippage_bps",
        ],
    )
    live_total_friction = _first_numeric(
        joined,
        ["expected_total_entry_friction_bps", "max_entry_friction_bps"],
    )
    live_formula_friction = live_expected_slippage + live_spread.clip(lower=0.0) / 2.0
    live_total_friction = live_total_friction.where(live_total_friction.notna(), live_formula_friction)

    realized_entry = _num(joined, "realized_entry_price")
    expected_fill = _first_numeric(joined, ["expected_fill_price", "expected_entry_price"])
    theoretical_entry = _first_numeric(
        joined,
        ["theoretical_entry_price", "policy_entry_price", "signal_price"],
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        actual_fill_vs_expected_bps = side_sign * (
            realized_entry.to_numpy(dtype=float)
            / np.maximum(expected_fill.to_numpy(dtype=float), 1e-12)
            - 1.0
        ) * 10000.0
        actual_fill_vs_theoretical_bps = side_sign * (
            realized_entry.to_numpy(dtype=float)
            / np.maximum(theoretical_entry.to_numpy(dtype=float), 1e-12)
            - 1.0
        ) * 10000.0

    out = pd.DataFrame(
        {
            "timestamp": joined["timestamp"],
            "symbol": joined["symbol"],
            "side": joined["side"],
            "strategy_id": joined["strategy_id"],
            "portfolio_decision": joined.get("portfolio_decision"),
            "was_traded": joined.get("was_traded"),
            "expected_policy_slippage_bps": expected_policy_slippage,
            "expected_policy_orderbook_slippage_bps": expected_policy_orderbook_slippage,
            "expected_policy_friction_bps": expected_policy_friction,
            "live_spread_bps": live_spread,
            "live_expected_slippage_bps": live_expected_slippage,
            "live_total_entry_friction_bps": live_total_friction,
            "actual_fill_vs_expected_bps": actual_fill_vs_expected_bps,
            "actual_fill_vs_theoretical_bps": actual_fill_vs_theoretical_bps,
            "policy_vs_live_slippage_delta_bps": live_expected_slippage
            - expected_policy_slippage,
            "policy_vs_live_friction_delta_bps": live_total_friction
            - expected_policy_friction,
            "signal_to_entry_seconds": _num(joined, "signal_to_entry_seconds"),
            "decision_to_entry_seconds": _num(joined, "decision_to_entry_seconds"),
        }
    )
    summary = _summarise_spread_slippage(out)
    return out, summary


def _numeric_summary(values: pd.Series) -> dict[str, Any]:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return {"n": 0}
    return {
        "n": int(len(vals)),
        "mean": float(vals.mean()),
        "median": float(vals.median()),
        "p90": float(vals.quantile(0.90)),
        "max": float(vals.max()),
    }


def _summarise_spread_slippage(rows: pd.DataFrame) -> dict[str, Any]:
    cols = [
        "expected_policy_slippage_bps",
        "expected_policy_friction_bps",
        "live_spread_bps",
        "live_expected_slippage_bps",
        "live_total_entry_friction_bps",
        "actual_fill_vs_expected_bps",
        "actual_fill_vs_theoretical_bps",
        "policy_vs_live_slippage_delta_bps",
        "policy_vs_live_friction_delta_bps",
        "signal_to_entry_seconds",
    ]
    summary: dict[str, Any] = {
        "rows": int(len(rows)),
        "traded_rows": int(pd.Series(rows.get("was_traded", False)).fillna(False).astype(bool).sum()),
        "columns": {col: _numeric_summary(rows[col]) for col in cols if col in rows.columns},
    }
    if "strategy_id" in rows.columns:
        by_strategy = {}
        for strategy_id, group in rows.groupby("strategy_id", dropna=False):
            by_strategy[str(strategy_id)] = {
                "rows": int(len(group)),
                "traded_rows": int(pd.Series(group.get("was_traded", False)).fillna(False).astype(bool).sum()),
                "policy_vs_live_friction_delta_bps": _numeric_summary(
                    group.get("policy_vs_live_friction_delta_bps", pd.Series(dtype=float))
                ),
                "live_total_entry_friction_bps": _numeric_summary(
                    group.get("live_total_entry_friction_bps", pd.Series(dtype=float))
                ),
            }
        summary["by_strategy"] = by_strategy
    return summary


def build_ledger_replay_field_coverage(
    prediction_ledger: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ledger = _normalise_ledger(prediction_ledger)
    live_traded = ledger.apply(_live_traded, axis=1) if not ledger.empty else pd.Series(dtype=bool)
    rows: list[dict[str, Any]] = []
    for group in REPLAY_FIELD_GROUPS:
        group_name = str(group["group"])
        scope = str(group.get("scope", "all"))
        scoped = ledger
        if scope == "traded":
            scoped = ledger.loc[live_traded.reindex(ledger.index, fill_value=False)].copy()
        for alternatives in group["required"]:
            present = _alternative_present(scoped, list(alternatives))
            missing = ~present
            rows.append(
                {
                    "field_group": group_name,
                    "scope": scope,
                    "accepted_alternatives": "|".join(alternatives),
                    "rows_checked": int(len(scoped)),
                    "present_rows": int(present.sum()) if len(scoped) else 0,
                    "missing_rows": int(missing.sum()) if len(scoped) else 0,
                    "coverage_rate": float(present.mean()) if len(scoped) else np.nan,
                    "missing_symbols_sample": ",".join(
                        scoped.loc[missing, "symbol"].astype(str).drop_duplicates().head(12).tolist()
                    )
                    if len(scoped) and "symbol" in scoped.columns
                    else "",
                    "missing_decision_sample": ",".join(
                        scoped.loc[missing, "portfolio_decision"].astype(str).drop_duplicates().head(8).tolist()
                    )
                    if len(scoped) and "portfolio_decision" in scoped.columns
                    else "",
                }
            )
    report = pd.DataFrame(rows)
    if report.empty:
        summary = {
            "ledger_rows": int(len(ledger)),
            "live_traded_rows": int(live_traded.sum()) if len(live_traded) else 0,
            "field_checks": 0,
            "failed_field_checks": 0,
            "critical_missing_rows": 0,
        }
        return report, summary
    failed = report[pd.to_numeric(report["missing_rows"], errors="coerce").fillna(0) > 0]
    traded_failed = failed[failed["scope"].eq("traded")]
    state_group = next(
        (
            group
            for group in REPLAY_FIELD_GROUPS
            if str(group.get("group")) == "exact_portfolio_state_replay"
        ),
        None,
    )
    state_complete = pd.Series(False, index=ledger.index, dtype=bool)
    if state_group is not None and not ledger.empty:
        state_complete = pd.Series(True, index=ledger.index, dtype=bool)
        for alternatives in state_group["required"]:
            state_complete &= _alternative_present(ledger, list(alternatives))
    summary = {
        "ledger_rows": int(len(ledger)),
        "live_traded_rows": int(live_traded.sum()) if len(live_traded) else 0,
        "field_checks": int(len(report)),
        "failed_field_checks": int(len(failed)),
        "failed_traded_field_checks": int(len(traded_failed)),
        "critical_missing_rows": int(pd.to_numeric(failed["missing_rows"], errors="coerce").fillna(0).sum()),
        "exact_portfolio_state_replayable_rows": int(state_complete.sum()) if len(state_complete) else 0,
        "exact_portfolio_state_replayable_rate": (
            float(state_complete.mean()) if len(state_complete) else np.nan
        ),
        "exact_portfolio_state_replayable_traded_rows": (
            int((state_complete & live_traded.reindex(ledger.index, fill_value=False)).sum())
            if len(state_complete)
            else 0
        ),
        "worst_missing": failed.sort_values("missing_rows", ascending=False)
        .head(10)[["field_group", "scope", "accepted_alternatives", "missing_rows", "coverage_rate"]]
        .to_dict(orient="records"),
    }
    return report, summary


def _build_live_candidate_table(ledger: pd.DataFrame) -> pd.DataFrame:
    incoming_row_ids = (
        ledger["_ledger_row_id"].to_numpy(copy=True)
        if "_ledger_row_id" in ledger.columns
        else None
    )
    work = _normalise_ledger(ledger)
    if incoming_row_ids is not None and len(incoming_row_ids) == len(work):
        work["_ledger_row_id"] = incoming_row_ids
    rank = _first_numeric(
        work,
        ["threshold_rank_score", "auction_rank_pct", "normalized_rank_score", "policy_rank_pct"],
    )
    reject_reason = work.get("portfolio_reject_reason", pd.Series("", index=work.index))
    reject_reason = reject_reason.fillna("").astype(str).str.lower()
    live_traded = work.apply(_live_traded, axis=1)
    # This replay diagnoses row-level policy parity from a live decision ledger;
    # it is not an independent exchange-state simulator. Rows live already
    # rejected for non-rank hard reasons must not consume replay auction slots,
    # otherwise capacity/friction/state rejects can displace the rows live
    # actually traded and create false live-accept/replay-reject mismatches.
    hard_veto = (
        (~live_traded)
        & reject_reason.ne("")
        & ~reject_reason.str.contains("rank_below_dynamic_threshold", na=False)
    )
    # Decision replay is a portfolio-policy diagnostic, not a second live
    # execution engine. Rows that live already hard-vetoed for source quality,
    # adverse entry movement, or symbol cooldown must not consume replay
    # auction slots and create false live-accept/replay-reject mismatches for
    # lower-ranked rows that live actually traded.
    rank = rank.mask(hard_veto, -np.inf)
    base_threshold = _first_numeric(
        work,
        ["final_threshold", "effective_threshold", "base_strategy_threshold", "initial_rank_threshold"],
        default=1.0,
    )
    entry = _first_numeric(
        work,
        [
            "theoretical_entry_price",
            "policy_entry_price",
            "expected_entry_price",
            "realized_entry_price",
        ],
        default=1.0,
    ).fillna(1.0)
    timestamp = work["timestamp"]
    holding_bars = pd.Series(4.0, index=work.index)
    gross_return = pd.Series(0.01, index=work.index)
    side_sign = np.where(work["side"].eq("short"), -1.0, 1.0)
    out = pd.DataFrame(
        {
            "_ledger_row_id": work["_ledger_row_id"],
            "_join_seq": work["_join_seq"],
            "timestamp": timestamp,
            "symbol": work["symbol"],
            "side": work["side"],
            "strategy_id": work["strategy_id"],
            "normalized_rank_score": rank,
            "strategy_rank_pct": _first_numeric(work, ["policy_rank_pct", "historical_rank_pct"]),
            "base_strategy_threshold": base_threshold,
            "calibrated_score": _first_numeric(
                work,
                ["raw_prediction_score", "meta_pred", "base_pred", "threshold_rank_score"],
                default=0.0,
            ).fillna(0.0),
            "entry_price": entry,
            "exit_timestamp": timestamp + pd.to_timedelta(holding_bars * 15, unit="m"),
            "exit_price": entry * (1.0 + side_sign * gross_return),
            "net_return": gross_return - 0.001,
            "gross_return": gross_return,
            "fees_bps": 10.0,
            "slippage_bps": _first_numeric(work, ["entry_slippage_proxy_bps"], default=0.0).fillna(0.0),
            "holding_bars": holding_bars,
            "simple_policy_exit_reason": "decision_replay_placeholder",
            "price_gap_bps": _first_numeric(
                work,
                ["entry_gap_bps", "adverse_signal_gap_bps", "price_gap_bps"],
                default=0.0,
            ).fillna(0.0).clip(lower=0.0),
            "expected_friction_bps": _first_numeric(
                work,
                ["expected_total_entry_friction_bps", "expected_friction_bps"],
                default=0.0,
            ).fillna(0.0),
            "liquidity_capacity_weight": 1.0,
            "market_mode": "perps",
        }
    )
    return out.dropna(subset=["timestamp", "symbol", "strategy_id", "normalized_rank_score"])


def _truthy(value: Any) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "1.0", "true", "yes", "y"}


def _persisted_auction_state_decisions(
    ledger: pd.DataFrame,
    params: Any,
) -> pd.DataFrame | None:
    """Replay a live auction from persisted pre-decision state.

    The generic historical replay fits an EV-priority curve from realized
    outcomes. A live decision ledger has no realized outcomes yet, so fitting
    that curve to placeholder returns is both unnecessary and numerically
    unstable. Current live ledgers persist the exact auction order and the
    portfolio state available before each candidate. Recompute the dynamic
    threshold from those causal inputs, then apply observable source-quality
    gates and capacity in auction order.
    """
    required = {
        "auction_rank_number",
        "portfolio_gate_rank_score",
        "portfolio_gate_initial_threshold",
        "portfolio_state_snapshot_json",
    }
    if not required.issubset(ledger.columns):
        return None
    auction_rank = pd.to_numeric(ledger["auction_rank_number"], errors="coerce")
    auction = ledger.loc[auction_rank.notna()].copy()
    if auction.empty:
        return None
    auction["_auction_group_ts"] = pd.to_datetime(
        auction.get("signal_bar_ts", auction["timestamp"]),
        utc=True,
        errors="coerce",
    )

    records: list[dict[str, Any]] = []
    for timestamp, group in auction.groupby(
        "_auction_group_ts", sort=True, dropna=False
    ):
        group = group.assign(
            _auction_rank=pd.to_numeric(
                group["auction_rank_number"], errors="coerce"
            )
        ).sort_values(["_auction_rank", "_ledger_row_id"], kind="stable")
        entries_this_bar = 0
        for _, row in group.iterrows():
            snapshot = _json_mapping(row.get("portfolio_state_snapshot_json"))
            capacity = snapshot.get("capacity") if isinstance(snapshot, Mapping) else None
            if not isinstance(capacity, Mapping):
                return None
            try:
                open_positions = int(capacity.get("open_positions"))
                open_notional = float(capacity.get("open_notional"))
                wallet = float(
                    capacity.get(
                        "wallet_value",
                        capacity.get("total_assets_quote"),
                    )
                )
                allocation_share_raw = capacity.get(
                    "wallet_investment_utilization"
                )
                allocation_share = (
                    float(allocation_share_raw)
                    if allocation_share_raw not in (None, "")
                    else float(open_notional) / float(wallet)
                )
            except (TypeError, ValueError):
                return None
            if not (
                np.isfinite(open_notional)
                and np.isfinite(wallet)
                and wallet > 0.0
                and np.isfinite(allocation_share)
            ):
                return None

            rank_score = float(
                pd.to_numeric(
                    pd.Series([row.get("portfolio_gate_rank_score")]),
                    errors="coerce",
                ).iloc[0]
            )
            initial_threshold = float(
                pd.to_numeric(
                    pd.Series([row.get("portfolio_gate_initial_threshold")]),
                    errors="coerce",
                ).iloc[0]
            )
            entry_cap = pd.to_numeric(
                pd.Series([row.get("auction_entry_cap")]), errors="coerce"
            ).iloc[0]
            capacity_already_full = (
                np.isfinite(entry_cap) and entries_this_bar >= int(entry_cap)
            ) or open_positions >= int(params.max_concurrent_positions)
            raw_signal_unreliable = _truthy(
                row.get("raw_signal_close_unreliable")
            )
            if np.isfinite(rank_score) and np.isfinite(initial_threshold):
                dynamic_threshold = dynamic_threshold_for_count(
                    max(
                        float(initial_threshold),
                        float(params.global_threshold_floor),
                    ),
                    int(open_positions),
                    params,
                    allocation_share=float(allocation_share),
                )
            elif capacity_already_full or raw_signal_unreliable:
                # A diagnostics-only auction can persist candidates after the
                # global entry cap has already fallen to zero, or after a raw
                # data-quality guard has failed. Their rank and threshold
                # fields are intentionally absent because no rank evaluation
                # occurred. The preceding deterministic rejection is enough
                # to replay the row exactly; do not abandon the entire
                # stateful audit for a synthetic generic replay.
                dynamic_threshold = np.nan
            else:
                return None
            logged_threshold = pd.to_numeric(
                pd.Series([row.get("portfolio_gate_final_threshold")]),
                errors="coerce",
            ).iloc[0]
            threshold_delta = (
                float(dynamic_threshold - logged_threshold)
                if np.isfinite(logged_threshold)
                else np.nan
            )

            accepted = False
            if raw_signal_unreliable:
                detail = str(row.get("raw_signal_close_unreliable_reason") or "")
                reason = "unreliable_raw_signal_close"
                if detail and detail.lower() not in {"nan", "none"}:
                    reason = f"{reason}:{detail}"
            elif capacity_already_full:
                reason = "auction_entry_cap_reached"
            elif rank_score < dynamic_threshold + float(
                params.threshold_viability_margin
            ):
                reason = "below_dynamic_threshold"
            elif open_positions >= int(params.max_concurrent_positions):
                reason = "max_concurrent_positions_reached"
            elif entries_this_bar >= int(params.max_new_entries_per_bar):
                reason = "max_new_entries_per_bar_reached"
            else:
                accepted = True
                entries_this_bar += 1
                reason = "accepted"

            records.append(
                {
                    "_ledger_row_id": row["_ledger_row_id"],
                    "timestamp": timestamp,
                    "symbol": str(row["symbol"]),
                    "side": str(row["side"]),
                    "strategy_id": str(row["strategy_id"]),
                    "accepted": bool(accepted),
                    "rejection_reason": reason,
                    "dynamic_threshold": float(dynamic_threshold),
                    "portfolio_priority": pd.to_numeric(
                        pd.Series([row.get("portfolio_priority")]),
                        errors="coerce",
                    ).iloc[0],
                    "position_size": pd.to_numeric(
                        pd.Series(
                            [
                                row.get(
                                    "position_size_quote",
                                    row.get("position_size"),
                                )
                            ]
                        ),
                        errors="coerce",
                    ).iloc[0],
                    "open_positions_before": int(open_positions),
                    "open_positions_after": int(open_positions + int(accepted)),
                    "replay_logged_threshold": logged_threshold,
                    "replay_threshold_delta": threshold_delta,
                }
            )
    return pd.DataFrame.from_records(records)


def _live_traded(row: pd.Series) -> bool:
    if "was_traded" in row and pd.notna(row["was_traded"]):
        try:
            return bool(row["was_traded"])
        except Exception:
            pass
    return str(row.get("portfolio_decision", "")).lower() in {"trade", "traded", "accepted"}


def _bool_series(values: pd.Series) -> pd.Series:
    def _one(value: Any) -> Any:
        if pd.isna(value):
            return np.nan
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"true", "1", "1.0", "yes", "y"}:
            return True
        if text in {"false", "0", "0.0", "no", "n"}:
            return False
        return np.nan

    return values.map(_one)


def _explanation(row: pd.Series) -> str:
    if bool(row.get("live_traded", False)):
        return "live_traded"
    reason = str(
        row.get("portfolio_reject_reason")
        or row.get("liquidity_reject_reason")
        or row.get("portfolio_decision")
        or ""
    )
    if "rank_below" in reason or "below_dynamic_threshold" in reason:
        return "rank_threshold"
    if "stale" in reason:
        return "live_stale_signal_or_data_gate"
    if "spread" in reason or "slippage" in reason or "friction" in reason:
        return "live_spread_slippage_gate"
    if "min_notional" in reason or "position_size" in reason or "wallet" in reason:
        return "live_sizing_or_wallet_gate"
    if "missing_policy_rank" in reason or "rank_reference" in reason:
        return "rank_reference_unavailable"
    if reason:
        return f"live_reject:{reason}"
    return "unexplained_live_not_traded"


def _add_direct_gate_reconciliation(merged: pd.DataFrame) -> pd.DataFrame:
    out = merged.copy()
    rank = _first_numeric(
        out,
        ["threshold_rank_score", "auction_rank_pct", "normalized_rank_score", "policy_rank_pct"],
    )
    threshold = _first_numeric(out, ["final_threshold", "effective_threshold", "initial_rank_threshold"])
    direct = rank >= threshold
    if "passed_rank_gate" in out.columns:
        passed = _bool_series(out["passed_rank_gate"])
        direct = direct.where(passed.isna(), passed.astype("boolean"))
    out["direct_rank_gate_would_open"] = direct.fillna(False).astype(bool)
    out["direct_rank_gate_rank"] = rank
    out["direct_rank_gate_threshold"] = threshold
    out["direct_rank_gate_matches_live_trade"] = (
        out["direct_rank_gate_would_open"].astype(bool)
        == out["live_traded"].astype(bool)
    )
    out["direct_rank_gate_gap_explanation"] = out.apply(
        lambda row: "match"
        if bool(row["direct_rank_gate_matches_live_trade"])
        else _explanation(row),
        axis=1,
    )
    return out


def build_live_decision_replay_reconciliation(
    prediction_ledger: pd.DataFrame,
    *,
    portfolio_policy_config_path: str | Path,
    initial_wallet: float = 10_000.0,
    dedupe_latest: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ledger = _normalise_ledger(prediction_ledger)
    if dedupe_latest:
        ledger = _dedupe_latest_decision_rows(ledger)
    candidates = _build_live_candidate_table(ledger)
    params = load_portfolio_policy_params(portfolio_policy_config_path)
    decisions = _persisted_auction_state_decisions(ledger, params)
    replay_mode = "persisted_auction_state"
    if decisions is None:
        replay_mode = "generic_candidate_replay"
        generic_params = replace(
            params,
            global_threshold_floor=0.0,
            occupancy_threshold_alpha=0.0,
            threshold_viability_margin=0.0,
        )
        decisions, _, _ = replay_candidates(
            candidates,
            generic_params,
            mode="global_auction",
            initial_wallet=float(initial_wallet),
            market_mode="perps",
        )
    if decisions is None or decisions.empty:
        merged = ledger.copy()
        merged["live_traded"] = merged.apply(_live_traded, axis=1)
        merged["replay_accepted"] = False
        merged["decision_match"] = merged["live_traded"] == merged["replay_accepted"]
        merged["replay_live_gap_class"] = np.where(
            merged["decision_match"],
            "match",
            "live_accept_replay_reject",
        )
        merged["replay_live_gap_explanation"] = merged.apply(_explanation, axis=1)
        merged = _add_direct_gate_reconciliation(merged)
        summary = {
            "rows": int(len(merged)),
            "replay_mode": replay_mode,
            "candidate_rows": int(len(candidates)),
            "replay_rows": 0,
            "live_traded": int(merged["live_traded"].sum()),
            "replay_accepted": 0,
            "decision_matches": int(merged["decision_match"].sum()),
            "decision_mismatches": int((~merged["decision_match"]).sum()),
            "gap_classes": merged["replay_live_gap_class"].value_counts(dropna=False).to_dict(),
            "gap_explanations": merged["replay_live_gap_explanation"].value_counts(dropna=False).to_dict(),
            "replay_rejection_reasons": {},
            "live_portfolio_reasons": merged.get(
                "portfolio_reject_reason", pd.Series(dtype=object)
            ).value_counts(dropna=False).to_dict(),
            "direct_rank_gate_would_open": int(merged["direct_rank_gate_would_open"].sum()),
            "direct_rank_gate_matches": int(
                merged["direct_rank_gate_matches_live_trade"].sum()
            ),
            "direct_rank_gate_mismatches": int(
                (~merged["direct_rank_gate_matches_live_trade"]).sum()
            ),
            "direct_rank_gate_gap_explanations": merged[
                "direct_rank_gate_gap_explanation"
            ].value_counts(dropna=False).to_dict(),
            "exact_portfolio_state_replayable_rows": 0,
            "exact_portfolio_state_replayable_traded_rows": 0,
            "exact_portfolio_state_replayable_note": (
                "Decision replay returned no accepted/rejected rows; "
                "candidate/rank parity is still reported separately."
            ),
        }
        return merged, summary
    key = ["timestamp", "symbol", "side", "strategy_id"]
    decision_cols = key + [
        "normalized_rank_score",
        "base_threshold",
        "accepted",
        "rejection_reason",
        "dynamic_threshold",
        "portfolio_priority",
        "position_size",
        "open_positions_before",
        "open_positions_after",
    ]
    replay_join = decisions.copy()
    replay_join["timestamp"] = pd.to_datetime(
        replay_join["timestamp"], utc=True, errors="coerce"
    )
    candidate_map = candidates[key + ["_ledger_row_id"]].copy()
    candidate_map["timestamp"] = pd.to_datetime(
        candidate_map["timestamp"], utc=True, errors="coerce"
    )
    for frame in (replay_join, candidate_map):
        frame["symbol"] = frame["symbol"].astype(str)
        frame["side"] = frame["side"].astype(str)
        frame["strategy_id"] = frame["strategy_id"].astype(str)
    candidate_map = candidate_map.drop_duplicates(key, keep="last")
    if "_ledger_row_id" not in replay_join.columns:
        replay_join = replay_join.merge(
            candidate_map[key + ["_ledger_row_id"]],
            on=key,
            how="left",
        )
    optional_replay_cols = [
        name
        for name in ("replay_logged_threshold", "replay_threshold_delta")
        if name in replay_join.columns
    ]
    merged = ledger.merge(
        replay_join[
            [
                "_ledger_row_id",
                "accepted",
                "rejection_reason",
                "dynamic_threshold",
                "portfolio_priority",
                "position_size",
                "open_positions_before",
                "open_positions_after",
                *optional_replay_cols,
            ]
        ],
        on="_ledger_row_id",
        how="left",
        suffixes=("", "_replay"),
    )
    merged["live_traded"] = merged.apply(_live_traded, axis=1)
    state_required = (
        ("portfolio_state_snapshot_json", "open_positions_before_json", "active_positions_before_json"),
        ("portfolio_state_snapshot_hash", "portfolio_state_hash"),
        ("wallet_before", "wallet_value"),
        ("open_positions_before", "open_positions_before_count"),
        ("cooldowns_before_json", "recent_losing_trade_cooldown_state_json"),
        ("portfolio_priority",),
    )
    replayable = pd.Series(True, index=merged.index, dtype=bool)
    for alternatives in state_required:
        replayable &= _alternative_present(merged, list(alternatives))
    merged["exact_portfolio_state_replayable"] = replayable
    merged["replay_accepted"] = merged["accepted"].fillna(False).astype(bool)
    merged["decision_match"] = merged["live_traded"] == merged["replay_accepted"]
    merged["replay_live_gap_class"] = merged.apply(
        lambda row: "match"
        if bool(row["decision_match"])
        else (
            "replay_accept_live_reject"
            if bool(row["replay_accepted"])
            else "live_accept_replay_reject"
        ),
        axis=1,
    )
    merged["replay_live_gap_explanation"] = merged.apply(_explanation, axis=1)
    merged = _add_direct_gate_reconciliation(merged)
    summary = {
        "rows": int(len(merged)),
        "replay_mode": replay_mode,
        "candidate_rows": int(len(candidates)),
        "replay_rows": int(len(decisions)),
        "live_traded": int(merged["live_traded"].sum()),
        "replay_accepted": int(merged["replay_accepted"].sum()),
        "decision_matches": int(merged["decision_match"].sum()),
        "decision_mismatches": int((~merged["decision_match"]).sum()),
        "gap_classes": merged["replay_live_gap_class"].value_counts(dropna=False).to_dict(),
        "gap_explanations": merged["replay_live_gap_explanation"].value_counts(dropna=False).to_dict(),
        "replay_rejection_reasons": merged["rejection_reason"].value_counts(dropna=False).to_dict(),
        "live_portfolio_reasons": merged.get(
            "portfolio_reject_reason", pd.Series(dtype=object)
        ).value_counts(dropna=False).to_dict(),
        "direct_rank_gate_would_open": int(merged["direct_rank_gate_would_open"].sum()),
        "direct_rank_gate_matches": int(
            merged["direct_rank_gate_matches_live_trade"].sum()
        ),
        "direct_rank_gate_mismatches": int(
            (~merged["direct_rank_gate_matches_live_trade"]).sum()
        ),
        "direct_rank_gate_gap_explanations": merged[
            "direct_rank_gate_gap_explanation"
        ].value_counts(dropna=False).to_dict(),
        "exact_portfolio_state_replayable_rows": int(
            merged["exact_portfolio_state_replayable"].sum()
        ),
        "exact_portfolio_state_replayable_traded_rows": int(
            (
                merged["exact_portfolio_state_replayable"].astype(bool)
                & merged["live_traded"].astype(bool)
            ).sum()
        ),
        "exact_portfolio_state_replayable_note": (
            "Rows without persisted portfolio state are candidate/rank replay rows, "
            "not exact stateful portfolio replay proof."
        ),
        "recomputed_threshold_max_abs_delta": (
            float(
                pd.to_numeric(
                    merged.get("replay_threshold_delta", pd.Series(dtype=float)),
                    errors="coerce",
                ).abs().max()
            )
            if "replay_threshold_delta" in merged.columns
            and pd.to_numeric(
                merged["replay_threshold_delta"], errors="coerce"
            ).notna().any()
            else np.nan
        ),
    }
    return merged, summary


def build_prediction_rank_parity_reconciliation(
    prediction_ledger: pd.DataFrame,
    *,
    data_root: str | Path,
    run_id: str,
    max_rows: int = 500,
    dedupe_latest: bool = True,
    tolerance: float = 1e-6,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Re-score logged live candidates and compare predictions/ranks 1:1.

    The ledger contains the selected model feature values used by live
    inference. Replaying those values through the deployed model bundle catches
    score, calibration, and rank-normalisation drift for both traded and
    rejected candidates without rebuilding the full market feature matrix.
    """
    ledger = _normalise_ledger(prediction_ledger)
    if dedupe_latest:
        ledger = _dedupe_latest_decision_rows(ledger)
    if max_rows > 0 and len(ledger) > int(max_rows):
        ledger = ledger.sort_values("timestamp").tail(int(max_rows)).copy()

    if ledger.empty:
        return pd.DataFrame(), {
            "rows": 0,
            "attempted_rows": 0,
            "success_rows": 0,
            "failed_rows": 0,
            "reason": "empty_ledger",
        }

    data_root_s = str(data_root)
    run_id_s = str(run_id)
    rows: list[dict[str, Any]] = []
    try:
        state = load_full_state(run_id_s, data_root_s)
        orchestrator = ModelOrchestrator(
            state,
            {
                "inference_model_timing_enabled": False,
                "preserve_logged_meta_model_derived_features": True,
                "strict_feature_parity_neutral_fill_nonfinite": True,
            },
        )
        calibration_data = load_calibration_curves(data_root_s, run_id_s)
        rank_store = PolicyRankReferenceStore(data_root=data_root_s, run_id=run_id_s)
        side_expert, canonical_postprocessor = _load_active_post_meta_chain(
            data_root_s, run_id_s
        )
        setup_error = ""
    except Exception as exc:
        orchestrator = None
        calibration_data = {}
        rank_store = None
        side_expert = None
        canonical_postprocessor = None
        setup_error = str(exc)

    for _, row in ledger.iterrows():
        symbol = str(row.get("symbol") or "")
        side = _normalise_side(row.get("side"), row.get("strategy_id"))
        strategy_id = str(row.get("strategy_id") or "")
        stored_calibrated = _safe_float_series_value(row.get("calibrated_score"))
        stored_base = _safe_float_series_value(row.get("base_pred"))
        stored_meta = _safe_float_series_value(row.get("meta_pred"))
        stored_policy_rank = _safe_float_series_value(row.get("policy_rank_pct"))
        stored_auction_rank = _safe_float_series_value(row.get("auction_rank_pct"))
        out = {
            "_ledger_row_id": row.get("_ledger_row_id"),
            "timestamp": row.get("timestamp"),
            "symbol": symbol,
            "side": side,
            "strategy_id": strategy_id,
            "portfolio_decision": row.get("portfolio_decision"),
            "was_traded": row.get("was_traded"),
            "stored_base_pred": stored_base,
            "stored_meta_pred": stored_meta,
            "stored_calibrated_score": stored_calibrated,
            "stored_v9_tail95_predecessor_rank": _safe_float_series_value(
                row.get("v9_tail95_predecessor_rank")
            ),
            "stored_score_regime_calibrated": _safe_float_series_value(
                row.get("score_regime_calibrated")
            ),
            "stored_expected_net_ev_after_1pct": _safe_float_series_value(
                row.get("expected_net_ev_after_1pct")
            ),
            "stored_expected_ev_rank_score": _safe_float_series_value(
                row.get("expected_ev_rank_score")
            ),
            "stored_policy_rank_pct": stored_policy_rank,
            "stored_auction_rank_pct": stored_auction_rank,
            "replay_status": "not_run",
            "replay_error": "",
        }
        for admission_field in ACTIVE_ADMISSION_FIELDS:
            out[f"stored_{admission_field}"] = row.get(admission_field)
        if orchestrator is None or rank_store is None:
            out["replay_status"] = "setup_failed"
            out["replay_error"] = setup_error
            rows.append(out)
            continue
        base_features = _feature_frame_from_json(
            row.get("base_model_feature_values_json"),
            symbol=symbol,
        )
        meta_features = _feature_frame_from_json(
            row.get("meta_model_feature_values_json"),
            symbol=symbol,
        )
        if base_features.empty or meta_features.empty:
            out["replay_status"] = "missing_feature_snapshot"
            rows.append(out)
            continue
        try:
            base_pred = orchestrator.predict_alpha(
                base_features,
                side=side,
                kind=strategy_id,
            )
            replay_meta, meta_replay_source = _logged_meta_prediction(
                orchestrator,
                meta_features,
                side=side,
                strategy_id=strategy_id,
                meta_model_key=row.get("meta_model_key"),
            )
            if not np.isfinite(replay_meta):
                meta_pred = orchestrator.predict_meta(
                    meta_features,
                    side=side,
                    kind=strategy_id,
                )
                replay_meta = (
                    float(meta_pred.iloc[0])
                    if isinstance(meta_pred, pd.Series) and not meta_pred.empty
                    else np.nan
                )
                meta_replay_source = f"materialized_fallback:{meta_replay_source}"
            replay_base = (
                float(base_pred.iloc[0])
                if isinstance(base_pred, pd.Series) and not base_pred.empty
                else np.nan
            )
            active_chain = _replay_active_post_meta_row(
                row,
                side=side,
                symbol=symbol,
                replay_base=replay_base,
                side_expert=side_expert,
                postprocessor=canonical_postprocessor,
            )
            if "replay_meta_pred" in active_chain:
                replay_meta = float(active_chain["replay_meta_pred"])
                meta_replay_source = str(active_chain.get("replay_meta_source") or "")
            replay_calibrated, _ = calibrated_score_and_threshold(
                raw_score=replay_meta,
                strategy_id=strategy_id,
                calibration_data=calibration_data,
                default_threshold=1.0,
            )
            policy_rank = rank_store.lookup(
                strategy_id=strategy_id,
                side=side,
                calibrated_score=replay_calibrated,
            )
            auction_rank = rank_store.lookup_auction(
                calibrated_score=replay_calibrated,
            )
            out.update(
                {
                    "replay_base_pred": replay_base,
                    "replay_meta_pred": replay_meta,
                    "replay_meta_source": meta_replay_source,
                    "replay_calibrated_score": float(replay_calibrated),
                    "replay_policy_rank_pct": float(policy_rank.policy_rank_pct),
                    "replay_policy_rank_reference_n": int(policy_rank.n_rows),
                    "replay_policy_rank_reference_source": policy_rank.source,
                    "replay_auction_rank_pct": float(auction_rank.policy_rank_pct),
                    "replay_auction_rank_reference_n": int(auction_rank.n_rows),
                    "replay_auction_rank_reference_source": auction_rank.source,
                    "replay_status": "ok",
                }
            )
            out.update(active_chain)
        except Exception as exc:
            out["replay_status"] = "replay_failed"
            out["replay_error"] = str(exc)
        rows.append(out)

    report = pd.DataFrame(rows)
    report, admission_summary = _replay_active_threshold_policy(
        report,
        ledger,
        policy_path=_active_threshold_policy_path(
            data_root=data_root_s,
            run_id=run_id_s,
        ),
        tolerance=float(tolerance),
    )
    compared_fields = (
        *ACTIVE_CHAIN_FIELDS,
        "calibrated_score",
        "policy_rank_pct",
        "auction_rank_pct",
    )
    for name in compared_fields:
        stored = f"stored_{name}"
        replay = f"replay_{name}"
        if stored in report.columns and replay in report.columns:
            report[f"{name}_abs_delta"] = _abs_delta(report[stored], report[replay])
            report[f"{name}_matches"] = report[f"{name}_abs_delta"] <= float(tolerance)

    ok = report.get("replay_status", pd.Series(dtype=object)).astype(str).eq("ok")
    summary: dict[str, Any] = {
        "rows": int(len(report)),
        "attempted_rows": int(len(report)),
        "success_rows": int(ok.sum()),
        "failed_rows": int((~ok).sum()),
        "max_rows": int(max_rows),
        "tolerance": float(tolerance),
        "status_counts": report.get("replay_status", pd.Series(dtype=object))
        .astype(str)
        .value_counts(dropna=False)
        .to_dict(),
    }
    for name in compared_fields:
        delta_col = f"{name}_abs_delta"
        match_col = f"{name}_matches"
        if delta_col not in report.columns:
            continue
        deltas = pd.to_numeric(report.loc[ok, delta_col], errors="coerce")
        summary[name] = {
            "n": int(deltas.notna().sum()),
            "max_abs_delta": float(deltas.max()) if deltas.notna().any() else np.nan,
            "mean_abs_delta": float(deltas.mean()) if deltas.notna().any() else np.nan,
            "mismatch_rows": int((~report.loc[ok, match_col].fillna(False).astype(bool)).sum())
            if match_col in report.columns
            else 0,
        }
    active_required = [
        name
        for name in ACTIVE_CHAIN_FIELDS
        if f"{name}_matches" in report.columns
    ]
    summary["active_chain_fields"] = active_required
    summary["active_chain_mismatch_rows"] = int(
        sum(
            (~report.loc[ok, f"{name}_matches"].fillna(False).astype(bool)).sum()
            for name in active_required
        )
    )
    summary["active_chain_pass"] = bool(
        active_required
        and len(active_required) == len(ACTIVE_CHAIN_FIELDS)
        and summary["active_chain_mismatch_rows"] == 0
    )
    summary["active_admission"] = admission_summary
    summary["active_admission_pass"] = bool(admission_summary.get("pass", False))
    summary["active_decision_chain_pass"] = bool(
        summary["active_chain_pass"] and summary["active_admission_pass"]
    )
    if "strategy_id" in report.columns:
        by_strategy: dict[str, Any] = {}
        for strategy_id, group in report.groupby("strategy_id", dropna=False):
            group_ok = group.get("replay_status", pd.Series(dtype=object)).astype(str).eq("ok")
            by_strategy[str(strategy_id)] = {
                "rows": int(len(group)),
                "success_rows": int(group_ok.sum()),
                "meta_pred_max_abs_delta": float(
                    pd.to_numeric(
                        group.loc[group_ok, "meta_pred_abs_delta"],
                        errors="coerce",
                    ).max()
                )
                if "meta_pred_abs_delta" in group.columns
                and pd.to_numeric(group.loc[group_ok, "meta_pred_abs_delta"], errors="coerce").notna().any()
                else np.nan,
                "auction_rank_max_abs_delta": float(
                    pd.to_numeric(
                        group.loc[group_ok, "auction_rank_pct_abs_delta"],
                        errors="coerce",
                    ).max()
                )
                if "auction_rank_pct_abs_delta" in group.columns
                and pd.to_numeric(group.loc[group_ok, "auction_rank_pct_abs_delta"], errors="coerce").notna().any()
                else np.nan,
            }
        summary["by_strategy"] = by_strategy
    return report, summary


def build_shadow_trade_reconciliation(
    trade_log: pd.DataFrame,
    *,
    tolerance_bps: float = 50.0,
    run_id: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if trade_log is None or trade_log.empty:
        return pd.DataFrame(), {
            "rows": 0,
            "shadow_rows": 0,
            "closed_shadow_rows": 0,
            "open_shadow_rows": 0,
            "exit_execution_parity_status": "pending_no_rows",
            "reason": "empty_trade_log",
        }
    work = trade_log.copy()
    side = work.get("side", pd.Series("", index=work.index)).astype(str).str.lower()
    action = work.get("action", pd.Series("", index=work.index)).astype(str).str.lower()
    lifecycle = work.get("lifecycle_event", pd.Series("", index=work.index)).astype(str).str.lower()
    status = work.get("status", pd.Series("", index=work.index)).astype(str).str.lower()
    has_shadow = _alternative_present(
        work,
        ["shadow_policy_schema", "shadow_entry_gap_bps", "shadow_latest_stop_price", "simple_policy_shadow"],
    )
    closed = (
        action.eq("exit")
        | lifecycle.str.contains("exit|close", regex=True, na=False)
        | status.isin({"closed", "filled", "completed"})
    )
    scoped = work.loc[has_shadow].copy()
    if scoped.empty:
        return pd.DataFrame(), {
            "rows": int(len(work)),
            "shadow_rows": 0,
            "closed_shadow_rows": 0,
            "open_shadow_rows": 0,
            "exit_execution_parity_status": "pending_no_shadow_rows",
            "reason": "no_shadow_rows",
        }
    scoped_side = scoped.get("side", pd.Series("", index=scoped.index)).astype(str).str.lower()
    live_exit = _first_numeric(
        scoped,
        ["realized_exit_price", "actual_exit_price", "exit_price"],
    )
    shadow_exit = _first_numeric(scoped, ["shadow_exit_price"])
    shadow_theoretical_exit = _first_numeric(
        scoped,
        ["shadow_theoretical_exit_price", "shadow_stop_trigger_price"],
    )
    live_stop = _first_numeric(scoped, ["final_placed_stop", "stop_price", "shadow_live_stop_price"])
    shadow_stop = _first_numeric(scoped, ["shadow_latest_stop_price", "shadow_initial_stop_price"])
    entry_gap = _first_numeric(scoped, ["shadow_entry_gap_bps", "entry_gap_bps"])
    stop_gap = _first_numeric(scoped, ["shadow_stop_gap_bps"])
    trigger_gap = _first_numeric(scoped, ["shadow_trigger_vs_live_exit_gap_bps"])
    side_sign = np.where(scoped_side.eq("short"), -1.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        exit_gap_bps = side_sign * (
            live_exit.to_numpy(dtype=float)
            / np.maximum(shadow_exit.to_numpy(dtype=float), 1e-12)
            - 1.0
        ) * 10000.0
        stop_gap_fallback = side_sign * (
            shadow_stop.to_numpy(dtype=float)
            / np.maximum(live_stop.to_numpy(dtype=float), 1e-12)
            - 1.0
        ) * 10000.0
        trigger_gap_fallback = side_sign * (
            live_exit.to_numpy(dtype=float)
            / np.maximum(shadow_theoretical_exit.to_numpy(dtype=float), 1e-12)
            - 1.0
        ) * 10000.0
    stop_gap = stop_gap.where(stop_gap.notna(), pd.Series(stop_gap_fallback, index=scoped.index))
    trigger_gap = trigger_gap.where(
        trigger_gap.notna(),
        pd.Series(trigger_gap_fallback, index=scoped.index),
    )
    out = pd.DataFrame(
        {
            "timestamp": scoped.get("timestamp"),
            "trade_id": scoped.get("trade_id"),
            "position_id": scoped.get("position_id"),
            "symbol": scoped.get("symbol"),
            "side": scoped_side,
            "strategy_id": scoped.get("strategy_id"),
            "lifecycle_event": scoped.get("lifecycle_event"),
            "status": scoped.get("status"),
            "shadow_status": scoped.get("shadow_status"),
            "entry_gap_bps": entry_gap,
            "live_stop_price": live_stop,
            "shadow_stop_price": shadow_stop,
            "shadow_vs_live_stop_gap_bps": stop_gap,
            "live_exit_price": live_exit,
            "shadow_exit_price": shadow_exit,
            "shadow_exit_price_source": scoped.get(
                "shadow_exit_price_source",
                pd.Series("", index=scoped.index),
            ),
            "shadow_theoretical_exit_price": shadow_theoretical_exit,
            "live_vs_shadow_exit_gap_bps": exit_gap_bps,
            "shadow_trigger_vs_live_exit_gap_bps": trigger_gap,
            "shadow_exit_reason": scoped.get("shadow_exit_reason"),
            "shadow_exit_return": _first_numeric(scoped, ["shadow_exit_return"]),
        }
    )
    out["entry_gap_within_tolerance"] = out["entry_gap_bps"].abs() <= float(tolerance_bps)
    out["stop_gap_within_tolerance"] = out["shadow_vs_live_stop_gap_bps"].abs() <= float(tolerance_bps)
    out["exit_gap_within_tolerance"] = out["live_vs_shadow_exit_gap_bps"].abs() <= float(tolerance_bps)
    scoped_closed = closed.reindex(scoped.index, fill_value=False)
    closed_out = out.loc[scoped_closed].copy()
    open_shadow_rows = int((~scoped_closed).sum())
    exit_gap_mismatch_rows = (
        int((~closed_out["exit_gap_within_tolerance"].fillna(False)).sum())
        if not closed_out.empty
        else 0
    )
    if closed_out.empty:
        parity_status = (
            "pending_open_positions" if open_shadow_rows > 0 else "pending_no_closed_rows"
        )
    elif exit_gap_mismatch_rows == 0:
        parity_status = "pass"
    else:
        parity_status = "fail"
    summary = {
        "rows": int(len(work)),
        "shadow_rows": int(len(out)),
        "closed_shadow_rows": int(len(closed_out)),
        "open_shadow_rows": open_shadow_rows,
        "tolerance_bps": float(tolerance_bps),
        "entry_gap_bps": _numeric_summary(out["entry_gap_bps"]),
        "shadow_vs_live_stop_gap_bps": _numeric_summary(out["shadow_vs_live_stop_gap_bps"]),
        "live_vs_shadow_exit_gap_bps": _numeric_summary(closed_out["live_vs_shadow_exit_gap_bps"]),
        "shadow_trigger_vs_live_exit_gap_bps": _numeric_summary(
            closed_out["shadow_trigger_vs_live_exit_gap_bps"]
        ),
        "entry_gap_mismatch_rows": int((~out["entry_gap_within_tolerance"].fillna(False)).sum()),
        "stop_gap_mismatch_rows": int((~out["stop_gap_within_tolerance"].fillna(False)).sum()),
        "exit_gap_mismatch_rows": exit_gap_mismatch_rows,
        "exit_execution_parity_status": parity_status,
        "shadow_status_counts": out.get("shadow_status", pd.Series(dtype=object)).value_counts(dropna=False).to_dict(),
    }
    if run_id:
        run_text = str(run_id)
        scoped_mask = pd.Series(False, index=out.index, dtype=bool)
        for col in ("position_id", "trade_id"):
            if col in out.columns:
                scoped_mask |= out[col].astype(str).str.contains(run_text, regex=False, na=False)
        scoped_out = out.loc[scoped_mask].copy()
        scoped_closed_mask = scoped_closed.reindex(out.index, fill_value=False).loc[scoped_out.index]
        scoped_closed_out = scoped_out.loc[scoped_closed_mask].copy()
        scoped_exit_mismatch_rows = (
            int((~scoped_closed_out["exit_gap_within_tolerance"].fillna(False)).sum())
            if not scoped_closed_out.empty
            else 0
        )
        if scoped_closed_out.empty:
            scoped_parity_status = (
                "pending_open_positions"
                if int((~scoped_closed_mask).sum()) > 0
                else "pending_no_closed_rows"
            )
        elif scoped_exit_mismatch_rows == 0:
            scoped_parity_status = "pass"
        else:
            scoped_parity_status = "fail"
        summary["current_run"] = {
            "run_id": run_text,
            "shadow_rows": int(len(scoped_out)),
            "closed_shadow_rows": int(len(scoped_closed_out)),
            "open_shadow_rows": int((~scoped_closed_mask).sum()),
            "live_vs_shadow_exit_gap_bps": _numeric_summary(
                scoped_closed_out["live_vs_shadow_exit_gap_bps"]
            ),
            "shadow_trigger_vs_live_exit_gap_bps": _numeric_summary(
                scoped_closed_out["shadow_trigger_vs_live_exit_gap_bps"]
            ),
            "exit_gap_mismatch_rows": scoped_exit_mismatch_rows,
            "exit_execution_parity_status": scoped_parity_status,
            "shadow_status_counts": scoped_out.get(
                "shadow_status", pd.Series(dtype=object)
            ).value_counts(dropna=False).to_dict(),
        }
    return out, summary


def _safe_float_series_value(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def execution_parity_audit_status(
    shadow_trade_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify the current-run exit reconciliation without conflating no evidence with pass."""
    current_run = shadow_trade_summary.get("current_run")
    scope = current_run if isinstance(current_run, Mapping) else shadow_trade_summary
    raw_status = str(scope.get("exit_execution_parity_status") or "")
    closed_shadow_rows = int(scope.get("closed_shadow_rows", 0) or 0)
    if raw_status == "pass" and closed_shadow_rows > 0:
        status = "pass"
    elif raw_status == "fail":
        status = "fail"
    else:
        status = "pending"
    return {
        "status": status,
        "reason": raw_status or "missing_exit_execution_parity_status",
        "scope": "current_run" if isinstance(current_run, Mapping) else "all_rows",
        "shadow_rows": int(scope.get("shadow_rows", 0) or 0),
        "closed_shadow_rows": closed_shadow_rows,
        "open_shadow_rows": int(scope.get("open_shadow_rows", 0) or 0),
        "exit_gap_mismatch_rows": int(scope.get("exit_gap_mismatch_rows", 0) or 0),
    }


def _render_markdown(
    *,
    spread_summary: dict[str, Any],
    decision_summary: dict[str, Any],
    field_summary: dict[str, Any],
    prediction_summary: Optional[dict[str, Any]] = None,
    shadow_trade_summary: Optional[dict[str, Any]] = None,
) -> str:
    prediction_summary = prediction_summary or {}
    shadow_trade_summary = shadow_trade_summary or {}
    current_run_summary = shadow_trade_summary.get("current_run") or {}
    current_run_lines = []
    if current_run_summary:
        current_run_lines = [
            "",
            "### Current Run Shadow Execution",
            f"- Run id: `{current_run_summary.get('run_id', '')}`",
            f"- Shadow rows: `{current_run_summary.get('shadow_rows', 0)}`",
            f"- Closed shadow rows: `{current_run_summary.get('closed_shadow_rows', 0)}`",
            f"- Live vs shadow exit gap: `{current_run_summary.get('live_vs_shadow_exit_gap_bps', {})}`",
            f"- Exit gap mismatches: `{current_run_summary.get('exit_gap_mismatch_rows', 0)}`",
            f"- Exit execution parity status: `{current_run_summary.get('exit_execution_parity_status', '')}`",
            f"- Status counts: `{current_run_summary.get('shadow_status_counts', {})}`",
        ]
    return "\n".join(
        [
            "# Execution and Decision Reconciliation",
            "",
            "## Prediction / Rank Parity",
            f"- Replayed rows: `{prediction_summary.get('success_rows', 0)}` / `{prediction_summary.get('attempted_rows', 0)}`",
            f"- Status counts: `{prediction_summary.get('status_counts', {})}`",
            f"- Active model/postprocessor chain pass: `{prediction_summary.get('active_chain_pass', False)}`",
            f"- Active 21-day EV70 admission pass: `{prediction_summary.get('active_admission_pass', False)}`",
            f"- Active decision chain pass: `{prediction_summary.get('active_decision_chain_pass', False)}`",
            f"- Active admission detail: `{prediction_summary.get('active_admission', {})}`",
            f"- Meta prediction delta: `{prediction_summary.get('meta_pred', {})}`",
            f"- Legacy shared-meta calibrated-score diagnostic: `{prediction_summary.get('calibrated_score', {})}`",
            f"- Legacy rank-reference diagnostic: `{prediction_summary.get('policy_rank_pct', {})}`",
            f"- Legacy auction-reference diagnostic: `{prediction_summary.get('auction_rank_pct', {})}`",
            "",
            "## Shadow Execution Realism",
            f"- Shadow rows: `{shadow_trade_summary.get('shadow_rows', 0)}`",
            f"- Closed shadow rows: `{shadow_trade_summary.get('closed_shadow_rows', 0)}`",
            f"- Entry gap: `{shadow_trade_summary.get('entry_gap_bps', {})}`",
            f"- Shadow vs live stop gap: `{shadow_trade_summary.get('shadow_vs_live_stop_gap_bps', {})}`",
            f"- Live vs shadow exit gap: `{shadow_trade_summary.get('live_vs_shadow_exit_gap_bps', {})}`",
            f"- Status counts: `{shadow_trade_summary.get('shadow_status_counts', {})}`",
            *current_run_lines,
            "",
            "## Spread / Slippage",
            f"- Rows: `{spread_summary.get('rows', 0)}`",
            f"- Traded rows: `{spread_summary.get('traded_rows', 0)}`",
            f"- Policy vs live friction delta: `{spread_summary.get('columns', {}).get('policy_vs_live_friction_delta_bps', {})}`",
            f"- Live total entry friction: `{spread_summary.get('columns', {}).get('live_total_entry_friction_bps', {})}`",
            "",
            "## Backtest / Live Open Decision",
            f"- Ledger rows: `{decision_summary.get('rows', 0)}`",
            f"- Live traded: `{decision_summary.get('live_traded', 0)}`",
            f"- Replay accepted: `{decision_summary.get('replay_accepted', 0)}`",
            f"- Decision mismatches: `{decision_summary.get('decision_mismatches', 0)}`",
            f"- Gap classes: `{decision_summary.get('gap_classes', {})}`",
            f"- Gap explanations: `{decision_summary.get('gap_explanations', {})}`",
            f"- Direct rank-gate would open: `{decision_summary.get('direct_rank_gate_would_open', 0)}`",
            f"- Direct rank-gate mismatches: `{decision_summary.get('direct_rank_gate_mismatches', 0)}`",
            f"- Direct rank-gate gap explanations: `{decision_summary.get('direct_rank_gate_gap_explanations', {})}`",
            f"- Exact portfolio-state replayable rows: `{decision_summary.get('exact_portfolio_state_replayable_rows', 0)}`",
            f"- Exact portfolio-state replayable traded rows: `{decision_summary.get('exact_portfolio_state_replayable_traded_rows', 0)}`",
            "",
            "## Replay Field Coverage",
            f"- Ledger rows: `{field_summary.get('ledger_rows', 0)}`",
            f"- Live traded rows: `{field_summary.get('live_traded_rows', 0)}`",
            f"- Exact portfolio-state replayable rows: `{field_summary.get('exact_portfolio_state_replayable_rows', 0)}`",
            f"- Exact portfolio-state replayable rate: `{field_summary.get('exact_portfolio_state_replayable_rate', 0)}`",
            f"- Failed field checks: `{field_summary.get('failed_field_checks', 0)}`",
            f"- Failed traded-field checks: `{field_summary.get('failed_traded_field_checks', 0)}`",
            f"- Critical missing rows: `{field_summary.get('critical_missing_rows', 0)}`",
            f"- Worst missing: `{field_summary.get('worst_missing', [])}`",
            "",
            "Note: decision replay uses live ledger candidates and deployed portfolio-policy gates. "
            "It is a final gate parity audit, not a PnL backtest.",
            "",
        ]
    )


def run_reconciliation(
    *,
    prediction_ledger_path: str | Path,
    portfolio_policy_config_path: str | Path,
    output_dir: str | Path,
    candidate_path: str | Path | None = None,
    trade_log_path: str | Path | None = None,
    data_root: str | Path | None = None,
    run_id: str | None = None,
    prediction_parity_max_rows: int = 500,
    shadow_tolerance_bps: float = 50.0,
    initial_wallet: float = 10_000.0,
    since: str | pd.Timestamp | None = None,
) -> dict[str, Any]:
    ledger = _filter_table_since(_read_table(prediction_ledger_path), since)
    candidates = _filter_table_since(_read_table(candidate_path), since)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spread_rows, spread_summary = build_spread_slippage_reconciliation(
        ledger,
        candidates=candidates,
    )
    field_rows, field_summary = build_ledger_replay_field_coverage(ledger)
    decision_rows, decision_summary = build_live_decision_replay_reconciliation(
        ledger,
        portfolio_policy_config_path=portfolio_policy_config_path,
        initial_wallet=initial_wallet,
    )
    if data_root is not None and run_id:
        prediction_rows, prediction_summary = build_prediction_rank_parity_reconciliation(
            ledger,
            data_root=data_root,
            run_id=str(run_id),
            max_rows=int(prediction_parity_max_rows),
        )
    else:
        prediction_rows = pd.DataFrame()
        prediction_summary = {
            "rows": 0,
            "attempted_rows": 0,
            "success_rows": 0,
            "failed_rows": 0,
            "reason": "data_root_or_run_id_not_provided",
        }
    trade_log = _filter_table_since(_read_table(trade_log_path), since)
    shadow_rows, shadow_summary = build_shadow_trade_reconciliation(
        trade_log,
        tolerance_bps=float(shadow_tolerance_bps),
        run_id=run_id,
    )
    execution_audit = execution_parity_audit_status(shadow_summary)
    spread_rows.to_csv(out_dir / "spread_slippage_reconciliation.csv", index=False)
    field_rows.to_csv(out_dir / "ledger_replay_field_coverage.csv", index=False)
    decision_rows.to_csv(out_dir / "live_decision_replay_reconciliation.csv", index=False)
    prediction_rows.to_csv(out_dir / "prediction_rank_parity_reconciliation.csv", index=False)
    shadow_rows.to_csv(out_dir / "shadow_trade_reconciliation.csv", index=False)
    (out_dir / "spread_slippage_reconciliation.json").write_text(
        json.dumps(_json_safe(spread_summary), indent=2),
        encoding="utf-8",
    )
    (out_dir / "ledger_replay_field_coverage.json").write_text(
        json.dumps(_json_safe(field_summary), indent=2),
        encoding="utf-8",
    )
    (out_dir / "live_decision_replay_reconciliation.json").write_text(
        json.dumps(_json_safe(decision_summary), indent=2),
        encoding="utf-8",
    )
    (out_dir / "prediction_rank_parity_reconciliation.json").write_text(
        json.dumps(_json_safe(prediction_summary), indent=2),
        encoding="utf-8",
    )
    (out_dir / "shadow_trade_reconciliation.json").write_text(
        json.dumps(_json_safe(shadow_summary), indent=2),
        encoding="utf-8",
    )
    (out_dir / "execution_parity_audit_status.json").write_text(
        json.dumps(_json_safe(execution_audit), indent=2),
        encoding="utf-8",
    )
    markdown = _render_markdown(
        spread_summary=spread_summary,
        decision_summary=decision_summary,
        field_summary=field_summary,
        prediction_summary=prediction_summary,
        shadow_trade_summary=shadow_summary,
    )
    (out_dir / "execution_and_decision_reconciliation.md").write_text(
        markdown,
        encoding="utf-8",
    )
    return {
        "spread_slippage": spread_summary,
        "field_coverage": field_summary,
        "decision_replay": decision_summary,
        "prediction_rank_parity": prediction_summary,
        "shadow_trade_reconciliation": shadow_summary,
        "execution_parity_audit": execution_audit,
        "output_dir": str(out_dir),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-ledger", required=True)
    parser.add_argument("--portfolio-policy-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate-path")
    parser.add_argument("--trade-log-path")
    parser.add_argument("--data-root")
    parser.add_argument("--run-id")
    parser.add_argument("--prediction-parity-max-rows", type=int, default=500)
    parser.add_argument("--shadow-tolerance-bps", type=float, default=50.0)
    parser.add_argument("--initial-wallet", type=float, default=10_000.0)
    parser.add_argument(
        "--since",
        help="Audit only rows at or after this UTC timestamp (ISO-8601).",
    )
    args = parser.parse_args(argv)
    result = run_reconciliation(
        prediction_ledger_path=args.prediction_ledger,
        portfolio_policy_config_path=args.portfolio_policy_config,
        output_dir=args.output_dir,
        candidate_path=args.candidate_path,
        trade_log_path=args.trade_log_path,
        data_root=args.data_root,
        run_id=args.run_id,
        prediction_parity_max_rows=args.prediction_parity_max_rows,
        shadow_tolerance_bps=args.shadow_tolerance_bps,
        initial_wallet=args.initial_wallet,
        since=args.since,
    )
    print(json.dumps(_json_safe(result), indent=2))
    return 1 if result["execution_parity_audit"]["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
