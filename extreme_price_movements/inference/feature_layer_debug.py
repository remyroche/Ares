"""Per-decision feature/model-frame dumps for live inference parity checks."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from extreme_price_movements.ebm_on_lgbm import (
    _compute_oof_bundle_tree_frame,
    _compute_soft_tree_features_ebm,
)
from extreme_price_movements.inference.model_orchestrator import (
    ModelOrchestrator,
    _alpha_prediction_frame_for_model,
    _effective_alpha_feature_contract,
    _extract_ebm_contract_model,
    _synthetic_ebm_raw_features,
)
from extreme_price_movements.inference.parity import strategy_core_id
from extreme_price_movements.utils import tprint


def feature_layer_debug_enabled(
    runtime_cfg: dict[str, Any] | None,
    *,
    live_test_mode: bool = False,
) -> bool:
    """Return whether live per-decision feature-layer dumps should be written."""
    cfg = runtime_cfg or {}
    raw = os.environ.get("EPM_LIVE_FEATURE_LAYER_DEBUG")
    if raw is None:
        raw = cfg.get("live_feature_layer_debug_enabled")
    if raw is None:
        # Keep this on in live-test because the overhead is bounded by the small
        # candidate set and this is the only way to inspect exact live frames.
        return bool(live_test_mode)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if np.isfinite(out) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _write_frame(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        df.to_parquet(tmp)
        tmp.replace(path)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        df.to_csv(path.with_suffix(".csv"))


def _feature_values_frame(row: pd.DataFrame) -> pd.DataFrame:
    if row is None or row.empty:
        return pd.DataFrame(columns=["feature", "value"])
    series = row.iloc[0]
    out = pd.DataFrame(
        {
            "feature": [str(c) for c in series.index],
            "value": [_safe_float(v) for v in series.to_numpy()],
        }
    )
    return out.sort_values("feature").reset_index(drop=True)


def _resolve_alpha_input(
    orchestrator: ModelOrchestrator,
    feature_row: pd.DataFrame,
    side: str,
    strategy_id: str,
) -> tuple[Any, pd.DataFrame, list[str], str]:
    model_key, model_info = orchestrator._alpha_model_info_for_kind(
        side,
        str(strategy_id),
    )
    if not isinstance(model_info, dict):
        return None, pd.DataFrame(), [], model_key
    model = model_info.get("model")
    feat_cols = _effective_alpha_feature_contract(model_info)
    X = orchestrator._align_alpha_feature_contract(feature_row, feat_cols)
    if X.empty:
        return model, X, feat_cols, model_key
    X = _alpha_prediction_frame_for_model(model, X, feat_cols)
    return model, X, feat_cols, model_key


def _resolve_meta_input(
    orchestrator: ModelOrchestrator,
    feature_row: pd.DataFrame,
    side: str,
    selected_strategy: str,
    base_pred: float,
) -> tuple[Any, pd.DataFrame, list[str], str]:
    selected = str(selected_strategy)
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
    # Mirror ModelOrchestrator.run_full_chain(): the selected strategy key is
    # injected first, then model-derived columns are materialized from base_pred.
    meta[selected] = float(base_pred)
    if core:
        meta[core] = float(base_pred)
    meta = orchestrator._materialize_alpha_model_meta_features(
        meta,
        model,
        side=side,
        kind=selected,
    )
    meta = orchestrator._materialize_meta_model_derived_features(
        meta,
        model,
        side=side,
        kind=selected,
    )
    X = meta.reindex(columns=feat_cols, fill_value=0.0).fillna(0.0)
    return model, X, feat_cols, key


def _model_raw_tree_frames(model: Any, X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    ebm = _extract_ebm_contract_model(model)
    if ebm is None:
        return pd.DataFrame(index=X.index), pd.DataFrame(index=X.index), {}
    raw_names = [str(c) for c in (getattr(ebm, "raw_selected_features", []) or [])]
    if not raw_names:
        return pd.DataFrame(index=X.index), pd.DataFrame(index=X.index), {}

    raw_df = X.reindex(columns=raw_names, fill_value=0.0)
    synthetic_raw_contract = all(re.fullmatch(r"f\d+", name) is not None for name in raw_names)
    positional_mapping = (
        getattr(ebm, "positional_feature_mapping", None)
        or getattr(ebm, "meta_positional_feature_mapping_", None)
        or getattr(model, "positional_feature_mapping", None)
        or getattr(model, "meta_positional_feature_mapping_", None)
        or {}
    )
    missing_real: list[str] = []
    if synthetic_raw_contract and isinstance(positional_mapping, dict) and positional_mapping:
        mapped = pd.DataFrame(index=X.index)
        for raw_name in raw_names:
            real_name = str(positional_mapping.get(raw_name, ""))
            if real_name and real_name in X.columns:
                mapped[raw_name] = X[real_name]
            else:
                mapped[raw_name] = 0.0
                missing_real.append(real_name or raw_name)
        raw_df = mapped
    elif synthetic_raw_contract and raw_df.abs().sum(axis=1).iloc[0] == 0.0 and X.shape[1] >= len(raw_names):
        raw_df = X.iloc[:, : len(raw_names)].copy()
        raw_df.columns = raw_names
    raw_df = raw_df.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    tree_names = [str(c) for c in (getattr(ebm, "tree_feature_names", []) or [])]
    if not tree_names:
        return raw_df, pd.DataFrame(index=X.index), {
            "raw_selected_features_n": len(raw_names),
            "tree_feature_names_n": 0,
            "missing_real_features": missing_real,
        }
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
    missing_tree = [name for name in tree_names if name not in tree_df.columns]
    for name in missing_tree:
        tree_df[name] = 0.0
    tree_df = tree_df.reindex(columns=tree_names, fill_value=0.0)
    tree_df = tree_df.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return raw_df, tree_df, {
        "raw_selected_features_n": len(raw_names),
        "tree_feature_names_n": len(tree_names),
        "missing_real_features": missing_real,
        "missing_tree_features": missing_tree,
        "tree_feature_config_oof": bool(
            isinstance(tree_config, dict) and tree_config.get("oof_tree_features")
        ),
    }


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


def dump_live_feature_layers(
    *,
    orchestrator: ModelOrchestrator,
    feature_row: pd.DataFrame,
    symbol: str,
    side: str,
    selected_strategy: str,
    chain_results: dict[str, Any],
    runtime_cfg: dict[str, Any] | None,
    timestamp: Any,
    signal_bar_ts: Any,
    feature_universe_symbols: list[str] | None = None,
) -> Path | None:
    """Persist exact model-frame layers used for a live decision."""
    cfg = runtime_cfg or {}
    run_id = str(cfg.get("run_id") or cfg.get("policy_artifact_run_id") or "latest")
    root = Path(
        cfg.get(
            "live_feature_layer_debug_dir",
            Path(str(cfg.get("data_root", "data")))
            / "artifacts"
            / run_id
            / "live_feature_layer_debug",
        )
    )
    decision_ts = pd.Timestamp(timestamp) if timestamp is not None else pd.Timestamp.now(tz="UTC")
    safe_symbol = str(symbol).replace("/", "_").replace(":", "_")
    safe_strategy = strategy_core_id(str(selected_strategy)).replace("/", "_").replace(":", "_")
    safe_ts = decision_ts.strftime("%Y%m%dT%H%M%S%fZ")
    out_dir = root / safe_ts / f"{side}_{safe_symbol}_{safe_strategy}"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_pred = _safe_float(chain_results.get("base_pred"))
    meta_pred = _safe_float(chain_results.get("meta_pred"))
    alpha_model, alpha_X, alpha_cols, alpha_key = _resolve_alpha_input(
        orchestrator,
        feature_row,
        side,
        str(selected_strategy),
    )
    meta_model, meta_X, meta_cols, meta_key = _resolve_meta_input(
        orchestrator,
        feature_row,
        side,
        str(selected_strategy),
        base_pred,
    )
    alpha_raw, alpha_tree, alpha_diag = _model_raw_tree_frames(alpha_model, alpha_X)
    meta_raw, meta_tree, meta_diag = _model_raw_tree_frames(meta_model, meta_X)

    _write_frame(out_dir / "post_transform_candidate_features.parquet", _feature_values_frame(feature_row))
    _write_frame(out_dir / "alpha_model_input.parquet", _frame_long(alpha_X, "alpha_model_input"))
    _write_frame(out_dir / "meta_model_input.parquet", _frame_long(meta_X, "meta_model_input"))
    _write_frame(out_dir / "alpha_ebm_raw_contract.parquet", _frame_long(alpha_raw, "alpha_ebm_raw_contract"))
    _write_frame(out_dir / "alpha_lgbm_tree_features.parquet", _frame_long(alpha_tree, "alpha_lgbm_tree_features"))
    _write_frame(out_dir / "meta_ebm_raw_contract.parquet", _frame_long(meta_raw, "meta_ebm_raw_contract"))
    _write_frame(out_dir / "meta_lgbm_tree_features.parquet", _frame_long(meta_tree, "meta_lgbm_tree_features"))
    all_layers = pd.concat(
        [
            _frame_long(alpha_raw, "alpha_ebm_raw_contract"),
            _frame_long(alpha_tree, "alpha_lgbm_tree_features"),
            _frame_long(meta_raw, "meta_ebm_raw_contract"),
            _frame_long(meta_tree, "meta_lgbm_tree_features"),
        ],
        ignore_index=True,
    )
    _write_frame(out_dir / "model_layers_long.parquet", all_layers)

    summary = {
        "schema_version": "live_feature_layer_debug_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "symbol": symbol,
        "side": side,
        "selected_strategy": selected_strategy,
        "strategy_id": strategy_core_id(str(selected_strategy)),
        "decision_ts": decision_ts.isoformat(),
        "signal_bar_ts": str(signal_bar_ts),
        "base_pred": base_pred,
        "meta_pred": meta_pred,
        "meta_prediction_source": chain_results.get("meta_prediction_source"),
        "calibrated_score": _safe_float(chain_results.get("calibrated_score")),
        "policy_rank_pct": _safe_float(chain_results.get("policy_rank_pct")),
        "rank_score_source": chain_results.get("rank_score_source"),
        "alpha_model_key": alpha_key,
        "meta_model_key": meta_key,
        "post_transform_candidate_features_n": int(feature_row.shape[1]) if isinstance(feature_row, pd.DataFrame) else 0,
        "feature_universe_n": len(feature_universe_symbols or []),
        "feature_universe_symbols": sorted(str(s) for s in (feature_universe_symbols or [])),
        "alpha_model_input_n": len(alpha_cols),
        "meta_model_input_n": len(meta_cols),
        "alpha_diag": alpha_diag,
        "meta_diag": meta_diag,
        "pre_causal_transform_status": (
            "not_available_in_exact_live_decision_path; live path consumes cached/"
            "merged post-transform feature frames. Use replay comparison for "
            "reconstructed pre-CausalTransform values."
        ),
        "prediction_replay_note": (
            "The stored model frames are exact for this decision row. If "
            "meta_prediction_source=batch_meta_after_base_gate, the live scalar "
            "prediction came from a batch call; use the ledger scalar as the "
            "prediction truth and these frames to inspect the row-level inputs."
        ),
        "files": {
            "post_transform_candidate_features": "post_transform_candidate_features.parquet",
            "alpha_model_input": "alpha_model_input.parquet",
            "meta_model_input": "meta_model_input.parquet",
            "alpha_ebm_raw_contract": "alpha_ebm_raw_contract.parquet",
            "alpha_lgbm_tree_features": "alpha_lgbm_tree_features.parquet",
            "meta_ebm_raw_contract": "meta_ebm_raw_contract.parquet",
            "meta_lgbm_tree_features": "meta_lgbm_tree_features.parquet",
            "model_layers_long": "model_layers_long.parquet",
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2))
    tprint(
        "Persisted live feature-layer debug dump: "
        f"symbol={symbol} side={side} strategy={strategy_core_id(str(selected_strategy))} "
        f"path={out_dir}"
    )
    return out_dir


def update_live_feature_layer_rank_summary(
    debug_dir: str | Path | None,
    *,
    decision: dict[str, Any] | None = None,
    chain_results: dict[str, Any] | None = None,
    gate_allowed: bool | None = None,
    gate_reason: str | None = None,
) -> None:
    """Attach post-policy-rank gate values to an existing live debug summary.

    Feature-layer dumps are written immediately after model scoring so they
    capture the exact model matrices. The policy/auction rank gate runs later
    after cross-strategy decision rows are assembled, so update the same
    summary file once those rank values exist.
    """
    if debug_dir is None:
        return
    path = Path(debug_dir) / "summary.json"
    if not path.exists():
        return
    try:
        summary = json.loads(path.read_text())
    except Exception:
        return
    decision = decision or {}
    chain_results = chain_results or {}

    def _first_value(key: str) -> Any:
        if key in chain_results:
            return chain_results.get(key)
        return decision.get(key)

    summary.update(
        {
            "rank_gate_updated_at": datetime.now(timezone.utc).isoformat(),
            "rank_gate_allowed": gate_allowed,
            "rank_gate_reason": gate_reason,
            "calibrated_score": _safe_float(_first_value("calibrated_score")),
            "policy_rank_pct": _safe_float(_first_value("policy_rank_pct")),
            "policy_rank_reference_n": _safe_float(
                _first_value("policy_rank_reference_n")
            ),
            "policy_rank_reference_source": _first_value(
                "policy_rank_reference_source"
            ),
            "auction_rank_pct": _safe_float(_first_value("auction_rank_pct")),
            "auction_rank_reference_n": _safe_float(
                _first_value("auction_rank_reference_n")
            ),
            "auction_rank_reference_source": _first_value(
                "auction_rank_reference_source"
            ),
            "auction_rank_score_source": _first_value("auction_rank_score_source"),
            "normalized_rank_score": _safe_float(
                _first_value("normalized_rank_score")
            ),
            "threshold_rank_score": _safe_float(
                _first_value("threshold_rank_score")
            ),
            "threshold_rank_score_source": _first_value(
                "threshold_rank_score_source"
            ),
            "effective_threshold": _safe_float(_first_value("effective_threshold")),
            "rank_score_source": _first_value("rank_score_source"),
        }
    )
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_json_safe(summary), indent=2))
    tmp.replace(path)
