#!/usr/bin/env python3
"""Build a live feature ledger for native reliability-blend components.

This is the explicit bridge between a live/final-fit policy score ledger and
persisted native q_fail / difficult-period component models.  It does not train
or approximate the blend.  It materializes live-equivalent component inputs and
audits whether every persisted component feature can be resolved before native
scoring is attempted.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_live_reliability_blend_scores import (  # noqa: E402
    HEAD_BY_STRATEGY_ID,
    OOF_TO_LIVE_FEATURES,
    _timestamp_features,
)
from scripts.run_reliability_blend_optuna import (  # noqa: E402
    _anchor_extra_features,
    _anchor_meta_drift_features,
    _lagged_by_symbol,
    _timestamp_anchor_state,
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(float(value)) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _read_ledger(path: Path, *, start: str | None, end: str | None) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    if "signal_bar_ts" in frame.columns and "timestamp" not in frame.columns:
        frame["timestamp"] = frame["signal_bar_ts"]
    if "timestamp" not in frame.columns:
        raise RuntimeError("ledger missing timestamp")
    if "strategy_id" not in frame.columns:
        raise RuntimeError("ledger missing strategy_id")
    if "symbol" not in frame.columns:
        raise RuntimeError("ledger missing symbol")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if start:
        frame = frame.loc[frame["timestamp"] >= pd.Timestamp(start, tz="UTC")]
    if end:
        frame = frame.loc[frame["timestamp"] <= pd.Timestamp(end, tz="UTC")]
    frame["strategy_id"] = frame["strategy_id"].astype(str)
    frame["head"] = frame["strategy_id"].map(HEAD_BY_STRATEGY_ID)
    frame = frame.loc[frame["head"].notna()].copy()
    frame = frame.dropna(subset=["timestamp", "symbol", "head"])
    frame = frame.sort_values(["timestamp", "strategy_id", "symbol"], kind="mergesort")
    duplicate = frame.duplicated(["timestamp", "strategy_id", "symbol"], keep=False)
    if bool(duplicate.any()):
        sample = frame.loc[duplicate, ["timestamp", "strategy_id", "symbol"]].head(10).to_dict("records")
        raise RuntimeError(f"duplicate component ledger keys: {sample}")
    if frame.empty:
        raise RuntimeError("ledger is empty after filtering")
    return frame.reset_index(drop=True)


def _score_col(frame: pd.DataFrame) -> str:
    for col in ("calibrated_score", "meta_pred", "raw_prediction_score"):
        if col in frame.columns:
            return col
    raise RuntimeError("ledger missing calibrated_score/meta_pred/raw_prediction_score")


def _rank_col(frame: pd.DataFrame) -> str:
    for col in ("policy_rank_pct", "policy_ref_rank_pct"):
        if col in frame.columns:
            return col
    raise RuntimeError("ledger missing policy_rank_pct/policy_ref_rank_pct")


def _safe_logit(values: pd.Series) -> pd.Series:
    arr = np.clip(pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return pd.Series(np.log(arr / (1.0 - arr)).astype("float32"), index=values.index)


def _add_if_present(out: pd.DataFrame, name: str, frame: pd.DataFrame, source: str) -> None:
    if source in frame.columns and name not in out.columns:
        out[name] = pd.to_numeric(frame[source], errors="coerce").astype("float32")


def _live_source_for_oof(name: str) -> str | None:
    for feature, _oof_col, live_col in OOF_TO_LIVE_FEATURES:
        if feature == name:
            if live_col == "__live_anchor_score__":
                return "calibrated_score"
            if live_col == "__live_rank__":
                return "policy_rank_pct"
            return live_col
    return None


def _candidate_source_names(name: str) -> list[str]:
    raw = str(name)
    candidates: list[str] = [raw]
    prefixes = (
        ("metaout__export_oof_", "meta_lgbm_"),
        ("metaout__oof_", "meta_lgbm_"),
        ("metaout__export_oof_", ""),
        ("metaout__oof_", ""),
        ("control__export__oof_", ""),
        ("control__oof_", ""),
        ("oof_", ""),
    )
    for prefix, replacement in prefixes:
        if raw.startswith(prefix):
            candidates.append(replacement + raw[len(prefix) :])
    # A few historical OOF/meta names have no literal live column. Resolve them
    # to the live anchor score contract instead of letting a native component
    # silently depend on an impossible column name.
    alias_map = {
        "pred": ("calibrated_score", "meta_pred", "raw_prediction_score"),
        "oof_pred": ("calibrated_score", "meta_pred", "raw_prediction_score"),
        "export_oof_pred": ("calibrated_score", "meta_pred", "raw_prediction_score"),
        "lgbm_prob": ("lgbm_prob", "calibrated_score"),
        "oof_lgbm_prob": ("lgbm_prob", "calibrated_score"),
        "meta_clf": ("meta_pred", "calibrated_score"),
        "oof_meta_clf": ("meta_pred", "calibrated_score"),
        "base_clf": ("base_pred",),
        "oof_base_clf": ("base_pred",),
        "p_move": ("raw_prediction_score", "meta_pred"),
        "oof_p_move": ("raw_prediction_score", "meta_pred"),
    }
    for candidate in list(candidates):
        for alias in alias_map.get(candidate, ()):
            candidates.append(alias)
    # Training artifacts sometimes collapse double underscores before
    # persisting meta-output names.
    candidates.extend([c.replace("__", "_") for c in list(candidates)])
    seen: set[str] = set()
    out: list[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _resolve_from_sources(out: pd.DataFrame, frame: pd.DataFrame, name: str) -> bool:
    if name in out.columns:
        return True
    live_source = _live_source_for_oof(name)
    if live_source:
        _add_if_present(out, name, frame, live_source)
        return name in out.columns
    for source in _candidate_source_names(name):
        if source in out.columns:
            out[name] = pd.to_numeric(out[source], errors="coerce").astype("float32")
            return True
        if source in frame.columns:
            out[name] = pd.to_numeric(frame[source], errors="coerce").astype("float32")
            return True
    return False


def _resolve_lagged_or_ts_feature(out: pd.DataFrame, frame: pd.DataFrame, name: str) -> bool:
    suffixes = (
        ("_diff_1obs_by_symbol", 1, "diff"),
        ("_diff_4obs_by_symbol", 4, "diff"),
        ("_diff_24obs_by_symbol", 24, "diff"),
        ("_minus_prev24_mean_by_symbol", 0, "minus_prev24"),
        ("_prev24_std_by_symbol", 0, "prev24_std"),
    )
    for suffix, _lag, _kind in suffixes:
        if name.endswith(suffix):
            base = name[: -len(suffix)]
            if base not in out.columns and not _resolve_from_sources(out, frame, base):
                return False
            values = pd.to_numeric(out[base], errors="coerce").to_numpy(dtype=np.float32)
            derived = _lagged_by_symbol(frame["timestamp"], frame["symbol"], values, lags=(1, 4, 24), prefix=base)
            if name in derived.columns:
                out[name] = pd.to_numeric(derived[name], errors="coerce").astype("float32")
                return True
            return False
    ts_suffixes = (
        ("__minus_ts_mean", "minus"),
        ("__z_ts", "z"),
        ("__rank_ts", "rank"),
    )
    for suffix, kind in ts_suffixes:
        if name.endswith(suffix):
            base = name[: -len(suffix)]
            if base not in out.columns and not _resolve_from_sources(out, frame, base):
                return False
            values = pd.to_numeric(out[base], errors="coerce")
            group_keys = [frame["head"].astype(str), pd.to_datetime(frame["timestamp"], utc=True)]
            grouped = values.groupby(group_keys, sort=False)
            if kind == "minus":
                out[name] = (values - grouped.transform("mean")).astype("float32")
            elif kind == "z":
                std = grouped.transform("std").replace(0.0, np.nan)
                out[name] = ((values - grouped.transform("mean")) / std).replace([np.inf, -np.inf], np.nan).astype("float32")
            else:
                out[name] = values.groupby(group_keys, sort=False).rank(method="average", pct=True).astype("float32")
            return True
    return False


def _resolve_timestamp_aggregate_feature(out: pd.DataFrame, frame: pd.DataFrame, name: str) -> bool:
    if not (name.startswith("mean__") or name.startswith("std__")):
        return False
    agg, base = name.split("__", 1)
    if not base or base == name:
        return False
    if base not in out.columns and not _resolve_required_column(out, frame, base):
        return False
    values = pd.to_numeric(out[base], errors="coerce")
    group_keys = [frame["head"].astype(str), pd.to_datetime(frame["timestamp"], utc=True)]
    grouped = values.groupby(group_keys, sort=False)
    if agg == "mean":
        out[name] = grouped.transform("mean").astype("float32")
    else:
        out[name] = grouped.transform("std").fillna(0.0).astype("float32")
    return True


def _resolve_interaction(out: pd.DataFrame, frame: pd.DataFrame, name: str) -> bool:
    prefix = "qfail_ix__"
    middle = "__x__"
    if not name.startswith(prefix) or middle not in name:
        return False
    left, right = name[len(prefix) :].split(middle, 1)
    if left not in out.columns and not _resolve_required_column(out, frame, left):
        return False
    if right not in out.columns and not _resolve_required_column(out, frame, right):
        return False
    out[name] = (
        pd.to_numeric(out[left], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        * pd.to_numeric(out[right], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    ).astype("float32")
    return True


def _resolve_required_column(out: pd.DataFrame, frame: pd.DataFrame, name: str) -> bool:
    return (
        _resolve_from_sources(out, frame, name)
        or _resolve_lagged_or_ts_feature(out, frame, name)
        or _resolve_timestamp_aggregate_feature(out, frame, name)
        or _resolve_interaction(out, frame, name)
    )


def _required_columns(bundle: dict[str, Any]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for head, payload in (bundle.get("heads", {}) or {}).items():
        cols: list[str] = []
        artifacts = list(payload.get("models", []) or [])
        full_fit = [
            artifact
            for artifact in artifacts
            if str(artifact.get("fold", "")).lower() == "full_fit"
            or str(artifact.get("model_scope", "")).lower() == "full_fit"
        ]
        # Native live scoring uses full_fit artifacts when present. Fold models
        # are OOF diagnostics and may depend on fold-local context that is not a
        # deployable live contract.
        for artifact in (full_fit or artifacts):
            for c in artifact.get("input_feature_columns", []) or []:
                c = str(c)
                if c not in cols:
                    cols.append(c)
        out[str(head)] = cols
    return out


def build_feature_ledger(frame: pd.DataFrame, required: dict[str, list[str]] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_col = _score_col(frame)
    rank_col = _rank_col(frame)
    out = frame[["timestamp", "symbol", "strategy_id", "head"]].copy()
    out["anchor_score"] = pd.to_numeric(frame[score_col], errors="coerce").astype("float32")
    out["anchor_rank_timestamp"] = pd.to_numeric(frame[rank_col], errors="coerce").astype("float32")
    out["anchor_p0"] = out["anchor_score"]
    out["anchor_logit0"] = _safe_logit(out["anchor_score"])
    out["anchor_rank0_by_timestamp"] = out["anchor_rank_timestamp"]

    for name, _oof_col, live_col in OOF_TO_LIVE_FEATURES:
        source = "calibrated_score" if live_col == "__live_anchor_score__" else "policy_rank_pct" if live_col == "__live_rank__" else live_col
        _add_if_present(out, name, frame, source)
        if name.startswith("oof_") and name in out.columns:
            out[f"control__export__{name}"] = out[name]

    extra = _anchor_extra_features(frame["timestamp"], out["anchor_score"].to_numpy(dtype=np.float32), out["anchor_rank_timestamp"].to_numpy(dtype=np.float32))
    drift = _anchor_meta_drift_features(frame["timestamp"], frame["symbol"], out["anchor_score"].to_numpy(dtype=np.float32), out["anchor_rank_timestamp"].to_numpy(dtype=np.float32))
    anchor_state = _timestamp_anchor_state(
        frame["timestamp"],
        out["anchor_score"].to_numpy(dtype=np.float32),
        out["anchor_rank_timestamp"].to_numpy(dtype=np.float32),
    )
    anchor_state = anchor_state.reindex(pd.to_datetime(frame["timestamp"], utc=True).to_numpy()).reset_index(drop=True)
    ts_feats = _timestamp_features(pd.DataFrame({"head": frame["head"], "timestamp": frame["timestamp"], "score": out["anchor_score"]}), "score")
    ts_feats = ts_feats.rename(
        columns={
            "score_minus_ts_mean": "anchor_score_minus_ts_mean",
            "score_ts_z": "anchor_score_ts_z",
            "score_ts_rank": "anchor_score_ts_rank",
        }
    )
    out = pd.concat(
        [
            out.reset_index(drop=True),
            extra.reset_index(drop=True),
            drift.reset_index(drop=True),
            anchor_state.reset_index(drop=True),
            ts_feats.reset_index(drop=True),
        ],
        axis=1,
    )

    # Add direct metaout aliases for all existing live diagnostics so required
    # metaout__oof_* names can resolve without relying on positional columns.
    for col in frame.columns:
        if col in {"timestamp", "symbol", "strategy_id", "head"}:
            continue
        vals = pd.to_numeric(frame[col], errors="coerce")
        if vals.notna().mean() == 0.0:
            continue
        if str(col).startswith("meta_lgbm_"):
            base = str(col)[len("meta_lgbm_") :]
            out[f"metaout__oof_{base}"] = vals.astype("float32")
            out[f"metaout__export_oof_{base}"] = vals.astype("float32")
        else:
            out[f"metaout__oof_{col}"] = vals.astype("float32")

    diagnostics: list[dict[str, Any]] = []
    required = required or {}
    for head, cols in required.items():
        mask = out["head"].astype(str) == str(head)
        for col in cols:
            resolved = _resolve_required_column(out, frame, col)
            finite_fraction = (
                float(pd.to_numeric(out.loc[mask, col], errors="coerce").replace([np.inf, -np.inf], np.nan).notna().mean())
                if resolved and bool(mask.any())
                else 0.0
            )
            diagnostics.append(
                {
                    "head": head,
                    "feature": col,
                    "resolved": bool(resolved),
                    "finite_fraction": finite_fraction,
                    "rows": int(mask.sum()),
                }
            )
    out = out.loc[:, ~out.columns.duplicated()].copy()
    for col in out.columns:
        if col not in {"timestamp", "symbol", "strategy_id", "head"}:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")
    return out, pd.DataFrame(diagnostics)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--component-model-bundle", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--fail-on-missing", action="store_true")
    args = parser.parse_args()

    frame = _read_ledger(args.ledger, start=args.start, end=args.end)
    bundle = joblib.load(args.component_model_bundle) if args.component_model_bundle else {}
    features, diagnostics = build_feature_ledger(frame, _required_columns(bundle))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = args.output_dir / "native_component_feature_ledger.parquet"
    diag_path = args.output_dir / "native_component_feature_ledger_diagnostics.csv"
    features.to_parquet(feature_path, index=False)
    diagnostics.to_csv(diag_path, index=False)
    missing = diagnostics.loc[~diagnostics.get("resolved", pd.Series(dtype=bool)).astype(bool)] if not diagnostics.empty else pd.DataFrame()
    low_coverage = diagnostics.loc[
        diagnostics.get("resolved", pd.Series(dtype=bool)).astype(bool)
        & (pd.to_numeric(diagnostics.get("finite_fraction", pd.Series(dtype=float)), errors="coerce") < 1.0)
    ] if not diagnostics.empty else pd.DataFrame()
    manifest = {
        "generated_by": "build_native_reliability_component_feature_ledger",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "ledger": str(args.ledger),
        "component_model_bundle": str(args.component_model_bundle) if args.component_model_bundle else None,
        "feature_path": str(feature_path),
        "diagnostics_path": str(diag_path),
        "rows": int(len(features)),
        "columns": int(len(features.columns)),
        "heads": sorted(features["head"].dropna().astype(str).unique().tolist()),
        "required_features": int(len(diagnostics)),
        "missing_required_features": int(len(missing)),
        "low_coverage_required_features": int(len(low_coverage)),
        "status": "ok" if int(len(missing)) == 0 else "missing_required_features",
        "missing_sample": missing.head(50).to_dict("records") if not missing.empty else [],
        "low_coverage_sample": low_coverage.head(50).to_dict("records") if not low_coverage.empty else [],
    }
    manifest_path = args.output_dir / "native_component_feature_ledger_manifest.json"
    manifest_path.write_text(json.dumps(_json_safe(manifest), indent=2) + "\n")
    if args.fail_on_missing and int(len(missing)) > 0:
        raise RuntimeError(f"missing required native component features: {int(len(missing))}; see {diag_path}")
    print(json.dumps(_json_safe(manifest), indent=2)[:6000])


if __name__ == "__main__":
    main()
