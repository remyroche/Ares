#!/usr/bin/env python3
"""Append frozen AE/GMM posterior features to replay candidate rows.

This is intended for policy-threshold ablations that need soft regime
membership at replay/inference parity.  It does not refit AE/GMM; it applies the
saved train-fitted transform state to feature-store rows aligned by
timestamp/symbol/side.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.features_gmm_ae import (  # noqa: E402
    load_ae_gmm_state_artifact,
    transform_ae_gmm_features,
)


DEFAULT_CANDIDATES = ROOT / (
    "data_perp/reports/"
    "s59_h5_2025start_monthly_v6_15mchart_base_frozenfs_fixedparams_may_july_combined_20260708/"
    "threshold_basis_ablation_july0108_top3_kraken/"
    "combined_history_july_replay_candidates.parquet"
)
DEFAULT_FEATURES_DIR = ROOT / "data_perp/features/20260708_180000"
DEFAULT_AE_GMM_STATE = ROOT / "data_perp/artifacts/s59_s52_frozen_native_shadow_20260709/ae_gmm_state/ae_gmm_state.pkl"


def _symbol_feature_path(features_dir: Path, symbol: str) -> Path:
    return features_dir / f"symbol={str(symbol).replace('/', '_')}.parquet"


def _side_code(value: Any) -> float:
    try:
        numeric = float(value)
        if np.isfinite(numeric):
            return -1.0 if numeric < 0.0 else 1.0
    except (TypeError, ValueError):
        pass
    text = str(value).strip().lower()
    if text in {"short", "sell", "-1"} or text.startswith("short"):
        return -1.0
    return 1.0


def _available_columns(path: Path) -> set[str]:
    try:
        import pyarrow.parquet as pq

        return set(pq.ParquetFile(path).schema_arrow.names)
    except Exception:
        frame = pd.read_parquet(path)
        return set(map(str, frame.columns))


def _load_feature_frame(path: Path, wanted_cols: list[str], *, read_all: bool = False) -> pd.DataFrame:
    available = _available_columns(path)
    read_cols = [col for col in wanted_cols if col in available]
    if bool(read_all):
        frame = pd.read_parquet(path)
    elif read_cols:
        frame = pd.read_parquet(path, columns=read_cols)
    else:
        frame = pd.read_parquet(path)
        frame = frame.iloc[:, 0:0]
    if not isinstance(frame.index, pd.DatetimeIndex):
        if "ts" in frame.columns:
            frame = frame.set_index("ts")
        elif "__ts__" in frame.columns:
            frame = frame.set_index("__ts__")
    frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")
    frame = frame.loc[frame.index.notna()].copy()
    return frame


def _build_group_features(
    group: pd.DataFrame,
    feature_frame: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    ts = pd.to_datetime(group["timestamp"], utc=True, errors="coerce")
    aligned = feature_frame.reindex(pd.DatetimeIndex(ts))
    aligned.index = group.index
    x = aligned.reindex(columns=feature_columns, fill_value=0.0).copy()
    if "side" in feature_columns:
        x["side"] = group.get("side", group.get("side_name", 1.0)).map(_side_code).astype(np.float32)
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return x.astype(np.float32, copy=False)


def _append_live_source_regime_inputs(
    raw: pd.DataFrame,
    *,
    required_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required_source = [
        col
        for col in required_columns
        if str(col).startswith("__regime_source_") and str(col).endswith("__")
    ]
    required_raw = [col for col in required_columns if str(col).startswith("__meta_raw__")]
    if not required_source and not required_raw:
        return raw, {"source_regime_requested": False}
    out = raw.copy()
    added_raw: list[str] = []
    for col in required_raw:
        raw_name = str(col)[len("__meta_raw__") :]
        if col in out.columns and pd.to_numeric(out[col], errors="coerce").notna().any():
            continue
        if raw_name in out.columns:
            out[col] = pd.to_numeric(out[raw_name], errors="coerce").astype(np.float32, copy=False)
            added_raw.append(col)
    added_source: list[str] = []
    missing_source: list[str] = []
    if required_source:
        try:
            from scripts.materialize_candidate_source_tags import (
                DEFAULT_CONFIG,
                ARCHETYPE_COLS,
                COMPONENT_COLS,
                build_archetype_scores,
                build_component_scores,
                build_feature_registry,
                load_config,
            )

            work = out.copy()
            if "__symbol__" not in work.columns:
                work["__symbol__"] = work.index.astype(str)
            if "__ts__" not in work.columns:
                work["__ts__"] = pd.NaT
            config = load_config(DEFAULT_CONFIG)
            config["timestamp_col"] = "__ts__"
            config["symbol_col"] = "__symbol__"
            registry = build_feature_registry(work, config)
            components, component_report = build_component_scores(work, registry, config)
            archetypes = build_archetype_scores(work, components, registry, config)
            archetypes["not_dirty_shock_score"] = (
                1.0 - pd.to_numeric(archetypes["dirty_shock_avoid_score"], errors="coerce")
            ).clip(0.0, 1.0).astype(np.float32)
            archetypes["loud_clean_source_score"] = (
                pd.to_numeric(archetypes["loud_breakout_impulse_score"], errors="coerce")
                * archetypes["not_dirty_shock_score"]
            ).clip(0.0, 1.0).astype(np.float32)

            def score_family(frame: pd.DataFrame, cols: tuple[str, ...]) -> pd.Series:
                present = [col for col in cols if col in frame.columns]
                if not present:
                    return pd.Series(0.5, index=frame.index, dtype=np.float32)
                values = frame[present].apply(pd.to_numeric, errors="coerce").astype(np.float32)
                return values.max(axis=1).fillna(0.5).clip(0.0, 1.0).astype(np.float32)

            family_scores = pd.DataFrame(
                {
                    "trend_following_score": score_family(
                        archetypes,
                        (
                            "quiet_continuation_score",
                            "run_entry_score",
                            "late_run_continuation_score",
                            "clean_run_entry_score",
                        ),
                    ),
                    "mean_reversion_score": score_family(archetypes, ("retest_reversal_score",)),
                    "vol_compression_score": score_family(
                        archetypes,
                        (
                            "compression_release_score",
                            "compression_capture_candidate_score",
                            "risk_adjusted_capture_candidate_score",
                            "clean_economic_capture_candidate_score",
                        ),
                    ),
                    "breakout_impulse_score": score_family(
                        archetypes,
                        (
                            "loud_breakout_impulse_score",
                            "loud_clean_execution_score",
                            "loud_clean_source_score",
                        ),
                    ),
                    "dirty_avoid_score": score_family(
                        archetypes,
                        ("dirty_shock_avoid_score", "misleading_location_risk_score"),
                    ),
                },
                index=work.index,
            ).astype(np.float32)
            source_values: dict[str, pd.Series] = {}
            for col in list(COMPONENT_COLS):
                if col in components.columns:
                    source_values[f"__regime_source_{col}__"] = components[col]
            for col in list(ARCHETYPE_COLS):
                if col in archetypes.columns:
                    source_values[f"__regime_source_{col}__"] = archetypes[col]
            for col in family_scores.columns:
                source_values[f"__regime_source_{col}__"] = family_scores[col]
            for col in required_source:
                series = source_values.get(col)
                if series is None:
                    missing_source.append(col)
                    continue
                out[col] = pd.to_numeric(series, errors="coerce").fillna(0.5).clip(0.0, 1.0).astype(
                    np.float32,
                    copy=False,
                )
                added_source.append(col)
            report = {
                "source_regime_requested": True,
                "source_regime_status": "ok",
                "source_regime_added_count": int(len(added_source)),
                "source_regime_missing_count": int(len(missing_source)),
                "source_regime_missing_sample": missing_source[:12],
                "meta_raw_added_count": int(len(added_raw)),
                "component_neutral_counts": dict((component_report or {}).get("component_neutral_counts", {})),
            }
            return out, report
        except Exception as exc:
            return out, {
                "source_regime_requested": True,
                "source_regime_status": f"failed:{type(exc).__name__}:{exc}",
                "source_regime_added_count": int(len(added_source)),
                "source_regime_missing_count": int(len(required_source)),
                "source_regime_missing_sample": required_source[:12],
                "meta_raw_added_count": int(len(added_raw)),
            }
    return out, {
        "source_regime_requested": bool(required_source),
        "source_regime_status": "not_requested",
        "source_regime_added_count": 0,
        "source_regime_missing_count": 0,
        "meta_raw_added_count": int(len(added_raw)),
    }


def materialize(
    *,
    candidates_path: Path,
    features_dir: Path,
    ae_gmm_state_path: Path,
    out_path: Path,
    materialize_live_source_regime: bool = False,
) -> dict[str, Any]:
    candidates = pd.read_parquet(candidates_path)
    candidates = candidates.copy()
    candidates["timestamp"] = pd.to_datetime(candidates["timestamp"], utc=True, errors="coerce")
    candidates["symbol"] = candidates["symbol"].astype(str)
    state = load_ae_gmm_state_artifact(ae_gmm_state_path)
    feature_columns = [str(col) for col in state.get("feature_columns", [])]
    non_side_feature_cols = [col for col in feature_columns if col != "side"]

    transformed_parts: list[pd.DataFrame] = []
    raw_parts: list[pd.DataFrame] = []
    missing_symbols: list[str] = []
    missing_feature_counts: dict[str, int] = {}
    parity_report: dict[str, Any] = {"source_regime_requested": False}
    matched_rows = 0
    for (symbol, side_name), group in candidates.groupby(["symbol", "side_name"], dropna=False, sort=True):
        path = _symbol_feature_path(features_dir, str(symbol))
        if not path.exists():
            missing_symbols.append(str(symbol))
            continue
        available = _available_columns(path)
        missing_feature_counts[str(symbol)] = int(len([col for col in non_side_feature_cols if col not in available]))
        features = _load_feature_frame(
            path,
            non_side_feature_cols,
            read_all=bool(materialize_live_source_regime),
        )
        sorted_group = group.sort_values("timestamp")
        x = _build_group_features(sorted_group, features, feature_columns)
        if bool(materialize_live_source_regime):
            ts = pd.to_datetime(sorted_group["timestamp"], utc=True, errors="coerce")
            raw = features.reindex(pd.DatetimeIndex(ts))
            raw.index = sorted_group.index
            raw["side"] = sorted_group.get("side", sorted_group.get("side_name", 1.0)).map(_side_code).astype(np.float32)
            raw["side_name"] = str(side_name).lower()
            raw["__symbol__"] = str(symbol)
            raw["__ts__"] = ts.to_numpy()
            raw_parts.append(raw)
        matched = int(features.index.isin(pd.DatetimeIndex(pd.to_datetime(group["timestamp"], utc=True))).sum())
        matched_rows += min(matched, len(group))

        if not bool(materialize_live_source_regime):
            transformed = transform_ae_gmm_features(x, state, index=x.index)
            transformed_parts.append(transformed)

    if bool(materialize_live_source_regime):
        if raw_parts:
            raw_all = pd.concat(raw_parts, axis=0).sort_index()
            raw_all, parity_report = _append_live_source_regime_inputs(
                raw_all,
                required_columns=feature_columns,
            )
            x_all = raw_all.reindex(columns=feature_columns, fill_value=0.0).replace(
                [np.inf, -np.inf],
                np.nan,
            ).fillna(0.0)
            for (_symbol, _side), x_group in x_all.assign(
                __symbol__=raw_all["__symbol__"].astype(str),
                side_name=raw_all["side_name"].astype(str),
                __ts__=pd.to_datetime(raw_all["__ts__"], utc=True, errors="coerce"),
            ).groupby(["__symbol__", "side_name"], sort=True):
                x_group = x_group.sort_values("__ts__")
                x_base = x_group.reindex(columns=feature_columns, fill_value=0.0).astype(np.float32, copy=False)
                transformed_parts.append(transform_ae_gmm_features(x_base, state, index=x_group.index))
        else:
            parity_report = {"source_regime_requested": bool(materialize_live_source_regime), "source_regime_status": "no_raw_parts"}

    if transformed_parts:
        generated = pd.concat(transformed_parts, axis=0).sort_index()
    else:
        generated = pd.DataFrame(index=candidates.index)

    posterior_cols = [
        col
        for col in generated.columns
        if str(col).startswith("gmm_cluster_posterior_")
        or str(col).startswith("gmm_prob_")
        or str(col)
        in {
            "gmm_posterior_max",
            "gmm_posterior_margin",
            "gmm_entropy",
            "cluster_entropy",
            "cluster_entropy_norm",
            "mahalanobis_distance",
            "expected_mahalanobis",
            "cluster_speed",
            "cluster_acceleration",
            "AE_reconstruction_error",
            "dae_reconstruction_error_zscore",
        }
    ]
    out = candidates.copy()
    for col in posterior_cols:
        out[col] = pd.to_numeric(generated.reindex(out.index)[col], errors="coerce").astype("float32")
    if "gmm_cluster_id" in generated.columns:
        out["frozen_aegmm_cluster_id"] = pd.to_numeric(
            generated.reindex(out.index)["gmm_cluster_id"],
            errors="coerce",
        ).astype("float32")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    coverage = {
        "generated_by": "materialize_replay_candidate_ae_gmm_posteriors",
        "candidates": str(candidates_path),
        "features_dir": str(features_dir),
        "ae_gmm_state_path": str(ae_gmm_state_path),
        "out_path": str(out_path),
        "input_rows": int(len(candidates)),
        "output_rows": int(len(out)),
        "symbols": int(candidates["symbol"].nunique()),
        "missing_symbols": sorted(set(missing_symbols)),
        "missing_symbols_count": int(len(set(missing_symbols))),
        "generated_columns": posterior_cols + (["frozen_aegmm_cluster_id"] if "gmm_cluster_id" in generated.columns else []),
        "generated_columns_count": int(len(posterior_cols) + (1 if "gmm_cluster_id" in generated.columns else 0)),
        "posterior_columns": [col for col in posterior_cols if str(col).startswith("gmm_cluster_posterior_")],
        "posterior_source": "frozen_train_fitted_ae_gmm_state",
        "materialize_live_source_regime": bool(materialize_live_source_regime),
        "source_regime_parity": parity_report,
        "feature_columns_count": int(len(feature_columns)),
        "median_missing_input_features_per_symbol": float(np.median(list(missing_feature_counts.values()))) if missing_feature_counts else None,
        "max_missing_input_features_per_symbol": int(max(missing_feature_counts.values())) if missing_feature_counts else None,
        "timestamp_min": str(candidates["timestamp"].min()),
        "timestamp_max": str(candidates["timestamp"].max()),
    }
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    return coverage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES_DIR)
    parser.add_argument("--ae-gmm-state", type=Path, default=DEFAULT_AE_GMM_STATE)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--materialize-live-source-regime",
        action="store_true",
        help="Compute live-equivalent __regime_source_* and __meta_raw__ inputs before frozen AE/GMM transform.",
    )
    args = parser.parse_args()
    summary = materialize(
        candidates_path=args.candidates,
        features_dir=args.features_dir,
        ae_gmm_state_path=args.ae_gmm_state,
        out_path=args.out,
        materialize_live_source_regime=bool(args.materialize_live_source_regime),
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
