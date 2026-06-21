#!/usr/bin/env python3
"""Run one global regime-specialist feature-engineering metrics pass.

This intentionally avoids the base/meta training loop. It builds a single
current-vs-history assessment frame across all provided strategy label rows,
hydrates a sampled row universe from the feature store, fits the regime
feature-engineering discriminator once, and writes compact diagnostics.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from extreme_price_movements.config import CFG
from extreme_price_movements.data_store import _feature_schema_names, get_feature_path
from extreme_price_movements.pipeline_steps import inject_features_into_datasets
from extreme_price_movements.regime_specialist_feature_engineering import (
    RegimeFeatureEngineeringConfig,
    build_regime_specialist_feature_engineering_artifact,
)
from extreme_price_movements.training_utils import (
    expand_feature_group_refs,
    get_base_feature_keys,
    get_meta_feature_keys,
)


def _log(message: str) -> None:
    ts = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{ts}] Global regime specialist metrics: {message}", flush=True)


def _elapsed(start: float) -> str:
    return f"{time.perf_counter() - start:.1f}s"


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if np.isfinite(out) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_selected_features(native_root: Path) -> tuple[list[str], list[dict[str, Any]]]:
    selected: list[str] = []
    models: list[dict[str, Any]] = []
    for model_path in sorted(native_root.glob("*/model.joblib")):
        model = joblib.load(model_path)
        feats = [str(c) for c in (getattr(model, "selected_features", []) or [])]
        models.append(
            {
                "model_dir": model_path.parent.name,
                "model_path": str(model_path),
                "selected_feature_count": int(len(feats)),
            }
        )
        selected.extend(feats)
    selected = list(dict.fromkeys([c for c in selected if c]))
    return selected, models


def _load_config_base_meta_features(cfg: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    base_long = get_base_feature_keys("long", cfg)
    base_short = get_base_feature_keys("short", cfg)
    base_aux: list[str] = []
    for name in ("exh_feature_keys", "spike_feature_keys"):
        vals = cfg.get(name, [])
        if isinstance(vals, (list, tuple)):
            base_aux.extend(expand_feature_group_refs(list(vals), cfg))

    meta_by_head: dict[str, list[str]] = {}
    for head in ("reg", "clf", "mfe", "mae", "asym"):
        meta_by_head[head] = get_meta_feature_keys(head, cfg)

    requested = list(
        dict.fromkeys(
            [
                str(c)
                for c in (
                    list(base_long)
                    + list(base_short)
                    + list(base_aux)
                    + [c for vals in meta_by_head.values() for c in vals]
                )
                if isinstance(c, str) and c
            ]
        )
    )
    diagnostics = {
        "base_long_count": int(len(base_long)),
        "base_short_count": int(len(base_short)),
        "base_aux_count": int(len(base_aux)),
        "meta_counts": {k: int(len(v)) for k, v in meta_by_head.items()},
        "requested_feature_count": int(len(requested)),
    }
    return requested, diagnostics


def _filter_feature_store_available(
    *,
    data_root: Path,
    ts_sig: pd.Timestamp,
    symbols: pd.Series,
    requested_features: list[str],
) -> tuple[list[str], dict[str, Any]]:
    stage_start = time.perf_counter()
    requested_set = set(requested_features)
    available: set[str] = set()
    inspected_symbols = 0
    empty_schema_symbols = 0
    symbol_list = sorted({str(s) for s in symbols.dropna().unique() if str(s)})
    _log(
        f"feature-store availability filter start: symbols={len(symbol_list)} "
        f"requested_features={len(requested_features)}"
    )
    for sym_i, sym in enumerate(symbol_list, start=1):
        fpath = get_feature_path(str(data_root), ts_sig, sym)
        schema = _feature_schema_names(str(fpath))
        if not schema:
            empty_schema_symbols += 1
            continue
        inspected_symbols += 1
        available.update(requested_set.intersection(schema))
        if len(available) == len(requested_set):
            break
        if sym_i % 25 == 0 or sym_i == len(symbol_list):
            _log(
                f"feature-store availability progress: symbols={sym_i}/{len(symbol_list)} "
                f"available={len(available)}/{len(requested_features)} "
                f"elapsed={_elapsed(stage_start)}"
            )

    filtered = [c for c in requested_features if c in available]
    missing = [c for c in requested_features if c not in available]
    diagnostics = {
        "filter_feature_store_available": True,
        "requested_feature_count": int(len(requested_features)),
        "available_feature_count": int(len(filtered)),
        "missing_feature_count": int(len(missing)),
        "inspected_symbols": int(inspected_symbols),
        "empty_schema_symbols": int(empty_schema_symbols),
        "missing_features_preview": missing[:100],
    }
    _log(
        f"feature-store availability complete: available={len(filtered)} "
        f"missing={len(missing)} inspected_symbols={inspected_symbols} "
        f"elapsed={_elapsed(stage_start)}"
    )
    return filtered, diagnostics


def _load_row_universe(data_root: Path, row_universe_run_id: str, label_run_id: str) -> pd.DataFrame:
    row_path = (
        data_root
        / "artifacts"
        / row_universe_run_id
        / "row_universe"
        / "train_row_universe_all.parquet"
    )
    if row_path.exists():
        return pd.read_parquet(row_path)
    labels_dir = data_root / "artifacts" / label_run_id / "labels"
    parts = []
    for path in sorted(labels_dir.glob("train_*.parquet")):
        df = pd.read_parquet(path, columns=["__ts__", "__symbol__"])
        parts.append(
            pd.DataFrame(
                {
                    "dataset": path.stem,
                    "timestamp": df["__ts__"],
                    "symbol": df["__symbol__"],
                }
            )
        )
    if not parts:
        raise FileNotFoundError(f"No row universe or train labels found under {data_root / 'artifacts' / label_run_id}")
    return pd.concat(parts, axis=0, ignore_index=True, copy=False)


def _sample_rows(
    rows: pd.DataFrame,
    *,
    max_rows: int,
    current_window_days: float,
    timestamp_balanced: bool = True,
    rows_per_timestamp_cap: int = 0,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = rows.loc[:, ["dataset", "timestamp", "symbol"]].copy()
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True, errors="coerce")
    rows = rows.dropna(subset=["timestamp", "symbol"])
    rows = rows.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    end = rows["timestamp"].max()
    start = end - pd.Timedelta(days=float(current_window_days))
    timestamp_count = int(rows["timestamp"].nunique()) if len(rows) else 0
    sampling_diag: dict[str, Any] = {
        "timestamp_balanced": bool(timestamp_balanced),
        "source_timestamp_count": timestamp_count,
    }
    if bool(timestamp_balanced) and timestamp_count > 0:
        if int(rows_per_timestamp_cap) > 0:
            cap = int(rows_per_timestamp_cap)
            cap_source = "explicit"
        else:
            cap = max(1, int(max_rows) // max(timestamp_count, 1))
            cap_source = "auto_budget"
        sampling_diag.update(
            {
                "rows_per_timestamp_cap": int(cap),
                "rows_per_timestamp_cap_source": cap_source,
            }
        )
        rng = np.random.default_rng(int(random_state))
        keep_parts: list[np.ndarray] = []
        # Sample the same way for current and historical timestamps so the
        # discriminator cannot learn the current window from row density alone.
        for positions in rows.groupby("timestamp", sort=False).indices.values():
            pos = np.asarray(positions, dtype=np.int64)
            if pos.size > cap:
                pos = rng.choice(pos, size=cap, replace=False).astype(np.int64)
            keep_parts.append(pos)
        keep = (
            np.sort(np.concatenate(keep_parts).astype(np.int64))
            if keep_parts
            else np.zeros(0, dtype=np.int64)
        )
        if len(keep) > int(max_rows):
            take = np.linspace(0, len(keep) - 1, int(max_rows)).round().astype(np.int64)
            keep = keep[np.unique(take)]
            sampling_diag["post_cap_budget_trimmed"] = True
        else:
            sampling_diag["post_cap_budget_trimmed"] = False
    else:
        current = (rows["timestamp"] >= start) & (rows["timestamp"] <= end)
        current_idx = np.flatnonzero(current.to_numpy(dtype=bool))
        hist_idx = np.flatnonzero(~current.to_numpy(dtype=bool))
        keep = current_idx
        remaining = max(0, int(max_rows) - len(keep))
        if remaining > 0 and len(hist_idx) > 0:
            if len(hist_idx) > remaining:
                take = np.linspace(0, len(hist_idx) - 1, remaining).round().astype(np.int64)
                hist_idx = hist_idx[np.unique(take)]
            keep = np.concatenate([hist_idx, keep])
        if len(keep) > int(max_rows):
            take = np.linspace(0, len(keep) - 1, int(max_rows)).round().astype(np.int64)
            keep = keep[np.unique(take)]
        sampling_diag.update(
            {
                "rows_per_timestamp_cap": None,
                "rows_per_timestamp_cap_source": "disabled",
                "post_cap_budget_trimmed": len(keep) > int(max_rows),
            }
        )
    sample = rows.iloc[np.sort(np.unique(keep))].reset_index(drop=True)
    sample_current = (sample["timestamp"] >= start) & (sample["timestamp"] <= end)

    def _rows_per_timestamp_summary(mask: pd.Series) -> dict[str, float | int]:
        counts = sample.loc[mask, "timestamp"].value_counts()
        if counts.empty:
            return {
                "timestamp_count": 0,
                "mean": 0.0,
                "p50": 0.0,
                "p90": 0.0,
                "max": 0,
            }
        arr = counts.to_numpy(dtype=np.float64)
        return {
            "timestamp_count": int(len(arr)),
            "mean": float(np.mean(arr)),
            "p50": float(np.quantile(arr, 0.50)),
            "p90": float(np.quantile(arr, 0.90)),
            "max": int(np.max(arr)),
        }

    diag = {
        "source_rows": int(len(rows)),
        "sample_rows": int(len(sample)),
        "current_rows": int(sample_current.sum()),
        "history_rows": int((sample["timestamp"] < start).sum()),
        "start": rows["timestamp"].min().isoformat() if len(rows) else None,
        "end": end.isoformat() if pd.notna(end) else None,
        "current_start": start.isoformat() if pd.notna(start) else None,
        "dataset_count": int(sample["dataset"].nunique()) if len(sample) else 0,
        "symbol_count": int(sample["symbol"].nunique()) if len(sample) else 0,
        "sampling": sampling_diag,
        "rows_per_timestamp": {
            "history": _rows_per_timestamp_summary(~sample_current),
            "current": _rows_per_timestamp_summary(sample_current),
        },
    }
    return sample, diag


def _artifact_summary(artifact: Any) -> dict[str, Any]:
    row_score_summary: dict[str, dict[str, float]] = {}
    for col in getattr(artifact, "row_scores", pd.DataFrame()).columns:
        arr = pd.to_numeric(artifact.row_scores[col], errors="coerce").to_numpy(dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        row_score_summary[str(col)] = {
            "mean": float(np.mean(finite)) if finite.size else float("nan"),
            "std": float(np.std(finite)) if finite.size else float("nan"),
            "p10": float(np.percentile(finite, 10)) if finite.size else float("nan"),
            "p50": float(np.percentile(finite, 50)) if finite.size else float("nan"),
            "p90": float(np.percentile(finite, 90)) if finite.size else float("nan"),
        }
    return {
        "schema_version": str(getattr(artifact, "schema_version", "")),
        "selected_features": list(getattr(artifact, "selected_features", []) or []),
        "selected_raw_features": list(getattr(artifact, "selected_raw_features", []) or []),
        "selected_pair_features": list(getattr(artifact, "selected_pair_features", []) or []),
        "selected_drift_features": list(getattr(artifact, "selected_drift_features", []) or []),
        "lgbm_features": list(getattr(artifact, "lgbm_features", []) or []),
        "elasticnet_features": list(getattr(artifact, "elasticnet_features", []) or []),
        "selected_feature_count": int(len(getattr(artifact, "selected_features", []) or [])),
        "selected_raw_feature_count": int(len(getattr(artifact, "selected_raw_features", []) or [])),
        "selected_pair_feature_count": int(len(getattr(artifact, "selected_pair_features", []) or [])),
        "selected_drift_feature_count": int(len(getattr(artifact, "selected_drift_features", []) or [])),
        "lgbm_feature_count": int(len(getattr(artifact, "lgbm_features", []) or [])),
        "elasticnet_feature_count": int(len(getattr(artifact, "elasticnet_features", []) or [])),
        "row_score_summary": row_score_summary,
        "diagnostics": getattr(artifact, "diagnostics", {}) or {},
    }


def main() -> int:
    total_start = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data_perp")
    parser.add_argument("--feature-run-id", required=True)
    parser.add_argument("--label-run-id", required=True)
    parser.add_argument("--preset-run-id", required=True)
    parser.add_argument("--row-universe-run-id", default="")
    parser.add_argument("--output-run-id", required=True)
    parser.add_argument(
        "--feature-source",
        choices=("config-base-meta", "native-selected"),
        default="config-base-meta",
        help="Candidate feature universe to hydrate before regime selection.",
    )
    parser.add_argument(
        "--output-name",
        default="",
        help="Subdirectory under artifacts/<run>/regime_specialist.",
    )
    parser.add_argument(
        "--no-filter-feature-store-available",
        action="store_true",
        help="Hydrate every requested key even if no sampled symbol schema exposes it.",
    )
    parser.add_argument("--max-rows", type=int, default=250000)
    parser.add_argument(
        "--rows-per-timestamp-cap",
        type=int,
        default=0,
        help=(
            "Maximum sampled rows per timestamp. Default 0 derives a cap from "
            "--max-rows / timestamp_count."
        ),
    )
    parser.add_argument(
        "--disable-timestamp-balanced-sampling",
        action="store_true",
        help=(
            "Use legacy sampling that keeps all current rows and thins history. "
            "Not recommended for domain-classifier diagnostics."
        ),
    )
    parser.add_argument("--current-window-days", type=float, default=28.0)
    parser.add_argument("--validation-metrics", action="store_true")
    parser.add_argument("--max-final-features", type=int, default=40)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    _log(
        f"run start: feature_run_id={args.feature_run_id} label_run_id={args.label_run_id} "
        f"preset_run_id={args.preset_run_id} output_run_id={args.output_run_id} "
        f"feature_source={args.feature_source} max_rows={args.max_rows} "
        f"current_window_days={args.current_window_days} validation={bool(args.validation_metrics)}"
    )

    data_root = Path(args.data_root)
    row_universe_run_id = args.row_universe_run_id or args.output_run_id
    cfg = dict(CFG)
    cfg["data_root"] = str(data_root)
    cfg["lgbm_require_native_preset"] = True
    cfg["feature_source_run_id"] = str(args.feature_run_id)
    native_root = data_root / "artifacts" / args.preset_run_id / "models" / "native"
    model_rows: list[dict[str, Any]] = []
    stage_start = time.perf_counter()
    if args.feature_source == "native-selected":
        requested_features, model_rows = _load_selected_features(native_root)
        feature_source_diag: dict[str, Any] = {
            "native_model_count": int(len(model_rows)),
            "requested_feature_count": int(len(requested_features)),
        }
        if not requested_features:
            raise RuntimeError(f"No selected features loaded from {native_root}")
    else:
        requested_features, feature_source_diag = _load_config_base_meta_features(cfg)
        if native_root.exists():
            _, model_rows = _load_selected_features(native_root)
        if not requested_features:
            raise RuntimeError("No config base/meta features resolved from CFG")
    _log(
        f"feature universe resolved: source={args.feature_source} "
        f"requested_features={len(requested_features)} native_models={len(model_rows)} "
        f"elapsed={_elapsed(stage_start)}"
    )

    stage_start = time.perf_counter()
    rows = _load_row_universe(data_root, row_universe_run_id, args.label_run_id)
    _log(
        f"row universe loaded: rows={len(rows)} source_run={row_universe_run_id} "
        f"elapsed={_elapsed(stage_start)}"
    )
    stage_start = time.perf_counter()
    sample, sample_diag = _sample_rows(
        rows,
        max_rows=int(args.max_rows),
        current_window_days=float(args.current_window_days),
        timestamp_balanced=not bool(args.disable_timestamp_balanced_sampling),
        rows_per_timestamp_cap=int(args.rows_per_timestamp_cap),
        random_state=int(args.random_state),
    )
    _log(f"row sample built: {sample_diag} elapsed={_elapsed(stage_start)}")

    stage_start = time.perf_counter()
    dataset = pd.DataFrame(
        {
            "__ts__": sample["timestamp"],
            "__symbol__": sample["symbol"],
        }
    )
    _log(f"hydration dataset prepared: rows={len(dataset)} elapsed={_elapsed(stage_start)}")
    ts_sig = pd.to_datetime(args.feature_run_id, format="%Y%m%d_%H%M%S").tz_localize("UTC")
    if args.no_filter_feature_store_available:
        selected_features = list(requested_features)
        availability_diag = {
            "filter_feature_store_available": False,
            "requested_feature_count": int(len(requested_features)),
            "available_feature_count": int(len(selected_features)),
            "missing_feature_count": 0,
        }
    else:
        selected_features, availability_diag = _filter_feature_store_available(
            data_root=data_root,
            ts_sig=ts_sig,
            symbols=sample["symbol"],
            requested_features=requested_features,
        )
    if not selected_features:
        raise RuntimeError(
            "No requested features are available in the sampled feature-store schemas"
        )
    _log(
        f"selected hydration features: requested={len(requested_features)} "
        f"selected={len(selected_features)} availability={availability_diag}"
    )

    stage_start = time.perf_counter()
    _log(
        f"feature hydration start: datasets=1 rows={len(dataset)} keys={len(selected_features)}"
    )
    hydrated = inject_features_into_datasets(
        {"global_regime_specialist": dataset},
        ts_sig,
        cfg,
        selected_features,
    )["global_regime_specialist"]
    _log(
        f"feature hydration complete: shape={hydrated.shape} elapsed={_elapsed(stage_start)}"
    )
    frame = hydrated.rename(columns={"__ts__": "timestamp", "__symbol__": "symbol"})
    ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    end = ts.max()
    current_start = end - pd.Timedelta(days=float(args.current_window_days))
    current_mask = (ts >= current_start) & (ts <= end)
    historical_mask = ts < current_start
    _log(
        f"assessment masks ready: end={end.isoformat() if pd.notna(end) else None} "
        f"current_start={current_start.isoformat() if pd.notna(current_start) else None} "
        f"current_rows={int(current_mask.sum())} historical_rows={int(historical_mask.sum())}"
    )

    config = RegimeFeatureEngineeringConfig(
        random_state=int(args.random_state),
        max_final_features=int(args.max_final_features),
        run_validation_diagnostics=bool(args.validation_metrics),
    )
    stage_start = time.perf_counter()
    _log(
        f"feature-engineering artifact build start: frame_shape={frame.shape} "
        f"candidate_features={len(selected_features)}"
    )
    artifact = build_regime_specialist_feature_engineering_artifact(
        frame,
        timestamp_col="timestamp",
        symbol_col="symbol",
        candidate_features=selected_features,
        current_mask=current_mask.to_numpy(dtype=bool),
        historical_mask=historical_mask.to_numpy(dtype=bool),
        config=config,
    )
    _log(
        f"feature-engineering artifact build complete: selected={len(artifact.selected_features)} "
        f"materialized={artifact.materialized_features.shape[1]} "
        f"row_score_cols={artifact.row_scores.shape[1]} elapsed={_elapsed(stage_start)}"
    )

    out_dir = (
        data_root
        / "artifacts"
        / args.output_run_id
        / "regime_specialist"
        / (
            args.output_name
            or (
                "global_once_config_base_meta"
                if args.feature_source == "config-base-meta"
                else "global_once"
            )
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    _log(f"writing outputs to {out_dir}")
    summary = {
        "mode": "global_regime_specialist_once",
        "feature_run_id": str(args.feature_run_id),
        "label_run_id": str(args.label_run_id),
        "preset_run_id": str(args.preset_run_id),
        "row_universe_run_id": str(row_universe_run_id),
        "feature_source": str(args.feature_source),
        "requested_feature_count": int(len(requested_features)),
        "selected_feature_union_count": int(len(selected_features)),
        "feature_source_diagnostics": feature_source_diag,
        "feature_store_availability": availability_diag,
        "native_models": model_rows,
        "sample": sample_diag,
        "artifact": _artifact_summary(artifact),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    _log("wrote summary.json")
    artifact.row_scores.reset_index(drop=True).to_parquet(out_dir / "row_scores.parquet")
    _log("wrote row_scores.parquet")
    artifact.materialized_features.reset_index(drop=True).to_parquet(
        out_dir / "materialized_features.parquet"
    )
    _log("wrote materialized_features.parquet")
    artifact.feature_report.to_parquet(out_dir / "feature_report.parquet")
    _log("wrote feature_report.parquet")
    with (out_dir / "artifact.pkl").open("wb") as f:
        pickle.dump(artifact, f, protocol=pickle.HIGHEST_PROTOCOL)
    _log(f"wrote artifact.pkl; run complete elapsed={_elapsed(total_start)}")
    print(json.dumps({"output_dir": str(out_dir), **summary}, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
