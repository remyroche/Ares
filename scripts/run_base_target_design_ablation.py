#!/usr/bin/env python3
"""Chronological base-target ablation on the frozen S59 feature/model contract.

This is intentionally not a production training entrypoint.  It isolates target
design by holding the following fixed for every arm: OOS fold rows, selected
features, LightGBM parameters, and the base candidate universe.  The cached
fold matrices are the exact matrices used by the reference base run; no feature
selection or HPO is performed here.

The source labels are already net of their materialized 1% round-trip cost.
Target definitions therefore use the net return directly and record that no
additional cost was subtracted.  Metrics use the same net field once.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from lightgbm import LGBMRegressor


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.target_design import (  # noqa: E402
    DEFAULT_TARGET_SPECS,
    TargetDesignSpec,
    build_target,
    build_training_weights,
    fit_target_reference,
    fit_training_weight_reference,
)


DEFAULT_BASE_REPORT = Path(
    "data_perp/reports/"
    "s59_h5_fullthroughjul10_base_configfull_freshmda_fixedparams_wf30_20260713"
)
DEFAULT_OUTPUT_ROOT = Path("data_perp/reports/base_target_design_ablation")
DEFAULT_FOLDS = (
    "2026-03-28_2026-04-27",
    "2026-04-27_2026-05-27",
    "2026-05-27_2026-06-26",
)
TOP_FRACTIONS = (0.10, 0.20)
IDENTITY_COLUMNS = ("__ts__", "__symbol__", "side_name", "__archetype_label_family__")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _read_fold(
    base_report: Path,
    fold: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cache = base_report / "_fold_cache" / fold
    required = ("x_train.parquet", "x_valid.parquet", "train.parquet", "valid.parquet", "valid_metrics.parquet")
    missing = [name for name in required if not (cache / name).exists()]
    if missing:
        raise FileNotFoundError(f"Fold {fold} missing cached files: {missing}")
    x_train = pd.read_parquet(cache / "x_train.parquet")
    x_valid = pd.read_parquet(cache / "x_valid.parquet")

    # Do not load the hundreds of label/path diagnostics that the target arms
    # never touch.  With 300k cached train rows this avoids a material and
    # otherwise needless memory peak before LightGBM starts its fit.
    train_path = cache / "train.parquet"
    valid_path = cache / "valid.parquet"
    train_available = set(pq.ParquetFile(train_path).schema.names)
    valid_available = set(pq.ParquetFile(valid_path).schema.names)
    train_requested = (
        "__ts__",
        "__first_touch_target_soft__",
        "__first_touch_capture_net__",
        "__u_policy_net__",
        "__y_ret__",
        "__first_touch_round_trip_cost__",
        "__barrier_pct__",
        "__sl__",
        "__tp__",
    )
    valid_requested = (
        "__ts__",
        "__symbol__",
        "side",
        "side_name",
        "__archetype_label_family__",
        "policy_archetype",
        "local_side_archetype",
        "source_archetype",
    )
    train = pd.read_parquet(train_path, columns=[name for name in train_requested if name in train_available])
    valid = pd.read_parquet(valid_path, columns=[name for name in valid_requested if name in valid_available])
    metrics = pd.read_parquet(cache / "valid_metrics.parquet")
    if not (len(x_train) == len(train) and len(x_valid) == len(valid) == len(metrics)):
        raise RuntimeError(f"Fold {fold} cached X/label rows are misaligned")
    return x_train, x_valid, train, valid.join(metrics, rsuffix="__metric"), metrics


def _normalize_side(frame: pd.DataFrame) -> pd.Series:
    if "side_name" in frame.columns:
        side = frame["side_name"].astype(str).str.lower()
        return side.where(side.isin(("long", "short")), "unknown")
    values = pd.to_numeric(frame.get("side", 1), errors="coerce").fillna(1.0)
    return pd.Series(np.where(values < 0.0, "short", "long"), index=frame.index)


def _archetype(frame: pd.DataFrame) -> pd.Series:
    for column in ("__archetype_label_family__", "policy_archetype", "local_side_archetype", "source_archetype"):
        if column in frame.columns:
            values = frame[column].astype(str)
            return values.where(values.notna() & (values != ""), "unknown")
    return pd.Series("unknown", index=frame.index, dtype=object)


def _model_params(model: LGBMRegressor) -> dict[str, Any]:
    params = dict(model.get_params(deep=False))
    # The reference fit is a regressor.  Keep that API/objective parity while
    # replacing only the target and training loss weights.
    params["objective"] = "regression"
    params["verbosity"] = -1
    return params


def _top_indices(score: np.ndarray, fraction: float) -> np.ndarray:
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"Invalid top fraction {fraction}")
    finite = np.nan_to_num(np.asarray(score, dtype=np.float64), nan=-np.inf)
    k = min(len(finite), max(1, int(np.ceil(len(finite) * float(fraction)))))
    return np.argsort(-finite, kind="mergesort")[:k]


def _safe_mean(values: Iterable[Any]) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def _metric_row(
    selected: pd.DataFrame,
    *,
    arm: str,
    fold: str,
    top_fraction: float,
    grouping: str,
    group_value: str,
) -> dict[str, Any]:
    ret = pd.to_numeric(selected.get("ret_net", selected.get("__y_ret__")), errors="coerce")
    if ret.isna().all():
        ret = pd.to_numeric(selected.get("__y_ret__"), errors="coerce")
    full_mae = pd.to_numeric(
        selected.get("first_touch_full_path_mae_norm", selected.get("mae_norm")), errors="coerce"
    )
    timeout = pd.to_numeric(selected.get("is_timeout", 0.0), errors="coerce").fillna(0.0)
    clean = pd.to_numeric(
        selected.get("clean_first_touch_exec", selected.get("first_touch_clean_exec", np.nan)),
        errors="coerce",
    )
    stop = pd.to_numeric(selected.get("first_touch_stop", np.nan), errors="coerce")
    ts = pd.to_datetime(selected["__ts__"], utc=True, errors="coerce")
    day_count = max(int(ts.dt.normalize().nunique()), 1)
    week = ts.dt.tz_localize(None).dt.to_period("W-SUN").astype(str)
    weekly = pd.DataFrame({"week": week, "ret": ret}).groupby("week", dropna=False)["ret"].mean()
    return {
        "arm": arm,
        "fold": fold,
        "top_fraction": float(top_fraction),
        "grouping": grouping,
        "group_value": str(group_value),
        "selected_rows": int(len(selected)),
        "trades_per_day": float(len(selected) / day_count),
        "mean_ev_after_1pct": _safe_mean(ret),
        "sum_ev_after_1pct": float(np.nansum(ret.to_numpy(dtype=np.float64))),
        "positive_ev_rate": _safe_mean(ret > 0.0),
        "clean_exec_precision": _safe_mean(clean),
        "full_path_bad_mae_rate": _safe_mean(full_mae >= 1.0),
        "timeout_rate": _safe_mean(timeout > 0.5),
        "stop_or_adverse_rate": _safe_mean(stop > 0.5),
        "worst_week_mean_ev": float(weekly.min()) if len(weekly) else float("nan"),
        "score_net_spearman": float(pd.Series(selected["score"]).corr(ret, method="spearman")),
    }


def _metric_rows(
    scored: pd.DataFrame,
    *,
    arm: str,
    fold: str,
    top_fraction: float,
) -> list[dict[str, Any]]:
    idx = _top_indices(scored["score"].to_numpy(dtype=np.float64), top_fraction)
    selected = scored.iloc[idx].copy()
    rows = [_metric_row(selected, arm=arm, fold=fold, top_fraction=top_fraction, grouping="overall", group_value="all")]
    groupings = {
        "side": "side_name",
        "archetype": "archetype_label_family",
        "month": "month",
        "week": "week_start",
    }
    for grouping, column in groupings.items():
        for value, sub in selected.groupby(column, dropna=False, sort=True):
            rows.append(_metric_row(sub, arm=arm, fold=fold, top_fraction=top_fraction, grouping=grouping, group_value=str(value)))
    for (side, archetype), sub in selected.groupby(["side_name", "archetype_label_family"], dropna=False, sort=True):
        rows.append(_metric_row(sub, arm=arm, fold=fold, top_fraction=top_fraction, grouping="side_x_archetype", group_value=f"{side}__{archetype}"))
    return rows


def _fit_arm(
    *,
    x_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    train: pd.DataFrame,
    reference_model: LGBMRegressor,
    spec: TargetDesignSpec,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    target_reference = fit_target_reference(train, spec)
    target, target_meta = build_target(train, spec, reference=target_reference)
    reference = fit_training_weight_reference(train)
    weights, weight_meta = build_training_weights(train, spec, reference)
    params = _model_params(reference_model)
    # Use the saved fold seed to make ``current_first_touch_soft`` a genuine
    # parity arm.  Only use the CLI seed when the reference artifact has none.
    if params.get("random_state") is None:
        params["random_state"] = int(seed)
    model = LGBMRegressor(**params)
    model.fit(x_train, target, sample_weight=weights)
    return model.predict(x_valid).astype(np.float32, copy=False), {
        "target": target_meta,
        "target_reference": target_reference,
        "weights": weight_meta,
        "params": params,
        "target_mean": float(np.mean(target)),
        "target_std": float(np.std(target)),
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-report", type=Path, default=DEFAULT_BASE_REPORT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--folds", default=",".join(DEFAULT_FOLDS), help="Comma-separated cached OOS folds.")
    parser.add_argument("--top-fractions", default="0.10,0.20")
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--max-train-rows", type=int, default=0, help="Optional deterministic cap for smoke only; 0 keeps the full cached fit matrix.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    base_report = Path(args.base_report)
    folds = tuple(value.strip() for value in str(args.folds).split(",") if value.strip())
    top_fractions = tuple(float(value.strip()) for value in str(args.top_fractions).split(",") if value.strip())
    output_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S"))
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "status": "planned" if args.dry_run else "running",
        "base_report": str(base_report),
        "folds": list(folds),
        "top_fractions": list(top_fractions),
        "target_specs": [asdict(spec) for spec in DEFAULT_TARGET_SPECS],
        "contract": {
            "features": "frozen cached base feature matrices",
            "params": "reference base fold LightGBM params",
            "feature_selection": "not rerun",
            "hpo": "not rerun",
            "cost": "metrics use materialized net return once; target arms use already-net source",
            "weights": "fit only on each fold train rows",
            "oos": "cached chronological base valid rows",
        },
    }
    if args.dry_run:
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8")
        print(json.dumps(manifest, indent=2, default=_json_default))
        return

    metric_rows: list[dict[str, Any]] = []
    score_parts: list[pd.DataFrame] = []
    fold_meta: dict[str, Any] = {}
    for fold_index, fold in enumerate(folds):
        x_train, x_valid, train, valid, _metrics = _read_fold(base_report, fold)
        model_path = base_report / "models" / fold / "base_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing reference model for {fold}: {model_path}")
        reference_model = joblib.load(model_path)
        if list(x_train.columns) != list(x_valid.columns):
            raise RuntimeError(f"Fold {fold} train/valid feature schema mismatch")
        if int(args.max_train_rows) > 0 and len(x_train) > int(args.max_train_rows):
            # Chronological spread, not random sampling.  This is a smoke-only
            # option and is recorded in the manifest.
            take = np.linspace(0, len(x_train) - 1, int(args.max_train_rows)).round().astype(np.int64)
            x_train = x_train.iloc[take].reset_index(drop=True)
            train = train.iloc[take].reset_index(drop=True)

        valid = valid.copy()
        valid["side_name"] = _normalize_side(valid)
        valid["archetype_label_family"] = _archetype(valid)
        valid_ts = pd.to_datetime(valid["__ts__"], utc=True, errors="coerce").dt.tz_localize(None)
        valid["month"] = valid_ts.dt.to_period("M").astype(str)
        valid["week_start"] = valid_ts.dt.to_period("W-SUN").astype(str)
        fold_meta[fold] = {"train_rows": int(len(train)), "valid_rows": int(len(valid)), "model_path": str(model_path)}

        production_score = reference_model.predict(x_valid).astype(np.float32, copy=False)
        arm_scores: dict[str, np.ndarray] = {"production_frozen": production_score}
        arm_meta: dict[str, Any] = {"production_frozen": {"source": str(model_path)}}
        for spec in DEFAULT_TARGET_SPECS:
            score, metadata = _fit_arm(
                x_train=x_train,
                x_valid=x_valid,
                train=train,
                reference_model=reference_model,
                spec=spec,
                seed=int(args.seed + fold_index),
            )
            arm_scores[spec.name] = score
            arm_meta[spec.name] = metadata

        identity = valid.reindex(columns=[column for column in IDENTITY_COLUMNS if column in valid.columns]).copy()
        identity["fold"] = fold
        for arm, score in arm_scores.items():
            scored = valid.copy()
            scored["score"] = score
            for fraction in top_fractions:
                metric_rows.extend(_metric_rows(scored, arm=arm, fold=fold, top_fraction=fraction))
            part = identity.copy()
            part["arm"] = arm
            part["score"] = score
            score_parts.append(part)
        fold_meta[fold]["arms"] = arm_meta
        print(f"completed {fold}: train={len(train):,} valid={len(valid):,} arms={len(arm_scores)}")
        # Cached frames are large.  Release them before loading the next OOS
        # fold instead of relying on the allocator to recover memory at an
        # arbitrary later point in the loop.
        del x_train, x_valid, train, valid, _metrics
        del reference_model, production_score, arm_scores, arm_meta, identity
        gc.collect()

    metrics = pd.DataFrame(metric_rows)
    scores = pd.concat(score_parts, axis=0, ignore_index=True)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    scores.to_parquet(output_dir / "oos_scores.parquet", index=False)
    overall = metrics.loc[metrics["grouping"].eq("overall")].copy()
    baseline = overall.loc[overall["arm"].eq("production_frozen")].set_index(["fold", "top_fraction"])
    delta_rows: list[dict[str, Any]] = []
    for row in overall.to_dict(orient="records"):
        key = (row["fold"], row["top_fraction"])
        if row["arm"] == "production_frozen" or key not in baseline.index:
            continue
        base = baseline.loc[key]
        delta_rows.append(
            {
                "arm": row["arm"],
                "fold": row["fold"],
                "top_fraction": row["top_fraction"],
                "delta_mean_ev_after_1pct": row["mean_ev_after_1pct"] - base["mean_ev_after_1pct"],
                "delta_clean_exec_precision": row["clean_exec_precision"] - base["clean_exec_precision"],
                "delta_full_path_bad_mae_rate": row["full_path_bad_mae_rate"] - base["full_path_bad_mae_rate"],
                "delta_timeout_rate": row["timeout_rate"] - base["timeout_rate"],
            }
        )
    pd.DataFrame(delta_rows).to_csv(output_dir / "delta_vs_production.csv", index=False)
    manifest.update({"status": "complete", "fold_metadata": fold_meta, "output_files": ["metrics.csv", "oos_scores.parquet", "delta_vs_production.csv"]})
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8")
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
