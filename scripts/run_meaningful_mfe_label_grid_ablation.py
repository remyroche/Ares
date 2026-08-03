#!/usr/bin/env python3
"""Run frozen-feature OOF models over the 12h/24h meaningful-MFE label grid."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_meaningful_mfe_catboost_v2_ablation import (  # noqa: E402
    DEFAULT_CONTEXT,
    DEFAULT_FEATURE_DIR,
    DEFAULT_SELECTION,
    _fit_catboost,
    _load_matrix,
    _predict,
)


SCHEMA = "meaningful_mfe_label_grid_ablation_v1"
IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")
SIDES = ("long", "short")


def _available_expanding_month_folds(
    decision: pd.Series,
    resolution: pd.Series,
    *,
    purge_hours: float,
) -> list[dict[str, Any]]:
    """Build all supported expanding folds; skip the leading no-history month."""

    decision = pd.to_datetime(decision, utc=True, errors="raise")
    resolution = pd.to_datetime(resolution, utc=True, errors="raise")
    months = pd.PeriodIndex(
        decision.dt.tz_localize(None).dt.to_period("M")
    ).unique().sort_values()
    folds: list[dict[str, Any]] = []
    for month in months:
        start = month.start_time.tz_localize("UTC")
        end = (month + 1).start_time.tz_localize("UTC")
        train = np.flatnonzero(
            (decision < start - pd.Timedelta(hours=purge_hours)).to_numpy()
            & (resolution < start).to_numpy()
        )
        valid = np.flatnonzero(
            (decision >= start).to_numpy() & (decision < end).to_numpy()
        )
        if not len(train) or not len(valid):
            continue
        folds.append(
            {
                "fold": len(folds) + 1,
                "month": str(month),
                "train_indices": train,
                "validation_indices": valid,
            }
        )
    return folds


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _metrics(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    *,
    model: str,
    scope: str,
) -> dict[str, Any]:
    hard = frame["favorable_first"].to_numpy(dtype=np.int8)
    soft = frame["soft_label"].to_numpy(dtype=float)
    net = frame["execution_net_ev_12h"].to_numpy(dtype=float)
    count = max(1, int(np.ceil(0.10 * len(frame))))
    selected = np.argsort(-prediction, kind="mergesort")[:count]
    return {
        "model": model,
        "scope": scope,
        "rows": int(len(frame)),
        "prevalence": float(hard.mean()),
        "auc": float(roc_auc_score(hard, prediction)) if np.unique(hard).size > 1 else np.nan,
        "average_precision": (
            float(average_precision_score(hard, prediction))
            if np.unique(hard).size > 1
            else np.nan
        ),
        "brier_hard": float(brier_score_loss(hard, prediction)),
        "log_loss_hard": float(log_loss(hard, np.clip(prediction, 1e-6, 1 - 1e-6))),
        "spearman_soft": float(pd.Series(prediction).corr(pd.Series(soft), method="spearman")),
        "top10_rows": int(count),
        "top10_precision": float(hard[selected].mean()),
        "top10_mean_net_ev_bps": float(net[selected].mean() * 10_000.0),
        "top10_positive_net_rate": float((net[selected] > 0.0).mean()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    labels = pd.read_parquet(args.label_grid)
    winner_summary = json.loads(args.winner_summary.read_text())
    required = [
        *IDENTITY,
        "grid_name",
        "label_resolution_utc",
        "label_valid",
        "soft_label",
        "favorable_first",
        "adverse_first",
        "timeout",
        "execution_net_ev_12h",
    ]
    missing = [column for column in required if column not in labels]
    if missing:
        raise ValueError("label grid is missing columns: " + ", ".join(missing))
    labels = labels.loc[labels["label_valid"]].copy()
    labels["__ts__"] = pd.to_datetime(labels["__ts__"], utc=True, errors="raise")
    labels["label_resolution_utc"] = pd.to_datetime(
        labels["label_resolution_utc"], utc=True, errors="raise"
    )
    grid_names = list(dict.fromkeys(labels["grid_name"].astype(str)))
    first = labels.loc[labels["grid_name"].eq(grid_names[0])].copy()
    first["side"] = first["side_name"].astype(str)
    first["__label_end_ts__"] = first["label_resolution_utc"]
    enriched, matrix, feature_payload = _load_matrix(
        first,
        context_path=args.context_path,
        feature_dir=args.feature_dir,
        selection_path=args.selection_path,
        archetype_contract_override=winner_summary["feature_contract"][
            "archetype_contract"
        ],
    )
    enriched["side_name"] = enriched["side"].astype(str)
    matrix.index = pd.MultiIndex.from_frame(enriched.loc[:, list(IDENTITY)])
    prediction_parts: list[pd.DataFrame] = []
    fold_reports: list[dict[str, Any]] = []
    for grid_index, grid_name in enumerate(grid_names):
        grid = labels.loc[labels["grid_name"].eq(grid_name)].copy()
        grid = grid.merge(
            enriched.loc[:, [*IDENTITY]].drop_duplicates(),
            on=list(IDENTITY),
            how="inner",
            validate="one_to_one",
        )
        grid = grid.sort_values(["__ts__", "__symbol__", "side_name"], kind="mergesort")
        grid_matrix = matrix.reindex(pd.MultiIndex.from_frame(grid.loc[:, list(IDENTITY)]))
        if grid_matrix.isna().all(axis=1).any():
            raise ValueError(f"{grid_name} could not recover every frozen feature row")
        components = {
            "hard_probability": np.full(len(grid), np.nan),
            "soft_probability": np.full(len(grid), np.nan),
            "competing_favorable_probability": np.full(len(grid), np.nan),
        }
        fold_id = np.full(len(grid), -1, dtype=np.int16)
        for side_index, side in enumerate(SIDES):
            positions = np.flatnonzero(grid["side_name"].astype(str).eq(side).to_numpy())
            local = grid.iloc[positions].reset_index(drop=True)
            local_x = grid_matrix.iloc[positions].reset_index(drop=True)
            winner = winner_summary["winners"][side]
            hard_features = [
                column for column in winner["hard"]["features"] if column in local_x
            ]
            soft_features = [
                column for column in winner["soft"]["features"] if column in local_x
            ]
            if not hard_features or not soft_features:
                raise ValueError(f"{grid_name}/{side} has no frozen winner features")
            risk_class = np.select(
                [
                    local["timeout"].to_numpy(dtype=bool),
                    local["adverse_first"].to_numpy(dtype=bool),
                    local["favorable_first"].to_numpy(dtype=bool),
                ],
                [0, 1, 2],
                default=-1,
            )
            for fold in _available_expanding_month_folds(
                local["__ts__"],
                local["label_resolution_utc"],
                purge_hours=float(local["horizon_hours"].iloc[0]),
            ):
                train = np.asarray(fold["train_indices"], dtype=int)
                valid = np.asarray(fold["validation_indices"], dtype=int)
                global_valid = positions[valid]
                hard_model = _fit_catboost(
                    "binary",
                    local_x.iloc[train][hard_features],
                    local.iloc[train]["favorable_first"].to_numpy(dtype=float),
                    winner["hard"]["geometry"],
                    seed=args.seed + 100_000 * grid_index + 10_000 * side_index + int(fold["fold"]),
                )
                soft_model = _fit_catboost(
                    "soft",
                    local_x.iloc[train][soft_features],
                    local.iloc[train]["soft_label"].to_numpy(dtype=float),
                    winner["soft"]["geometry"],
                    seed=args.seed + 100_000 * grid_index + 10_000 * side_index + 100 + int(fold["fold"]),
                )
                competing_model = _fit_catboost(
                    "multiclass",
                    local_x.iloc[train][hard_features],
                    risk_class[train],
                    winner["hard"]["geometry"],
                    seed=args.seed + 100_000 * grid_index + 10_000 * side_index + 200 + int(fold["fold"]),
                )
                components["hard_probability"][global_valid] = _predict(
                    hard_model, "binary", local_x.iloc[valid][hard_features]
                )
                components["soft_probability"][global_valid] = _predict(
                    soft_model, "soft", local_x.iloc[valid][soft_features]
                )
                components["competing_favorable_probability"][global_valid] = _predict(
                    competing_model, "multiclass", local_x.iloc[valid][hard_features]
                )[:, 2]
                fold_id[global_valid] = int(fold["fold"])
                fold_reports.append(
                    {
                        "grid_name": grid_name,
                        "side": side,
                        "fold": int(fold["fold"]),
                        "train_rows": int(len(train)),
                        "validation_rows": int(len(valid)),
                        "train_max_resolution_utc": local.iloc[train]["label_resolution_utc"].max(),
                        "validation_start_utc": local.iloc[valid]["__ts__"].min(),
                        "hard_features": len(hard_features),
                        "soft_features": len(soft_features),
                    }
                )
        output = grid.loc[:, required].copy()
        output["oof_fold"] = fold_id
        for name, values in components.items():
            output[name] = values
        prediction_parts.append(output)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    metric_rows: list[dict[str, Any]] = []
    for grid_name, grid in predictions.groupby("grid_name", sort=True):
        finite = grid[
            [
                "hard_probability",
                "soft_probability",
                "competing_favorable_probability",
            ]
        ].notna().all(axis=1)
        sample = grid.loc[finite].copy()
        for model in (
            "hard_probability",
            "soft_probability",
            "competing_favorable_probability",
        ):
            prediction = sample[model].to_numpy(dtype=float)
            metric_rows.append(
                {
                    "grid_name": grid_name,
                    **_metrics(sample, prediction, model=model, scope="pooled_global"),
                }
            )
            for side in SIDES:
                mask = sample["side_name"].eq(side).to_numpy()
                metric_rows.append(
                    {
                        "grid_name": grid_name,
                        **_metrics(
                            sample.loc[mask],
                            prediction[mask],
                            model=model,
                            scope=f"side_{side}",
                        ),
                    }
                )
    metrics = pd.DataFrame(metric_rows)
    args.output_dir.mkdir(parents=True)
    prediction_path = args.output_dir / "oof_predictions.parquet"
    metrics_path = args.output_dir / "metrics.csv"
    folds_path = args.output_dir / "folds.csv"
    predictions.to_parquet(prediction_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    pd.DataFrame(fold_reports).to_csv(folds_path, index=False)
    manifest = {
        "schema": SCHEMA,
        "status": "completed_research_oof_not_promotion_evidence",
        "contract": {
            "feature_selection_hpo": "frozen per-side hard/soft winners from the prior April-authorized v2 study; no label-grid retuning",
            "folds": "expanding monthly OOF with each grid's true 12h/24h resolution timestamps",
            "ranking": "one pooled global top10 across timestamps and sides for aggregate economics; side scopes are diagnostics",
            "cost": "execution_net_ev_12h already contains cost; no second subtraction",
            "use": "label learnability comparison only; not a trade admission score",
        },
        "inputs": {
            "label_grid": str(args.label_grid),
            "label_grid_sha256": _sha256(args.label_grid),
            "winner_summary": str(args.winner_summary),
            "winner_summary_sha256": _sha256(args.winner_summary),
            "selection": str(args.selection_path),
            "selection_sha256": feature_payload["selection_sha256"],
        },
        "grid_names": grid_names,
        "outputs": {
            "predictions": {"path": str(prediction_path), "sha256": _sha256(prediction_path)},
            "metrics": {"path": str(metrics_path), "sha256": _sha256(metrics_path)},
            "folds": {"path": str(folds_path), "sha256": _sha256(folds_path)},
        },
    }
    _write_json(args.output_dir / "manifest.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-grid", type=Path, required=True)
    parser.add_argument(
        "--winner-summary",
        type=Path,
        default=Path("data_perp/artifacts/meaningful_mfe_catboost_v2_ablation_20260725_v1/summary.json"),
    )
    parser.add_argument("--context-path", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--selection-path", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    print(json.dumps(run(_parser().parse_args()), indent=2, default=str))
