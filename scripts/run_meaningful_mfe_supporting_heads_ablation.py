#!/usr/bin/env python3
"""Train frozen-feature supporting heads and test causal incremental value."""

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
from scripts.run_meaningful_mfe_label_grid_ablation import (  # noqa: E402
    IDENTITY,
    SIDES,
    _available_expanding_month_folds,
)


SCHEMA = "meaningful_mfe_supporting_heads_ablation_v1"
SUPPORT_TARGETS = {
    "early_path_quality": "early_3bar_path_quality",
    "economic_barrier_time_quality": "economic_barrier_time_quality",
    "slope_quality": "__slope_quality__",
}
EVENT_SCORES = (
    "hard_probability",
    "soft_probability",
    "competing_favorable_probability",
)


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


def _support_metrics(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    target_column: str,
    *,
    head: str,
    scope: str,
) -> dict[str, Any]:
    target = frame[target_column].to_numpy(dtype=float)
    event = frame["favorable_first"].to_numpy(dtype=float)
    net = frame["execution_net_ev_12h"].to_numpy(dtype=float)
    count = max(1, int(np.ceil(0.10 * len(frame))))
    selected = np.argsort(-prediction, kind="mergesort")[:count]
    return {
        "head": head,
        "target": target_column,
        "scope": scope,
        "rows": int(len(frame)),
        "target_mean": float(target.mean()),
        "prediction_mean": float(prediction.mean()),
        "mae": float(np.mean(np.abs(prediction - target))),
        "spearman": float(pd.Series(prediction).corr(pd.Series(target), method="spearman")),
        "top10_target_mean": float(target[selected].mean()),
        "top10_favorable_first_rate": float(event[selected].mean()),
        "top10_mean_net_ev_bps": float(net[selected].mean() * 10_000.0),
    }


def _event_metrics(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    *,
    arm: str,
    scope: str,
) -> dict[str, Any]:
    target = frame["favorable_first"].to_numpy(dtype=np.int8)
    net = frame["execution_net_ev_12h"].to_numpy(dtype=float)
    count = max(1, int(np.ceil(0.10 * len(frame))))
    selected = np.argsort(-prediction, kind="mergesort")[:count]
    return {
        "arm": arm,
        "scope": scope,
        "evaluation": "july_forward_meta_oos",
        "rows": int(len(frame)),
        "auc": (
            float(roc_auc_score(target, prediction))
            if np.unique(target).size > 1
            else np.nan
        ),
        "average_precision": (
            float(average_precision_score(target, prediction))
            if np.unique(target).size > 1
            else np.nan
        ),
        "brier": float(brier_score_loss(target, prediction)),
        "top10_rows": int(count),
        "top10_favorable_first_rate": float(target[selected].mean()),
        "top10_mean_net_ev_bps": float(net[selected].mean() * 10_000.0),
        "top10_positive_net_rate": float((net[selected] > 0.0).mean()),
    }


def causal_incremental_stack(
    frame: pd.DataFrame,
    *,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit June side-local stacks and score July once on identical rows."""

    decision = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    resolution = pd.to_datetime(frame["label_resolution_utc"], utc=True, errors="raise")
    july_start = pd.Timestamp("2026-07-01T00:00:00Z")
    train_mask = (
        decision.ge(pd.Timestamp("2026-06-01T00:00:00Z"))
        & decision.lt(july_start - pd.Timedelta(hours=12))
        & resolution.lt(july_start)
    )
    evaluation_mask = decision.ge(july_start)
    arm_features = {
        "event_only": list(EVENT_SCORES),
        "event_plus_early_path": [*EVENT_SCORES, "pred_early_path_quality"],
        "event_plus_barrier_time": [
            *EVENT_SCORES,
            "pred_economic_barrier_time_quality",
        ],
        "event_plus_slope": [*EVENT_SCORES, "pred_slope_quality"],
        "event_plus_all_support": [
            *EVENT_SCORES,
            "pred_early_path_quality",
            "pred_economic_barrier_time_quality",
            "pred_slope_quality",
        ],
    }
    rows: list[pd.DataFrame] = []
    for arm_index, (arm, features) in enumerate(arm_features.items()):
        prediction = np.full(len(frame), np.nan)
        for side_index, side in enumerate(SIDES):
            train = train_mask & frame["side_name"].eq(side)
            valid = evaluation_mask & frame["side_name"].eq(side)
            if not train.any() or not valid.any():
                continue
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=0.25,
                    max_iter=2_000,
                    random_state=seed + 100 * arm_index + side_index,
                    solver="lbfgs",
                ),
            )
            model.fit(
                frame.loc[train, features],
                frame.loc[train, "favorable_first"].to_numpy(dtype=np.int8),
            )
            prediction[valid] = model.predict_proba(frame.loc[valid, features])[:, 1]
        valid = evaluation_mask.to_numpy() & np.isfinite(prediction)
        part = frame.loc[
            valid,
            [
                *IDENTITY,
                "label_resolution_utc",
                "favorable_first",
                "execution_net_ev_12h",
            ],
        ].copy()
        part["arm"] = arm
        part["stack_probability"] = prediction[valid]
        part["meta_train_start_utc"] = pd.Timestamp("2026-06-01T00:00:00Z")
        part["meta_train_end_exclusive_utc"] = july_start
        part["meta_evaluation_start_utc"] = july_start
        rows.append(part)
    predictions = pd.concat(rows, ignore_index=True)
    metric_rows: list[dict[str, Any]] = []
    for arm, sample in predictions.groupby("arm", sort=True):
        metric_rows.append(
            _event_metrics(
                sample,
                sample["stack_probability"].to_numpy(dtype=float),
                arm=arm,
                scope="pooled_global",
            )
        )
        for side in SIDES:
            side_sample = sample.loc[sample["side_name"].eq(side)]
            metric_rows.append(
                _event_metrics(
                    side_sample,
                    side_sample["stack_probability"].to_numpy(dtype=float),
                    arm=arm,
                    scope=f"side_{side}",
                )
            )
    metrics = pd.DataFrame(metric_rows)
    return predictions, metrics


def selection_replacement_economics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compare each pooled-global July top-10 selection with event-only."""

    identity = list(IDENTITY)
    by_arm = {
        arm: sample.copy()
        for arm, sample in predictions.groupby("arm", sort=True)
    }
    baseline = by_arm["event_only"]

    def selected(sample: pd.DataFrame) -> pd.DataFrame:
        count = max(1, int(np.ceil(0.10 * len(sample))))
        return sample.nlargest(count, "stack_probability", keep="first")

    base_selected = selected(baseline)
    base_ids = pd.MultiIndex.from_frame(base_selected[identity])
    base_ev = float(base_selected["execution_net_ev_12h"].mean() * 10_000.0)
    rows: list[dict[str, Any]] = []
    for arm, sample in by_arm.items():
        arm_selected = selected(sample)
        arm_ids = pd.MultiIndex.from_frame(arm_selected[identity])
        added = arm_selected.loc[~arm_ids.isin(base_ids)]
        dropped = base_selected.loc[~base_ids.isin(arm_ids)]
        arm_ev = float(arm_selected["execution_net_ev_12h"].mean() * 10_000.0)
        rows.append(
            {
                "arm": arm,
                "evaluation": "july_forward_meta_oos",
                "selection": "one_pooled_global_top10",
                "selected_rows": int(len(arm_selected)),
                "overlap_rows": int(arm_ids.isin(base_ids).sum()),
                "added_rows": int(len(added)),
                "dropped_rows": int(len(dropped)),
                "event_only_net_ev_bps": base_ev,
                "arm_net_ev_bps": arm_ev,
                "incremental_net_ev_bps": arm_ev - base_ev,
                "added_mean_net_ev_bps": (
                    float(added["execution_net_ev_12h"].mean() * 10_000.0)
                    if len(added)
                    else np.nan
                ),
                "dropped_mean_net_ev_bps": (
                    float(dropped["execution_net_ev_12h"].mean() * 10_000.0)
                    if len(dropped)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    grid = pd.read_parquet(args.label_grid)
    grid = grid.loc[grid["grid_name"].eq(args.grid_name) & grid["label_valid"]].copy()
    event = pd.read_parquet(args.event_predictions)
    event = event.loc[event["grid_name"].eq(args.grid_name)].copy()
    winner_summary = json.loads(args.winner_summary.read_text())
    required_grid = [
        *IDENTITY,
        "label_resolution_utc",
        "horizon_hours",
        "favorable_first",
        "execution_net_ev_12h",
        "early_3bar_path_quality",
        "economic_barrier_time_quality",
        "future_close_slope_atr_per_hour_clip_10",
    ]
    missing = [column for column in required_grid if column not in grid]
    if missing:
        raise ValueError("supporting label grid is missing: " + ", ".join(missing))
    grid["__slope_quality__"] = 1.0 / (
        1.0
        + np.exp(
            -np.clip(
                grid["future_close_slope_atr_per_hour_clip_10"].to_numpy(dtype=float)
                / 0.5,
                -40.0,
                40.0,
            )
        )
    )
    first = grid.copy()
    first["side"] = first["side_name"].astype(str)
    first["__label_end_ts__"] = pd.to_datetime(
        first["label_resolution_utc"], utc=True, errors="raise"
    )
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
    grid = grid.merge(
        enriched.loc[:, list(IDENTITY)].drop_duplicates(),
        on=list(IDENTITY),
        how="inner",
        validate="one_to_one",
    ).sort_values(["__ts__", "__symbol__", "side_name"], kind="mergesort")
    local_matrix = matrix.reindex(pd.MultiIndex.from_frame(grid.loc[:, list(IDENTITY)]))
    predictions = {
        f"pred_{head}": np.full(len(grid), np.nan)
        for head in SUPPORT_TARGETS
    }
    fold_rows: list[dict[str, Any]] = []
    for side_index, side in enumerate(SIDES):
        positions = np.flatnonzero(grid["side_name"].eq(side).to_numpy())
        side_frame = grid.iloc[positions].reset_index(drop=True)
        side_x = local_matrix.iloc[positions].reset_index(drop=True)
        features = [
            column
            for column in winner_summary["winners"][side]["soft"]["features"]
            if column in side_x
        ]
        params = winner_summary["winners"][side]["soft"]["geometry"]
        for fold in _available_expanding_month_folds(
            side_frame["__ts__"],
            side_frame["label_resolution_utc"],
            purge_hours=float(side_frame["horizon_hours"].iloc[0]),
        ):
            train = np.asarray(fold["train_indices"], dtype=int)
            valid = np.asarray(fold["validation_indices"], dtype=int)
            global_valid = positions[valid]
            for target_index, (head, target_column) in enumerate(SUPPORT_TARGETS.items()):
                model = _fit_catboost(
                    "soft",
                    side_x.iloc[train][features],
                    side_frame.iloc[train][target_column].to_numpy(dtype=float),
                    params,
                    seed=(
                        args.seed
                        + 100_000 * side_index
                        + 1_000 * int(fold["fold"])
                        + 100 * target_index
                    ),
                )
                predictions[f"pred_{head}"][global_valid] = _predict(
                    model, "soft", side_x.iloc[valid][features]
                )
            fold_rows.append(
                {
                    "side": side,
                    "fold": int(fold["fold"]),
                    "month": fold["month"],
                    "train_rows": int(len(train)),
                    "validation_rows": int(len(valid)),
                    "train_max_resolution_utc": side_frame.iloc[train]["label_resolution_utc"].max(),
                    "validation_start_utc": side_frame.iloc[valid]["__ts__"].min(),
                    "features": int(len(features)),
                }
            )
    support = grid.loc[:, required_grid + ["__slope_quality__"]].copy()
    for column, values in predictions.items():
        support[column] = values
    finite_support = support[list(predictions)].notna().all(axis=1)
    support_metrics: list[dict[str, Any]] = []
    for head, target_column in SUPPORT_TARGETS.items():
        sample = support.loc[finite_support].copy()
        prediction = sample[f"pred_{head}"].to_numpy(dtype=float)
        support_metrics.append(
            _support_metrics(
                sample, prediction, target_column, head=head, scope="pooled_global"
            )
        )
        for side in SIDES:
            mask = sample["side_name"].eq(side).to_numpy()
            support_metrics.append(
                _support_metrics(
                    sample.loc[mask],
                    prediction[mask],
                    target_column,
                    head=head,
                    scope=f"side_{side}",
                )
            )

    event_keep = [*IDENTITY, *EVENT_SCORES]
    stack_source = support.merge(
        event.loc[:, event_keep],
        on=list(IDENTITY),
        how="inner",
        validate="one_to_one",
    )
    stack_source = stack_source.loc[
        stack_source[[*EVENT_SCORES, *predictions]].notna().all(axis=1)
    ].reset_index(drop=True)
    stack_predictions, stack_metrics = causal_incremental_stack(
        stack_source, seed=args.seed + 500_000
    )
    replacement_metrics = selection_replacement_economics(stack_predictions)
    args.output_dir.mkdir(parents=True)
    paths = {
        "support_predictions": args.output_dir / "supporting_head_oof_predictions.parquet",
        "support_metrics": args.output_dir / "supporting_head_metrics.csv",
        "folds": args.output_dir / "supporting_head_folds.csv",
        "stack_predictions": args.output_dir / "incremental_stack_july_predictions.parquet",
        "stack_metrics": args.output_dir / "incremental_stack_july_metrics.csv",
        "replacement_metrics": args.output_dir
        / "incremental_stack_july_replacement_economics.csv",
    }
    support.to_parquet(paths["support_predictions"], index=False)
    pd.DataFrame(support_metrics).to_csv(paths["support_metrics"], index=False)
    pd.DataFrame(fold_rows).to_csv(paths["folds"], index=False)
    stack_predictions.to_parquet(paths["stack_predictions"], index=False)
    stack_metrics.to_csv(paths["stack_metrics"], index=False)
    replacement_metrics.to_csv(paths["replacement_metrics"], index=False)
    manifest = {
        "schema": SCHEMA,
        "status": "completed_research_oof_not_promotion_evidence",
        "contract": {
            "grid": args.grid_name,
            "features_hpo": "frozen April-authorized per-side soft-head winners; no supporting-label HPO",
            "support_heads": SUPPORT_TARGETS,
            "slope_transform": "sigmoid(fixed clip[-10,10] ATR/hour / 0.5)",
            "folds": "expanding monthly OOF with exact label resolution and horizon purge",
            "incremental_stack": "per-side fixed logistic C=0.25 fit on resolved June OOF predictions; one-shot July OOS",
            "economics": "one pooled global July top10 across timestamps and sides; cost already embedded once",
        },
        "inputs": {
            "label_grid": {"path": str(args.label_grid), "sha256": _sha256(args.label_grid)},
            "event_predictions": {"path": str(args.event_predictions), "sha256": _sha256(args.event_predictions)},
            "winner_summary": {"path": str(args.winner_summary), "sha256": _sha256(args.winner_summary)},
            "selection_sha256": feature_payload["selection_sha256"],
        },
        "outputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
    }
    _write_json(args.output_dir / "manifest.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label-grid",
        type=Path,
        default=Path("data_perp/artifacts/meaningful_mfe_label_grid_20260727_v2/meaningful_mfe_label_grid.parquet"),
    )
    parser.add_argument(
        "--event-predictions",
        type=Path,
        default=Path("data_perp/artifacts/meaningful_mfe_label_grid_ablation_20260727_v1/oof_predictions.parquet"),
    )
    parser.add_argument(
        "--winner-summary",
        type=Path,
        default=Path("data_perp/artifacts/meaningful_mfe_catboost_v2_ablation_20260725_v1/summary.json"),
    )
    parser.add_argument("--context-path", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--selection-path", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--grid-name", default="h12_u1p5atr")
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    print(json.dumps(run(_parser().parse_args()), indent=2, default=str))
