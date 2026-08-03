#!/usr/bin/env python3
"""Cross-era transfer diagnostic for exact economic-failure labels.

The historical side is a reconstructed execution architecture, not a backcast
of the current model.  The forward historical->current test is therefore a
semantic-transfer diagnostic over explicitly common health fields.  The
current->historical direction is reported only as a non-causal reverse
diagnostic.  No result from this script is promotion eligible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)

from run_historical_exact_model_failure_ablation import (
    _model,
    event_operating_curve,
    grouped_oof,
    market_feature_columns,
    risk_tail_economics,
)


DEFAULT_HISTORICAL = (
    "data_perp/artifacts/"
    "historical_reconstructed_execution_failure_labels_20260729_v2/"
    "hourly_current_health_and_failure_labels.parquet"
)
DEFAULT_CURRENT = (
    "data_perp/artifacts/"
    "current_lineage_exact_failure_labels_resolved_july19_20260729_v1/"
    "hourly_current_health_and_failure_labels.parquet"
)
DEFAULT_CATALOG = (
    "data_perp/artifacts/"
    "historical_reconstructed_execution_health_20260729_v1/"
    "common_health_catalog.csv"
)
DEFAULT_MARKET = (
    "data_perp/artifacts/regime_transition_research_20260726_v3/"
    "hourly_transition_dataset.parquet"
)
DEFAULT_ACTIVE = (
    "data_perp/artifacts/"
    "regime_transition_active_head_chronological_oos_20260729_v2/"
    "chronological_oos.parquet"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, (Path, pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def common_health_columns(catalog: pd.DataFrame) -> list[str]:
    required = {"feature", "cross_era_common"}
    if not required.issubset(catalog.columns):
        raise ValueError(f"catalog missing {sorted(required - set(catalog))}")
    selected = catalog.loc[
        catalog["cross_era_common"].astype(bool), "feature"
    ].astype(str).tolist()
    prohibited = {
        "health__alpha_uncertainty_mean",
        "health__catboost_entropy_mean",
    }
    if prohibited.intersection(selected):
        raise AssertionError("non-comparable health fields entered common set")
    return selected


def causal_trailing_robust_z(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    lookback_days: int = 21,
    min_history_hours: int = 72,
) -> tuple[pd.DataFrame, list[str]]:
    """Add within-era robust z-scores using strictly earlier observations."""
    work = frame.sort_values(["era", "source_utc"]).copy()
    output_columns = [f"health_z__{name.removeprefix('health__')}" for name in columns]
    for output_column in output_columns:
        work[output_column] = np.nan
    for _, indices in work.groupby("era", sort=False).groups.items():
        local = work.loc[indices].sort_values("source_utc")
        timestamp = pd.DatetimeIndex(local["source_utc"])
        for source_column, output_column in zip(columns, output_columns):
            values = pd.Series(
                pd.to_numeric(local[source_column], errors="coerce").to_numpy(float),
                index=timestamp,
            )
            rolling = values.rolling(
                f"{int(lookback_days)}D",
                closed="left",
                min_periods=int(min_history_hours),
            )
            median = rolling.median()
            q25 = rolling.quantile(0.25)
            q75 = rolling.quantile(0.75)
            scale = (q75 - q25) / 1.349
            standardized = (values - median) / scale.where(scale.abs() > 1e-12)
            work.loc[local.index, output_column] = standardized.to_numpy(float)
    return work.sort_index(), output_columns


def add_active_interactions(
    frame: pd.DataFrame, normalized_health: Sequence[str]
) -> tuple[pd.DataFrame, list[str]]:
    work = frame.copy()
    preferred_suffixes = (
        "mapped_ev_std",
        "base_residual_rank_abs_gap",
        "mapped_ev_coverage",
        "recent_resolved_net_ev_hl3d",
        "recent_resolved_hit_rate_hl3d",
        "recent_resolved_mapping_error_hl3d",
        "recent_resolved_cost_bps_hl3d",
    )
    lookup = {name.removeprefix("health_z__"): name for name in normalized_health}
    interaction_columns: list[str] = []
    active = pd.to_numeric(
        work["active_transition_probability_oos"], errors="coerce"
    )
    for suffix in preferred_suffixes:
        if suffix not in lookup:
            continue
        name = f"interaction__active_x__{suffix}"
        work[name] = active * pd.to_numeric(work[lookup[suffix]], errors="coerce")
        interaction_columns.append(name)
    return work, interaction_columns


def _metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {
        "rows": int(len(y)),
        "positive_rows": int(y.sum()),
        "prevalence": float(y.mean()) if len(y) else np.nan,
        "brier": float(brier_score_loss(y, prediction)) if len(y) else np.nan,
        "f1_at_0_5": float(
            f1_score(y, prediction >= 0.5, zero_division=0)
        ) if len(y) else np.nan,
    }
    if len(np.unique(y)) == 2:
        result["pr_auc"] = float(average_precision_score(y, prediction))
        result["roc_auc"] = float(roc_auc_score(y, prediction))
    else:
        result["pr_auc"] = np.nan
        result["roc_auc"] = np.nan
    return result


def fit_transfer(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    *,
    features: Sequence[str],
    target_column: str,
    seed: int,
) -> np.ndarray:
    train_y = train[target_column].astype(int).to_numpy()
    if len(np.unique(train_y)) != 2:
        raise ValueError("transfer training data must contain both classes")
    model = _model(int(seed))
    model.fit(
        train[list(features)].apply(pd.to_numeric, errors="coerce"),
        train_y,
    )
    return model.predict_proba(
        evaluation[list(features)].apply(pd.to_numeric, errors="coerce")
    )[:, 1]


def _load_panel(
    path: Path,
    *,
    era: str,
    exact_current_lineage: bool,
) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["source_utc"] = pd.to_datetime(
        frame["source_utc"], utc=True, errors="raise"
    )
    if frame["source_utc"].duplicated().any():
        raise ValueError(f"{era} health has duplicate source hours")
    frame = frame.loc[frame["label_window_complete"].astype(bool)].copy()
    frame["era"] = era
    frame["exact_current_lineage"] = bool(exact_current_lineage)
    return frame


def run(args: argparse.Namespace) -> dict[str, Any]:
    historical_path = Path(args.historical)
    current_path = Path(args.current)
    catalog_path = Path(args.catalog)
    market_path = Path(args.market)
    active_path = Path(args.active_oos)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")

    historical = _load_panel(
        historical_path,
        era="historical_reconstructed",
        exact_current_lineage=False,
    )
    current = _load_panel(
        current_path,
        era="current_exact",
        exact_current_lineage=True,
    )
    catalog = pd.read_csv(catalog_path)
    common_health = common_health_columns(catalog)
    for name, local in (("historical", historical), ("current", current)):
        missing = sorted(set(common_health) - set(local))
        if missing:
            raise ValueError(f"{name} panel missing common fields: {missing}")

    market = pd.read_parquet(market_path)
    market["source_utc"] = pd.to_datetime(
        market["source_utc"], utc=True, errors="raise"
    )
    active = pd.read_parquet(active_path).rename(
        columns={"prediction": "active_transition_probability_oos"}
    )
    active["source_utc"] = pd.to_datetime(
        active["source_utc"], utc=True, errors="raise"
    )
    shared_context = market.merge(
        active[["source_utc", "active_transition_probability_oos"]],
        on="source_utc",
        how="inner",
        validate="one_to_one",
    )
    frame = pd.concat([historical, current], ignore_index=True).merge(
        shared_context,
        on="source_utc",
        how="inner",
        validate="many_to_one",
        suffixes=("", "__market"),
    )
    if frame.empty:
        raise ValueError("no cross-era context overlap")
    frame, normalized_health = causal_trailing_robust_z(
        frame,
        common_health,
        lookback_days=int(args.lookback_days),
        min_history_hours=int(args.min_history_hours),
    )
    frame, interactions = add_active_interactions(frame, normalized_health)
    market_features = market_feature_columns(market)
    feature_sets = {
        "market_only": market_features,
        "common_health_raw_only": common_health,
        "common_health_causal_z_only": normalized_health,
        "market_plus_common_health_raw": market_features + common_health,
        "market_plus_common_health_causal_z": market_features + normalized_health,
        "market_plus_causal_z_plus_active": market_features
        + normalized_health
        + ["active_transition_probability_oos"],
        "market_plus_causal_z_plus_active_interactions": market_features
        + normalized_health
        + ["active_transition_probability_oos"]
        + interactions,
    }

    output.mkdir(parents=True, exist_ok=False)
    metric_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    operating_rows: list[dict[str, Any]] = []
    tail_rows: list[dict[str, Any]] = []
    predictions = frame[
        [
            "source_utc",
            "era",
            "exact_current_lineage",
            "post_12h_net_ev_mean",
            "post_minus_pre_mapping_residual",
            "active_transition_probability_oos",
            "target__economic_failure_broad_active",
            "target__economic_failure_broad_event_id",
            "target__economic_failure_strict_active",
            "target__economic_failure_strict_event_id",
        ]
    ].copy()
    fold_provenance: dict[str, Any] = {}

    directions = (
        ("historical_to_current", "historical_reconstructed", "current_exact", True),
        ("current_to_historical_reverse", "current_exact", "historical_reconstructed", False),
    )
    for label_index, label in enumerate(("broad", "strict")):
        target = f"target__economic_failure_{label}_active"
        event = f"target__economic_failure_{label}_event_id"
        for set_index, (feature_set, features) in enumerate(feature_sets.items()):
            for direction, train_era, evaluation_era, causal_direction in directions:
                train = frame.loc[frame["era"].eq(train_era)].copy()
                evaluation = frame.loc[frame["era"].eq(evaluation_era)].copy()
                prediction = fit_transfer(
                    train,
                    evaluation,
                    features=features,
                    target_column=target,
                    seed=int(args.seed) + 1000 * label_index + 20 * set_index,
                )
                column = f"prediction__{label}__{feature_set}__{direction}"
                predictions.loc[evaluation.index, column] = prediction
                y = evaluation[target].astype(int).to_numpy()
                metric_rows.append(
                    {
                        "failure_label": label,
                        "validation": direction,
                        "causal_direction": causal_direction,
                        "train_era": train_era,
                        "evaluation_era": evaluation_era,
                        "feature_set": feature_set,
                        "feature_count": len(features),
                        **_metrics(y, prediction),
                    }
                )
                for row in event_operating_curve(
                    evaluation,
                    prediction,
                    target_column=target,
                    event_column=event,
                ):
                    operating_rows.append(
                        {
                            "failure_label": label,
                            "validation": direction,
                            "feature_set": feature_set,
                            **row,
                        }
                    )
                tail_rows.append(
                    {
                        "failure_label": label,
                        "validation": direction,
                        "feature_set": feature_set,
                        **risk_tail_economics(
                            evaluation, prediction, target_column=target
                        ),
                    }
                )
                local_output = evaluation.assign(prediction=prediction)
                for month, local in local_output.groupby(
                    local_output["source_utc"].dt.strftime("%Y-%m"), sort=True
                ):
                    monthly_rows.append(
                        {
                            "failure_label": label,
                            "validation": direction,
                            "feature_set": feature_set,
                            "month": month,
                            **_metrics(
                                local[target].astype(int).to_numpy(),
                                local["prediction"].to_numpy(float),
                            ),
                        }
                    )

            for era_index, era in enumerate(("historical_reconstructed", "current_exact")):
                local = frame.loc[frame["era"].eq(era)].copy()
                prediction, provenance, _ = grouped_oof(
                    local,
                    features=features,
                    target_column=target,
                    event_column=event,
                    folds=int(args.folds),
                    seed=int(args.seed)
                    + 1000 * label_index
                    + 20 * set_index
                    + era_index,
                )
                validation = f"within_{era}_grouped_oof"
                predictions.loc[local.index, f"prediction__{label}__{feature_set}__{validation}"] = prediction
                metric_rows.append(
                    {
                        "failure_label": label,
                        "validation": validation,
                        "causal_direction": False,
                        "train_era": era,
                        "evaluation_era": era,
                        "feature_set": feature_set,
                        "feature_count": len(features),
                        **_metrics(local[target].astype(int).to_numpy(), prediction),
                    }
                )
                fold_provenance[f"{label}__{feature_set}__{validation}"] = provenance

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(monthly_rows).to_csv(output / "monthly_metrics.csv", index=False)
    pd.DataFrame(operating_rows).to_csv(
        output / "event_operating_curve.csv", index=False
    )
    pd.DataFrame(tail_rows).to_csv(
        output / "risk_tail_economics.csv", index=False
    )
    predictions.to_parquet(
        output / "predictions.parquet", index=False, compression="zstd"
    )
    _write_json(output / "fold_provenance.json", fold_provenance)

    forward = metrics.loc[metrics["validation"].eq("historical_to_current")]
    winners = []
    for label, local in forward.groupby("failure_label", sort=True):
        winner = local.sort_values(["pr_auc", "brier"], ascending=[False, True]).iloc[0]
        baseline = local.loc[local["feature_set"].eq("market_only")].iloc[0]
        winners.append(
            {
                "failure_label": label,
                "feature_set": winner["feature_set"],
                "pr_auc": winner["pr_auc"],
                "market_only_pr_auc": baseline["pr_auc"],
                "delta_pr_auc_vs_market_only": winner["pr_auc"] - baseline["pr_auc"],
                "brier": winner["brier"],
            }
        )
    source_paths = {
        "historical": historical_path,
        "current": current_path,
        "catalog": catalog_path,
        "market": market_path,
        "active_oos": active_path,
    }
    manifest = {
        "schema": "cross_era_execution_failure_transfer_v1",
        "status": "CROSS_ERA_DIAGNOSTIC_COMPLETE",
        "promotion_eligible": False,
        "lineage_contract": {
            "historical": (
                "historical reconstructed execution architecture; exact labels "
                "for that architecture; not a current-model backcast"
            ),
            "current": (
                "exact current execution-EV lineage, including resolved retired "
                "forward rows used only as evaluation in the causal direction"
            ),
        },
        "validation_contract": {
            "historical_to_current": (
                "causal temporal and semantic transfer diagnostic"
            ),
            "current_to_historical_reverse": (
                "non-causal reverse diagnostic only"
            ),
            "within_era": (
                "grouped OOF research baseline; not walk-forward or promotion evidence"
            ),
        },
        "joined_coverage": {
            era: {
                "rows": len(local),
                "start_utc": local["source_utc"].min(),
                "end_utc": local["source_utc"].max(),
            }
            for era, local in frame.groupby("era", sort=True)
        },
        "context_coverage_limit": (
            "joined evaluation ends at the minimum market/active-context endpoint; "
            "hours after that endpoint are not claimed as evaluated"
        ),
        "common_health_feature_count": len(common_health),
        "common_health_features": common_health,
        "excluded_non_comparable_fields": [
            "health__alpha_uncertainty_mean",
            "health__catboost_entropy_mean",
        ],
        "causal_health_normalization": {
            "lookback_days": int(args.lookback_days),
            "min_history_hours": int(args.min_history_hours),
            "closed": "left",
            "scope": "within era",
        },
        "feature_sets": {name: len(columns) for name, columns in feature_sets.items()},
        "forward_winners": winners,
        "sources": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in source_paths.items()
        },
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--historical", default=DEFAULT_HISTORICAL)
    value.add_argument("--current", default=DEFAULT_CURRENT)
    value.add_argument("--catalog", default=DEFAULT_CATALOG)
    value.add_argument("--market", default=DEFAULT_MARKET)
    value.add_argument("--active-oos", default=DEFAULT_ACTIVE)
    value.add_argument("--output-dir", required=True)
    value.add_argument("--lookback-days", type=int, default=21)
    value.add_argument("--min-history-hours", type=int, default=72)
    value.add_argument("--folds", type=int, default=4)
    value.add_argument("--seed", type=int, default=20260729)
    return value


if __name__ == "__main__":
    print(json.dumps(_safe(run(parser().parse_args())), indent=2, sort_keys=True))
