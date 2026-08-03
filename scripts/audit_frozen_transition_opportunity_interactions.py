#!/usr/bin/env python3
"""Test frozen transition scores only through economic-context interactions.

March supplies resolved training rows and fixed risk thresholds.  April is a
single research diagnostic.  The audit compares a direct-score calibration,
context main effects, and predeclared active-transition interactions.  It does
not create a transition veto, exposure rule, or promotion candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MODEL = ROOT / (
    "data_perp/artifacts/canonical_raw_feature_direct_utility_multitask_"
    "20260729_v1"
)
DEFAULT_HEALTH = ROOT / (
    "data_perp/artifacts/historical_exact_model_health_failure_20260729_v3/"
    "hourly_exact_model_health_and_failure_labels.parquet"
)
DEFAULT_ACTIVE = ROOT / (
    "data_perp/artifacts/regime_transition_active_head_chronological_oos_"
    "20260729_v2/chronological_oos.parquet"
)
DEFAULT_DESTINATION = ROOT / (
    "data_perp/artifacts/regime_transition_destination_chronological_oos_"
    "20260729_v1/destination_chronological_oos.parquet"
)

WINNER_FEATURE_ARM = "base_transition_health"
WINNER_TASK_ARM = "economic_multitask"
RISK_COLUMNS = (
    "risk__low_opportunity",
    "risk__adverse",
    "risk__exit_conversion",
    "risk__timeout",
    "risk__negative_recent_health",
    "risk__recent_mapping_error",
    "risk__liquidity_cost",
    "risk__low_map_support",
    "risk__destination_uncertainty",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def corr(left: pd.Series, right: pd.Series) -> float:
    local = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(local) < 3 or local.left.nunique() < 2 or local.right.nunique() < 2:
        return np.nan
    value = spearmanr(local.left, local.right).statistic
    return float(value) if np.isfinite(value) else np.nan


def stable_top(frame: pd.DataFrame, score: str, fraction: float) -> pd.DataFrame:
    count = max(1, int(math.ceil(len(frame) * fraction)))
    order = np.lexsort(
        (
            frame["candidate_id"].astype(str).to_numpy(),
            -pd.to_numeric(frame[score], errors="raise").to_numpy(),
        )
    )
    return frame.iloc[order[:count]].copy()


def attach_frozen_context(
    predictions: pd.DataFrame,
    active: pd.DataFrame,
    destination: pd.DataFrame,
    health: pd.DataFrame,
) -> pd.DataFrame:
    work = predictions.copy()
    work["__ts__"] = pd.to_datetime(work["__ts__"], utc=True, errors="raise")
    active_local = active.loc[:, ["source_utc", "prediction"]].copy()
    active_local["source_utc"] = pd.to_datetime(
        active_local["source_utc"], utc=True, errors="raise"
    )
    active_local = active_local.drop_duplicates("source_utc", keep="last").rename(
        columns={"prediction": "active_transition_probability"}
    )
    work = work.merge(
        active_local,
        left_on="__ts__",
        right_on="source_utc",
        how="left",
        validate="many_to_one",
    ).drop(columns="source_utc")
    destination_columns = [
        "source_utc",
        "destination_entropy",
        "destination_confidence",
    ]
    destination_local = destination.loc[:, destination_columns].copy()
    destination_local["source_utc"] = pd.to_datetime(
        destination_local["source_utc"], utc=True, errors="raise"
    )
    destination_local = destination_local.drop_duplicates("source_utc", keep="last")
    work = work.merge(
        destination_local,
        left_on="__ts__",
        right_on="source_utc",
        how="left",
        validate="many_to_one",
    ).drop(columns="source_utc")
    health_columns = [
        "source_utc",
        "health__recent_resolved_net_ev_hl3d",
        "health__recent_resolved_mapping_error_hl3d",
        "health__recent_resolved_cost_bps_hl3d",
        "health__low_map_support_share",
    ]
    health_local = health.loc[:, health_columns].copy()
    health_local["source_utc"] = pd.to_datetime(
        health_local["source_utc"], utc=True, errors="raise"
    )
    health_local = health_local.drop_duplicates("source_utc", keep="last")
    work = work.merge(
        health_local,
        left_on="__ts__",
        right_on="source_utc",
        how="left",
        validate="many_to_one",
    ).drop(columns="source_utc")
    work["risk__low_opportunity"] = 1.0 - pd.to_numeric(
        work["diagnostic__opportunity"], errors="coerce"
    )
    work["risk__adverse"] = pd.to_numeric(
        work["diagnostic__adverse_magnitude"], errors="coerce"
    )
    work["risk__exit_conversion"] = pd.to_numeric(
        work["diagnostic__exit_conversion_loss"], errors="coerce"
    )
    work["risk__timeout"] = pd.to_numeric(
        work["diagnostic__timeout"], errors="coerce"
    )
    work["risk__negative_recent_health"] = -pd.to_numeric(
        work["health__recent_resolved_net_ev_hl3d"], errors="coerce"
    )
    work["risk__recent_mapping_error"] = pd.to_numeric(
        work["health__recent_resolved_mapping_error_hl3d"], errors="coerce"
    ).abs()
    work["risk__liquidity_cost"] = pd.to_numeric(
        work["health__recent_resolved_cost_bps_hl3d"], errors="coerce"
    )
    work["risk__low_map_support"] = pd.to_numeric(
        work["health__low_map_support_share"], errors="coerce"
    )
    work["risk__destination_uncertainty"] = pd.to_numeric(
        work["destination_entropy"], errors="coerce"
    )
    return work


def fit_side_local_ridge(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    alpha: float = 10.0,
) -> np.ndarray:
    prediction = pd.Series(np.nan, index=test.index, dtype=float)
    for side in ("long", "short"):
        train_side = train.loc[train.side_name.eq(side)].copy()
        test_side = test.loc[test.side_name.eq(side)].copy()
        train_x = train_side.loc[:, features].apply(pd.to_numeric, errors="coerce")
        test_x = test_side.loc[:, features].apply(pd.to_numeric, errors="coerce")
        median = train_x.median()
        train_x = train_x.fillna(median).fillna(0.0)
        test_x = test_x.fillna(median).fillna(0.0)
        scaler = StandardScaler().fit(train_x)
        model = Ridge(alpha=alpha).fit(
            scaler.transform(train_x),
            pd.to_numeric(train_side.execution_net_ev_12h, errors="raise"),
        )
        prediction.loc[test_side.index] = model.predict(scaler.transform(test_x))
    if prediction.isna().any():
        raise ValueError("side-local interaction audit left missing predictions")
    return prediction.to_numpy()


def threshold_flags(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_result = train.copy()
    test_result = test.copy()
    for side in ("long", "short"):
        train_side = train.loc[train.side_name.eq(side)]
        for column in ("active_transition_probability", *RISK_COLUMNS):
            threshold = float(
                pd.to_numeric(train_side[column], errors="coerce").quantile(0.80)
            )
            flag = f"high__{column}"
            train_result.loc[train_result.side_name.eq(side), flag] = (
                pd.to_numeric(
                    train_result.loc[train_result.side_name.eq(side), column],
                    errors="coerce",
                )
                .ge(threshold)
                .astype(float)
            )
            test_result.loc[test_result.side_name.eq(side), flag] = (
                pd.to_numeric(
                    test_result.loc[test_result.side_name.eq(side), column],
                    errors="coerce",
                )
                .ge(threshold)
                .astype(float)
            )
            train_result.loc[
                train_result.side_name.eq(side), f"threshold__{column}"
            ] = threshold
            test_result.loc[
                test_result.side_name.eq(side), f"threshold__{column}"
            ] = threshold
    return train_result, test_result


def conditional_interactions(april: pd.DataFrame) -> pd.DataFrame:
    rows = []
    active = april["high__active_transition_probability"].eq(1.0)
    for risk in RISK_COLUMNS:
        modifier = april[f"high__{risk}"].eq(1.0)
        cells = {}
        for active_value in (False, True):
            for modifier_value in (False, True):
                local = april.loc[
                    active.eq(active_value) & modifier.eq(modifier_value)
                ]
                cells[(active_value, modifier_value)] = (
                    float(local.execution_net_ev_12h.mean()) if len(local) else np.nan
                )
                rows.append(
                    {
                        "modifier": risk,
                        "metric": "cell",
                        "active_high": active_value,
                        "modifier_high": modifier_value,
                        "rows": int(len(local)),
                        "mean_net_bps": (
                            float(local.execution_net_ev_12h.mean() * 1e4)
                            if len(local)
                            else np.nan
                        ),
                        "difference_in_differences_bps": np.nan,
                    }
                )
        did = (
            cells[(True, True)]
            - cells[(True, False)]
            - cells[(False, True)]
            + cells[(False, False)]
        )
        rows.append(
            {
                "modifier": risk,
                "metric": "difference_in_differences",
                "active_high": None,
                "modifier_high": None,
                "rows": int(len(april)),
                "mean_net_bps": np.nan,
                "difference_in_differences_bps": did * 1e4,
            }
        )
    return pd.DataFrame(rows)


def evaluate_scores(april: pd.DataFrame, score_columns: list[str]) -> pd.DataFrame:
    rows = []
    for score in score_columns:
        rows.append(
            {
                "score": score,
                "fraction": 1.0,
                "rows": int(len(april)),
                "rank_ic": corr(april[score], april.execution_net_ev_12h),
                "mae": float(
                    np.mean(
                        np.abs(
                            pd.to_numeric(april[score], errors="raise")
                            - pd.to_numeric(
                                april.execution_net_ev_12h, errors="raise"
                            )
                        )
                    )
                ),
                "mean_net_bps": float(april.execution_net_ev_12h.mean() * 1e4),
                "long_share": float(april.side_name.eq("long").mean()),
            }
        )
        for fraction in (0.01, 0.05, 0.10, 0.20):
            selected = stable_top(april, score, fraction)
            rows.append(
                {
                    "score": score,
                    "fraction": fraction,
                    "rows": int(len(selected)),
                    "rank_ic": corr(
                        selected[score], selected.execution_net_ev_12h
                    ),
                    "mae": float(
                        np.mean(
                            np.abs(
                                selected[score]
                                - selected.execution_net_ev_12h
                            )
                        )
                    ),
                    "mean_net_bps": float(
                        selected.execution_net_ev_12h.mean() * 1e4
                    ),
                    "long_share": float(selected.side_name.eq("long").mean()),
                }
            )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    model_root = Path(args.model_root)
    health_path = Path(args.health)
    active_path = Path(args.active)
    destination_path = Path(args.destination)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    march = pd.read_parquet(model_root / "march_selection_predictions.parquet")
    march = march.loc[
        march.feature_arm.eq(WINNER_FEATURE_ARM)
        & march.task_arm.eq(WINNER_TASK_ARM)
    ].copy()
    march["execution_label_end_utc"] = pd.to_datetime(
        march["execution_label_end_utc"], utc=True, errors="raise"
    )
    march = march.loc[
        march.execution_label_end_utc.lt(pd.Timestamp("2025-04-01", tz="UTC"))
    ].copy()
    april = pd.read_parquet(
        model_root / "april_reused_diagnostic_predictions.parquet"
    )
    active = pd.read_parquet(active_path)
    destination = pd.read_parquet(destination_path)
    health = pd.read_parquet(health_path)
    march = attach_frozen_context(march, active, destination, health)
    april = attach_frozen_context(april, active, destination, health)
    march, april = threshold_flags(march, april)
    main_features = [
        "direct_net_score",
        "active_transition_probability",
        *RISK_COLUMNS,
    ]
    interaction_features = list(main_features)
    for risk in RISK_COLUMNS:
        column = f"interaction__active_x__{risk}"
        march[column] = march.active_transition_probability * march[risk]
        april[column] = april.active_transition_probability * april[risk]
        interaction_features.append(column)
    april["score__direct_calibrated"] = fit_side_local_ridge(
        march, april, ["direct_net_score"], alpha=float(args.ridge_alpha)
    )
    april["score__context_main"] = fit_side_local_ridge(
        march, april, main_features, alpha=float(args.ridge_alpha)
    )
    april["score__active_interactions"] = fit_side_local_ridge(
        march, april, interaction_features, alpha=float(args.ridge_alpha)
    )
    metrics = evaluate_scores(
        april,
        [
            "direct_net_score",
            "score__direct_calibrated",
            "score__context_main",
            "score__active_interactions",
        ],
    )
    conditional = conditional_interactions(april)
    threshold_columns = [
        "side_name",
        *[f"threshold__{name}" for name in ("active_transition_probability", *RISK_COLUMNS)],
    ]
    thresholds = (
        march.loc[:, threshold_columns]
        .groupby("side_name", observed=True)
        .first()
        .reset_index()
    )
    output.mkdir(parents=True, exist_ok=False)
    paths = {
        "metrics": output / "april_score_metrics.parquet",
        "conditional": output / "april_conditional_interactions.parquet",
        "thresholds": output / "march_frozen_thresholds.parquet",
        "april": output / "april_interaction_scores.parquet",
    }
    metrics.to_parquet(paths["metrics"], index=False, compression="zstd")
    conditional.to_parquet(paths["conditional"], index=False, compression="zstd")
    thresholds.to_parquet(paths["thresholds"], index=False, compression="zstd")
    april.to_parquet(paths["april"], index=False, compression="zstd")
    report = {
        "schema": "frozen_transition_opportunity_interaction_audit_v1",
        "status": "RESEARCH_DIAGNOSTIC_COMPLETE_NO_CONTROL_OR_PROMOTION",
        "calendar": {
            "march_resolved_training_rows": int(len(march)),
            "april_diagnostic_rows": int(len(april)),
        },
        "winner": {
            "feature_arm": WINNER_FEATURE_ARM,
            "task_arm": WINNER_TASK_ARM,
        },
        "models": {
            "side_local": True,
            "ridge_alpha_fixed": float(args.ridge_alpha),
            "hpo": False,
            "arms": [
                "direct calibration",
                "economic context main effects",
                "active-transition x economic-risk interactions",
            ],
        },
        "selection": (
            "one pooled-global April top1/5/10/20 per score with candidate-ID "
            "tie break; never timestamp/side quotas"
        ),
        "explicitly_forbidden": [
            "transition veto",
            "transition exposure reduction",
            "April model selection",
            "portfolio replay",
            "promotion claim",
        ],
        "sources": {
            "model_manifest": {
                "path": str((model_root / "manifest.json").resolve()),
                "sha256": sha256(model_root / "manifest.json"),
            },
            "health": {"path": str(health_path.resolve()), "sha256": sha256(health_path)},
            "active": {"path": str(active_path.resolve()), "sha256": sha256(active_path)},
            "destination": {
                "path": str(destination_path.resolve()),
                "sha256": sha256(destination_path),
            },
        },
        "outputs": {
            key: {"path": str(path.resolve()), "sha256": sha256(path)}
            for key, path in paths.items()
        },
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "promotion_eligible": False,
    }
    manifest = output / "manifest.json"
    manifest.write_text(
        json.dumps(safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "manifest.sha256").write_text(sha256(manifest) + "\n")
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model-root", type=Path, default=DEFAULT_MODEL)
    result.add_argument("--health", type=Path, default=DEFAULT_HEALTH)
    result.add_argument("--active", type=Path, default=DEFAULT_ACTIVE)
    result.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    result.add_argument("--ridge-alpha", type=float, default=10.0)
    result.add_argument("--output-dir", type=Path, required=True)
    return result


def main() -> None:
    print(json.dumps(safe(run(parser().parse_args())), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
