#!/usr/bin/env python3
"""Evaluate rolling-origin archetype residual states and requested placebo tests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_ROOT = ROOT / "data_perp/reports/global_residual_state_discovery_20260711_v1"
DEFAULT_PREDICTIONS = (
    DEFAULT_ROOT / "global_side_latent_states/rolling_origin_state_predictions.parquet"
)

STATE_MODEL_SPECS = (
    (
        "gmm",
        "global_state_id",
        "global_state_expected_negative_ev",
        "global_state_expected_positive_surprise",
    ),
    (
        "hmm",
        "global_hmm_state_id",
        "global_hmm_expected_negative_ev",
        "global_hmm_expected_positive_surprise",
    ),
)


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _auprc(target: pd.Series, score: pd.Series) -> float:
    target = pd.to_numeric(target, errors="coerce")
    score = pd.to_numeric(score, errors="coerce")
    valid = target.notna() & score.notna()
    if valid.sum() < 20 or target[valid].nunique() < 2:
        return np.nan
    return float(average_precision_score(target[valid].astype(int), score[valid]))


def _recognition_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups = ["oos_month", "side_name", "archetype_policy_key"]
    for keys, local in frame.groupby(groups, observed=True, sort=True):
        for state_model, _, risk_col, opportunity_col in STATE_MODEL_SPECS:
            if risk_col not in local or opportunity_col not in local:
                continue
            risk = pd.to_numeric(local[risk_col], errors="coerce")
            opportunity = pd.to_numeric(local[opportunity_col], errors="coerce")
            negative = pd.to_numeric(local["target_negative_ev"], errors="coerce").gt(
                0.0
            )
            positive = pd.to_numeric(
                local["target_positive_surprise"], errors="coerce"
            ).gt(0.0)
            top_risk = risk.ge(risk.quantile(0.95))
            top_opportunity = opportunity.ge(opportunity.quantile(0.95))
            weights = pd.to_numeric(local["selected_rows"], errors="coerce").fillna(0.0)
            rows.append(
                {
                    "oos_month": keys[0],
                    "side_name": keys[1],
                    "archetype_policy_key": keys[2],
                    "state_model": state_model,
                    "hours": len(local),
                    "selected_rows": int(weights.sum()),
                    "negative_ev_auprc": _auprc(negative, risk),
                    "positive_surprise_auprc": _auprc(positive, opportunity),
                    "negative_ev_precision_top5pct": float(negative[top_risk].mean()),
                    "positive_surprise_precision_top5pct": float(
                        positive[top_opportunity].mean()
                    ),
                    "mean_ev": float(
                        np.average(
                            pd.to_numeric(
                                local["target_mean_ev"], errors="coerce"
                            ).fillna(0),
                            weights=np.maximum(weights, 1e-8),
                        )
                    ),
                    "top_opportunity_ev": float(
                        pd.to_numeric(
                            local.loc[top_opportunity, "target_mean_ev"],
                            errors="coerce",
                        ).mean()
                    ),
                    "top_risk_ev": float(
                        pd.to_numeric(
                            local.loc[top_risk, "target_mean_ev"], errors="coerce"
                        ).mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def _event_recognition(
    frame: pd.DataFrame, root: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    membership = pd.read_parquet(root / "unreliability_event_membership.parquet")
    events = pd.read_csv(root / "unreliability_event_catalog.csv")
    high_priority = events.loc[
        events["event_class"].isin(["adverse", "payoff_disagreement"])
        & events["discovery_eligible"].astype(bool)
        & events["bootstrap_survival"].astype(bool)
        & events["bh_surviving_cells"].gt(0)
    ].nlargest(25, "adverse_priority", keep="first")
    membership = membership[membership["event_id"].isin(high_priority["event_id"])]
    membership["day"] = pd.to_datetime(membership["day"], utc=True)
    side_day = (
        membership.groupby(["day", "side_name", "archetype_policy_key"], observed=True)[
            "event_id"
        ]
        .agg(lambda values: "|".join(sorted(set(map(str, values)))))
        .rename("event_ids")
        .reset_index()
    )
    work = frame.copy()
    work["day"] = pd.to_datetime(work["__ts__"], utc=True).dt.floor("D")
    work = work.merge(
        side_day,
        on=["day", "side_name", "archetype_policy_key"],
        how="left",
        validate="many_to_one",
    )
    work["is_event"] = work["event_ids"].notna()
    rows: list[dict[str, Any]] = []
    for keys, local in work.groupby(
        ["oos_month", "side_name", "archetype_policy_key"], observed=True
    ):
        for state_model, _, risk_col, _ in STATE_MODEL_SPECS:
            if risk_col not in local:
                continue
            risk = pd.to_numeric(local[risk_col], errors="coerce")
            alert = risk.ge(risk.quantile(0.95))
            event_hours = local["is_event"].astype(bool)
            rows.append(
                {
                    "oos_month": keys[0],
                    "side_name": keys[1],
                    "archetype_policy_key": keys[2],
                    "state_model": state_model,
                    "event_hour_recall": float(alert[event_hours].mean())
                    if event_hours.any()
                    else np.nan,
                    "false_alert_hours": int((alert & ~event_hours).sum()),
                    "false_alert_hours_per_month": int((alert & ~event_hours).sum()),
                    "alert_hours": int(alert.sum()),
                    "event_hours": int(event_hours.sum()),
                }
            )
    event_work = frame.copy()
    event_work["day"] = pd.to_datetime(event_work["__ts__"], utc=True).dt.floor("D")
    event_work = event_work.merge(
        membership[
            ["day", "side_name", "archetype_policy_key", "event_id"]
        ].drop_duplicates(),
        on=["day", "side_name", "archetype_policy_key"],
        how="inner",
        validate="many_to_many",
    )
    event_score_parts: list[pd.DataFrame] = []
    for state_model, _, risk_col, _ in STATE_MODEL_SPECS:
        if risk_col not in event_work:
            continue
        part = (
            event_work.groupby(
                ["event_id", "side_name", "archetype_policy_key"], observed=True
            )
            .agg(
                max_state_risk=(risk_col, "max"),
                mean_state_risk=(risk_col, "mean"),
                recognized_hours=("__ts__", "nunique"),
            )
            .reset_index()
        )
        part["state_model"] = state_model
        event_score_parts.append(part)
    event_scores = pd.concat(event_score_parts, ignore_index=True, sort=False).merge(
        events[["event_id", "event_priority", "mean_ev", "discovery_eligible"]],
        on="event_id",
        how="left",
    )
    return pd.DataFrame(rows), event_scores


def _component_enrichment(frame: pd.DataFrame, root: Path) -> pd.DataFrame:
    membership = pd.read_parquet(root / "unreliability_event_membership.parquet")
    membership["day"] = pd.to_datetime(membership["day"], utc=True)
    side_day = (
        membership.groupby(["day", "side_name", "archetype_policy_key"], observed=True)[
            "event_id"
        ]
        .agg(lambda values: "|".join(sorted(set(map(str, values)))))
        .rename("event_ids")
        .reset_index()
    )
    work = frame.copy()
    work["day"] = pd.to_datetime(work["__ts__"], utc=True).dt.floor("D")
    work = work.merge(
        side_day,
        on=["day", "side_name", "archetype_policy_key"],
        how="left",
        validate="many_to_one",
    )
    rows: list[dict[str, Any]] = []
    for state_model, state_id_col, _, _ in STATE_MODEL_SPECS:
        if state_id_col not in work:
            continue
        for keys, local in work.groupby(
            ["oos_month", "side_name", "archetype_policy_key", state_id_col],
            observed=True,
        ):
            event_counts = local["event_ids"].dropna().value_counts()
            rows.append(
                {
                    "oos_month": keys[0],
                    "side_name": keys[1],
                    "archetype_policy_key": keys[2],
                    "state_model": state_model,
                    "state_id": int(keys[3]),
                    "hours": len(local),
                    "occupancy": float(
                        len(local)
                        / len(
                            work[
                                (work["oos_month"] == keys[0])
                                & (work["side_name"] == keys[1])
                                & (work["archetype_policy_key"] == keys[2])
                            ]
                        )
                    ),
                    "mean_signed_surprise": float(
                        pd.to_numeric(
                            local["target_signed_surprise"], errors="coerce"
                        ).mean()
                    ),
                    "negative_surprise_rate": float(
                        pd.to_numeric(local["target_signed_surprise"], errors="coerce")
                        .lt(0)
                        .mean()
                    ),
                    "positive_surprise_rate": float(
                        pd.to_numeric(local["target_signed_surprise"], errors="coerce")
                        .gt(0)
                        .mean()
                    ),
                    "negative_ev_rate": float(
                        pd.to_numeric(local["target_negative_ev"], errors="coerce")
                        .gt(0)
                        .mean()
                    ),
                    "mean_ev": float(
                        pd.to_numeric(local["target_mean_ev"], errors="coerce").mean()
                    ),
                    "event_count": int(len(event_counts)),
                    "largest_event_share": float(
                        event_counts.iloc[0] / event_counts.sum()
                    )
                    if len(event_counts)
                    else np.nan,
                    "multi_event_useful": bool(
                        len(event_counts) >= 3
                        and (event_counts.iloc[0] / event_counts.sum()) <= 0.50
                    )
                    if len(event_counts)
                    else False,
                }
            )
    return pd.DataFrame(rows)


def _placebos(frame: pd.DataFrame, draws: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(20260711)
    rows: list[dict[str, Any]] = []
    for (side, archetype), local in frame.groupby(
        ["side_name", "archetype_policy_key"], observed=True
    ):
        local = local.sort_values("__ts__", kind="stable").reset_index(drop=True)
        target = pd.to_numeric(local["target_negative_ev"], errors="coerce").gt(0.0)
        for state_model, _, risk_col, _ in STATE_MODEL_SPECS:
            if risk_col not in local:
                continue
            risk = pd.to_numeric(local[risk_col], errors="coerce")
            observed = _auprc(target, risk)
            shuffled: list[float] = []
            for _ in range(int(draws)):
                shuffled_target = target.copy()
                for _, positions in local.groupby(
                    "oos_month", observed=True
                ).indices.items():
                    shuffled_target.iloc[positions] = rng.permutation(
                        target.iloc[positions].to_numpy()
                    )
                shuffled.append(_auprc(shuffled_target, risk))
            rows.append(
                {
                    "side_name": side,
                    "archetype_policy_key": archetype,
                    "state_model": state_model,
                    "placebo": "shuffle_labels_within_month",
                    "observed_auprc": observed,
                    "placebo_mean_auprc": float(np.nanmean(shuffled)),
                    "placebo_q95_auprc": float(np.nanquantile(shuffled, 0.95)),
                    "passes_placebo": bool(observed > np.nanquantile(shuffled, 0.95)),
                }
            )
            for days in (7, 14, 21, 30):
                offset = (days * 24) % max(len(target), 1)
                shifted = pd.Series(
                    np.roll(target.to_numpy(dtype=bool), offset), index=target.index
                )
                shifted_score = _auprc(shifted, risk)
                rows.append(
                    {
                        "side_name": side,
                        "archetype_policy_key": archetype,
                        "state_model": state_model,
                        "placebo": f"circular_shift_{days}d",
                        "observed_auprc": observed,
                        "placebo_mean_auprc": shifted_score,
                        "placebo_q95_auprc": shifted_score,
                        "passes_placebo": bool(observed > shifted_score),
                    }
                )
            random_scores = [
                _auprc(target, pd.Series(rng.permutation(risk.to_numpy())))
                for _ in range(draws)
            ]
            rows.append(
                {
                    "side_name": side,
                    "archetype_policy_key": archetype,
                    "state_model": state_model,
                    "placebo": "random_assignment_preserving_occupancy",
                    "observed_auprc": observed,
                    "placebo_mean_auprc": float(np.nanmean(random_scores)),
                    "placebo_q95_auprc": float(np.nanquantile(random_scores, 0.95)),
                    "passes_placebo": bool(
                        observed > np.nanquantile(random_scores, 0.95)
                    ),
                }
            )
            if "placebo_target_mean_ev" in local:
                placebo_target = pd.to_numeric(
                    local["placebo_target_mean_ev"], errors="coerce"
                ).lt(0.0)
                lower_rank_auprc = _auprc(placebo_target, risk)
                rows.append(
                    {
                        "side_name": side,
                        "archetype_policy_key": archetype,
                        "state_model": state_model,
                        "placebo": "lower_ranked_nontraded_population",
                        "observed_auprc": observed,
                        "placebo_mean_auprc": lower_rank_auprc,
                        "placebo_q95_auprc": lower_rank_auprc,
                        "passes_placebo": bool(observed > lower_rank_auprc),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--placebo-draws", type=int, default=100)
    args = parser.parse_args()
    output = Path(args.root) / "global_side_latent_states" / "validation"
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(args.predictions)
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True)
    metrics = _recognition_rows(frame)
    events, event_scores = _event_recognition(frame, Path(args.root))
    placebos = _placebos(frame, args.placebo_draws)
    components = _component_enrichment(frame, Path(args.root))
    metrics.to_csv(output / "state_recognition_by_month_archetype.csv", index=False)
    events.to_csv(output / "event_recall_false_alerts.csv", index=False)
    event_scores.to_csv(output / "event_state_risk_scores.csv", index=False)
    placebos.to_csv(output / "placebo_tests.csv", index=False)
    components.to_csv(output / "component_enrichment_stability.csv", index=False)
    summary = {
        "schema": "archetype_residual_state_validation_v1",
        "fit_partition": "archetype_policy_key",
        "prediction_rows": len(frame),
        "months": sorted(frame["oos_month"].astype(str).unique()),
        "mean_negative_ev_auprc": float(metrics["negative_ev_auprc"].mean()),
        "mean_positive_surprise_auprc": float(
            metrics["positive_surprise_auprc"].mean()
        ),
        "recognition_by_state_model": (
            metrics.groupby("state_model", observed=True)[
                ["negative_ev_auprc", "positive_surprise_auprc"]
            ]
            .mean()
            .to_dict(orient="index")
        ),
        "placebos_passed": int(placebos["passes_placebo"].sum()),
        "placebos_total": int(len(placebos)),
        "leakage_contract": "All recognition metrics use each fold's non-training month only.",
    }
    (output / "manifest.json").write_text(
        json.dumps(_safe(summary), indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
