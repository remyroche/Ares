#!/usr/bin/env python3
"""Event-centred active-transition diagnostics on canonical exact economics.

This report freezes each score stream's one pooled global top-k book and slices
it around canonical transition events without reranking by event, timestamp,
side or asset.  It combines exact 12-hour policy economics, active-transition
grouped-OOF probabilities and conditional destination grouped-OOF
probabilities.  Results are research-only because the transition models are
grouped OOF rather than chronological policy-OOS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")
DESTINATION_COLUMNS = tuple(f"p_destination__state_{index}" for index in range(5))


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


def _validation_disclosure(
    active_contract: str, destination_contract: str
) -> dict[str, str]:
    supported = {
        "grouped_oof": "grouped OOF; non-chronological",
        "chronological_label_oos_pooled_geometry": (
            "expanding-month chronological label OOS; pooled upstream state geometry"
        ),
    }
    if active_contract not in supported or destination_contract not in supported:
        raise ValueError("unsupported transition validation contract")
    return {
        "active": supported[active_contract],
        "destination": supported[destination_contract],
        "blocker": (
            f"active={supported[active_contract]}; "
            f"destination={supported[destination_contract]}; only 13 "
            "exact-policy events overlap and operating thresholds are "
            "evaluated on that same event cohort"
        ),
    }


def _stable_top_k(
    frame: pd.DataFrame,
    *,
    score_column: str,
    fraction: float,
) -> pd.DataFrame:
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("top-k fraction must be in (0,1]")
    score = pd.to_numeric(frame[score_column], errors="raise").to_numpy(float)
    if not np.isfinite(score).all():
        raise ValueError(f"{score_column} contains non-finite values")
    count = max(1, int(math.ceil(float(fraction) * len(frame))))
    order = np.lexsort((frame["candidate_id"].astype(str).to_numpy(), -score))
    return frame.iloc[order[:count]].copy()


def _episode_count(mask: np.ndarray, timestamps: pd.Series) -> int:
    if not len(mask):
        return 0
    active_times = pd.to_datetime(timestamps.loc[mask], utc=True).sort_values()
    if active_times.empty:
        return 0
    gaps = active_times.diff().gt(pd.Timedelta(hours=1))
    return int(1 + gaps.iloc[1:].sum())


def _safe_spearman(x: pd.Series, y: pd.Series) -> float:
    x_values = pd.to_numeric(x, errors="coerce").to_numpy(float)
    y_values = pd.to_numeric(y, errors="coerce").to_numpy(float)
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    if valid.sum() < 3 or np.unique(x_values[valid]).size < 2:
        return float("nan")
    return float(spearmanr(x_values[valid], y_values[valid]).statistic)


def _window_metrics(
    frame: pd.DataFrame,
    *,
    score_column: str,
    prefix: str,
) -> dict[str, Any]:
    if frame.empty:
        return {
            f"{prefix}_rows": 0,
            f"{prefix}_hours": 0,
            f"{prefix}_mean_score": np.nan,
            f"{prefix}_mean_gross_bps": np.nan,
            f"{prefix}_mean_cost_bps": np.nan,
            f"{prefix}_mean_net_bps": np.nan,
            f"{prefix}_positive_net_rate": np.nan,
            f"{prefix}_full_stop_rate": np.nan,
            f"{prefix}_timeout_rate": np.nan,
            f"{prefix}_trailing_rate": np.nan,
            f"{prefix}_adverse_exit_rate": np.nan,
            f"{prefix}_mean_mfe_bps": np.nan,
            f"{prefix}_mean_mae_bps": np.nan,
            f"{prefix}_score_net_rank_ic": np.nan,
            f"{prefix}_net_minus_score_bps": np.nan,
        }
    gross = pd.to_numeric(frame["execution_gross_ev_12h"], errors="raise")
    cost = pd.to_numeric(frame["execution_cost_return"], errors="raise")
    net = pd.to_numeric(frame["execution_net_ev_12h"], errors="raise")
    exit_class = frame["execution_exit_class"].astype(str)
    score = pd.to_numeric(frame[score_column], errors="raise")
    return {
        f"{prefix}_rows": int(len(frame)),
        f"{prefix}_hours": int(frame["__ts__"].nunique()),
        f"{prefix}_mean_score": float(score.mean()),
        f"{prefix}_mean_gross_bps": float(10_000.0 * gross.mean()),
        f"{prefix}_mean_cost_bps": float(10_000.0 * cost.mean()),
        f"{prefix}_mean_net_bps": float(10_000.0 * net.mean()),
        f"{prefix}_positive_net_rate": float(net.gt(0.0).mean()),
        f"{prefix}_full_stop_rate": float(exit_class.eq("full_stop").mean()),
        f"{prefix}_timeout_rate": float(exit_class.eq("timeout").mean()),
        f"{prefix}_trailing_rate": float(exit_class.eq("trailing").mean()),
        f"{prefix}_adverse_exit_rate": float(exit_class.eq("adverse_exit").mean()),
        f"{prefix}_mean_mfe_bps": float(
            10_000.0 * frame["execution_mfe_return_12h"].mean()
        ),
        f"{prefix}_mean_mae_bps": float(
            10_000.0 * frame["execution_mae_return_12h"].mean()
        ),
        f"{prefix}_score_net_rank_ic": _safe_spearman(score, net),
        f"{prefix}_net_minus_score_bps": float(
            10_000.0 * (net - score).mean()
        ),
    }


def _destination_summary(
    event_id: str,
    destination: pd.DataFrame,
    anchor: pd.Timestamp,
    destination_state: int,
) -> dict[str, Any]:
    local = destination.loc[destination["target__event_id"].astype(str).eq(str(event_id))].copy()
    if local.empty:
        return {
            "destination_prediction": None,
            "destination_prediction_correct": False,
            "destination_confidence": np.nan,
            "destination_entropy": np.nan,
            "destination_prediction_source_utc": pd.NaT,
        }
    local["source_utc"] = pd.to_datetime(local["source_utc"], utc=True)
    pre_or_onset = local.loc[local["source_utc"].le(anchor)]
    if pre_or_onset.empty:
        pre_or_onset = local
    probabilities = pre_or_onset.loc[:, DESTINATION_COLUMNS].apply(
        pd.to_numeric, errors="raise"
    )
    confidence = probabilities.max(axis=1)
    best_index = confidence.idxmax()
    best = probabilities.loc[best_index].to_numpy(float)
    entropy = -float(
        np.sum(np.where(best > 0.0, best * np.log(np.clip(best, 1e-12, 1.0)), 0.0))
    )
    prediction = str(pre_or_onset.loc[best_index, "predicted_destination"])
    return {
        "destination_prediction": prediction,
        "destination_prediction_correct": prediction == f"state_{destination_state}",
        "destination_confidence": float(best.max()),
        "destination_entropy": entropy,
        "destination_prediction_source_utc": pre_or_onset.loc[
            best_index, "source_utc"
        ],
    }


def _active_event_summary(
    event: pd.Series,
    active: pd.DataFrame,
    thresholds: Sequence[float],
) -> dict[str, Any]:
    event_id = str(event["event_id"])
    anchor = pd.Timestamp(event["anchor_source_utc"])
    end = pd.Timestamp(event["transition_end_utc"])
    local = active.loc[
        active["source_utc"].between(
            anchor - pd.Timedelta(hours=3),
            end + pd.Timedelta(hours=1),
            inclusive="left",
        )
    ].copy()
    onset_row = active.loc[active["source_utc"].eq(anchor)]
    during = active.loc[
        active["source_utc"].ge(anchor) & active["source_utc"].lt(end)
    ]
    output: dict[str, Any] = {
        "active_probability_at_onset": float(onset_row["prediction"].iloc[0])
        if len(onset_row)
        else np.nan,
        "active_probability_max_during": float(during["prediction"].max())
        if len(during)
        else np.nan,
    }
    for threshold in thresholds:
        label = str(threshold).replace(".", "p")
        alerts = local.loc[local["prediction"].ge(float(threshold))]
        output[f"active_detected_threshold_{label}"] = bool(
            during["prediction"].ge(float(threshold)).any()
        )
        output[f"active_alert_hours_threshold_{label}"] = int(len(alerts))
        output[f"active_alert_episodes_threshold_{label}"] = _episode_count(
            local["prediction"].ge(float(threshold)).to_numpy(),
            local["source_utc"],
        )
        output[f"active_first_alert_offset_hours_threshold_{label}"] = (
            float((alerts["source_utc"].min() - anchor) / pd.Timedelta(hours=1))
            if len(alerts)
            else np.nan
        )
    return output


def build_event_report(
    candidates: pd.DataFrame,
    active: pd.DataFrame,
    events: pd.DataFrame,
    destination: pd.DataFrame,
    *,
    score_columns: Sequence[str],
    top_k_fraction: float,
    thresholds: Sequence[float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = candidates.loc[candidates["mapped_eligible"].astype(bool)].copy()
    work["__ts__"] = pd.to_datetime(work["__ts__"], utc=True, errors="raise")
    active = active.copy()
    active["source_utc"] = pd.to_datetime(active["source_utc"], utc=True, errors="raise")
    if active["source_utc"].duplicated().any():
        raise ValueError("active OOF must have one row per source hour")
    events = events.copy()
    for column in (
        "anchor_source_utc",
        "transition_start_utc",
        "transition_end_utc",
    ):
        events[column] = pd.to_datetime(events[column], utc=True, errors="raise")
    minimum = work["__ts__"].min()
    maximum = work["__ts__"].max()
    overlap_events = events.loc[
        events["anchor_source_utc"].between(minimum, maximum, inclusive="both")
    ].copy()
    if overlap_events.empty:
        raise ValueError("no transition events overlap canonical candidates")
    active_join = active.rename(
        columns={
            "prediction": "active_transition_probability_oof",
            "target__transition_active": "expost_transition_active",
            "target__event_id": "transition_event_id",
        }
    )
    work = work.merge(
        active_join[
            [
                "source_utc",
                "active_transition_probability_oof",
                "expost_transition_active",
                "transition_event_id",
            ]
        ],
        left_on="__ts__",
        right_on="source_utc",
        how="left",
        validate="many_to_one",
    )
    if work["active_transition_probability_oof"].isna().any():
        raise ValueError("canonical event report lacks active OOF coverage")
    frozen_books = {
        score_column: _stable_top_k(
            work, score_column=score_column, fraction=top_k_fraction
        )
        for score_column in score_columns
    }
    rows: list[dict[str, Any]] = []
    for _, event in overlap_events.iterrows():
        event_id = str(event["event_id"])
        anchor = pd.Timestamp(event["anchor_source_utc"])
        end = pd.Timestamp(event["transition_end_utc"])
        event_common = {
            "event_id": event_id,
            "source_state": int(event["source_state"]),
            "destination_state": int(event["destination_state"]),
            "transition_pair": str(event["transition_archetype"]),
            "anchor_source_utc": anchor,
            "transition_end_utc": end,
            "transition_duration_hours": float(
                (end - anchor) / pd.Timedelta(hours=1)
            ),
            "market_transition_severity": float(event["robust_pre_post_shift"]),
            **_destination_summary(
                event_id, destination, anchor, int(event["destination_state"])
            ),
            **_active_event_summary(event, active, thresholds),
        }
        for score_column, book in frozen_books.items():
            before = book.loc[
                book["__ts__"].ge(anchor - pd.Timedelta(hours=12))
                & book["__ts__"].lt(anchor)
            ]
            during = book.loc[
                book["__ts__"].ge(anchor) & book["__ts__"].lt(end)
            ]
            after = book.loc[
                book["__ts__"].ge(end)
                & book["__ts__"].lt(end + pd.Timedelta(hours=12))
            ]
            row = {
                **event_common,
                "score_stream": score_column,
                **_window_metrics(before, score_column=score_column, prefix="before"),
                **_window_metrics(during, score_column=score_column, prefix="during"),
                **_window_metrics(after, score_column=score_column, prefix="after"),
            }
            row["net_damage_during_vs_before_bps"] = (
                row["before_mean_net_bps"] - row["during_mean_net_bps"]
            )
            row["gross_damage_during_vs_before_bps"] = (
                row["before_mean_gross_bps"] - row["during_mean_gross_bps"]
            )
            row["stop_rate_change_during_vs_before"] = (
                row["during_full_stop_rate"] - row["before_full_stop_rate"]
            )
            row["rank_ic_change_during_vs_before"] = (
                row["during_score_net_rank_ic"] - row["before_score_net_rank_ic"]
            )
            row["economically_damaging"] = bool(
                np.isfinite(row["during_mean_net_bps"])
                and row["during_mean_net_bps"] < 0.0
                and row["net_damage_during_vs_before_bps"] > 0.0
            )
            rows.append(row)
    report = pd.DataFrame(rows)
    severity_threshold = float(
        report.drop_duplicates("event_id")["market_transition_severity"].quantile(0.75)
    )
    report["top25_severity"] = report["market_transition_severity"].ge(
        severity_threshold
    )
    operating_rows: list[dict[str, Any]] = []
    total_days = (
        active["source_utc"].max() - active["source_utc"].min()
    ) / pd.Timedelta(days=1)
    event_unique = report.drop_duplicates("event_id")
    for threshold in thresholds:
        label = str(threshold).replace(".", "p")
        detected_column = f"active_detected_threshold_{label}"
        false_mask = (
            active["prediction"].ge(float(threshold))
            & ~active["target__transition_active"].astype(bool)
        ).to_numpy()
        false_episodes = _episode_count(false_mask, active["source_utc"])
        row = {
            "threshold": float(threshold),
            "event_count": int(len(event_unique)),
            "event_recall": float(event_unique[detected_column].mean()),
            "top25_severity_event_recall": float(
                event_unique.loc[
                    event_unique["market_transition_severity"].ge(
                        severity_threshold
                    ),
                    detected_column,
                ].mean()
            ),
            "false_alert_episodes": false_episodes,
            "false_alert_episodes_per_30d": float(
                false_episodes * 30.0 / max(float(total_days), 1.0)
            ),
            "median_first_alert_offset_hours": float(
                event_unique[
                    f"active_first_alert_offset_hours_threshold_{label}"
                ].median()
            ),
            "mean_alert_episodes_per_event": float(
                event_unique[
                    f"active_alert_episodes_threshold_{label}"
                ].mean()
            ),
        }
        for score_column in score_columns:
            score_events = report.loc[report["score_stream"].eq(score_column)]
            damaging = score_events.loc[score_events["economically_damaging"]]
            row[f"damaging_{score_column}_event_count"] = int(len(damaging))
            row[f"damaging_{score_column}_event_recall"] = (
                float(damaging[detected_column].mean())
                if len(damaging)
                else np.nan
            )
        operating_rows.append(row)
    operating = pd.DataFrame(operating_rows)
    pair_summary = (
        report.groupby(
            ["score_stream", "source_state", "destination_state", "transition_pair"],
            observed=True,
            sort=True,
        )
        .agg(
            event_count=("event_id", "nunique"),
            damaging_event_count=("economically_damaging", "sum"),
            mean_net_damage_bps=("net_damage_during_vs_before_bps", "mean"),
            mean_gross_damage_bps=("gross_damage_during_vs_before_bps", "mean"),
            mean_stop_rate_change=("stop_rate_change_during_vs_before", "mean"),
            mean_rank_ic_change=("rank_ic_change_during_vs_before", "mean"),
            mean_destination_confidence=("destination_confidence", "mean"),
            destination_accuracy=("destination_prediction_correct", "mean"),
            mean_transition_severity=("market_transition_severity", "mean"),
        )
        .reset_index()
    )
    return report, operating, pair_summary


def bootstrap_event_summary(
    report: pd.DataFrame,
    *,
    draws: int = 2_000,
    seed: int = 20260729,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for score_stream, local in report.groupby("score_stream", sort=True):
        damage = pd.to_numeric(
            local["net_damage_during_vs_before_bps"], errors="coerce"
        ).dropna().to_numpy(float)
        if not len(damage):
            continue
        samples = rng.choice(damage, size=(int(draws), len(damage)), replace=True)
        means = samples.mean(axis=1)
        rows.append(
            {
                "score_stream": score_stream,
                "event_count": int(len(damage)),
                "damaging_event_count": int(
                    local["economically_damaging"].fillna(False).sum()
                ),
                "mean_net_damage_bps": float(damage.mean()),
                "median_net_damage_bps": float(np.median(damage)),
                "bootstrap_mean_damage_p05_bps": float(np.quantile(means, 0.05)),
                "bootstrap_mean_damage_p50_bps": float(np.quantile(means, 0.50)),
                "bootstrap_mean_damage_p95_bps": float(np.quantile(means, 0.95)),
                "bootstrap_probability_mean_damage_positive": float(
                    np.mean(means > 0.0)
                ),
                "destination_accuracy": float(
                    local["destination_prediction_correct"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def destination_event_abstention_curve(
    report: pd.DataFrame, thresholds: Sequence[float]
) -> pd.DataFrame:
    events = report.drop_duplicates("event_id").copy()
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        accepted = pd.to_numeric(
            events["destination_confidence"], errors="coerce"
        ).ge(float(threshold))
        local = events.loc[accepted]
        rows.append(
            {
                "minimum_confidence": float(threshold),
                "event_count": int(len(events)),
                "accepted_events": int(len(local)),
                "coverage": float(accepted.mean()),
                "accuracy": float(
                    local["destination_prediction_correct"].mean()
                )
                if len(local)
                else np.nan,
                "mean_confidence": float(local["destination_confidence"].mean())
                if len(local)
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    candidates_path = Path(args.mapped_candidates)
    active_path = Path(args.active_oof)
    events_path = Path(args.events)
    destination_path = Path(args.destination_oof)
    report, operating, pair_summary = build_event_report(
        pd.read_parquet(candidates_path),
        pd.read_parquet(active_path),
        pd.read_parquet(events_path),
        pd.read_parquet(destination_path),
        score_columns=args.score_columns,
        top_k_fraction=float(args.top_k_fraction),
        thresholds=args.thresholds,
    )
    output.mkdir(parents=True, exist_ok=False)
    event_path = output / "canonical_event_metrics.parquet"
    operating_path = output / "active_operating_curve.csv"
    pair_path = output / "transition_pair_impacts.csv"
    summary_path = output / "score_event_summary.csv"
    destination_abstention_path = output / "destination_event_abstention_curve.csv"
    event_summary = bootstrap_event_summary(report)
    destination_abstention = destination_event_abstention_curve(
        report, args.destination_confidence_thresholds
    )
    report.to_parquet(event_path, index=False, compression="zstd")
    operating.to_csv(operating_path, index=False)
    pair_summary.to_csv(pair_path, index=False)
    event_summary.to_csv(summary_path, index=False)
    destination_abstention.to_csv(destination_abstention_path, index=False)
    validation = _validation_disclosure(
        args.active_validation_contract,
        args.destination_validation_contract,
    )
    manifest = {
        "schema": "active_transition_canonical_event_impacts_v2",
        "status": "RESEARCH_ONLY_COMMON_LINEAGE_EVENT_REPORT_COMPLETE",
        "promotion_eligible": False,
        "promotion_blocker": validation["blocker"],
        "validation_contract": {
            "active": validation["active"],
            "destination": validation["destination"],
        },
        "event_count": int(report["event_id"].nunique()),
        "event_score_rows": int(len(report)),
        "score_streams": list(args.score_columns),
        "selection_contract": (
            "one pooled global top-k per score stream, frozen before event slicing; "
            "no event/timestamp/side reranking"
        ),
        "event_windows": {
            "before": "[-12h, onset)",
            "during": "[onset, transition_end)",
            "after": "[transition_end, transition_end+12h)",
        },
        "active_thresholds": [float(value) for value in args.thresholds],
        "destination_confidence_thresholds": [
            float(value) for value in args.destination_confidence_thresholds
        ],
        "uncertainty_contract": (
            "deterministic 2,000-draw event bootstrap; descriptive because "
            "events are few and at least one transition component remains "
            "non-chronological or uses pooled upstream geometry"
        ),
        "sources": {
            "mapped_candidates": {
                "path": str(candidates_path),
                "sha256": _sha256(candidates_path),
            },
            "active_oof": {
                "path": str(active_path),
                "sha256": _sha256(active_path),
            },
            "events": {
                "path": str(events_path),
                "sha256": _sha256(events_path),
            },
            "destination_oof": {
                "path": str(destination_path),
                "sha256": _sha256(destination_path),
            },
        },
        "outputs": {},
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
    }
    outputs = {
        "event_metrics": event_path,
        "active_operating_curve": operating_path,
        "transition_pair_impacts": pair_path,
        "score_event_summary": summary_path,
        "destination_event_abstention_curve": destination_abstention_path,
    }
    manifest["outputs"] = {
        name: {"path": str(path), "sha256": _sha256(path)}
        for name, path in outputs.items()
    }
    manifest_path = output / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    root = Path("/Users/remyroche/Documents/Ares")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mapped-candidates",
        type=Path,
        default=root
        / (
            "data_perp/artifacts/historical_causal_score_economics_mapping_20260729_v1/"
            "canonical_base__score_base_alpha/causal_mapped_candidates.parquet"
        ),
    )
    parser.add_argument(
        "--active-oof",
        type=Path,
        default=root
        / "data_perp/artifacts/regime_transition_active_head_20260726_v1/grouped_oof.parquet",
    )
    parser.add_argument(
        "--events",
        type=Path,
        default=root
        / "data_perp/artifacts/regime_transition_research_20260726_v3/transition_events.parquet",
    )
    parser.add_argument(
        "--destination-oof",
        type=Path,
        default=root
        / (
            "data_perp/artifacts/regime_transition_classifier_ablation_20260726_v2/"
            "destination_grouped_oof.parquet"
        ),
    )
    parser.add_argument(
        "--active-validation-contract",
        choices=("grouped_oof", "chronological_label_oos_pooled_geometry"),
        default="grouped_oof",
    )
    parser.add_argument(
        "--destination-validation-contract",
        choices=("grouped_oof", "chronological_label_oos_pooled_geometry"),
        default="grouped_oof",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--score-columns", nargs="+", default=("score_raw", "mapped_direct_net")
    )
    parser.add_argument("--top-k-fraction", type=float, default=0.10)
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=(0.25, 0.50, 0.75)
    )
    parser.add_argument(
        "--destination-confidence-thresholds",
        type=float,
        nargs="+",
        default=(0.0, 0.50, 0.60, 0.70, 0.80),
    )
    return parser


def main() -> None:
    print(json.dumps(_safe(run(_parser().parse_args())), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
