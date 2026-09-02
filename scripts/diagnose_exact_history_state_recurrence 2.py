#!/usr/bin/env python3
"""Diagnose recurrence of pre-May market states in exact-policy OOF streams."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.cluster import MiniBatchKMeans
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.reconstruct_janfeb2025_execution_ev_12h_oof import (  # noqa: E402
    eligible_raw_features,
    normalize_symbol,
    source_paths,
)
from scripts.run_exact_policy_capture_support_ablation import (  # noqa: E402
    apply_recent_mapping_frame,
)

SCHEMA = "exact_history_state_recurrence_diagnostic_v1"
IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")
SIDE = "side_name"
SIGNAL_TIME = "__ts__"
DECISION_TIME = "execution_decision_utc"
RESOLUTION_TIME = "execution_label_end_utc"
TARGET = "execution_net_ev_12h"
HISTORICAL_SCORE = "historical_direct_ev_oof"
CURRENT_SCORE = "causal_recent_isotonic_ev"


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
    temporary.write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def common_state_features(
    historical_paths: Sequence[Path],
    current_columns: Sequence[str],
) -> list[str]:
    historical = set(eligible_raw_features(historical_paths))
    current = {
        column.removeprefix("capture_candidate__")
        for column in current_columns
        if column.startswith("capture_candidate__")
    }
    return sorted(historical & current)


def select_one_global_topk(
    frame: pd.DataFrame,
    score_column: str,
    *,
    fraction: float = 0.10,
) -> np.ndarray:
    score = pd.to_numeric(frame[score_column], errors="coerce").to_numpy(dtype=float)
    valid = np.flatnonzero(np.isfinite(score))
    selected = np.zeros(len(frame), dtype=bool)
    if not len(valid):
        return selected
    count = max(1, int(np.ceil(float(fraction) * len(valid))))
    order = np.argsort(-score[valid], kind="mergesort")[:count]
    selected[valid[order]] = True
    return selected


def recurrence_gate(
    weekly: pd.DataFrame,
    *,
    minimum_rows: int,
    minimum_historical_weeks: int,
    minimum_recent_weeks: int,
    minimum_sign_consistency: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (side, state), group in weekly.groupby([SIDE, "state_id"], sort=True):
        eligible = group.loc[group["selected_rows"].ge(int(minimum_rows))].copy()
        historical = eligible.loc[eligible["era"].eq("historical")]
        recent = eligible.loc[eligible["era"].eq("may_june")]
        july = eligible.loc[eligible["era"].eq("july")]
        reference = pd.concat([historical, recent], ignore_index=True)
        signs = np.sign(reference["selected_mean_net_bps"].to_numpy(dtype=float))
        positive_share = float((signs > 0).mean()) if len(signs) else np.nan
        negative_share = float((signs < 0).mean()) if len(signs) else np.nan
        consistency = (
            max(positive_share, negative_share)
            if np.isfinite(positive_share) and np.isfinite(negative_share)
            else 0.0
        )
        reference_sign = (
            1
            if positive_share > negative_share
            else (-1 if negative_share > positive_share else 0)
        )
        july_mean = (
            float(
                np.average(
                    july["selected_mean_net_bps"],
                    weights=july["selected_rows"],
                )
            )
            if len(july)
            else np.nan
        )
        july_sign = int(np.sign(july_mean)) if np.isfinite(july_mean) else 0
        eligible_state = (
            historical["week"].nunique() >= int(minimum_historical_weeks)
            and recent["week"].nunique() >= int(minimum_recent_weeks)
            and consistency >= float(minimum_sign_consistency)
            and july["week"].nunique() >= 1
            and reference_sign != 0
            and july_sign == reference_sign
        )
        rows.append(
            {
                SIDE: side,
                "state_id": int(state),
                "historical_supported_weeks": int(historical["week"].nunique()),
                "may_june_supported_weeks": int(recent["week"].nunique()),
                "july_supported_weeks": int(july["week"].nunique()),
                "reference_sign_consistency": consistency,
                "reference_sign": reference_sign,
                "july_selected_mean_net_bps": july_mean,
                "july_sign": july_sign,
                "eligible_recurring_state": bool(eligible_state),
            }
        )
    return pd.DataFrame(rows)


def _load_historical_raw(paths: Sequence[Path], features: Sequence[str]) -> pd.DataFrame:
    use = [*IDENTITY, *features]
    parts = []
    for path in paths:
        part = pd.read_parquet(path, columns=use)
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True)
    frame[SIGNAL_TIME] = pd.to_datetime(frame[SIGNAL_TIME], utc=True, errors="raise")
    frame["__symbol__"] = frame["__symbol__"].map(normalize_symbol)
    frame[SIDE] = frame[SIDE].astype(str).str.lower()
    if frame.duplicated(list(IDENTITY), keep=False).any():
        raise ValueError("historical raw ledgers contain duplicate identities")
    return frame


def _timestamp_geometry(
    frame: pd.DataFrame,
    features: Sequence[str],
) -> tuple[pd.DataFrame, list[str]]:
    numeric = frame.loc[:, features].apply(pd.to_numeric, errors="coerce")
    work = frame.loc[:, [SIDE, SIGNAL_TIME]].copy()
    work = pd.concat([work, numeric], axis=1)
    median = work.groupby([SIDE, SIGNAL_TIME], sort=True)[list(features)].median()
    q25 = work.groupby([SIDE, SIGNAL_TIME], sort=True)[list(features)].quantile(0.25)
    q75 = work.groupby([SIDE, SIGNAL_TIME], sort=True)[list(features)].quantile(0.75)
    median.columns = [f"median__{column}" for column in features]
    spread = q75 - q25
    spread.columns = [f"iqr__{column}" for column in features]
    result = pd.concat([median, spread], axis=1).reset_index()
    return result, [*median.columns, *spread.columns]


def _fit_states(
    historical: pd.DataFrame,
    current: pd.DataFrame,
    state_columns: Sequence[str],
    *,
    clusters: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    historical_parts = []
    current_parts = []
    report: dict[str, Any] = {}
    for side_index, side in enumerate(("long", "short")):
        fit = historical.loc[historical[SIDE].eq(side)].copy().reset_index(drop=True)
        score = current.loc[current[SIDE].eq(side)].copy().reset_index(drop=True)
        if len(fit) < 100 or score.empty:
            raise ValueError(f"insufficient timestamp geometry for side {side}")
        transform = make_pipeline(
            SimpleImputer(strategy="median"),
            RobustScaler(quantile_range=(25.0, 75.0)),
        )
        fit_x = transform.fit_transform(fit.loc[:, state_columns])
        score_x = transform.transform(score.loc[:, state_columns])
        model = MiniBatchKMeans(
            n_clusters=int(clusters),
            random_state=int(seed + side_index),
            n_init=10,
            batch_size=min(1024, len(fit)),
        ).fit(fit_x)
        fit_state = model.predict(fit_x)
        score_state = model.predict(score_x)
        fit_distance = np.sqrt(
            np.sum((fit_x - model.cluster_centers_[fit_state]) ** 2, axis=1)
        )
        score_distance = np.sqrt(
            np.sum((score_x - model.cluster_centers_[score_state]) ** 2, axis=1)
        )
        threshold = float(np.quantile(fit_distance, 0.95))
        fit["state_id"] = fit_state.astype(np.int16)
        fit["state_distance"] = fit_distance
        fit["state_ood"] = fit_distance > threshold
        score["state_id"] = score_state.astype(np.int16)
        score["state_distance"] = score_distance
        score["state_ood"] = score_distance > threshold
        historical_parts.append(fit)
        current_parts.append(score)
        report[side] = {
            "fit_timestamps": int(len(fit)),
            "current_timestamps": int(len(score)),
            "clusters": int(clusters),
            "historical_occupancy": {
                str(key): int(value)
                for key, value in fit["state_id"].value_counts().sort_index().items()
            },
            "current_occupancy": {
                str(key): int(value)
                for key, value in score["state_id"].value_counts().sort_index().items()
            },
            "historical_distance_p95": threshold,
            "current_ood_rate": float(score["state_ood"].mean()),
        }
    return (
        pd.concat(historical_parts, ignore_index=True),
        pd.concat(current_parts, ignore_index=True),
        report,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    historical_paths = source_paths(
        args.labels_root,
        start_month=args.historical_start_month,
        end_month=args.historical_end_month,
    )
    current_schema = pq.read_schema(args.current_features).names
    features = common_state_features(historical_paths, current_schema)
    if len(features) < int(args.minimum_common_features):
        raise ValueError(f"only {len(features)} common raw state features")

    historical_oof = pd.read_parquet(args.historical_oof)
    historical_raw = _load_historical_raw(historical_paths, features)
    historical = historical_oof.merge(
        historical_raw,
        on=list(IDENTITY),
        how="left",
        validate="one_to_one",
    )
    if historical[features].notna().any(axis=1).mean() < 0.99:
        raise ValueError("historical raw feature join coverage is below 99%")
    historical["raw_score"] = pd.to_numeric(
        historical[HISTORICAL_SCORE], errors="coerce"
    )
    historical["mapped_score"], historical_mapping = apply_recent_mapping_frame(
        historical,
        historical["raw_score"].to_numpy(dtype=float),
        scope="global",
    )
    historical["evidence_tier"] = "jan_apr_exact_oof"

    current_features = pd.read_parquet(args.current_features)
    current_score = pd.read_parquet(args.current_scores)
    current = current_score.merge(
        current_features.loc[
            :,
            [
                *IDENTITY,
                *[f"capture_candidate__{column}" for column in features],
            ],
        ],
        on=list(IDENTITY),
        how="inner",
        validate="one_to_one",
    )
    current = current.rename(
        columns={f"capture_candidate__{column}": column for column in features}
    )
    current["mapped_score"] = pd.to_numeric(
        current[CURRENT_SCORE], errors="coerce"
    )
    current["evidence_tier"] = "may_july_current_oof_forward"

    historical_state, state_columns = _timestamp_geometry(historical, features)
    current_state, current_state_columns = _timestamp_geometry(current, features)
    if state_columns != current_state_columns:
        raise RuntimeError("historical/current timestamp state columns differ")
    historical_state, current_state, state_report = _fit_states(
        historical_state,
        current_state,
        state_columns,
        clusters=args.clusters,
        seed=args.random_state,
    )
    state_key = [SIDE, SIGNAL_TIME]
    historical = historical.merge(
        historical_state.loc[
            :, [*state_key, "state_id", "state_distance", "state_ood"]
        ],
        on=state_key,
        how="left",
        validate="many_to_one",
    )
    current = current.merge(
        current_state.loc[
            :, [*state_key, "state_id", "state_distance", "state_ood"]
        ],
        on=state_key,
        how="left",
        validate="many_to_one",
    )
    historical["selected_global_top10"] = select_one_global_topk(
        historical, "mapped_score"
    )
    current["selected_global_top10"] = select_one_global_topk(
        current, "mapped_score"
    )
    combined = pd.concat(
        [
            historical.loc[
                :,
                [
                    *IDENTITY,
                    DECISION_TIME,
                    RESOLUTION_TIME,
                    TARGET,
                    "mapped_score",
                    "state_id",
                    "state_distance",
                    "state_ood",
                    "selected_global_top10",
                    "evidence_tier",
                ],
            ],
            current.loc[
                :,
                [
                    *IDENTITY,
                    DECISION_TIME,
                    RESOLUTION_TIME,
                    TARGET,
                    "mapped_score",
                    "state_id",
                    "state_distance",
                    "state_ood",
                    "selected_global_top10",
                    "evidence_tier",
                ],
            ],
        ],
        ignore_index=True,
    )
    decision = pd.to_datetime(combined[DECISION_TIME], utc=True, errors="raise")
    combined["month"] = decision.dt.strftime("%Y-%m")
    combined["week"] = (
        decision.dt.tz_localize(None).dt.to_period("W-SUN").astype(str)
    )
    combined["era"] = np.select(
        [
            decision.lt(pd.Timestamp("2025-05-01", tz="UTC")),
            decision.lt(pd.Timestamp("2026-07-01", tz="UTC")),
        ],
        ["historical", "may_june"],
        default="july",
    )
    weekly_rows = []
    for keys, group in combined.groupby(
        ["evidence_tier", "era", SIDE, "state_id", "week"], sort=True
    ):
        selected = group.loc[group["selected_global_top10"]]
        weekly_rows.append(
            {
                "evidence_tier": keys[0],
                "era": keys[1],
                SIDE: keys[2],
                "state_id": int(keys[3]),
                "week": keys[4],
                "candidate_rows": int(len(group)),
                "selected_rows": int(len(selected)),
                "selected_mean_net_bps": (
                    float(selected[TARGET].mean() * 10_000.0)
                    if len(selected)
                    else np.nan
                ),
                "selected_positive_rate": (
                    float((selected[TARGET] > 0.0).mean())
                    if len(selected)
                    else np.nan
                ),
            }
        )
    weekly = pd.DataFrame(weekly_rows)
    gate = recurrence_gate(
        weekly,
        minimum_rows=args.minimum_selected_rows_per_week,
        minimum_historical_weeks=args.minimum_historical_weeks,
        minimum_recent_weeks=args.minimum_recent_weeks,
        minimum_sign_consistency=args.minimum_sign_consistency,
    )
    occupancy = (
        pd.concat(
            [
                historical_state.assign(evidence_tier="jan_apr_reference"),
                current_state.assign(evidence_tier="may_july_forward"),
            ],
            ignore_index=True,
        )
        .assign(
            month=lambda x: pd.to_datetime(
                x[SIGNAL_TIME], utc=True, errors="raise"
            ).dt.strftime("%Y-%m")
        )
        .groupby(["evidence_tier", "month", SIDE, "state_id"], sort=True)
        .agg(
            timestamp_rows=(SIGNAL_TIME, "size"),
            mean_distance=("state_distance", "mean"),
            ood_rate=("state_ood", "mean"),
        )
        .reset_index()
    )

    args.output_dir.mkdir(parents=True)
    paths = {
        "candidate_assignments": args.output_dir / "state_candidate_assignments.parquet",
        "timestamp_states": args.output_dir / "timestamp_state_geometry.parquet",
        "weekly_economics": args.output_dir / "state_weekly_selected_economics.csv",
        "recurrence_gate": args.output_dir / "state_recurrence_gate.csv",
        "occupancy": args.output_dir / "state_monthly_occupancy.csv",
    }
    combined.to_parquet(paths["candidate_assignments"], index=False, compression="zstd")
    pd.concat(
        [
            historical_state.assign(evidence_tier="jan_apr_reference"),
            current_state.assign(evidence_tier="may_july_forward"),
        ],
        ignore_index=True,
    ).to_parquet(paths["timestamp_states"], index=False, compression="zstd")
    weekly.to_csv(paths["weekly_economics"], index=False)
    gate.to_csv(paths["recurrence_gate"], index=False)
    occupancy.to_csv(paths["occupancy"], index=False)
    manifest = {
        "schema": SCHEMA,
        "status": "completed_diagnostic_not_a_trade_gate",
        "contract": {
            "state_features": (
                "intersection of raw PIT fields available in the January-April "
                "historical ledgers and frozen May-July feature universe"
            ),
            "geometry": (
                "per-side robust timestamp cross-sectional median/IQR; fixed "
                "KMeans fitted outcome-free on January-April and applied unchanged "
                "to May-July"
            ),
            "historical_score": (
                "strict two-layer execution-EV OOF followed by causal global "
                "21-day correction"
            ),
            "current_score": "canonical causal recent-isotonic execution EV",
            "ranking": (
                "one pooled global top10 inside each non-comparable evidence tier; "
                "state/week rows only slice those frozen selections and never rerank"
            ),
            "use": (
                "recurrence and OOD diagnosis only; historical and current tiers "
                "use different candidate models and are never pooled as PnL"
            ),
        },
        "features": features,
        "state_columns": state_columns,
        "states": state_report,
        "historical_mapping": historical_mapping,
        "recurrence": {
            "minimum_selected_rows_per_week": args.minimum_selected_rows_per_week,
            "minimum_historical_weeks": args.minimum_historical_weeks,
            "minimum_recent_weeks": args.minimum_recent_weeks,
            "minimum_sign_consistency": args.minimum_sign_consistency,
            "eligible_states": int(gate["eligible_recurring_state"].sum()),
        },
        "rows": {
            "historical_candidates": int(len(historical)),
            "current_candidates": int(len(current)),
            "historical_timestamps": int(len(historical_state)),
            "current_timestamps": int(len(current_state)),
        },
        "inputs": {
            "historical_oof": {
                "path": str(args.historical_oof),
                "sha256": _sha256(args.historical_oof),
            },
            "current_features": {
                "path": str(args.current_features),
                "sha256": _sha256(args.current_features),
            },
            "current_scores": {
                "path": str(args.current_scores),
                "sha256": _sha256(args.current_scores),
            },
            "historical_ledgers": [
                {"path": str(path), "sha256": _sha256(path)}
                for path in historical_paths
            ],
        },
        "outputs": {
            key: {"path": str(path), "sha256": _sha256(path)}
            for key, path in paths.items()
        },
    }
    _write_json(args.output_dir / "manifest.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--historical-oof",
        type=Path,
        default=Path(
            "data_perp/artifacts/janapr2025_execution_ev_exact1m_two_layer_oof_20260727_v1/"
            "two_layer_direct_ev_strict_oof.parquet"
        ),
    )
    parser.add_argument(
        "--labels-root",
        type=Path,
        default=Path(
            "data_perp/artifacts/"
            "20260720_s59_h5_signalclose_causal_trailing_cost100bps_labels_v2/labels"
        ),
    )
    parser.add_argument("--historical-start-month", default="2025-01")
    parser.add_argument("--historical-end-month", default="2025-04")
    parser.add_argument(
        "--current-features",
        type=Path,
        default=Path(
            "data_perp/artifacts/exact_policy_capture_feature_universe_20260727_v2/"
            "capture_feature_universe.parquet"
        ),
    )
    parser.add_argument(
        "--current-scores",
        type=Path,
        default=Path(
            "data_perp/artifacts/"
            "execution_ev_context_clean_recent_mapping_forward_july19_20260726_v1/"
            "mapped_oof.parquet"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clusters", type=int, default=4)
    parser.add_argument("--minimum-common-features", type=int, default=20)
    parser.add_argument("--minimum-selected-rows-per-week", type=int, default=100)
    parser.add_argument("--minimum-historical-weeks", type=int, default=3)
    parser.add_argument("--minimum-recent-weeks", type=int, default=2)
    parser.add_argument("--minimum-sign-consistency", type=float, default=0.75)
    parser.add_argument("--random-state", type=int, default=20260727)
    return parser


if __name__ == "__main__":
    print(json.dumps(run(_parser().parse_args()), indent=2, default=str))
