#!/usr/bin/env python3
"""Build a complete causal parent stream for V9 + MLP monthly walk-forward.

Raw meta scores are converted to historical percentiles using only earlier OOS
scores. Residual-state assignments are regenerated from the frozen quarterly
state revision valid at each decision month, so missing state artifacts cannot
silently remove calendar days.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from scripts.run_meta_v9_ev_mapped_side_residual_ablation import (
    _augment_from_feature_store,
)


KEYS = ["__ts__", "__symbol__", "side_name", "archetype_policy_key"]
OUTCOMES = [
    "ev_after_1pct",
    "clean_exec",
    "dirty_positive",
    "full_path_bad_mae_1r",
    "timeout",
]
SCORE_COLUMN_ALIASES = ("score_meta_base_soft_label", "score")


def _read_prediction_shards(directory: Path) -> pd.DataFrame:
    paths = sorted(directory.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No prediction shards under {directory}")
    columns = [
        "__ts__",
        "__symbol__",
        "side_name",
        "archetype_policy_key",
        "__archetype_policy_key__",
        *SCORE_COLUMN_ALIASES,
        *OUTCOMES,
    ]
    parts: list[pd.DataFrame] = []
    for path in paths:
        available = set(pq.read_schema(path).names)
        part = pd.read_parquet(path, columns=[c for c in columns if c in available])
        if "archetype_policy_key" not in part:
            part["archetype_policy_key"] = part["__archetype_policy_key__"]
        if "score_meta_base_soft_label" not in part:
            if "score" not in part:
                raise ValueError(f"Prediction shard has no supported score: {path}")
            part["score_meta_base_soft_label"] = pd.to_numeric(
                part["score"], errors="coerce"
            ).astype(np.float32)
        part["__ts__"] = pd.to_datetime(part["__ts__"], utc=True, errors="coerce")
        parts.append(part)
    return (
        pd.concat(parts, ignore_index=True, copy=False)
        .drop_duplicates(KEYS, keep="last")
        .sort_values(KEYS, kind="stable")
        .reset_index(drop=True)
    )


def _causal_historical_ranks(
    prior_scores: np.ndarray,
    evaluation: pd.DataFrame,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    result = np.full(len(evaluation), np.nan, dtype=np.float32)
    score = pd.to_numeric(
        evaluation["score_meta_base_soft_label"], errors="coerce"
    ).to_numpy(dtype=np.float64, copy=False)
    timestamps = pd.to_datetime(evaluation["__ts__"], utc=True)
    months = timestamps.dt.strftime("%Y-%m")
    history = np.asarray(prior_scores, dtype=np.float64)
    history = history[np.isfinite(history)]
    rows: list[dict[str, object]] = []
    for month in sorted(months.unique()):
        current = months.eq(month).to_numpy()
        reference = np.sort(history)
        values = score[current]
        finite = np.isfinite(values)
        ranks = np.full(len(values), np.nan, dtype=np.float64)
        if len(reference):
            left = np.searchsorted(reference, values[finite], side="left")
            right = np.searchsorted(reference, values[finite], side="right")
            ranks[finite] = (left + right) / (2.0 * len(reference))
        result[current] = ranks.astype(np.float32)
        rows.append(
            {
                "month": month,
                "reference_rows": int(len(reference)),
                "scored_rows": int(current.sum()),
                "rank_coverage": float(np.isfinite(ranks).mean()),
            }
        )
        history = np.concatenate([history, values[finite]])
    return result, rows


def _causal_trailing_day_ranks(
    prior: pd.DataFrame,
    evaluation: pd.DataFrame,
    *,
    lookback_days: int,
    min_reference_rows: int = 1,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Return daily scores ranked against a strictly earlier trailing window.

    The reference for a decision day contains only scores from the preceding
    ``lookback_days`` UTC days.  No score from the current day is admitted,
    including earlier intraday rows.  This matches a pre-open, global
    historical-rank contract and is safe for the candidate stream whose raw
    meta score is named simply ``score``.
    """

    if int(lookback_days) < 1:
        raise ValueError("lookback_days must be positive")
    value_column = "score_meta_base_soft_label"
    required = {"__ts__", value_column}
    for name, frame in (("prior", prior), ("evaluation", evaluation)):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} rank frame missing columns: {sorted(missing)}")

    eval_ts = pd.to_datetime(evaluation["__ts__"], utc=True, errors="coerce")
    eval_day = eval_ts.dt.floor("D")
    result = np.full(len(evaluation), np.nan, dtype=np.float32)
    scores = pd.to_numeric(evaluation[value_column], errors="coerce").to_numpy(
        dtype=np.float64, copy=False
    )
    prior_frame = prior.loc[:, ["__ts__", value_column]].copy()
    prior_frame["__ts__"] = pd.to_datetime(prior_frame["__ts__"], utc=True, errors="coerce")
    prior_frame["__day"] = prior_frame["__ts__"].dt.floor("D")
    prior_frame["__score"] = pd.to_numeric(
        prior_frame[value_column], errors="coerce"
    )
    prior_by_day = {
        pd.Timestamp(day): group["__score"].to_numpy(dtype=np.float64, copy=False)
        for day, group in prior_frame.loc[
            prior_frame["__day"].notna()
        ].groupby("__day", sort=True, observed=True)
    }

    history: deque[tuple[pd.Timestamp, np.ndarray]] = deque()
    # Prior rows are eligible only if they are before the first evaluated day.
    # The loop then adds each evaluated day only after all of its rows were
    # ranked, which prevents same-day information leakage.
    days = sorted(pd.Timestamp(day) for day in eval_day.dropna().unique())
    summaries: list[dict[str, object]] = []
    for day in days:
        lower = day - pd.Timedelta(days=int(lookback_days))
        while history and history[0][0] < lower:
            history.popleft()
        # Add eligible external history on demand. This is kept separate from
        # evaluated days so a shared source directory is safe.
        if not history:
            for prior_day, values in prior_by_day.items():
                if lower <= prior_day < day:
                    finite = values[np.isfinite(values)]
                    if len(finite):
                        history.append((prior_day, finite))
        reference = (
            np.sort(np.concatenate([values for _, values in history]))
            if history
            else np.empty(0, dtype=np.float64)
        )
        current = eval_day.eq(day).to_numpy()
        values = scores[current]
        finite = np.isfinite(values)
        ranks = np.full(len(values), np.nan, dtype=np.float64)
        if len(reference) >= int(min_reference_rows):
            left = np.searchsorted(reference, values[finite], side="left")
            right = np.searchsorted(reference, values[finite], side="right")
            ranks[finite] = (left + right) / (2.0 * len(reference))
        result[current] = ranks.astype(np.float32)
        current_finite = values[finite]
        if len(current_finite):
            history.append((day, current_finite))
        summaries.append(
            {
                "day": day.strftime("%Y-%m-%d"),
                "reference_rows": int(len(reference)),
                "scored_rows": int(current.sum()),
                "rank_coverage": float(np.isfinite(ranks).mean()),
            }
        )
    return result, summaries


def _state_input_columns(state: object) -> list[str]:
    columns: list[str] = []
    for model in state.local_models.values():
        columns.extend(model.feature_columns)
    if state.market_model is not None:
        columns.extend(state.market_model.feature_columns)
    return list(dict.fromkeys(columns))


def _read_context(
    path: Path,
    columns: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    available = set(pq.read_schema(path).names)
    required = ["__ts__", "__symbol__", "side_name"]
    read_columns = list(dict.fromkeys([*required, *columns]))
    missing = sorted(set(required).difference(available))
    if missing:
        raise ValueError(f"Context source missing keys: {missing}")
    timestamp_type = pq.read_schema(path).field("__ts__").type
    if timestamp_type.tz is None:
        filter_start = start.tz_convert(None).to_pydatetime()
        filter_end = end.tz_convert(None).to_pydatetime()
    else:
        filter_start = start.to_pydatetime()
        filter_end = end.to_pydatetime()
    frame = pd.read_parquet(
        path,
        columns=[c for c in read_columns if c in available],
        filters=[("__ts__", ">=", filter_start), ("__ts__", "<", filter_end)],
    )
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    return frame.drop_duplicates(required, keep="last")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-prediction-shards", type=Path, default=None)
    parser.add_argument("--prior-meta-prediction-shards", type=Path, default=None)
    parser.add_argument(
        "--meta-predictions-file",
        type=Path,
        default=None,
        help="Single promoted meta OOS parquet with frozen rank/probability columns.",
    )
    parser.add_argument(
        "--rank-column",
        default="historical_rank",
        help="Frozen train-reference rank in --meta-predictions-file.",
    )
    parser.add_argument(
        "--hit-probability-column",
        default="hit_probability",
        help="Observable soft-hit score in --meta-predictions-file.",
    )
    parser.add_argument(
        "--causal-rank-mode",
        choices=("monthly_prior_oos", "trailing_days_global"),
        default="monthly_prior_oos",
        help=(
            "Rank contract when raw prediction shards are supplied. The default "
            "uses earlier OOS months only; trailing_days_global ranks each UTC "
            "day solely against the prior trailing-day score distribution."
        ),
    )
    parser.add_argument(
        "--rank-lookback-days",
        type=int,
        default=8,
        help="Prior calendar-day lookback for --causal-rank-mode=trailing_days_global.",
    )
    parser.add_argument(
        "--rank-min-reference-rows",
        type=int,
        default=1,
        help="Minimum strictly-prior score rows required to emit a trailing-day rank.",
    )
    parser.add_argument("--context-source", type=Path, default=None)
    parser.add_argument(
        "--feature-store-dir",
        type=Path,
        default=None,
        help="Logical per-symbol feature store used when no context parquet is supplied.",
    )
    parser.add_argument("--historical-state-artifact", type=Path, required=True)
    parser.add_argument("--state-revision-april", type=Path, required=True)
    parser.add_argument("--state-revision-july", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--evaluation-start",
        default=None,
        help=(
            "Optional inclusive UTC boundary applied to meta-prediction-shards. "
            "This permits one shard directory to provide both prior history and "
            "the bounded evaluation stream without copying parquet files."
        ),
    )
    parser.add_argument(
        "--evaluation-end",
        default=None,
        help="Optional exclusive UTC boundary applied to meta-prediction-shards.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.meta_predictions_file is not None:
        available = set(pq.read_schema(args.meta_predictions_file).names)
        required = [*KEYS, *OUTCOMES, args.rank_column, args.hit_probability_column]
        missing = sorted(set(required).difference(available))
        if missing:
            raise ValueError(
                f"Promoted meta predictions are missing required columns: {missing}"
            )
        evaluation = pd.read_parquet(args.meta_predictions_file, columns=required)
        evaluation = evaluation.rename(
            columns={
                args.rank_column: "historical_rank",
                args.hit_probability_column: "hit_probability",
            }
        )
        evaluation["__ts__"] = pd.to_datetime(
            evaluation["__ts__"], utc=True, errors="coerce"
        )
        evaluation = evaluation.drop_duplicates(KEYS, keep="last").sort_values(
            KEYS, kind="stable"
        )
        prior = pd.DataFrame()
    else:
        if args.meta_prediction_shards is None or args.prior_meta_prediction_shards is None:
            raise ValueError(
                "Provide either --meta-predictions-file or both prediction shard arguments."
            )
        evaluation = _read_prediction_shards(args.meta_prediction_shards)
        prior = _read_prediction_shards(args.prior_meta_prediction_shards)
    requested_start = (
        pd.Timestamp(args.evaluation_start, tz="UTC")
        if args.evaluation_start is not None
        else None
    )
    requested_end = (
        pd.Timestamp(args.evaluation_end, tz="UTC")
        if args.evaluation_end is not None
        else None
    )
    if requested_start is not None:
        evaluation = evaluation.loc[evaluation["__ts__"].ge(requested_start)]
    if requested_end is not None:
        evaluation = evaluation.loc[evaluation["__ts__"].lt(requested_end)]
    if evaluation.empty:
        raise ValueError(
            "No evaluation rows remain after applying the requested date window"
        )
    evaluation_start = evaluation["__ts__"].min()
    if args.meta_predictions_file is None:
        prior = prior.loc[prior["__ts__"].lt(evaluation_start)].copy()
        if args.causal_rank_mode == "monthly_prior_oos":
            ranks, rank_manifest = _causal_historical_ranks(
                pd.to_numeric(
                    prior["score_meta_base_soft_label"], errors="coerce"
                ).to_numpy(),
                evaluation,
            )
            rank_contract = "monthly prior-OOS empirical CDF; current month excluded"
        else:
            ranks, rank_manifest = _causal_trailing_day_ranks(
                prior,
                evaluation,
                lookback_days=int(args.rank_lookback_days),
                min_reference_rows=int(args.rank_min_reference_rows),
            )
            rank_contract = (
                "global trailing prior-day empirical CDF; current UTC day excluded; "
                f"lookback_days={int(args.rank_lookback_days)}"
            )
        evaluation["historical_rank"] = ranks
        evaluation["hit_probability"] = pd.to_numeric(
            evaluation["score_meta_base_soft_label"], errors="coerce"
        ).astype(np.float32)
    else:
        evaluation["historical_rank"] = pd.to_numeric(
            evaluation["historical_rank"], errors="coerce"
        ).astype(np.float32)
        evaluation["hit_probability"] = pd.to_numeric(
            evaluation["hit_probability"], errors="coerce"
        ).astype(np.float32)
        rank_manifest = [
            {
                "source": str(args.meta_predictions_file),
                "rank_column": str(args.rank_column),
                "rows": int(len(evaluation)),
                "rank_coverage": float(evaluation["historical_rank"].notna().mean()),
            }
        ]
        rank_contract = "promoted meta train-reference rank; frozen before OOS scoring"

    april_state = joblib.load(args.state_revision_april)
    july_state = joblib.load(args.state_revision_july)
    state_inputs = list(
        dict.fromkeys(
            [*_state_input_columns(april_state), *_state_input_columns(july_state)]
        )
    )
    if args.feature_store_dir is not None:
        observable = _augment_from_feature_store(
            evaluation.copy(), args.feature_store_dir, state_inputs
        )
        context_contract = {
            "feature_store_dir": str(args.feature_store_dir),
            "requested_features": len(state_inputs),
        }
    else:
        if args.context_source is None:
            raise ValueError("Provide --context-source or --feature-store-dir.")
        context = _read_context(
            args.context_source,
            state_inputs,
            evaluation["__ts__"].min().floor("D"),
            evaluation["__ts__"].max().ceil("D") + pd.Timedelta(days=1),
        )
        observable = evaluation.merge(
            context,
            on=["__ts__", "__symbol__", "side_name"],
            how="left",
            validate="many_to_one",
        )
        context_contract = {"context_source": str(args.context_source)}
    state_parts: list[pd.DataFrame] = []
    state_rows: list[dict[str, object]] = []
    july_start = pd.Timestamp("2026-07-01", tz="UTC")
    for name, state, mask in (
        ("2026-04_revision", april_state, observable["__ts__"].lt(july_start)),
        ("2026-07_revision", july_state, observable["__ts__"].ge(july_start)),
    ):
        part = observable.loc[mask].copy()
        if part.empty:
            continue
        part["score"] = pd.to_numeric(
            part["historical_rank"], errors="coerce"
        ).astype(np.float32)
        assessed = state.annotate_outcomes_for_assessment(part)
        transformed = state.transform_oos(
            part.drop(columns=[c for c in OUTCOMES if c in part], errors="ignore")
        )
        transformed = transformed.reset_index(drop=True)
        keys = part[KEYS].reset_index(drop=True)
        outcomes = part[OUTCOMES].reset_index(drop=True)
        assessment_columns = [
            column
            for column in assessed.columns
            if column not in part.columns or column == "resid_event_class"
        ]
        assessment = assessed[assessment_columns].reset_index(drop=True)
        state_part = pd.concat(
            [keys, outcomes, assessment, transformed], axis=1
        )
        state_part = state_part.loc[:, ~state_part.columns.duplicated(keep="last")]
        state_parts.append(state_part)
        state_rows.append(
            {
                "revision": name,
                "rows": int(len(part)),
                "days": int(part["__ts__"].dt.floor("D").nunique()),
                "input_feature_column_coverage": float(
                    len(set(state_inputs).intersection(part.columns))
                    / max(len(state_inputs), 1)
                ),
                "input_value_coverage": float(
                    part.reindex(columns=state_inputs).notna().mean(axis=0).mean()
                ),
            }
        )
    generated_state = pd.concat(state_parts, ignore_index=True, copy=False)
    if len(generated_state) != len(evaluation):
        raise AssertionError(
            f"state coverage mismatch: {len(generated_state):,} != {len(evaluation):,}"
        )

    historical_schema = set(pq.read_schema(args.historical_state_artifact).names)
    generated_schema = set(generated_state.columns)
    shared = sorted(historical_schema.intersection(generated_schema))
    historical = pd.read_parquet(args.historical_state_artifact, columns=shared)
    historical["__ts__"] = pd.to_datetime(historical["__ts__"], utc=True)
    historical = historical.loc[historical["__ts__"].lt(evaluation_start)]
    complete_state = pd.concat(
        [historical, generated_state.reindex(columns=shared)],
        ignore_index=True,
        copy=False,
    ).drop_duplicates(KEYS, keep="last")

    parent_columns = [*KEYS, "historical_rank", "hit_probability", *OUTCOMES]
    parent_path = args.output_dir / "complete_parent_oos_predictions.parquet"
    state_path = args.output_dir / "complete_oos_residual_event_states.parquet"
    evaluation[parent_columns].to_parquet(parent_path, index=False, compression="zstd")
    complete_state.to_parquet(state_path, index=False, compression="zstd")
    day_counts = (
        evaluation.assign(day=evaluation["__ts__"].dt.floor("D"))
        .groupby(evaluation["__ts__"].dt.strftime("%Y-%m"), observed=True)
        .agg(rows=("__ts__", "size"), days=("day", "nunique"))
        .reset_index(names="month")
    )
    day_counts.to_csv(args.output_dir / "coverage_by_month.csv", index=False)
    manifest = {
        "schema": "complete_meta_residual_parent_v1",
        "parent_rows": int(len(evaluation)),
        "parent_start": evaluation["__ts__"].min().isoformat(),
        "parent_end": evaluation["__ts__"].max().isoformat(),
        "requested_evaluation_start": (
            requested_start.isoformat() if requested_start is not None else None
        ),
        "requested_evaluation_end_exclusive": (
            requested_end.isoformat() if requested_end is not None else None
        ),
        "causal_rank_contract": rank_contract,
        "causal_rank_mode": (
            str(args.causal_rank_mode)
            if args.meta_predictions_file is None
            else "frozen_input"
        ),
        "rank_folds": rank_manifest,
        "state_revisions": state_rows,
        "sources": {
            "meta_prediction_shards": str(args.meta_prediction_shards),
            "prior_meta_prediction_shards": str(args.prior_meta_prediction_shards),
            "meta_predictions_file": str(args.meta_predictions_file),
            "context": context_contract,
            "historical_state_artifact": str(args.historical_state_artifact),
            "state_revision_april": str(args.state_revision_april),
            "state_revision_july": str(args.state_revision_july),
        },
        "parent_path": str(parent_path),
        "state_path": str(state_path),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
