#!/usr/bin/env python3
"""Adjacent-July execution-EV adaptation with explicit zero fallback.

The frozen input score is the exact causal 21-day side x predicted-archetype
EV correction.  An adapter for a July block may use only resolved residuals
from the immediately preceding July block.  The first block and every
under-supported side therefore receive an exact zero correction.

A frozen, outcome-free state basis is fitted once on pre-July rows.  A
state-conditional residual is evaluated only when prior July blocks show
recurring, stable and materially different state mappings.  Calendar labels
and regime/state sample weights are never used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.causal_execution_regimes import (  # noqa: E402
    CausalRegimeStateModel,
)
from scripts.diagnose_causal_execution_ev_regimes import (  # noqa: E402
    STATE_FEATURES,
)
from scripts.run_execution_ev_recent_residual_shrinkage_ablation import (  # noqa: E402
    policy_global_topk_mask,
)


SCHEMA = "adjacent_july_state_adapter_ablation_v1"
DECISION = "execution_decision_utc"
RESOLUTION = "execution_label_end_utc"
TARGET = "execution_net_ev_12h"
SIDE = "side_name"
BASE_SCORE = (
    "catboost__residual__without_hpo__all_features"
    "__recent_ev_catboost_predicted_archetype"
)
IDENTITY = ("__ts__", "__symbol__", SIDE, "candidate_id")
DEFAULT_SCORES = ROOT / (
    "data_perp/artifacts/"
    "execution_ev_context_clean_exact_recent_correction_forward_july19_20260726_v2/"
    "mapped_oof_and_forward.parquet"
)
DEFAULT_FEATURES = ROOT / (
    "data_perp/artifacts/"
    "execution_ev_context_clean_regime_input_forward_july19_20260726_v1/"
    "joined.parquet"
)
DEFAULT_MARKET_STATE_ROWS = ROOT / (
    "data_perp/artifacts/"
    "execution_ev_raw_market_state_transition_heads_20260726_v1/"
    "raw_market_state_transition_rows.parquet"
)
DEFAULT_OUTPUT = ROOT / (
    "data_perp/artifacts/"
    "adjacent_july_state_adapter_ablation_20260726_v1"
)
JULY_BLOCKS = (
    ("july_01_05", "2026-07-01T00:00:00Z", "2026-07-06T00:00:00Z"),
    ("july_06_12", "2026-07-06T00:00:00Z", "2026-07-13T00:00:00Z"),
    ("july_13_19", "2026-07-13T00:00:00Z", "2026-07-20T00:00:00Z"),
)
SHRINKAGES = (0.0, 0.25, 0.50, 1.0)


def _as_utc(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="raise")


def load_frame(
    scores_path: Path,
    features_path: Path,
    market_state_path: Path | None = None,
) -> pd.DataFrame:
    scores = pd.read_parquet(scores_path)
    features = pd.read_parquet(features_path)
    required_score = {
        *IDENTITY,
        DECISION,
        RESOLUTION,
        TARGET,
        BASE_SCORE,
        "evaluation_origin",
    }
    required_feature = {"candidate_id", *STATE_FEATURES}
    if missing := required_score.difference(scores.columns):
        raise ValueError(f"score input missing: {sorted(missing)}")
    if missing := required_feature.difference(features.columns):
        raise ValueError(f"feature input missing: {sorted(missing)}")
    if scores["candidate_id"].duplicated().any():
        raise ValueError("score input candidate_id must be unique")
    if features["candidate_id"].duplicated().any():
        raise ValueError("feature input candidate_id must be unique")
    feature_columns = [
        column
        for column in features.columns
        if column not in scores.columns or column == "candidate_id"
    ]
    work = scores.merge(
        features.loc[:, feature_columns],
        on="candidate_id",
        how="inner",
        validate="one_to_one",
    )
    if market_state_path is not None:
        market = pd.read_parquet(market_state_path)
        market_columns = [
            column
            for column in market.columns
            if column == "candidate_id"
            or (
                column.startswith("mkt_state__")
                and column.endswith("__h0")
            )
        ]
        if "candidate_id" not in market_columns:
            raise ValueError("market-state input lacks candidate_id")
        if market["candidate_id"].duplicated().any():
            raise ValueError("market-state candidate_id must be unique")
        work = work.merge(
            market.loc[:, market_columns],
            on="candidate_id",
            how="inner",
            validate="one_to_one",
        )
    for column in ("__ts__", DECISION, RESOLUTION):
        work[column] = _as_utc(work[column])
    numeric = work.loc[:, [TARGET, BASE_SCORE, *STATE_FEATURES]].apply(
        pd.to_numeric, errors="coerce"
    )
    finite = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    work = work.loc[finite].copy()
    work.loc[:, numeric.columns] = numeric.loc[finite]
    work = work.sort_values(
        [DECISION, "__symbol__", SIDE, "candidate_id"], kind="stable"
    ).reset_index(drop=True)
    return work


def add_frozen_states(
    frame: pd.DataFrame,
    *,
    reference_end: pd.Timestamp = pd.Timestamp("2026-07-01T00:00:00Z"),
    market_coverage_threshold: float = 0.95,
) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    """Fit one pre-July state basis per side and freeze it for every July block."""

    pre_july = frame.loc[_as_utc(frame[DECISION]).lt(reference_end)]
    candidate_market_inputs = [
        column
        for column in frame.columns
        if column.startswith("mkt_state__") and column.endswith("__h0")
    ]
    market_coverage = {
        column: float(
            pd.to_numeric(pre_july[column], errors="coerce").notna().mean()
        )
        for column in candidate_market_inputs
    }
    selected_market_inputs = [
        column
        for column, coverage in market_coverage.items()
        if coverage >= float(market_coverage_threshold)
    ]
    state_source_features = [*STATE_FEATURES, *selected_market_inputs]
    output_parts: list[pd.DataFrame] = []
    reports: dict[str, Any] = {}
    state_inputs: list[str] = []
    decision = _as_utc(frame[DECISION])
    for side in ("long", "short"):
        local = frame.loc[frame[SIDE].astype(str).eq(side)].copy()
        local_decision = _as_utc(local[DECISION])
        reference = local.loc[local_decision.lt(reference_end)].copy()
        if len(reference) < 500:
            raise ValueError(f"insufficient pre-July state support for {side}")
        model = CausalRegimeStateModel.fit(reference, state_source_features)
        transformed = model.transform(local).reset_index(drop=True)
        local = local.reset_index(drop=True)
        local = pd.concat([local, transformed], axis=1)
        stable = list(model.predictor_feature_columns)
        state_inputs = sorted(set(state_inputs).union(stable))
        reports[side] = {
            "reference_rows": int(len(reference)),
            "reference_decision_max": local_decision.loc[
                local_decision.lt(reference_end)
            ].max().isoformat(),
            "selected_k": int(model.selected_k),
            "selection": model.selection,
            "predictor_features": stable,
            "state_source_features": state_source_features,
            "market_input_coverage": market_coverage,
            "selected_market_inputs": selected_market_inputs,
            "market_coverage_threshold": float(market_coverage_threshold),
            "state_id_contract": (
                "frozen pre-July per-side basis; diagnostic/specialist routing "
                "only, never numeric ordinal input"
            ),
        }
        output_parts.append(local)
    return (
        pd.concat(output_parts, ignore_index=True).sort_values(
            [DECISION, "__symbol__", SIDE, "candidate_id"], kind="stable"
        ).reset_index(drop=True),
        reports,
        state_inputs,
    )


def _fit_adapter(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    min_rows: int,
    iterations: int,
    seed: int,
    n_jobs: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit side-local recent residual models; unsupported sides return zero."""

    from catboost import CatBoostRegressor

    delta = np.zeros(len(evaluation), dtype=float)
    report: dict[str, Any] = {}
    for side_index, side in enumerate(("long", "short")):
        train_mask = train[SIDE].astype(str).eq(side).to_numpy()
        eval_mask = evaluation[SIDE].astype(str).eq(side).to_numpy()
        fit = train.loc[train_mask]
        future = evaluation.loc[eval_mask]
        if len(fit) < int(min_rows) or future.empty:
            report[side] = {
                "status": "zero_fallback",
                "reason": (
                    "insufficient_prior_adjacent_july_rows"
                    if len(fit) < int(min_rows)
                    else "no_evaluation_rows"
                ),
                "train_rows": int(len(fit)),
                "evaluation_rows": int(len(future)),
            }
            continue
        x_train = fit.loc[:, list(feature_columns)].apply(
            pd.to_numeric, errors="coerce"
        )
        x_eval = future.loc[:, list(feature_columns)].apply(
            pd.to_numeric, errors="coerce"
        )
        medians = x_train.median(axis=0).fillna(0.0)
        x_train = x_train.fillna(medians)
        x_eval = x_eval.fillna(medians)
        residual = (
            fit[TARGET].to_numpy(dtype=float)
            - fit[BASE_SCORE].to_numpy(dtype=float)
        )
        model = CatBoostRegressor(
            loss_function="RMSE",
            iterations=int(iterations),
            learning_rate=0.035,
            depth=4,
            l2_leaf_reg=15.0,
            random_strength=0.5,
            bagging_temperature=1.0,
            bootstrap_type="Bayesian",
            random_seed=int(seed + side_index),
            thread_count=int(n_jobs),
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(x_train, residual)
        prediction = np.asarray(model.predict(x_eval), dtype=float)
        prediction = np.clip(prediction, -0.01, 0.01)
        delta[eval_mask] = prediction
        report[side] = {
            "status": "fit_on_prior_adjacent_july_residuals",
            "train_rows": int(len(fit)),
            "evaluation_rows": int(len(future)),
            "mean_delta": float(np.mean(prediction)),
            "mean_abs_delta": float(np.mean(np.abs(prediction))),
            "clip": [-0.01, 0.01],
        }
    return delta, report


def specialist_eligibility(
    prior: pd.DataFrame,
    *,
    min_state_rows: int = 100,
    min_recurring_states: int = 2,
    min_effect_range: float = 0.002,
    min_week_rank_correlation: float = 0.50,
) -> dict[str, Any]:
    """Training-only gate for state-specialist evaluation.

    A state is recurring when it has adequate support in at least two completed
    July blocks.  State mappings must retain sign and rank across the two most
    recent blocks and differ by at least 20 bps within a side.
    """

    if prior.empty or "july_block" not in prior:
        return {
            "eligible": False,
            "sides": {
                side: {
                    "eligible": False,
                    "reason": "fewer_than_two_completed_july_blocks",
                }
                for side in ("long", "short")
            },
            "decision": "zero_fallback_do_not_test_state_specialist",
        }
    result: dict[str, Any] = {"eligible": True, "sides": {}}
    for side in ("long", "short"):
        local = prior.loc[prior[SIDE].astype(str).eq(side)].copy()
        if local["july_block"].nunique() < 2:
            side_report = {
                "eligible": False,
                "reason": "fewer_than_two_completed_july_blocks",
            }
            result["sides"][side] = side_report
            result["eligible"] = False
            continue
        grouped = (
            local.assign(
                economic_residual=local[TARGET].to_numpy(dtype=float)
                - local[BASE_SCORE].to_numpy(dtype=float)
            )
            .groupby(["july_block", "causal_regime_state"], observed=True)
            ["economic_residual"]
            .agg(["mean", "count"])
            .reset_index()
        )
        blocks = list(local["july_block"].drop_duplicates())[-2:]
        pivot_mean = grouped.pivot(
            index="causal_regime_state", columns="july_block", values="mean"
        )
        pivot_count = grouped.pivot(
            index="causal_regime_state", columns="july_block", values="count"
        )
        common = [
            state
            for state in pivot_mean.index
            if all(
                block in pivot_mean.columns
                and pd.notna(pivot_mean.loc[state, block])
                and float(pivot_count.loc[state, block]) >= int(min_state_rows)
                for block in blocks
            )
        ]
        if len(common) >= 2:
            left = pivot_mean.loc[common, blocks[-2]].to_numpy(dtype=float)
            right = pivot_mean.loc[common, blocks[-1]].to_numpy(dtype=float)
            rank_correlation = float(
                pd.Series(left).corr(pd.Series(right), method="spearman")
            )
            sign_consistency = float(np.mean(np.sign(left) == np.sign(right)))
            effect_range = float(
                min(np.ptp(left), np.ptp(right))
            )
        else:
            rank_correlation = float("nan")
            sign_consistency = 0.0
            effect_range = 0.0
        eligible = bool(
            len(common) >= int(min_recurring_states)
            and np.isfinite(rank_correlation)
            and rank_correlation >= float(min_week_rank_correlation)
            and sign_consistency >= 0.75
            and effect_range >= float(min_effect_range)
        )
        result["sides"][side] = {
            "eligible": eligible,
            "completed_blocks": blocks,
            "recurring_states": [int(value) for value in common],
            "rank_correlation": (
                rank_correlation if np.isfinite(rank_correlation) else None
            ),
            "sign_consistency": sign_consistency,
            "minimum_within_block_effect_range": effect_range,
            "thresholds": {
                "min_state_rows": int(min_state_rows),
                "min_recurring_states": int(min_recurring_states),
                "min_effect_range": float(min_effect_range),
                "min_week_rank_correlation": float(min_week_rank_correlation),
                "min_sign_consistency": 0.75,
            },
        }
        result["eligible"] = bool(result["eligible"] and eligible)
    result["decision"] = (
        "test_state_specialist"
        if result["eligible"]
        else "zero_fallback_do_not_test_state_specialist"
    )
    return result


def _state_prior_delta(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    eligibility: dict[str, Any],
    *,
    parent_shrink: float = 0.50,
    prior_strength: float = 500.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    delta = np.zeros(len(evaluation), dtype=float)
    if not bool(eligibility.get("eligible")):
        return delta, {
            "status": "zero_fallback",
            "reason": "recurring_state_gate_failed",
        }
    side_report: dict[str, Any] = {}
    for side in ("long", "short"):
        train_mask = train[SIDE].astype(str).eq(side)
        eval_mask = evaluation[SIDE].astype(str).eq(side)
        fit = train.loc[train_mask].copy()
        fit["economic_residual"] = (
            fit[TARGET].to_numpy(dtype=float)
            - fit[BASE_SCORE].to_numpy(dtype=float)
        )
        stats = fit.groupby("causal_regime_state", observed=True)[
            "economic_residual"
        ].agg(["mean", "count"])
        correction: dict[int, float] = {}
        for state, row in stats.iterrows():
            support_shrink = float(row["count"] / (row["count"] + prior_strength))
            correction[int(state)] = float(
                parent_shrink * support_shrink * row["mean"]
            )
        values = (
            evaluation.loc[eval_mask, "causal_regime_state"]
            .map(correction)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        delta[np.flatnonzero(eval_mask.to_numpy())] = values
        side_report[side] = {
            "status": "frozen_state_conditional_shrunk_residual",
            "train_rows": int(len(fit)),
            "state_corrections": correction,
        }
    return np.clip(delta, -0.01, 0.01), {
        "status": "state_specialist_tested",
        "sides": side_report,
    }


def _metrics(selected: pd.DataFrame) -> dict[str, Any]:
    if selected.empty:
        return {
            "selected_rows": 0,
            "mean_net_ev": None,
            "mean_net_ev_bps": None,
            "positive_rate": None,
            "sum_net_ev": 0.0,
        }
    return {
        "selected_rows": int(len(selected)),
        "mean_net_ev": float(selected[TARGET].mean()),
        "mean_net_ev_bps": float(10_000.0 * selected[TARGET].mean()),
        "positive_rate": float(selected[TARGET].gt(0.0).mean()),
        "sum_net_ev": float(selected[TARGET].sum()),
    }


def evaluate_predictions(
    predictions: pd.DataFrame,
    *,
    top_k_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    weekly_rows: list[dict[str, Any]] = []
    pooled_rows: list[dict[str, Any]] = []
    for arm, group in predictions.groupby("arm", sort=True):
        for block, local in group.groupby("july_block", sort=False):
            mask = policy_global_topk_mask(local, "score", top_k_fraction)
            weekly_rows.append(
                {
                    "arm": arm,
                    "july_block": block,
                    "scope": "weekly_global_topk_diagnostic",
                    "evaluation_rows": int(len(local)),
                    **_metrics(local.loc[mask]),
                }
            )
        global_mask = policy_global_topk_mask(group, "score", top_k_fraction)
        fixed_global = group.assign(pooled_global_selected=global_mask)
        latest_block = JULY_BLOCKS[-1][0]
        segments = [
            ("all_july", fixed_global),
            (
                "latest_july_block",
                fixed_global.loc[fixed_global["july_block"].eq(latest_block)],
            ),
        ]
        for segment, local in segments:
            for scope in ("pooled", "long", "short"):
                scoped = (
                    local
                    if scope == "pooled"
                    else local.loc[local[SIDE].astype(str).eq(scope)]
                )
                selected = scoped.loc[scoped["pooled_global_selected"]]
                pooled_rows.append(
                    {
                        "arm": arm,
                        "segment": segment,
                        "scope": scope,
                        "evaluation_rows": int(len(scoped)),
                        "selection_coverage": float(
                            len(selected) / max(len(scoped), 1)
                        ),
                        **_metrics(selected),
                    }
                )
    return pd.DataFrame(weekly_rows), pd.DataFrame(pooled_rows)


def run(args: argparse.Namespace) -> dict[str, Path]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    market_state_path = (
        args.market_state_rows
        if args.market_state_rows is not None and args.market_state_rows.is_file()
        else None
    )
    frame = load_frame(args.scores, args.features, market_state_path)
    frame, state_report, state_inputs = add_frozen_states(
        frame,
        market_coverage_threshold=args.market_coverage_threshold,
    )
    selected_market_inputs = sorted(
        {
            column
            for side_report in state_report.values()
            for column in side_report.get("selected_market_inputs", [])
        }
    )
    adapter_features = list(
        dict.fromkeys(
            [
                *STATE_FEATURES,
                *selected_market_inputs,
                *state_inputs,
            ]
        )
    )
    prediction_parts: list[pd.DataFrame] = []
    block_reports: dict[str, Any] = {}
    decision = _as_utc(frame[DECISION])
    resolution = _as_utc(frame[RESOLUTION])
    completed_parts: list[pd.DataFrame] = []
    previous_start: pd.Timestamp | None = None
    previous_end: pd.Timestamp | None = None
    for block_index, (name, start_raw, end_raw) in enumerate(JULY_BLOCKS):
        start = pd.Timestamp(start_raw)
        end = pd.Timestamp(end_raw)
        evaluation = frame.loc[decision.ge(start) & decision.lt(end)].copy()
        if evaluation.empty:
            continue
        evaluation["july_block"] = name
        if previous_start is None or previous_end is None:
            train = frame.iloc[0:0].copy()
        else:
            train = frame.loc[
                decision.ge(previous_start)
                & decision.lt(previous_end)
                & resolution.lt(start)
            ].copy()
            train["july_block"] = JULY_BLOCKS[block_index - 1][0]
        delta, adapter_report = _fit_adapter(
            train,
            evaluation,
            adapter_features,
            min_rows=args.min_adapter_rows,
            iterations=args.iterations,
            seed=args.random_state + 100 * block_index,
            n_jobs=args.n_jobs,
        )
        prior = (
            pd.concat(completed_parts, ignore_index=True)
            if completed_parts
            else frame.iloc[0:0].copy()
        )
        eligibility = specialist_eligibility(
            prior,
            min_state_rows=args.min_state_rows,
        )
        specialist_delta, specialist_report = _state_prior_delta(
            train,
            evaluation,
            eligibility,
        )
        arms: list[tuple[str, np.ndarray]] = [
            (f"adapter_shrink_{shrink:.2f}", shrink * delta)
            for shrink in SHRINKAGES
        ]
        arms.append(("state_specialist_shrunk", specialist_delta))
        identity_columns = [
            column
            for column in (
                *IDENTITY,
                DECISION,
                RESOLUTION,
                TARGET,
                "evaluation_origin",
                "causal_regime_state",
            )
            if column in evaluation
        ]
        for arm, correction in arms:
            part = evaluation.loc[:, identity_columns].copy()
            part["july_block"] = name
            part["arm"] = arm
            part["base_score"] = evaluation[BASE_SCORE].to_numpy(dtype=float)
            part["adapter_delta_before_shrink"] = delta
            part["applied_correction"] = correction
            part["score"] = part["base_score"] + correction
            prediction_parts.append(part)
        completed_parts.append(evaluation)
        block_reports[name] = {
            "start": start.isoformat(),
            "end_exclusive": end.isoformat(),
            "evaluation_rows": int(len(evaluation)),
            "prior_adjacent_rows": int(len(train)),
            "max_prior_label_resolution": (
                _as_utc(train[RESOLUTION]).max().isoformat()
                if len(train)
                else None
            ),
            "adapter": adapter_report,
            "specialist_eligibility": eligibility,
            "specialist": specialist_report,
        }
        previous_start = start
        previous_end = end
    predictions = pd.concat(prediction_parts, ignore_index=True)
    weekly, pooled = evaluate_predictions(
        predictions, top_k_fraction=args.top_k_fraction
    )
    args.output_dir.mkdir(parents=True)
    paths = {
        "predictions": args.output_dir / "adjacent_july_predictions.parquet",
        "weekly": args.output_dir / "weekly_metrics.csv",
        "pooled": args.output_dir / "pooled_global_metrics.csv",
        "report": args.output_dir / "report.json",
    }
    predictions.to_parquet(paths["predictions"], index=False)
    weekly.to_csv(paths["weekly"], index=False)
    pooled.to_csv(paths["pooled"], index=False)
    report = {
        "schema": SCHEMA,
        "status": "completed_research_oos_not_promotion_eligible",
        "contract": {
            "base_score": BASE_SCORE,
            "base_provenance": (
                "historical outer OOF plus frozen-final-fit forward OOS; exact "
                "causal 21d side x predicted-archetype EV correction"
            ),
            "adapter": (
                "side-local residual model trained only on immediately prior "
                "July block rows whose 12h label resolved before evaluation"
            ),
            "zero_fallback": (
                "first July block and every side below min support receive "
                "exactly zero correction"
            ),
            "state_basis": (
                "one outcome-free pre-July fit per side, frozen across all "
                "July blocks; no calendar/regime sample weights"
            ),
            "selection": (
                "one pooled global top-k across all adjacent-week OOS rows; "
                "weekly results are stability diagnostics only"
            ),
        },
        "sources": {
            "scores": str(args.scores.resolve()),
            "features": str(args.features.resolve()),
            "market_state_rows": (
                str(market_state_path.resolve())
                if market_state_path is not None
                else None
            ),
        },
        "rows": int(len(frame)),
        "adapter_features": adapter_features,
        "state_fit": state_report,
        "blocks": block_reports,
        "arms": [f"adapter_shrink_{value:.2f}" for value in SHRINKAGES]
        + ["state_specialist_shrunk"],
        "search_breadth": {
            "fixed_adapter_shrinkages": list(SHRINKAGES),
            "specialist_gate": "one predeclared eligibility contract",
        },
        "outputs": {key: str(path.resolve()) for key, path in paths.items()},
    }
    paths["report"].write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return paths


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument(
        "--market-state-rows", type=Path, default=DEFAULT_MARKET_STATE_ROWS
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--iterations", type=int, default=160)
    parser.add_argument("--min-adapter-rows", type=int, default=500)
    parser.add_argument("--min-state-rows", type=int, default=100)
    parser.add_argument("--market-coverage-threshold", type=float, default=0.95)
    parser.add_argument("--top-k-fraction", type=float, default=0.10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=3)
    return parser


def main() -> None:
    paths = run(_parser().parse_args())
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
