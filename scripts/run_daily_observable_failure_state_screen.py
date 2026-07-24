#!/usr/bin/env python3
"""Screen causal daily failure-state probabilities from the unified handoff.

This is deliberately a research gate, not a policy overlay.  It aggregates
only decision-time columns per side x archetype x UTC day, derives daily loss
labels after outcome resolution, and delegates chronological fitting, purging,
nonlinear MI screening and calibration to ``failure_detector``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from extreme_price_movements.unsupervised_regime_learning.failure_detector import (
    ProspectiveFailureDetectorConfig,
    chronological_failure_detection,
    is_batch_layout_dependent_ae_gmm_feature,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data_perp/reports/residual_state_full_source_regimes_20260720/jan_jun_candidates_frozen_aegmm_durable_transitions.parquet"
DEFAULT_LABELS_DIR = (
    ROOT
    / "data_perp/artifacts/20260713_s59_h5_fullthroughjul10_trailing_cost100bps_labels/labels"
)
CANONICAL_RESIDUAL_EVENT_CONTEXT = (
    "resid_event_aegmm_local_support_log1p",
    "resid_event_aegmm_gmm_entropy",
    "resid_event_aegmm_expected_market_peer_surprise",
    "resid_event_aegmm_expected_ev_timestamp_neutral_surprise",
)
RESIDUAL_EVENT_CONTEXT_KEYS = (
    "__ts__", "__symbol__", "side_name", "archetype_policy_key",
)

OUTCOME_COLUMNS = {
    "ev_after_1pct", "clean_exec", "dirty_positive", "full_path_bad_mae_1r",
    "timeout", "first_touch_bad_mae_1r", "adverse_excursion", "target",
}
IDENTITY_COLUMNS = {
    "__ts__", "timestamp", "__symbol__", "symbol", "side_name",
    "archetype_policy_key", "day", "month", "week_start",
}
LABEL_OUTCOME_COLUMNS = {
    "__u_policy_net__",
    "__long_path_clean_exec_label__",
    "__long_path_dirty_positive_label__",
    "__path_full_bad_mae_1r__",
    "__first_touch_timeout__",
    "__first_touch_mae_to_sl__",
}
LABEL_NON_OBSERVABLE_PREFIXES = (
    "__y_", "__mfe", "__mae", "__tp", "__sl", "__is_timeout__",
    "__quality", "__bars_to", "__barrier", "__n_", "__w_", "__source__",
    "__u_", "__r_", "__first_touch", "__trailing", "__path", "__long_path",
    "__max_", "__underwater", "__area_", "target__", "availability__",
)
LABEL_NON_OBSERVABLE_EXACT = {
    "candidate_id", "timeframe", "side", "__side__", "__regime_family__",
    "__archetype_label_family__", "__archetype_label_source__",
    "__archetype_policy_key__", "__archetype_policy_role__",
    "__archetype_policy_confidence__", "__archetype_policy_tp_r__",
    "__archetype_policy_sl_r__", "__archetype_policy_trail_r__",
    "__archetype_policy_max_bars_to_mfe__", "__archetype_policy_max_barrier__",
}


def _is_observable_label_feature(name: str) -> bool:
    """Keep only decision-time label-store columns for retrospective joins."""

    # Label parquet files intentionally co-locate the pre-entry feature frame
    # with many resolved path/outcome fields.  For double-underscore columns,
    # use a narrow positive contract instead of maintaining a brittle outcome
    # deny-list as labels evolve.
    if name.startswith("__") and not name.startswith(("__regime_", "__meta_raw_")):
        return False
    return (
        name not in IDENTITY_COLUMNS
        and name not in LABEL_OUTCOME_COLUMNS
        and name not in LABEL_NON_OBSERVABLE_EXACT
        and not name.startswith(LABEL_NON_OBSERVABLE_PREFIXES)
    )


def _observable_columns(frame: pd.DataFrame, maximum: int) -> list[str]:
    candidates = []
    for name in frame.columns:
        if name in IDENTITY_COLUMNS or name in OUTCOME_COLUMNS:
            continue
        if name in LABEL_OUTCOME_COLUMNS:
            continue
        if name.startswith((
            "target__", "availability__", "meta_aux_",
            *LABEL_NON_OBSERVABLE_PREFIXES,
        )):
            continue
        if is_batch_layout_dependent_ae_gmm_feature(name):
            continue
        values = pd.to_numeric(frame[name], errors="coerce")
        if values.notna().mean() >= 0.90 and values.nunique(dropna=True) >= 4:
            candidates.append(name)
    # The screen itself performs nonlinear selection; this cap protects daily
    # aggregation memory while retaining the broad observable contract.
    return candidates[:maximum]


def build_daily_state(
    frame: pd.DataFrame,
    *,
    maximum_features: int,
    feature_columns: Iterable[str] | None = None,
    negative_hit_surprise_pp: float = 0.15,
    positive_hit_surprise_pp: float = 0.15,
    relative_ev_threshold: float = 0.005,
) -> tuple[pd.DataFrame, list[str]]:
    work = frame.copy()
    work["day"] = pd.to_datetime(work["__ts__"], utc=True).dt.floor("D")
    work["side_name"] = work["side_name"].astype(str).str.lower()
    work["archetype_policy_key"] = work["archetype_policy_key"].astype(str)
    if feature_columns is None:
        features = _observable_columns(work, maximum_features)
    else:
        # Context fields are deliberately prepended to the broad candidate
        # universe. A field can therefore arrive through both routes; retain
        # its first occurrence so pandas never returns a two-dimensional
        # duplicate-name frame to the numeric coverage check.
        features = list(dict.fromkeys(name for name in feature_columns if name in work))
        features = _observable_columns(
            work.loc[:, [*IDENTITY_COLUMNS.intersection(work.columns), *features]],
            maximum_features,
        )
    group = ["day", "side_name", "archetype_policy_key"]
    numeric = work.loc[:, [*group, *features]].copy()
    numeric.loc[:, features] = numeric.loc[:, features].apply(pd.to_numeric, errors="coerce")
    # Median is robust to symbol/candidate composition.  p90 retains the
    # intensity of synchronized market shocks without using outcomes.
    median = numeric.groupby(group, observed=True)[features].median().add_prefix("state_med__")
    p90 = numeric.groupby(group, observed=True)[features].quantile(0.90).add_prefix("state_p90__")
    outcome = work.groupby(group, observed=True).agg(
        mean_ev_after_1pct=("ev_after_1pct", "mean"),
        clean_exec_rate=("clean_exec", "mean"),
        dirty_positive_rate=("dirty_positive", "mean"),
        selected_rows=("ev_after_1pct", "size"),
    )
    state = pd.concat([median, p90, outcome], axis=1).reset_index()
    # Same-day side references remove broad market movement from the residual
    # target. They are labels only: the detector never receives them as inputs.
    side_reference = (
        work.groupby(["day", "side_name"], observed=True)
        .agg(
            side_mean_ev_after_1pct=("ev_after_1pct", "mean"),
            side_clean_exec_rate=("clean_exec", "mean"),
        )
        .reset_index()
    )
    state = state.merge(
        side_reference, on=["day", "side_name"], how="left", validate="many_to_one"
    )
    state["relative_ev_after_1pct"] = (
        state["mean_ev_after_1pct"] - state["side_mean_ev_after_1pct"]
    )
    state["signed_hit_surprise"] = (
        state["clean_exec_rate"] - state["side_clean_exec_rate"]
    )
    state["adverse_event"] = state["mean_ev_after_1pct"].lt(0.0)
    state["negative_pnl_day"] = state["adverse_event"]
    state["event_block"] = np.where(state["adverse_event"], "negative_ev", "normal")
    state["failure_mode"] = pd.NA
    state["failure_mode_available_day"] = pd.NaT
    state["target__any_failure"] = state["adverse_event"]
    state["target__negative_ev_day"] = state["negative_pnl_day"]
    state["target__negative_relative_ev_day"] = state[
        "relative_ev_after_1pct"
    ].le(-abs(float(relative_ev_threshold)))
    state["target__negative_hit_surprise_day"] = state[
        "signed_hit_surprise"
    ].le(-abs(float(negative_hit_surprise_pp)))
    state["target__positive_hit_surprise_day"] = state[
        "signed_hit_surprise"
    ].ge(abs(float(positive_hit_surprise_pp)))
    state["target__failure_severity"] = (-state["mean_ev_after_1pct"]).clip(lower=0.0)
    return state, [*median.columns.tolist(), *p90.columns.tolist()]


def _label_path(labels_dir: Path, side: str, month: pd.Timestamp) -> Path:
    return labels_dir / (
        f"train_global_{side}_5_{month.year}_{month.month:02d}.parquet"
    )


def _top10_equivalent_tail(
    frame: pd.DataFrame,
    *,
    score_column: str,
    fraction_of_top30: float,
) -> pd.DataFrame:
    """Select a causal top-10%-equivalent tail from a side top-30 stream."""

    if not 0.0 < fraction_of_top30 <= 1.0:
        raise ValueError("tail fraction must be in (0, 1]")
    work = frame.copy()
    score = pd.to_numeric(work[score_column], errors="coerce")
    work["tail_rank_within_side_top30"] = (
        score.groupby([work["__ts__"], work["side_name"]], observed=True)
        .rank(method="first", ascending=False, pct=True)
    )
    return work.loc[work["tail_rank_within_side_top30"].le(fraction_of_top30)].copy()


def _read_label_observable_columns(labels_dir: Path) -> list[str]:
    sample = next(labels_dir.glob("train_global_long_5_*.parquet"), None)
    if sample is None:
        raise FileNotFoundError(f"No long label shards found in {labels_dir}")
    columns = pd.read_parquet(sample, columns=None).columns.tolist()
    return [name for name in columns if _is_observable_label_feature(name)]


def _load_monthly_frozen_history(
    candidate_root: Path,
    labels_dir: Path,
    *,
    maximum_features: int,
    tail_fraction_of_top30: float,
    negative_hit_surprise_pp: float,
    positive_hit_surprise_pp: float,
    relative_ev_threshold: float,
    residual_event_context_root: Path | None = None,
    months: set[str] | None = None,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Build daily state from frozen scores, joining outcomes only after scoring.

    Candidate ledgers are frozen model outputs.  Label shards are read solely
    for resolved targets and additional pre-entry market columns; outcome
    columns are explicitly excluded from every detector input.
    """

    root_manifest = candidate_root / "history_manifest.json"
    if not root_manifest.exists():
        raise FileNotFoundError(f"Missing frozen-history manifest: {root_manifest}")
    provenance = json.loads(root_manifest.read_text())
    if provenance.get("evidence_scope") != "frozen_backcast_diagnostic":
        raise ValueError("candidate root is not a frozen-backcast diagnostic source")
    label_features = _read_label_observable_columns(labels_dir)
    candidate_paths = sorted(candidate_root.glob("monthly/*/candidates.parquet"))
    if months:
        candidate_paths = [path for path in candidate_paths if path.parent.name in months]
    if not candidate_paths:
        raise FileNotFoundError(f"No monthly candidate ledgers under {candidate_root}")

    daily_states: list[pd.DataFrame] = []
    coverage: list[dict[str, object]] = []
    state_features: list[str] | None = None
    for candidate_path in candidate_paths:
        month = pd.Timestamp(candidate_path.parent.name + "-01", tz="UTC")
        candidates = pd.read_parquet(candidate_path)
        candidates["__ts__"] = pd.to_datetime(candidates["__ts__"], utc=True)
        candidates["side_name"] = candidates["side_name"].astype(str).str.lower()
        candidates = _top10_equivalent_tail(
            candidates,
            score_column="score_meta_base_soft_label",
            fraction_of_top30=tail_fraction_of_top30,
        )
        context_columns: list[str] = []
        if residual_event_context_root is not None:
            context_path = (
                residual_event_context_root / "monthly" / candidate_path.parent.name
                / "residual_event_context.parquet"
            )
            if not context_path.exists():
                raise FileNotFoundError(f"Missing residual-event context shard: {context_path}")
            context = pd.read_parquet(context_path)
            required_context = [*RESIDUAL_EVENT_CONTEXT_KEYS, *CANONICAL_RESIDUAL_EVENT_CONTEXT]
            missing_context = [name for name in required_context if name not in context]
            if missing_context:
                raise ValueError(f"Residual-event context missing fields: {missing_context}")
            context["__ts__"] = pd.to_datetime(context["__ts__"], utc=True)
            context["__symbol__"] = context["__symbol__"].astype(str)
            context["side_name"] = context["side_name"].astype(str).str.lower()
            context["archetype_policy_key"] = context["archetype_policy_key"].astype(str)
            candidates = candidates.merge(
                context.loc[:, required_context], on=list(RESIDUAL_EVENT_CONTEXT_KEYS), how="left", validate="m:1"
            )
            missing_rate = candidates.loc[:, list(CANONICAL_RESIDUAL_EVENT_CONTEXT)].isna().any(axis=1).mean()
            if missing_rate > 0.0:
                raise ValueError(
                    f"Residual-event context did not fully join for {candidate_path.parent.name}: "
                    f"missing_rate={missing_rate:.4f}"
                )
            context_columns = list(CANONICAL_RESIDUAL_EVENT_CONTEXT)
        candidate_feature_columns = _observable_columns(
            candidates, maximum=maximum_features
        )
        label_frames: list[pd.DataFrame] = []
        for side in ("long", "short"):
            label_path = _label_path(labels_dir, side, month)
            if not label_path.exists():
                raise FileNotFoundError(f"Missing label shard: {label_path}")
            available = pd.read_parquet(label_path, columns=None).columns
            selected = [
                "__ts__", "__symbol__", "side_name", *LABEL_OUTCOME_COLUMNS,
                *[name for name in label_features if name in available],
            ]
            labels = pd.read_parquet(label_path, columns=selected)
            labels["__ts__"] = pd.to_datetime(labels["__ts__"], utc=True)
            labels["side_name"] = labels["side_name"].astype(str).str.lower()
            label_frames.append(labels)
        labels = pd.concat(label_frames, ignore_index=True)
        join_keys = ["__ts__", "__symbol__", "side_name"]
        merged = candidates.merge(
            labels,
            on=join_keys,
            how="left",
            validate="m:1",
            suffixes=("", "__label"),
        )
        joined = merged["__u_policy_net__"].notna()
        coverage.append({
            "month": month.strftime("%Y-%m"),
            "candidate_rows_top10_equivalent": int(len(merged)),
            "outcome_joined_rows": int(joined.sum()),
            "outcome_join_rate": float(joined.mean()) if len(merged) else np.nan,
            "symbols": int(merged["__symbol__"].nunique()),
        })
        merged = merged.loc[joined].copy()
        merged["ev_after_1pct"] = (
            pd.to_numeric(merged["__u_policy_net__"], errors="coerce") - 0.01
        )
        merged["clean_exec"] = merged["__long_path_clean_exec_label__"].astype("boolean")
        merged["dirty_positive"] = merged["__long_path_dirty_positive_label__"].astype("boolean")
        merged["full_path_bad_mae_1r"] = merged["__path_full_bad_mae_1r__"].astype("boolean")
        merged["timeout"] = merged["__first_touch_timeout__"].astype("boolean")
        merged["first_touch_bad_mae_1r"] = (
            pd.to_numeric(merged["__first_touch_mae_to_sl__"], errors="coerce").ge(1.0)
        )
        merged["adverse_excursion"] = pd.to_numeric(
            merged["__first_touch_mae_to_sl__"], errors="coerce"
        )
        observable_features = [
            *context_columns,
            *candidate_feature_columns,
            *[name for name in label_features if name in merged],
        ]
        state, current_features = build_daily_state(
            merged,
            maximum_features=maximum_features,
            feature_columns=observable_features,
            negative_hit_surprise_pp=negative_hit_surprise_pp,
            positive_hit_surprise_pp=positive_hit_surprise_pp,
            relative_ev_threshold=relative_ev_threshold,
        )
        if state_features is None:
            state_features = current_features
        else:
            # Every shard uses the same schema. An intersection makes later
            # chronological screening robust to rare monthly availability gaps.
            state_features = [name for name in state_features if name in current_features]
        daily_states.append(state)

    combined = pd.concat(daily_states, ignore_index=True)
    assert state_features is not None
    # Require the final daily-state features to be broadly observed across the
    # full history. This happens after joining all months, not month-by-month.
    usable = [
        name for name in state_features
        if combined[name].notna().mean() >= 0.90
        and pd.to_numeric(combined[name], errors="coerce").nunique(dropna=True) >= 4
    ]
    return combined, usable, pd.DataFrame(coverage)


def _write_summary_tables(
    output: Path,
    *,
    state: pd.DataFrame,
    report: pd.DataFrame,
    selected: pd.DataFrame,
) -> None:
    """Persist compact, comparable summaries for residual-state follow-up."""

    if not report.empty:
        group = ["failure_mode", "side_name", "archetype_policy_key"]
        metrics = (
            report.groupby(group, observed=True, as_index=False)
            .agg(
                folds=("fold_index", "size"),
                oos_days=("oos_days", "sum"),
                event_days=("oos_positive_days", "sum"),
                average_precision=("average_precision", "mean"),
                alert_precision=("precision", "mean"),
                alert_recall=("recall", "mean"),
                alert_lift=("lift", "mean"),
                brier=("brier", "mean"),
            )
            .sort_values(["failure_mode", "average_precision"], ascending=[True, False])
        )
        metrics.to_csv(output / "oos_metrics_by_mode_side_archetype.csv", index=False)
    if not selected.empty:
        frequencies = selected.copy()
        frequencies["base_feature"] = frequencies["feature"].str.replace(
            r"^state_(?:med|p90)__", "", regex=True
        )
        frequencies = (
            frequencies.groupby(["failure_mode", "base_feature"], observed=True, as_index=False)
            .agg(
                selected_rows=("feature", "size"),
                archetype_cells=("archetype_policy_key", "nunique"),
                mean_mutual_information=("mutual_information", "mean"),
                mean_tail_lift=("tail_lift", "mean"),
                mean_screen_score=("score", "mean"),
            )
            .sort_values(
                ["failure_mode", "archetype_cells", "selected_rows", "mean_screen_score"],
                ascending=[True, False, False, False],
            )
        )
        frequencies.to_csv(output / "selected_feature_frequency_by_mode.csv", index=False)
    target_columns = [name for name in state if name.startswith("target__")]
    if target_columns:
        prevalence = (
            state.groupby(["side_name", "archetype_policy_key"], observed=True)[target_columns]
            .mean()
            .reset_index()
        )
        prevalence.to_csv(output / "daily_target_prevalence_by_side_archetype.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--candidate-root", type=Path,
        help="Frozen compact candidate-history root. Overrides --input.",
    )
    parser.add_argument(
        "--negative-hit-surprise-pp", type=float, default=0.15,
        help="Side-relative clean-rate shortfall defining a negative surprise day.",
    )
    parser.add_argument(
        "--positive-hit-surprise-pp", type=float, default=0.15,
        help="Side-relative clean-rate excess defining a positive surprise day.",
    )
    parser.add_argument(
        "--relative-ev-threshold", type=float, default=0.005,
        help="Side-relative daily EV shortfall defining an adverse residual day.",
    )
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument(
        "--residual-event-context-root", type=Path,
        help="Frozen observable residual-event context sidecar generated before outcomes are joined.",
    )
    parser.add_argument(
        "--months", nargs="*", default=[],
        help="Optional YYYY-MM shards. Useful when a frozen candidate root spans contracts under separate review.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-observable-features", type=int, default=220)
    parser.add_argument(
        "--tail-fraction-of-top30", type=float, default=1.0 / 3.0,
        help="Per timestamp x side fraction retained from a top-30 candidate stream.",
    )
    parser.add_argument("--min-train-days", type=int, default=90)
    args = parser.parse_args()
    coverage = pd.DataFrame()
    source_manifest: dict[str, object]
    if args.candidate_root is not None:
        state, features, coverage = _load_monthly_frozen_history(
            args.candidate_root,
            args.labels_dir,
            maximum_features=args.max_observable_features,
            tail_fraction_of_top30=args.tail_fraction_of_top30,
            negative_hit_surprise_pp=args.negative_hit_surprise_pp,
            positive_hit_surprise_pp=args.positive_hit_surprise_pp,
            relative_ev_threshold=args.relative_ev_threshold,
            residual_event_context_root=args.residual_event_context_root,
            months={str(month) for month in args.months} or None,
        )
        source_manifest = {
            "schema": "daily_observable_failure_state_frozen_history_v1",
            "evidence_scope": "frozen_backcast_diagnostic",
            "candidate_root": str(args.candidate_root),
            "labels_dir": str(args.labels_dir),
            "outcome_join": "labels joined only after frozen candidate scoring",
            "tail_definition": "top fraction within each timestamp x side top-30 stream",
            "tail_fraction_of_top30": args.tail_fraction_of_top30,
            "negative_hit_surprise_pp": args.negative_hit_surprise_pp,
            "positive_hit_surprise_pp": args.positive_hit_surprise_pp,
            "relative_ev_threshold": args.relative_ev_threshold,
            "detector_feature_count": len(features),
            "excluded_batch_layout_dependent_ae_gmm": True,
            "residual_event_context_root": str(args.residual_event_context_root or ""),
            "residual_event_context_columns": list(CANONICAL_RESIDUAL_EVENT_CONTEXT)
            if args.residual_event_context_root is not None else [],
            "months": list(args.months),
        }
    else:
        frame = pd.read_parquet(args.input)
        state, features = build_daily_state(
            frame, maximum_features=args.max_observable_features
        )
        source_manifest = {
            "schema": "daily_observable_failure_state_input_v1",
            "evidence_scope": "source_defined_by_input",
            "input": str(args.input),
            "detector_feature_count": len(features),
            "excluded_batch_layout_dependent_ae_gmm": True,
        }
    config = ProspectiveFailureDetectorConfig(
        min_train_days=args.min_train_days, eval_days=30, inner_validation_days=30,
        min_positive_days=4, max_features=24, mi_bins=8, lead_days=(1, 3),
    )
    predictions, report, selected = chronological_failure_detection(
        state, config=config, feature_columns=features
    )
    args.output.mkdir(parents=True, exist_ok=True)
    state.to_parquet(args.output / "daily_observable_state.parquet", index=False)
    predictions.to_parquet(args.output / "oos_failure_probabilities.parquet", index=False)
    report.to_csv(args.output / "failure_detector_report.csv", index=False)
    selected.to_csv(args.output / "selected_observable_features.csv", index=False)
    _write_summary_tables(args.output, state=state, report=report, selected=selected)
    if not coverage.empty:
        coverage.to_csv(args.output / "monthly_outcome_join_coverage.csv", index=False)
    source_manifest["daily_state_rows"] = int(len(state))
    source_manifest["prediction_rows"] = int(len(predictions))
    source_manifest["selected_feature_rows"] = int(len(selected))
    (args.output / "source_manifest.json").write_text(
        json.dumps(source_manifest, indent=2, default=str) + "\n"
    )


if __name__ == "__main__":
    main()
