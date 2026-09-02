#!/usr/bin/env python3
"""Outcome-free, live-aligned historical state-recurrence diagnostic.

This deliberately answers only whether the observable May--July candidate
environment is represented by earlier observable states.  It does not load a
score, target, label-resolution time, policy result, calendar feature, or
sample weight.  Weekly blocks are evaluation partitions and audit labels only.

The state basis is the 23 raw point-in-time fields verified against the frozen
current feature universe.  For every evaluation week and side, the scaler and
KMeans geometry are fit solely on rows whose execution decision precedes the
week.  Historical and earlier current rows are then transformed by that
week-specific frozen geometry for occupancy and drift diagnostics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.causal_execution_regimes import CausalRegimeStateModel  # noqa: E402


SCHEMA = "historical_state_recurrence_observability_v2"
IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")
DECISION = "execution_decision_utc"
SOURCE_DECISION = "__decision_ts__"
HISTORICAL_OOF = ROOT / (
    "data_perp/artifacts/janapr2025_execution_ev_exact1m_two_layer_oof_20260727_v1/"
    "two_layer_direct_ev_strict_oof.parquet"
)
CURRENT_CANONICAL = ROOT / (
    "data_perp/artifacts/execution_ev_canonical_exact_policy_regime_input_20260727_v3/"
    "joined.parquet"
)
CAPTURE_UNIVERSE = ROOT / (
    "data_perp/artifacts/exact_policy_capture_feature_universe_20260727_v2/"
    "capture_feature_universe.parquet"
)
CAPTURE_MANIFEST = ROOT / (
    "data_perp/artifacts/exact_policy_capture_feature_universe_20260727_v2/"
    "feature_universe_manifest.json"
)
SOURCE_ROOT = ROOT / (
    "data_perp/artifacts/20260720_s59_h5_signalclose_causal_trailing_cost100bps_labels_v2/"
    "labels"
)
DEFAULT_OUTPUT = ROOT / (
    "data_perp/artifacts/historical_state_recurrence_observability_20260727_v2"
)

# These are the causal source-ledger fields that also occur in the frozen live
# capture universe, have full current coverage, and passed the paired current
# source/store Spearman >= 0.95 parity audit.  The list is intentionally fixed:
# no outcome, calendar, score, policy, or feature-selection signal can enter.
STATE_FEATURES = (
    "atr_compression_ratio",
    "breakout_24h",
    "dir_path_edge_2h",
    "dir_path_risk_skew_2h",
    "distance_to_resistance_daily_vwap_atr",
    "dn_vol",
    "dn_vol_6",
    "ema20_slope_5h",
    "fvg",
    "jump_intensity",
    "log_bars_since_above_1atr",
    "log_bars_since_above_2atr",
    "memory_asymmetry_1ATR",
    "memory_asymmetry_2ATR",
    "memory_asymmetry_3ATR",
    "press_12",
    "range_12h_pct",
    "range_16h_pct",
    "range_24h_pct",
    "rejection_proxy",
    "spread_proxy_abs_return_bps_robust_z",
    "trend_acceleration",
    "zscore_price_200",
)
LIVE_PREFIX = "capture_candidate__"


def _utc(values: Iterable[Any] | pd.Series, *, column: str) -> pd.Series:
    result = pd.Series(pd.to_datetime(values, utc=True, errors="coerce"))
    if result.isna().any():
        raise ValueError(f"{column} contains null or invalid UTC timestamps")
    return result


def _normalise_identity(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    missing = sorted(set(IDENTITY) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} missing identity columns: {missing}")
    out = frame.copy()
    out["__ts__"] = _utc(out["__ts__"], column=f"{name}.__ts__").to_numpy()
    out["__symbol__"] = (
        out["__symbol__"].astype(str).str.strip().str.replace("/", "_", regex=False)
    )
    out["side_name"] = out["side_name"].astype(str).str.lower()
    if not out["side_name"].isin(("long", "short")).all():
        raise ValueError(f"{name}.side_name must be canonical long/short")
    out["candidate_id"] = out["candidate_id"].astype(str)
    if out.duplicated(list(IDENTITY), keep=False).any():
        raise ValueError(f"{name} contains duplicate exact identities")
    return out


def source_paths(source_root: Path, months: Iterable[str]) -> list[Path]:
    paths = [
        source_root / f"train_global_{side}_5_{month}.parquet"
        for month in months
        for side in ("long", "short")
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing source ledgers: " + ", ".join(missing))
    return paths


def load_source_rows(paths: Iterable[Path]) -> pd.DataFrame:
    columns = [*IDENTITY, SOURCE_DECISION, *STATE_FEATURES]
    parts = [pd.read_parquet(path, columns=columns) for path in paths]
    source = _normalise_identity(pd.concat(parts, ignore_index=True), name="source")
    source[DECISION] = _utc(source.pop(SOURCE_DECISION), column=SOURCE_DECISION).to_numpy()
    expected = source["__ts__"] + pd.Timedelta(hours=1)
    if not source[DECISION].equals(expected):
        raise ValueError("source decision timestamp must equal signal timestamp + 1h")
    for feature in STATE_FEATURES:
        source[feature] = pd.to_numeric(source[feature], errors="coerce").astype(np.float32)
    return source.sort_values([DECISION, "__symbol__", "side_name", "candidate_id"], kind="stable").reset_index(drop=True)


def _strict_join(
    population: pd.DataFrame,
    source: pd.DataFrame,
    *,
    name: str,
    require_all_population: bool = True,
) -> pd.DataFrame:
    population = _normalise_identity(population, name=name)
    if DECISION not in population:
        raise ValueError(f"{name} missing {DECISION}")
    population[DECISION] = _utc(population[DECISION], column=f"{name}.{DECISION}").to_numpy()
    merged = population.merge(
        source.loc[:, [*IDENTITY, DECISION, *STATE_FEATURES]],
        on=list(IDENTITY),
        how="left",
        validate="one_to_one",
        suffixes=("_population", "_source"),
        indicator=True,
    )
    if require_all_population and not merged["_merge"].eq("both").all():
        raise ValueError(f"{name}/source identity join is incomplete")
    if not merged["_merge"].eq("both").all():
        merged = merged.loc[merged["_merge"].eq("both")].copy()
    if not merged[f"{DECISION}_population"].equals(merged[f"{DECISION}_source"]):
        raise ValueError(f"{name}/source execution decision timestamps disagree")
    merged = merged.drop(columns=["_merge", f"{DECISION}_population"]).rename(
        columns={f"{DECISION}_source": DECISION}
    )
    return merged


def load_panels(
    *,
    historical_oof: Path,
    current_canonical: Path,
    source_root: Path,
) -> tuple[pd.DataFrame, dict[str, int]]:
    historical = pd.read_parquet(
        historical_oof,
        columns=[*IDENTITY, DECISION, "base_margin_to_cutoff"],
    )
    historical = historical.loc[
        pd.to_numeric(historical["base_margin_to_cutoff"], errors="coerce").ge(0.0)
    ].copy()
    historical_source = load_source_rows(
        source_paths(source_root, ("2025_01", "2025_02", "2025_03", "2025_04"))
    )
    historical = _strict_join(historical, historical_source, name="historical_oof")
    historical["panel_origin"] = "historical_strict_oof_top30"

    current = pd.read_parquet(
        current_canonical,
        columns=[*IDENTITY, DECISION, "base_margin_to_cutoff"],
    )
    if not pd.to_numeric(current["base_margin_to_cutoff"], errors="coerce").ge(0.0).all():
        raise ValueError("current canonical panel must be the post-base top30 candidate stream")
    current_source = load_source_rows(
        source_paths(source_root, ("2026_05", "2026_06", "2026_07"))
    )
    current = _strict_join(current, current_source, name="current_canonical")
    current["panel_origin"] = "current_canonical_top30"

    keep = [*IDENTITY, DECISION, *STATE_FEATURES, "panel_origin"]
    combined = pd.concat(
        [historical.loc[:, keep], current.loc[:, keep]], ignore_index=True
    )
    combined = combined.sort_values(
        [DECISION, "__symbol__", "side_name", "candidate_id"], kind="stable"
    ).reset_index(drop=True)
    if combined.duplicated(list(IDENTITY), keep=False).any():
        raise ValueError("combined panel contains duplicate exact identities")
    return combined, {
        "historical_rows": int(len(historical)),
        "current_rows": int(len(current)),
        "combined_rows": int(len(combined)),
    }


def verify_live_alignment(
    current_panel: pd.DataFrame,
    *,
    capture_universe: Path,
    capture_manifest: Path,
    minimum_coverage: float,
    minimum_spearman: float,
) -> pd.DataFrame:
    """Verify that diagnostic fields have a live causal counterpart.

    This validation is deliberately outside the state fit.  It is a provenance
    check only and cannot select, weight, rank, or transform any state row.
    """

    manifest = json.loads(capture_manifest.read_text(encoding="utf-8"))
    eligible = set(manifest["eligible_full_period_feature_columns"])
    required_live = {f"{LIVE_PREFIX}{feature}" for feature in STATE_FEATURES}
    if missing := sorted(required_live - eligible):
        raise ValueError(f"state fields are not live-eligible: {missing}")
    live = pd.read_parquet(capture_universe, columns=[*IDENTITY, *sorted(required_live)])
    live = _normalise_identity(live, name="capture_universe")
    current = current_panel.loc[current_panel["panel_origin"].eq("current_canonical_top30")].copy()
    joined = current.merge(live, on=list(IDENTITY), how="left", validate="one_to_one")
    if len(joined) != len(current):
        raise ValueError("current/live alignment join changed row count")
    rows: list[dict[str, Any]] = []
    for feature in STATE_FEATURES:
        source = pd.to_numeric(joined[feature], errors="coerce")
        live_values = pd.to_numeric(joined[f"{LIVE_PREFIX}{feature}"], errors="coerce")
        valid = source.notna() & live_values.notna()
        coverage = float(valid.mean())
        spearman = float(source.loc[valid].corr(live_values.loc[valid], method="spearman")) if valid.sum() >= 3 else float("nan")
        if coverage < float(minimum_coverage) or not np.isfinite(spearman) or spearman < float(minimum_spearman):
            raise ValueError(
                f"live parity fails for {feature}: coverage={coverage:.6f}, spearman={spearman:.6f}"
            )
        rows.append(
            {
                "feature": feature,
                "live_feature": f"{LIVE_PREFIX}{feature}",
                "rows": int(len(joined)),
                "paired_coverage": coverage,
                "spearman": spearman,
            }
        )
    return pd.DataFrame(rows)


def weekly_blocks(
    frame: pd.DataFrame, *, start: pd.Timestamp, end: pd.Timestamp
) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("weekly boundaries must be timezone-aware UTC")
    observed_end = _utc(frame[DECISION], column=DECISION).max() + pd.Timedelta(nanoseconds=1)
    final = min(end, observed_end)
    blocks: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
    cursor = start
    while cursor < final:
        stop = min(cursor + pd.Timedelta(days=7), final)
        if _utc(frame[DECISION], column=DECISION).between(cursor, stop, inclusive="left").any():
            blocks.append((cursor, stop, f"block_{cursor:%Y%m%d}"))
        cursor = stop
    return blocks


def diagnose_expanding_states(
    frame: pd.DataFrame,
    *,
    first_evaluation: pd.Timestamp,
    end: pd.Timestamp,
    min_state_fit_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit side-local state geometry only on pre-block observable rows."""

    decisions = _utc(frame[DECISION], column=DECISION)
    summaries: list[dict[str, Any]] = []
    emitted: list[pd.DataFrame] = []
    for block_start, block_end, block_name in weekly_blocks(
        frame, start=first_evaluation, end=end
    ):
        evaluation_mask = decisions.between(block_start, block_end, inclusive="left")
        fit_mask = decisions.lt(block_start)
        for side in ("long", "short"):
            fit = frame.loc[fit_mask & frame["side_name"].eq(side)].copy()
            evaluation = frame.loc[evaluation_mask & frame["side_name"].eq(side)].copy()
            row: dict[str, Any] = {
                "evaluation_block": block_name,
                "start_utc": block_start,
                "end_exclusive_utc": block_end,
                "side_name": side,
                "state_fit_rows": int(len(fit)),
                "evaluation_rows": int(len(evaluation)),
                "status": "evaluated",
            }
            if len(fit) < int(min_state_fit_rows) or evaluation.empty:
                row["status"] = "insufficient_prior_state_rows_or_evaluation"
                summaries.append(row)
                continue
            model = CausalRegimeStateModel.fit(fit, STATE_FEATURES)
            transformed_fit = model.transform(fit.loc[:, STATE_FEATURES])
            transformed_eval = model.transform(evaluation.loc[:, STATE_FEATURES])
            drift = model.training_drift(transformed_fit, transformed_eval)
            state_counts = transformed_eval["causal_regime_state"].value_counts().sort_index()
            row.update(
                {
                    "selected_k": int(model.selected_k),
                    "state_distribution_js": float(drift["state_distribution_js"]),
                    "eval_minimum_state_occupancy": float(drift["eval_minimum_state_occupancy"]),
                    "eval_mean_ood_z": float(drift["eval_mean_ood_z"]),
                    "eval_p95_ood_z": float(drift["eval_p95_ood_z"]),
                    "eval_ood_distance_gt_p99_fraction": float(
                        (transformed_eval["causal_regime_distance_percentile"] > 0.99).mean()
                    ),
                    "eval_mean_entropy": float(transformed_eval["causal_regime_entropy"].mean()),
                    "eval_p95_entropy": float(transformed_eval["causal_regime_entropy"].quantile(0.95)),
                    "eval_mean_top2_margin": float(transformed_eval["causal_regime_top2_margin"].mean()),
                    "evaluation_state_counts": json.dumps(
                        {str(key): int(value) for key, value in state_counts.items()}, sort_keys=True
                    ),
                    "state_fit_max_decision_utc": _utc(fit[DECISION], column=DECISION).max(),
                }
            )
            summaries.append(row)
            audit_columns = [*IDENTITY, DECISION, "panel_origin"]
            diagnostic = pd.concat(
                [evaluation.loc[:, audit_columns].reset_index(drop=True), transformed_eval.reset_index(drop=True)],
                axis=1,
            )
            diagnostic["evaluation_block"] = block_name
            diagnostic["state_fit_cutoff_utc"] = block_start
            emitted.append(diagnostic)
    summary = pd.DataFrame(summaries)
    rows = pd.concat(emitted, ignore_index=True) if emitted else pd.DataFrame()
    return summary, rows


def run(args: argparse.Namespace) -> dict[str, Path]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    panel, counts = load_panels(
        historical_oof=args.historical_oof,
        current_canonical=args.current_canonical,
        source_root=args.source_root,
    )
    alignment = verify_live_alignment(
        panel,
        capture_universe=args.capture_universe,
        capture_manifest=args.capture_manifest,
        minimum_coverage=args.minimum_live_coverage,
        minimum_spearman=args.minimum_live_spearman,
    )
    first = pd.Timestamp(args.first_evaluation, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    summary, rows = diagnose_expanding_states(
        panel,
        first_evaluation=first,
        end=end,
        min_state_fit_rows=args.min_state_fit_rows,
    )
    args.output_dir.mkdir(parents=True)
    alignment_path = args.output_dir / "live_alignment.csv"
    summary_path = args.output_dir / "state_diagnostics_by_week.csv"
    rows_path = args.output_dir / "weekly_state_diagnostic_rows.parquet"
    manifest_path = args.output_dir / "manifest.json"
    alignment.to_csv(alignment_path, index=False)
    summary.to_csv(summary_path, index=False)
    rows.to_parquet(rows_path, index=False, compression="zstd")
    manifest = {
        "schema": SCHEMA,
        "status": "completed_outcome_free_observability_diagnostic_not_economic_evidence",
        "contract": {
            "state_inputs": list(STATE_FEATURES),
            "state_inputs_count": len(STATE_FEATURES),
            "state_source_time": "source-ledger signal __ts__, strictly one hour before execution decision",
            "fit": "per-side expanding CausalRegimeStateModel fit on decision times strictly before each evaluation block",
            "excluded": ["scores", "outcomes", "targets", "label resolution", "calendar features", "sample weights", "policy actions"],
            "blocks": "weekly UTC evaluation partitions only; never state features or labels",
            "live_alignment": "paired source/capture validation only; values never enter selection or fitting",
        },
        "inputs": {
            "historical_oof": str(args.historical_oof),
            "current_canonical": str(args.current_canonical),
            "source_root": str(args.source_root),
            "capture_universe": str(args.capture_universe),
            "capture_manifest": str(args.capture_manifest),
        },
        "panel_rows": counts,
        "evaluation": {
            "first_evaluation": first.isoformat(),
            "end": end.isoformat(),
            "min_state_fit_rows": int(args.min_state_fit_rows),
            "evaluated_side_blocks": int(summary["status"].eq("evaluated").sum()),
        },
        "outputs": {
            "live_alignment": str(alignment_path.resolve()),
            "state_diagnostics": str(summary_path.resolve()),
            "diagnostic_rows": str(rows_path.resolve()),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    return {"manifest": manifest_path, "alignment": alignment_path, "summary": summary_path, "rows": rows_path}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-oof", type=Path, default=HISTORICAL_OOF)
    parser.add_argument("--current-canonical", type=Path, default=CURRENT_CANONICAL)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--capture-universe", type=Path, default=CAPTURE_UNIVERSE)
    parser.add_argument("--capture-manifest", type=Path, default=CAPTURE_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--first-evaluation", default="2026-05-05")
    parser.add_argument("--end", default="2026-07-20")
    parser.add_argument("--min-state-fit-rows", type=int, default=500)
    parser.add_argument("--minimum-live-coverage", type=float, default=0.99)
    parser.add_argument("--minimum-live-spearman", type=float, default=0.95)
    return parser


if __name__ == "__main__":
    print(json.dumps({key: str(value) for key, value in run(_parser().parse_args()).items()}, indent=2))
