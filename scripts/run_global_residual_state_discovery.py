#!/usr/bin/env python3
"""Run archetype-specific residual-state discovery from frozen predictions.

The command is intentionally staged.  ``events`` and ``audit`` are cheap enough
to run during iteration; latent-model fitting is delegated to the companion
rolling-origin runner so July remains an untouched generalization period.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.config import CFG  # noqa: E402
from extreme_price_movements.residual_state_discovery import (  # noqa: E402
    FEATURE_CONCEPTS,
    ReliabilityEventConfig,
    audit_feature_concepts,
    discover_reliability_events,
    feature_quality_metrics,
)
from scripts.score_compare_meta_residual_july_oos import (  # noqa: E402
    _append_store_features,
)

DEFAULT_LEDGER = ROOT / (
    "data_perp/reports/train_meta_residual_archetype_enhancement_20260711_v1/"
    "champion_frozen_single_source_202501_20260710/"
    "frozen_champion_single_source_ledger.parquet"
)
DEFAULT_FEATURE_ROOT = ROOT / "data_perp/features/20260710_170000"
DEFAULT_CANDIDATE_ROOT = DEFAULT_LEDGER.parent / "candidate_shards"
DEFAULT_OUTPUT = ROOT / "data_perp/reports/global_residual_state_discovery_20260711_v1"
LEDGER_COLUMNS = (
    "__ts__",
    "__symbol__",
    "side_name",
    "archetype_policy_key",
    "hit_probability",
    "ev_after_1pct",
    "clean_exec",
    "dirty_positive",
    "first_touch_bad_mae_1r",
    "full_path_bad_mae_1r",
    "timeout",
    "selected_for_monitor",
    "outcomes_available",
    "historical_rank",
    "base_score",
    "score_meta_base_soft_label",
    "production_adjusted_rank",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8"
    )


def _configured_feature_names(value: Any, key: str = "") -> set[str]:
    names: set[str] = set()
    key_lower = str(key).lower()
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            names.update(_configured_feature_names(child_value, str(child_key)))
    elif isinstance(value, (list, tuple, set)) and (
        "feature" in key_lower or key_lower.endswith("keys")
    ):
        names.update(str(item) for item in value if isinstance(item, str))
    return names


def _feature_schema(
    feature_root: Path, pattern: str = "*.parquet"
) -> tuple[set[str], list[dict[str, Any]]]:
    files = sorted(feature_root.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No symbol feature files under {feature_root}")
    columns: set[str] = set()
    schemas: list[dict[str, Any]] = []
    for path in files:
        parquet = pq.ParquetFile(path)
        names = set(parquet.schema_arrow.names)
        columns.update(names)
        schemas.append(
            {
                "path": str(path),
                "rows": int(parquet.metadata.num_rows),
                "columns": int(len(names)),
            }
        )
    return columns, schemas


def _candidate_feature_coverage(
    candidate_root: Path,
    requested: Iterable[str],
) -> dict[str, float]:
    names = sorted(set(map(str, requested)))
    finite = {name: 0 for name in names}
    total = {name: 0 for name in names}
    for path in sorted(candidate_root.glob("candidates_*.parquet")):
        available = set(pq.ParquetFile(path).schema_arrow.names)
        columns = [name for name in names if name in available]
        if not columns:
            continue
        part = pd.read_parquet(path, columns=columns)
        for name in columns:
            values = pd.to_numeric(part[name], errors="coerce")
            finite[name] += int(np.isfinite(values.to_numpy(dtype=float)).sum())
            total[name] += int(len(values))
    return {
        name: float(finite[name] / total[name]) if total[name] else 0.0
        for name in names
    }


def _feature_store_sample_coverage(
    candidate_root: Path,
    feature_root: Path,
    requested: Iterable[str],
) -> dict[str, float]:
    names = sorted(set(map(str, requested)))
    samples: list[pd.DataFrame] = []
    for path in sorted(candidate_root.glob("candidates_*.parquet")):
        available = set(pq.ParquetFile(path).schema_arrow.names)
        keys = [
            name for name in ("__ts__", "__symbol__", "side_name") if name in available
        ]
        part = (
            pd.read_parquet(path, columns=keys)
            .iloc[:: max(1, pq.ParquetFile(path).metadata.num_rows // 1500)]
            .head(1500)
        )
        samples.append(part)
    sample = pd.concat(samples, ignore_index=True) if samples else pd.DataFrame()
    if sample.empty:
        return {name: 0.0 for name in names}
    enriched, _ = _append_store_features(sample, feature_root, names)
    return {
        name: float(pd.to_numeric(enriched.get(name), errors="coerce").notna().mean())
        if name in enriched
        else 0.0
        for name in names
    }


def _feature_store_quality_sample(
    candidate_root: Path,
    feature_root: Path,
    requested: Iterable[str],
    ledger_path: Path | None = None,
) -> pd.DataFrame:
    names = sorted(set(map(str, requested)))
    samples: list[pd.DataFrame] = []
    for path in sorted(candidate_root.glob("candidates_*.parquet")):
        rows = int(pq.ParquetFile(path).metadata.num_rows)
        keys = ["__ts__", "__symbol__", "side_name"]
        part = (
            pd.read_parquet(path, columns=keys).iloc[:: max(1, rows // 1200)].head(1200)
        )
        samples.append(part)
    sample = pd.concat(samples, ignore_index=True)
    if ledger_path is not None and ledger_path.exists():
        july = pd.read_parquet(
            ledger_path, columns=["__ts__", "__symbol__", "side_name"]
        )
        july["__ts__"] = pd.to_datetime(july["__ts__"], utc=True, errors="coerce")
        july = july.loc[july["__ts__"].ge(pd.Timestamp("2026-07-01", tz="UTC"))]
        if not july.empty:
            sample = pd.concat(
                [sample, july.iloc[:: max(1, len(july) // 3000)].head(3000)],
                ignore_index=True,
            )
    enriched, _ = _append_store_features(sample, feature_root, names)
    enriched["__ts__"] = pd.to_datetime(enriched["__ts__"], utc=True, errors="coerce")
    return enriched


def _load_selected_ledger(path: Path, start: str, end: str) -> pd.DataFrame:
    available = set(pq.ParquetFile(path).schema_arrow.names)
    columns = [name for name in LEDGER_COLUMNS if name in available]
    frame = pd.read_parquet(path, columns=columns)
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    mask = frame["__ts__"].ge(pd.Timestamp(start, tz="UTC"))
    mask &= frame["__ts__"].lt(pd.Timestamp(end, tz="UTC"))
    if "selected_for_monitor" in frame.columns:
        mask &= frame["selected_for_monitor"].fillna(False).astype(bool)
    # ``outcomes_available`` only documents the appended July tail.  Historical
    # backcast rows carry complete realized fields but intentionally leave that
    # provenance flag null, so the realized EV itself is the full-history gate.
    mask &= pd.to_numeric(frame["ev_after_1pct"], errors="coerce").notna()
    return frame.loc[mask].reset_index(drop=True)


def _known_period_coverage(events: pd.DataFrame) -> pd.DataFrame:
    periods = (
        (
            "aug_10_14_2025",
            "2025-08-10",
            "2025-08-14",
            "late_trend_dispersion_or_narrow_btc_led_rally",
            "asset_minus_market_return|breadth_participation|price_minus_oi_recovery|dispersion_change",
        ),
        (
            "sep_22_2025",
            "2025-09-22",
            "2025-09-22",
            "broad_leveraged_long_liquidation",
            "price_down_oi_down_breadth|oi_flush_acceleration|funding_crowding|downside_synchronization",
        ),
        (
            "year_end_2025",
            "2025-12-29",
            "2026-01-03",
            "thin_liquidity_year_end_false_recovery",
            "volume_participation|volume_concentration|oi_concentration|range_per_unit_volume",
        ),
        (
            "mar_11_2026",
            "2026-03-11",
            "2026-03-11",
            "headline_transition_or_unresolved_risk_off",
            "price_recovery|oi_recovery|funding_sign_transition|downside_correlation",
        ),
        (
            "may_06_2026",
            "2026-05-06",
            "2026-05-06",
            "localized_questionable_long_or_residual_bucket_instability",
            "asset_minus_market_state|missingness|dispersion|liquidity",
        ),
        (
            "jun_05_06_2026",
            "2026-06-05",
            "2026-06-06",
            "systemic_deleveraging_with_premature_long_recovery",
            "oi_flush_age|oi_deceleration|breadth_recovery|downside_synchronization",
        ),
        (
            "jul_03_2026",
            "2026-07-03",
            "2026-07-03",
            "positive_clean_surprise_but_negative_payoff_asymmetry",
            "winner_loser_magnitude|mae_mfe|time_to_hit|conditional_ev",
        ),
        (
            "jul_08_2026",
            "2026-07-08",
            "2026-07-08",
            "risk_off_shock_with_direction_payoff_mismatch",
            "shock_intensity|downside_synchronization|range_expansion|oi_response|side_payoff_asymmetry",
        ),
    )
    rows: list[dict[str, Any]] = []
    event_start = pd.to_datetime(events.get("event_start"), utc=True, errors="coerce")
    event_end = pd.to_datetime(events.get("event_end"), utc=True, errors="coerce")
    for name, start, end, hypothesis, expected_features in periods:
        start_ts = pd.Timestamp(start, tz="UTC")
        end_ts = pd.Timestamp(end, tz="UTC")
        overlap = events.loc[event_start.le(end_ts) & event_end.ge(start_ts)]
        adverse = (
            overlap.loc[overlap["event_class"].isin(["adverse", "payoff_disagreement"])]
            if "event_class" in overlap
            else overlap.iloc[0:0]
        )
        rows.append(
            {
                "period": name,
                "start": start_ts,
                "end": end_ts,
                "hypothesis": hypothesis,
                "expected_feature_families": expected_features,
                "detected_events": int(len(overlap)),
                "eligible_events": int(overlap.get("discovery_eligible", False).sum()),
                "event_ids": "|".join(map(str, overlap.get("event_id", []))),
                "adverse_events": int(len(adverse)),
                "adverse_event_ids": "|".join(map(str, adverse.get("event_id", []))),
                "max_priority": float(overlap["event_priority"].max())
                if len(overlap)
                else np.nan,
                "max_adverse_priority": float(adverse["adverse_priority"].max())
                if len(adverse)
                else np.nan,
                "mean_ev": float(overlap["mean_ev"].mean()) if len(overlap) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _event_month_summary(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    work = events.copy()
    work["calendar_month"] = (
        pd.to_datetime(work["event_start"], utc=True).dt.to_period("M").astype(str)
    )
    return (
        work.groupby(["calendar_month", "side_name"], observed=True)
        .agg(
            event_count=("event_id", "nunique"),
            eligible_events=("discovery_eligible", "sum"),
            bootstrap_survival_rate=("bootstrap_survival", "mean"),
            mean_event_ev=("mean_ev", "mean"),
            worst_event_ev=("worst_ev", "min"),
            median_duration_days=("event_duration_days", "median"),
            selected_rows=("selected_rows", "sum"),
        )
        .reset_index()
    )


def _path_economic_month_summary(cells: pd.DataFrame) -> pd.DataFrame:
    """Summarize mutually exclusive path/economics mechanisms by local stream."""
    if cells.empty:
        return pd.DataFrame()
    work = cells.copy()
    work["calendar_month"] = (
        pd.to_datetime(work["day"], utc=True).dt.to_period("M").astype(str)
    )
    rate_columns = [
        "acute_adverse_rate",
        "slow_timeout_loss_rate",
        "clean_negative_ev_rate",
        "dirty_negative_ev_rate",
        "durable_clean_positive_rate",
    ]
    available = [name for name in rate_columns if name in work.columns]
    aggregations: dict[str, tuple[str, str]] = {
        "selected_rows": ("selected_rows", "sum"),
        "mean_ev_after_cost": ("mean_ev_after_cost", "mean"),
        "signed_hit_surprise": ("signed_hit_surprise", "mean"),
    }
    aggregations.update({name: (name, "mean") for name in available})
    return (
        work.groupby(
            ["calendar_month", "side_name", "archetype_policy_key"],
            observed=True,
            sort=True,
        )
        .agg(**aggregations)
        .reset_index()
    )


def _event_mechanism_summary(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    mechanism = "state_failure_mechanism"
    if mechanism not in events.columns:
        return pd.DataFrame()
    return (
        events.groupby(
            ["side_name", "archetype_policy_key", mechanism],
            observed=True,
            sort=True,
        )
        .agg(
            event_count=("event_id", "nunique"),
            eligible_events=("discovery_eligible", "sum"),
            selected_rows=("selected_rows", "sum"),
            mean_event_ev=("mean_ev", "mean"),
            worst_event_ev=("worst_ev", "min"),
            median_autocorrelation=("autocorrelation_strength", "median"),
        )
        .reset_index()
    )


def run_events(args: argparse.Namespace, output: Path) -> dict[str, Any]:
    selected = _load_selected_ledger(Path(args.ledger), args.start, args.end)
    config = ReliabilityEventConfig(
        causal_min_days=int(args.causal_min_days),
        bootstrap_draws=int(args.bootstrap_draws),
    )
    result = discover_reliability_events(selected, config)
    result.daily_cells.to_parquet(
        output / "daily_reliability_cells.parquet", index=False
    )
    result.events.to_csv(output / "unreliability_event_catalog.csv", index=False)
    result.event_membership.to_parquet(
        output / "unreliability_event_membership.parquet", index=False
    )
    result.summary.to_csv(
        output / "unreliability_event_summary_side_archetype.csv", index=False
    )
    result.sensitivity.to_csv(
        output / "unreliability_event_definition_sensitivity.csv", index=False
    )
    _event_month_summary(result.events).to_csv(
        output / "unreliability_event_month_summary.csv", index=False
    )
    _path_economic_month_summary(result.daily_cells).to_csv(
        output / "path_economic_label_month_summary.csv", index=False
    )
    _event_mechanism_summary(result.events).to_csv(
        output / "unreliability_event_mechanism_summary.csv", index=False
    )
    known = _known_period_coverage(result.events)
    known.to_csv(output / "known_period_event_coverage.csv", index=False)
    manifest = {
        **result.manifest,
        "ledger": str(Path(args.ledger).resolve()),
        "selected_rows_loaded": int(len(selected)),
        "start": args.start,
        "end_exclusive": args.end,
        "policy_selection_column": "selected_for_monitor (identical to threshold_basis_selected)",
        "policy": "ev_target_archetype_reachable_match_current_activity_8d_hr_off_regimecal_v1",
        "history_contract": {
            "2025-01_to_2026-06": "fixed-model retrospective discovery backcast",
            "2026-07-01_to_2026-07-10": "frozen post-fit generalization only",
        },
        "config": asdict(config),
    }
    _write_json(output / "event_manifest.json", manifest)
    return manifest


def run_audit(args: argparse.Namespace, output: Path) -> dict[str, Any]:
    feature_root = Path(args.feature_root)
    candidate_root = Path(args.candidate_root)
    raw_columns, raw_schemas = _feature_schema(feature_root, "symbol=*.parquet")
    candidate_columns, candidate_schemas = _feature_schema(
        candidate_root, "candidates_*.parquet"
    )
    columns = raw_columns | candidate_columns
    configured = _configured_feature_names(CFG)
    audit = audit_feature_concepts(columns, configured_columns=configured)
    required_candidates = {
        name
        for aliases in FEATURE_CONCEPTS.values()
        for kind in ("exact", "proxy")
        for name in aliases.get(kind, ())
    }
    coverage = _candidate_feature_coverage(candidate_root, required_candidates)
    store_coverage = _feature_store_sample_coverage(
        candidate_root, feature_root, required_candidates
    )
    audit["candidate_finite_coverage"] = (
        audit["matched_features"]
        .fillna("")
        .map(
            lambda value: max(
                [coverage.get(name, 0.0) for name in str(value).split("|") if name],
                default=0.0,
            )
        )
    )
    audit["feature_store_finite_coverage"] = (
        audit["matched_features"]
        .fillna("")
        .map(
            lambda value: max(
                [
                    store_coverage.get(name, 0.0)
                    for name in str(value).split("|")
                    if name
                ],
                default=0.0,
            )
        )
    )
    audit["effective_finite_coverage"] = audit[
        ["candidate_finite_coverage", "feature_store_finite_coverage"]
    ].max(axis=1)
    configured_or_schema = audit["status"].ne("missing")
    audit.loc[
        configured_or_schema & audit["effective_finite_coverage"].lt(0.80), "status"
    ] = "unreliable_coverage"
    audit.to_csv(output / "feature_concept_audit.csv", index=False)
    coverage = (
        audit["status"]
        .value_counts(dropna=False)
        .rename_axis("status")
        .reset_index(name="concepts")
    )
    coverage.to_csv(output / "feature_concept_audit_summary.csv", index=False)
    quality_sample = _feature_store_quality_sample(
        candidate_root, feature_root, required_candidates, Path(args.ledger)
    )
    quality = feature_quality_metrics(
        quality_sample,
        sorted(required_candidates),
        timestamp_col="__ts__",
        symbol_col="__symbol__",
    )
    quality.to_csv(output / "feature_quality_sample.csv", index=False)
    quality_sample["calendar_month"] = quality_sample["__ts__"].dt.strftime("%Y-%m")
    month_rows: list[dict[str, Any]] = []
    asset_rows: list[dict[str, Any]] = []
    for name in sorted(required_candidates):
        if name not in quality_sample:
            continue
        finite = pd.to_numeric(quality_sample[name], errors="coerce").notna()
        month = finite.groupby(quality_sample["calendar_month"], observed=True).mean()
        asset = finite.groupby(quality_sample["__symbol__"], observed=True).mean()
        month_rows.extend(
            {"feature": name, "calendar_month": key, "finite_coverage": float(value)}
            for key, value in month.items()
        )
        asset_rows.extend(
            {"feature": name, "symbol": key, "finite_coverage": float(value)}
            for key, value in asset.items()
        )
    pd.DataFrame(month_rows).to_csv(
        output / "feature_coverage_by_month.csv", index=False
    )
    pd.DataFrame(asset_rows).to_csv(
        output / "feature_coverage_by_asset.csv", index=False
    )
    manifest = {
        "schema": "global_residual_feature_audit_v1",
        "feature_root": str(feature_root.resolve()),
        "candidate_root": str(candidate_root.resolve()),
        "symbol_files": len(raw_schemas),
        "candidate_shards": len(candidate_schemas),
        "raw_union_feature_columns": len(raw_columns),
        "candidate_union_feature_columns": len(candidate_columns),
        "union_feature_columns": len(columns),
        "configured_feature_columns": len(configured),
        "required_concepts": len(FEATURE_CONCEPTS),
        "quality_sample_rows": len(quality_sample),
        "status_counts": {
            str(key): int(value)
            for key, value in audit["status"].value_counts().items()
        },
        "files": raw_schemas,
        "candidate_files": candidate_schemas,
    }
    _write_json(output / "feature_audit_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("events", "audit", "all"), default="all")
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-07-11")
    parser.add_argument("--causal-min-days", type=int, default=20)
    parser.add_argument("--bootstrap-draws", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifests: dict[str, Any] = {}
    if args.stage in {"events", "all"}:
        manifests["events"] = run_events(args, output)
        print(
            json.dumps(
                _json_safe(
                    {"stage": "events", "status": "complete", **manifests["events"]}
                )
            ),
            flush=True,
        )
    if args.stage in {"audit", "all"}:
        manifests["audit"] = run_audit(args, output)
        audit_console = {
            key: value
            for key, value in manifests["audit"].items()
            if key not in {"files", "candidate_files"}
        }
        print(
            json.dumps(
                _json_safe({"stage": "audit", "status": "complete", **audit_console})
            ),
            flush=True,
        )
    _write_json(
        output / "manifest.json",
        {
            "schema": "global_residual_state_discovery_stage_a_b_v1",
            "stages": manifests,
            "latent_state_scope": (
                "one frozen AE or MLP plus GMM per side x existing inference archetype; "
                "the observable market representation remains shared per side x timestamp"
            ),
        },
    )


if __name__ == "__main__":
    main()
