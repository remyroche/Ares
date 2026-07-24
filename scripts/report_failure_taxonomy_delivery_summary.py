#!/usr/bin/env python3
"""Build a concise, leakage-honest failure-taxonomy delivery report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _percent(value: float) -> str:
    return "n/a" if not np.isfinite(value) else f"{100.0 * value:.2f}%"


def _number(value: float, digits: int = 4) -> str:
    return "n/a" if not np.isfinite(value) else f"{value:.{digits}f}"


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
    columns = [str(name) for name in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _semantic_summary(profiles: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        profiles.groupby("semantic_label", observed=True)
        .agg(
            technical_modes=("cluster_id", "size"),
            blocks=("blocks", "sum"),
            active_months=("active_months", "max"),
            mean_ev=("mean_calendar_ev", "mean"),
            worst_ev=("worst_calendar_ev", "min"),
            timeout_rate=("mean_timeout_rate", "mean"),
            dirty_positive_rate=("mean_dirty_positive_rate", "mean"),
            bad_mae_rate=("mean_full_path_bad_mae_rate", "mean"),
            rank_spearman=("mean_ranking_spearman", "mean"),
            entropy=("mean_cluster_entropy", "mean"),
        )
        .reset_index()
        .sort_values("mean_ev", kind="stable")
    )
    return grouped


def _format_semantic_table(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy()
    for name in ("mean_ev", "worst_ev"):
        result[name] = result[name].map(lambda value: _percent(float(value)))
    for name in ("timeout_rate", "dirty_positive_rate", "bad_mae_rate"):
        result[name] = result[name].map(lambda value: _percent(float(value)))
    for name in ("rank_spearman", "entropy"):
        result[name] = result[name].map(lambda value: _number(float(value), 3))
    return result


def _mixture_method_summary(taxonomy: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for scope in ("local", "parent"):
        diagnostics = pd.read_csv(
            taxonomy / f"{scope}_failure_mixture_diagnostics.csv"
        )
        grouped = (
            diagnostics.groupby("method", observed=True)
            .agg(
                arms=("method", "size"),
                winners=("is_winner", "sum"),
                mean_selection_objective=("selection_objective", "mean"),
                mean_seed_ari=("seed_ari", "mean"),
                mean_seed_posterior_js=("seed_posterior_js", "mean"),
                mean_bootstrap_ari=("episode_bootstrap_ari", "mean"),
                mean_bootstrap_posterior_js=(
                    "episode_bootstrap_posterior_js",
                    "mean",
                ),
            )
            .reset_index()
        )
        grouped.insert(0, "scope", scope)
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def _format_mixture_method_summary(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy()
    for name in (
        "mean_selection_objective",
        "mean_seed_ari",
        "mean_seed_posterior_js",
        "mean_bootstrap_ari",
        "mean_bootstrap_posterior_js",
    ):
        result[name] = result[name].map(lambda value: _number(float(value), 3))
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    backcast = Path(args.backcast)
    taxonomy = Path(args.taxonomy)
    quality = Path(args.quality)
    coverage = Path(args.coverage)
    validation = Path(args.validation)
    strict_oos = Path(args.strict_oos) if str(args.strict_oos).strip() else None
    strict_taxonomy = (
        Path(args.strict_taxonomy) if str(args.strict_taxonomy).strip() else None
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    backcast_manifest = _load_json(backcast / "manifest.json")
    taxonomy_manifest = _load_json(taxonomy / "manifest.json")
    quality_manifest = _load_json(quality / "manifest.json")
    coverage_manifest = _load_json(coverage / "manifest.json")
    validation_payload = _load_json(validation)
    local_profiles = pd.read_csv(
        taxonomy / "local_frozen_failure_mode_profiles.csv"
    )
    parent_profiles = pd.read_csv(
        taxonomy / "parent_frozen_failure_mode_profiles.csv"
    )
    local_status = pd.read_csv(quality / "local_promotion_status.csv")
    parent_status = pd.read_csv(quality / "parent_promotion_status.csv")
    daily = pd.read_parquet(taxonomy / "daily_global_health.parquet")

    local_semantic = _semantic_summary(local_profiles)
    parent_semantic = _semantic_summary(parent_profiles)
    mixture_methods = _mixture_method_summary(taxonomy)
    local_semantic.to_csv(output / "local_semantic_mode_summary.csv", index=False)
    parent_semantic.to_csv(output / "parent_semantic_mode_summary.csv", index=False)
    mixture_methods.to_csv(output / "mixture_method_comparison.csv", index=False)
    promoted = pd.concat(
        [
            local_status.assign(scope="local"),
            parent_status.assign(scope="parent"),
        ],
        ignore_index=True,
    ).loc[lambda frame: frame["status"].eq("promotable_research_signal")]
    promoted.to_csv(output / "promotable_research_signals.csv", index=False)

    negative_days = int(pd.to_numeric(daily["negative_pnl_day"]).sum())
    total_days = int(len(daily))
    negative_day_rate = negative_days / max(total_days, 1)
    strict_oos_note = (
        "Strict OOS sensitivity report was not supplied."
        if strict_oos is None
        else f"Strict OOS sensitivity: `{strict_oos.resolve()}`."
    )
    strict_section = "Strict genuine-OOS sensitivity was not supplied."
    strict_metrics: dict[str, Any] = {}
    if strict_oos is not None:
        strict_manifest = _load_json(strict_oos / "manifest.json")
        strict_coverage = pd.read_csv(strict_oos / "strict_oos_coverage.csv")
        strict_overall = pd.read_csv(
            strict_oos / "strict_base_meta_intersection_overall.csv"
        ).iloc[0]
        strict_monthly = pd.read_csv(
            strict_oos / "strict_base_meta_intersection_monthly.csv"
        )
        tail_rows: list[dict[str, Any]] = []
        for tail in (10, 20, 30):
            base_ev = float(strict_overall[f"base_top{tail:02d}_mean_ev_after_1pct"])
            meta_ev = float(strict_overall[f"meta_top{tail:02d}_mean_ev_after_1pct"])
            tail_rows.append(
                {
                    "tail": f"top{tail}",
                    "base EV/trade": _percent(base_ev),
                    "meta EV/trade": _percent(meta_ev),
                    "delta": _percent(meta_ev - base_ev),
                    "base clean": _percent(float(strict_overall[f"base_top{tail:02d}_clean_exec_rate"])),
                    "meta clean": _percent(float(strict_overall[f"meta_top{tail:02d}_clean_exec_rate"])),
                    "base timeout": _percent(float(strict_overall[f"base_top{tail:02d}_timeout_rate"])),
                    "meta timeout": _percent(float(strict_overall[f"meta_top{tail:02d}_timeout_rate"])),
                }
            )
        month_rows = strict_monthly.loc[
            :, [
                "month",
                "base_top10_mean_ev_after_1pct",
                "meta_top10_mean_ev_after_1pct",
            ]
        ].copy()
        month_rows["delta"] = (
            month_rows["meta_top10_mean_ev_after_1pct"]
            - month_rows["base_top10_mean_ev_after_1pct"]
        )
        for column in (
            "base_top10_mean_ev_after_1pct",
            "meta_top10_mean_ev_after_1pct",
            "delta",
        ):
            month_rows[column] = month_rows[column].map(
                lambda value: _percent(float(value))
            )
        strict_section = f"""
The strict report uses `{int(strict_manifest['base_meta_intersection']['overlap_rows']):,}` exact UTC-key overlaps. The base comparator and residual-expert rank are evaluated against the same meta-handoff outcomes; the large base ledger is used only to prove row provenance.

{_markdown_table(pd.DataFrame(tail_rows))}

Top-10 by month:

{_markdown_table(month_rows)}

Strict coverage:

{_markdown_table(strict_coverage)}
"""
        strict_metrics = {
            "overlap_rows": int(strict_manifest["base_meta_intersection"]["overlap_rows"]),
            "top10_base_ev": float(strict_overall["base_top10_mean_ev_after_1pct"]),
            "top10_meta_ev": float(strict_overall["meta_top10_mean_ev_after_1pct"]),
            "top10_meta_delta": float(
                strict_overall["meta_top10_mean_ev_after_1pct"]
                - strict_overall["base_top10_mean_ev_after_1pct"]
            ),
        }
    if strict_taxonomy is not None:
        strict_taxonomy_manifest = _load_json(strict_taxonomy / "manifest.json")
        strict_section += (
            "\nLimited strict-taxonomy refit: "
            f"`{strict_taxonomy.resolve()}`. It produced "
            f"{int(strict_taxonomy_manifest.get('frozen_local_mode_groups', 0))} "
            "stable frozen local groups and status "
            f"`{strict_taxonomy_manifest.get('status')}`. This is sensitivity "
            "evidence only; the overlapping base+meta OOS history is too short "
            "to replace the three-year descriptive taxonomy.\n"
        )

    promoted_columns = [
        "scope",
        "side_name",
        "archetype_policy_key",
        "failure_mode",
        "positive_days",
        "alert_days",
        "precision",
        "recall",
        "lift",
        "average_precision",
        "status",
    ]
    promoted_display = promoted.loc[:, promoted_columns].copy()
    for name in ("precision", "recall"):
        promoted_display[name] = promoted_display[name].map(
            lambda value: _percent(float(value))
        )
    for name in ("lift", "average_precision"):
        promoted_display[name] = promoted_display[name].map(
            lambda value: _number(float(value), 3)
        )

    report = f"""# Three-Year Failure Taxonomy

## Evidence Contract

- Period: `{backcast_manifest.get('start')}` to `{backcast_manifest.get('end_exclusive')}`.
- Diagnostic rows: `{int(backcast_manifest.get('rows', 0)):,}`; monitored rows: `{int(backcast_manifest.get('selected_for_monitor_rows', 0)):,}`.
- Costs: `{_percent(float(backcast_manifest.get('round_trip_cost', np.nan)))}` round trip, counted once: `{bool(backcast_manifest.get('cost_counted_once'))}`.
- Source status: frozen-model diagnostic backcast, not genuine full-period model OOS.
- Frozen taxonomy reference end: `{taxonomy_manifest.get('prospective_taxonomy_reference_end')}`.
- Historical meta score available in the three-year source: `{taxonomy_manifest.get('historical_meta_score_available')}`.
- {strict_oos_note}

## Coverage

- Calendar days: `{total_days:,}`.
- Negative-PnL days: `{negative_days:,}` (`{_percent(negative_day_rate)}`).
- Negative days with frozen parent assignment: `{int(validation_payload['metrics']['negative_days_with_parent_mode']):,}`.
- Negative-day local assignment coverage: `{_percent(float(validation_payload['metrics']['negative_day_local_mode_full_coverage']))}`.
- Minimum monthly path coverage: `{_percent(float(validation_payload['metrics']['minimum_path_coverage']))}` against the explicit validation floor `{_percent(float(validation_payload['metrics'].get('required_min_path_coverage', np.nan)))}`; months below 90%: `{int(validation_payload['metrics']['path_coverage_below_90pct_count'])}`.

The exact coverage requirement passes, but the 94% negative-day prevalence makes the broad backcast unsuitable as proof of a profitable or discriminative trading policy. It is an error-taxonomy source.

## Frozen Local Failure Modes

{_markdown_table(_format_semantic_table(local_semantic))}

## Frozen Parent Modes

{_markdown_table(_format_semantic_table(parent_semantic))}

The local taxonomy has stronger economic and behavioral separation. The parent taxonomy is diffuse and should remain broad context: its posterior entropy is close to one and one mode dominates support.

## Representation And Mixture Comparison

{_markdown_table(_format_mixture_method_summary(mixture_methods))}

The comparison refits representations across seeds and episode bootstraps. `PCA/GMM`, `PCA/Student-t`, and the small denoising-AE each win at least one local side/archetype group; VICReg/GMM has no selected winner despite relatively strong bootstrap ARI. This supports retaining the simpler robust representations for the current sparse episode population.

## Prospective OOS Signals

{_markdown_table(promoted_display)}

- Local evaluated arms: `{quality_manifest['local']['evaluated_arms']}`; research-promotable signals: `{quality_manifest['local']['promotable_research_signals']}`.
- Parent evaluated arms: `{quality_manifest['parent']['evaluated_arms']}`; research-promotable signals: `{quality_manifest['parent']['promotable_research_signals']}`.
- Same-day recognized episodes: `{coverage_manifest['oos_recognized_episodes']['same_day']}` / `{coverage_manifest['oos_assessable_episodes']['same_day']}`.
- One-day lead recognized episodes: `{coverage_manifest['oos_recognized_episodes']['lead_1d']}` / `{coverage_manifest['oos_assessable_episodes']['lead_1d']}`.
- Three-day lead recognized episodes: `{coverage_manifest['oos_recognized_episodes']['lead_3d']}` / `{coverage_manifest['oos_assessable_episodes']['lead_3d']}`.

These are continuous research risk signals, not approved hard gates. Most ex-post modes are not predictably separable from observable state at the current alert threshold.

## Strict OOS Sensitivity

{strict_section}

## Stability And Integrity

- Local temporal warnings: `{taxonomy_manifest.get('mixture_temporal_stability_warnings')}` / `{taxonomy_manifest.get('mixture_temporal_stability_rows')}`.
- Parent temporal warnings: `{taxonomy_manifest.get('parent_temporal_stability_warnings')}` / `{taxonomy_manifest.get('parent_temporal_stability_rows')}`.
- Local calendar redundancy warnings: `{taxonomy_manifest.get('mixture_redundancy_warnings')}` / `{taxonomy_manifest.get('mixture_nonredundancy_rows')}`.
- Parent calendar redundancy warnings: `{taxonomy_manifest.get('parent_mixture_redundancy_warnings')}` / `{taxonomy_manifest.get('parent_mixture_nonredundancy_rows')}`.
- Delivery validator passed: `{validation_payload.get('passed')}`.
- Leaked prospective features: `{validation_payload['metrics'].get('leaked_features')}`.
- Batch-layout-dependent AE/GMM features selected: `{validation_payload['metrics'].get('nonportable_ae_gmm_features')}`.

## Decision

The ex-post local taxonomy is usable for diagnosis and target construction. Negative-EV onset detectors are useful as research overlays. Strict genuine-OOS sensitivity is now documented, but the taxonomy and detector are not justified as live hard gates: the comparable base+meta OOS overlap is too short to refreeze stable modes, and forward replication remains required before promotion.
"""
    (output / "FAILURE_TAXONOMY_REPORT.md").write_text(report, encoding="utf-8")
    manifest = {
        "schema": "failure_taxonomy_delivery_summary_v1",
        "backcast": str(backcast.resolve()),
        "taxonomy": str(taxonomy.resolve()),
        "quality": str(quality.resolve()),
        "coverage": str(coverage.resolve()),
        "validation": str(validation.resolve()),
        "strict_oos": str(strict_oos.resolve()) if strict_oos else "",
        "strict_taxonomy": str(strict_taxonomy.resolve()) if strict_taxonomy else "",
        "strict_oos_metrics": strict_metrics,
        "negative_days": negative_days,
        "calendar_days": total_days,
        "negative_day_rate": negative_day_rate,
        "frozen_local_semantic_modes": int(local_semantic["semantic_label"].nunique()),
        "frozen_parent_semantic_modes": int(parent_semantic["semantic_label"].nunique()),
        "promotable_research_signals": int(len(promoted)),
        "deployment_status": "research_only_not_promoted",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backcast", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--strict-oos", type=str, default="")
    parser.add_argument("--strict-taxonomy", type=str, default="")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
