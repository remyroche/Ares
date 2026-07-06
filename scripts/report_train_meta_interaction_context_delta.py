#!/usr/bin/env python3
"""Report direct-context vs interaction-context train_meta deltas.

The materialized interaction feature set is useful only if it improves top-k
trade quality without hiding path risk.  This report compares two train_meta
smoke directories with the same schema and summarizes aggregate and
side/archetype deltas.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_direct_context_risk_aware_train_meta_smoke import _json_safe  # noqa: E402


DEFAULT_BASE_DIR = Path(
    "data_perp/reports/contextual_tp_sl_ablation_workflow_v14_runtime_health_20260701/"
    "direct_cross_asset_meta_context_v1/train_meta_direct_context_feature_set_v1/risk_aware_train_meta_smoke_v1"
)
DEFAULT_CANDIDATE_DIR = Path(
    "data_perp/reports/contextual_tp_sl_ablation_workflow_v14_runtime_health_20260701/"
    "direct_cross_asset_meta_context_v1/train_meta_direct_context_feature_set_v1/"
    "train_meta_interaction_context_feature_set_v1/risk_aware_train_meta_smoke_v1"
)
DEFAULT_FEATURE_SET_DIR = DEFAULT_CANDIDATE_DIR.parent
DEFAULT_OUT_DIR = DEFAULT_FEATURE_SET_DIR / "interaction_context_delta_report_v1"
DEFAULT_SELECTORS = (
    "s0_ev_only",
    "s12_ev_clean_strong_risk",
    "s13_ev_clean_fullsl_neutral_timeout",
    "s14_cell_prior_fullsl_s12",
    "s15_cell_prior_fullsl_timeout_s12",
    "s16_cell_prior_clean_risk_s12",
    "s17_cell_prior_ev_fullsl_s12",
    "s18_long_cell_prior_ev_fullsl_s12",
    "s19_long_s16_short_s12",
    "s20_long_s14_short_s12",
    "s21_long_s18_short_s12",
)
DEFAULT_REFERENCE_SELECTOR = "s12_ev_clean_strong_risk"


METRIC_COLUMNS = (
    "precision_positive_ev",
    "ev_weighted_precision",
    "mean_ev_after_1pct",
    "full_sl_rate",
    "timeout_rate",
    "clean_exec_proxy_rate",
)


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _metric_summary(frame: pd.DataFrame, selectors: tuple[str, ...]) -> pd.DataFrame:
    selected = frame[frame["selector"].astype(str).isin(selectors)].copy()
    if selected.empty:
        return pd.DataFrame()
    return selected.groupby(["selector", "top_frac"], as_index=False).agg(
        months=("month", "nunique"),
        selected_rows=("selected_rows", "sum"),
        precision_positive_ev=("precision_positive_ev", "mean"),
        ev_weighted_precision=("ev_weighted_precision", "mean"),
        mean_ev_after_1pct=("mean_ev_after_1pct", "mean"),
        full_sl_rate=("full_sl_rate", "mean"),
        timeout_rate=("timeout_rate", "mean"),
        clean_exec_proxy_rate=("clean_exec_proxy_rate", "mean"),
    )


def _delta_frame(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    key_cols: list[str],
    selectors: tuple[str, ...],
) -> pd.DataFrame:
    candidate = candidate[candidate["selector"].astype(str).isin(selectors)].copy()
    baseline = baseline[baseline["selector"].astype(str).isin(selectors)].copy()
    merged = candidate.merge(baseline, on=key_cols, suffixes=("_candidate", "_baseline"))
    for col in METRIC_COLUMNS:
        if f"{col}_candidate" in merged.columns and f"{col}_baseline" in merged.columns:
            merged[f"delta_{col}"] = merged[f"{col}_candidate"] - merged[f"{col}_baseline"]
    return merged


def _aggregate_delta_summary(delta: pd.DataFrame) -> pd.DataFrame:
    if delta.empty:
        return pd.DataFrame()
    agg_cols = {f"mean_delta_{col}": (f"delta_{col}", "mean") for col in METRIC_COLUMNS if f"delta_{col}" in delta}
    return delta.groupby(["selector", "top_frac"], as_index=False).agg(**agg_cols)


def _cell_delta_summary(delta: pd.DataFrame) -> pd.DataFrame:
    if delta.empty:
        return pd.DataFrame()
    top10 = delta[np.isclose(pd.to_numeric(delta["top_frac"], errors="coerce"), 0.10)].copy()
    if top10.empty:
        return pd.DataFrame()
    return top10.groupby(["selector", "side_name", "source_archetype"], as_index=False).agg(
        months=("month", "nunique"),
        mean_delta_precision=("delta_precision_positive_ev", "mean"),
        mean_delta_weighted_precision=("delta_ev_weighted_precision", "mean"),
        mean_delta_ev=("delta_mean_ev_after_1pct", "mean"),
        mean_delta_full_sl=("delta_full_sl_rate", "mean"),
        mean_delta_timeout=("delta_timeout_rate", "mean"),
        mean_delta_clean=("delta_clean_exec_proxy_rate", "mean"),
        better_ev_months=("delta_mean_ev_after_1pct", lambda s: int((pd.to_numeric(s, errors="coerce") > 0).sum())),
        lower_full_sl_months=("delta_full_sl_rate", lambda s: int((pd.to_numeric(s, errors="coerce") < 0).sum())),
        lower_timeout_months=("delta_timeout_rate", lambda s: int((pd.to_numeric(s, errors="coerce") < 0).sum())),
    )


def _selector_delta_vs_reference(
    frame: pd.DataFrame,
    *,
    reference_selector: str,
    key_cols: list[str],
    selectors: tuple[str, ...],
) -> pd.DataFrame:
    selected = frame[frame["selector"].astype(str).isin(selectors)].copy()
    reference = selected[selected["selector"].astype(str).eq(reference_selector)].copy()
    current = selected[~selected["selector"].astype(str).eq(reference_selector)].copy()
    if reference.empty or current.empty:
        return pd.DataFrame()
    merged = current.merge(reference, on=key_cols, suffixes=("_candidate", "_reference"))
    for col in METRIC_COLUMNS:
        if f"{col}_candidate" in merged.columns and f"{col}_reference" in merged.columns:
            merged[f"delta_{col}"] = merged[f"{col}_candidate"] - merged[f"{col}_reference"]
    return merged


def _selector_delta_summary(delta: pd.DataFrame, *, group_cols: list[str]) -> pd.DataFrame:
    if delta.empty:
        return pd.DataFrame()
    top10 = delta[np.isclose(pd.to_numeric(delta["top_frac"], errors="coerce"), 0.10)].copy()
    if top10.empty:
        return pd.DataFrame()
    return top10.groupby(group_cols, as_index=False).agg(
        months=("month", "nunique"),
        mean_delta_precision=("delta_precision_positive_ev", "mean"),
        mean_delta_weighted_precision=("delta_ev_weighted_precision", "mean"),
        mean_delta_ev=("delta_mean_ev_after_1pct", "mean"),
        mean_delta_full_sl=("delta_full_sl_rate", "mean"),
        mean_delta_timeout=("delta_timeout_rate", "mean"),
        mean_delta_clean=("delta_clean_exec_proxy_rate", "mean"),
    )


def _fit_event_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    fit = pd.read_csv(path)
    if fit.empty or "feature_count" not in fit.columns:
        return pd.DataFrame()
    return fit.groupby("month", as_index=False).agg(
        fits=("status", "size"),
        used_features_min=("feature_count", "min"),
        used_features_mean=("feature_count", "mean"),
        used_features_max=("feature_count", "max"),
        all_null_feature_mean=("all_null_feature_count", "mean"),
        constant_feature_mean=("constant_feature_count", "mean"),
    )


def _write_report(
    path: Path,
    *,
    manifest: dict[str, Any],
    base_summary: pd.DataFrame,
    candidate_summary: pd.DataFrame,
    aggregate_delta: pd.DataFrame,
    cell_summary: pd.DataFrame,
    selector_delta_summary: pd.DataFrame,
    cell_selector_delta_summary: pd.DataFrame,
    fit_summary: pd.DataFrame,
    availability: pd.DataFrame,
) -> None:
    top10_delta = aggregate_delta[np.isclose(pd.to_numeric(aggregate_delta.get("top_frac", np.nan), errors="coerce"), 0.10)].copy()
    top10_candidate = candidate_summary[
        np.isclose(pd.to_numeric(candidate_summary.get("top_frac", np.nan), errors="coerce"), 0.10)
    ].copy()
    sparse = availability.head(20) if not availability.empty else pd.DataFrame()
    worst_cells = cell_summary.sort_values(["mean_delta_ev", "mean_delta_full_sl"], ascending=[True, False]).head(20)
    best_cells = cell_summary.sort_values(["mean_delta_ev", "mean_delta_precision"], ascending=[False, False]).head(20)
    long_selector_deltas = (
        cell_selector_delta_summary[cell_selector_delta_summary["side_name"].astype(str).eq("long")].copy()
        if not cell_selector_delta_summary.empty and "side_name" in cell_selector_delta_summary.columns
        else pd.DataFrame()
    )
    lines = [
        "# Train Meta Interaction Context Delta Report",
        "",
        "## Scope",
        "",
        f"- Baseline smoke: `{manifest['baseline_dir']}`",
        f"- Candidate smoke: `{manifest['candidate_dir']}`",
        "- Metrics are top-k precision, EV after 1% round-trip cost, full-SL/bad-MAE proxy, timeout, and clean-exec proxy.",
        "- Candidate features are live-predictable direct context plus side/archetype/context interactions.",
        "",
        "## Candidate Top10",
        "",
        top10_candidate.to_markdown(index=False) if not top10_candidate.empty else "No top10 candidate metrics.",
        "",
        "## Top10 Delta vs Baseline",
        "",
        top10_delta.to_markdown(index=False) if not top10_delta.empty else "No top10 deltas.",
        "",
        f"## Candidate Top10 Delta vs `{manifest['reference_selector']}`",
        "",
        selector_delta_summary.to_markdown(index=False)
        if not selector_delta_summary.empty
        else "No candidate selector deltas.",
        "",
        f"## Long Cell Delta vs `{manifest['reference_selector']}`",
        "",
        long_selector_deltas.to_markdown(index=False)
        if not long_selector_deltas.empty
        else "No long-cell selector deltas.",
        "",
        "## Best Side x Archetype Deltas",
        "",
        best_cells.to_markdown(index=False) if not best_cells.empty else "No cell deltas.",
        "",
        "## Weak Side x Archetype Deltas",
        "",
        worst_cells.to_markdown(index=False) if not worst_cells.empty else "No weak cell deltas.",
        "",
        "## Fit Feature Availability",
        "",
        fit_summary.to_markdown(index=False) if not fit_summary.empty else "No fit-event availability summary.",
        "",
        "## Sparsest Materialized Features",
        "",
        sparse.to_markdown(index=False) if not sparse.empty else "No feature availability summary.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    *,
    baseline_dir: Path,
    candidate_dir: Path,
    feature_set_dir: Path,
    output_dir: Path,
    selectors: tuple[str, ...],
    reference_selector: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_agg = _load_csv(baseline_dir / "risk_aware_train_meta_aggregate.csv")
    candidate_agg = _load_csv(candidate_dir / "risk_aware_train_meta_aggregate.csv")
    baseline_cell = _load_csv(baseline_dir / "risk_aware_train_meta_by_cell.csv")
    candidate_cell = _load_csv(candidate_dir / "risk_aware_train_meta_by_cell.csv")

    base_summary = _metric_summary(baseline_agg, selectors)
    candidate_summary = _metric_summary(candidate_agg, selectors)
    aggregate_delta = _aggregate_delta_summary(
        _delta_frame(candidate_agg, baseline_agg, key_cols=["month", "selector", "top_frac"], selectors=selectors)
    )
    cell_delta = _delta_frame(
        candidate_cell,
        baseline_cell,
        key_cols=["month", "side_name", "source_archetype", "selector", "top_frac"],
        selectors=selectors,
    )
    cell_summary = _cell_delta_summary(cell_delta)
    selector_delta = _selector_delta_vs_reference(
        candidate_agg,
        reference_selector=reference_selector,
        key_cols=["month", "top_frac"],
        selectors=selectors,
    )
    selector_delta_summary = _selector_delta_summary(selector_delta, group_cols=["selector_candidate", "top_frac"])
    cell_selector_delta = _selector_delta_vs_reference(
        candidate_cell,
        reference_selector=reference_selector,
        key_cols=["month", "side_name", "source_archetype", "top_frac"],
        selectors=selectors,
    )
    cell_selector_delta_summary = _selector_delta_summary(
        cell_selector_delta,
        group_cols=["selector_candidate", "side_name", "source_archetype"],
    )
    fit_summary = _fit_event_summary(candidate_dir / "risk_aware_train_meta_fit_events.csv")
    availability_path = feature_set_dir / "train_meta_interaction_context_feature_availability_summary.csv"
    availability = pd.read_csv(availability_path) if availability_path.exists() else pd.DataFrame()

    outputs = {
        "baseline_summary": output_dir / "baseline_topk_summary.csv",
        "candidate_summary": output_dir / "candidate_topk_summary.csv",
        "aggregate_delta": output_dir / "interaction_context_aggregate_delta.csv",
        "cell_delta": output_dir / "interaction_context_cell_delta.csv",
        "cell_delta_summary": output_dir / "interaction_context_cell_delta_summary.csv",
        "selector_delta_vs_reference": output_dir / "interaction_context_selector_delta_vs_reference.csv",
        "selector_delta_vs_reference_summary": output_dir / "interaction_context_selector_delta_vs_reference_summary.csv",
        "cell_selector_delta_vs_reference": output_dir / "interaction_context_cell_selector_delta_vs_reference.csv",
        "cell_selector_delta_vs_reference_summary": output_dir / "interaction_context_cell_selector_delta_vs_reference_summary.csv",
        "fit_feature_summary": output_dir / "interaction_context_fit_feature_summary.csv",
        "report": output_dir / "train_meta_interaction_context_delta_report.md",
        "manifest": output_dir / "manifest.json",
    }
    base_summary.to_csv(outputs["baseline_summary"], index=False)
    candidate_summary.to_csv(outputs["candidate_summary"], index=False)
    aggregate_delta.to_csv(outputs["aggregate_delta"], index=False)
    cell_delta.to_csv(outputs["cell_delta"], index=False)
    cell_summary.to_csv(outputs["cell_delta_summary"], index=False)
    selector_delta.to_csv(outputs["selector_delta_vs_reference"], index=False)
    selector_delta_summary.to_csv(outputs["selector_delta_vs_reference_summary"], index=False)
    cell_selector_delta.to_csv(outputs["cell_selector_delta_vs_reference"], index=False)
    cell_selector_delta_summary.to_csv(outputs["cell_selector_delta_vs_reference_summary"], index=False)
    fit_summary.to_csv(outputs["fit_feature_summary"], index=False)
    manifest = {
        "scope": "train_meta_interaction_context_delta_report",
        "baseline_dir": str(baseline_dir),
        "candidate_dir": str(candidate_dir),
        "feature_set_dir": str(feature_set_dir),
        "selectors": list(selectors),
        "reference_selector": reference_selector,
        "rows": {
            "aggregate_delta": int(len(aggregate_delta)),
            "cell_delta": int(len(cell_delta)),
            "cell_delta_summary": int(len(cell_summary)),
            "selector_delta_vs_reference": int(len(selector_delta)),
            "cell_selector_delta_vs_reference": int(len(cell_selector_delta)),
        },
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    outputs["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    _write_report(
        outputs["report"],
        manifest=manifest,
        base_summary=base_summary,
        candidate_summary=candidate_summary,
        aggregate_delta=aggregate_delta,
        cell_summary=cell_summary,
        selector_delta_summary=selector_delta_summary,
        cell_selector_delta_summary=cell_selector_delta_summary,
        fit_summary=fit_summary,
        availability=availability,
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--feature-set-dir", type=Path, default=DEFAULT_FEATURE_SET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--selectors", nargs="+", default=list(DEFAULT_SELECTORS))
    parser.add_argument("--reference-selector", default=DEFAULT_REFERENCE_SELECTOR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run(
        baseline_dir=args.baseline_dir,
        candidate_dir=args.candidate_dir,
        feature_set_dir=args.feature_set_dir,
        output_dir=args.output_dir,
        selectors=tuple(str(selector) for selector in args.selectors),
        reference_selector=str(args.reference_selector),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
