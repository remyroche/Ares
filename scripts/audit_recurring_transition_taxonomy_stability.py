#!/usr/bin/env python3
"""Audit recurring transition taxonomy evidence without touching the trading stack.

The report deliberately keeps two different objects apart:

* persistent *regime* discovery (the multihorizon observable panel); and
* finite-horizon *transition* morphology / stable-vs-transition labels.

It is an evidence inventory and stability audit.  It does not add fields to a
residual learner, fit a routing model, or claim that any ex-post phase is
available at a decision timestamp.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_PANEL = ROOT / "data_perp/artifacts/regime_multiview_panel_2022_2026_20260730_v2/multiview_regime_features.parquet"
DEFAULT_CATALOGUE = ROOT / "data_perp/artifacts/transition_pattern_catalogue_20260730_v6"
DEFAULT_PATH = ROOT / "data_perp/artifacts/regime_transition_path_geometry_diagnostic_20260730_v1"
DEFAULT_CALENDAR = ROOT / "data_perp/artifacts/stack_performance_calendar_2022_2026_20260730_v3/performance_period_metrics.parquet"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/recurring_transition_taxonomy_stability_20260730_v1"
SCHEMA = "recurring_transition_taxonomy_stability_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def panel_summary(panel_path: Path) -> dict[str, Any]:
    """Return factual scope/coverage only; the panel is never outcome-fitted here."""

    frame = _load_parquet(panel_path)
    timestamp = next((name for name in ("source_utc", "__ts__", "timestamp") if name in frame), None)
    if timestamp is None:
        raise ValueError("multiview panel has no recognised timestamp field")
    values = pd.to_datetime(frame[timestamp], utc=True, errors="coerce")
    numeric = frame.select_dtypes(include=[np.number]).columns.tolist()
    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "numeric_columns": int(len(numeric)),
        "timestamp_field": timestamp,
        "time_start_utc": values.min(),
        "time_end_utc": values.max(),
        "null_fraction_numeric": float(frame[numeric].isna().mean().mean()) if numeric else None,
        "role": "outcome-free causal/trailing multihorizon observable panel; referenced for discovery scope, not merged with transition labels",
    }


def summarize_morphology(morphology: pd.DataFrame, support: pd.DataFrame | None = None) -> pd.DataFrame:
    """Summarise fold-local GMM morphology without pretending component IDs align."""

    data = morphology.copy()
    data["anchor_source_utc"] = pd.to_datetime(data["anchor_source_utc"], utc=True, errors="coerce")
    data["era"] = data["anchor_source_utc"].dt.year.astype("Int64")
    probability_columns = sorted(column for column in data if column.startswith("morphology__posterior_"))
    data["posterior_max"] = data[probability_columns].max(axis=1) if probability_columns else np.nan
    rows: list[dict[str, Any]] = []
    for (fold, component), group in data.groupby(["oof_fold", "morphology__component_id"], dropna=False, sort=True):
        record: dict[str, Any] = {
            "oof_fold": int(fold),
            "fold_local_component": str(component),
            "events": int(len(group)),
            "eras": int(group["era"].nunique()),
            "era_values": ",".join(str(value) for value in sorted(group["era"].dropna().unique())),
            "source_destination_pairs": int(group[["source_state", "destination_state"]].drop_duplicates().shape[0]),
            "mean_posterior_max": float(group["posterior_max"].mean()),
            "mean_entropy": float(group["morphology__entropy"].mean()),
            "mean_top2_margin": float(group["morphology__top2_margin"].mean()),
            "abstention_rate": float(pd.to_numeric(group["morphology__abstained"], errors="coerce").mean()),
            "recurs_across_eras": bool(group["era"].nunique() >= 2),
            "cross_fold_alignment": "NOT_IDENTIFIABLE_COMPONENT_IDS_ARE_FOLD_LOCAL",
        }
        if support is not None and not support.empty:
            support_component = "morphology_component_id" if "morphology_component_id" in support else "morphology__component_id"
            matched = support.loc[(support["oof_fold"].eq(fold)) & (support[support_component].astype(str).eq(str(component)))]
            if not matched.empty:
                last = matched.iloc[-1]
                # The catalogue's recurrence table describes the component
                # support available to the fold-local fit.  It is not the
                # holdout event support in this row, hence the deliberately
                # explicit names below.
                record["catalogue_fit_support_pass"] = bool(last.get("support_pass", False))
                record["catalogue_fit_events"] = int(last.get("events", 0))
                record["catalogue_fit_eras"] = int(last.get("eras", 0))
        rows.append(record)
    return pd.DataFrame(rows)


def morphology_classifier_agreement(morphology: pd.DataFrame, classifier: pd.DataFrame) -> pd.DataFrame:
    merged = morphology.merge(classifier, on=["event_id", "anchor_source_utc", "oof_fold"], how="inner", validate="one_to_one")
    probability_columns = sorted(column for column in classifier if column.startswith("classifier__p_m"))
    if not probability_columns:
        return pd.DataFrame(columns=["slice", "events", "agreement", "mean_confidence"])
    merged["predicted_component"] = merged[probability_columns].idxmax(axis=1).str.replace("classifier__p_", "", regex=False)
    merged["agreement"] = merged["predicted_component"].eq(merged["morphology__component_id"].astype(str))
    merged["confidence"] = merged[probability_columns].max(axis=1)
    records = []
    for key, group in [("all_events", merged), ("non_abstained", merged.loc[merged["morphology__abstained"].eq(0)])]:
        records.append({"slice": key, "events": int(len(group)), "agreement": float(group["agreement"].mean()) if len(group) else np.nan, "mean_confidence": float(group["confidence"].mean()) if len(group) else np.nan})
    for component, group in merged.loc[merged["morphology__abstained"].eq(0)].groupby("morphology__component_id", sort=True):
        records.append({"slice": f"component::{component}", "events": int(len(group)), "agreement": float(group["agreement"].mean()), "mean_confidence": float(group["confidence"].mean())})
    return pd.DataFrame(records)


def stable_transition_metrics(
    frame: pd.DataFrame,
    *,
    probability_column: str = "classifier__p_1",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """OOF discrimination and reliability of one stable-v-transition arm."""

    data = frame.copy()
    data["anchor_source_utc"] = pd.to_datetime(data["anchor_source_utc"], utc=True, errors="coerce")
    data["era"] = data["anchor_source_utc"].dt.year.astype("Int64")
    y = pd.to_numeric(data["target__stable_vs_transition"], errors="coerce")
    p = pd.to_numeric(data[probability_column], errors="coerce")
    data = data.loc[y.notna() & p.notna()].copy()

    def measure(name: str, group: pd.DataFrame) -> dict[str, Any]:
        labels = pd.to_numeric(group["target__stable_vs_transition"], errors="coerce")
        scores = pd.to_numeric(group[probability_column], errors="coerce")
        diverse = labels.nunique() == 2
        return {
            "slice": name,
            "rows": int(len(group)),
            "transition_rate": float(labels.mean()) if len(group) else np.nan,
            "roc_auc": float(roc_auc_score(labels, scores)) if diverse else np.nan,
            "average_precision": float(average_precision_score(labels, scores)) if diverse else np.nan,
            "brier": float(brier_score_loss(labels, scores)) if len(group) else np.nan,
            "mean_probability": float(scores.mean()) if len(group) else np.nan,
        }
    records = [measure("all_oof", data)] + [measure(f"era::{era}", group) for era, group in data.groupby("era", sort=True)] + [measure(f"fold::{fold}", group) for fold, group in data.groupby("oof_fold", sort=True)]
    reliability = data.assign(bin=pd.qcut(data[probability_column], q=min(5, data[probability_column].nunique()), duplicates="drop"))
    bins = (reliability.groupby("bin", observed=True)
            .agg(rows=("event_id", "size"), mean_probability=(probability_column, "mean"), observed_transition_rate=("target__stable_vs_transition", "mean"))
            .reset_index())
    bins["bin"] = bins["bin"].astype(str)
    return pd.DataFrame(records), bins


def path_side_consistency(path_summary: pd.DataFrame) -> pd.DataFrame:
    """Side-separated descriptive transition-phase geometry; no performance join."""

    data = path_summary.loc[path_summary["taxonomy"].eq("transition_phase_ex_post")].copy()
    if data.empty:
        return pd.DataFrame()
    keep_metrics = {"net_ev_12h", "peak_mfe_atr", "opportunity_probability", "mae_before_meaningful_mfe_atr", "time_to_meaningful_mfe_hours", "future_slope_atr_per_hour"}
    data = data.loc[data["metric"].isin(keep_metrics)]
    return data.loc[:, ["context_value", "side_name", "metric", "condition", "metric_available", "n_candidates", "n_decision_hours", "mean", "ci95_low", "ci95_high"]].sort_values(["context_value", "metric", "side_name"]).reset_index(drop=True)


def performance_era_summary(calendar: pd.DataFrame) -> pd.DataFrame:
    data = calendar.loc[calendar["period_type"].eq("month") & calendar["complete_for_percentage"].astype(bool)].copy()
    data["period_start_utc"] = pd.to_datetime(data["period_start_utc"], utc=True, errors="coerce")
    data["era"] = data["period_start_utc"].dt.year.astype("Int64")
    columns = ["alpha_rank_ic", "execution_net_rank_ic", "tail_execution_net_rank_ic", "mean_net_bps"]
    summaries = data.groupby("era", sort=True).agg(
        complete_months=("period", "size"),
        meaningful_ic_months=("meaningfully_positive_ic", "sum"),
        meaningful_ev_months=("meaningfully_positive_ev", "sum"),
        meaningful_both_months=("meaningfully_positive_ic_and_ev", "sum"),
        **{f"mean_{column}": (column, "mean") for column in columns},
    ).reset_index()
    return summaries


def tool_inventory(root: Path, catalogue_dir: Path) -> pd.DataFrame:
    """Separate executable/running evidence from source-code availability."""

    entries = [
        ("GMM_transition_morphology", "implemented_and_run", catalogue_dir / "morphology_oof.parquet", "fold-local OOF transition morphology; labels cannot be made into persistent regime states"),
        ("LightGBM_stable_transition", "implemented_and_run", catalogue_dir / "stable_transition_oof.parquet", "OOF stable-v-transition probability"),
        ("LightGBM_morphology_classifier", "implemented_and_run", catalogue_dir / "morphology_classifier_oof.parquet", "OOF classifier of fold-local morphology IDs"),
        ("BOCPD", "implemented_and_run", root / "data_perp/artifacts/regime_transition_changepoint_ablation_20260727_v2/manifest.json", "online univariate BOCPD transition onset context; not a state taxonomy"),
        ("KMeans_state_recurrence", "implemented_and_run_limited_scope", root / "data_perp/artifacts/exact_history_state_recurrence_20260727_v1/manifest.json", "outcome-free fixed geometry, Jan-Jul 2025 only; not comparable to 2022-26 transition catalogue"),
        ("AE_GMM_context", "implemented_and_run_limited_scope", root / "data_perp/artifacts/packb_downstream_context_20260725_v2_31_8_frozen_ae_gmm/manifest.json", "candidate context Apr-Jul 2026, not a long-history transition taxonomy"),
        ("HDBSCAN", "implemented_only", root / "scripts/run_short_mixed_failure_cluster_overlay.py", "code exists; no eligible materialized 2022-26 taxonomy evidence found"),
        ("Categorical_HMM", "implemented_only", root / "extreme_price_movements/global_residual_latent_state.py", "code exists; no eligible materialized 2022-26 taxonomy evidence found"),
        ("Bayesian_Rule_List", "implemented_and_run_native_fallback" if (catalogue_dir / "stable_transition_brl_oof.parquet").exists() else "implemented_only", catalogue_dir / "stable_transition_brl_oof.parquet", "ordered low-cardinality binary rule list with Beta-Binomial posterior/MAP objective; native fallback is not MCMC BRL"),
    ]
    rows = []
    for tool, status, evidence, note in entries:
        executable = True
        if tool == "Bayesian_Rule_List":
            # The native MAP fallback is dependency-free; imodels only changes
            # the optional MCMC backend, never whether this challenger runs.
            executable = True
        if tool == "Categorical_HMM":
            executable = importlib.util.find_spec("hmmlearn") is not None
            if not executable:
                status = "implemented_dependency_unavailable"
        if tool == "HDBSCAN":
            executable = importlib.util.find_spec("hdbscan") is not None
            if not executable:
                status = "implemented_dependency_unavailable"
        rows.append({"tool": tool, "status": status, "dependency_executable": executable, "evidence_path": str(evidence), "evidence_exists": evidence.exists(), "note": note})
    return pd.DataFrame(rows)


def render_report(manifest: dict[str, Any]) -> str:
    """Human-readable companion to the signed machine-readable manifest."""

    stable = manifest["headline"]["stable_transition_all_oof"]
    morphology = manifest["headline"]["morphology_classifier_non_abstained"]
    brl = manifest["headline"].get("stable_transition_brl_all_oof")
    panel = manifest["panel"]
    return "\n".join([
        "# Recurring transition taxonomy and stability audit",
        "",
        "## Scope",
        "",
        f"The audit references the outcome-free multihorizon panel ({panel['rows']:,} hourly rows, {panel['numeric_columns']:,} numeric fields; {panel['time_start_utc']} to {panel['time_end_utc']}). It does not fit or feed a residual learner.",
        "",
        "Persistent regime discovery and finite-horizon transition labels remain separate. In particular, no ex-post transition phase is presented as a decision-time regime field.",
        "",
        "## Evidence that was actually run",
        "",
        f"The OOF LightGBM stable-v-transition classifier has ROC-AUC {stable['roc_auc']:.4f}, average precision {stable['average_precision']:.4f}, and Brier {stable['brier']:.4f} on {stable['rows']} observations.",
        f"The OOF morphology classifier agrees with the fold-local GMM morphology label on {morphology['agreement']:.1%} of {morphology['events']} non-abstained events. This is predictability of a fold-local morphology, not proof that GMM component IDs align across folds.",
        (f"The dependency-free rule-list challenger ({brl['backend']}) has OOF ROC-AUC {brl['roc_auc']:.4f}, average precision {brl['average_precision']:.4f}, and Brier {brl['brier']:.4f} on {brl['rows']} observations. It is an ordered Beta-Binomial MAP list, not an MCMC Bayesian Rule List." if brl else "No BRL OOF artifact was available for this report."),
        f"{manifest['headline']['recurrent_fold_local_components']} fold-local components appear in at least two calendar eras. Cross-fold semantic alignment remains unmeasured because the GMM components were fitted independently per fold.",
        "",
        "## Interpretation limits",
        "",
        "The path-geometry table is side-specific and descriptive. The performance calendar is aggregate by era and is not joined to phase outcomes. Therefore this artifact does not establish that a transition morphology improves trading, authorise a gate, or select a state-routing policy.",
        "",
        "See `tool_inventory.csv` for implemented-versus-run status, `gmm_fold_local_component_support.csv` for fold-local recurrence/uncertainty, and `stable_transition_lgbm_oof_metrics.csv` for era and fold variation.",
        "",
    ])


def materialize_audit(
    *,
    panel_path: Path = DEFAULT_PANEL,
    catalogue_dir: Path = DEFAULT_CATALOGUE,
    path_dir: Path = DEFAULT_PATH,
    calendar_path: Path = DEFAULT_CALENDAR,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    panel_path, catalogue_dir, path_dir, calendar_path, output_dir = map(Path, (panel_path, catalogue_dir, path_dir, calendar_path, output_dir))
    morphology = _load_parquet(catalogue_dir / "morphology_oof.parquet")
    stable = _load_parquet(catalogue_dir / "stable_transition_oof.parquet")
    classifier = _load_parquet(catalogue_dir / "morphology_classifier_oof.parquet")
    brl_path = catalogue_dir / "stable_transition_brl_oof.parquet"
    stable_brl = _load_parquet(brl_path) if brl_path.exists() else pd.DataFrame()
    support_path = catalogue_dir / "morphology_recurrence_support.csv"
    support = pd.read_csv(support_path) if support_path.exists() else None
    path = pd.read_csv(path_dir / "path_geometry_by_context.csv")
    calendar = _load_parquet(calendar_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    component = summarize_morphology(morphology, support)
    agreement = morphology_classifier_agreement(morphology, classifier)
    stable_metrics, reliability = stable_transition_metrics(stable)
    brl_metrics, brl_reliability = stable_transition_metrics(stable_brl, probability_column="brl__p_transition") if not stable_brl.empty else (pd.DataFrame(), pd.DataFrame())
    if not brl_metrics.empty:
        brl_metrics.insert(0, "backend", str(stable_brl["classifier_backend"].iloc[0]))
    paths = path_side_consistency(path)
    performance = performance_era_summary(calendar)
    inventory = tool_inventory(ROOT, catalogue_dir)
    for name, data in {
        "tool_inventory.csv": inventory,
        "gmm_fold_local_component_support.csv": component,
        "morphology_classifier_oof_agreement.csv": agreement,
        "stable_transition_lgbm_oof_metrics.csv": stable_metrics,
        "stable_transition_lgbm_reliability.csv": reliability,
        "stable_transition_brl_oof_metrics.csv": brl_metrics,
        "stable_transition_brl_reliability.csv": brl_reliability,
        "transition_phase_side_path_geometry.csv": paths,
        "performance_era_summary.csv": performance,
    }.items():
        data.to_csv(output_dir / name, index=False)

    inputs = {"multihorizon_panel": panel_path, "catalogue": catalogue_dir / "manifest.json", "path_geometry": path_dir / "manifest.json", "performance_calendar": calendar_path}
    manifest = {
        "schema": SCHEMA,
        "purpose": "independent recurring transition taxonomy and stability audit; no residual learner inputs or routing are created",
        "promotion_eligible": False,
        "research_only": True,
        "separation_contract": {
            "persistent_regime": "multihorizon observable panel and separately inventoried discovery tools",
            "transition": "finite-horizon morphology and stable-v-transition labels",
            "forbidden": "no merge of regime and transition state; no ex-post phase as decision-time field; no residual learner integration",
        },
        "panel": panel_summary(panel_path),
        "counts": {
            "morphology_events": int(len(morphology)), "stable_transition_rows": int(len(stable)),
            "stable_transition_brl_rows": int(len(stable_brl)),
            "fold_local_components": int(len(component)), "path_rows": int(len(paths)), "performance_eras": int(len(performance)),
        },
        "headline": {
            "stable_transition_all_oof": stable_metrics.iloc[0].to_dict(),
            "stable_transition_brl_all_oof": brl_metrics.iloc[0].to_dict() if not brl_metrics.empty else None,
            "morphology_classifier_all_events": agreement.iloc[0].to_dict(),
            "morphology_classifier_non_abstained": agreement.iloc[1].to_dict(),
            "recurrent_fold_local_components": int(component["recurs_across_eras"].sum()) if not component.empty else 0,
            "cross_fold_alignment": "not estimated: GMM component IDs are fold-local; semantic alignment needs a separately frozen prototype/matching experiment",
        },
        "inputs_sha256": {name: _sha256(path) for name, path in inputs.items() if path.is_file()},
        "limitations": [
            "GMM component identifiers are fold-local; no cross-fold component alignment is claimed.",
            "Catalogue fit-support is recorded separately from held-out event support.",
            "Path geometry is only materialized through 2024 and is descriptive, not OOF execution evidence.",
            "The performance calendar aggregates the canonical stack by era and does not attribute PnL to a transition morphology.",
        ],
        "output_sha256": {item.name: _sha256(item) for item in sorted(output_dir.iterdir()) if item.is_file()},
    }
    (output_dir / "REPORT.md").write_text(render_report(manifest), encoding="utf-8")
    manifest["output_sha256"]["REPORT.md"] = _sha256(output_dir / "REPORT.md")
    _write_json(output_dir / "manifest.json", manifest)
    digest = _sha256(output_dir / "manifest.json")
    (output_dir / "manifest.sha256").write_text(f"{digest}  manifest.json\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--catalogue", type=Path, default=DEFAULT_CATALOGUE)
    parser.add_argument("--path-geometry", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--calendar", type=Path, default=DEFAULT_CALENDAR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = materialize_audit(panel_path=args.panel, catalogue_dir=args.catalogue, path_dir=args.path_geometry, calendar_path=args.calendar, output_dir=args.output)
    print(json.dumps(_safe({"output": str(args.output), "headline": manifest["headline"]}), indent=2))


if __name__ == "__main__":
    main()
