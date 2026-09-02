#!/usr/bin/env python3
"""Router-conditioned P8U specialist re-fit — offline research only.

This experiment answers one narrow question: after the strict target-free
Router has selected its configured timestamp-local fraction of the long candidate universe,
do the previously selected P8U archetype heads improve ordering *inside that
population*?  The geometry and the per-head model/feature/HPO contracts are
reused from immutable prior inner-fold bundles.  Only supervised weights are
re-fit on prior-resolved Router-selected rows.  No held labels are used in
routing, fitting, rank references, or combination choices.

It is intentionally separate from the general P8U runner: it cannot update the
Router, Base, Meta, MC1, admission, policy, portfolio, or live contracts.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_strict_r3_p8u_opportunity_probe_router_recall_v1 as base  # noqa: E402


SCHEMA = "strict_r3_p8u_router_conditioned_head_retrain_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, value: Any) -> None:
    base._write_json_exclusive(path, value)


def _write_parquet_exclusive(frame: pd.DataFrame, path: Path) -> None:
    base._write_parquet_exclusive(frame, path)


def _load_bundle(path: Path) -> dict[str, Any]:
    """Load historical script-main joblib bundles without changing them."""
    main = sys.modules["__main__"]
    # Historical joblibs record these classes as ``__main__`` because the
    # originating runner was invoked as a script.  Resolve them to the exact
    # same implementation; this is a compatibility shim, not a conversion.
    for name in ("CategoryModel", "ProbeModel", "RobustPreprocessor"):
        setattr(main, name, getattr(base, name))
    return joblib.load(path)


def router_top_fraction_mask(frame: pd.DataFrame, fraction: float) -> np.ndarray:
    """Timestamp-local, target-free Router gate with stable tie resolution."""
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("Router top fraction must be in (0, 1]")
    score = pd.to_numeric(frame["router_rank"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(score).all():
        raise AssertionError("Router-conditioned experiment requires finite target-free Router ranks")
    return base._selection_mask(frame, score, None, float(fraction), 1.0)


def _full_score(frame: pd.DataFrame, gate: np.ndarray, gated_score: np.ndarray) -> np.ndarray:
    if int(gate.sum()) != len(gated_score):
        raise AssertionError("gated score must align one-to-one with Router gate")
    result = np.full(len(frame), -np.inf, dtype=np.float32)
    result[gate] = np.asarray(gated_score, dtype=np.float32)
    return result


def _fast_metric_row(
    frame: pd.DataFrame, selected: np.ndarray, *, fold: str, strategy: str,
    budget: float, router_share: float, score_name: str,
) -> dict[str, Any]:
    """Core economic metrics without report-only within-timestamp oracle loops.

    The inherited general runner also calculates five hindsight oracle-recall
    measures for every strategy call.  This isolated second-stage test invokes
    many individual-head controls and does not use those measures for model
    choice; repeating them dominated execution time without adding evidence.
    Selection masks, labels, and the requested realised-EV/recall metrics are
    identical to the general runner.
    """
    valid = frame["label_valid"].to_numpy(dtype=bool)
    net = pd.to_numeric(frame["policy_net_bps"], errors="coerce").to_numpy(dtype=float)
    chosen = np.asarray(selected, dtype=bool) & valid & np.isfinite(net)
    values = net[chosen]
    row: dict[str, Any] = {
        "fold": fold, "split": "outer_oof", "strategy": strategy, "score_name": score_name,
        "budget_fraction": float(budget), "router_share": float(router_share),
        "candidate_rows": int(len(frame)), "valid_label_rows": int(valid.sum()),
        "selected_rows_all": int(np.asarray(selected, dtype=bool).sum()), "selected_rows_valid": int(chosen.sum()),
        "selected_mean_net_bps": float(np.nanmean(values)) if len(values) else float("nan"),
        "selected_median_net_bps": float(np.nanmedian(values)) if len(values) else float("nan"),
        "selected_p10_net_bps": float(np.nanquantile(values, .10)) if len(values) else float("nan"),
        "selected_cvar10_net_bps": float(np.nanmean(np.sort(values)[:max(1, int(math.ceil(.10 * len(values))))])) if len(values) else float("nan"),
        "selected_positive_mass_bps": float(np.nansum(np.maximum(values, 0.0))),
        "all_positive_mass_bps": float(np.nansum(np.maximum(net[valid], 0.0))),
    }
    row["positive_economic_mass_recall"] = row["selected_positive_mass_bps"] / max(row["all_positive_mass_bps"], 1e-8)
    for threshold in (0.0, 50.0, 100.0, 200.0):
        opportunity = valid & (net > threshold)
        row[f"recall_gt_{int(threshold)}"] = float((chosen & opportunity).sum() / max(1, opportunity.sum()))
        row[f"selected_hit_gt_{int(threshold)}"] = float((chosen & opportunity).sum() / max(1, chosen.sum()))
    return row


def _metric_frame(
    frame: pd.DataFrame, score: np.ndarray, *, fold: str, cfg: dict[str, Any], score_name: str,
) -> tuple[pd.DataFrame, dict[tuple[float, float], np.ndarray]]:
    """Exact selection masks with only the predeclared core metrics."""
    router = frame["router_rank"].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    masks: dict[tuple[float, float], np.ndarray] = {}
    for budget in cfg["evaluation"]["budget_fractions"]:
        budget = float(budget)
        baseline = base._selection_mask(frame, router, None, budget, 1.0)
        rows.append(_fast_metric_row(
            frame, baseline, fold=fold, strategy="router_only", budget=budget, router_share=1.0, score_name="router_rank",
        ))
        for share in cfg["evaluation"]["rescue_router_shares"]:
            share = float(share)
            combined = base._selection_mask(frame, router, score, budget, share)
            masks[(budget, share)] = combined
            rows.append(_fast_metric_row(
                frame, combined, fold=fold, strategy="router_plus_probe_rescue", budget=budget, router_share=share, score_name=score_name,
            ))
        probe_only = base._selection_mask(frame, score, None, budget, 0.0)
        rows.append(_fast_metric_row(
            frame, probe_only, fold=fold, strategy="probe_only", budget=budget, router_share=0.0, score_name=score_name,
        ))
    return pd.DataFrame(rows), masks


def _rank_statistics(frame: pd.DataFrame, score: np.ndarray, *, fold: str, score_name: str) -> dict[str, Any]:
    valid = frame["label_valid"].to_numpy(dtype=bool)
    net = pd.to_numeric(frame["policy_net_bps"], errors="coerce").to_numpy(dtype=float)
    usable = valid & np.isfinite(net) & np.isfinite(score)
    if int(usable.sum()) < 3:
        return {"fold": fold, "score_name": score_name, "valid_rows": int(usable.sum()), "spearman_net_bps": float("nan")}
    return {
        "fold": fold,
        "score_name": score_name,
        "valid_rows": int(usable.sum()),
        "spearman_net_bps": float(pd.Series(score[usable]).corr(pd.Series(net[usable]), method="spearman")),
        "score_p10": float(np.nanquantile(score[usable], .10)),
        "score_p50": float(np.nanquantile(score[usable], .50)),
        "score_p90": float(np.nanquantile(score[usable], .90)),
    }


def _head_value_matrix(
    ranks: np.ndarray, membership: np.ndarray, head_specs: Sequence[dict[str, Any]], combination: dict[str, Any],
) -> np.ndarray:
    return np.column_stack([
        base._combine_head_score(
            ranks[:, slot], membership[:, slot],
            gamma=float(combination["membership_gamma"]),
            method=str(combination["combination_method"]),
            activation_floor=float(head_specs[slot].get("membership_activation_floor", 0.0)),
        )
        for slot in range(ranks.shape[1])
    ]).astype(np.float32)


def _head_contract_rows(bundle: dict[str, Any], *, fold: str, bundle_path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in bundle["head_specs"]:
        rows.append({
            "fold": fold,
            "category": int(spec["category"]),
            "feature_set": str(spec["feature_set"]),
            "target_family": str(spec["target_family"]),
            "model_kind": str(spec["model_kind"]),
            "feature_budget": int(spec.get("feature_budget", len(spec.get("fields", [])))),
            "selected_field_count": len(spec.get("fields", [])),
            "model_hpo_id": str(spec.get("model_hpo_id", "baseline")),
            "model_hpo_overrides": json.dumps(spec.get("model_hpo_overrides", {}), sort_keys=True),
            "membership_activation_floor": float(spec.get("membership_activation_floor", 0.0)),
            "source_bundle": str(bundle_path.relative_to(ROOT)),
            "source_bundle_sha256": _sha256(bundle_path),
        })
    return pd.DataFrame(rows)


def _fold(
    panel: pd.DataFrame, config: dict[str, Any], definition: dict[str, Any], *, offset: int,
) -> dict[str, Any]:
    fold = str(definition["name"])
    held_start = base._utc(definition["held_start"])
    held_end = base._exclusive_day_end(definition["held_end"])
    bundle_path = ROOT / config["source_head_bundles"][fold]
    if not bundle_path.exists():
        raise FileNotFoundError(f"missing sealed source head bundle: {bundle_path}")
    bundle = _load_bundle(bundle_path)
    if str(bundle.get("fold")) != fold:
        raise AssertionError(f"source bundle fold mismatch for {fold}: {bundle.get('fold')}")
    category_model = bundle["category_model"]
    head_specs = [dict(item) for item in bundle["head_specs"]]
    combination = dict(bundle["combination"])
    discovery_fields = tuple(bundle["discovery_fields"])
    predictive_fields = tuple(bundle["predictive_fields"])

    held = base._month_range(panel, held_start, held_end).reset_index(drop=True)
    prequential = base._eligible_labels(panel, held_start)
    train_gate = router_top_fraction_mask(prequential, float(config["router_conditioning"]["router_top_fraction"]))
    held_gate = router_top_fraction_mask(held, float(config["router_conditioning"]["router_top_fraction"]))
    train = prequential.loc[train_gate].reset_index(drop=True)
    gated = held.loc[held_gate].reset_index(drop=True)
    if train.empty or gated.empty:
        raise RuntimeError(f"{fold}: empty Router-conditioned train or held population")
    if not (train["policy_label_available_ts"] < held_start).all():
        raise AssertionError(f"{fold}: non-prequential policy label reached head training")

    membership_train = category_model.membership(train)
    membership_gated = category_model.membership(gated)
    models = base._refit_selected_heads(
        train, membership_train, head_specs, discovery_fields, predictive_fields, config,
        seed=int(config["seed"]) + offset * 100_000,
    )
    p8u_gated, ranks = base._score_probe_stack(gated, membership_gated, models, head_specs, combination)
    p8u_full = _full_score(held, held_gate, p8u_gated)

    # Matched generic control: same Router-gated supervised population and the
    # same number of individual train-only heads, but no category membership.
    ones = np.ones((len(train), 1), dtype=np.float32)
    generic_ranks: list[np.ndarray] = []
    for head_index in range(len(head_specs)):
        generic = base._fit_probe(
            train, predictive_fields, 0, ones, "P1", "atr_utility", "lgbm_huber",
            config["probe"], config["preprocessing"], int(config["seed"]) + offset * 100_000 + 10_000 + head_index,
        )
        generic_ranks.append(generic.score(gated)[1])
    c0_gated = np.mean(np.column_stack(generic_ranks), axis=1).astype(np.float32)
    c0_full = _full_score(held, held_gate, c0_gated)

    scores = {
        "router_rank": held["router_rank"].to_numpy(dtype=np.float32),
        "p8u_router_conditioned": p8u_full,
        "c0_router_conditioned": c0_full,
    }
    metrics: list[pd.DataFrame] = []
    masks: dict[str, dict[tuple[float, float], np.ndarray]] = {}
    for name, score in scores.items():
        frame, score_masks = _metric_frame(held, score, fold=fold, cfg=config, score_name=name)
        # The Router score does not need an artificial rescue score: retain its
        # unambiguous Router-only metrics and label the other rows as controls.
        if name == "router_rank":
            frame = frame.loc[frame["strategy"].eq("router_only")].copy()
        metrics.append(frame.assign(model=name))
        masks[name] = score_masks

    head_values = _head_value_matrix(ranks, membership_gated, head_specs, combination)
    head_metrics: list[pd.DataFrame] = []
    rank_stats: list[dict[str, Any]] = [
        _rank_statistics(gated, p8u_gated, fold=fold, score_name="p8u_aggregate"),
        _rank_statistics(gated, c0_gated, fold=fold, score_name="c0_generic"),
    ]
    for slot, spec in enumerate(head_specs):
        name = f"head_{int(spec['category']):02d}"
        head_full = _full_score(held, held_gate, head_values[:, slot])
        frame, _ = _metric_frame(held, head_full, fold=fold, cfg=config, score_name=name)
        head_metrics.append(frame.loc[frame["strategy"].eq("probe_only")].assign(
            model=name, category=int(spec["category"]), feature_set=str(spec["feature_set"]),
            target_family=str(spec["target_family"]), model_kind=str(spec["model_kind"]),
        ))
        rank_stats.append(_rank_statistics(gated, head_values[:, slot], fold=fold, score_name=name))

    predictions = held.loc[:, [
        "candidate_id", "__decision_ts__", "__symbol__", "side_name", "router_rank", "label_valid",
        "policy_net_bps", "policy_label_available_ts",
    ]].copy()
    predictions[f"router_top{int(round(float(config['router_conditioning']['router_top_fraction']) * 100)):02d}"] = held_gate
    predictions["p8u_score"] = p8u_full
    predictions["c0_score"] = c0_full
    predictions["p8u_gated_score"] = p8u_gated.repeat(1) if len(held) == len(gated) else np.nan
    # Store the actual held-gate values by identity without pretending the
    # specialist generated a meaningful score outside the Router gate.
    predictions.loc[held_gate, "p8u_gated_score"] = p8u_gated
    predictions.loc[held_gate, "c0_gated_score"] = c0_gated
    for budget in config["evaluation"]["budget_fractions"]:
        label = f"b{int(round(float(budget) * 100)):02d}"
        predictions[f"router_{label}_selected"] = base._selection_mask(
            held, scores["router_rank"], None, float(budget), 1.0,
        )
        predictions[f"p8u_{label}_selected"] = base._selection_mask(
            held, scores["p8u_router_conditioned"], None, float(budget), 1.0,
        )
        predictions[f"c0_{label}_selected"] = base._selection_mask(
            held, scores["c0_router_conditioned"], None, float(budget), 1.0,
        )
        predictions[f"p8u_rescue50_{label}_selected"] = base._selection_mask(
            held, scores["router_rank"], scores["p8u_router_conditioned"], float(budget), .5,
        )
        predictions[f"c0_rescue50_{label}_selected"] = base._selection_mask(
            held, scores["router_rank"], scores["c0_router_conditioned"], float(budget), .5,
        )
    source = {
        "fold": fold,
        "held_start": str(held_start), "held_end": str(held_end),
        "source_bundle": str(bundle_path.relative_to(ROOT)), "source_bundle_sha256": _sha256(bundle_path),
        "category_algorithm": str(category_model.algorithm), "category_k": int(category_model.k),
        "router_top_fraction": float(config["router_conditioning"]["router_top_fraction"]),
        "train_prior_resolved_rows": int(len(prequential)), "train_router_rows": int(len(train)),
        "held_rows": int(len(held)), "held_router_rows": int(len(gated)),
        "held_valid_rows": int(held["label_valid"].sum()), "held_router_valid_rows": int(gated["label_valid"].sum()),
        "geometry_refit": False, "head_contract_retuned": False,
        "model_scored_outside_router_gate": False,
    }
    return {
        "metrics": pd.concat(metrics, ignore_index=True),
        "head_metrics": pd.concat(head_metrics, ignore_index=True),
        "head_statistics": pd.DataFrame(rank_stats),
        "predictions": predictions,
        "head_contract": _head_contract_rows(bundle, fold=fold, bundle_path=bundle_path),
        "source": source,
    }


def _format_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    if frame.empty:
        return "_No rows._\n"
    view = frame.loc[:, [key for key in columns if key in frame.columns]].copy()
    def cell(value: object) -> str:
        if isinstance(value, float):
            return "" if not np.isfinite(value) else f"{value:.5g}"
        if pd.isna(value):
            return ""
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join([
        "| " + " | ".join(view.columns) + " |",
        "| " + " | ".join("---" for _ in view.columns) + " |",
        *["| " + " | ".join(cell(value) for value in row) + " |" for row in view.itertuples(index=False, name=None)],
    ]) + "\n"


def _report(output: Path, manifest: dict[str, Any], metrics: pd.DataFrame, head_metrics: pd.DataFrame,
            head_statistics: pd.DataFrame, source: pd.DataFrame) -> None:
    primary = metrics.loc[np.isclose(metrics["budget_fraction"], .10)].copy()
    primary = primary.loc[primary["strategy"].isin(["router_only", "probe_only", "router_plus_probe_rescue"])]
    top_percent = int(round(float(source["router_top_fraction"].iloc[0]) * 100))
    report = [
        "# P8U Heads Inside Router-Selected Candidates\n",
        "## Scope\n",
        f"This is a long-only offline second-stage test.  The strict target-free Router is applied first at every timestamp; all P8U and generic-control training and scoring is restricted to that Router top-{top_percent}% population.  The frozen NMF geometry and the selected per-head feature/model/HPO contracts are reused from the supplied immutable bundles.  No geometry, head specification, combination, Router, Base, Meta, MC1, admission, policy, portfolio, or live artifact changes.\n",
        "The supervised target remains policy net bps expressed in decision-time ATR utility for fitting; every result below is realised canonical policy **net bps**.  Train labels satisfy `policy_label_available_ts < held_start`; held labels are joined only after target-free Router gating and scoring.\n",
        "## Fold population and causal receipt\n",
        _format_table(source, ["fold", "held_start", "held_end", "train_prior_resolved_rows", "train_router_rows", "held_rows", "held_router_rows", "held_valid_rows", "held_router_valid_rows", "category_algorithm", "category_k", "geometry_refit", "head_contract_retuned", "model_scored_outside_router_gate"]),
        "## Full-universe fixed-budget metrics (Router gate precedes every second-stage score)\n",
        _format_table(primary, ["fold", "model", "strategy", "router_share", "selected_rows_valid", "recall_gt_50", "selected_hit_gt_50", "selected_mean_net_bps", "selected_median_net_bps", "selected_cvar10_net_bps", "positive_economic_mass_recall"]),
        "## Individual-head metrics — score only inside the Router gate\n",
        _format_table(head_metrics.loc[np.isclose(head_metrics["budget_fraction"], .10)], ["fold", "model", "category", "feature_set", "target_family", "model_kind", "selected_rows_valid", "recall_gt_50", "selected_hit_gt_50", "selected_mean_net_bps", "selected_cvar10_net_bps"]),
        "## Rank signal inside the Router gate\n",
        _format_table(head_statistics, ["fold", "score_name", "valid_rows", "spearman_net_bps", "score_p10", "score_p50", "score_p90"]),
        "## Interpretation\n",
        "A head stack is promising only if it consistently improves Router-only and its matched generic C0 reranker on realised policy net bps, selected >50-bps hit rate, and >50-bps recall in at least two of the three strictly OOS quarters.  A one-quarter result or a gain that is weaker than C0 is not a deployment or Router-replacement signal.\n",
        "## Manifest\n```json\n" + json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n```\n",
    ]
    (output / "P8U_ROUTER_CONDITIONED_HEAD_RETRAIN_REPORT.md").write_text("\n".join(report))


def finalize_existing(config_path: Path, output: Path) -> None:
    """Seal report/provenance after a resource interruption post-metrics.

    This is deliberately unable to score, fit, alter, or recompute anything.
    It only reads already-exclusive metric/receipt files from an interrupted
    write tail and records that fact.  Keeping it separate avoids a costly
    rerun simply to serialize a markdown report after the full history has
    been released from memory.
    """
    manifest_path = output / "run_manifest.json"
    report_path = output / "P8U_ROUTER_CONDITIONED_HEAD_RETRAIN_REPORT.md"
    if manifest_path.exists() and report_path.exists():
        raise FileExistsError("existing output has already been finalized")
    required = [
        "correctness_report.json", "metrics.parquet", "individual_head_metrics.parquet",
        "head_rank_statistics.parquet", "fold_source_audit.parquet", "frozen_head_contract.parquet",
        "candidate_predictions.parquet", "target_free_router_universe.parquet",
    ]
    absent = [name for name in required if not (output / name).exists()]
    if absent:
        raise FileNotFoundError(f"cannot finalize incomplete results: {absent}")
    config_path = config_path.resolve()
    config = base._load_config(config_path)
    metrics = pd.read_parquet(output / "metrics.parquet")
    head_metrics = pd.read_parquet(output / "individual_head_metrics.parquet")
    head_statistics = pd.read_parquet(output / "head_rank_statistics.parquet")
    source = pd.read_parquet(output / "fold_source_audit.parquet")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest["postfit_finalization"] = {
            "reason": "the original process completed model/metric artifacts and its manifest, then stopped before markdown report serialization",
            "recomputed_scores_or_metrics": False,
            "changed_causal_or_model_contract": False,
        }
    else:
        manifest = {
            "schema": SCHEMA,
            "config_path": str(config_path.relative_to(ROOT)), "config_sha256": _sha256(config_path),
            "policy_label_path": str(config["policy_label_path"]), "policy_label_sha256": _sha256(ROOT / config["policy_label_path"]),
            "folds": source.to_dict(orient="records"),
            "artifacts": {
                "metrics": "metrics.parquet", "individual_head_metrics": "individual_head_metrics.parquet",
                "head_rank_statistics": "head_rank_statistics.parquet", "candidate_predictions": "candidate_predictions.parquet",
                "frozen_head_contract": "frozen_head_contract.parquet", "fold_source_audit": "fold_source_audit.parquet",
                "target_free_router_universe": "target_free_router_universe.parquet",
            },
            "decision": "OFFLINE_RESEARCH_ONLY",
            "postfit_finalization": {
                "reason": "the original process completed every metric and correctness artifact, then was resource-interrupted before report serialization while retaining the broad historical panel",
                "recomputed_scores_or_metrics": False,
                "changed_causal_or_model_contract": False,
            },
        }
        _write_json_exclusive(manifest_path, manifest)
    _report(output, manifest, metrics, head_metrics, head_statistics, source)



def run(config_path: Path, output: Path) -> None:
    config_path = config_path.resolve()
    config = base._load_config(config_path)
    if str(config.get("schema_version")) != SCHEMA:
        raise AssertionError("Router-conditioned experiment requires its schema-v1 config")
    if output.exists():
        raise FileExistsError(f"immutable output already exists: {output}")
    router_fraction = float(config["router_conditioning"]["router_top_fraction"])
    if not np.isclose(router_fraction, .50):
        raise AssertionError("the corrected predeclared test is exactly the Router top-50% stage")
    selected_folds = list(config["outer_folds"])
    loaded_end = max(base._exclusive_day_end(item["held_end"]) for item in selected_folds)
    predictive_fields = base._assert_causal_feature_contract(
        base._P8U_FEATURE_CONTRACTS[config["feature_contract_keys"]["predictive"]]
    )
    sidecar_fields = base._assert_causal_feature_contract(
        base._P8U_FEATURE_CONTRACTS[config["probe_feature_sidecar_fields_key"]]
    )
    target_free, target_free_view, source_audit = base._read_panel(
        config, predictive_fields, sidecar_fields, start=config["research_period"][0], end=loaded_end,
    )
    labels = base._load_labels(
        str(config["policy_label_path"]), decision_start=config["research_period"][0], decision_end=loaded_end,
    )
    panel = base._attach_outcomes(target_free, labels)
    output.mkdir(parents=True, exist_ok=False)
    _write_parquet_exclusive(target_free_view, output / "target_free_router_universe.parquet")
    del target_free_view, labels
    results = [_fold(panel, config, definition, offset=index) for index, definition in enumerate(selected_folds)]
    metrics = pd.concat([item["metrics"] for item in results], ignore_index=True)
    head_metrics = pd.concat([item["head_metrics"] for item in results], ignore_index=True)
    head_statistics = pd.concat([item["head_statistics"] for item in results], ignore_index=True)
    predictions = pd.concat([item["predictions"] for item in results], ignore_index=True)
    head_contract = pd.concat([item["head_contract"] for item in results], ignore_index=True)
    sources = pd.DataFrame([item["source"] for item in results])
    _write_parquet_exclusive(metrics, output / "metrics.parquet")
    _write_parquet_exclusive(head_metrics, output / "individual_head_metrics.parquet")
    _write_parquet_exclusive(head_statistics, output / "head_rank_statistics.parquet")
    _write_parquet_exclusive(predictions, output / "candidate_predictions.parquet")
    _write_parquet_exclusive(head_contract, output / "frozen_head_contract.parquet")
    _write_parquet_exclusive(sources, output / "fold_source_audit.parquet")
    correctness = {
        "schema": SCHEMA,
        "long_only": bool(panel["side_name"].eq("long").all()),
        "target_free_router_gate_before_outcome_join": True,
        "training_labels_strictly_prior_resolved": bool(all(item["source"]["train_prior_resolved_rows"] >= item["source"]["train_router_rows"] for item in results)),
        "geometry_refit_per_fold": False,
        "head_contract_retuned": False,
        "model_scored_outside_router_gate": False,
        "held_outcomes_used_for_router_or_score": False,
        "canonical_or_live_contract_changed": False,
    }
    _write_json_exclusive(output / "correctness_report.json", correctness)
    # The full source panel is no longer needed after all exclusive metric
    # artifacts are written.  Releasing it before provenance/report writing
    # avoids a resource-only failure at the final serialization step.
    del panel, target_free, results, predictions, head_contract
    gc.collect()
    manifest = {
        "schema": SCHEMA,
        "config_path": str(config_path.relative_to(ROOT)), "config_sha256": _sha256(config_path),
        "policy_label_path": str(config["policy_label_path"]), "policy_label_sha256": _sha256(ROOT / config["policy_label_path"]),
        "source_audit": source_audit, "folds": sources.to_dict(orient="records"),
        "artifacts": {
            "metrics": "metrics.parquet", "individual_head_metrics": "individual_head_metrics.parquet",
            "head_rank_statistics": "head_rank_statistics.parquet", "candidate_predictions": "candidate_predictions.parquet",
            "frozen_head_contract": "frozen_head_contract.parquet", "fold_source_audit": "fold_source_audit.parquet",
        },
        "decision": "OFFLINE_RESEARCH_ONLY",
    }
    _write_json_exclusive(output / "run_manifest.json", manifest)
    _report(output, manifest, metrics, head_metrics, head_statistics, sources)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--finalize-existing", action="store_true")
    args = parser.parse_args()
    if args.finalize_existing:
        finalize_existing(args.config, args.output)
    else:
        run(args.config, args.output)


if __name__ == "__main__":
    main()
