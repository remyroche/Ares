#!/usr/bin/env python3
"""Strict-OOF broad P8U rescue and Router/P8U blend ablation.

This is deliberately an offline Router-stage experiment.  Unlike the prior
Router-conditioned P8U check, P8U is fit and scored on the full target-free
candidate universe, so it can rescue candidates that the Router would exclude
from its timestamp-local top-50% capacity.  Geometry and selected per-head
contracts are loaded from immutable fold bundles; neither geometry nor HPO is
rediscovered here.  No downstream stack, admission, policy, portfolio, or live
artifact is read or changed.
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

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_strict_r3_p8u_opportunity_probe_router_recall_v1 as base  # noqa: E402
import run_strict_r3_p8u_router_conditioned_head_retrain_v1 as conditioned  # noqa: E402


SCHEMA = "strict_r3_p8u_router50_broad_rescue_recall_v1"


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


def _timestamp_rank(frame: pd.DataFrame, score: np.ndarray) -> np.ndarray:
    """Target-free deterministic [0,1] percentile rank at each timestamp."""
    values = np.asarray(score, dtype=float)
    if len(values) != len(frame) or not np.isfinite(values).all():
        raise ValueError("timestamp-rank inputs must be finite and identity-aligned")
    work = pd.DataFrame({
        "__decision_ts__": frame["__decision_ts__"].to_numpy(),
        "candidate_id": frame["candidate_id"].to_numpy(),
        "score": values,
    }, index=frame.index)
    # Ascending gives lowest=0 and highest=1.  Candidate ID gives a stable
    # resolution for tied values without depending on source row order.
    ordered = work.sort_values(
        ["__decision_ts__", "score", "candidate_id"],
        ascending=[True, True, True], kind="stable",
    )
    position = ordered.groupby("__decision_ts__", sort=False).cumcount().to_numpy(dtype=float)
    size = ordered.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size").to_numpy(dtype=float)
    rank = np.where(size > 1.0, position / (size - 1.0), .5)
    result = np.empty(len(frame), dtype=np.float32)
    result[ordered.index.to_numpy(dtype=int)] = rank.astype(np.float32)
    return result


def _valid_selection_metrics(
    frame: pd.DataFrame, selected: np.ndarray, *, fold: str, family: str, arm: str,
    router_share: float | None = None, p8u_weight: float | None = None,
) -> dict[str, Any]:
    valid = frame["label_valid"].to_numpy(dtype=bool)
    net = pd.to_numeric(frame["policy_net_bps"], errors="coerce").to_numpy(dtype=float)
    selected = np.asarray(selected, dtype=bool)
    chosen = selected & valid & np.isfinite(net)
    values = net[chosen]
    row: dict[str, Any] = {
        "fold": fold,
        "family": family,
        "arm": arm,
        "router_share": np.nan if router_share is None else float(router_share),
        "p8u_weight": np.nan if p8u_weight is None else float(p8u_weight),
        "candidate_rows": int(len(frame)),
        "valid_label_rows": int((valid & np.isfinite(net)).sum()),
        "selected_rows_all": int(selected.sum()),
        "selected_rows_valid": int(chosen.sum()),
        "selected_fraction_all": float(selected.mean()),
        "selected_mean_net_bps": float(np.nanmean(values)) if len(values) else float("nan"),
        "selected_median_net_bps": float(np.nanmedian(values)) if len(values) else float("nan"),
        "selected_p10_net_bps": float(np.nanquantile(values, .10)) if len(values) else float("nan"),
        "selected_cvar10_net_bps": float(np.nanmean(np.sort(values)[:max(1, math.ceil(.10 * len(values)))])) if len(values) else float("nan"),
        "selected_total_net_bps": float(np.nansum(values)),
        "selected_positive_mass_bps": float(np.nansum(np.maximum(values, 0.0))),
        "all_positive_mass_bps": float(np.nansum(np.maximum(net[valid & np.isfinite(net)], 0.0))),
    }
    row["positive_economic_mass_recall"] = row["selected_positive_mass_bps"] / max(row["all_positive_mass_bps"], 1e-8)
    for threshold in (0.0, 50.0, 100.0, 200.0):
        opportunity = valid & np.isfinite(net) & (net > threshold)
        row[f"recall_gt_{int(threshold)}"] = float((chosen & opportunity).sum() / max(1, opportunity.sum()))
        row[f"selected_hit_gt_{int(threshold)}"] = float((chosen & opportunity).sum() / max(1, chosen.sum()))
    return row


def _assert_exact_capacity(frame: pd.DataFrame, selected: np.ndarray, fraction: float) -> None:
    check = frame.loc[:, ["__decision_ts__"]].copy()
    check["selected"] = np.asarray(selected, dtype=bool)
    counts = check.groupby("__decision_ts__", sort=False).agg(total=("selected", "size"), kept=("selected", "sum"))
    expected = np.maximum(1, np.ceil(counts["total"].to_numpy(dtype=float) * float(fraction)).astype(int))
    if not np.array_equal(counts["kept"].to_numpy(dtype=int), expected):
        raise AssertionError("timestamp-local capacity differs from frozen Router top-50% contract")


def _wide_predictions(frame: pd.DataFrame, *, router_rank: np.ndarray, p8u_score: np.ndarray,
                      p8u_rank: np.ndarray, generic_score: np.ndarray, generic_rank: np.ndarray) -> pd.DataFrame:
    return frame.loc[:, [
        "candidate_id", "__decision_ts__", "__symbol__", "side_name", "router_rank",
        "label_valid", "policy_net_bps", "policy_label_available_ts",
    ]].assign(
        router_timestamp_rank=_timestamp_rank(frame, router_rank),
        p8u_score=p8u_score.astype(np.float32),
        p8u_timestamp_rank=p8u_rank.astype(np.float32),
        generic_score=generic_score.astype(np.float32),
        generic_timestamp_rank=generic_rank.astype(np.float32),
    )


def _fit_generic_control(
    train: pd.DataFrame, held: pd.DataFrame, head_specs: Sequence[dict[str, Any]],
    discovery_fields: Sequence[str], predictive_fields: Sequence[str], config: dict[str, Any], *, seed: int,
) -> np.ndarray:
    """Same full-universe head count/contracts, but no archetype membership."""
    membership = np.ones((len(train), 1), dtype=np.float32)
    ranks: list[np.ndarray] = []
    for slot, spec in enumerate(head_specs):
        fields = discovery_fields if str(spec["feature_set"]) == "P0" else predictive_fields
        model = base._fit_probe(
            train, fields, 0, membership, str(spec["feature_set"]),
            str(spec["target_family"]), str(spec["model_kind"]), config["probe"], config["preprocessing"],
            seed + 10_000 + slot * 1009,
            feature_budget=int(spec.get("feature_budget", len(fields))),
            model_overrides=dict(spec.get("model_hpo_overrides", {})),
        )
        ranks.append(model.score(held)[1])
    return np.mean(np.column_stack(ranks), axis=1).astype(np.float32)


def _fold(train: pd.DataFrame, held: pd.DataFrame, config: dict[str, Any], definition: dict[str, Any], *, offset: int) -> dict[str, Any]:
    fold = str(definition["name"])
    held_start = base._utc(definition["held_start"])
    held_end = base._exclusive_day_end(definition["held_end"])
    bundle_path = ROOT / config["source_head_bundles"][fold]
    if not bundle_path.exists():
        raise FileNotFoundError(f"missing sealed source head bundle: {bundle_path}")
    bundle = conditioned._load_bundle(bundle_path)
    if str(bundle.get("fold")) != fold:
        raise AssertionError(f"source bundle fold mismatch: {bundle.get('fold')} != {fold}")
    category_model = bundle["category_model"]
    head_specs = [dict(item) for item in bundle["head_specs"]]
    combination = dict(bundle["combination"])
    discovery_fields = tuple(bundle["discovery_fields"])
    predictive_fields = tuple(bundle["predictive_fields"])

    if train.empty or held.empty:
        raise RuntimeError(f"{fold}: empty strict-prequential training or held population")
    if not (train["policy_label_available_ts"] < held_start).all():
        raise AssertionError(f"{fold}: later-resolved label reached P8U training")

    membership_train = category_model.membership(train)
    membership_held = category_model.membership(held)
    models = base._refit_selected_heads(
        train, membership_train, head_specs, discovery_fields, predictive_fields, config,
        seed=int(config["seed"]) + offset * 100_000,
    )
    p8u_score, _ = base._score_probe_stack(held, membership_held, models, head_specs, combination)
    generic_score = _fit_generic_control(
        train, held, head_specs, discovery_fields, predictive_fields, config,
        seed=int(config["seed"]) + offset * 100_000,
    )
    router_rank = held["router_rank"].to_numpy(dtype=np.float32)
    p8u_rank = _timestamp_rank(held, p8u_score)
    router_ts_rank = _timestamp_rank(held, router_rank)
    generic_rank = _timestamp_rank(held, generic_score)
    retained = float(config["evaluation"]["retained_fraction"])

    metrics: list[dict[str, Any]] = []
    selections: dict[str, np.ndarray] = {}
    router_only = base._selection_mask(held, router_rank, None, retained, 1.0)
    _assert_exact_capacity(held, router_only, retained)
    selections["router_100"] = router_only
    metrics.append(_valid_selection_metrics(held, router_only, fold=fold, family="capacity_rescue", arm="router_100", router_share=1.0))

    for share in config["evaluation"]["rescue_router_shares"]:
        share = float(share)
        if np.isclose(share, 1.0):
            continue
        selection = base._selection_mask(held, router_rank, p8u_rank, retained, share)
        _assert_exact_capacity(held, selection, retained)
        # P8U can only consume the explicitly reserved capacity after Router's
        # retained seats have been taken; it never displaces a Router seat by
        # a score tie or by looking at outcomes.
        selections[f"router{int(round(share * 100)):02d}_p8u{int(round((1.0-share)*100)):02d}_rescue"] = selection
        metrics.append(_valid_selection_metrics(
            held, selection, fold=fold, family="capacity_rescue",
            arm=f"router{int(round(share * 100)):02d}_p8u{int(round((1.0-share)*100)):02d}_rescue",
            router_share=share,
        ))

    for weight in config["evaluation"]["blend_p8u_weights"]:
        weight = float(weight)
        blend = (1.0 - weight) * router_ts_rank + weight * p8u_rank
        selection = base._selection_mask(held, blend, None, retained, 1.0)
        _assert_exact_capacity(held, selection, retained)
        name = f"router{int(round((1.0-weight)*100)):02d}_p8u{int(round(weight*100)):02d}_blend"
        selections[name] = selection
        metrics.append(_valid_selection_metrics(
            held, selection, fold=fold, family="score_blend", arm=name, p8u_weight=weight,
        ))

    # This is a matched broad-model negative control, not a candidate for
    # promotion.  It establishes whether frozen archetype memberships add
    # signal over otherwise comparable broad full-universe P8U models.
    generic_blend = .5 * router_ts_rank + .5 * generic_rank
    generic_selection = base._selection_mask(held, generic_blend, None, retained, 1.0)
    _assert_exact_capacity(held, generic_selection, retained)
    selections["router50_generic50_blend"] = generic_selection
    metrics.append(_valid_selection_metrics(
        held, generic_selection, fold=fold, family="matched_generic_control", arm="router50_generic50_blend", p8u_weight=.5,
    ))

    predictions = _wide_predictions(
        held, router_rank=router_rank, p8u_score=p8u_score, p8u_rank=p8u_rank,
        generic_score=generic_score, generic_rank=generic_rank,
    )
    for name, selection in selections.items():
        predictions[f"{name}_selected"] = selection
    source = {
        "fold": fold,
        "held_start": str(held_start), "held_end": str(held_end),
        "source_bundle": str(bundle_path.relative_to(ROOT)), "source_bundle_sha256": _sha256(bundle_path),
        "category_algorithm": str(category_model.algorithm), "category_k": int(category_model.k),
        "train_prior_resolved_rows": int(len(train)), "held_rows": int(len(held)),
        "held_valid_rows": int(held["label_valid"].sum()),
        "geometry_refit": False, "head_contract_retuned": False,
        "p8u_scored_full_target_free_universe": True,
    }
    return {"metrics": pd.DataFrame(metrics), "predictions": predictions, "source": source}


def _month_path(root: Path, month: pd.Timestamp, filename: str) -> Path:
    return root / f"month={month:%Y-%m}" / filename


def _read_parquet_once(path: Path, *, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Bound the source-panel peak RSS to one month without changing inputs."""
    return pd.read_parquet(path, columns=None if columns is None else list(columns))


def _materialize_fold_panel(
    config: dict[str, Any], *, definition: dict[str, Any], predictive_fields: Sequence[str], sidecar_fields: Sequence[str],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    """Read one fold in monthly slices, keeping all held candidates and capped train rows.

    The original generic runner builds a multi-year wide panel before applying
    an already-declared ``max_train_rows`` cap.  That is unnecessary for a
    fixed-head re-fit and can exceed memory.  Here label *availability* is used
    solely to select the same kind of strictly prior-resolved training sample;
    target values remain unjoined until the target-free train/held panels have
    been materialised.  Held candidates are never filtered by label/path.
    """
    fold = str(definition["name"])
    held_start = base._utc(definition["held_start"])
    held_end = base._exclusive_day_end(definition["held_end"])
    labels = base._load_labels(
        str(config["policy_label_path"]), decision_start=config["research_period"][0], decision_end=held_end,
    )
    prior = labels.loc[
        labels["policy_path_valid"].eq(True)
        & labels["policy_label_available_ts"].notna()
        & labels["policy_net_bps"].notna()
        & labels["policy_label_available_ts"].lt(held_start)
    ].copy()
    # Retain the frozen model cap, but apply it once before loading wide rows.
    # This is an offline memory-equivalent materialisation of the established
    # train cap, not a held-period feature/target selection.
    cap = int(config["probe"]["max_train_rows"])
    prior = prior.sort_values(["__decision_ts__", "candidate_id"], kind="stable")
    sampled_prior = base._timestamp_sample(prior, cap, seed + 37).reset_index(drop=True)
    train_ids = set(sampled_prior["candidate_id"].astype(str))
    if not train_ids:
        raise RuntimeError(f"{fold}: no prior-resolved label-valid training identities")

    base_fields = tuple(field for field in predictive_fields if field not in set(sidecar_fields))
    parts: list[pd.DataFrame] = []
    held_target_free: list[pd.DataFrame] = []
    audit: list[dict[str, Any]] = []
    months = base._month_starts(config["research_period"][0], held_end - pd.Timedelta(nanoseconds=1))
    for month in months:
        feature_source = base._source_for(month, config["feature_sources"])
        score_source = base._source_for(month, config["router_score_sources"])
        feature_path = _month_path(ROOT / feature_source["root"], month, "causal_feature_universe.parquet")
        sidecar_path = ROOT / config["probe_feature_sidecar_root"] / f"month={month:%Y-%m}" / "causal_probe_intraday_features.parquet"
        score_root = ROOT / score_source["root"]
        score_path = score_root / "target_free_scores" / f"month={month:%Y-%m}.parquet"
        if not score_path.exists():
            score_path = _month_path(score_root, month, "raw_oos_predictions.parquet")
        if not feature_path.exists() or not sidecar_path.exists() or not score_path.exists():
            raise FileNotFoundError(f"{fold}: missing declared monthly source for {month:%Y-%m}")
        feature = _read_parquet_once(feature_path, columns=[*base.IDENTITY_COLUMNS, *base_fields])
        feature["__decision_ts__"] = pd.to_datetime(feature["__decision_ts__"], utc=True)
        feature["side_name"] = feature["side_name"].astype(str).str.lower()
        feature = feature.loc[feature["side_name"].eq(str(config["side"]).lower())].copy()
        is_held = feature["__decision_ts__"].ge(held_start) & feature["__decision_ts__"].lt(held_end)
        is_train = feature["candidate_id"].astype(str).isin(train_ids)
        feature = feature.loc[is_held | is_train].copy()
        if feature.empty:
            continue
        sidecar = _read_parquet_once(sidecar_path, columns=[*base.IDENTITY_COLUMNS, *sidecar_fields])
        sidecar = sidecar.loc[sidecar["candidate_id"].isin(feature["candidate_id"])].copy()
        if sidecar["candidate_id"].duplicated().any():
            raise AssertionError(f"{fold}/{month:%Y-%m}: duplicate causal sidecar identity")
        identity = feature[list(base.IDENTITY_COLUMNS)].merge(
            sidecar[list(base.IDENTITY_COLUMNS)], on="candidate_id", how="left",
            suffixes=("_feature", "_sidecar"), validate="one_to_one",
        )
        for field in base.IDENTITY_COLUMNS[1:]:
            left, right = identity[f"{field}_feature"], identity[f"{field}_sidecar"]
            if field == "__decision_ts__":
                left, right = pd.to_datetime(left, utc=True), pd.to_datetime(right, utc=True)
            if not left.eq(right).fillna(False).all():
                raise AssertionError(f"{fold}/{month:%Y-%m}: sidecar changed target-free {field}")
        merged = feature.merge(sidecar[["candidate_id", *sidecar_fields]], on="candidate_id", how="left", validate="one_to_one")
        if merged[list(sidecar_fields)].isna().all(axis=None):
            raise AssertionError(f"{fold}/{month:%Y-%m}: causal sidecar provided no required values")
        score_probe = _read_parquet_once(score_path)
        rank_column = next((name for name in ("router_primary_rank", "router_primary_only_rank", "router_full_ae_rank") if name in score_probe.columns), None)
        if rank_column is None:
            raise AssertionError(f"{fold}/{month:%Y-%m}: no accepted Router rank")
        score = score_probe.loc[score_probe["candidate_id"].isin(merged["candidate_id"]), ["candidate_id", rank_column]].rename(columns={rank_column: "router_rank"})
        merged = merged.merge(score, on="candidate_id", how="inner", validate="one_to_one")
        if len(merged) != len(feature):
            raise AssertionError(f"{fold}/{month:%Y-%m}: Router join lost a target-free candidate")
        held_part = merged.loc[merged["__decision_ts__"].ge(held_start) & merged["__decision_ts__"].lt(held_end)].copy()
        if not held_part.empty:
            held_target_free.append(held_part[list(base.IDENTITY_COLUMNS) + ["router_rank"]].copy())
        parts.append(merged)
        audit.append({
            "fold": fold, "month": str(month.date()), "feature_path": str(feature_path.relative_to(ROOT)),
            "sidecar_path": str(sidecar_path.relative_to(ROOT)), "score_path": str(score_path.relative_to(ROOT)),
            "rows_materialized": int(len(merged)), "rows_held_target_free": int(len(held_part)),
        })
        del feature, sidecar, score_probe, score, merged, held_part
        gc.collect()
    target_free = pd.concat(parts, ignore_index=True)
    if target_free["candidate_id"].duplicated().any():
        raise AssertionError(f"{fold}: duplicate materialised target-free candidate identity")
    held_before_labels = pd.concat(held_target_free, ignore_index=True)
    if held_before_labels["candidate_id"].duplicated().any():
        raise AssertionError(f"{fold}: duplicate held target-free candidate identity")
    panel = base._attach_outcomes(target_free, labels)
    train = panel.loc[panel["candidate_id"].astype(str).isin(train_ids)].copy()
    train = base._eligible_labels(train, held_start).reset_index(drop=True)
    held = panel.loc[panel["__decision_ts__"].ge(held_start) & panel["__decision_ts__"].lt(held_end)].reset_index(drop=True)
    if len(train) > cap:
        raise AssertionError(f"{fold}: materialized train cap exceeded")
    receipt = {
        "fold": fold, "prior_label_candidate_rows": int(len(prior)), "sampled_prior_label_rows": int(len(sampled_prior)),
        "materialized_train_rows": int(len(train)), "materialized_held_rows": int(len(held)),
        "training_cap": cap, "held_universe_complete_before_outcome_join": True,
        "training_sample_uses_prior_label_availability_only": True,
    }
    del labels, panel, target_free, parts, held_target_free
    gc.collect()
    return train, held, held_before_labels, audit, receipt


def _pooled_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selection_columns = [column for column in predictions.columns if column.endswith("_selected")]
    for column in selection_columns:
        arm = column.removesuffix("_selected")
        family = "capacity_rescue" if arm.endswith("_rescue") or arm == "router_100" else (
            "matched_generic_control" if "generic" in arm else "score_blend"
        )
        selected = predictions[column].to_numpy(dtype=bool)
        row = _valid_selection_metrics(predictions, selected, fold="pooled_2025_q2_q4", family=family, arm=arm)
        rows.append(row)
    return pd.DataFrame(rows)


def _format(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    if frame.empty:
        return "_No rows._\n"
    view = frame.loc[:, [column for column in columns if column in frame.columns]].copy()
    def cell(value: object) -> str:
        if isinstance(value, float):
            return "" if not np.isfinite(value) else f"{value:.5g}"
        return "" if pd.isna(value) else str(value).replace("|", "\\|")
    return "\n".join([
        "| " + " | ".join(view.columns) + " |",
        "| " + " | ".join("---" for _ in view.columns) + " |",
        *["| " + " | ".join(cell(value) for value in row) + " |" for row in view.itertuples(index=False, name=None)],
    ]) + "\n"


def _report(output: Path, config_path: Path, source: pd.DataFrame, metrics: pd.DataFrame, pooled: pd.DataFrame) -> None:
    primary = pooled.sort_values(["recall_gt_50", "recall_gt_100", "recall_gt_200"], ascending=False, kind="stable")
    report = [
        "# Broad P8U Rescue for Router Top-50% Recall\n",
        "## Scope\n",
        "This offline, long-only, strict-OOF study asks whether a broadly trained P8U specialist stack can increase winner recall inside the Router's fixed timestamp-local 50% capacity. P8U is scored across the entire target-free universe; it can therefore rescue rows outside Router top-50%. Each fold reuses an immutable frozen NMF geometry and selected head contracts. Geometry, HPO, Router, Base, Meta, MC1, admission, policy, portfolio, and live artifacts are untouched.\n",
        "## Causal receipt\n",
        _format(source, ["fold", "held_start", "held_end", "train_prior_resolved_rows", "held_rows", "held_valid_rows", "category_algorithm", "category_k", "geometry_refit", "head_contract_retuned", "p8u_scored_full_target_free_universe"]),
        "## Pooled full-universe Router-capacity results\n",
        "Every arm retains exactly 50% at each timestamp. `recall_gt_50/100/200` are the primary requested measures; realised policy net economics are guardrails rather than a held-period tuning criterion.\n",
        _format(primary, ["family", "arm", "router_share", "p8u_weight", "selected_rows_valid", "recall_gt_50", "recall_gt_100", "recall_gt_200", "selected_hit_gt_50", "selected_mean_net_bps", "selected_cvar10_net_bps", "positive_economic_mass_recall"]),
        "## Strict-OOF results by quarter\n",
        _format(metrics, ["fold", "family", "arm", "router_share", "p8u_weight", "selected_rows_valid", "recall_gt_50", "recall_gt_100", "recall_gt_200", "selected_mean_net_bps", "selected_cvar10_net_bps"]),
        "## Interpretation\n",
        "A capacity rescue improves the Router only if it raises all or most Recall@50/100/200 measures without material economic deterioration across multiple held quarters. A score blend is a separate candidate mechanism: it reorders all rows inside the same 50% capacity, rather than reserving explicit P8U seats. These held metrics are comparative research evidence only; no arm is selected or promoted by this run.\n",
        f"Config: `{config_path.relative_to(ROOT)}`\n",
    ]
    (output / "P8U_ROUTER50_BROAD_RESCUE_RECALL_REPORT.md").write_text("\n".join(report))


def run(config_path: Path, output: Path) -> None:
    config_path = config_path.resolve()
    config = base._load_config(config_path)
    if str(config.get("schema_version")) != SCHEMA:
        raise AssertionError("unexpected broad-rescue schema")
    if output.exists():
        raise FileExistsError(f"immutable output already exists: {output}")
    if not np.isclose(float(config["router"]["retained_fraction"]), .50):
        raise AssertionError("this predeclared experiment preserves exact Router top-50% capacity")

    folds = list(config["outer_folds"])
    predictive_fields = base._assert_causal_feature_contract(base._P8U_FEATURE_CONTRACTS[config["feature_contract_keys"]["predictive"]])
    sidecar_fields = base._assert_causal_feature_contract(base._P8U_FEATURE_CONTRACTS[config["probe_feature_sidecar_fields_key"]])
    output.mkdir(parents=True, exist_ok=False)
    results: list[dict[str, Any]] = []
    source_audit: list[dict[str, Any]] = []
    materialization_receipts: list[dict[str, Any]] = []
    for index, definition in enumerate(folds):
        train, held, held_target_free, fold_audit, receipt = _materialize_fold_panel(
            config, definition=definition, predictive_fields=predictive_fields, sidecar_fields=sidecar_fields,
            seed=int(config["seed"]) + index * 100_000,
        )
        # Seal complete held candidates before their policy outcomes are ever
        # joined into a score evaluation.  Training rows are intentionally not
        # persisted as a wide panel: their role is the fixed-cap supervised fit.
        _write_parquet_exclusive(held_target_free, output / "target_free_held_universe" / f"fold={definition['name']}.parquet")
        result = _fold(train, held, config, definition, offset=index)
        result["source"].update(receipt)
        results.append(result)
        source_audit.extend(fold_audit)
        materialization_receipts.append(receipt)
        del train, held, held_target_free, result
        gc.collect()
    metrics = pd.concat([item["metrics"] for item in results], ignore_index=True)
    predictions = pd.concat([item["predictions"] for item in results], ignore_index=True)
    source = pd.DataFrame([item["source"] for item in results])
    pooled = _pooled_metrics(predictions)
    _write_parquet_exclusive(metrics, output / "fold_metrics.parquet")
    _write_parquet_exclusive(pooled, output / "pooled_metrics.parquet")
    _write_parquet_exclusive(predictions, output / "candidate_predictions.parquet")
    _write_parquet_exclusive(source, output / "fold_source_audit.parquet")
    correctness = {
        "schema": SCHEMA,
        "long_only": True,
        "target_free_universe_written_before_outcome_join": True,
        "full_universe_p8u_score": True,
        "training_labels_strictly_prior_resolved": True,
        "held_outcomes_used_for_router_or_score": False,
        "geometry_refit_per_fold": False,
        "head_contract_retuned": False,
        "all_arms_exact_timestamp_local_router50_capacity": True,
        "canonical_or_live_contract_changed": False,
    }
    _write_json_exclusive(output / "correctness_report.json", correctness)
    del results
    gc.collect()
    manifest = {
        "schema": SCHEMA,
        "config_path": str(config_path.relative_to(ROOT)), "config_sha256": _sha256(config_path),
        "policy_label_path": str(config["policy_label_path"]), "policy_label_sha256": _sha256(ROOT / config["policy_label_path"]),
        "source_audit": source_audit, "materialization_receipts": materialization_receipts, "folds": source.to_dict(orient="records"),
        "artifacts": {"fold_metrics": "fold_metrics.parquet", "pooled_metrics": "pooled_metrics.parquet", "candidate_predictions": "candidate_predictions.parquet", "target_free_held_universe": "target_free_held_universe/"},
        "decision": "OFFLINE_RESEARCH_ONLY",
    }
    _write_json_exclusive(output / "run_manifest.json", manifest)
    _report(output, config_path, source, metrics, pooled)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.config, args.output)


if __name__ == "__main__":
    main()
