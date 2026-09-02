#!/usr/bin/env python3
"""Refit and seal the selected P8U Router/F72 Base/Under inference models.

Historical P8U results are score ledgers, so they cannot be reconstructed bit
for bit into their discarded fold boosters.  This producer instead performs a
single, explicit final refit at a named UTC cutoff:

* Router: current 3-month, 28-day-reserve strict-resolved full universe;
* Base: current 3-month, 28-day-reserve Router-OOF selected population;
* Under: current 4-month, 28-day-reserve *Base-OOF* population.

That OOF separation is intentional: the Under never trains on in-sample final
Base scores.  No label is joined to the parity sample.  The output is a
model-only package; it does not package MC1, policy, portfolio, or exchange
authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_strict_r3_economic_recall_router as router  # noqa: E402
import run_strict_r3_p8u_meta_target_query_grid_v1 as meta  # noqa: E402
import run_strict_r3_p8u_precision_preservation_screen_v1 as base_stage  # noqa: E402
import run_strict_r3_p8u_precision_preservation_weight_funnel_v1 as base_weights  # noqa: E402
import run_strict_r3_p8u_precision_preservation_winner_hpo_v1 as base_hpo  # noqa: E402
import run_strict_r3_router_single_base_prescreen_v1 as base_data  # noqa: E402
import select_strict_r3_p8u_meta_fullfeatures_v1 as under_select  # noqa: E402
from extreme_price_movements.inference.p8u_model_package import (  # noqa: E402
    BASE_GEOMETRY,
    MODEL_ROLES,
    P8UModelBundle,
    SCHEMA,
    _ModelState,
    role_file_entry,
    sha256_file,
)


SEED = 1729
IDENTITY = ("candidate_id", "__decision_ts__", "side_name")
ROUTER_CONTRACT = ROOT / "data_perp/artifacts/strict_r3_p8u_router_oof_apr25_jul26_successorlabels_20260828_v1/run_contract.json"
BASE_SELECTION = ROOT / "data_perp/artifacts/strict_r3_b0_family_addback_20260826_v1_policy_ordinal_base_g3/selection.json"
BASE_HPO = ROOT / "data_perp/artifacts/strict_r3_p8u_precision_preservation_hpo_raw_cat_20260827_v2/run_manifest.json"
UNDER_CONTRACT = ROOT / "data_perp/artifacts/strict_r3_p8u_meta_under_fullfeatures_selection_20260828_v2/contracts/under_f120.json"
UNDER_SOURCE_CONFIG = ROOT / "config/strict_r3_p8u_meta_target_query_grid_20260828_v1.json"
CANONICAL_CONTRACT = ROOT / "config/strict_r3_p8u_routed_f72_underf120_research_canonical_20260828_v6.json"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _utc(value: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _hash_array(values: np.ndarray) -> str:
    work = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(work.dtype).encode())
    digest.update(str(tuple(work.shape)).encode())
    digest.update(work.tobytes())
    return digest.hexdigest()


def _hash_identity(frame: pd.DataFrame) -> str:
    work = frame.loc[:, list(IDENTITY)].copy()
    work["__decision_ts__"] = pd.to_datetime(work["__decision_ts__"], utc=True, errors="raise")
    work = work.sort_values(["__decision_ts__", "candidate_id", "side_name"], kind="stable")
    digest = hashlib.sha256()
    for row in work.itertuples(index=False, name=None):
        digest.update("|".join(map(str, row)).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _safe_hash(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


def _source_receipts(paths: Mapping[str, Path]) -> dict[str, dict[str, str | None]]:
    return {
        name: {"path": str(path.relative_to(ROOT)), "sha256": _safe_hash(path)}
        for name, path in paths.items()
    }


def _mkdir_exclusive(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"immutable output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def _router_final_fit(*, cutoff: pd.Timestamp) -> tuple[Any, tuple[str, ...], np.ndarray, np.ndarray, dict[str, Any]]:
    contract = _json(ROUTER_CONTRACT)
    fields = tuple(str(value) for value in contract["feature_contract"])
    ranker = dict(contract["ranker"])
    reserve_days, train_months, train_cap = int(contract["reserve_days"]), int(contract["train_months"]), int(contract["train_cap"])
    reserve = cutoff - pd.Timedelta(days=reserve_days)
    start = reserve - pd.DateOffset(months=train_months)
    feature_roots = tuple(ROOT / str(value) for value in contract["feature_roots"])
    policy_path = ROOT / str(contract["policy_path"])
    target_free = router._window_features(feature_roots, start, reserve, (*IDENTITY, *fields))
    policy = router._policy_window(policy_path, start, reserve)
    capped = router._deterministic_query_cap(target_free, cap=train_cap)
    train = router._prepare_train(capped, None, policy, reserve)
    targets = router._targets(train, str(contract["primary_target"]), include_auxiliary=False)
    name, inverse, labels = targets["main"]
    fitted = router._fit_target(
        train, fields, str(name), "main", bool(inverse), np.asarray(labels, dtype=np.int32), SEED + 1009, ranker,
    )
    matrix = router._matrix(train, fields, fitted.medians)
    audit = {
        "training_start": str(start), "reserve_cutoff": str(reserve), "rows_before_cap": int(len(target_free)),
        "rows_capped": int(len(capped)), "rows_resolved": int(len(train)), "queries": int(train["__decision_ts__"].nunique()),
        "target": str(name), "target_classes": int(fitted.classes), "feature_matrix_sha256": _hash_array(matrix),
        "labels_sha256": _hash_array(np.asarray(labels, dtype=np.int32)), "identity_sha256": _hash_identity(train),
        "weight_summary": fitted.weight_summary,
    }
    return fitted.model, fields, fitted.medians, fitted.reference.sorted_values, audit


def _base_final_fit(*, cutoff: pd.Timestamp) -> tuple[Any, tuple[str, ...], np.ndarray, dict[str, Any]]:
    hpo_manifest = _json(BASE_HPO)
    params = dict(hpo_manifest["hpo"]["winner"])
    fields = base_data._load_f72_fields(BASE_SELECTION)
    reserve_days = int(hpo_manifest["strict_oof"]["reserve_days"])
    train_months = int(hpo_manifest["strict_oof"]["train_months"])
    train_cap = int(hpo_manifest["strict_oof"]["train_cap_complete_queries"])
    reserve = cutoff - pd.Timedelta(days=reserve_days)
    start = reserve - pd.DateOffset(months=train_months)
    feature_roots = (
        ROOT / "data_perp/artifacts/strict_r3_incumbent_meta_causal_features_20260827_preaug_v1",
        ROOT / "data_perp/artifacts/strict_r3_incumbent_meta_causal_features_20260827_v1",
    )
    label_root = ROOT / "data_perp/artifacts/strict_r3_router50_single_base_target_labels_20260827_v1"
    router_root = ROOT / "data_perp/artifacts/strict_r3_p8u_router_oof_apr25_jul26_successorlabels_20260828_v1"
    window, coverage = base_data._load_window(
        candidate_root=None, feature_root=feature_roots, label_root=label_root, router_root=router_root,
        start=start, end=reserve, fields=fields,
    )
    arm = base_stage.Arm("raw_bps", "t1_raw_bps", "equal_width6")
    train = base_stage._train_rows(window, arm, reserve, train_cap)
    labels, geometry = base_stage._labels(train, arm)
    x_train, medians = base_data._numeric_matrix(train, fields)
    fit, valid = base_hpo._inner_masks(train)
    weights = base_weights._query_safe_weights(train, labels, "tail_linear_125")
    fit_frame = train.loc[fit].reset_index(drop=True)
    valid_frame = train.loc[valid].reset_index(drop=True)
    model = CatBoostRanker(
        loss_function="QueryRMSE", eval_metric="NDCG:top=10", iterations=2000,
        learning_rate=float(params["learning_rate"]), depth=int(params["max_depth"]),
        l2_leaf_reg=float(params["lambda_l2"]), random_strength=float(params["random_strength"]),
        rsm=float(params["feature_fraction"]), bootstrap_type="Bernoulli", subsample=float(params["bagging_fraction"]),
        random_seed=SEED, thread_count=1, verbose=False, allow_writing_files=False,
        od_type="Iter", od_wait=30,
    )
    model.fit(
        Pool(x_train[fit], labels[fit], group_id=base_hpo._qid(fit_frame), weight=weights[fit]),
        eval_set=Pool(x_train[valid], labels[valid], group_id=base_hpo._qid(valid_frame), weight=weights[valid]),
        use_best_model=True, verbose=False,
    )
    audit = {
        "training_start": str(start), "reserve_cutoff": str(reserve), "window_rows": int(len(window)),
        "rows_resolved": int(len(train)), "queries": int(train["__decision_ts__"].nunique()), "feature_count": len(fields),
        "target_geometry": geometry, "target_sha256": _hash_array(labels), "feature_matrix_sha256": _hash_array(x_train),
        "identity_sha256": _hash_identity(train), "weight_sha256": _hash_array(weights),
        "inner_fit_rows": int(fit.sum()), "inner_validation_rows": int(valid.sum()), "best_tree_count": int(model.tree_count_),
        "router_gate_source": str(router_root.relative_to(ROOT)), "coverage": coverage,
    }
    return model, fields, medians.to_numpy(np.float32), audit


def _under_final_fit(*, cutoff: pd.Timestamp) -> tuple[Any, tuple[str, ...], np.ndarray, dict[str, Any]]:
    config = _json(UNDER_SOURCE_CONFIG)
    spec = meta.Spec(raw=config, config_path=UNDER_SOURCE_CONFIG)
    arm = next(item for item in meta._arm_specs(config, None) if item.name == "under_bps100__timestamp")
    if arm.family != "under" or arm.query != "timestamp" or arm.scale != "bps" or float(arm.threshold) != 100.0:
        raise AssertionError("sealed Under F120 target contract has changed")
    fields = tuple(str(value) for value in _json(UNDER_CONTRACT)["selected_features"])
    reserve_days = int(config["folds"]["resolved_label_reserve_days"])
    train_months = int(config["folds"]["train_months"])
    max_train_rows = int(config["folds"]["max_train_rows"])
    max_query_rows = int(config["folds"]["max_query_rows"])
    anchor_days = int(config["folds"]["anchor_block_days"])
    reserve = cutoff - pd.Timedelta(days=reserve_days)
    start = reserve - pd.DateOffset(months=train_months)
    base_root = ROOT / str(config["source"]["base_target_free_root"])
    feature_roots = tuple(ROOT / str(value) for value in config["source"]["full_feature_roots"])
    policy = meta._read_policy(ROOT / str(config["source"]["policy_labels"]))
    path_root = ROOT / str(config["source"]["path_labels"])
    target_free = meta._read_base_features(base_root=base_root, feature_roots=feature_roots, start=start, end=reserve, fields=fields)
    labelled = meta._labelled(target_free, policy, path_root, start, reserve)
    valid_resolution = meta._valid_label(labelled, reserve)
    strict = labelled.loc[valid_resolution].copy().reset_index(drop=True)
    anchors = meta._prequential_anchor(strict, block_days=anchor_days)
    labels, residual, target_info = meta._train_target(strict, arm, anchor=anchors)
    usable = labels >= 0
    sampled = meta._sample_queries(strict.loc[usable].copy(), max_train_rows, SEED)
    lookup = pd.DataFrame({"candidate_id": strict.candidate_id.astype(str), "__target__": labels})
    sampled = sampled.merge(lookup, on="candidate_id", how="left", validate="one_to_one")
    if sampled["__target__"].isna().any():
        raise AssertionError("Under sampled frame lost strict-prequential target")
    order, _query, groups = meta._bounded_queries(sampled, meta._query_ids(sampled, arm.query), max_query_rows)
    sampled = sampled.iloc[order].reset_index(drop=True)
    y = sampled["__target__"].to_numpy(np.int32)
    if len(sampled) < 20_000 or len(np.unique(y)) < 2:
        raise AssertionError("insufficient selected Under support")
    raw_matrix = meta._matrix(sampled, fields)
    medians = np.nanmedian(raw_matrix, axis=0).astype(np.float32)
    medians[~np.isfinite(medians)] = 0.0
    x = raw_matrix.copy()
    missing = ~np.isfinite(x)
    if missing.any():
        x[missing] = np.broadcast_to(medians, x.shape)[missing]
    model = under_select._model(classes=int(np.max(y)) + 1, seed=SEED, n_jobs=min(4, os.cpu_count() or 1))
    model.fit(x, y, group=groups)
    audit = {
        "training_start": str(start), "reserve_cutoff": str(reserve), "target_free_rows": int(len(target_free)),
        "strict_resolved_rows": int(len(strict)), "prequential_anchor_rows": int(np.isfinite(anchors).sum()),
        "sampled_rows": int(len(sampled)), "queries": int(len(groups)), "classes": int(np.max(y) + 1),
        "target_info": target_info, "feature_matrix_sha256": _hash_array(x), "target_sha256": _hash_array(y),
        "identity_sha256": _hash_identity(sampled), "residual_sha256": _hash_array(np.asarray(residual, dtype=np.float32)),
        "base_score_provenance": "strict-OOF F72 Base target-free ledger; never final in-sample Base predictions",
        "base_ledger": str(base_root.relative_to(ROOT)),
    }
    return model, fields, medians, audit


def _sample_full_features(*, fields: Sequence[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    roots = (
        ROOT / "data_perp/artifacts/strict_r3_incumbent_meta_causal_features_20260827_preaug_v1",
        ROOT / "data_perp/artifacts/strict_r3_incumbent_meta_causal_features_20260827_v1",
    )
    result = router._window_features(roots, start, end, (*IDENTITY, *fields))
    if result.empty:
        raise AssertionError("no target-free holdout sample available for package parity")
    if result.duplicated(IDENTITY).any() or not result.side_name.eq("long").all():
        raise AssertionError("invalid target-free package parity sample")
    return result.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def _assert_parity(*, raw_bundle: P8UModelBundle, packaged: P8UModelBundle, sample: pd.DataFrame) -> dict[str, Any]:
    before = raw_bundle.score_stack(sample)
    after = packaged.score_stack(sample)
    detail: dict[str, Any] = {"rows_full": int(len(sample)), "router50_rows": int(len(before[1]))}
    for name, left, right, columns in (
        ("router", before[0], after[0], ("router_raw_score", "router_primary_rank")),
        ("base", before[1], after[1], ("base_score", "base_rank_ts")),
        ("under", before[2], after[2], ("under_raw_score", "under_rank_ts")),
    ):
        merged = left.merge(right, on=list(IDENTITY), how="outer", suffixes=("_raw", "_loaded"), indicator=True, validate="one_to_one")
        if not merged["_merge"].eq("both").all():
            raise AssertionError(f"{name}: model reload changed score identities")
        maxima: dict[str, float] = {}
        for column in columns:
            delta = np.abs(merged[f"{column}_raw"].to_numpy(float) - merged[f"{column}_loaded"].to_numpy(float))
            maxima[column] = float(np.nanmax(delta)) if len(delta) else 0.0
            if not np.allclose(merged[f"{column}_raw"], merged[f"{column}_loaded"], rtol=0.0, atol=1e-6, equal_nan=False):
                raise AssertionError(f"{name}: reload prediction delta exceeds 1e-6 for {column}")
        detail[name] = {"rows": int(len(merged)), "max_abs_delta": maxima}
    return detail


def run(*, cutoff: pd.Timestamp, out: Path, parity_start: pd.Timestamp, parity_end: pd.Timestamp) -> Path:
    if cutoff.tzinfo is None or cutoff.tz_convert("UTC") != cutoff:
        raise ValueError("cutoff must be UTC")
    _mkdir_exclusive(out)
    temporary = out.with_name(f".{out.name}.build-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.mkdir()
    try:
        router_model, router_fields, router_medians, router_reference, router_audit = _router_final_fit(cutoff=cutoff)
        base_model, base_fields, base_medians, base_audit = _base_final_fit(cutoff=cutoff)
        under_model, under_fields, under_medians, under_audit = _under_final_fit(cutoff=cutoff)
        model_specs = {
            "router_model": (router_model, "lightgbm_booster_text", router_fields, router_medians, router_reference, ("router_raw_score", "router_primary_rank")),
            "base_model": (base_model, "catboost_ranker_cbm", base_fields, base_medians, None, ("base_score", "base_rank_ts")),
            "under_model": (under_model, "lightgbm_booster_text", tuple((*BASE_GEOMETRY, *under_fields)), under_medians, None, ("under_raw_score", "under_rank_ts")),
        }
        states: dict[str, _ModelState] = {}
        models: dict[str, Any] = {}
        role_entries: dict[str, dict[str, object]] = {}
        for role, (model, fmt, fields, medians, reference, outputs) in model_specs.items():
            role_root = temporary / "models" / role
            role_root.mkdir(parents=True)
            model_path = role_root / ("model.cbm" if fmt == "catboost_ranker_cbm" else "model.txt")
            state_path = role_root / "state.npz"
            if fmt == "catboost_ranker_cbm":
                model.save_model(str(model_path))
            else:
                model.booster_.save_model(str(model_path))
            payload = {"medians": np.asarray(medians, dtype=np.float32)}
            if reference is not None:
                payload["rank_reference"] = np.asarray(reference, dtype=np.float32)
            np.savez_compressed(state_path, **payload)
            states[role] = _ModelState(role, tuple(fields), np.asarray(medians, dtype=np.float32), None if reference is None else np.asarray(reference, dtype=np.float32), model_path, fmt)
            models[role] = model
            role_entries[role] = role_file_entry(temporary, role=role, model_path=model_path, state_path=state_path, model_format=fmt, feature_order=fields, output_fields=outputs)
        manifest = {
            "schema": SCHEMA,
            "package_kind": "P8U Router50 + F72 Base + Under F120 model-only final refit",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "training_cutoff": cutoff.isoformat(),
            "side": "long",
            "routing": {"fraction": .50, "scope": "complete point-in-time candidate universe", "tie_break": "candidate_id ascending"},
            "models": role_entries,
            "inference_flow": ["router_model", "exact timestamp Router50", "base_model", "under_model"],
            "source_receipts": _source_receipts({
                "canonical_contract": CANONICAL_CONTRACT, "router_contract": ROUTER_CONTRACT,
                "base_selection": BASE_SELECTION, "base_hpo": BASE_HPO, "under_contract": UNDER_CONTRACT,
                "under_source_config": UNDER_SOURCE_CONFIG,
            }),
            "training": {"router_model": router_audit, "base_model": base_audit, "under_model": under_audit},
            "known_boundaries": [
                "Historical fold boosters were not persisted; this is a transparent final refit from selected frozen contracts.",
                "Under training consumes strict-OOF Base scores; final live Under scoring consumes final Base scores only after Router50.",
                "This package does not include MC1, policy, portfolio, execution, or exchange authority.",
            ],
        }
        _write_json(temporary / "manifest.json", manifest)
        raw_bundle = P8UModelBundle(root=temporary, manifest=manifest, states=states, models=models)
        packaged = P8UModelBundle.load(temporary, verify_hashes=True)
        union = tuple(dict.fromkeys((*router_fields, *base_fields, *under_fields)))
        sample = _sample_full_features(fields=union, start=parity_start, end=parity_end)
        parity = _assert_parity(raw_bundle=raw_bundle, packaged=packaged, sample=sample)
        _write_json(temporary / "parity_report.json", {
            "status": "pass", "target_free_sample": {"start": str(parity_start), "end": str(parity_end), "feature_union_count": len(union)},
            "load_predict_parity": parity, "router50_only_downstream": True,
        })
        # The intended immutable output root was reserved above, so replace
        # the empty directory only after every model and parity check exists.
        out.rmdir()
        os.replace(temporary, out)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        if out.exists() and not any(out.iterdir()):
            out.rmdir()
        raise
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", default="2026-08-28T00:00:00Z")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--parity-start", default="2026-07-30T00:00:00Z")
    parser.add_argument("--parity-end", default="2026-07-31T00:00:00Z")
    args = parser.parse_args()
    output = args.out.resolve()
    if ROOT not in output.parents:
        raise ValueError("output must remain inside the repository")
    print(run(cutoff=_utc(args.cutoff), out=output, parity_start=_utc(args.parity_start), parity_end=_utc(args.parity_end)))


if __name__ == "__main__":
    main()
