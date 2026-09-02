#!/usr/bin/env python3
"""Cross-model screen for one frozen strict-R3 P8u Meta contract.

This is deliberately a *model-family* comparison, not another target or
feature-selection stage.  It reuses the exact selected Meta target/query,
feature contract, training-window construction, and external BaseStableMeta
diagnostics.  Every held score is persisted without outcomes before policy
labels are used for diagnostics.

The script has no MC1, admission, portfolio, live, or exchange side effect.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool
from lightgbm import LGBMRanker

import run_strict_r3_p8u_meta_lgbm_objective_screen_v1 as objective
import run_strict_r3_p8u_meta_target_query_grid_v1 as screen


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "strict_r3_p8u_meta_crossmodel_v1"
IDENTITY = screen.IDENTITY
SEED = 1729
MODEL_FAMILIES = (
    "lgbm_xendcg",
    "catboost_queryrmse",
    "catboost_yetirank",
)


def _trial_receipts(
    *, arm: screen.Arm, feature_contract: Path, feature_count: int,
    sample_weight_profile: Mapping[str, Any] | None, label_gain: Sequence[float],
    model_candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Expose model-family candidates through the common descriptor contract.

    The downstream GateProxy descriptor builder intentionally consumes only
    immutable target-free scores plus a compact trial receipt.  Keeping that
    receipt structurally identical to the Rank-XENDCG screen means that the
    CatBoost comparison cannot receive a bespoke or weaker downstream proxy.
    This is lineage metadata only; it is not read by fitting or scoring.
    """
    loss = {
        "lgbm_xendcg": "rank_xendcg",
        "catboost_queryrmse": "QueryRMSE",
        "catboost_yetirank": "YetiRank:mode=NDCG;top=12",
    }
    return [
        {
            "name": str(candidate["name"]),
            "target": arm.name,
            "arm_name": arm.name,
            "parent_contract": str(feature_contract),
            "additive_feature_family": "selected_contract",
            "feature_mode": "frozen_selected",
            "sample_weight": dict(sample_weight_profile) if sample_weight_profile is not None else None,
            "model": {
                "objective": loss[str(candidate["model_family"])],
                "common_budget": not bool(candidate.get("params")),
                "hpo_params": dict(candidate.get("params", {})),
            },
            "gain": [float(value) for value in label_gain],
            "truncation": 12 if str(candidate["model_family"]) == "catboost_yetirank" else None,
            "sigmoid": None,
        }
        for candidate in model_candidates
    ]


def _model_candidates(
    plan_path: Path | None, selected_trial_plan: Path | None = None,
    only_names: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Return immutable common-budget candidates or a sealed HPO bank.

    A candidate bank is intentionally a *parameter proposal* only.  It cannot
    alter target, query, feature, sample-weight, or label-gain lineage; those
    come solely from the enclosing frozen Meta contract.  Its target-free
    scores subsequently pass through the same descriptor/GateProxy path as a
    model-family screen.
    """
    if plan_path is not None and selected_trial_plan is not None:
        raise AssertionError("choose either a generic model HPO bank or a frozen selected-trial plan")
    if selected_trial_plan is not None:
        # A strict-MC1 confirmation needs score coverage before its first held
        # month.  Reconstruct the exact already-shortlisted HPO candidates from
        # the GateProxy receipt rather than copying parameters into a new,
        # potentially divergent configuration file.
        raw = json.loads(selected_trial_plan.read_text())
        if not isinstance(raw, list) or not raw:
            raise AssertionError("selected-trial plan must be a non-empty list")
        objective_family = {
            "rank_xendcg": "lgbm_xendcg",
            "QueryRMSE": "catboost_queryrmse",
            "YetiRank:mode=NDCG;top=12": "catboost_yetirank",
        }
        candidates: list[dict[str, Any]] = []
        names: set[str] = set()
        for record in raw:
            if not isinstance(record, Mapping):
                raise AssertionError("selected-trial record must be an object")
            name = str(record.get("trial", ""))
            trial_config = record.get("trial_config")
            if not name or name in names or not isinstance(trial_config, Mapping):
                raise AssertionError("invalid or duplicate selected-trial record")
            model = trial_config.get("model")
            if not isinstance(model, Mapping):
                raise AssertionError(f"{name}: selected-trial model receipt missing")
            family = objective_family.get(str(model.get("objective", "")))
            params = model.get("hpo_params", {})
            if family not in MODEL_FAMILIES or not isinstance(params, Mapping):
                raise AssertionError(f"{name}: unsupported selected-trial model receipt")
            names.add(name)
            candidates.append({"name": name, "model_family": family, "params": dict(params)})
        return _restrict_candidates(candidates, only_names)
    if plan_path is None:
        return _restrict_candidates(
            [{"name": family, "model_family": family, "params": {}} for family in MODEL_FAMILIES], only_names,
        )
    payload = json.loads(plan_path.read_text())
    raw = payload.get("candidates") if isinstance(payload, Mapping) else payload
    if not isinstance(raw, list) or not raw:
        raise AssertionError("model HPO plan must contain a non-empty candidates list")
    result: list[dict[str, Any]] = []
    names: set[str] = set()
    for item in raw:
        if not isinstance(item, Mapping):
            raise AssertionError("model HPO candidate must be an object")
        name, family = str(item.get("name", "")), str(item.get("model_family", ""))
        params = item.get("params", {})
        if not name or name in names or family not in MODEL_FAMILIES or not isinstance(params, Mapping):
            raise AssertionError("invalid or duplicate model HPO candidate")
        names.add(name)
        result.append({"name": name, "model_family": family, "params": dict(params)})
    return _restrict_candidates(result, only_names)


def _restrict_candidates(
    candidates: list[dict[str, Any]], only_names: Sequence[str] | None,
) -> list[dict[str, Any]]:
    """Restrict a sealed candidate list without rewriting its parameters."""
    if not only_names:
        return candidates
    wanted = tuple(str(name) for name in only_names)
    if len(wanted) != len(set(wanted)):
        raise AssertionError("duplicate --only-candidate name")
    available = {str(candidate["name"]) for candidate in candidates}
    unknown = sorted(set(wanted).difference(available))
    if unknown:
        raise AssertionError(f"requested candidate is absent from sealed plan: {unknown}")
    result = [candidate for candidate in candidates if str(candidate["name"]) in set(wanted)]
    if not result:
        raise AssertionError("candidate filter was empty")
    return result


def _validate_weight_compatibility(
    *, sample_weight_profile: Mapping[str, Any] | None,
    model_candidates: Sequence[Mapping[str, Any]],
) -> None:
    """Reject a misleading weighted YetiRank comparison before any fitting."""
    if sample_weight_profile is not None and any(
        str(candidate["model_family"]) == "catboost_yetirank"
        for candidate in model_candidates
    ):
        raise AssertionError(
            "CatBoost YetiRank does not support the selected object-level "
            "sample-weight profile; compare it only in an explicitly "
            "unweighted matched family screen"
        )
def _apply_source_override(raw: dict[str, Any], override_path: Path | None) -> dict[str, Any]:
    """Replace only source-lineage roots while preserving the frozen arms.

    This is for a received target-free bridge whose Base and feature identities
    have already been audited.  Target/query/fold/weight definitions are never
    accepted from the override, which prevents a historical continuation from
    silently changing the experiment being confirmed.
    """
    if override_path is None:
        return raw
    payload = json.loads(override_path.read_text())
    source = payload.get("source", payload) if isinstance(payload, Mapping) else None
    if not isinstance(source, Mapping):
        raise AssertionError("source override must be a source mapping")
    allowed = {"base_target_free_root", "full_feature_roots", "base_f72_contract", "policy_labels", "path_labels"}
    unknown = sorted(set(source).difference(allowed))
    if unknown:
        raise AssertionError(f"source override has unsupported keys: {unknown}")
    required = {"base_target_free_root", "full_feature_roots", "policy_labels", "path_labels"}
    if not required.issubset(source) or not isinstance(source["full_feature_roots"], list):
        raise AssertionError("source override lacks the required Base, feature, policy, or path roots")
    result = dict(raw)
    result["source"] = {**dict(raw["source"]), **dict(source)}
    return result


def _once(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _progress(path: Path, payload: Mapping[str, Any]) -> None:
    with (path / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True, default=str) + "\n")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    members = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
    for member in members:
        digest.update(str(member).encode())
        with member.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _arms(raw: Mapping[str, Any]) -> dict[str, screen.Arm]:
    return {arm.name: arm for arm in screen._arm_specs(raw, None)}


def _qid(frame: pd.DataFrame) -> np.ndarray:
    # The family screen must preserve the selected target/query definition;
    # a Magnitude or State winner may legitimately use Base-band blocks rather
    # than one exact-timestamp query.  These IDs are fit-only bookkeeping.
    if "__rank_query_id__" not in frame:
        raise AssertionError("prepared family screen is missing frozen ranking-query IDs")
    codes, _ = pd.factorize(frame["__rank_query_id__"], sort=True)
    if np.any(codes < 0):
        raise AssertionError("invalid exact-timestamp query IDs")
    return codes.astype(np.int64)


def _fit_predict(
    *, family: str, train_x: np.ndarray, labels: np.ndarray, group: Sequence[int],
    qid: np.ndarray, held_x: np.ndarray, sample_weight: np.ndarray | None,
    label_gain: Sequence[float], seed: int, params: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Fit one cheap common-budget challenger on query-safe inputs only."""
    if family not in MODEL_FAMILIES:
        raise ValueError(f"unsupported model family {family!r}")
    if sum(int(value) for value in group) != len(labels) or min(group) < 2:
        raise AssertionError("invalid prepared ranking groups")
    if len(qid) != len(labels):
        raise AssertionError("invalid prepared CatBoost query IDs")
    requested = dict(params or {})
    common = {
        "n_estimators": 220,
        "learning_rate": 0.05,
        "max_depth": 4,
        "num_leaves": 15,
        "min_child_samples": 350,
        "min_split_gain": 0.001,
        "colsample_bytree": 0.80,
        "subsample": 0.82,
        "subsample_freq": 1,
        "reg_alpha": 0.02,
        "reg_lambda": 8.0,
        "random_state": int(seed),
        "n_jobs": 1,
        "verbosity": -1,
    }
    if family == "lgbm_xendcg":
        allowed = set(common)
        unknown = sorted(set(requested).difference(allowed))
        if unknown:
            raise AssertionError(f"LGBM HPO candidate has unsupported parameters: {unknown}")
        common.update(requested)
        model = LGBMRanker(
            objective="rank_xendcg",
            metric="ndcg",
            label_gain=[float(value) for value in label_gain],
            **common,
        )
        model.fit(train_x, labels, group=list(group), sample_weight=sample_weight)
        output = model.predict(held_x)
    elif family == "catboost_queryrmse":
        common_cat: dict[str, Any] = {
            "iterations": 220, "learning_rate": 0.05, "depth": 4,
            "l2_leaf_reg": 8.0, "random_strength": 0.5, "rsm": 0.80,
            "random_seed": int(seed), "thread_count": 1,
            "verbose": False, "allow_writing_files": False,
        }
        unknown = sorted(set(requested).difference(common_cat))
        if unknown:
            raise AssertionError(f"QueryRMSE HPO candidate has unsupported parameters: {unknown}")
        common_cat.update(requested)
        model = CatBoostRanker(
            loss_function="QueryRMSE",
            eval_metric="NDCG:top=10",
            **common_cat,
        )
        model.fit(Pool(train_x, labels, group_id=qid, weight=sample_weight))
        output = model.predict(held_x)
    elif family == "catboost_yetirank":
        common_cat = {
            "iterations": 220, "learning_rate": 0.05, "depth": 4,
            "l2_leaf_reg": 8.0, "random_strength": 0.5, "rsm": 0.80,
            "random_seed": int(seed), "thread_count": 1,
            "verbose": False, "allow_writing_files": False,
        }
        unknown = sorted(set(requested).difference(common_cat))
        if unknown:
            raise AssertionError(f"YetiRank HPO candidate has unsupported parameters: {unknown}")
        common_cat.update(requested)
        model = CatBoostRanker(
            loss_function="YetiRank:mode=NDCG;top=12",
            eval_metric="NDCG:top=10",
            **common_cat,
        )
        model.fit(Pool(train_x, labels, group_id=qid, weight=sample_weight))
        output = model.predict(held_x)
    values = np.asarray(output, dtype=np.float32)
    if not np.isfinite(values).all():
        raise AssertionError(f"{family}: non-finite held score")
    return values


def _score(
    *, prepared: objective.PreparedFold, arm: screen.Arm, family: str,
    sample_weight_profile: Mapping[str, Any] | None, label_gain: Sequence[float], seed: int,
    trial_name: str, model_params: Mapping[str, Any] | None,
) -> pd.DataFrame:
    sample_weight, weight_audit = objective._sample_weight(
        train=prepared.train_frame, labels=prepared.labels, profile=sample_weight_profile,
    )
    raw = _fit_predict(
        family=family,
        train_x=prepared.train_x,
        labels=prepared.labels,
        group=prepared.groups,
        qid=_qid(prepared.train_frame),
        held_x=prepared.held_x,
        sample_weight=sample_weight,
        label_gain=label_gain, seed=seed, params=model_params,
    )
    if arm.family == "over":
        raw *= -1.0
    score = prepared.held_target_free.loc[:, list(IDENTITY) + ["base_score", "base_rank_ts"]].copy().reset_index(drop=True)
    score["meta_raw_score"] = raw
    rank = score.loc[:, ["candidate_id", "__decision_ts__"]].copy()
    rank["value"] = raw
    score["meta_rank_ts"] = screen._rank_desc(rank, "value")
    score["arm"] = arm.name
    score["family"] = arm.family
    score["scale"] = arm.scale
    score["query_contract"] = arm.query
    score["trial"] = str(trial_name)
    score["held_month"] = f"{prepared.held_month:%Y-%m}"
    score["target_free"] = True
    score["fit_weight_profile"] = str(weight_audit["profile"])
    return score


def _weight_trial(path: Path, name: str) -> tuple[Mapping[str, Any] | None, tuple[float, ...], Mapping[str, Any]]:
    """Read one sealed Rank-XENDCG weight winner without re-specifying it."""
    payload = json.loads(path.read_text())
    trials = payload.get("trials") if isinstance(payload, Mapping) else payload
    if not isinstance(trials, list):
        raise AssertionError("weight receipt must contain a trial list")
    selected = [trial for trial in trials if isinstance(trial, Mapping) and str(trial.get("name")) == name]
    if len(selected) != 1:
        raise AssertionError("exactly one declared sample-weight winner is required")
    trial = selected[0]
    model = trial.get("model")
    gain = trial.get("gain")
    if not isinstance(model, Mapping) or str(model.get("objective")) != "rank_xendcg":
        raise AssertionError("cross-model screen requires a sealed Rank-XENDCG weight winner")
    if not isinstance(gain, list) or len(gain) < 2:
        raise AssertionError("weight winner lacks a valid label-gain schedule")
    profile = trial.get("sample_weight")
    if profile is not None and not isinstance(profile, Mapping):
        raise AssertionError("invalid sample-weight profile")
    return profile, tuple(float(value) for value in gain), dict(trial)


def _validate_final_lgbm_hpo(*, enabled: bool, model_candidates: Sequence[Mapping[str, Any]]) -> None:
    """Keep the final regular-HPO receipt semantically narrower than a family screen.

    The first screen is intentionally a small cross-family comparison.  Once
    a family has been rejected by strict MC1, the later bounded search is a
    real LightGBM depth/regularisation HPO.  Mixing CatBoost candidates back
    into that receipt would make the stage name misleading and would reopen a
    family decision already falsified by MC1.
    """
    if not enabled:
        return
    if len(model_candidates) < 4:
        raise AssertionError("final LightGBM HPO requires at least four predeclared candidates")
    families = {str(candidate.get("model_family")) for candidate in model_candidates}
    if families != {"lgbm_xendcg"}:
        raise AssertionError("final LightGBM HPO may contain only lgbm_xendcg candidates")


def run(
    *, config: Path, arm_name: str, feature_contract: Path, out: Path,
    weight_trials: Path, weight_trial_name: str,
    held_month_values: Sequence[str] | None = None, workers: int = 3,
    model_plan: Path | None = None, selected_trial_plan: Path | None = None,
    source_override: Path | None = None, only_candidate_names: Sequence[str] | None = None,
    final_lgbm_hpo: bool = False,
) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    raw = _apply_source_override(json.loads(config.read_text()), source_override)
    arm = _arms(raw).get(arm_name)
    if arm is None:
        raise ValueError(f"unknown arm {arm_name!r}")
    sample_weight_profile, label_gain, weight_trial = _weight_trial(weight_trials, weight_trial_name)
    fields, continuous_fields, declared_sidecar = objective._read_contract(feature_contract)
    model_candidates = _model_candidates(model_plan, selected_trial_plan, only_candidate_names)
    _validate_final_lgbm_hpo(enabled=bool(final_lgbm_hpo), model_candidates=model_candidates)
    # CatBoost's pairwise YetiRank objective ignores object-level weights.
    # Permitting it in a weighted cross-family run would make the resulting
    # model comparison non-matched while looking like one.  Require a
    # dedicated unweighted YetiRank control instead; QueryRMSE and
    # Rank-XENDCG may still share the selected per-row profile.
    _validate_weight_compatibility(
        sample_weight_profile=sample_weight_profile,
        model_candidates=model_candidates,
    )
    descriptor_trials = _trial_receipts(
        arm=arm, feature_contract=feature_contract, feature_count=len(fields),
        sample_weight_profile=sample_weight_profile, label_gain=label_gain,
        model_candidates=model_candidates,
    )
    panel_fields = tuple(field for field in fields if field not in set(continuous_fields))
    spec = screen.Spec(raw=raw, config_path=config)
    months = tuple(screen._utc_month(value) for value in (held_month_values or spec.folds["held_months"]))
    if len(months) < 3 or tuple(sorted(months)) != months:
        raise ValueError("need at least three strictly chronological held months")
    base_root = ROOT / str(spec.source["base_target_free_root"])
    feature_roots = tuple(ROOT / str(value) for value in spec.source["full_feature_roots"])
    policy_path = ROOT / str(spec.source["policy_labels"])
    path_root = ROOT / str(spec.source["path_labels"])
    policy = screen._read_policy(policy_path)
    out.mkdir(parents=True)
    preflight_months = tuple(screen._month_range(months[0] - pd.DateOffset(months=5), screen._month_end(months[-1])))
    coverage = screen._preflight(spec, panel_fields, preflight_months)
    coverage.to_parquet(out / "source_coverage_audit.parquet", index=False, compression="zstd")
    if continuous_fields:
        if declared_sidecar is None:
            raise AssertionError("continuous-regime feature contract is missing a sidecar")
        sidecar = objective._preflight_continuous_sidecar(
            base_root=base_root, path=declared_sidecar, fields=continuous_fields, months=preflight_months,
        )
        sidecar.to_parquet(out / "continuous_regime_sidecar_coverage_audit.parquet", index=False, compression="zstd")
    _once(out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": (
            "offline strict-OOF final LightGBM depth/regularisation HPO; no MC1/admission/portfolio/live/exchange mutation"
            if final_lgbm_hpo else
            "offline strict-OOF Meta cross-model screen; no MC1/admission/portfolio/live/exchange mutation"
        ),
        "base_contract": raw["base_contract"],
        "arm": dataclasses.asdict(arm),
        "feature_contract": str(feature_contract),
        "feature_contract_sha256": _sha(feature_contract),
        "feature_count": len(fields),
        # These aliases make the output consumable by the shared strict-OOF
        # descriptor/GateProxy reader.  They refer to the same immutable
        # selected contract above; no legacy parent feature contract is used.
        "meta_feature_contract": str(feature_contract),
        "meta_feature_count": len(fields),
        "trials": descriptor_trials,
        "weight_trials": str(weight_trials),
        "weight_trials_sha256": _sha(weight_trials),
        "weight_trial_name": str(weight_trial_name),
        "weight_trial": weight_trial,
        "held_months": [f"{month:%Y-%m}" for month in months],
        "model_families": sorted({str(candidate["model_family"]) for candidate in model_candidates}),
        "model_candidates": model_candidates,
        "model_plan": str(model_plan) if model_plan else None,
        "model_plan_sha256": _sha(model_plan) if model_plan else None,
        "selected_trial_plan": str(selected_trial_plan) if selected_trial_plan else None,
        "selected_trial_plan_sha256": _sha(selected_trial_plan) if selected_trial_plan else None,
        "source_override": str(source_override) if source_override else None,
        "source_override_sha256": _sha(source_override) if source_override else None,
        "only_candidate_names": [str(name) for name in only_candidate_names] if only_candidate_names else None,
        "common_budget": {
            "iterations": 220, "learning_rate": 0.05, "depth": 4,
            "feature_fraction": 0.80, "bagging_fraction": 0.82,
            "l2": 8.0, "one_thread_per_fit": True,
        },
        "hpo_stage": "final_lgbm_regular" if final_lgbm_hpo else "short_cross_family_screen",
        "selection": (
            "GateProxy screens the predeclared final LightGBM HPO descriptors; only a fresh strict-MC1-confirmed finalist may advance"
            if final_lgbm_hpo else
            "GateProxy screens the fixed target-free model descriptors; only a strict MC1-confirmed finalist may enter full HPO"
        ),
        "source": raw["source"],
        "source_hashes": {"base": _sha(base_root), "policy": _sha(policy_path), "path": _sha(path_root)},
        "causality": raw["causality"],
    })
    all_weekly: list[pd.DataFrame] = []
    all_bands: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    metrics_by_candidate: dict[str, list[dict[str, Any]]] = {
        str(candidate["name"]): [] for candidate in model_candidates
    }
    for fold_index, month in enumerate(months):
        prepared = objective._prepare_fold(
            base_root=base_root,
            feature_roots=feature_roots,
            policy=policy,
            path_root=path_root,
                arm=arm,
            fields=fields,
            panel_fields=panel_fields,
            continuous_sidecar=declared_sidecar,
            continuous_fields=continuous_fields,
            spec=spec,
                held_month=month,
                seed=int(spec.folds["seed"]) + fold_index,
                materialize_held_labels=True,
            )
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(int(workers), len(model_candidates))) as pool:
            futures = {
                pool.submit(
                    _score, prepared=prepared, arm=arm, family=str(candidate["model_family"]),
                    sample_weight_profile=sample_weight_profile, label_gain=label_gain,
                    seed=SEED + 10_000 * idx + fold_index,
                    trial_name=str(candidate["name"]), model_params=candidate.get("params"),
                ): candidate
                for idx, candidate in enumerate(model_candidates)
            }
            for future in concurrent.futures.as_completed(futures):
                candidate = futures[future]
                trial_name, family = str(candidate["name"]), str(candidate["model_family"])
                score = future.result()
                path = out / "target_free_scores" / trial_name / f"month={month:%Y-%m}.parquet"
                path.parent.mkdir(parents=True, exist_ok=True)
                score.to_parquet(path, index=False, compression="zstd")
                weekly, bands, metrics = screen._metrics(
                    score=score,
                    held_labelled=prepared.held_labelled,
                    held_anchor=prepared.held_anchor,
                    spec=spec,
                )
                weekly["trial"] = trial_name
                weekly["arm"] = arm.name
                weekly["held_month"] = f"{month:%Y-%m}"
                all_weekly.append(weekly)
                if not bands.empty:
                    bands["trial"] = trial_name
                    bands["arm"] = arm.name
                    bands["held_month"] = f"{month:%Y-%m}"
                    all_bands.append(bands)
                audit = {**prepared.audit, "trial": trial_name, "model_family": family, **metrics}
                audits.append(audit)
                metrics_by_candidate[trial_name].append(metrics)
                _progress(out, {"event": "fold_complete", "trial": trial_name, "model_family": family, "held_month": f"{month:%Y-%m}", "target_free_score": str(path), **metrics})
    rows: list[dict[str, Any]] = []
    for candidate in model_candidates:
        trial_name, family = str(candidate["name"]), str(candidate["model_family"])
        aggregate = pd.DataFrame(metrics_by_candidate[trial_name]).mean(numeric_only=True).to_dict()
        rows.append({"trial": trial_name, "model_family": family, "arm": arm.name, **aggregate})
    summary = pd.DataFrame(rows).sort_values(
        ["sstable_meta", "conditional_mi_meta_policy_given_base", "mean_top2_substitution_ev_bps", "trial"],
        ascending=[False, False, False, True], kind="stable",
    ).reset_index(drop=True)
    summary["rank"] = np.arange(1, len(summary) + 1, dtype=int)
    summary.to_parquet(out / "cross_model_summary.parquet", index=False, compression="zstd")
    # ``objective_summary`` is an interface receipt for the common descriptor
    # builder.  Model-family selection still has no direct authority: the
    # GateProxy consumes target-free scores and later MC1 remains the only
    # promotion gate.
    summary.to_parquet(out / "objective_summary.parquet", index=False, compression="zstd")
    summary.head(1).to_parquet(out / "cross_model_winner.parquet", index=False, compression="zstd")
    pd.DataFrame(audits).to_parquet(out / "cross_model_fold_metrics.parquet", index=False, compression="zstd")
    pd.concat(all_weekly, ignore_index=True).to_parquet(out / "weekly_sstable_meta.parquet", index=False, compression="zstd")
    (pd.concat(all_bands, ignore_index=True) if all_bands else pd.DataFrame()).to_parquet(
        out / "base_band_conversion_metrics.parquet", index=False, compression="zstd"
    )
    correctness = {
        "p8u_base_target_free_score_source": True,
        "declared_meta_features_merged_by_exact_identity": True,
        "all_train_labels_resolved_before_reserve": True,
        "train_residual_anchor_strict_prequential": True,
        "held_scores_persisted_before_held_outcome_metrics": True,
        "all_candidates_share_frozen_target_query_feature_and_fold_contract": True,
        "all_families_share_the_sealed_fit_only_sample_weight_contract": True,
        "selected_query_contract_is_preserved_without_timestamp_substitution": True,
        "model_families_are_limited_to_lgbm_xendcg_catboost_queryrmse_catboost_yetirank": True,
        "all_feature_medians_fit_train_only": True,
        "no_policy_or_path_field_in_target_free_inputs": True,
        "no_mc1_admission_portfolio_live_or_exchange_mutation": True,
        "shared_descriptor_gateproxy_contract_is_present": True,
    }
    if final_lgbm_hpo:
        correctness["final_lgbm_hpo_is_predeclared_and_frozen"] = True
        correctness["no_rejected_model_family_is_reopened"] = True
    else:
        correctness["no_full_hpo_performed"] = True
    _once(out / "correctness_report.json", correctness)
    _progress(out, {"event": "complete", "winner": str(summary.iloc[0].trial), "model_family": str(summary.iloc[0].model_family)})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--feature-contract", type=Path, required=True)
    parser.add_argument("--weight-trials", type=Path, required=True)
    parser.add_argument("--weight-trial-name", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--held-months", nargs="+")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--model-plan", type=Path, help="immutable model-specific HPO candidate bank")
    parser.add_argument("--selected-trial-plan", type=Path, help="frozen GateProxy shortlist for causal prehistory scoring")
    parser.add_argument("--source-override", type=Path, help="audited target-free historical source bridge only")
    parser.add_argument("--only-candidate", action="append", help="subset a sealed candidate plan without changing parameters")
    parser.add_argument("--final-lgbm-hpo", action="store_true", help="declare a preselected-family final LightGBM regularisation HPO")
    args = parser.parse_args()
    if int(args.workers) < 1:
        raise ValueError("--workers must be positive")
    print(run(
        config=args.config.resolve(),
        arm_name=str(args.arm),
        feature_contract=args.feature_contract.resolve(),
        weight_trials=args.weight_trials.resolve(),
        weight_trial_name=str(args.weight_trial_name),
        out=args.out.resolve(),
        held_month_values=args.held_months,
        workers=int(args.workers),
        model_plan=args.model_plan.resolve() if args.model_plan else None,
        selected_trial_plan=args.selected_trial_plan.resolve() if args.selected_trial_plan else None,
        source_override=args.source_override.resolve() if args.source_override else None,
        only_candidate_names=args.only_candidate,
        final_lgbm_hpo=bool(args.final_lgbm_hpo),
    ))


if __name__ == "__main__":
    main()
