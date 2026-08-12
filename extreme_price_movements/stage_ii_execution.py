"""Executable, fail-closed Stage-II development orchestration.

Stage I deliberately persists a narrow score ledger.  Stage II needs both that
ledger *and* pre-materialised causal/meta fields plus realised path coordinates
for train-only discovery.  This module makes that extra dependency explicit;
it never manufactures a path coordinate from an outcome score or silently
falls back to an ordinary Stage-I meta replay.
"""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .lgbm_pipeline import _fit_lgbm_model
from .stage_i_strict_oof import _strict_train_mask, _validation_blocks
from .stage_ii_meta_archetype_funnel import (
    MetaOOFPredictor,
    StageIIFunnelError,
    StageIIMetaPredictionRequest,
    StageIIMetaPredictionResult,
)
from .stage_ii_meta_archetypes import SideLocalMetaArchetypeState
from .stage_i_causal_admission import apply_causal_21d_side_admission
from .stage_ii_production_oos import StageIILockedOOSScoringResult


SCHEMA = "stage_ii_executable_development_v1"


class StageIIExecutionError(ValueError):
    """The Stage-I handoff or Stage-II execution boundary is incomplete."""


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(series: pd.Series, *, name: str) -> pd.Series:
    out = pd.to_datetime(series, utc=True, errors="coerce")
    if out.isna().any():
        raise StageIIExecutionError(f"{name} must contain finite UTC timestamps")
    return out


def _as_bool(series: pd.Series, *, name: str) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(float)
    if not np.isfinite(values).all() or not np.isin(values, (0.0, 1.0)).all():
        raise StageIIExecutionError(f"{name} must be explicit boolean/0/1")
    return values.astype(bool)


def _base_fold_catalogue(provenance: pd.DataFrame) -> tuple[pd.DataFrame, tuple[dict[str, Any], ...]]:
    required = {"side", "layer", "fold_id", "train_max_label_available_ts", "validation_start_ts", "validation_end_ts", "strict_prior_resolved"}
    missing = sorted(required.difference(provenance.columns))
    if missing:
        raise StageIIExecutionError(f"Stage-I fold provenance lacks: {missing}")
    base = provenance.loc[provenance.layer.astype(str).eq("base_r3")].copy()
    if base.empty or not _as_bool(base.strict_prior_resolved, name="Stage-I base strict_prior_resolved").all():
        raise StageIIExecutionError("Stage-I direct R3 provenance is absent or non-causal")
    base["side"] = base.side.astype(str).str.lower()
    if not base.side.isin(("long", "short")).all() or base.duplicated(["side", "fold_id"]).any():
        raise StageIIExecutionError("Stage-I base provenance has duplicate/noncanonical side folds")
    base["train_max_label_available_ts"] = _utc(base.train_max_label_available_ts, name="base train cutoff")
    base["validation_start_ts"] = _utc(base.validation_start_ts, name="base validation start")
    base["validation_end_ts"] = _utc(base.validation_end_ts, name="base validation end")
    if not base.train_max_label_available_ts.lt(base.validation_start_ts).all():
        raise StageIIExecutionError("Stage-I base fold is not strictly prior-resolved")
    # Fold ids are side-local in Stage I.  The Stage-II public contract names
    # a single catalogue, so namespace them deterministically rather than
    # pretending long fold 0 and short fold 0 are one fitted model.
    base["stage_ii_fold_id"] = np.arange(len(base), dtype=np.int32)
    catalog = tuple({
        "fold_id": int(row.stage_ii_fold_id),
        "train_max_label_available_ts": row.train_max_label_available_ts.isoformat(),
        "validation_start_ts": row.validation_start_ts.isoformat(),
        "validation_end_ts": row.validation_end_ts.isoformat(),
    } for row in base.sort_values(["side", "fold_id"], kind="stable").itertuples(index=False))
    return base, catalog


def build_stage_ii_ledger(
    *,
    stage_i_predictions: pd.DataFrame,
    stage_i_fold_provenance: pd.DataFrame,
    enriched_ledger: pd.DataFrame,
    required_enriched_columns: Sequence[str],
) -> tuple[pd.DataFrame, tuple[dict[str, Any], ...]]:
    """Join an immutable Stage-I OOF ledger to an explicit path/context ledger.

    Only rows with direct same-side base OOF predictions and a prior-resolved
    base bps map can enter Stage II.  The latter condition intentionally
    removes neutral value-map burn-in rows rather than falsely asserting that
    an unavailable map had causal support.
    """
    source_required = {
        "candidate_id", "side_name", "decision_ts", "label_available_ts", "exact_net_bps", "exact_gross_bps",
        "base_strict_oof_available", "base_fold_id", "r3_p_adverse", "r3_p_weak", "r3_p_clear",
        "prequential_base_expected_net_bps", "value_map__value_map_max_label_available_ts",
    }
    missing = sorted(source_required.difference(stage_i_predictions.columns))
    if missing:
        raise StageIIExecutionError(f"completed Stage-I predictions lack direct base handoff fields: {missing}")
    enrich_required = {"candidate_id", "side_name", "decision_ts", *map(str, required_enriched_columns)}
    missing = sorted(enrich_required.difference(enriched_ledger.columns))
    if missing:
        raise StageIIExecutionError(f"enriched Stage-II ledger lacks required causal/path fields: {missing}")
    pred = stage_i_predictions.copy()
    pred["side_name"] = pred.side_name.astype(str).str.lower()
    pred["decision_ts"] = _utc(pred.decision_ts, name="Stage-I decision_ts")
    pred["label_available_ts"] = _utc(pred.label_available_ts, name="Stage-I label_available_ts")
    if pred.duplicated(["candidate_id", "side_name", "decision_ts"]).any():
        raise StageIIExecutionError("completed Stage-I predictions have duplicate immutable identities")
    # Keep only direct base OOF rows.  The original Stage-I meta residual is
    # intentionally ignored: Stage II refits the residual in matched controls.
    pred = pred.loc[_as_bool(pred.base_strict_oof_available, name="base_strict_oof_available")].copy()
    map_cutoff = pd.to_datetime(pred["value_map__value_map_max_label_available_ts"], utc=True, errors="coerce")
    pred = pred.loc[map_cutoff.notna() & map_cutoff.lt(pred.decision_ts)].copy()
    if pred.empty:
        raise StageIIExecutionError("no Stage-I rows have strict OOF base and prior-resolved base-map support")
    folds, catalogue = _base_fold_catalogue(stage_i_fold_provenance)
    pred = pred.merge(
        folds.loc[:, ["side", "fold_id", "stage_ii_fold_id", "train_max_label_available_ts"]],
        left_on=["side_name", "base_fold_id"], right_on=["side", "fold_id"], how="left", validate="many_to_one", sort=False,
    )
    if pred.stage_ii_fold_id.isna().any():
        raise StageIIExecutionError("a Stage-I base prediction references an absent strict fold")
    enrich = enriched_ledger.loc[:, list(dict.fromkeys(["candidate_id", "side_name", "decision_ts", *map(str, required_enriched_columns)]))].copy()
    enrich["side_name"] = enrich.side_name.astype(str).str.lower()
    enrich["decision_ts"] = _utc(enrich.decision_ts, name="enriched decision_ts")
    if enrich.duplicated(["candidate_id", "side_name", "decision_ts"]).any():
        raise StageIIExecutionError("enriched Stage-II ledger has duplicate identities")
    work = pred.merge(enrich, on=["candidate_id", "side_name", "decision_ts"], how="inner", validate="one_to_one", sort=False)
    if len(work) != len(pred):
        raise StageIIExecutionError("enriched Stage-II ledger does not cover every eligible Stage-I base OOF row")
    numeric = list(map(str, required_enriched_columns))
    if not np.isfinite(work.loc[:, numeric].apply(pd.to_numeric, errors="coerce").to_numpy(float)).all():
        raise StageIIExecutionError("required Stage-II causal/path inputs must be finite; no implicit imputation is allowed here")
    if "symbol" not in work.columns:
        raise StageIIExecutionError("completed Stage-I handoff lacks immutable symbol identity")
    work["symbol"] = work["symbol"].astype(str)
    if work.symbol.str.strip().eq("").any():
        raise StageIIExecutionError("Stage-II symbol identity is unavailable")
    work["r3_is_strict_oof"] = True
    work["r3_source_side"] = work.side_name
    work["r3_fit_end_ts"] = work.train_max_label_available_ts
    work["r3_score_semantics"] = "same_side_direct_strict_oof_probabilities_without_conversion"
    work["r3_oof_fold_id"] = work.stage_ii_fold_id.astype(np.int32)
    work["base_map_is_prequential"] = True
    work["base_map_source_side"] = work.side_name
    work["base_map_max_label_available_ts"] = pd.to_datetime(work["value_map__value_map_max_label_available_ts"], utc=True)
    work["signal_close_ts"] = work.decision_ts - pd.Timedelta(hours=1)
    work["total_cost_bps"] = np.float32(100.0)
    return work.sort_values(["decision_ts", "side_name", "candidate_id"], kind="stable").reset_index(drop=True), catalogue


def make_side_local_strict_meta_predictor(
    *,
    side_by_candidate_id: Mapping[str, str],
    params: Mapping[str, Any],
    n_validation_folds: int = 4,
    min_train_rows: int = 500,
) -> MetaOOFPredictor:
    """Return a matched, side-local strict-OOF residual runner for Stage II."""
    frozen = dict(params)
    if str(frozen.get("objective", "huber")).lower() != "huber":
        raise StageIIExecutionError("Stage-II residual params must declare Huber objective")
    frozen["objective"] = "huber"
    if int(n_validation_folds) < 1 or int(min_train_rows) < 3:
        raise StageIIExecutionError("Stage-II OOF controls require positive folds and >=3 prior rows")

    def predict(request: StageIIMetaPredictionRequest) -> StageIIMetaPredictionResult:
        ids = np.asarray(request.candidate_ids, dtype=object).astype(str)
        if len(set(ids)) != len(ids):
            raise StageIIExecutionError("Stage-II predictor requires globally unique candidate ids")
        side = np.asarray([side_by_candidate_id.get(value, "") for value in ids], dtype=object)
        if not np.isin(side, ("long", "short")).all():
            raise StageIIExecutionError("Stage-II predictor cannot resolve a canonical side for every candidate")
        decision = _utc(pd.Series(request.decision_timestamps), name="meta decision timestamps")
        available = _utc(pd.Series(request.label_available_timestamps), name="meta label timestamps")
        if not available.gt(decision).all():
            raise StageIIExecutionError("Stage-II residual labels must resolve after decision")
        target = np.asarray(request.target_residual_bps, dtype=np.float32)
        if not np.isfinite(target).all():
            raise StageIIExecutionError("Stage-II residual target must be finite")
        prediction = np.full(len(ids), np.nan, dtype=np.float32)
        fold_ids = np.full(len(ids), -1, dtype=np.int32)
        folds: list[dict[str, Any]] = []
        next_fold = 0
        for side_name in ("long", "short"):
            positions = np.flatnonzero(side == side_name)
            if not len(positions):
                continue
            local_decision = decision.iloc[positions].reset_index(drop=True)
            local_available = available.iloc[positions].reset_index(drop=True)
            blocks = _validation_blocks(local_decision, local_available, n_folds=n_validation_folds, min_train_rows=min_train_rows)
            for block in blocks:
                validation = positions[np.asarray(block, dtype=np.int32)]
                start = decision.iloc[validation].min()
                train_mask = (side == side_name) & _strict_train_mask(available, start)
                train = np.flatnonzero(train_mask)
                if len(train) < min_train_rows:
                    raise StageIIExecutionError("Stage-II matched residual fold has insufficient prior-resolved same-side rows")
                if not available.iloc[train].lt(start).all():
                    raise AssertionError("unresolved labels entered Stage-II residual fit")
                model = _fit_lgbm_model(
                    request.frame.iloc[train].loc[:, list(request.feature_columns)], target[train], None,
                    classifier=False, params=dict(frozen), objective_mode="stage_i_residual",
                )
                prediction[validation] = np.asarray(model.predict(request.frame.iloc[validation].loc[:, list(request.feature_columns)]), dtype=np.float32)
                fold_ids[validation] = next_fold
                folds.append({"fold_id": next_fold, "train_max_label_available_ts": available.iloc[train].max().isoformat(), "validation_start_ts": start.isoformat(), "validation_end_ts": decision.iloc[validation].max().isoformat(), "side": side_name, "train_rows": int(len(train)), "validation_rows": int(len(validation))})
                next_fold += 1
        if not np.isfinite(prediction).all() or (fold_ids < 0).any():
            raise StageIIExecutionError("Stage-II residual OOF left unmatched rows; burn-in must be excluded before controls")
        return StageIIMetaPredictionResult(
            candidate_ids=request.candidate_ids, predicted_residual_bps=prediction, oof_fold_ids=fold_ids,
            provenance={"strict_oof": True, "layer": "meta_residual", "score_semantics": "raw_predicted_residual_bps", "base_model_changed": False, "base_handoff": dict(request.base_handoff_provenance), "feature_columns": tuple(request.feature_columns), "folds": folds, "params": frozen},
        )
    return predict


def write_development_checkpoint(
    output_dir: str | Path,
    *,
    stage_i_oos_dir: str | Path,
    enriched_ledger: str | Path,
    result: Any,
    base_fold_catalogue: Sequence[Mapping[str, Any]],
    candidate_spec: Mapping[str, Any],
) -> Path:
    """Publish a restart-safe Stage-II development result, not an OOS result."""
    root = Path(output_dir).resolve()
    if root.exists():
        raise StageIIExecutionError("Stage-II development output must use a new immutable path")
    root.mkdir(parents=True)
    try:
        result.candidate_audit.to_parquet(root / "candidate_audit.parquet", index=False, compression="zstd")
        result.economic_stability.to_parquet(root / "economic_stability.parquet", index=False, compression="zstd")
        result.causal_predictability.to_parquet(root / "causal_predictability.parquet", index=False, compression="zstd")
        result.control_metrics.to_parquet(root / "control_metrics.parquet", index=False, compression="zstd")
        result.selected_contributions.to_parquet(root / "selected_contributions.parquet", index=False, compression="zstd")
        result.admission_audit.to_parquet(root / "causal_21d_admission_audit.parquet", index=False, compression="zstd")
        if result.oof_features is not None:
            result.oof_features.to_parquet(root / "selected_archetype_oof_features.parquet", index=False, compression="zstd")
        manifest = {"schema": SCHEMA, "status": "complete", "stage_i_oos_dir": str(Path(stage_i_oos_dir).resolve()), "stage_i_oos_manifest_sha256": file_sha256(Path(stage_i_oos_dir) / "manifest.json"), "enriched_ledger": str(Path(enriched_ledger).resolve()), "enriched_ledger_sha256": file_sha256(enriched_ledger), "base_r3_oof_fold_catalog": list(base_fold_catalogue), "candidate_spec": dict(candidate_spec), "result_manifest": dict(result.manifest), "selected_candidate_id": result.selected_candidate_id, "selected_control_arm": result.selected_control_arm, "selection_kind": "development_only; locked OOS requires separately frozen winner and one-shot scorer", "files": sorted(path.name for path in root.iterdir() if path.is_file())}
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    except Exception:
        import shutil
        shutil.rmtree(root, ignore_errors=True)
        raise
    return root


def validate_enriched_ledger_manifest(
    manifest: Mapping[str, Any],
    *,
    ledger_path: str | Path,
    required_causal_columns: Sequence[str],
    required_path_columns: Sequence[str],
) -> None:
    """Validate provenance for an externally materialised Stage-II ledger.

    Materialisation is deliberately owned by canonical path-label/context
    producers.  The execution layer only accepts an immutable manifest that
    binds the actual parquet bytes, common decision identity, causal inputs and
    realised path labels.  This prevents a later OOS run from quietly deriving
    labels from scores or a different candidate population.
    """
    if manifest.get("schema") != "stage_ii_enriched_path_context_ledger_v1":
        raise StageIIExecutionError("unsupported Stage-II enriched-ledger manifest schema")
    if str(manifest.get("ledger_sha256", "")) != file_sha256(ledger_path):
        raise StageIIExecutionError("enriched-ledger manifest does not bind supplied parquet bytes")
    identities = tuple(map(str, manifest.get("identity_columns", ())))
    if identities != ("candidate_id", "symbol", "side_name", "signal_close_ts", "decision_ts", "label_available_ts"):
        raise StageIIExecutionError("enriched-ledger manifest must bind the canonical executable identity")
    causal = set(map(str, manifest.get("causal_columns", ())))
    path = set(map(str, manifest.get("path_descriptor_columns", ())))
    if not set(map(str, required_causal_columns)).issubset(causal):
        raise StageIIExecutionError("enriched-ledger manifest omits a frozen causal input")
    if not set(map(str, required_path_columns)).issubset(path):
        raise StageIIExecutionError("enriched-ledger manifest omits a frozen realised path descriptor")
    labels = manifest.get("label_lineage")
    context = manifest.get("context_lineage")
    if not isinstance(labels, Mapping) or not isinstance(context, Mapping):
        raise StageIIExecutionError("enriched-ledger manifest requires canonical label and causal-context lineage")
    for name, lineage in (("label", labels), ("context", context)):
        for field in ("artifact_path", "artifact_sha256", "identity_sha256"):
            if not str(lineage.get(field, "")).strip():
                raise StageIIExecutionError(f"enriched-ledger {name} lineage lacks {field}")


def _selected_archetype_columns(arm: str, *, components: int) -> tuple[str, ...]:
    from .stage_ii_meta_archetypes import META_ARCHETYPE_PREFIX, membership_feature_names

    soft = tuple([
        *membership_feature_names(components), f"{META_ARCHETYPE_PREFIX}prob__unknown",
        f"{META_ARCHETYPE_PREFIX}entropy", f"{META_ARCHETYPE_PREFIX}confidence",
        f"{META_ARCHETYPE_PREFIX}support_log1p", f"{META_ARCHETYPE_PREFIX}available",
    ])
    prior = (f"{META_ARCHETYPE_PREFIX}prior_residual_bps",)
    if arm == "soft_memberships":
        return soft
    if arm == "prior":
        return prior
    if arm == "both":
        return (*soft, *prior)
    raise StageIIExecutionError("locked Stage-II scorer cannot execute a non-archetype control arm")


def _admission_columns(
    frame: pd.DataFrame,
    *,
    score_column: str,
    prefix: str,
) -> pd.DataFrame:
    """Apply the canonical 21d map and retain row-level causal provenance."""
    mapped, audit = apply_causal_21d_side_admission(
        frame, score_column=score_column, net_column="exact_net_bps",
        decision_column="decision_ts", label_available_column="label_available_ts",
        identity_column="candidate_id",
    )
    audit = audit.loc[:, ["snapshot_utc", "side_name", "reference_max_label_available_ts"]].copy()
    audit["__day"] = pd.to_datetime(audit.snapshot_utc, utc=True).dt.normalize()
    if audit.duplicated(["__day", "side_name"]).any():
        raise StageIIExecutionError("21-day admission audit has duplicate side/day provenance")
    mapped["__day"] = pd.to_datetime(mapped.decision_ts, utc=True).dt.normalize()
    mapped = mapped.merge(audit.drop(columns="snapshot_utc"), on=["__day", "side_name"], how="left", validate="many_to_one", sort=False)
    values = mapped["causal_21d_side_expected_net_bps"].to_numpy(float)
    flags = mapped["causal_21d_side_admitted_ge_50bps"].to_numpy(bool)
    # A no-support row is retained as a visible causal non-admission.  Its
    # unavailable map has no fabricated "last label" timestamp.
    if flags[~np.isfinite(values)].any() or not np.array_equal(flags, np.nan_to_num(values, nan=-np.inf) >= 50.0):
        raise AssertionError("canonical admission mapping/flag drift")
    mapped[f"{prefix}_causal_21d_side_expected_net_bps"] = values
    mapped[f"{prefix}_causal_21d_side_admitted_ge_50bps"] = flags
    mapped[f"{prefix}_causal_21d_admission_source_side"] = mapped.side_name.astype(str)
    mapped[f"{prefix}_causal_21d_admission_is_prequential"] = True
    mapped[f"{prefix}_causal_21d_admission_max_label_available_ts"] = pd.to_datetime(mapped.reference_max_label_available_ts, utc=True, errors="coerce")
    mapped[f"{prefix}_causal_21d_admission_window_days"] = 21
    return mapped.drop(columns=["__day"])


def make_locked_stage_ii_scorer(
    *,
    full_ledger: pd.DataFrame,
    candidate_config: Any,
    causal_feature_cols: Sequence[str],
    meta_feature_cols: Sequence[str],
    selected_control_arm: str,
    meta_params: Mapping[str, Any],
) -> Any:
    """Build the one-shot frozen Stage-II scorer used by locked OOS release.

    The callable accepts only the production module's history/development and
    locked identity context.  It performs no candidate comparison, HPO, or arm
    choice: a single conversion recogniser and side-local residual model are
    fitted on history+development then applied once to the supplied evaluation
    identity.  Realised path columns are stripped before every transform.
    """
    arm = str(selected_control_arm)
    additions = _selected_archetype_columns(arm, components=int(candidate_config.components))
    params = dict(meta_params)
    if str(params.get("objective", "huber")).lower() != "huber":
        raise StageIIExecutionError("locked Stage-II residual params must declare Huber")
    params["objective"] = "huber"
    base_meta = tuple(dict.fromkeys((
        "prequential_base_expected_net_bps", "r3_p_adverse", "r3_p_weak", "r3_p_clear", *map(str, meta_feature_cols),
    )))
    required = {"candidate_id", "side_name", "decision_ts", "label_available_ts", "exact_net_bps", *base_meta}
    if missing := sorted(required.difference(full_ledger.columns)):
        raise StageIIExecutionError(f"locked Stage-II full ledger lacks frozen fields: {missing}")

    def scorer(context: Mapping[str, Any]) -> StageIILockedOOSScoringResult:
        history = context["history"].copy()
        development = context["development"].copy()
        evaluation_identity = context["evaluation_identity"].copy()
        train_ids = set(pd.concat([history, development], ignore_index=True).candidate_id.astype(str))
        eval_ids = set(evaluation_identity.candidate_id.astype(str))
        if train_ids.intersection(eval_ids):
            raise StageIIExecutionError("locked Stage-II train/evaluation identities overlap")
        training = full_ledger.loc[full_ledger.candidate_id.astype(str).isin(train_ids)].copy()
        evaluation = full_ledger.loc[full_ledger.candidate_id.astype(str).isin(eval_ids)].copy()
        if len(training) != len(history) + len(development) or len(evaluation) != len(evaluation_identity):
            raise StageIIExecutionError("locked Stage-II scorer identity population drift")
        eval_start = pd.to_datetime(evaluation.decision_ts, utc=True).min()
        train_available = pd.to_datetime(training.label_available_ts, utc=True)
        if not train_available.lt(eval_start).all():
            raise StageIIExecutionError("locked Stage-II recogniser/residual fit includes unresolved evaluation-era labels")
        state = SideLocalMetaArchetypeState(candidate_config, causal_feature_cols).fit(training)
        train_causal = training.drop(columns=[candidate_config.exact_net_col, *candidate_config.path_descriptor_cols], errors="ignore")
        eval_causal = evaluation.drop(columns=[candidate_config.exact_net_col, *candidate_config.path_descriptor_cols], errors="ignore")
        train_archetype = state.transform(train_causal)
        eval_archetype = state.transform(eval_causal)
        if not pd.to_numeric(eval_archetype["meta_conversion_arch_available"], errors="coerce").eq(1.0).all():
            raise StageIIExecutionError("frozen Stage-II recogniser is unavailable for part of locked evaluation")
        train_design = pd.concat([training.loc[:, list(base_meta)].reset_index(drop=True), train_archetype.loc[:, list(additions)].reset_index(drop=True)], axis=1)
        eval_design = pd.concat([evaluation.loc[:, list(base_meta)].reset_index(drop=True), eval_archetype.loc[:, list(additions)].reset_index(drop=True)], axis=1)
        if not np.isfinite(train_design.apply(pd.to_numeric, errors="coerce").to_numpy(float)).all() or not np.isfinite(eval_design.apply(pd.to_numeric, errors="coerce").to_numpy(float)).all():
            raise StageIIExecutionError("frozen Stage-II model input is incomplete")
        prediction = np.full(len(evaluation), np.nan, dtype=np.float32)
        model_text: list[str] = []
        for side in ("long", "short"):
            train_mask = training.side_name.astype(str).str.lower().eq(side).to_numpy()
            eval_mask = evaluation.side_name.astype(str).str.lower().eq(side).to_numpy()
            if not eval_mask.any():
                continue
            if int(train_mask.sum()) < max(int(candidate_config.min_train_rows), 3):
                raise StageIIExecutionError(f"locked Stage-II {side} residual has insufficient frozen training support")
            target = (training.exact_net_bps.to_numpy(np.float32) - training.prequential_base_expected_net_bps.to_numpy(np.float32))[train_mask]
            model = _fit_lgbm_model(train_design.loc[train_mask], target, None, classifier=False, params=dict(params), objective_mode="stage_i_residual")
            prediction[eval_mask] = np.asarray(model.predict(eval_design.loc[eval_mask]), dtype=np.float32)
            booster = getattr(model, "booster_", None)
            model_text.append(booster.model_to_string() if booster is not None else repr(model))
        if not np.isfinite(prediction).all():
            raise StageIIExecutionError("locked Stage-II residual did not score the complete evaluation identity")
        out = evaluation.copy().reset_index(drop=True)
        for column in eval_archetype.columns:
            out[column] = eval_archetype[column].to_numpy()
        out["r3_raw_clear_minus_adverse"] = out.r3_p_clear.to_numpy(float) - out.r3_p_adverse.to_numpy(float)
        out["base_is_strict_oof"] = True
        out["base_source_side"] = out.side_name.astype(str)
        out["base_score_semantics"] = "same_side_direct_strict_oof_probabilities_without_conversion"
        out["base_oof_fold_id"] = out.r3_oof_fold_id.astype(np.int32)
        out["base_train_max_label_available_ts"] = out.r3_fit_end_ts
        out["meta_raw_predicted_residual_bps"] = prediction
        out["meta_reconstructed_expected_net_bps"] = out.prequential_base_expected_net_bps.to_numpy(np.float32) + prediction
        out["meta_is_strict_oof"] = True
        out["meta_source_side"] = out.side_name.astype(str)
        out["meta_score_semantics"] = "raw_predicted_residual_bps"
        out["meta_oof_fold_id"] = 0
        out["meta_train_max_label_available_ts"] = train_available.max()
        for layer in ("base", "meta"):
            score = "prequential_base_expected_net_bps" if layer == "base" else "meta_reconstructed_expected_net_bps"
            combined = pd.concat([training, out], ignore_index=True, sort=False)
            combined = _admission_columns(combined, score_column=score, prefix=layer)
            mapped = combined.loc[combined.candidate_id.astype(str).isin(eval_ids)].copy()
            mapped = mapped.set_index("candidate_id").reindex(out.candidate_id.astype(str)).reset_index()
            for name in (f"{layer}_causal_21d_side_expected_net_bps", f"{layer}_causal_21d_side_admitted_ge_50bps", f"{layer}_causal_21d_admission_source_side", f"{layer}_causal_21d_admission_is_prequential", f"{layer}_causal_21d_admission_max_label_available_ts", f"{layer}_causal_21d_admission_window_days"):
                out[name] = mapped[name].to_numpy()
        for prefix in ("base", "meta"):
            out[f"{prefix}_candidate_id"] = out.candidate_id
            out[f"{prefix}_symbol"] = out.symbol
            out[f"{prefix}_decision_ts"] = out.decision_ts
            out[f"{prefix}_side_name"] = out.side_name
        model_sha = sha256("\n".join(model_text).encode("utf-8")).hexdigest()
        return StageIILockedOOSScoringResult(out, {
            "model_sha256": model_sha,
            "reselection_forbidden": True, "hpo_forbidden": True,
            "selected_discovery_candidate_id": context["winner_manifest"].selected_discovery_candidate_id,
            "selected_control_arm": context["winner_manifest"].selected_control_arm,
            # The caller adds immutable winner/base/fold bindings expected by
            # stage_ii_production_oos before validation.
        })
    return scorer


__all__ = ["SCHEMA", "StageIIExecutionError", "build_stage_ii_ledger", "file_sha256", "make_locked_stage_ii_scorer", "make_side_local_strict_meta_predictor", "validate_enriched_ledger_manifest", "write_development_checkpoint"]
