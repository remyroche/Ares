#!/usr/bin/env python3
"""Run a staged downstream-economic AE/GMM representation comparison.

No expensive work is launched unless ``--stage`` requests it. The canonical
screen fits base once before five contiguous OOS months, promotes variants on
the first three months, then fits meta once and evaluates the final two months.
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
import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.ae_gmm_economic_ablation import (  # noqa: E402
    BASELINE_ARM,
    AEGMMArm,
    add_baseline_deltas,
    arm_model_features,
    base_selection_ranking,
    default_arms,
    economic_metrics,
    load_arms,
    load_feature_contract,
    model_ae_gmm_features,
    split_months,
    state_path_for_arm,
    strip_ae_gmm_features,
    write_feature_contract,
)
from extreme_price_movements.features_gmm_ae import load_ae_gmm_state_artifact  # noqa: E402
from extreme_price_movements.features_gmm_ae import (  # noqa: E402
    ae_gmm_cycle_reference_indices,
    ae_gmm_cycle_sample_identity_hash,
)
from scripts.report_s52_trailing_regime_meta_handoff import run_handoff_only  # noqa: E402
from scripts.report_train_meta_extended_pool_ablation_metrics import build_report  # noqa: E402
from scripts.run_materialized_trailing_label_topk_lgbm_hpo import (  # noqa: E402
    _fit_cycle_ae_gmm_state,
    _load_projected_labels,
    run_hpo,
)
from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    _load_feature_store_columns,
)
from scripts.run_s52_train_meta_regime_handoff_smoke import (  # noqa: E402
    _load_fixed_model_params,
    run_smoke,
)


DEFAULT_BASE_FEATURES = Path(
    "data_perp/reports/"
    "s59_h5_singlecycle_aegmm_bme_fs_fixedparams_wf30_20260716_v1/"
    "topk_lgbm_feature_selection_by_fold.csv"
)
DEFAULT_CURRENT_STATE = Path(
    "data_perp/reports/"
    "s59_h5_singlecycle_aegmm_bme_fs_fixedparams_wf30_20260716_v1/"
    "ae_gmm_states/cycle__global_state.pkl"
)
DEFAULT_BASE_PARAMS = Path("docs/promoted_s59_singlecycle_base_params.json")
DEFAULT_META_CONTRACT = Path(
    "extreme_price_movements/config/meta_v9_anchor_oldparams_residual_backbone_v1.json"
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _csv_values(raw: str, cast: Any) -> tuple[Any, ...]:
    return tuple(cast(value.strip()) for value in str(raw).split(",") if value.strip())


def _arm_map(arms: Sequence[AEGMMArm]) -> dict[str, AEGMMArm]:
    return {arm.arm_id: arm for arm in arms}


def _write_matrix_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _file_sha256(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sample_feature_value_hash(
    frame: pd.DataFrame,
    *,
    indices: np.ndarray,
    columns: Sequence[str],
) -> str:
    """Hash sampled covariates without materializing a second dense matrix."""

    digest = hashlib.sha256()
    digest.update("\n".join(map(str, columns)).encode("utf-8"))
    sampled = frame.iloc[np.asarray(indices, dtype=np.int64)]
    for column in columns:
        values = pd.to_numeric(sampled[str(column)], errors="coerce").astype(
            np.float32, copy=False
        )
        digest.update(
            pd.util.hash_pandas_object(values, index=False)
            .to_numpy(dtype=np.uint64, copy=False)
            .tobytes()
        )
    return digest.hexdigest()


def _write_or_validate_resume_contract(
    path: Path,
    payload: dict[str, Any],
    *,
    rerun: bool,
) -> None:
    normalized = _json_safe(payload)
    if path.exists() and not rerun:
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != normalized:
            raise ValueError(
                f"Refusing to reuse stale ablation output with a changed contract: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(normalized, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _export_model_feature_usage(model_root: Path, output_path: Path) -> Path | None:
    """Export fitted tree importance with an explicit AE/GMM-family marker."""

    rows: list[dict[str, Any]] = []
    for model_path in sorted(model_root.rglob("*.joblib")):
        try:
            model = joblib.load(model_path)
        except Exception:
            continue
        importance = getattr(model, "feature_importances_", None)
        names = list(getattr(model, "feature_name_", []) or [])
        if not names:
            columns_path = model_path.parent / "columns.json"
            if columns_path.exists():
                payload = json.loads(columns_path.read_text(encoding="utf-8"))
                names = list(payload.get("feature_names") or payload.get("columns") or [])
        if importance is None or len(names) != len(importance):
            continue
        state_names = set(model_ae_gmm_features(include_hard_ids=True))
        for feature, value in zip(names, np.asarray(importance).tolist()):
            raw = str(feature)
            state_feature = raw
            for prefix in ("base_lgbm_", "meta_lgbm_"):
                state_feature = state_feature.removeprefix(prefix)
            rows.append(
                {
                    "model_path": str(model_path),
                    "feature": raw,
                    "importance": float(value),
                    "is_ae_gmm_feature": bool(state_feature in state_names),
                }
            )
    if not rows:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    totals = frame.groupby("model_path", observed=True)["importance"].transform("sum")
    frame["importance_share"] = frame["importance"] / totals.replace(0.0, np.nan)
    frame.sort_values(["model_path", "importance"], ascending=[True, False]).to_csv(
        output_path, index=False
    )
    return output_path


def _ensure_round_trip_cost(ledger: pd.DataFrame, expected: float) -> dict[str, Any]:
    column = next(
        (
            name
            for name in (
                "__first_touch_round_trip_cost__",
                "round_trip_cost",
                "embedded_round_trip_cost",
            )
            if name in ledger.columns
        ),
        None,
    )
    if column is None:
        raise ValueError(
            "Cannot verify the economic ablation's round-trip cost; no cost column "
            "is present in the base ledger."
        )
    values = pd.to_numeric(ledger[column], errors="coerce").dropna()
    median = float(values.median()) if len(values) else float("nan")
    if not math.isfinite(median) or not math.isclose(median, float(expected), abs_tol=5e-5):
        raise ValueError(
            f"Round-trip cost mismatch: expected={expected:.6f} "
            f"observed_median={median:.6f} column={column}"
        )
    return {
        "column": column,
        "expected": float(expected),
        "observed_median": median,
        "observed_min": float(values.min()),
        "observed_max": float(values.max()),
        "double_count_guard": "ledger EV is consumed without subtracting cost again",
    }


def _load_state_fit_frame(
    *,
    labels_path: Path,
    feature_dir: Path,
    input_features: Sequence[str],
) -> pd.DataFrame:
    required = list(dict.fromkeys(map(str, input_features)))
    frame, _ = _load_projected_labels(
        labels_path,
        selected_features=[],
        ae_gmm_input_features=required,
    )
    missing = [name for name in required if name not in frame.columns]
    if missing:
        feature_matrix, _ = _load_feature_store_columns(
            frame,
            feature_dir=feature_dir,
            selected_features=missing,
        )
        if not feature_matrix.empty:
            additions = [name for name in missing if name in feature_matrix.columns]
            frame = pd.concat(
                [
                    frame.reset_index(drop=True),
                    feature_matrix.loc[:, additions].reset_index(drop=True).astype(np.float32, copy=False),
                ],
                axis=1,
                copy=False,
            )
    unavailable = [name for name in required if name not in frame.columns]
    if unavailable:
        raise ValueError(
            f"AE/GMM full-period reference is missing {len(unavailable)} inputs: "
            f"{unavailable[:20]}"
        )
    return frame


def fit_candidate_states(
    *,
    arms: Sequence[AEGMMArm],
    labels_path: Path,
    feature_dir: Path,
    state_root: Path,
    rerun: bool,
) -> dict[str, dict[str, Any]]:
    fit_arms = [arm for arm in arms if arm.mode == "fit"]
    if not fit_arms:
        return {}
    union_inputs = list(
        dict.fromkeys(feature for arm in fit_arms for feature in arm.input_features)
    )
    frame = _load_state_fit_frame(
        labels_path=labels_path,
        feature_dir=feature_dir,
        input_features=union_inputs,
    )
    ts = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    symbols = frame["__symbol__"].astype(str)
    sides = frame["side"].astype(str)
    reference_end = ts.max() + pd.Timedelta(hours=1)
    reference_window = {
        "fold": "whole_available_period_bme_transductive",
        "valid_start": reference_end,
        "valid_end": reference_end + pd.Timedelta(hours=1),
        "train_start": None,
    }
    outputs: dict[str, dict[str, Any]] = {}
    feature_value_hashes: dict[tuple[str, int, str], str] = {}
    for arm in fit_arms:
        destination = state_path_for_arm(arm, state_root)
        assert destination is not None
        reference_cap = max(
            int(arm.ae_max_train_rows), int(arm.gmm_max_train_rows)
        )
        reference_indices = ae_gmm_cycle_reference_indices(
            ts,
            symbols=symbols,
            sides=sides,
            max_rows=reference_cap,
        )
        value_hash_key = (
            str(arm.manifest()["input_feature_hash"]),
            int(reference_cap),
            ae_gmm_cycle_sample_identity_hash(
                ts,
                symbols=symbols,
                sides=sides,
                indices=reference_indices,
            ),
        )
        if value_hash_key not in feature_value_hashes:
            feature_value_hashes[value_hash_key] = _sample_feature_value_hash(
                frame,
                indices=reference_indices,
                columns=arm.input_features,
            )
        state_contract = {
            "schema": "ae_gmm_ablation_state_resume_contract_v1",
            "arm": arm.manifest(),
            "labels_path": str(labels_path.resolve()),
            "feature_dir": str(feature_dir.resolve()),
            "reference_rows_available": int(len(frame)),
            "reference_start": str(ts.min()),
            "reference_end": str(ts.max()),
            "reference_rows_sampled": int(len(reference_indices)),
            "reference_sample_identity_hash": value_hash_key[2],
            "reference_sample_feature_value_hash": feature_value_hashes[
                value_hash_key
            ],
        }
        _write_or_validate_resume_contract(
            destination.parent / "ablation_state_resume_contract.json",
            state_contract,
            rerun=rerun,
        )
        if destination.exists() and not rerun:
            state = load_ae_gmm_state_artifact(destination)
            actual_inputs = tuple(map(str, state.get("feature_columns", [])))
            actual_components = int(state.get("gmm_n_components", 0) or 0)
            actual_covariance = str(state.get("gmm_covariance_type", ""))
            actual_reg_covar = float(state.get("gmm_reg_covar", float("nan")))
            if actual_inputs != tuple(arm.input_features):
                raise ValueError(
                    f"Cached state input contract does not match arm {arm.arm_id}"
                )
            if arm.cluster_candidates and actual_components not in set(arm.cluster_candidates):
                raise ValueError(
                    f"Cached state components={actual_components} do not match "
                    f"arm {arm.arm_id} candidates={arm.cluster_candidates}"
                )
            if (
                arm.covariance_type_candidates
                and actual_covariance not in set(arm.covariance_type_candidates)
            ):
                raise ValueError(
                    f"Cached state covariance={actual_covariance!r} does not match "
                    f"arm {arm.arm_id} candidates={arm.covariance_type_candidates}"
                )
            if (
                arm.reg_covar_candidates
                and not any(
                    math.isclose(actual_reg_covar, float(candidate), abs_tol=1e-12)
                    for candidate in arm.reg_covar_candidates
                )
            ):
                raise ValueError(
                    f"Cached state reg_covar={actual_reg_covar} does not match "
                    f"arm {arm.arm_id} candidates={arm.reg_covar_candidates}"
                )
            for key, expected_value in (
                ("ae_max_train_rows", arm.ae_max_train_rows),
                ("gmm_max_train_rows", arm.gmm_max_train_rows),
            ):
                if int(state.get(key, 0) or 0) != int(expected_value):
                    raise ValueError(
                        f"Cached state {key}={state.get(key)!r} does not match "
                        f"arm {arm.arm_id} value={expected_value}"
                    )
            if str(state.get("cycle_reference_sample_identity_hash", "")) != str(
                state_contract["reference_sample_identity_hash"]
            ):
                raise ValueError(
                    f"Cached state reference sample does not match arm {arm.arm_id}"
                )
            outputs[arm.arm_id] = {
                "state_path": str(destination),
                "status": "reused",
            }
            continue
        state_path, contract = _fit_cycle_ae_gmm_state(
            frame=frame,
            ts_utc=ts,
            reference_window=reference_window,
            feature_columns=arm.input_features,
            input_feature_columns=arm.input_features,
            max_train_rows=int(arm.ae_max_train_rows),
            gmm_max_train_rows=int(arm.gmm_max_train_rows),
            ae_max_iter=int(arm.ae_max_iter),
            artifact_dir=destination.parent,
            seed=int(arm.seed),
            cluster_candidates=arm.cluster_candidates or None,
            reg_covar_candidates=arm.reg_covar_candidates or None,
            covariance_type_candidates=arm.covariance_type_candidates or None,
        )
        if state_path != destination:
            raise RuntimeError(
                f"Unexpected state artifact path for {arm.arm_id}: {state_path}"
            )
        outputs[arm.arm_id] = {
            **contract,
            "status": "fitted",
            "leakage_classification": "outcome_free_representation_transductive",
        }
    return outputs


def run_base_arms(
    *,
    arms: Sequence[AEGMMArm],
    production_features: Sequence[str],
    core_features: Sequence[str],
    state_root: Path,
    base_root: Path,
    labels_path: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    fixed_params_json: Path,
    months: Sequence[str],
    train_window_days: int,
    seed: int,
    rerun: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for arm in arms:
        arm_out = base_root / arm.arm_id
        manifest_path = arm_out / "manifest.json"
        ledger_path = arm_out / "best_oos_scored_ledger.parquet"
        feature_contract = write_feature_contract(
            arm_out / "model_feature_contract.json",
            arm_model_features(
                arm,
                production_features=production_features,
                core_features=core_features,
            ),
            source=arm.arm_id,
        )
        state_path = state_path_for_arm(arm, state_root)
        run_contract = {
            "schema": "ae_gmm_base_arm_resume_contract_v1",
            "arm": arm.manifest(),
            "months": list(map(str, months)),
            "train_window_days": int(train_window_days),
            "labels_path": str(labels_path),
            "feature_dir": str(feature_dir),
            "feature_list_csv": str(feature_list_csv),
            "fixed_params_json": str(fixed_params_json),
            "fixed_params_sha256": _file_sha256(fixed_params_json),
            "feature_contract_sha256": _file_sha256(feature_contract),
            "state_path": str(state_path) if state_path else None,
            "state_sha256": _file_sha256(state_path),
        }
        _write_or_validate_resume_contract(
            arm_out / "ablation_resume_contract.json",
            run_contract,
            rerun=rerun,
        )
        if not (manifest_path.exists() and ledger_path.exists() and not rerun):
            run_hpo(
                labels_path=labels_path,
                feature_dir=feature_dir,
                feature_list_csv=feature_list_csv,
                output_dir=arm_out,
                months=list(map(str, months)),
                max_feature_store_features=None,
                max_train_rows=0,
                feature_selection_sample_rows=45_000,
                hpo_max_train_rows=45_000,
                n_trials=0,
                seed=int(seed),
                include_ae_gmm_state_features=arm.mode != "none",
                ae_gmm_state_feature_max_train_rows=int(arm.ae_max_train_rows),
                ae_gmm_state_feature_gmm_max_train_rows=int(arm.gmm_max_train_rows),
                ae_gmm_state_feature_max_iter=int(arm.ae_max_iter),
                feature_selection_top_n=0,
                feature_selection_target_mode="target_soft",
                feature_selection_method="mda",
                max_oos_model_age_days=0,
                single_fit_oos_window=True,
                train_window_days=int(train_window_days),
                fixed_params_json=fixed_params_json,
                fixed_selected_features_csv=feature_contract,
                fixed_ae_gmm_state_pkl=state_path,
                rerun_hpo=False,
                rerun_ae_gmm_hpo=False,
                save_fold_models=True,
                save_final_model=False,
                two_phase_wide_feature_selection=True,
            )
        records.append(
            {
                "arm": arm.arm_id,
                "base_dir": str(arm_out),
                "ledger_path": str(ledger_path),
                "state_path": str(state_path) if state_path else None,
                "feature_contract": str(feature_contract),
            }
        )
        usage_path = _export_model_feature_usage(
            arm_out / "models", arm_out / "base_model_feature_usage.csv"
        )
        if usage_path is not None:
            records[-1]["feature_usage"] = str(usage_path)
        gc.collect()
    return records


def report_base_arms(
    *,
    arms: Sequence[AEGMMArm],
    base_root: Path,
    report_root: Path,
    all_months: Sequence[str],
    selection_months: Sequence[str],
    expected_cost: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_metrics: list[pd.DataFrame] = []
    selection_metrics: list[pd.DataFrame] = []
    cost_records: list[dict[str, Any]] = []
    for arm in arms:
        ledger_path = base_root / arm.arm_id / "best_oos_scored_ledger.parquet"
        if not ledger_path.exists():
            continue
        ledger = pd.read_parquet(ledger_path)
        cost_records.append({"arm": arm.arm_id, **_ensure_round_trip_cost(ledger, expected_cost)})
        all_metrics.append(
            economic_metrics(ledger, arm=arm.arm_id, months=all_months)
        )
        selection_metrics.append(
            economic_metrics(ledger, arm=arm.arm_id, months=selection_months)
        )
    if not all_metrics:
        raise RuntimeError("No completed base ledgers found")
    report_root.mkdir(parents=True, exist_ok=True)
    metrics = pd.concat(all_metrics, ignore_index=True)
    selection = pd.concat(selection_metrics, ignore_index=True)
    deltas = add_baseline_deltas(metrics)
    ranking = base_selection_ranking(selection, months=selection_months)
    metrics.to_csv(report_root / "base_all_oos_metrics.csv", index=False)
    deltas.to_csv(report_root / "base_all_oos_delta_vs_baseline.csv", index=False)
    selection.to_csv(report_root / "base_selection_period_metrics.csv", index=False)
    ranking.to_csv(report_root / "base_representation_ranking.csv", index=False)
    pd.DataFrame(cost_records).to_csv(report_root / "base_cost_audit.csv", index=False)
    return metrics, ranking


def run_meta_finalists(
    *,
    finalists: Sequence[str],
    arm_by_id: dict[str, AEGMMArm],
    meta_production_features: Sequence[str],
    meta_core_features: Sequence[str],
    state_root: Path,
    base_root: Path,
    handoff_root: Path,
    meta_root: Path,
    labels_path: Path,
    feature_dir: Path,
    meta_contract_path: Path,
    meta_train_months: Sequence[str],
    meta_oos_months: Sequence[str],
    round_trip_cost: float,
    seed: int,
    rerun: bool,
) -> list[dict[str, Any]]:
    params = _load_fixed_model_params(meta_contract_path)
    records: list[dict[str, Any]] = []
    for arm_id in finalists:
        arm = arm_by_id[arm_id]
        base_ledger = base_root / arm_id / "best_oos_scored_ledger.parquet"
        handoff_dir = handoff_root / arm_id
        state_path = state_path_for_arm(arm, state_root)
        handoff_path = handoff_dir / "train_meta_regime_handoff.parquet"
        if not (handoff_path.exists() and not rerun):
            run_handoff_only(
                ledger_path=base_ledger,
                output_dir=handoff_dir,
                label_context_dir=labels_path,
                feature_dir=feature_dir,
                feature_store_scope="all_safe",
                fixed_ae_gmm_state_pkl=state_path,
                fit_months=list(map(str, meta_train_months)),
                holdout_month=str(meta_oos_months[0]),
                selected_col="selected_top30",
                embedded_round_trip_cost=float(round_trip_cost),
                executable_cost_floor=float(round_trip_cost),
            )
        meta_features = arm_model_features(
            arm,
            production_features=meta_production_features,
            core_features=meta_core_features,
        )
        meta_feature_contract = write_feature_contract(
            meta_root / arm_id / "model_feature_contract.json",
            meta_features,
            source=arm_id,
        )
        arm_out = meta_root / arm_id
        run_contract = {
            "schema": "ae_gmm_meta_arm_resume_contract_v1",
            "arm": arm.manifest(),
            "meta_train_months": list(map(str, meta_train_months)),
            "meta_oos_months": list(map(str, meta_oos_months)),
            "handoff_path": str(handoff_path),
            "handoff_sha256": _file_sha256(handoff_path),
            "meta_contract_path": str(meta_contract_path),
            "meta_contract_sha256": _file_sha256(meta_contract_path),
            "feature_contract_sha256": _file_sha256(meta_feature_contract),
            "round_trip_cost": float(round_trip_cost),
        }
        _write_or_validate_resume_contract(
            arm_out / "ablation_resume_contract.json",
            run_contract,
            rerun=rerun,
        )
        prediction_path = arm_out / "s52_train_meta_regime_handoff_smoke_predictions.parquet"
        if not (prediction_path.exists() and not rerun):
            run_smoke(
                handoff_dir=handoff_dir,
                ledger_path=handoff_dir / "s52_trailing_regime_scored_ledger.parquet",
                handoff_path=handoff_path,
                out_dir=arm_out,
                frontier="top30",
                seed=int(seed),
                train_scope="selected",
                enable_base_prior_features=True,
                enable_reliability_features=True,
                enable_support_drift_features=False,
                enable_hit_surprise_features=False,
                feature_selection_top_n=0,
                feature_selection_target="ev_frontier",
                feature_selection_method="lgbm_pipeline",
                max_oos_model_age_days=0,
                single_fit_oos_window=True,
                validation_scope="chronological",
                model_train_max_rows=0,
                model_params=params,
                model_profile_name=f"ae_gmm_economic_ablation::{arm_id}",
                meta_head_mode="single_base_soft_label",
                minimal_artifacts=False,
                fixed_selected_features=meta_features,
                side_specific_single_head=False,
                eval_months=list(map(str, meta_oos_months)),
                force_prediction_shards=False,
                combine_prediction_shards=True,
                save_fold_models=True,
            )
        records.append(
            {
                "arm": arm_id,
                "handoff_dir": str(handoff_dir),
                "meta_dir": str(arm_out),
                "meta_feature_contract": str(meta_feature_contract),
            }
        )
        usage_path = _export_model_feature_usage(
            arm_out, arm_out / "meta_model_feature_usage.csv"
        )
        if usage_path is not None:
            records[-1]["feature_usage"] = str(usage_path)
        gc.collect()
    return records


def _choose_finalists(
    ranking: pd.DataFrame,
    *,
    count: int,
    available: set[str],
) -> list[str]:
    ordered = [str(value) for value in ranking["arm"] if str(value) in available]
    challengers = [value for value in ordered if value != BASELINE_ARM][: max(int(count), 0)]
    return list(dict.fromkeys([BASELINE_ARM, *challengers]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--feature-list-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-production-features", type=Path, default=DEFAULT_BASE_FEATURES)
    parser.add_argument("--base-fixed-params", type=Path, default=DEFAULT_BASE_PARAMS)
    parser.add_argument("--meta-production-contract", type=Path, default=DEFAULT_META_CONTRACT)
    parser.add_argument("--current-ae-gmm-state", type=Path, default=DEFAULT_CURRENT_STATE)
    parser.add_argument("--arms-json", type=Path, default=None)
    parser.add_argument("--oos-months", default="2026-02,2026-03,2026-04,2026-05,2026-06")
    parser.add_argument("--components", default="3,4,5,6,7,8")
    parser.add_argument("--reg-covars", default="0.0005,0.001,0.003")
    parser.add_argument("--covariance-types", default="diag")
    parser.add_argument("--meta-finalists", type=int, default=3)
    parser.add_argument("--base-train-window-days", type=int, default=365)
    parser.add_argument("--round-trip-cost", type=float, default=0.01)
    parser.add_argument("--min-group-rows", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stage",
        choices=("plan", "states", "base", "meta", "report", "all"),
        default="plan",
    )
    parser.add_argument("--rerun", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    periods = split_months(_csv_values(args.oos_months, str))
    arms = (
        load_arms(args.arms_json)
        if args.arms_json is not None
        else default_arms(
            current_state_path=args.current_ae_gmm_state,
            components=_csv_values(args.components, int),
            reg_covars=_csv_values(args.reg_covars, float),
            covariance_types=_csv_values(args.covariance_types, str),
            seed=int(args.seed),
        )
    )
    arm_by_id = _arm_map(arms)
    if BASELINE_ARM not in arm_by_id:
        raise ValueError(f"Arm configuration must include {BASELINE_ARM}")
    base_production = load_feature_contract(args.base_production_features)
    meta_production = load_feature_contract(args.meta_production_contract)
    base_core = strip_ae_gmm_features(base_production)
    meta_core = strip_ae_gmm_features(meta_production)
    state_root = args.output_root / "states"
    base_root = args.output_root / "base"
    handoff_root = args.output_root / "meta_handoffs"
    meta_root = args.output_root / "meta"
    report_root = args.output_root / "reports"
    manifest_path = args.output_root / "matrix_manifest.json"
    manifest: dict[str, Any] = {
        "schema": "ae_gmm_downstream_economic_ablation_matrix_v1",
        "stage_requested": args.stage,
        "periods": periods,
        "base_train_window_days": int(args.base_train_window_days),
        "round_trip_cost": float(args.round_trip_cost),
        "base_production_feature_count": len(base_production),
        "base_core_feature_count": len(base_core),
        "meta_production_feature_count": len(meta_production),
        "meta_core_feature_count": len(meta_core),
        "arms": [arm.manifest() for arm in arms],
        "leakage_contract": {
            "candidate_representation": (
                "outcome-free beginning/middle/end fit over the whole available period; "
                "representation-transductive and not untouched OOS"
            ),
            "incumbent_representation": (
                "reused exactly as serialized; its arm manifest discloses whether "
                "configuration selection was outcome-free or target-informed"
            ),
            "base": "one 365-day fit before month 1; fixed predictions over five OOS months",
            "base_promotion": "first three base OOS months only",
            "meta": "one fit on the first three base OOS months; final two months OOS",
            "cost": "1% embedded once in labels/ledger; no report-time subtraction",
        },
    }
    _write_matrix_manifest(manifest_path, manifest)
    if args.stage == "plan":
        print(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))
        return 0

    if args.stage in {"states", "all"}:
        manifest["state_results"] = fit_candidate_states(
            arms=arms,
            labels_path=args.labels_path,
            feature_dir=args.feature_dir,
            state_root=state_root,
            rerun=bool(args.rerun),
        )
        _write_matrix_manifest(manifest_path, manifest)
        if args.stage == "states":
            return 0

    if args.stage in {"base", "all"}:
        manifest["base_results"] = run_base_arms(
            arms=arms,
            production_features=base_production,
            core_features=base_core,
            state_root=state_root,
            base_root=base_root,
            labels_path=args.labels_path,
            feature_dir=args.feature_dir,
            feature_list_csv=args.feature_list_csv,
            fixed_params_json=args.base_fixed_params,
            months=periods["base_oos"],
            train_window_days=int(args.base_train_window_days),
            seed=int(args.seed),
            rerun=bool(args.rerun),
        )
        _, ranking = report_base_arms(
            arms=arms,
            base_root=base_root,
            report_root=report_root,
            all_months=periods["base_oos"],
            selection_months=periods["base_selection"],
            expected_cost=float(args.round_trip_cost),
        )
        manifest["base_ranking"] = ranking.to_dict(orient="records")
        _write_matrix_manifest(manifest_path, manifest)
        if args.stage == "base":
            return 0

    ranking_path = report_root / "base_representation_ranking.csv"
    if not ranking_path.exists():
        _, ranking = report_base_arms(
            arms=arms,
            base_root=base_root,
            report_root=report_root,
            all_months=periods["base_oos"],
            selection_months=periods["base_selection"],
            expected_cost=float(args.round_trip_cost),
        )
    else:
        ranking = pd.read_csv(ranking_path)
    finalists = _choose_finalists(
        ranking,
        count=int(args.meta_finalists),
        available=set(arm_by_id),
    )
    manifest["meta_finalists"] = finalists
    if args.stage in {"meta", "all"}:
        manifest["meta_results"] = run_meta_finalists(
            finalists=finalists,
            arm_by_id=arm_by_id,
            meta_production_features=meta_production,
            meta_core_features=meta_core,
            state_root=state_root,
            base_root=base_root,
            handoff_root=handoff_root,
            meta_root=meta_root,
            labels_path=args.labels_path,
            feature_dir=args.feature_dir,
            meta_contract_path=args.meta_production_contract,
            meta_train_months=periods["meta_train"],
            meta_oos_months=periods["meta_oos"],
            round_trip_cost=float(args.round_trip_cost),
            seed=int(args.seed),
            rerun=bool(args.rerun),
        )
        _write_matrix_manifest(manifest_path, manifest)
        if args.stage == "meta":
            return 0

    if args.stage in {"report", "all"}:
        manifest["meta_report"] = build_report(
            root_dir=meta_root,
            out_dir=report_root / "meta",
            min_group_rows=int(args.min_group_rows),
        )
        _write_matrix_manifest(manifest_path, manifest)
    print(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
