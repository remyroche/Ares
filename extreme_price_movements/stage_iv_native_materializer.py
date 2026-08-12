"""Load hash-bound Stage-IV native cells from declarative artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .lgbm_pipeline import _fit_lgbm_model
from .stage_i_adapter_winner_bundle import StageIAdapterWinnerBundle
from .stage_i_causal_admission import Causal21dAdmissionSpec
from .stage_i_strict_oof import _multiclass_probabilities
from .stage_i_target_adapter import (
    CUMULATIVE_ORDINAL5_O,
    FOLD_QUANTILE_RESIDUAL3,
    LEGACY_R3_MULTICLASS3,
    SOFT_SCALAR_S,
    StageITargetContract,
    canonical_sha256,
    file_sha256,
    fit_cumulative_ordinal5_estimator,
)
from .stage_i_base_target_ablation import recover_ordinal_simplex
from .stage_i_target_specific_oos import (
    DirectCorrectnessState,
    DIRECT_BASE_INPUT_SEMANTICS,
    DIRECT_FQ3_SEMANTICS,
    _clean_params,
)
from .stage_iv_native_artifact_runner import (
    FROZEN_MODEL_SERIALIZATION_FORMAT,
    NativeBasePrediction,
    StageIVNativeCell,
    StageIVNativePlan,
    StageIVNativeFrozenArtifact,
    StageIVNativeFrozenOOSPlan,
    StageIVNativeRunnerError,
    StageIVNativeRunnerSpec,
)


MATERIALIZER_SCHEMA = "stage_iv_native_explicit_cell_spec_v1"
FROZEN_OOS_MATERIALIZER_SCHEMA = "stage_iv_native_frozen_oos_spec_v1"
_GENERATED_META = {
    "base_raw_score", "base_output_entropy", "base_output_top2_margin",
    "base_output_max_probability",
}


class StageIVNativeMaterializationError(StageIVNativeRunnerError):
    """A declarative source, hash, or winner contract is invalid."""


@dataclass(frozen=True)
class StageIVNativeLaunch:
    cells: tuple[StageIVNativeCell, ...]
    runner_spec: StageIVNativeRunnerSpec
    winner_bundle: StageIAdapterWinnerBundle
    launch_manifest: Mapping[str, Any]


@dataclass(frozen=True)
class StageIVNativeFrozenOOSLaunch:
    plans: tuple[StageIVNativeFrozenOOSPlan, ...]
    admission_spec: Causal21dAdmissionSpec
    launch_manifest: Mapping[str, Any]


class _NativeWinnerModel:
    def __init__(self, model: Any, family: str) -> None:
        self.model, self.family = model, family

    def predict_native(self, frame: pd.DataFrame) -> NativeBasePrediction:
        if self.family == SOFT_SCALAR_S:
            score = np.clip(np.asarray(self.model.predict(frame), float), 0.0, 1.0)
            states = np.column_stack([1.0 - score, score])
        elif self.family == CUMULATIVE_ORDINAL5_O:
            states = recover_ordinal_simplex(
                self.model.predict_cumulative_probability(frame)
            )
            score = states @ (np.arange(5, dtype=float) / 4.0)
        elif self.family == LEGACY_R3_MULTICLASS3:
            states = _multiclass_probabilities(self.model, frame)
            score = states[:, 2] - states[:, 0]
        else:
            raise StageIVNativeMaterializationError(
                f"unsupported frozen native base family {self.family}"
            )
        return NativeBasePrediction(
            np.asarray(score, dtype=np.float32),
            np.asarray(states, dtype=np.float32),
        )


def native_winner_base_fitter(
    frame: pd.DataFrame, target: np.ndarray, weight: np.ndarray,
    layer: str, params: Mapping[str, Any],
) -> _NativeWinnerModel:
    contract = StageITargetContract.from_dict(params["__target_contract__"])
    frozen = {key: value for key, value in params.items() if not key.startswith("__")}
    if contract.family == SOFT_SCALAR_S:
        model = _fit_lgbm_model(
            frame, np.asarray(target, dtype=np.float32), weight, classifier=False,
            params=_clean_params(frozen, objective="regression_l1"),
            objective_mode=f"stage_iv_native_{layer}_S",
        )
    elif contract.family == CUMULATIVE_ORDINAL5_O:
        model = fit_cumulative_ordinal5_estimator(
            frame, np.asarray(target, dtype=np.int8), weight,
            params=_clean_params(frozen, objective="binary"),
        )
    elif contract.family == LEGACY_R3_MULTICLASS3:
        model = _fit_lgbm_model(
            frame, np.asarray(target, dtype=np.int8), weight, classifier=True,
            params=_clean_params(frozen, objective="multiclass", num_class=3),
            objective_mode=f"stage_iv_native_{layer}_R3",
        )
    else:
        raise StageIVNativeMaterializationError(
            "Stage IV winner machinery supports only S/O/R3 native bases"
        )
    return _NativeWinnerModel(model, contract.family)


def direct_fq3_winner_meta_fitter(
    frame: pd.DataFrame, labels: np.ndarray, weight: np.ndarray,
    _layer: str, params: Mapping[str, Any],
) -> Any:
    contract = StageITargetContract.from_dict(params["__target_contract__"])
    metadata = dict(contract.metadata)
    if (
        contract.family != FOLD_QUANTILE_RESIDUAL3
        or metadata.get("meta_target_semantics") != DIRECT_FQ3_SEMANTICS
        or metadata.get("base_input_semantics") != DIRECT_BASE_INPUT_SEMANTICS
    ):
        raise StageIVNativeMaterializationError("winner meta contract is not direct FQ3")
    frozen = {key: value for key, value in params.items() if not key.startswith("__")}
    return _fit_lgbm_model(
        frame, np.asarray(labels, dtype=np.int8), weight, classifier=True,
        params=_clean_params(frozen, objective="multiclass", num_class=3),
        objective_mode="stage_iv_native_direct_FQ3",
    )


def _path(base: Path, value: Any) -> Path:
    result = Path(str(value))
    return result if result.is_absolute() else (base / result).resolve()


def _read_json(path: Path) -> Mapping[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise StageIVNativeMaterializationError(f"expected JSON object: {path}")
    return raw


def _verify_file(path: Path, expected: Any, name: str) -> str:
    if not path.is_file():
        raise StageIVNativeMaterializationError(f"{name} is absent: {path}")
    observed = file_sha256(path)
    if observed != str(expected):
        raise StageIVNativeMaterializationError(f"{name} SHA-256 drift")
    return observed


def _winner_contract_hashes(bundle: StageIAdapterWinnerBundle, side: str) -> tuple[str, str]:
    cell = bundle.cell(side)
    feature_sha = canonical_sha256({
        "base": list(cell.base_features), "meta": list(cell.meta_features),
    })
    parameter_sha = canonical_sha256({
        "base": dict(cell.base_params), "meta": dict(cell.meta_params),
        "base_target_contract_sha256": cell.base_target_contract.sha256,
        "meta_target_contract_sha256": cell.meta_target_contract.sha256,
    })
    return feature_sha, parameter_sha


def _ledger_plan(
    *, ledger: pd.DataFrame, side: str, source: Mapping[str, Any],
    winner: StageIAdapterWinnerBundle, cell: Mapping[str, Any],
) -> StageIVNativePlan:
    frozen = winner.cell(side)
    meta_metadata = dict(frozen.meta_target_contract.metadata)
    if frozen.base_target_contract.family not in {
        SOFT_SCALAR_S, CUMULATIVE_ORDINAL5_O, LEGACY_R3_MULTICLASS3,
    }:
        raise StageIVNativeMaterializationError(f"{side} winner base is not native S/O/R3")
    if (
        frozen.meta_target_contract.family != FOLD_QUANTILE_RESIDUAL3
        or meta_metadata.get("meta_target_semantics") != DIRECT_FQ3_SEMANTICS
        or meta_metadata.get("base_input_semantics") != DIRECT_BASE_INPUT_SEMANTICS
    ):
        raise StageIVNativeMaterializationError(f"{side} winner meta is not direct FQ3")
    if any("mapped" in name.lower() or "expected_net" in name.lower()
           for name in frozen.meta_features):
        raise StageIVNativeMaterializationError(
            f"{side} winner meta feature contract contains pre-mapped bps"
        )
    feature_sha, parameter_sha = _winner_contract_hashes(winner, side)
    if source.get("winner_feature_contract_sha256") != feature_sha:
        raise StageIVNativeMaterializationError(f"{side} winner feature contract SHA drift")
    if source.get("winner_parameter_contract_sha256") != parameter_sha:
        raise StageIVNativeMaterializationError(f"{side} winner parameter contract SHA drift")
    columns = dict(source.get("columns", {}))
    required = {
        "candidate_id", "symbol", "decision_ts", "label_available_ts",
        "base_target", "exact_net_bps",
    }
    if set(columns) != required:
        raise StageIVNativeMaterializationError(
            f"{side} ledger column contract must declare exactly {sorted(required)}"
        )
    names = {key: str(value) for key, value in columns.items()}
    raw_meta = tuple(
        name for name in frozen.meta_features
        if name not in _GENERATED_META and not name.startswith("base_state_p")
    )
    selected = tuple(dict.fromkeys((*frozen.base_features, *raw_meta)))
    missing = sorted({*names.values(), *selected}.difference(ledger.columns))
    if missing:
        raise StageIVNativeMaterializationError(f"{side} ledger lacks fields: {missing[:12]}")
    burns = dict(cell.get("burn_ins", {}))
    if set(burns) != {"broad", "tail", "meta", "handoff_history"}:
        raise StageIVNativeMaterializationError(
            "each explicit cell requires all three burn-ins and handoff history"
        )
    base_params = {
        **dict(frozen.base_params),
        "__target_contract__": frozen.base_target_contract.to_dict(),
    }
    meta_params = {
        **dict(frozen.meta_params),
        "__target_contract__": frozen.meta_target_contract.to_dict(),
    }
    score_domain = (
        (-1.0, 1.0)
        if frozen.base_target_contract.family == LEGACY_R3_MULTICLASS3 else (0.0, 1.0)
    )
    return StageIVNativePlan(
        side=side, candidate_ids=ledger[names["candidate_id"]],
        symbols=ledger[names["symbol"]], frame=ledger,
        base_target=ledger[names["base_target"]],
        exact_net_bps=ledger[names["exact_net_bps"]],
        decision_timestamps=ledger[names["decision_ts"]],
        label_available_timestamps=ledger[names["label_available_ts"]],
        broad_feature_names=tuple(frozen.base_features),
        tail_feature_names=tuple(frozen.base_features), meta_feature_names=raw_meta,
        broad_params=base_params, tail_params=base_params, meta_params=meta_params,
        tail_fraction=float(cell["tail_fraction"]),
        broad_min_train_rows=int(burns["broad"]),
        tail_min_train_rows=int(burns["tail"]),
        meta_min_train_rows=int(burns["meta"]),
        min_handoff_history_rows=int(burns["handoff_history"]),
        n_validation_folds=int(cell.get("n_validation_folds", 4)),
        broad_output_route=str(cell["broad_output_route"]),
        score_domain=score_domain,
        sample_weight=(
            None if source.get("sample_weight_column") is None
            else ledger[str(source["sample_weight_column"])]
        ),
        cost_bps=100.0,
    )


def load_stage_iv_native_launch(cell_spec_path: str | Path) -> StageIVNativeLaunch:
    """Materialise only the explicitly enumerated cells in a hash-bound spec."""
    path = Path(cell_spec_path).resolve()
    raw, base = _read_json(path), path.parent
    if raw.get("schema") != MATERIALIZER_SCHEMA:
        raise StageIVNativeMaterializationError("unsupported Stage-IV cell-spec schema")
    winner_ref = dict(raw.get("winner_bundle", {}))
    winner_path = _path(base, winner_ref.get("path"))
    winner_file_sha = _verify_file(
        winner_path, winner_ref.get("file_sha256"), "winner bundle"
    )
    winner = StageIAdapterWinnerBundle.from_dict(_read_json(winner_path))
    if winner.sha256 != winner_ref.get("contract_sha256"):
        raise StageIVNativeMaterializationError("winner bundle semantic SHA-256 drift")
    sources = dict(raw.get("side_ledgers", {}))
    if set(sources) != {"long", "short"}:
        raise StageIVNativeMaterializationError("cell spec requires immutable long/short ledgers")
    ledgers: dict[str, pd.DataFrame] = {}
    ledger_hashes: dict[str, str] = {}
    for side, source_value in sources.items():
        source = dict(source_value)
        ledger_path = _path(base, source.get("path"))
        ledger_hashes[side] = _verify_file(
            ledger_path, source.get("sha256"), f"{side} ledger"
        )
        ledgers[side] = pd.read_parquet(ledger_path)
    declared = raw.get("cells")
    if not isinstance(declared, Sequence) or isinstance(declared, (str, bytes)) or not declared:
        raise StageIVNativeMaterializationError("cell spec requires an explicit ordered cells list")
    cells: list[StageIVNativeCell] = []
    for item in declared:
        if not isinstance(item, Mapping):
            raise StageIVNativeMaterializationError("every explicit cell must be an object")
        cell_id = str(item.get("cell_id", "")).strip()
        if not cell_id:
            raise StageIVNativeMaterializationError("explicit cell_id must be non-empty")
        plans = tuple(
            _ledger_plan(
                ledger=ledgers[side], side=side, source=dict(sources[side]),
                winner=winner, cell=item,
            )
            for side in ("long", "short")
        )
        cells.append(StageIVNativeCell(
            cell_id=cell_id, plans=plans,
            source_lineage={
                "cell_spec": file_sha256(path), "winner_bundle_file": winner_file_sha,
                "winner_bundle_contract": winner.sha256,
                "long_ledger": ledger_hashes["long"], "short_ledger": ledger_hashes["short"],
            },
        ))
    runner_raw = dict(raw.get("runner", {}))
    admission = Causal21dAdmissionSpec(**dict(runner_raw.pop("admission_spec", {})))
    runner = StageIVNativeRunnerSpec(admission_spec=admission, **runner_raw)
    runner.validate()
    launch_manifest = {
        "schema": MATERIALIZER_SCHEMA, "cell_spec_path": str(path),
        "cell_spec_sha256": file_sha256(path), "winner_bundle_path": str(winner_path),
        "winner_bundle_file_sha256": winner_file_sha,
        "winner_bundle_contract_sha256": winner.sha256,
        "ledger_sha256": ledger_hashes, "explicit_cell_ids": [cell.cell_id for cell in cells],
        "factorial_generation": False,
    }
    return StageIVNativeLaunch(tuple(cells), runner, winner, launch_manifest)


def _safe_model_descriptor(base: Path, value: Any, *, role: str) -> dict[str, str]:
    """Resolve and integrity-check one declared frozen model file.

    The format is intentionally explicit.  A SHA-256 match is verified before
    the trusted local joblib payload is deserialised; arbitrary formats and
    implicit pickle paths are never accepted by the frozen-OOS CLI.
    """
    item = dict(value) if isinstance(value, Mapping) else {}
    if str(item.get("format", "")) != FROZEN_MODEL_SERIALIZATION_FORMAT:
        raise StageIVNativeMaterializationError(
            f"{role} must use {FROZEN_MODEL_SERIALIZATION_FORMAT}"
        )
    path = _path(base, item.get("path"))
    digest = _verify_file(path, item.get("sha256"), f"{role} frozen model")
    return {"path": str(path), "sha256": digest, "format": FROZEN_MODEL_SERIALIZATION_FORMAT}


def _load_safe_joblib_model(descriptor: Mapping[str, str], *, role: str) -> Any:
    # `joblib` is reached only after the descriptor's exact content digest was
    # validated.  This is a trusted-local artifact boundary, not a general
    # user-supplied pickle loader.
    import joblib

    path = Path(str(descriptor["path"]))
    _verify_file(path, descriptor["sha256"], f"{role} frozen model")
    model = joblib.load(path)
    method = "predict_native" if role in {"broad_model", "tail_model"} else "predict_proba"
    if not callable(getattr(model, method, None)):
        raise StageIVNativeMaterializationError(
            f"{role} deserialised model lacks required {method} interface"
        )
    return model


def _frozen_frame(base: Path, value: Any, *, name: str) -> tuple[pd.DataFrame, str]:
    item = dict(value) if isinstance(value, Mapping) else {}
    path = _path(base, item.get("path"))
    digest = _verify_file(path, item.get("sha256"), name)
    return pd.read_parquet(path), digest


def _frozen_oos_contract_sha(payload: Mapping[str, Any]) -> str:
    return canonical_sha256(payload)


def load_stage_iv_native_frozen_oos_launch(spec_path: str | Path) -> StageIVNativeFrozenOOSLaunch:
    """Load only hash-bound, already-fitted Stage-IV OOS artifacts.

    This materializer has no fitter, HPO, or cell-selection parameters.  The
    direct-FQ3 state is JSON data, while all three executable models must be
    separately declared and content-hashed serialized artifacts.
    """
    path = Path(spec_path).resolve()
    raw, base = _read_json(path), path.parent
    if raw.get("schema") != FROZEN_OOS_MATERIALIZER_SCHEMA:
        raise StageIVNativeMaterializationError("unsupported frozen Stage-IV OOS spec schema")
    sides = dict(raw.get("sides", {}))
    if set(sides) != {"long", "short"}:
        raise StageIVNativeMaterializationError("frozen OOS spec requires long and short artifacts")
    admission = Causal21dAdmissionSpec(**dict(raw.get("admission_spec", {})))
    plans: list[StageIVNativeFrozenOOSPlan] = []
    manifest_artifacts: dict[str, Any] = {}
    for side in ("long", "short"):
        item = dict(sides[side])
        frozen = dict(item.get("artifact", {}))
        if str(frozen.get("side", "")).lower() != side:
            raise StageIVNativeMaterializationError(f"{side} frozen artifact side mismatch")
        models_raw = dict(frozen.get("model_artifacts", {}))
        if set(models_raw) != {"broad_model", "tail_model", "meta_model"}:
            raise StageIVNativeMaterializationError(f"{side} requires exactly three frozen model artifacts")
        models = {role: _safe_model_descriptor(base, models_raw[role], role=f"{side}/{role}") for role in sorted(models_raw)}
        model_manifest_sha = canonical_sha256(models)
        if model_manifest_sha != str(frozen.get("model_artifact_manifest_sha256", "")):
            raise StageIVNativeMaterializationError(f"{side} frozen model artifact manifest SHA-256 drift")
        handoff, handoff_sha = _frozen_frame(base, frozen.get("pre_oos_handoff_history"), name=f"{side} handoff history")
        reference, reference_sha = _frozen_frame(base, frozen.get("pre_oos_mapping_reference"), name=f"{side} mapping reference")
        state_raw = frozen.get("direct_fq3_state")
        if not isinstance(state_raw, Mapping):
            raise StageIVNativeMaterializationError(f"{side} direct FQ3 state must be JSON object data")
        try:
            fq3_state = DirectCorrectnessState(**dict(state_raw))
        except Exception as exc:
            raise StageIVNativeMaterializationError(f"{side} direct FQ3 state is invalid") from exc
        feature_sets = {name: tuple(map(str, frozen.get(name, ()))) for name in ("broad_feature_names", "tail_feature_names", "meta_feature_names")}
        if any(not names for names in feature_sets.values()):
            raise StageIVNativeMaterializationError(f"{side} frozen feature contracts must be non-empty")
        artifact_payload = {
            "artifact_id": str(frozen.get("artifact_id", "")), "side": side,
            "freeze_cutoff_timestamp": str(frozen.get("freeze_cutoff_timestamp", "")),
            "model_artifact_manifest_sha256": model_manifest_sha,
            "handoff_history_sha256": handoff_sha, "mapping_reference_sha256": reference_sha,
            "direct_fq3_state": dict(state_raw), "feature_sets": {key: list(value) for key, value in feature_sets.items()},
            "broad_output_route": str(frozen.get("broad_output_route", "")),
            "tail_fraction": frozen.get("tail_fraction"), "min_handoff_history_rows": frozen.get("min_handoff_history_rows"),
            "score_domain": list(frozen.get("score_domain", ())),
        }
        artifact_sha = _frozen_oos_contract_sha(artifact_payload)
        if artifact_sha != str(frozen.get("artifact_sha256", "")):
            raise StageIVNativeMaterializationError(f"{side} frozen artifact SHA-256 drift")
        panel, panel_sha = _frozen_frame(base, item.get("oos_panel"), name=f"{side} OOS panel")
        columns = dict(item.get("columns", {}))
        required = {"candidate_id", "symbol", "decision_ts", "label_available_ts", "exact_net_bps"}
        if set(columns) != required or set(map(str, columns.values())).difference(panel.columns):
            raise StageIVNativeMaterializationError(f"{side} OOS panel column contract is invalid")
        model_objects = {role: _load_safe_joblib_model(models[role], role=role) for role in models}
        artifact = StageIVNativeFrozenArtifact(
            artifact_id=str(frozen["artifact_id"]), artifact_sha256=artifact_sha,
            freeze_cutoff_timestamp=frozen["freeze_cutoff_timestamp"], side=side,
            broad_model=model_objects["broad_model"], tail_model=model_objects["tail_model"], meta_model=model_objects["meta_model"],
            direct_fq3_state=fq3_state,
            broad_feature_names=feature_sets["broad_feature_names"], tail_feature_names=feature_sets["tail_feature_names"], meta_feature_names=feature_sets["meta_feature_names"],
            broad_output_route=str(frozen["broad_output_route"]), tail_fraction=float(frozen["tail_fraction"]),
            min_handoff_history_rows=int(frozen["min_handoff_history_rows"]), score_domain=tuple(map(float, frozen["score_domain"])),
            pre_oos_handoff_history=handoff, pre_oos_mapping_reference=reference,
            model_artifacts=models, model_artifact_manifest_sha256=model_manifest_sha,
        )
        plans.append(StageIVNativeFrozenOOSPlan(
            artifact=artifact, candidate_ids=panel[str(columns["candidate_id"])], symbols=panel[str(columns["symbol"])], frame=panel,
            exact_net_bps=panel[str(columns["exact_net_bps"])], decision_timestamps=panel[str(columns["decision_ts"])], label_available_timestamps=panel[str(columns["label_available_ts"])],
        ))
        manifest_artifacts[side] = {"artifact_sha256": artifact_sha, "model_artifact_manifest_sha256": model_manifest_sha, "model_files": models, "oos_panel_sha256": panel_sha}
    return StageIVNativeFrozenOOSLaunch(tuple(plans), admission, {
        "schema": FROZEN_OOS_MATERIALIZER_SCHEMA, "spec_path": str(path), "spec_sha256": file_sha256(path),
        "artifacts": manifest_artifacts, "frozen_only": True,
        "forbidden": ["fit", "hpo", "feature_selection", "cell_reselection"],
    })


__all__ = [
    "MATERIALIZER_SCHEMA", "FROZEN_OOS_MATERIALIZER_SCHEMA", "StageIVNativeLaunch", "StageIVNativeFrozenOOSLaunch", "StageIVNativeMaterializationError",
    "direct_fq3_winner_meta_fitter", "load_stage_iv_native_launch",
    "load_stage_iv_native_frozen_oos_launch", "native_winner_base_fitter",
]
