"""Run-scoped training and validation for the promoted meta postprocessor chain."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


POLICY_ID = "meta_residual_v9_tail95_market_state_mlp_hier_ev_v1"
INPUT_MANIFEST_NAME = "meta_postprocessor_inputs.json"
ARTIFACT_NAME = "composite_policy_regime_ev_calibration.json"

_INPUT_ARGS = {
    "expanded_source": "--expanded-source",
    "champion_ledger": "--champion-ledger",
    "train_oof_predictions_dir": "--train-oof-predictions-dir",
    "train_oof_rank_cache": "--train-oof-rank-cache",
    "state_artifact": "--state-artifact",
    "context_state_artifact": "--context-state-artifact",
    "parent_eval_predictions": "--parent-eval-predictions",
    "frozen_encoder_artifact": "--frozen-encoder-artifact",
}


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def meta_postprocessor_enabled(cfg: Mapping[str, Any]) -> bool:
    env = os.environ.get("EPM_META_POSTPROCESSOR_ENABLED")
    return _truthy(env) if env is not None else bool(
        cfg.get("meta_postprocessor_enabled", False)
    )


def run_scoped_postprocessor_dir(data_root: str | Path, run_id: str) -> Path:
    return Path(data_root) / "artifacts" / str(run_id) / "meta_postprocessors"


def run_scoped_postprocessor_artifact(
    data_root: str | Path, run_id: str
) -> Path | None:
    path = run_scoped_postprocessor_dir(data_root, run_id) / ARTIFACT_NAME
    return path if path.exists() else None


def _input_manifest(cfg: Mapping[str, Any], run_root: Path) -> dict[str, str]:
    manifest_path = Path(
        str(
            cfg.get("meta_postprocessor_input_manifest")
            or os.environ.get("EPM_META_POSTPROCESSOR_INPUT_MANIFEST")
            or run_root / INPUT_MANIFEST_NAME
        )
    )
    payload: dict[str, Any] = {}
    if manifest_path.exists():
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            payload.update(loaded.get("inputs") or loaded)
    for key in _INPUT_ARGS:
        cfg_key = f"meta_postprocessor_{key}"
        env_key = f"EPM_META_POSTPROCESSOR_{key.upper()}"
        value = cfg.get(cfg_key) or os.environ.get(env_key)
        if value:
            payload[key] = str(value)
    return {str(k): str(v) for k, v in payload.items() if v}


def validate_meta_postprocessor_bundle(bundle_dir: Path) -> dict[str, Any]:
    artifact_path = bundle_dir / ARTIFACT_NAME
    manifest_path = bundle_dir / "manifest.json"
    model_dir = bundle_dir / "policy_models"
    missing = [
        str(path)
        for path in (artifact_path, manifest_path, model_dir)
        if not path.exists()
    ]
    if missing:
        raise RuntimeError("incomplete meta postprocessor bundle: " + ", ".join(missing))
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(artifact.get("policy_id")) != POLICY_ID:
        raise RuntimeError(
            f"unexpected meta postprocessor policy_id={artifact.get('policy_id')!r}"
        )
    if artifact.get("predecessor") != (
        "meta_residual_extreme_local_champion_overlay_ooftrain_tieaware_downonly_"
        "20260712_v9::forced_local_tail_0.950"
    ):
        raise RuntimeError("bundle does not use the required v9 95% predecessor")
    if artifact.get("blacklisted_side_archetypes") != [
        "long||long_dirtyavoid_sparse_questionable"
    ]:
        raise RuntimeError("bundle long dirty-avoid blacklist contract is missing")
    if not artifact.get("strict_required_features"):
        raise RuntimeError("bundle must fail closed on missing inference features")
    mapping = artifact.get("expected_ev_mapping") or {}
    if mapping.get("schema") not in {
        "hierarchical_monotonic_expected_ev_v1",
        "hierarchical_monotonic_expected_ev_v2",
    }:
        raise RuntimeError("bundle expected-EV mapping is missing or invalid")
    model_paths = [
        bundle_dir / str(effect.get("model_path"))
        for effect in artifact.get("effects") or []
        if isinstance(effect, dict) and effect.get("model_path")
    ]
    absent_models = [str(path) for path in model_paths if not path.exists()]
    if absent_models:
        raise RuntimeError("bundle model sidecars are missing: " + ", ".join(absent_models))
    if manifest.get("schema") == "meta_market_state_encoder_expanding_walkforward_v1":
        parent_start = str(manifest.get("parent_meta_oos_start") or "")
        if not parent_start or parent_start > "2026-01-01":
            raise RuntimeError(
                "parent-meta OOS eligibility must start no later than 2026-01-01"
            )
        if str(manifest.get("walkforward_policy_start")) != "2026-04-01":
            raise RuntimeError("unexpected postprocessor policy OOS boundary")
        if not manifest.get("mlp_hpo_params"):
            raise RuntimeError("postprocessor manifest is missing effective MLP params")
    return {
        "policy_id": POLICY_ID,
        "artifact_path": str(artifact_path),
        "model_count": len(model_paths),
        "expected_ev_local_curves": len(mapping.get("local") or {}),
        "strict_required_features": True,
    }


def train_meta_postprocessors(
    cfg: Mapping[str, Any],
    *,
    run_id: str,
) -> dict[str, Any]:
    """Train the composite chain from explicit current-run OOF input contracts."""
    data_root = Path(str(cfg["data_root"]))
    run_root = data_root / "artifacts" / str(run_id)
    inputs = _input_manifest(cfg, run_root)
    required = set(_INPUT_ARGS) - {"context_state_artifact"}
    required.discard("frozen_encoder_artifact")
    missing = sorted(key for key in required if key not in inputs)
    if missing:
        raise RuntimeError(
            "meta postprocessor training requires explicit run-scoped inputs: "
            + ", ".join(missing)
            + f"; write {run_root / INPUT_MANIFEST_NAME}"
        )
    missing_paths = sorted(
        key for key, value in inputs.items() if not Path(value).exists()
    )
    if missing_paths:
        raise FileNotFoundError(
            "meta postprocessor input paths do not exist: " + ", ".join(missing_paths)
        )
    output_dir = run_scoped_postprocessor_dir(data_root, run_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    predecessor_dir = output_dir / "v9_tail95_predecessor"
    predecessor_command = [
        sys.executable,
        "-u",
        "scripts/run_meta_residual_extreme_local_champion_overlay.py",
        "--output-dir",
        str(predecessor_dir),
        "--train-start",
        str(cfg.get("meta_postprocessor_train_start", "2025-04-01")),
        "--train-end",
        str(
            cfg.get(
                "meta_postprocessor_predecessor_train_end",
                cfg.get("meta_postprocessor_walkforward_policy_start", "2026-04-01"),
            )
        ),
        "--eval-end",
        str(cfg.get("meta_postprocessor_eval_end", "2026-08-01")),
    ]
    for key in (
        "champion_ledger",
        "train_oof_predictions_dir",
        "train_oof_rank_cache",
        "state_artifact",
        "parent_eval_predictions",
    ):
        predecessor_command.extend([_INPUT_ARGS[key], inputs[key]])

    command = [
        sys.executable,
        "-u",
        "scripts/run_meta_market_state_encoder_ablation.py",
        "--arms",
        "mlp_direct",
        "--output-dir",
        str(output_dir),
        "--expanding-walkforward",
        "--parent-meta-oos-start",
        str(cfg.get("meta_postprocessor_parent_meta_oos_start", "2026-01-01")),
        "--walkforward-tuning-start",
        str(cfg.get("meta_postprocessor_walkforward_tuning_start", "2026-02-01")),
        "--walkforward-policy-start",
        str(cfg.get("meta_postprocessor_walkforward_policy_start", "2026-04-01")),
        "--history-train-end",
        str(
            cfg.get(
                "meta_postprocessor_history_train_end",
                cfg.get("meta_postprocessor_walkforward_policy_start", "2026-04-01"),
            )
        ),
        "--predecessor-artifact",
        str(predecessor_dir),
    ]
    params_json = cfg.get("meta_postprocessor_mlp_params_json") or os.environ.get(
        "EPM_META_POSTPROCESSOR_MLP_PARAMS_JSON"
    )
    if params_json:
        command.extend(["--mlp-params-json", str(params_json)])
    run_hpo = bool(cfg.get("meta_postprocessor_run_mlp_hpo", False)) or _truthy(
        os.environ.get("EPM_META_POSTPROCESSOR_RUN_MLP_HPO")
    )
    if run_hpo:
        command.extend(
            [
                "--run-mlp-hpo",
                "--mlp-hpo-trials",
                str(int(cfg.get("meta_postprocessor_mlp_hpo_trials", 20))),
            ]
        )
    for key, flag in _INPUT_ARGS.items():
        if key in inputs:
            command.extend([flag, inputs[key]])
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    predecessor_log_path = output_dir / "v9_tail95_training.log"
    with predecessor_log_path.open("w", encoding="utf-8") as log:
        predecessor_result = subprocess.run(
            predecessor_command,
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if predecessor_result.returncode != 0:
        raise RuntimeError(
            "v9 tail95 predecessor training failed "
            f"rc={predecessor_result.returncode}; log={predecessor_log_path}"
        )

    log_path = output_dir / "training.log"
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"meta postprocessor training failed rc={result.returncode}; log={log_path}"
        )
    validation = validate_meta_postprocessor_bundle(output_dir)
    os.environ["EPM_REGIME_EV_CALIBRATION_ARTIFACT"] = str(
        (output_dir / ARTIFACT_NAME).resolve()
    )
    os.environ["EPM_META_POSTPROCESSOR_REQUIRED"] = "1"
    pointer = {
        "schema": "meta_postprocessor_pointer_v1",
        "run_id": str(run_id),
        **validation,
        "input_manifest": inputs,
    }
    (run_root / "meta_postprocessor_pointer.json").write_text(
        json.dumps(pointer, indent=2) + "\n", encoding="utf-8"
    )
    return pointer
