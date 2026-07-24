#!/usr/bin/env python3
"""Staged search for economically useful alternatives to sequential DAE/GMM.

The default stage is ``plan``. Expensive encoder, density, base, and meta jobs
must be requested explicitly. Representation promotion remains outcome-free;
economic labels are first consumed by the frozen downstream stages.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.ae_gmm_economic_ablation import (  # noqa: E402
    add_baseline_deltas,
    economic_metrics,
    load_feature_contract,
    split_months,
)
from extreme_price_movements.alternative_latent_encoders import (  # noqa: E402
    AlternativeLatentEncoder,
    NativeLatentOutput,
)
from extreme_price_movements.alternative_representation_search import (  # noqa: E402
    EncoderCandidate,
    all_encoder_candidates,
    cap_candidates,
    encoder_config,
    evaluate_common_panel,
    fit_encoder_candidate,
    idec_final_candidates,
    materialize_representation_features,
    panel_identifier,
    robust_panel_proxy_score,
    select_family_finalists,
)
from extreme_price_movements.mutual_information_clustering import (  # noqa: E402
    FrozenEmbeddingMutualInformationClustering,
    MutualInformationClusteringConfig,
)
from extreme_price_movements.representation_proxy_metrics import (  # noqa: E402
    GmmPanelFit,
    GmmPanelSpec,
    align_diagonal_gmm_components,
    diagonal_gmm_state,
    diagonal_gmm_statistics,
    evaluate_ood_proxy,
    normalized_entropy,
    refine_diagonal_gmm,
    refinement_promotion_diagnostics,
    reorder_posteriors_to_reference,
)
from extreme_price_movements.representation_search_cache import (  # noqa: E402
    cached_side_conditioned_donor_map,
    load_reference_cache,
    prepare_reference_cache,
)
from scripts.report_s52_trailing_regime_meta_handoff import (
    run_handoff_only,  # noqa: E402
)
from scripts.run_ae_gmm_economic_ablation_matrix import (  # noqa: E402
    _load_state_fit_frame,
)
from scripts.run_materialized_trailing_label_topk_lgbm_hpo import run_hpo  # noqa: E402

DEFAULT_CONFIG = ROOT / "extreme_price_movements/config/alternative_representation_search_v1.json"
DEFAULT_INPUT_CONTRACT = ROOT / (
    "data_perp/reports/s59_h5_singlecycle_aegmm_bme_fs_fixedparams_wf30_20260716_v1/"
    "base_raw_candidate_features.csv"
)
DEFAULT_BASE_PARAMS = ROOT / "docs/promoted_s59_singlecycle_base_params.json"
DEFAULT_META_CONTRACT = ROOT / "extreme_price_movements/config/meta_v9_anchor_oldparams_residual_backbone_v1.json"
DEFAULT_META_PARAMS_MANIFEST = ROOT / (
    "data_perp/reports/meta_v9_recovery_20260717/"
    "residual_state_mda95_hier_newaegmm_hpo150_v1/"
    "staged_selection_hpo_manifest.json"
)
STAGES = (
    "plan",
    "prepare",
    "encoders",
    "proxy",
    "idec_final",
    "density1",
    "density2",
    "overlap",
    "iic",
    "base",
    "meta",
    "report",
    "all",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (Path, pd.Timestamp, pd.Period)):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != "alternative_representation_search_v1":
        raise ValueError(f"Unsupported search config: {path}")
    return payload


def _validate_downstream_label_contract(labels_path: Path) -> dict[str, Any]:
    """Reject materialized labels that allow execution on the signal candle.

    Representation selection is outcome-free, but the base/meta economic comparison
    is only meaningful when every arm uses the same causal first executable bar.
    Materialized hourly labels must therefore start execution after the signal candle
    has closed, regardless of whether an additional delayed-entry policy is enabled.
    """

    labels_path = Path(labels_path)
    manifest_path = labels_path / "labels_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Downstream representation comparisons require a materialized label "
            f"manifest: {manifest_path}"
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = payload.get("materialized_side_archetype_trailing_labels", {})
    if not isinstance(contract, Mapping):
        raise ValueError(f"Invalid materialized-label contract: {manifest_path}")
    timeframe = str(contract.get("timeframe", "")).strip()
    entry_delay_hours = contract.get("entry_delay_hours")
    path_start = str(contract.get("path_start_contract", "")).strip()
    if timeframe != "1h" or int(entry_delay_hours or 0) != 1:
        raise ValueError(
            "Representation comparison labels must use the canonical one-hour "
            f"close offset; got timeframe={timeframe!r}, "
            f"entry_delay_hours={entry_delay_hours!r}"
        )
    if path_start != "signal_timestamp_plus_timeframe_then_optional_delayed_execution":
        raise ValueError(
            "Representation comparison labels do not declare the required causal "
            f"path-start contract: {path_start!r}"
        )
    round_trip_cost = float(contract.get("round_trip_cost", float("nan")))
    if not math.isfinite(round_trip_cost) or round_trip_cost <= 0.0:
        raise ValueError(
            "Representation comparison labels must declare a positive round-trip "
            f"cost; got {round_trip_cost!r}"
        )
    return {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "timeframe": timeframe,
        "entry_delay_hours": int(entry_delay_hours),
        "path_start_contract": path_start,
        "round_trip_cost": round_trip_cost,
    }


def _candidate_manifest(
    config: Mapping[str, Any], *, max_per_family: int
) -> list[EncoderCandidate]:
    return cap_candidates(
        all_encoder_candidates(config),
        max_per_family=int(max_per_family),
        seed=int(config["seed"]),
    )


def _baseline_candidate_id(config: Mapping[str, Any]) -> str:
    controls = list(config.get("baseline_learning_curve", []))
    if not controls or not str(controls[0].get("id", "")):
        raise ValueError("Alternative representation search requires one incumbent baseline")
    return str(controls[0]["id"])


def _coverage_preserving_points(
    points: Sequence[Sequence[Any]], *, maximum: int, seed: int
) -> list[Sequence[Any]]:
    """Select grid points while covering every axis value before duplication."""
    values = list(points)
    if int(maximum) <= 0 or len(values) <= int(maximum):
        return values
    rng = np.random.default_rng(int(seed))
    tie_order = rng.permutation(len(values)).tolist()
    tie_rank = {int(index): rank for rank, index in enumerate(tie_order)}
    token_sets = [
        {(axis, json.dumps(value, sort_keys=True, default=str)) for axis, value in enumerate(point)}
        for point in values
    ]
    frequency: dict[tuple[int, str], int] = {}
    for tokens in token_sets:
        for token in tokens:
            frequency[token] = frequency.get(token, 0) + 1
    uncovered = set(frequency)
    remaining = set(range(len(values)))
    selected: list[int] = []
    while remaining and len(selected) < int(maximum):
        best = max(
            remaining,
            key=lambda index: (
                sum(
                    1.0 / max(frequency[token], 1)
                    for token in token_sets[index] & uncovered
                ),
                -tie_rank[index],
            ),
        )
        selected.append(int(best))
        remaining.remove(best)
        uncovered.difference_update(token_sets[best])
    return [values[index] for index in selected]


def _load_input_frame(args: argparse.Namespace, features: Sequence[str]) -> pd.DataFrame:
    return _load_state_fit_frame(
        labels_path=args.labels_path,
        feature_dir=args.feature_dir,
        input_features=features,
    )


def _resolve_feature_contract(args: argparse.Namespace) -> tuple[list[str], str]:
    """Recover the immutable cached feature list when an old source CSV is pruned."""

    if args.input_feature_contract.exists():
        return load_feature_contract(args.input_feature_contract), _sha256(
            args.input_feature_contract
        )
    cache_dir = args.output_root / "cache/reference"
    if not (cache_dir / "manifest.json").exists():
        raise FileNotFoundError(
            f"Feature contract is missing and no reference cache exists: {args.input_feature_contract}"
        )
    manifest, _raw, _keys = load_reference_cache(cache_dir)
    features = list(map(str, manifest.feature_names))
    recovered = cache_dir / "recovered_input_feature_contract.csv"
    if not recovered.exists():
        pd.DataFrame({"feature": features}).to_csv(recovered, index=False)
    args.base_feature_list = recovered
    return features, f"reference_cache:{manifest.feature_value_hash}"


def prepare(args: argparse.Namespace, config: Mapping[str, Any], features: Sequence[str]) -> None:
    cache_dir = args.output_root / "cache/reference"
    sample_sizes = {
        int(config["sampling"]["reference_rows"]),
        int(config["sampling"]["encoder_comparison_rows"]),
        int(config["sampling"]["ssl_proxy_rows"]),
        int(config["common_gmm_panel"]["rows"]),
        int(config["gmm_search"]["stage1"]["rows"]),
        int(config["gmm_search"]["stage2"]["rows"]),
    }
    for point in config["baseline_learning_curve"]:
        sample_sizes.add(int(point["dae_rows"]))
        sample_sizes.add(int(point["gmm_rows"]))
    cache_reused = False
    manifest = None
    if (cache_dir / "manifest.json").exists() and not args.rerun:
        cached_manifest, cached_scaled, _cached_keys = load_reference_cache(cache_dir)
        required_samples = [
            cached_manifest.sample_indices.get(str(size)) for size in sample_sizes
        ]
        cache_reused = bool(
            list(cached_manifest.feature_names) == list(map(str, features))
            and int(cached_manifest.reference_rows)
            == int(config["sampling"]["reference_rows"])
            and cached_scaled.shape
            == (int(cached_manifest.reference_rows), len(features))
            and (Path(cached_manifest.raw_path)).is_file()
            and all(path is not None and Path(path).is_file() for path in required_samples)
        )
        if cache_reused:
            manifest = cached_manifest
    if manifest is None:
        frame = _load_input_frame(args, features)
        manifest = prepare_reference_cache(
            frame,
            feature_names=features,
            output_dir=cache_dir,
            reference_rows=int(config["sampling"]["reference_rows"]),
            scaler_rows=int(config["sampling"]["encoder_comparison_rows"]),
            sample_sizes=sorted(sample_sizes),
        )
    _write_json(
        args.output_root / "cache/source_contract.json",
        {
            "schema": "alternative_representation_source_contract_v1",
            "labels_path": str(args.labels_path.resolve()),
            "feature_dir": str(args.feature_dir.resolve()),
            "input_contract": str(args.input_feature_contract.resolve()),
            "input_contract_sha256": (
                args.input_contract_sha256
                if hasattr(args, "input_contract_sha256")
                else _sha256(args.input_feature_contract)
            ),
            "reference_manifest": manifest.to_dict(),
            "reference_cache_reused": bool(cache_reused),
            "outcomes_consumed": False,
        },
    )


def _load_candidate(path: Path) -> EncoderCandidate:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return EncoderCandidate(**payload)


def _sample_indices(manifest: Any, rows: int) -> np.ndarray:
    path = manifest.sample_indices.get(str(int(rows)))
    if path is None:
        raise ValueError(f"Reference cache does not contain the {rows}-row sample")
    return np.load(path, mmap_mode="r")


def _candidate_fit_rows(candidate: EncoderCandidate, config: Mapping[str, Any]) -> int:
    if candidate.family in {"incumbent", "legacy_incumbent"}:
        return int(candidate.config.get("dae_max_train_rows", 100_000))
    if candidate.family == "ssl":
        return int(config["sampling"]["ssl_proxy_rows"])
    return int(config["sampling"]["encoder_comparison_rows"])


def _shared_encoder_key(candidate: EncoderCandidate) -> str:
    """Identity for work that is unchanged by the downstream output view."""
    values = {
        key: value
        for key, value in candidate.config.items()
        if key not in {"gmm_rows", "conditional", "proxy_parent_id"}
    }
    return json.dumps(values, sort_keys=True, default=str)


def _link_or_copy(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _idec_pretraining_cache_path(
    *,
    cache_dir: Path,
    candidate: EncoderCandidate,
    reference_contract: Mapping[str, Any],
    fit_rows: int,
    seed: int,
    device: str,
) -> Path:
    """Key only the reconstruction architecture and immutable input contract."""
    cfg = encoder_config(candidate, seed=int(seed), device=str(device))
    payload = {
        "schema": "idec_pretraining_cache_v1",
        "reference": dict(reference_contract),
        "fit_rows": int(fit_rows),
        "architecture": {
            "latent_dim": int(cfg.latent_dim),
            "hidden_dim": int(cfg.hidden_dim),
            "residual_blocks": int(cfg.residual_blocks),
            "epochs": int(cfg.epochs),
            "pretrain_epochs": int(cfg.pretrain_epochs),
            "pretraining_fraction": cfg.pretraining_fraction,
            "batch_size": int(cfg.batch_size),
            "learning_rate": float(cfg.learning_rate),
            "weight_decay": float(cfg.weight_decay),
            "random_state": int(cfg.random_state),
        },
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return cache_dir / f"{digest}.joblib"


def fit_encoders(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    candidates: Sequence[EncoderCandidate],
) -> None:
    manifest, raw, keys = load_reference_cache(args.output_root / "cache/reference")
    raw_values = np.load(manifest.raw_path, mmap_mode="r")
    sides = keys["side"].to_numpy(copy=False)
    reference_contract = {
        "row_identity_hash": manifest.row_identity_hash,
        "feature_value_hash": manifest.feature_value_hash,
        "feature_names": list(manifest.feature_names),
    }
    reusable: dict[str, Path] = {}
    fit_value_cache: dict[int, np.ndarray] = {}
    fit_side_cache: dict[int, np.ndarray] = {}
    pretraining_cache_dir = args.output_root / "cache" / "idec_pretraining"
    for candidate in candidates:
        candidate_dir = args.output_root / "encoders" / candidate.candidate_id
        reuse_key = _shared_encoder_key(candidate)
        if (candidate_dir / "encoder.joblib").exists() and not args.rerun:
            contract_path = candidate_dir / "reference_contract.json"
            if not contract_path.exists() or json.loads(contract_path.read_text(encoding="utf-8")) != reference_contract:
                raise ValueError(
                    f"Cached encoder {candidate.candidate_id} does not match the active reference cache"
                )
            reusable.setdefault(reuse_key, candidate_dir)
            continue
        frozen_state = str(candidate.config.get("frozen_encoder_state_path", ""))
        if frozen_state:
            state_path = Path(frozen_state)
            if not state_path.exists():
                raise FileNotFoundError(f"Frozen incumbent encoder is missing: {state_path}")
            state = joblib.load(state_path)
            encoder = AlternativeLatentEncoder.from_state(state, device=args.device)
            if encoder.feature_center.size != raw_values.shape[1]:
                raise ValueError(
                    "Frozen incumbent feature width does not match the active reference frame"
                )
            native = encoder.transform_native(raw_values, sides=sides)
            candidate_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(encoder.to_state(), candidate_dir / "encoder.joblib", compress=3)
            np.savez(
                candidate_dir / "native_outputs.npz",
                latent=np.asarray(native.latent, dtype=np.float32),
                reconstruction_error=(
                    np.empty((0,), dtype=np.float32)
                    if native.reconstruction_error is None
                    else np.asarray(native.reconstruction_error, dtype=np.float32)
                ),
                cluster_probabilities=(
                    np.empty((0,), dtype=np.float32)
                    if native.cluster_probabilities is None
                    else np.asarray(native.cluster_probabilities, dtype=np.float32)
                ),
                mean=(
                    np.empty((0,), dtype=np.float32)
                    if native.mean is None
                    else np.asarray(native.mean, dtype=np.float32)
                ),
                logvar=(
                    np.empty((0,), dtype=np.float32)
                    if native.logvar is None
                    else np.asarray(native.logvar, dtype=np.float32)
                ),
            )
            _write_json(candidate_dir / "candidate.json", candidate.to_dict())
            _write_json(candidate_dir / "reference_contract.json", reference_contract)
            source_contract = Path(
                str(candidate.config.get("source_reference_contract_path", ""))
            )
            _write_json(
                candidate_dir / "frozen_encoder_provenance.json",
                {
                    "source_encoder_state": str(state_path.resolve()),
                    "source_encoder_sha256": _sha256(state_path),
                    "source_reference_contract": (
                        str(source_contract.resolve()) if source_contract.exists() else None
                    ),
                    "source_reference_contract_sha256": (
                        _sha256(source_contract) if source_contract.exists() else None
                    ),
                    "active_reference_contract": reference_contract,
                    "encoder_retrained": False,
                    "transformed_on_active_reference": True,
                },
            )
            reusable[reuse_key] = candidate_dir
            del encoder, native
            gc.collect()
            continue
        source = reusable.get(reuse_key)
        if source is not None and not args.rerun:
            candidate_dir.mkdir(parents=True, exist_ok=True)
            for name in (
                "encoder.joblib",
                "native_outputs.npz",
                "native_latent.npy",
                "native_reconstruction_error.npy",
                "native_cluster_probabilities.npy",
                "native_mean.npy",
                "native_logvar.npy",
            ):
                if not (source / name).exists():
                    continue
                destination = candidate_dir / name
                _link_or_copy(source / name, destination)
            _write_json(candidate_dir / "candidate.json", candidate.to_dict())
            _write_json(candidate_dir / "reference_contract.json", reference_contract)
            continue
        fit_rows = _candidate_fit_rows(candidate, config)
        indices = _sample_indices(manifest, fit_rows)
        if int(fit_rows) not in fit_value_cache:
            fit_value_cache[int(fit_rows)] = np.ascontiguousarray(
                raw_values[indices], dtype=np.float32
            )
            fit_side_cache[int(fit_rows)] = np.asarray(sides[indices]).copy()
        fit_values = fit_value_cache[int(fit_rows)]
        fit_sides = fit_side_cache[int(fit_rows)]
        donor_regimes = None
        if str(candidate.config.get("donor_policy", "none")) == "causal_compatible_group":
            regime_columns = (
                manifest.feature_groups.get("regime_source", ())
                or manifest.feature_groups.get("market_context", ())
            )
            if not regime_columns:
                raise ValueError("Conditional donor augmentation requires a frozen regime-source feature")
            values = np.asarray(fit_values[:, int(regime_columns[0])], dtype=np.float64)
            finite = values[np.isfinite(values)]
            if not len(finite):
                raise ValueError("Conditional donor regime feature contains no finite values")
            cuts = np.unique(np.quantile(finite, [0.2, 0.4, 0.6, 0.8]))
            donor_regimes = np.digitize(values, cuts).astype(np.int8)
        idec_pretraining_state = None
        if candidate.family == "idec":
            pretraining_path = _idec_pretraining_cache_path(
                cache_dir=pretraining_cache_dir,
                candidate=candidate,
                reference_contract=reference_contract,
                fit_rows=int(fit_rows),
                seed=int(config["seed"]),
                device=args.device,
            )
            if pretraining_path.exists() and not args.rerun:
                idec_pretraining_state = joblib.load(pretraining_path)
            else:
                pretraining_path.parent.mkdir(parents=True, exist_ok=True)
                pretrainer = AlternativeLatentEncoder(
                    encoder_config(candidate, seed=int(config["seed"]), device=args.device)
                )
                idec_pretraining_state = pretrainer.fit_idec_pretraining_state(
                    fit_values,
                    sides=fit_sides,
                    feature_group_indices=manifest.feature_groups,
                )
                joblib.dump(idec_pretraining_state, pretraining_path, compress=3)
        fit_encoder_candidate(
            candidate,
            fit_values=fit_values,
            fit_sides=fit_sides,
            transform_values=raw_values,
            transform_sides=sides,
            feature_group_indices=manifest.feature_groups,
            fit_donor_regime_labels=donor_regimes,
            idec_pretraining_state=idec_pretraining_state,
            output_dir=candidate_dir,
            seed=int(config["seed"]),
            device=args.device,
        )
        _write_json(candidate_dir / "reference_contract.json", reference_contract)
        reusable[reuse_key] = candidate_dir
        gc.collect()


def _restore_candidate_encoder(candidate_dir: Path, device: str) -> AlternativeLatentEncoder:
    state = joblib.load(candidate_dir / "encoder.joblib")
    return AlternativeLatentEncoder.from_state(state, device=device)


def _load_native(candidate_dir: Path) -> NativeLatentOutput:
    latent_path = candidate_dir / "native_latent.npy"
    if latent_path.exists():
        def optional_mmap(name: str) -> np.ndarray | None:
            path = candidate_dir / f"native_{name}.npy"
            if not path.exists():
                return None
            values = np.load(path, mmap_mode="r")
            return None if values.size == 0 else values

        return NativeLatentOutput(
            latent=np.load(latent_path, mmap_mode="r"),
            reconstruction_error=optional_mmap("reconstruction_error"),
            cluster_probabilities=optional_mmap("cluster_probabilities"),
            mean=optional_mmap("mean"),
            logvar=optional_mmap("logvar"),
        )
    with np.load(candidate_dir / "native_outputs.npz", allow_pickle=False) as data:
        def optional(name: str) -> np.ndarray | None:
            values = data[name]
            return None if values.size == 0 else values.astype(np.float32, copy=False)

        return NativeLatentOutput(
            latent=data["latent"].astype(np.float32, copy=False),
            reconstruction_error=optional("reconstruction_error"),
            cluster_probabilities=optional("cluster_probabilities"),
            mean=optional("mean"),
            logvar=optional("logvar"),
        )


def run_proxy(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    candidates: Sequence[EncoderCandidate],
    append_existing: bool = False,
) -> pd.DataFrame:
    manifest, _scaled, keys = load_reference_cache(args.output_root / "cache/reference")
    raw = np.load(manifest.raw_path, mmap_mode="r")
    scale = np.load(manifest.scale_path, mmap_mode="r")
    fit_indices = _sample_indices(manifest, int(config["sampling"]["encoder_comparison_rows"]))
    rng = np.random.default_rng(int(config["seed"]) + 311)
    perturb_contract_path = args.output_root / "cache/reference/perturbation_contract.json"
    perturb_contract = {
        "row_identity_hash": manifest.row_identity_hash,
        "feature_value_hash": manifest.feature_value_hash,
        "seed": int(config["seed"]),
        "mild_noise_std_robust_units": 0.01,
    }
    cached_perturb_contract = (
        json.loads(perturb_contract_path.read_text(encoding="utf-8"))
        if perturb_contract_path.exists()
        else None
    )
    rebuild_perturbations = bool(args.rerun or cached_perturb_contract != perturb_contract)
    mild_path = args.output_root / "cache/reference/mild_synthetic.npy"
    if not mild_path.exists() or rebuild_perturbations:
        mild_out = np.lib.format.open_memmap(
            mild_path, mode="w+", dtype=np.float32, shape=raw.shape
        )
        for start in range(0, len(raw), 20_000):
            stop = min(start + 20_000, len(raw))
            block = np.asarray(raw[start:stop], dtype=np.float32).copy()
            block += rng.normal(0.0, 0.01, block.shape).astype(np.float32) * scale.reshape(1, -1)
            mild_out[start:stop] = block
        mild_out.flush()
        del mild_out
    harmless = np.load(mild_path, mmap_mode="r")
    donor_map = cached_side_conditioned_donor_map(
        keys["side"].to_numpy(copy=False),
        output_path=args.output_root / "cache/reference/structural_donor_map.npy",
        seed=int(config["seed"]) + 317,
    )
    structural_group = max(
        (indices for name, indices in manifest.feature_groups.items() if name != "side"),
        key=len,
    )
    structural_path = args.output_root / "cache/reference/structural_synthetic.npy"
    if not structural_path.exists() or rebuild_perturbations:
        structural_out = np.lib.format.open_memmap(
            structural_path, mode="w+", dtype=np.float32, shape=raw.shape
        )
        columns = np.asarray(structural_group, dtype=np.int64)
        for start in range(0, len(raw), 20_000):
            stop = min(start + 20_000, len(raw))
            block = np.asarray(raw[start:stop], dtype=np.float32).copy()
            if len(columns):
                local_donors = donor_map[start:stop]
                block[:, columns] = raw[local_donors[:, None], columns[None, :]]
            structural_out[start:stop] = block
        structural_out.flush()
        del structural_out
    structural = np.load(structural_path, mmap_mode="r")
    _write_json(perturb_contract_path, perturb_contract)
    panel_config = config["common_gmm_panel"]
    specs = [
        GmmPanelSpec(int(k), "diag", float(panel_config["reg_covar"][0]))
        for k in panel_config["components"]
    ]
    months = pd.to_datetime(keys["__ts__"], utc=True).dt.to_period("M").astype(str)
    strata = {
        "symbol": keys["__symbol__"].astype(str).to_numpy(),
        "calendar_period": months.to_numpy(),
        "side": keys["side"].astype(str).to_numpy(),
    }
    regime_indices = [
        *manifest.feature_groups.get("regime_source", ()),
        *manifest.feature_groups.get("market_context", ()),
    ]
    if regime_indices:
        regime_values = np.asarray(raw[:, int(regime_indices[0])], dtype=np.float64)
        finite = regime_values[np.isfinite(regime_values)]
        if len(finite):
            cuts = np.unique(np.quantile(finite, [0.2, 0.4, 0.6, 0.8]))
            strata["major_market_regime"] = np.digitize(regime_values, cuts).astype(str)
    all_panel_reports: list[pd.DataFrame] = []
    ood_reports: dict[str, dict[str, Any]] = {}
    rejected_candidates: list[dict[str, Any]] = []
    proxy_reusable: dict[str, Path] = {}
    for candidate in candidates:
        candidate_dir = args.output_root / "encoders" / candidate.candidate_id
        proxy_path = candidate_dir / "proxy_panel.csv"
        perturb_latent: np.ndarray | None = None
        proxy_key = _shared_encoder_key(candidate)
        proxy_source = proxy_reusable.get(proxy_key)
        if proxy_source is not None and not args.rerun and not proxy_path.exists():
            for name in ("proxy_panel.csv", "common_panel.joblib", "ood_proxy.json"):
                _link_or_copy(proxy_source / name, candidate_dir / name)
        if proxy_path.exists() and not args.rerun:
            report = pd.read_csv(proxy_path)
        else:
            encoder = _restore_candidate_encoder(candidate_dir, args.device)
            native = _load_native(candidate_dir)
            local_fit_indices = fit_indices
            if candidate.family in {"incumbent", "legacy_incumbent"}:
                local_fit_indices = _sample_indices(
                    manifest, int(candidate.config.get("gmm_rows", len(fit_indices)))
                )
            nearby_indices: list[np.ndarray] = []
            resample_rng = np.random.default_rng(
                int(config["seed"]) + 100_003 + int(len(local_fit_indices))
            )
            for _repeat in range(int(config["proxy"].get("nearby_reference_resamples", 0))):
                jitter = resample_rng.integers(
                    -2, 3, size=len(local_fit_indices), dtype=np.int64
                )
                nearby_indices.append(
                    np.unique(
                        np.clip(
                            np.asarray(local_fit_indices, dtype=np.int64) + jitter,
                            0,
                            len(raw) - 1,
                        )
                    )
                )
            perturb_latent = encoder.transform(harmless, sides=keys["side"].to_numpy())
            try:
                fits, report = evaluate_common_panel(
                    native.latent,
                    fit_indices=local_fit_indices,
                    perturb_latent=perturb_latent,
                    strata=strata,
                    seeds=[int(config["seed"]) + value for value in range(int(config["proxy"]["seed_repeats"]))],
                    specs=specs,
                    n_init=int(panel_config["n_init"]),
                    nearby_fit_indices=nearby_indices,
                    retry_reg_covars=tuple(
                        panel_config.get("retry_reg_covar", (0.01, 0.03, 0.1))
                    ),
                )
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                rejection = {
                    "candidate_id": candidate.candidate_id,
                    "family": candidate.family,
                    "status": "rejected_proxy_density_unstable",
                    "error": str(exc),
                }
                rejected_candidates.append(rejection)
                _write_json(candidate_dir / "proxy_rejection.json", rejection)
                del native, encoder
                gc.collect()
                continue
            fit_failures = list(report.attrs.get("fit_failures", []))
            if fit_failures:
                _write_json(candidate_dir / "proxy_fit_warnings.json", {"failures": fit_failures})
            joblib.dump(fits, candidate_dir / "common_panel.joblib", compress=3)
            report.to_csv(proxy_path, index=False)
        proxy_reusable.setdefault(proxy_key, candidate_dir)
        report = report.copy()
        report["candidate_id"] = candidate.candidate_id
        report["family"] = candidate.family
        all_panel_reports.append(report)

        fits = joblib.load(candidate_dir / "common_panel.joblib")
        provisional = report.sort_values("proxy_score", ascending=False).iloc[0]
        panel = next(item for item in fits if panel_identifier(item) == provisional["panel_id"])
        encoder = _restore_candidate_encoder(candidate_dir, args.device)
        native = _load_native(candidate_dir)
        clean_stats = diagonal_gmm_statistics(native.latent, panel.state)
        structural_latent = encoder.transform(structural, sides=keys["side"].to_numpy())
        structural_stats = diagonal_gmm_statistics(structural_latent, panel.state)
        del structural_latent
        if perturb_latent is None:
            perturb_latent = encoder.transform(
                harmless, sides=keys["side"].to_numpy()
            )
        mild_stats = diagonal_gmm_statistics(
            perturb_latent, panel.state
        )
        del perturb_latent
        chronological = np.argsort(pd.to_datetime(keys["__ts__"], utc=True).view("int64"))
        split = max(1, int(0.8 * len(chronological)))
        later = chronological[split:]
        earlier = chronological[:split]
        ood = evaluate_ood_proxy(
            clean_stats["ood_score"][later],
            mild_stats["ood_score"][later],
            structural_stats["ood_score"][later],
            clean_stats["ood_score"][earlier[-max(1, len(later)):]],
            calibration_scores=clean_stats["ood_score"][earlier],
            aligned_clean_scores={
                "mild_synthetic": clean_stats["ood_score"][later],
                "structural_synthetic": clean_stats["ood_score"][later],
                "natural_temporal": None,
            },
        )
        ood_reports[candidate.candidate_id] = ood
        _write_json(candidate_dir / "ood_proxy.json", ood)
        del clean_stats, structural_stats, mild_stats, native, encoder
        gc.collect()

    if not all_panel_reports:
        return pd.DataFrame()
    panel_report = pd.concat(all_panel_reports, ignore_index=True)
    combined_panel_path = args.output_root / "proxy_panel_all_candidates.csv"
    if append_existing and combined_panel_path.exists():
        previous_panel = pd.read_csv(combined_panel_path)
        panel_report = pd.concat([previous_panel, panel_report], ignore_index=True)
        panel_report = panel_report.drop_duplicates(
            ["candidate_id", "panel_id"], keep="last"
        )
    # Rank every panel point against the other points in the same encoder
    # family. This keeps the proxy comparable across candidates; per-candidate
    # ranks would make every three-point panel look equally strong.
    panel_report["proxy_score"] = np.nan
    for _family, indices in panel_report.groupby("family", observed=True).groups.items():
        panel_report.loc[indices, "proxy_score"] = robust_panel_proxy_score(
            panel_report.loc[indices]
        )
    panel_report.to_csv(combined_panel_path, index=False)
    summaries: list[dict[str, Any]] = []
    summary_candidates = list(candidates)
    if append_existing:
        summary_candidates = [
            _load_candidate(path)
            for path in sorted((args.output_root / "encoders").glob("*/candidate.json"))
        ]
    for candidate in summary_candidates:
        local = panel_report.loc[panel_report["candidate_id"].eq(candidate.candidate_id)]
        if local.empty:
            continue
        best = local.sort_values("proxy_score", ascending=False).iloc[0]
        ood = ood_reports.get(candidate.candidate_id)
        if ood is None:
            ood = json.loads(
                (args.output_root / "encoders" / candidate.candidate_id / "ood_proxy.json").read_text(
                    encoding="utf-8"
                )
            )
        score_sets = ood["score_sets"]
        synthetic_mean_monotonic = bool(
            float(score_sets["clean_untouched_later"]["mean"])
            <= float(score_sets["mild_synthetic"]["mean"])
            <= float(score_sets["structural_synthetic"]["mean"])
        )
        synthetic_median_monotonic = bool(
            float(score_sets["clean_untouched_later"]["median"])
            <= float(score_sets["mild_synthetic"]["median"])
            <= float(score_sets["structural_synthetic"]["median"])
        )
        summaries.append(
            {
                "candidate_id": candidate.candidate_id,
                "family": candidate.family,
                "stage": candidate.stage,
                "output_mode": candidate.output_mode,
                "best_panel_id": str(best["panel_id"]),
                "panel_proxy_score": float(best["proxy_score"]),
                "ood_proxy_score": float(
                    0.20 * synthetic_mean_monotonic
                    + 0.20 * synthetic_median_monotonic
                    + 0.25
                    * float(
                        ood["clean_corrupted_separation"]["structural_synthetic"][
                            "probability_corrupted_gt_clean"
                        ]
                    )
                    + 0.15
                    * float(
                        ood["clean_corrupted_separation"]["mild_synthetic"][
                            "probability_corrupted_gt_clean"
                        ]
                    )
                    + 0.20
                    * float(
                        ood["clean_corrupted_separation"]["natural_temporal"][
                            "probability_corrupted_gt_clean"
                        ]
                    )
                    - 0.20
                    * float(
                        ood["untouched_later_false_positive_rate"]["elevated_or_extreme"]
                    )
                ),
                "best_panel_components": int(best["n_components"]),
                "best_panel_seed_instability": float(best["seed_excess_instability"]),
                "best_panel_perturb_tv": float(best["posterior_perturb_tv"]),
                "best_panel_ood_rank_consistency": float(best["ood_rank_consistency"]),
                "ood_mean_monotonic": synthetic_mean_monotonic,
                "ood_untouched_later_fpr": float(
                    ood["untouched_later_false_positive_rate"]["elevated_or_extreme"]
                ),
            }
        )
    candidate_summary = pd.DataFrame(summaries)
    candidate_summary["best_robust_panel_score"] = np.nan
    for _family, indices in candidate_summary.groupby("family", observed=True).groups.items():
        local = candidate_summary.loc[indices]
        panel_rank = local["panel_proxy_score"].rank(pct=True, method="average")
        ood_rank = local["ood_proxy_score"].rank(pct=True, method="average")
        candidate_summary.loc[indices, "best_robust_panel_score"] = (
            0.85 * panel_rank + 0.15 * ood_rank
        )
    summary = select_family_finalists(
        candidate_summary,
        top_per_family=int(config["proxy"]["promote_per_family"]),
    )
    summary.to_csv(args.output_root / "proxy_candidate_summary.csv", index=False)
    if rejected_candidates:
        pd.DataFrame(rejected_candidates).to_csv(
            args.output_root / "proxy_rejected_candidates.csv", index=False
        )
    _write_saturation_report(args, config, summary)
    return summary


def run_idec_final(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    summary = pd.read_csv(args.output_root / "proxy_candidate_summary.csv")
    proxy_ids = summary.loc[
        summary["family"].eq("idec")
        & summary["stage"].eq("proxy")
        & summary["promoted"].fillna(False).astype(bool),
        "candidate_id",
    ].head(int(config["proxy"]["promote_per_family"]))
    parents = [
        _load_candidate(args.output_root / "encoders" / candidate_id / "candidate.json")
        for candidate_id in proxy_ids
    ]
    finalists = cap_candidates(
        idec_final_candidates(config, parents),
        max_per_family=int(args.max_idec_final_candidates),
        seed=int(config["seed"]) + 41,
    )
    fit_encoders(args, config, finalists)
    run_proxy(args, config, finalists, append_existing=True)


def _linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    x -= x.mean(axis=0, keepdims=True)
    y -= y.mean(axis=0, keepdims=True)
    cross = np.linalg.norm(x.T @ y, ord="fro") ** 2
    denom = np.linalg.norm(x.T @ x, ord="fro") * np.linalg.norm(y.T @ y, ord="fro")
    return float(cross / max(denom, 1e-12))


def _write_saturation_report(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    summary: pd.DataFrame,
) -> bool:
    incumbent = summary.loc[summary["family"].eq("incumbent")]
    if incumbent.empty:
        return False
    rows: list[dict[str, Any]] = []
    native_by_id = {
        row.candidate_id: _load_native(args.output_root / "encoders" / row.candidate_id)
        for row in incumbent.itertuples()
    }
    control_ids = [str(point["id"]) for point in config["baseline_learning_curve"]]
    comparisons = list(zip(control_ids[:-1], control_ids[1:]))
    for low_id, high_id in comparisons:
        if low_id not in native_by_id or high_id not in native_by_id:
            continue
        low, high = native_by_id[low_id], native_by_id[high_id]
        low_recon = float(np.mean(low.reconstruction_error)) if low.reconstruction_error is not None else np.nan
        high_recon = float(np.mean(high.reconstruction_error)) if high.reconstruction_error is not None else np.nan
        latent_cka = _linear_cka(low.latent, high.latent)
        rows.append(
            {
                "from": low_id,
                "to": high_id,
                "relative_reconstruction_gain": float((low_recon - high_recon) / max(abs(low_recon), 1e-12)),
                "latent_linear_cka": latent_cka,
                "latent_representation_change": 1.0 - latent_cka,
                "proxy_score_delta": float(
                    incumbent.set_index("candidate_id").loc[high_id, "best_robust_panel_score"]
                    - incumbent.set_index("candidate_id").loc[low_id, "best_robust_panel_score"]
                ),
            }
        )
    report = pd.DataFrame(rows)
    report.to_csv(args.output_root / "baseline_saturation.csv", index=False)
    if not report.empty:
        last = report.iloc[-1]
        material = bool(
            float(last["relative_reconstruction_gain"])
            >= float(config["baseline_saturation"]["min_relative_reconstruction_gain"])
            or float(last["proxy_score_delta"])
            >= float(config["baseline_saturation"]["min_gmm_stability_gain"])
        )
        _write_json(
            args.output_root / "baseline_saturation_decision.json",
            {
                "scaled_incumbent_control_material": material,
                "criteria": config["baseline_saturation"],
                "comparisons": report.to_dict("records"),
            },
        )
        return material
    return False


def _dae_gmm_input(native: NativeLatentOutput) -> tuple[np.ndarray, dict[str, Any]]:
    """Build the observable density input for a standard denoising AE.

    A DAE exposes a deterministic latent representation and reconstruction
    error, but not VAE posterior moments or ELBO.  Reconstruction error is a
    genuine novelty coordinate: a state can be close in latent direction while
    still being poorly reconstructed.  Use ``log1p`` before the fitted latent
    transform so rare reconstruction tails do not dominate the GMM geometry.
    """

    latent = np.asarray(native.latent, dtype=np.float32)
    if latent.ndim != 2 or latent.shape[1] < 2:
        raise ValueError("DAE latent output must be a non-trivial 2D matrix")
    error = native.reconstruction_error
    if error is None or len(error) != len(latent):
        raise ValueError("DAE/GMM density input requires aligned reconstruction error")
    novelty = np.log1p(
        np.maximum(np.asarray(error, dtype=np.float32).reshape(-1), 0.0)
    ).reshape(-1, 1)
    values = np.ascontiguousarray(np.concatenate([latent, novelty], axis=1), dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("DAE/GMM density input contains non-finite values")
    return values, {
        "schema": "dae_latent_plus_reconstruction_novelty_v1",
        "latent_dim": int(latent.shape[1]),
        "derivative_columns": ["reconstruction_error_log1p"],
    }


def _preprocess_latent(
    fit: np.ndarray, all_values: np.ndarray, mode: str
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    fit64 = np.asarray(fit, dtype=np.float64)
    all64 = np.asarray(all_values, dtype=np.float64)
    if mode == "raw":
        return fit64, all64, {"mode": "raw"}
    center = fit64.mean(axis=0)
    centered_fit = fit64 - center
    centered_all = all64 - center
    if mode == "standardized":
        scale = np.maximum(centered_fit.std(axis=0), 1e-8)
        return centered_fit / scale, centered_all / scale, {
            "mode": mode, "center": center, "scale": scale
        }
    if mode == "whitened":
        covariance = np.cov(centered_fit, rowvar=False)
        values, vectors = np.linalg.eigh(np.atleast_2d(covariance))
        transform = vectors @ np.diag(1.0 / np.sqrt(np.maximum(values, 1e-8))) @ vectors.T
        return centered_fit @ transform, centered_all @ transform, {
            "mode": mode, "center": center, "transform": transform
        }
    raise ValueError(f"Unknown latent preprocessing: {mode}")


def _apply_latent_transform(values: np.ndarray, state: Mapping[str, Any]) -> np.ndarray:
    transformed = np.asarray(values, dtype=np.float64)
    mode = str(state["mode"])
    if mode == "raw":
        return transformed
    centered = transformed - np.asarray(state["center"], dtype=np.float64)
    if mode == "standardized":
        return centered / np.asarray(state["scale"], dtype=np.float64)
    if mode == "whitened":
        return centered @ np.asarray(state["transform"], dtype=np.float64)
    raise ValueError(f"Unknown serialized latent preprocessing: {mode}")


def _stage2_promoted_embeddings(
    args: argparse.Namespace,
    summary: pd.DataFrame,
    *,
    minimum: int,
    maximum: int,
) -> pd.DataFrame:
    stage1_path = args.output_root / "density_stage1_summary.csv"
    if not stage1_path.exists():
        return summary
    stage1 = pd.read_csv(stage1_path).sort_values(
        ["family", "density_proxy_score", "density_id"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    best_per_embedding = stage1.drop_duplicates("candidate_id", keep="first")
    family_count = max(1, int(best_per_embedding["family"].nunique()))
    per_family = max(1, int(maximum) // family_count)
    selected_rows = best_per_embedding.groupby(
        "family", observed=True, group_keys=False
    ).head(per_family)
    if len(selected_rows) < int(minimum):
        remainder = best_per_embedding.loc[
            ~best_per_embedding["candidate_id"].isin(selected_rows["candidate_id"])
        ].head(int(minimum) - len(selected_rows))
        selected_rows = pd.concat([selected_rows, remainder], ignore_index=True)
    selected = selected_rows.head(int(maximum))[["candidate_id"]]
    return summary.merge(selected, on="candidate_id", how="inner")


def run_density_search(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    *,
    stage: int,
) -> pd.DataFrame:
    summary = pd.read_csv(args.output_root / "proxy_candidate_summary.csv")
    promoted = summary.loc[summary["promoted"].fillna(False).astype(bool)]
    section = config["gmm_search"][f"stage{stage}"]
    if int(stage) == 2:
        promoted = _stage2_promoted_embeddings(
            args,
            promoted,
            minimum=int(section.get("promoted_candidates_min", 6)),
            maximum=int(section.get("promoted_candidates_max", 10)),
        )
    manifest, _scaled, _keys = load_reference_cache(args.output_root / "cache/reference")
    fit_indices = _sample_indices(manifest, int(section["rows"]))
    rows: list[dict[str, Any]] = []
    for candidate_row in promoted.itertuples():
        candidate_dir = args.output_root / "encoders" / candidate_row.candidate_id
        native = _load_native(candidate_dir)
        density_input, density_input_schema = _dae_gmm_input(native)
        for mode in section["latent_preprocessing"]:
            fit_values, transformed, transform_state = _preprocess_latent(
                density_input[fit_indices], density_input, str(mode)
            )
            transform_state["density_input_schema"] = density_input_schema
            heldout = np.setdiff1d(np.arange(len(transformed)), fit_indices, assume_unique=False)
            heldout = heldout[-min(len(heldout), 50_000) :]
            evaluation_indices = np.unique(
                np.concatenate(
                    [
                        np.asarray(fit_indices[: min(50_000, len(fit_indices))]),
                        np.asarray(heldout),
                    ]
                )
            )
            evaluation_values = transformed[evaluation_indices]
            for covariance, components, reg in itertools.product(
                section["covariance_types"],
                section["components"],
                section["reg_covar"],
            ):
                density_id = f"{candidate_row.candidate_id}__s{stage}__{mode}__{covariance}__k{components}__r{reg:g}"
                density_dir = args.output_root / f"density_stage{stage}" / density_id
                model_path = density_dir / "gmm.joblib"
                if model_path.exists() and not args.rerun:
                    model = joblib.load(model_path)
                else:
                    density_dir.mkdir(parents=True, exist_ok=True)
                    model = GaussianMixture(
                        n_components=int(components),
                        covariance_type=str(covariance),
                        reg_covar=float(reg),
                        n_init=int(section["n_init"]),
                        random_state=int(config["seed"]),
                        max_iter=300,
                    ).fit(fit_values)
                    joblib.dump(model, model_path, compress=3)
                    joblib.dump(
                        transform_state,
                        density_dir / "latent_transform.joblib",
                        compress=3,
                    )
                probability = model.predict_proba(evaluation_values).astype(np.float32)
                entropy = normalized_entropy(probability)
                rows.append(
                    {
                        "candidate_id": str(candidate_row.candidate_id),
                        "family": str(candidate_row.family),
                        "density_id": density_id,
                        "stage": int(stage),
                        "components": int(components),
                        "covariance_type": str(covariance),
                        "reg_covar": float(reg),
                        "latent_preprocessing": str(mode),
                        "heldout_nll": float(-model.score(transformed[heldout]))
                        if len(heldout)
                        else float(-model.score(evaluation_values)),
                        "min_occupancy": float(probability.mean(axis=0).min()),
                        "entropy_mass_near_zero": float(np.mean(entropy <= 0.05)),
                        "entropy_mass_near_one": float(np.mean(entropy >= 0.95)),
                        "converged": bool(model.converged_),
                    }
                )
        gc.collect()
    report = pd.DataFrame(rows)
    if not report.empty:
        report["density_proxy_score"] = (
            report.groupby("family", observed=True)["heldout_nll"].rank(pct=True, ascending=False)
            + report.groupby("family", observed=True)["min_occupancy"].rank(pct=True, ascending=True)
            - 0.5 * report.groupby("family", observed=True)["entropy_mass_near_zero"].rank(pct=True, ascending=True)
            - 0.5 * report.groupby("family", observed=True)["entropy_mass_near_one"].rank(pct=True, ascending=True)
        )
        report["family_rank"] = report.groupby("family", observed=True)["density_proxy_score"].rank(method="first", ascending=False)
    report.to_csv(args.output_root / f"density_stage{stage}_summary.csv", index=False)
    return report


def run_overlap_refinement(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    path = args.output_root / "density_stage2_summary.csv"
    if not path.exists():
        return
    summary = pd.read_csv(path)
    proxy_modes = pd.read_csv(args.output_root / "proxy_candidate_summary.csv").loc[
        :, ["candidate_id", "output_mode"]
    ].drop_duplicates("candidate_id")
    summary = summary.merge(
        proxy_modes, on="candidate_id", how="left", validate="many_to_one"
    )
    overlap = config["gmm_search"]["overlap_refinement"]
    candidates = (
        summary.loc[
            summary["covariance_type"].eq("diag")
            & summary["output_mode"].fillna("embedding_gmm").eq("embedding_gmm")
        ]
        .sort_values(
            ["family", "density_proxy_score", "density_id"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        .groupby("family", observed=True, group_keys=False)
        .head(int(overlap.get("candidates_per_family", 2)))
    )
    requested_source_ids = list(getattr(args, "overlap_source_density_ids", []))
    if requested_source_ids:
        available = summary.set_index("density_id", drop=False)
        missing = [density_id for density_id in requested_source_ids if density_id not in available.index]
        if missing:
            raise ValueError(
                "Requested overlap source density IDs are unavailable: " + ", ".join(missing)
            )
        candidates = available.loc[requested_source_ids]
        if isinstance(candidates, pd.Series):
            candidates = candidates.to_frame().T
        candidates = candidates.reset_index(drop=True)
        unsupported = candidates.loc[~candidates["covariance_type"].eq("diag"), "density_id"]
        if not unsupported.empty:
            raise ValueError(
                "Bhattacharyya refinement currently supports diagonal GMMs only; "
                "requested tied source: " + ", ".join(unsupported.astype(str))
            )
    manifest, _scaled, _keys = load_reference_cache(args.output_root / "cache/reference")
    density_fit_indices = np.asarray(
        _sample_indices(manifest, int(config["gmm_search"]["stage2"]["rows"])),
        dtype=np.int64,
    )
    rows: list[dict[str, Any]] = []
    for row in candidates.itertuples():
        density_dir = args.output_root / "density_stage2" / row.density_id
        model = joblib.load(density_dir / "gmm.joblib")
        transform_state = joblib.load(density_dir / "latent_transform.joblib")
        native = _load_native(args.output_root / "encoders" / row.candidate_id)
        density_input, density_input_schema = _dae_gmm_input(native)
        if transform_state.get("density_input_schema") not in (None, density_input_schema):
            raise ValueError("Overlap refinement has a mismatched DAE/GMM input schema")
        transformed = _apply_latent_transform(density_input, transform_state)
        fit_values = transformed[
            density_fit_indices[
                : min(int(overlap.get("fit_rows", 20_000)), len(density_fit_indices))
            ]
        ]
        heldout_mask = np.ones(len(transformed), dtype=bool)
        heldout_mask[density_fit_indices] = False
        heldout_indices = np.flatnonzero(heldout_mask)
        heldout_values = transformed[heldout_indices[-min(50_000, len(heldout_indices)) :]]
        if not len(heldout_values):
            rows.append(
                {
                    "density_id": row.density_id,
                    "candidate_id": row.candidate_id,
                    "promotion_eligible": False,
                    "skipped_reason": "no_untouched_reference_rows_after_stage2_fit",
                }
            )
            continue
        refinement_tensor_cache: dict[tuple[int, tuple[int, ...], str], Any] = {}
        for metric, penalty in itertools.product(overlap["metrics"], overlap["lambdas"]):
            refined = refine_diagonal_gmm(
                fit_values,
                model,
                overlap_lambda=float(penalty),
                overlap_metric=str(metric),
                min_variance=float(overlap["variance_floor"]),
                steps=int(overlap.get("steps", 50)),
                device=args.device,
                tensor_cache=refinement_tensor_cache,
            )
            diagnostics = refinement_promotion_diagnostics(
                heldout_values,
                model,
                refined["state"],
                overlap_metric=str(metric),
                max_heldout_nll_degradation=float(overlap["max_heldout_nll_degradation"]),
            )
            refinement_id = f"{row.density_id}__{metric}__lambda{float(penalty):g}"
            refinement_dir = args.output_root / "overlap_refinement" / refinement_id
            refinement_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(refined["state"], refinement_dir / "refined_gmm_state.joblib", compress=3)
            _write_json(refinement_dir / "promotion_diagnostics.json", diagnostics)
            rows.append(
                {
                    "density_id": row.density_id,
                    "candidate_id": row.candidate_id,
                    "metric": metric,
                    "overlap_lambda": float(penalty),
                    "refined": bool(refined["refined"]),
                    "final_nll": refined.get("final_nll"),
                    "final_overlap": refined.get("final_overlap"),
                    "heldout_nll_degradation": diagnostics["heldout_nll"]["mean_nll_degradation"],
                    "bounded_overlap_delta": diagnostics["bounded_overlap_delta"],
                    "promotion_eligible_nll_overlap": diagnostics["promotion_eligible"],
                    "seed_stability_check": "pending_common_panel_recheck",
                    "ood_ranking_check": "pending_common_panel_recheck",
                    # Fail closed: a refined density is not a downstream candidate
                    # until both required rechecks have been materialized.
                    "promotion_eligible": False,
                }
            )
    report = pd.DataFrame(rows)
    if not report.empty and "promotion_eligible_nll_overlap" in report:
        preliminary = report.loc[
            report["promotion_eligible_nll_overlap"].fillna(False).astype(bool)
            & report["overlap_lambda"].gt(0)
        ].sort_values(
            ["density_id", "metric", "bounded_overlap_delta", "heldout_nll_degradation"],
            ascending=[True, True, True, True],
            kind="mergesort",
        )
        selected_indices = preliminary.groupby(
            ["density_id", "metric"], observed=True
        ).head(1).index
        for selected_index in selected_indices:
            selected = report.loc[selected_index]
            source = candidates.loc[candidates["density_id"].eq(selected["density_id"])].iloc[0]
            density_dir = args.output_root / "density_stage2" / str(selected["density_id"])
            transform_state = joblib.load(density_dir / "latent_transform.joblib")
            native = _load_native(
                args.output_root / "encoders" / str(selected["candidate_id"])
            )
            density_input, density_input_schema = _dae_gmm_input(native)
            if transform_state.get("density_input_schema") not in (None, density_input_schema):
                raise ValueError("Overlap refinement has a mismatched DAE/GMM input schema")
            transformed = _apply_latent_transform(density_input, transform_state)
            fit_values = transformed[
                density_fit_indices[
                    : min(int(overlap.get("fit_rows", 20_000)), len(density_fit_indices))
                ]
            ]
            heldout_mask = np.ones(len(transformed), dtype=bool)
            heldout_mask[density_fit_indices] = False
            heldout_values = transformed[np.flatnonzero(heldout_mask)[-50_000:]]
            baseline_states: list[dict[str, Any]] = []
            refined_states: list[dict[str, Any]] = []
            for seed_offset in range(int(overlap.get("seed_rechecks", 3))):
                seeded = GaussianMixture(
                    n_components=int(source["components"]),
                    covariance_type="diag",
                    reg_covar=float(source["reg_covar"]),
                    n_init=1,
                    random_state=int(config["seed"]) + 50_000 + seed_offset,
                    max_iter=300,
                ).fit(fit_values)
                baseline_states.append(diagonal_gmm_state(seeded))
                refined_states.append(
                    refine_diagonal_gmm(
                        fit_values,
                        seeded,
                        overlap_lambda=float(selected["overlap_lambda"]),
                        overlap_metric=str(selected["metric"]),
                        min_variance=float(overlap["variance_floor"]),
                        steps=int(overlap.get("steps", 50)),
                        device="cpu",
                    )["state"]
                )

            def occupancy_instability(states: Sequence[Mapping[str, Any]]) -> float:
                reference = states[0]
                occupancies: list[np.ndarray] = []
                for state in states:
                    probability = diagonal_gmm_statistics(
                        heldout_values, state
                    )["posteriors"]
                    if state is not reference:
                        probability = reorder_posteriors_to_reference(
                            probability,
                            align_diagonal_gmm_components(reference, state),
                        )
                    occupancies.append(np.mean(probability, axis=0))
                pairwise = [
                    float(np.sum(np.abs(occupancies[left] - occupancies[right])))
                    for left in range(len(occupancies))
                    for right in range(left + 1, len(occupancies))
                ]
                return float(np.mean(pairwise)) if pairwise else 0.0

            baseline_seed_instability = occupancy_instability(baseline_states)
            refined_seed_instability = occupancy_instability(refined_states)
            seed_pass = bool(
                refined_seed_instability <= baseline_seed_instability + 0.01
            )
            source_model = joblib.load(density_dir / "gmm.joblib")
            refined_path = (
                args.output_root
                / "overlap_refinement"
                / f"{selected['density_id']}__{selected['metric']}__lambda{float(selected['overlap_lambda']):g}"
                / "refined_gmm_state.joblib"
            )
            refined_state = joblib.load(refined_path)
            calibration = fit_values[-min(20_000, len(fit_values)) :]
            natural = heldout_values[: max(1, len(heldout_values) // 2)]
            clean = heldout_values[-max(1, len(heldout_values) // 2) :]
            rng = np.random.default_rng(int(config["seed"]) + 60_001)
            scale = np.maximum(np.std(fit_values, axis=0), 1e-6)
            mild = clean + rng.normal(0.0, 0.01, clean.shape) * scale
            structural = clean + rng.normal(0.0, 0.05, clean.shape) * scale

            def ood_report(state: Mapping[str, Any] | GaussianMixture) -> dict[str, Any]:
                return evaluate_ood_proxy(
                    diagonal_gmm_statistics(clean, state)["ood_score"],
                    diagonal_gmm_statistics(mild, state)["ood_score"],
                    diagonal_gmm_statistics(structural, state)["ood_score"],
                    diagonal_gmm_statistics(natural, state)["ood_score"],
                    calibration_scores=diagonal_gmm_statistics(calibration, state)[
                        "ood_score"
                    ],
                    aligned_clean_scores={
                        "mild_synthetic": diagonal_gmm_statistics(clean, state)[
                            "ood_score"
                        ],
                        "structural_synthetic": diagonal_gmm_statistics(clean, state)[
                            "ood_score"
                        ],
                        "natural_temporal": None,
                    },
                )

            baseline_ood = ood_report(source_model)
            refined_ood = ood_report(refined_state)
            baseline_sep = float(
                baseline_ood["clean_corrupted_separation"]["structural_synthetic"][
                    "probability_corrupted_gt_clean"
                ]
            )
            refined_sep = float(
                refined_ood["clean_corrupted_separation"]["structural_synthetic"][
                    "probability_corrupted_gt_clean"
                ]
            )
            baseline_fpr = float(
                baseline_ood["untouched_later_false_positive_rate"][
                    "elevated_or_extreme"
                ]
            )
            refined_fpr = float(
                refined_ood["untouched_later_false_positive_rate"][
                    "elevated_or_extreme"
                ]
            )
            refined_means = refined_ood["score_sets"]
            synthetic_monotonic = bool(
                float(refined_means["clean_untouched_later"]["mean"])
                <= float(refined_means["mild_synthetic"]["mean"])
                <= float(refined_means["structural_synthetic"]["mean"])
            )
            ood_pass = bool(
                synthetic_monotonic
                and refined_sep >= baseline_sep - 0.02
                and refined_fpr <= baseline_fpr + 0.01
            )
            report.loc[selected_index, "seed_stability_check"] = "pass" if seed_pass else "fail"
            report.loc[selected_index, "ood_ranking_check"] = "pass" if ood_pass else "fail"
            report.loc[selected_index, "baseline_seed_instability"] = baseline_seed_instability
            report.loc[selected_index, "refined_seed_instability"] = refined_seed_instability
            report.loc[selected_index, "baseline_ood_separation"] = baseline_sep
            report.loc[selected_index, "refined_ood_separation"] = refined_sep
            report.loc[selected_index, "promotion_eligible"] = bool(seed_pass and ood_pass)
            report.loc[selected_index, "refinement_state_path"] = str(refined_path)
            report.loc[selected_index, "family"] = str(source["family"])
            report.loc[selected_index, "output_mode"] = str(source.get("output_mode", "embedding_gmm"))
    report.to_csv(args.output_root / "overlap_refinement_summary.csv", index=False)
    if not report.empty and "promotion_eligible" in report:
        promoted = report.loc[report["promotion_eligible"].fillna(False).astype(bool)].copy()
        if not promoted.empty:
            promoted["source_density_id"] = promoted["density_id"]
            promoted["density_id"] = promoted.apply(
                lambda row: f"{row['source_density_id']}__{row['metric']}__lambda{float(row['overlap_lambda']):g}",
                axis=1,
            )
            promoted["stage"] = "refined"
            promoted.to_csv(
                args.output_root / "overlap_promoted_summary.csv", index=False
            )


def run_iic(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    summary = pd.read_csv(args.output_root / "proxy_candidate_summary.csv")
    promoted = summary.loc[summary["promoted"].fillna(False).astype(bool)]
    iic = config["iic"]
    manifest, _scaled, _keys = load_reference_cache(args.output_root / "cache/reference")
    fit_indices = _sample_indices(
        manifest, int(config["sampling"]["encoder_comparison_rows"])
    )
    rows: list[dict[str, Any]] = []
    for row in promoted.itertuples():
        latent = _load_native(args.output_root / "encoders" / row.candidate_id).latent
        rng = np.random.default_rng(int(config["seed"]) + 701)
        weak = latent + rng.normal(0.0, 0.01, latent.shape).astype(np.float32)
        strong = latent + rng.normal(0.0, 0.05, latent.shape).astype(np.float32)
        grid = list(itertools.product(
            iic["clusters"], iic["overclusters"], iic["view_pairs"], iic["mi_weight"],
            iic["marginal_balance_weight"], iic["bottleneck_dim"], iic["reconstruction_aux_weight"],
        ))
        grid = [point for point in grid if not (float(point[-1]) > 0.0 and int(point[-2]) == 0)]
        grid = _coverage_preserving_points(
            grid,
            maximum=int(args.iic_trials_per_embedding),
            seed=int(config["seed"]) + 701,
        )
        for clusters, overclusters, view_pair, mi_weight, balance, bottleneck, reconstruction in grid:
            model = FrozenEmbeddingMutualInformationClustering(
                MutualInformationClusteringConfig(
                    cluster_counts=(int(clusters),),
                    overcluster_counts=(int(overclusters),),
                    shared_bottleneck_dim=None if int(bottleneck) == 0 else int(bottleneck),
                    mutual_information_weight=float(mi_weight),
                    marginal_balance_weight=float(balance),
                    reconstruction_weight=float(reconstruction),
                    epochs=int(iic.get("epochs", 50)),
                    batch_size=int(iic.get("batch_size", 512)),
                    random_state=int(config["seed"]),
                    device=args.device,
                )
            ).fit(
                weak[fit_indices],
                (weak if view_pair == "weak_weak" else strong)[fit_indices],
            )
            diagnostics = model.diagnostics(latent)
            head = diagnostics[f"cluster_{int(clusters)}"]
            pair_diagnostics = model.diagnostics(weak, strong)
            pair_head = pair_diagnostics[f"cluster_{int(clusters)}"]
            iic_id = hashlib.sha256(
                f"{row.candidate_id}|{clusters}|{overclusters}|{view_pair}|{mi_weight}|{balance}|{bottleneck}|{reconstruction}".encode()
            ).hexdigest()[:16]
            out = args.output_root / "iic" / iic_id
            model.save(out / "model.pkl")
            rows.append(
                {
                    "iic_id": iic_id,
                    "candidate_id": row.candidate_id,
                    "family": row.family,
                    "clusters": int(clusters),
                    "overclusters": int(overclusters),
                    "view_pair": view_pair,
                    "mi_weight": float(mi_weight),
                    "balance_weight": float(balance),
                    "bottleneck": int(bottleneck),
                    "reconstruction_weight": float(reconstruction),
                    "min_occupancy": float(np.min(head["occupancy"])),
                    "mean_entropy": float(np.mean(head["normalized_entropy"])),
                    "entropy_mass_near_zero": float(np.mean(head["normalized_entropy"] <= 0.05)),
                    "entropy_mass_near_one": float(np.mean(head["normalized_entropy"] >= 0.95)),
                    "assignment_mutual_information": float(pair_head["mutual_information"]),
                    "assignment_conditional_entropy": float(pair_head["conditional_entropy"]),
                }
            )
    report = pd.DataFrame(rows)
    if not report.empty:
        grouped = report.groupby("family", observed=True)
        report["iic_proxy_score"] = (
            grouped["assignment_mutual_information"].rank(pct=True, ascending=True)
            + grouped["assignment_conditional_entropy"].rank(pct=True, ascending=False)
            + grouped["min_occupancy"].rank(pct=True, ascending=True)
            - 0.5 * grouped["entropy_mass_near_zero"].rank(pct=True, ascending=True)
            - 0.5 * grouped["entropy_mass_near_one"].rank(pct=True, ascending=True)
        )
        report = report.sort_values(
            ["family", "iic_proxy_score", "iic_id"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        report["family_rank"] = report.groupby("family", observed=True).cumcount() + 1
        report["density_id"] = "iic__" + report["iic_id"].astype(str)
        report["stage"] = "iic"
        report["output_mode"] = "iic"
    report.to_csv(args.output_root / "iic_summary.csv", index=False)


def _best_density_rows(args: argparse.Namespace, top_per_family: int) -> pd.DataFrame:
    path = args.output_root / "density_stage2_summary.csv"
    if not path.exists():
        path = args.output_root / "density_stage1_summary.csv"
    summary = pd.read_csv(path)
    proxy = pd.read_csv(args.output_root / "proxy_candidate_summary.csv")
    modes = proxy.loc[:, ["candidate_id", "output_mode"]].drop_duplicates("candidate_id")
    summary = summary.merge(modes, on="candidate_id", how="left", validate="many_to_one")
    summary["output_mode"] = summary["output_mode"].fillna("embedding_gmm")
    summary = summary.sort_values(
        ["family", "density_proxy_score", "density_id"],
        ascending=[True, False, True],
        kind="mergesort",
    ).drop_duplicates("candidate_id", keep="first")
    summary["embedding_family_rank"] = summary.groupby(
        "family", observed=True
    ).cumcount() + 1
    return summary.loc[
        summary["embedding_family_rank"].le(int(top_per_family))
    ].copy()


def _all_density_rows(args: argparse.Namespace) -> pd.DataFrame:
    """Return every materialized density arm, including explicit refinements.

    The normal search promotes density models by an outcome-free proxy.  A
    controlled density ablation instead needs to carry an explicitly named
    collection of models into the downstream base/meta comparison.  Keep this
    lookup separate from proxy promotion so callers cannot silently substitute
    a proxy winner for a requested arm.
    """

    tables: list[pd.DataFrame] = []
    for stage in (1, 2):
        path = args.output_root / f"density_stage{stage}_summary.csv"
        if not path.exists():
            continue
        table = pd.read_csv(path).copy()
        table["stage"] = int(stage)
        tables.append(table)
    refinement_path = args.output_root / "overlap_refinement_summary.csv"
    if refinement_path.exists():
        refined = pd.read_csv(refinement_path).copy()
        if not refined.empty:
            source = refined["density_id"].astype(str)
            refined["source_density_id"] = source
            refined["density_id"] = (
                source
                + "__"
                + refined["metric"].astype(str)
                + "__lambda"
                + refined["overlap_lambda"].astype(float).map(lambda value: f"{value:g}")
            )
            refined["stage"] = "refined"
            refined["refinement_state_path"] = refined["density_id"].map(
                lambda density_id: str(
                    args.output_root
                    / "overlap_refinement"
                    / density_id
                    / "refined_gmm_state.joblib"
                )
            )
            tables.append(refined)
    if not tables:
        raise FileNotFoundError("No density summaries are available")
    result = pd.concat(tables, ignore_index=True, sort=False)
    if "candidate_id" not in result or "density_id" not in result:
        raise ValueError("Density summaries are missing candidate_id or density_id")
    proxy_path = args.output_root / "proxy_candidate_summary.csv"
    if proxy_path.exists():
        proxy = pd.read_csv(proxy_path)
        if "output_mode" in proxy:
            modes = proxy.loc[:, ["candidate_id", "output_mode"]].drop_duplicates(
                "candidate_id"
            )
            result = result.drop(columns=["output_mode"], errors="ignore").merge(
                modes, on="candidate_id", how="left", validate="many_to_one"
            )
    if "output_mode" not in result:
        result["output_mode"] = "embedding_gmm"
    else:
        result["output_mode"] = result["output_mode"].fillna("embedding_gmm")
    if result["density_id"].duplicated().any():
        duplicates = result.loc[result["density_id"].duplicated(), "density_id"].astype(str)
        raise ValueError("Density identifiers must be unique: " + ", ".join(duplicates.head(5)))
    return result


def _requested_density_rows(args: argparse.Namespace) -> pd.DataFrame | None:
    requested = list(getattr(args, "downstream_density_ids", []))
    if not requested:
        return None
    available = _all_density_rows(args).set_index("density_id", drop=False)
    missing = [density_id for density_id in requested if density_id not in available.index]
    if missing:
        raise ValueError(
            "Requested downstream density IDs are unavailable: " + ", ".join(missing)
        )
    selected = available.loc[requested]
    if isinstance(selected, pd.Series):
        selected = selected.to_frame().T
    return selected.reset_index(drop=True)


def _downstream_candidate_rows(args: argparse.Namespace, top_per_family: int) -> pd.DataFrame:
    requested = _requested_density_rows(args)
    if requested is not None:
        return requested
    density = _best_density_rows(args, top_per_family=top_per_family)
    refinement_path = args.output_root / "overlap_promoted_summary.csv"
    if refinement_path.exists():
        refinements = pd.read_csv(refinement_path).sort_values(
            ["family", "bounded_overlap_delta", "heldout_nll_degradation"],
            ascending=[True, True, True],
            kind="mergesort",
        )
        refinements = refinements.groupby(
            "family", observed=True, group_keys=False
        ).head(1)
        for column in density.columns:
            if column not in refinements:
                refinements[column] = np.nan
        for column in refinements.columns:
            if column not in density:
                density[column] = np.nan
        density = pd.concat(
            [density, refinements.loc[:, density.columns]], ignore_index=True
        )
    iic_path = args.output_root / "iic_summary.csv"
    if not iic_path.exists():
        return density
    iic = pd.read_csv(iic_path)
    iic = iic.loc[iic["family_rank"].le(1)].copy()
    for column in density.columns:
        if column not in iic:
            iic[column] = np.nan
    for column in iic.columns:
        if column not in density:
            density[column] = np.nan
    return pd.concat([density, iic.loc[:, density.columns]], ignore_index=True)


def _sklearn_density_statistics(values: np.ndarray, model: GaussianMixture) -> dict[str, np.ndarray]:
    z = np.asarray(values, dtype=np.float64)
    posterior = model.predict_proba(z)
    delta = z[:, None, :] - np.asarray(model.means_, dtype=np.float64)[None, :, :]
    if model.covariance_type == "diag":
        variance = np.maximum(np.asarray(model.covariances_, dtype=np.float64), 1e-12)
        mahal_sq = np.sum(delta * delta / variance[None, :, :], axis=2)
    elif model.covariance_type == "tied":
        precision = np.linalg.pinv(np.asarray(model.covariances_, dtype=np.float64))
        mahal_sq = np.einsum("nkd,de,nke->nk", delta, precision, delta, optimize=True)
    else:
        raise ValueError(f"Unsupported downstream GMM covariance: {model.covariance_type}")
    log_density = model.score_samples(z)
    return {
        "posteriors": posterior.astype(np.float32),
        "mahalanobis": np.sqrt(np.maximum(mahal_sq, 0.0)).astype(np.float32),
        "log_density": log_density.astype(np.float32),
        "ood_score": (-log_density).astype(np.float32),
    }


def _transform_with_density(
    native: NativeLatentOutput, density_dir: Path
) -> tuple[GmmPanelFit, np.ndarray, dict[str, np.ndarray]]:
    model = joblib.load(density_dir / "gmm.joblib")
    transform_state = joblib.load(density_dir / "latent_transform.joblib")
    density_input, density_input_schema = _dae_gmm_input(native)
    expected_schema = transform_state.get("density_input_schema")
    if expected_schema is not None and expected_schema != density_input_schema:
        raise ValueError("DAE/GMM density input schema does not match the frozen transform")
    values = _apply_latent_transform(density_input, transform_state)
    if model.covariance_type == "diag":
        state = diagonal_gmm_state(model)
    else:
        diagonal = np.diag(np.asarray(model.covariances_, dtype=np.float64))
        state = {
            "weights": np.asarray(model.weights_, dtype=np.float64),
            "means": np.asarray(model.means_, dtype=np.float64),
            "covariances": np.repeat(diagonal[None, :], model.n_components, axis=0),
        }
    panel = GmmPanelFit(
        GmmPanelSpec(int(model.n_components), str(model.covariance_type), float(model.reg_covar)),
        int(model.random_state),
        state,
    )
    return panel, values.astype(np.float32), _sklearn_density_statistics(values, model)


def _full_native_cache(
    args: argparse.Namespace,
    features: Sequence[str],
    *,
    candidate_id: str,
) -> tuple[pd.DataFrame, NativeLatentOutput]:
    """Transform the full downstream universe once per frozen encoder.

    A density search often evaluates many GMMs against one DAE embedding.  The
    DAE transform is identical across those arms and is substantially more
    expensive than the subsequent 17-dimensional GMM score.  Persist only the
    observable outputs required by every density model, using mmapable float32
    arrays to avoid repeated dense encoder inference and pandas copies.
    """

    cache_dir = args.output_root / "downstream_native_cache" / str(candidate_id)
    keys_path = cache_dir / "keys.parquet"
    latent_path = cache_dir / "latent.npy"
    reconstruction_path = cache_dir / "reconstruction_error.npy"
    contract_path = cache_dir / "contract.json"
    encoder_path = args.output_root / "encoders" / str(candidate_id) / "encoder.joblib"
    contract = {
        "schema": "frozen_encoder_full_universe_cache_v1",
        "candidate_id": str(candidate_id),
        "encoder_sha256": _sha256(encoder_path),
        "input_feature_contract_sha256": str(getattr(args, "input_contract_sha256", "")),
        "input_feature_count": int(len(features)),
        "labels_path": str(Path(args.labels_path).resolve()),
        "feature_dir": str(Path(args.feature_dir).resolve()),
    }
    if (
        keys_path.exists()
        and latent_path.exists()
        and reconstruction_path.exists()
        and contract_path.exists()
        and json.loads(contract_path.read_text(encoding="utf-8")) == contract
        and not args.rerun
    ):
        keys = pd.read_parquet(keys_path)
        native = NativeLatentOutput(
            latent=np.load(latent_path, mmap_mode="r"),
            reconstruction_error=np.load(reconstruction_path, mmap_mode="r"),
        )
        if len(keys) != len(native.latent) or len(keys) != len(native.reconstruction_error):
            raise ValueError("Cached full encoder output has inconsistent row counts")
        return keys, native

    frame = _load_input_frame(args, features)
    keys = frame.loc[:, ["__ts__", "__symbol__", "side"]].copy()
    duplicate = keys.duplicated(["__ts__", "__symbol__", "side"])
    unique_positions = np.flatnonzero(~duplicate.to_numpy())
    unique = frame.iloc[unique_positions].reset_index(drop=True)
    unique_keys = unique.loc[:, ["__ts__", "__symbol__", "side"]].copy()
    encoder = _restore_candidate_encoder(
        args.output_root / "encoders" / str(candidate_id), args.device
    )
    native = encoder.transform_native(
        unique.loc[:, list(features)].to_numpy(np.float32),
        sides=unique["side"].to_numpy(),
    )
    if native.reconstruction_error is None:
        raise ValueError("Full DAE cache requires reconstruction error for the GMM input")
    cache_dir.mkdir(parents=True, exist_ok=True)
    unique_keys.to_parquet(keys_path, index=False, compression="zstd")
    np.save(latent_path, np.asarray(native.latent, dtype=np.float32), allow_pickle=False)
    np.save(
        reconstruction_path,
        np.asarray(native.reconstruction_error, dtype=np.float32),
        allow_pickle=False,
    )
    _write_json(contract_path, contract)
    return unique_keys, NativeLatentOutput(
        latent=np.load(latent_path, mmap_mode="r"),
        reconstruction_error=np.load(reconstruction_path, mmap_mode="r"),
    )


def _materialize_full_sidecar(
    args: argparse.Namespace,
    features: Sequence[str],
    row: Any,
) -> Path:
    output = args.output_root / "downstream_sidecars" / f"{row.density_id}.parquet"
    manifest, _scaled, _keys = load_reference_cache(args.output_root / "cache/reference")
    model_path = (
        args.output_root / "iic" / str(row.iic_id) / "model.pkl"
        if str(row.stage) == "iic"
        else (
            Path(str(row.refinement_state_path))
            if str(row.stage) == "refined"
            else args.output_root
            / f"density_stage{int(row.stage)}"
            / row.density_id
            / "gmm.joblib"
        )
    )
    transform_path = (
        None
        if str(row.stage) == "iic"
        else (
            args.output_root / "density_stage2" / str(row.source_density_id) / "latent_transform.joblib"
            if str(row.stage) == "refined"
            else args.output_root
            / f"density_stage{int(row.stage)}"
            / row.density_id
            / "latent_transform.joblib"
        )
    )
    sidecar_contract = {
        "row_identity_hash": manifest.row_identity_hash,
        "feature_value_hash": manifest.feature_value_hash,
        "candidate_id": str(row.candidate_id),
        "encoder_sha256": _sha256(args.output_root / "encoders" / row.candidate_id / "encoder.joblib"),
        "assignment_model_sha256": _sha256(model_path),
        "density_transform_sha256": (
            _sha256(transform_path) if transform_path is not None else None
        ),
        "density_input_schema": (
            "dae_latent_plus_reconstruction_novelty_v1"
            if transform_path is not None
            else None
        ),
        "output_mode": str(getattr(row, "output_mode", "embedding_gmm")),
    }
    contract_path = output.with_suffix(".contract.json")
    if output.exists() and not args.rerun:
        if not contract_path.exists() or json.loads(contract_path.read_text(encoding="utf-8")) != sidecar_contract:
            raise ValueError(f"Cached sidecar {output} does not match its encoder/reference contract")
        return output
    unique, native = _full_native_cache(
        args, features, candidate_id=str(row.candidate_id)
    )
    if str(row.stage) == "iic":
        model = FrozenEmbeddingMutualInformationClustering.load(
            args.output_root / "iic" / str(row.iic_id) / "model.pkl"
        )
        head_name = f"cluster_{int(row.clusters)}"
        diagnostics = model.diagnostics(native.latent)[head_name]
        sidecar = unique.loc[:, ["__ts__", "__symbol__", "side"]].reset_index(drop=True).copy()
        for index in range(native.latent.shape[1]):
            sidecar[f"repr_latent_{index:02d}"] = native.latent[:, index]
        probabilities = np.asarray(diagnostics["probabilities"], dtype=np.float32)
        for index in range(probabilities.shape[1]):
            sidecar[f"repr_iic_posterior_{index:02d}"] = probabilities[:, index]
        sidecar["repr_iic_entropy_norm"] = diagnostics["normalized_entropy"]
        sidecar["repr_iic_posterior_margin"] = diagnostics["margin"]
        sidecar["repr_iic_assignment_uncertainty"] = 1.0 - probabilities.max(axis=1)
        sidecar["repr_density_available"] = np.float32(0.0)
        if native.reconstruction_error is not None:
            sidecar["repr_reconstruction_error"] = native.reconstruction_error
        output.parent.mkdir(parents=True, exist_ok=True)
        sidecar.to_parquet(output, index=False, compression="zstd")
        _write_json(contract_path, sidecar_contract)
        return output
    if str(row.stage) == "refined":
        source_dir = args.output_root / "density_stage2" / str(row.source_density_id)
        transform_state = joblib.load(source_dir / "latent_transform.joblib")
        density_input, density_input_schema = _dae_gmm_input(native)
        if transform_state.get("density_input_schema") not in (None, density_input_schema):
            raise ValueError("Refined density sidecar has a mismatched DAE/GMM input schema")
        transformed = _apply_latent_transform(density_input, transform_state).astype(np.float32)
        refined_state = joblib.load(Path(str(row.refinement_state_path)))
        stats = diagonal_gmm_statistics(transformed, refined_state)
        panel = GmmPanelFit(
            GmmPanelSpec(
                int(np.asarray(refined_state["weights"]).shape[0]),
                "diag",
                0.003,
            ),
            0,
            refined_state,
        )
        sidecar = materialize_representation_features(
            keys=unique,
            native=native,
            panel=panel,
            reference_indices=np.arange(min(100_000, len(unique)), dtype=np.int64),
            output_mode="embedding_gmm",
            density_statistics=stats,
        )
        sidecar["repr_gmm_reconstruction_log1p"] = density_input[:, -1]
        output.parent.mkdir(parents=True, exist_ok=True)
        sidecar.to_parquet(output, index=False, compression="zstd")
        _write_json(contract_path, sidecar_contract)
        return output
    density_input, _density_input_schema = _dae_gmm_input(native)
    panel, transformed, density_stats = _transform_with_density(
        native,
        args.output_root / f"density_stage{int(row.stage)}" / row.density_id,
    )
    proxy_summary = pd.read_csv(args.output_root / "proxy_candidate_summary.csv")
    proxy_match = proxy_summary.loc[
        proxy_summary["candidate_id"].eq(str(row.candidate_id))
    ]
    consistency_score = (
        1.0 - float(proxy_match.iloc[0]["best_panel_perturb_tv"])
        if not proxy_match.empty
        else None
    )
    sidecar = materialize_representation_features(
        keys=unique,
        native=native,
        panel=panel,
        reference_indices=np.arange(min(100_000, len(unique)), dtype=np.int64),
        output_mode=str(getattr(row, "output_mode", "embedding_gmm")),
        density_statistics=density_stats,
        perturbation_consistency_score=consistency_score,
    )
    sidecar["repr_gmm_reconstruction_log1p"] = density_input[:, -1]
    output.parent.mkdir(parents=True, exist_ok=True)
    sidecar.to_parquet(output, index=False, compression="zstd")
    _write_json(contract_path, sidecar_contract)
    return output


def run_base(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    features: Sequence[str],
) -> None:
    rows = _downstream_candidate_rows(
        args,
        top_per_family=int(config["downstream"].get("base_density_per_family", 1)),
    )
    periods = split_months(args.oos_months)
    base_contract = config["downstream"]["base"]
    common_output = args.output_root / "base_common_selection"
    common_selected = (
        common_output / "_feature_selection_bme_sample" / "selected_features.json"
    )
    if not common_selected.is_file():
        run_hpo(
            labels_path=args.labels_path,
            feature_dir=args.feature_dir,
            feature_list_csv=args.base_feature_list,
            output_dir=common_output,
            months=periods["base_oos"],
            max_feature_store_features=None,
            max_train_rows=0,
            feature_selection_sample_rows=int(base_contract["mda_fit_rows"])
            + int(base_contract["mda_eval_rows"]),
            hpo_max_train_rows=0,
            n_trials=0,
            seed=int(config["seed"]),
            include_ae_gmm_state_features=False,
            ae_gmm_state_feature_max_train_rows=0,
            ae_gmm_state_feature_gmm_max_train_rows=0,
            ae_gmm_state_feature_max_iter=0,
            feature_selection_top_n=0,
            feature_selection_target_mode="target_soft",
            feature_selection_method="mda",
            max_oos_model_age_days=0,
            single_fit_oos_window=True,
            train_window_days=int(base_contract["train_days"]),
            fixed_params_json=args.base_params,
            save_fold_models=False,
            save_final_model=False,
            two_phase_wide_feature_selection=True,
            hpo_only=True,
        )
    common_payload = json.loads(common_selected.read_text(encoding="utf-8"))
    common_features = list(map(str, common_payload.get("selected_features", [])))
    if not common_features:
        raise RuntimeError(f"Common base MDA selected no features: {common_selected}")
    for row in rows.itertuples():
        sidecar = _materialize_full_sidecar(args, features, row)
        output = args.output_root / "base" / row.density_id
        sidecar_columns = [
            column
            for column in __import__(
                "pyarrow.parquet", fromlist=["ParquetFile"]
            ).ParquetFile(sidecar).schema.names
            if column not in {"__ts__", "__symbol__", "side"}
        ]
        candidate_contract = (
            args.output_root / "base_feature_contracts" / f"{row.density_id}.json"
        )
        contract_payload = {
            "schema": "alternative_representation_fixed_base_features_v1",
            "selected_features": list(dict.fromkeys([*common_features, *sidecar_columns])),
            "common_mda_contract": str(common_selected),
            "common_feature_count": len(common_features),
            "representation_feature_count": len(sidecar_columns),
        }
        existing_contract = (
            json.loads(candidate_contract.read_text(encoding="utf-8"))
            if candidate_contract.exists()
            else None
        )
        completed_ledger = output / "best_oos_scored_ledger.parquet"
        if completed_ledger.exists() and existing_contract == contract_payload and not args.rerun:
            continue
        _write_json(candidate_contract, contract_payload)
        run_hpo(
            labels_path=args.labels_path,
            feature_dir=args.feature_dir,
            feature_list_csv=args.base_feature_list,
            output_dir=output,
            months=periods["base_oos"],
            max_feature_store_features=None,
            max_train_rows=0,
            feature_selection_sample_rows=int(base_contract["mda_fit_rows"])
            + int(base_contract["mda_eval_rows"]),
            hpo_max_train_rows=0,
            n_trials=0,
            seed=int(config["seed"]),
            include_ae_gmm_state_features=False,
            ae_gmm_state_feature_max_train_rows=0,
            ae_gmm_state_feature_gmm_max_train_rows=0,
            ae_gmm_state_feature_max_iter=0,
            feature_selection_top_n=0,
            feature_selection_target_mode="target_soft",
            feature_selection_method="mda",
            max_oos_model_age_days=0,
            single_fit_oos_window=True,
            train_window_days=int(base_contract["train_days"]),
            fixed_params_json=args.base_params,
            fixed_selected_features_csv=candidate_contract,
            save_fold_models=True,
            save_final_model=False,
            two_phase_wide_feature_selection=True,
            external_feature_sidecar_path=sidecar,
        )


def _base_promoted_rows(
    args: argparse.Namespace,
    *,
    top_per_family: int,
    months: Sequence[str],
    baseline_candidate_id: str,
) -> pd.DataFrame:
    candidates = _downstream_candidate_rows(args, top_per_family=top_per_family)
    rows: list[dict[str, Any]] = []
    for candidate in candidates.itertuples():
        ledger_path = args.output_root / "base" / candidate.density_id / "best_oos_scored_ledger.parquet"
        if not ledger_path.exists():
            continue
        metrics = economic_metrics(
            pd.read_parquet(ledger_path),
            arm=str(candidate.density_id),
            months=months,
        )
        overall = metrics.loc[metrics["scope"].eq("overall")].set_index("top_frac")
        ev = {
            fraction: float(overall.loc[fraction, "mean_ev_after_1pct"])
            if fraction in overall.index
            else float("-inf")
            for fraction in (0.10, 0.20, 0.30)
        }
        rows.append(
            {
                **candidate._asdict(),
                "base_top10_ev": ev[0.10],
                "base_top20_ev": ev[0.20],
                "base_top30_ev": ev[0.30],
                "base_outer_score": 0.40 * ev[0.10] + 0.35 * ev[0.20] + 0.25 * ev[0.30],
            }
        )
    ranked = pd.DataFrame(rows)
    if ranked.empty:
        return ranked
    ranked = ranked.sort_values(
        ["family", "base_outer_score", "density_id"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    ranked["base_family_rank"] = ranked.groupby("family", observed=True).cumcount() + 1
    requested = list(getattr(args, "downstream_density_ids", []))
    if requested:
        ranked["promoted_to_meta"] = ranked["density_id"].astype(str).isin(requested)
    else:
        ranked["promoted_to_meta"] = ranked["base_family_rank"].le(int(top_per_family)) | ranked[
            "candidate_id"
        ].eq(str(baseline_candidate_id))
    ranked.to_csv(args.output_root / "base_outer_promotion.csv", index=False)
    promoted = ranked.loc[ranked["promoted_to_meta"]].copy()
    if requested:
        promoted = promoted.set_index("density_id", drop=False).loc[requested].reset_index(drop=True)
    return promoted


def _relative_output_subdir(value: str, *, argument: str) -> Path:
    """Return a safe experiment-relative directory for isolated meta comparisons."""
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"{argument} must be a non-empty relative path")
    return path


def _meta_comparison_rows(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    *,
    months: Sequence[str],
) -> pd.DataFrame:
    """Select representation arms without changing the base-source contract."""
    rows = _base_promoted_rows(
        args,
        top_per_family=int(config["downstream"]["report_top_per_family"]),
        months=months,
        baseline_candidate_id=_baseline_candidate_id(config),
    )
    requested = list(getattr(args, "meta_candidate_ids", [])) or list(
        getattr(args, "downstream_density_ids", [])
    )
    if not requested:
        return rows
    available = set(rows["density_id"].astype(str))
    missing = sorted(set(requested).difference(available))
    if missing:
        raise ValueError(
            "Requested meta comparison density IDs are unavailable: " + ", ".join(missing)
        )
    selected = rows.loc[rows["density_id"].astype(str).isin(requested)].copy()
    if selected.empty:
        raise ValueError("No representation arms selected for the meta comparison")
    return selected


def _prediction_row_index(frame: pd.DataFrame) -> pd.MultiIndex:
    """Stable inference identity used for cross-representation comparison."""
    side_col = "side" if "side" in frame.columns else "side_name"
    if side_col not in frame.columns:
        raise ValueError("Meta prediction frame has no side or side_name identity")
    keys = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(frame["__ts__"], utc=True, errors="coerce").astype("int64"),
            "__symbol__": frame["__symbol__"].astype(str),
            "__side__": frame[side_col].astype(str),
        }
    )
    if keys.isna().any(axis=None) or bool(keys.duplicated().any()):
        raise ValueError("Meta prediction frame has invalid or duplicate comparison keys")
    return pd.MultiIndex.from_frame(keys)


def _score_column_for_metrics(frame: pd.DataFrame) -> str | None:
    candidates = [
        column
        for column in frame.columns
        if str(column).endswith("_ev_residual_expert_hier_mapped")
    ]
    if not candidates:
        candidates = [
            column
            for column in ("score_meta_raw", "score_base_rank", "score")
            if column in frame.columns
        ]
    return str(candidates[0]) if candidates else None


def _augment_handoff_with_sidecar(handoff: Path, sidecar: Path, output: Path) -> Path:
    base = pd.read_parquet(handoff)
    import pyarrow.parquet as pq

    feature_columns = [
        column
        for column in pq.ParquetFile(sidecar).schema.names
        if column not in {"__ts__", "__symbol__", "side"}
    ]
    context = pd.read_parquet(sidecar, columns=["__ts__", "__symbol__", "side", *feature_columns])

    def canonical_side(frame: pd.DataFrame, source: str) -> None:
        if "side" in frame:
            values = pd.to_numeric(frame["side"], errors="coerce")
        elif "__side__" in frame:
            values = pd.to_numeric(frame["__side__"], errors="coerce")
        elif "side_name" in frame:
            names = frame["side_name"].astype(str).str.lower()
            values = names.map({"long": 1, "short": -1})
        else:
            raise ValueError(
                f"{source} has no supported side key; expected side, __side__, or side_name"
            )
        if values.isna().any() or not values.isin((-1, 1)).all():
            raise ValueError(f"{source} contains invalid side values for representation join")
        frame["side"] = values.astype(np.int8)

    for frame, source in ((base, "meta handoff"), (context, "representation sidecar")):
        frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
        frame["__symbol__"] = frame["__symbol__"].astype(str)
        canonical_side(frame, source)
    overlap = [column for column in feature_columns if column in base.columns]
    base = base.drop(columns=overlap, errors="ignore")
    merged = base.merge(context, on=["__ts__", "__symbol__", "side"], how="left", validate="many_to_one")
    if float(merged[feature_columns].notna().all(axis=1).mean()) < 0.999:
        raise ValueError("Meta representation sidecar coverage is incomplete")
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output, index=False, compression="zstd")
    return output


def run_meta(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    periods = split_months(args.oos_months)
    base_rows = _meta_comparison_rows(
        args,
        config,
        months=periods["base_oos"],
    )
    production_meta = load_feature_contract(args.meta_contract)
    meta_contract_config = config["downstream"]["meta"]
    meta_output_root = args.output_root / args.meta_output_subdir
    handoff_output_root = args.output_root / "meta_handoff" / args.meta_output_subdir
    contract_output_root = args.output_root / "meta_contracts" / args.meta_output_subdir
    fixed_base_density_id = str(args.fixed_meta_base_density_id or "")
    shared_handoffs: dict[str, Path] = {}
    reused_handoff_dir = (
        Path(args.reuse_fixed_meta_handoff).resolve()
        if args.reuse_fixed_meta_handoff is not None
        else None
    )
    for row in base_rows.itertuples():
        source_density_id = fixed_base_density_id or str(row.density_id)
        base_dir = args.output_root / "base" / source_density_id
        ledger = base_dir / "best_oos_scored_ledger.parquet"
        if not ledger.exists():
            raise FileNotFoundError(
                f"Missing fixed base ledger for meta arm {row.density_id}: {ledger}"
            )
        output = meta_output_root / row.density_id
        handoff_dir = handoff_output_root / row.density_id
        output_manifest = output / "representation_meta_contract.json"
        comparison_contract = {
            "schema": "alternative_representation_fixed_base_meta_v1",
            "representation_density_id": str(row.density_id),
            "base_density_id": source_density_id,
            "base_ledger": str(ledger),
            "base_ledger_sha256": _sha256(ledger),
            "meta_output_subdir": str(args.meta_output_subdir),
            "oos_months": list(periods["meta_oos"]),
            "feature_contract": str(args.meta_contract),
        }
        if reused_handoff_dir is not None:
            comparison_contract["reused_fixed_base_handoff_dir"] = str(reused_handoff_dir)
            comparison_contract["reused_fixed_base_handoff_manifest_sha256"] = _sha256(
                reused_handoff_dir / "manifest.json"
            )
        if (
            (output / "metrics.csv").exists()
            and output_manifest.exists()
            and json.loads(output_manifest.read_text(encoding="utf-8")) == comparison_contract
            and not args.rerun
        ):
            continue
        source_handoff_dir = shared_handoffs.get(source_density_id)
        if source_handoff_dir is None:
            source_contract = {
                "schema": "alternative_representation_fixed_base_handoff_v1",
                "base_density_id": source_density_id,
                "base_ledger": str(ledger),
                "base_ledger_sha256": _sha256(ledger),
                "fit_months": list(periods["meta_train"]),
                "holdout_month": str(periods["meta_oos"][0]),
                "selected_col": "selected_top30",
                "embedded_round_trip_cost": 0.01,
                "executable_cost_floor": 0.01,
            }
            if reused_handoff_dir is not None:
                manifest_path = reused_handoff_dir / "manifest.json"
                handoff_path = reused_handoff_dir / "train_meta_regime_handoff.parquet"
                scored_path = reused_handoff_dir / "s52_trailing_regime_scored_ledger.parquet"
                if not (manifest_path.exists() and handoff_path.exists() and scored_path.exists()):
                    raise FileNotFoundError(
                        "Reusable fixed-base handoff is incomplete: "
                        f"{reused_handoff_dir}"
                    )
                provenance = json.loads(manifest_path.read_text(encoding="utf-8"))
                source_ledger = Path(str(provenance.get("ledger_path", ""))).resolve()
                if source_ledger != ledger.resolve():
                    raise ValueError(
                        "Reusable fixed-base handoff has a different base ledger: "
                        f"{source_ledger} != {ledger.resolve()}"
                    )
                expected_fields = {
                    "fit_months": list(periods["meta_train"]),
                    "holdout_month": str(periods["meta_oos"][0]),
                    "selected_col": "selected_top30",
                    "embedded_round_trip_cost": 0.01,
                    "executable_cost_floor": 0.01,
                }
                mismatched = {
                    key: (provenance.get(key), value)
                    for key, value in expected_fields.items()
                    if provenance.get(key) != value
                }
                if mismatched:
                    raise ValueError(
                        "Reusable fixed-base handoff does not match the requested contract: "
                        f"{mismatched}"
                    )
                source_handoff_dir = reused_handoff_dir
                source_contract["reused_handoff_dir"] = str(reused_handoff_dir)
                source_contract["reused_handoff_manifest_sha256"] = _sha256(manifest_path)
            else:
                source_handoff_dir = (
                    handoff_output_root / "_fixed_base_source" / source_density_id
                )
                source_manifest = source_handoff_dir / "fixed_base_handoff_contract.json"
                source_handoff = source_handoff_dir / "train_meta_regime_handoff.parquet"
                source_scored = source_handoff_dir / "s52_trailing_regime_scored_ledger.parquet"
                if not (
                    source_handoff.exists()
                    and source_scored.exists()
                    and source_manifest.exists()
                    and json.loads(source_manifest.read_text(encoding="utf-8")) == source_contract
                    and not args.rerun
                ):
                    run_handoff_only(
                        ledger_path=ledger,
                        output_dir=source_handoff_dir,
                        label_context_dir=args.labels_path,
                        feature_dir=args.feature_dir,
                        feature_store_scope="all_safe",
                        fixed_ae_gmm_state_pkl=None,
                        fit_months=periods["meta_train"],
                        holdout_month=periods["meta_oos"][0],
                        selected_col="selected_top30",
                        embedded_round_trip_cost=0.01,
                        executable_cost_floor=0.01,
                        context_cache_dir=args.output_root / "meta_feature_context_cache",
                    )
                    _write_json(source_manifest, source_contract)
            shared_handoffs[source_density_id] = source_handoff_dir
        sidecar = args.output_root / "downstream_sidecars" / f"{row.density_id}.parquet"
        augmented = _augment_handoff_with_sidecar(
            source_handoff_dir / "train_meta_regime_handoff.parquet",
            sidecar,
            handoff_dir / "train_meta_regime_handoff_with_representation.parquet",
        )
        added = [
            column
            for column in __import__("pyarrow.parquet", fromlist=["ParquetFile"]).ParquetFile(sidecar).schema.names
            if column not in {"__ts__", "__symbol__", "side"}
        ]
        contract = contract_output_root / f"{row.density_id}.json"
        _write_json(contract, {"selected_feature_union": list(dict.fromkeys([*production_meta, *added]))})
        eval_start = pd.Timestamp(pd.Period(periods["meta_oos"][0]).start_time).strftime("%Y-%m-%d")
        eval_end = pd.Timestamp((pd.Period(periods["meta_oos"][-1]) + 1).start_time).strftime("%Y-%m-%d")
        command = [
            sys.executable,
            str(ROOT / "scripts/run_meta_v9_ev_mapped_side_residual_ablation.py"),
            "--source-mode", "current_handoff",
            "--handoff", str(augmented),
            "--scored-ledger", str(source_handoff_dir / "s52_trailing_regime_scored_ledger.parquet"),
            "--feature-dir", str(args.feature_dir),
            "--contract", str(contract),
            "--out-dir", str(output),
            "--calibration-month", periods["meta_train"][-1],
            "--eval-start", eval_start,
            "--eval-end", eval_end,
            "--oos-fit-mode", "frozen_pre_eval",
            "--selection-mode", "staged_mda",
            "--backbone-score", "base",
            "--feature-selection-max-rows", str(int(meta_contract_config.get("feature_selection_rows", 45_000))),
            "--reuse-hpo-params-manifest", str(args.meta_params_manifest),
        ]
        subprocess.run(command, cwd=ROOT, check=True)
        _write_json(output_manifest, comparison_contract)


def report(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    periods = split_months(args.oos_months)
    meta_rows = _meta_comparison_rows(
        args,
        config,
        months=periods["base_oos"],
    )
    base_rows = _downstream_candidate_rows(
        args,
        top_per_family=int(config["downstream"].get("base_density_per_family", 1)),
    )
    base_metrics: list[pd.DataFrame] = []
    meta_metrics: list[pd.DataFrame] = []
    meta_predictions: dict[str, tuple[str, pd.DataFrame, str]] = {}
    for row in base_rows.itertuples():
        ledger = args.output_root / "base" / row.density_id / "best_oos_scored_ledger.parquet"
        if ledger.exists():
            metrics = economic_metrics(
                pd.read_parquet(ledger), arm=str(row.density_id), months=periods["base_oos"]
            )
            metrics["family"] = str(row.family)
            base_metrics.append(metrics)
    meta_output_root = args.output_root / args.meta_output_subdir
    for row in meta_rows.itertuples():
        meta_path = meta_output_root / row.density_id / "metrics.csv"
        if meta_path.exists():
            metrics = pd.read_csv(meta_path)
            metrics.insert(0, "density_id", str(row.density_id))
            metrics.insert(1, "family", str(row.family))
            meta_metrics.append(metrics)
        prediction_path = meta_output_root / row.density_id / "oos_predictions.parquet"
        if prediction_path.exists():
            predictions = pd.read_parquet(prediction_path)
            score_col = _score_column_for_metrics(predictions)
            if score_col is not None:
                detailed = economic_metrics(
                    predictions,
                    arm=str(row.density_id),
                    score_col=score_col,
                    months=periods["meta_oos"],
                )
                detailed["family"] = str(row.family)
                detailed.to_csv(
                    meta_output_root / row.density_id / "economic_metrics_detailed_raw_universe.csv",
                    index=False,
                )
                meta_predictions[str(row.density_id)] = (
                    str(row.family),
                    predictions,
                    score_col,
                )
    report_dir = args.output_root / args.report_subdir
    report_dir.mkdir(parents=True, exist_ok=True)
    if base_metrics:
        combined_base = pd.concat(base_metrics, ignore_index=True)
        baseline_rows = base_rows.loc[
            base_rows["candidate_id"].eq(_baseline_candidate_id(config))
        ]
        if not baseline_rows.empty:
            combined_base = add_baseline_deltas(
                combined_base,
                baseline_arm=str(baseline_rows.iloc[0]["density_id"]),
            )
        combined_base.to_csv(
            report_dir / "base_all_promoted_embeddings_metrics.csv", index=False
        )
    if meta_metrics:
        pd.concat(meta_metrics, ignore_index=True).to_csv(
            report_dir / "meta_top2_per_family_metrics.csv", index=False
        )
    common_rows_report: list[dict[str, Any]] = []
    common_meta: list[pd.DataFrame] = []
    if meta_predictions:
        key_sets = {
            density_id: _prediction_row_index(predictions)
            for density_id, (_family, predictions, _score_col) in meta_predictions.items()
        }
        shared_index = next(iter(key_sets.values()))
        for keys in list(key_sets.values())[1:]:
            shared_index = shared_index.intersection(keys, sort=False)
        if len(shared_index) <= 0:
            raise RuntimeError("Meta representation arms have no shared OOS prediction rows")
        for density_id, (family, predictions, score_col) in meta_predictions.items():
            row_index = key_sets[density_id]
            common = predictions.loc[row_index.isin(shared_index)].copy(deep=False)
            detailed = economic_metrics(
                common,
                arm=density_id,
                score_col=score_col,
                months=periods["meta_oos"],
            )
            detailed["family"] = family
            detailed.to_csv(
                    meta_output_root / density_id / "economic_metrics_detailed_common_rows.csv",
                index=False,
            )
            common_meta.append(detailed)
            common_rows_report.append(
                {
                    "density_id": density_id,
                    "family": family,
                    "candidate_rows": int(len(predictions)),
                    "shared_rows": int(len(common)),
                    "shared_row_coverage": float(len(common) / max(len(predictions), 1)),
                }
            )
    if common_meta:
        combined_meta = pd.concat(common_meta, ignore_index=True)
        baseline_meta_rows = meta_rows.loc[
            meta_rows["candidate_id"].eq(_baseline_candidate_id(config))
        ]
        if not baseline_meta_rows.empty:
            combined_meta = add_baseline_deltas(
                combined_meta,
                baseline_arm=str(baseline_meta_rows.iloc[0]["density_id"]),
            )
        combined_meta.to_csv(report_dir / "meta_top2_per_family_economic_metrics_common_rows.csv", index=False)
        pd.DataFrame(common_rows_report).to_csv(
            report_dir / "meta_common_row_coverage.csv", index=False
        )
    _write_json(
        report_dir / "report_manifest.json",
        {
            "schema": "alternative_representation_final_report_v1",
            "top_per_family": int(config["downstream"]["report_top_per_family"]),
            "base_oos_months": periods["base_oos"],
            "meta_train_months": periods["meta_train"],
            "meta_oos_months": periods["meta_oos"],
            "comparison_contract": "common OOS rows, costs, labels, global top-k basis",
            "authoritative_meta_economic_metrics": str(
                report_dir / "meta_top2_per_family_economic_metrics_common_rows.csv"
            ),
            "meta_common_row_coverage": str(report_dir / "meta_common_row_coverage.csv"),
            "meta_output_subdir": str(args.meta_output_subdir),
            "fixed_meta_base_density_id": str(args.fixed_meta_base_density_id or ""),
            "meta_outputs": [str(path) for path in sorted(meta_output_root.glob("*/manifest.json"))],
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--labels-path", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--input-feature-contract", type=Path, default=DEFAULT_INPUT_CONTRACT)
    parser.add_argument("--base-feature-list", type=Path, default=DEFAULT_INPUT_CONTRACT)
    parser.add_argument("--base-params", type=Path, default=DEFAULT_BASE_PARAMS)
    parser.add_argument("--meta-contract", type=Path, default=DEFAULT_META_CONTRACT)
    parser.add_argument(
        "--meta-params-manifest",
        type=Path,
        default=DEFAULT_META_PARAMS_MANIFEST,
        help="Frozen long/short meta parameters; feature selection still reruns per candidate.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES, default="plan")
    parser.add_argument(
        "--fixed-meta-base-density-id",
        default="",
        help=(
            "Use this completed base ledger for every representation meta arm. "
            "This isolates the incremental value of representation features at meta."
        ),
    )
    parser.add_argument(
        "--reuse-fixed-meta-handoff",
        type=Path,
        default=None,
        help=(
            "Existing handoff-only directory to reuse after exact ledger, period, "
            "selection, and cost-contract verification."
        ),
    )
    parser.add_argument(
        "--meta-candidate-ids",
        default="",
        help="Comma-separated density IDs to include in a focused meta comparison.",
    )
    parser.add_argument(
        "--downstream-density-ids",
        default="",
        help=(
            "Comma-separated density IDs to run through base and meta exactly as "
            "specified. This bypasses proxy-family promotion for a controlled "
            "density ablation."
        ),
    )
    parser.add_argument(
        "--overlap-source-density-ids",
        default="",
        help=(
            "Comma-separated diagonal stage-2 density IDs to refine. This "
            "bypasses outcome-free proxy selection for a controlled refinement."
        ),
    )
    parser.add_argument(
        "--meta-output-subdir",
        default="meta",
        help="Relative output directory for meta predictions; defaults to meta.",
    )
    parser.add_argument(
        "--report-subdir",
        default="reports",
        help="Relative output directory for reports; defaults to reports.",
    )
    parser.add_argument("--oos-months", default="2026-02,2026-03,2026-04,2026-05,2026-06")
    parser.add_argument(
        "--max-candidates-per-family",
        type=int,
        default=12,
        help="Deterministic proxy cap per family; use 0 for the complete grid.",
    )
    parser.add_argument(
        "--max-idec-final-candidates",
        type=int,
        default=12,
        help="Deterministic cap after the four IDEC proxy parents; use 0 for the full final grid.",
    )
    parser.add_argument(
        "--iic-trials-per-embedding",
        type=int,
        default=8,
        help="Deterministic hierarchical sample of the frozen-head IIC grid; 0 runs all points.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    args.oos_months = [value.strip() for value in args.oos_months.split(",") if value.strip()]
    args.meta_candidate_ids = [
        value.strip() for value in str(args.meta_candidate_ids).split(",") if value.strip()
    ]
    args.downstream_density_ids = [
        value.strip()
        for value in str(args.downstream_density_ids).split(",")
        if value.strip()
    ]
    args.overlap_source_density_ids = [
        value.strip()
        for value in str(args.overlap_source_density_ids).split(",")
        if value.strip()
    ]
    args.meta_output_subdir = _relative_output_subdir(
        str(args.meta_output_subdir), argument="--meta-output-subdir"
    )
    args.report_subdir = _relative_output_subdir(
        str(args.report_subdir), argument="--report-subdir"
    )
    return args


def main() -> int:
    args = parse_args()
    config = _load_config(args.config)
    label_contract = (
        _validate_downstream_label_contract(args.labels_path)
        if args.stage in {"base", "meta", "report", "all"}
        else None
    )
    features, args.input_contract_sha256 = _resolve_feature_contract(args)
    candidates = _candidate_manifest(
        config, max_per_family=int(args.max_candidates_per_family)
    )
    periods = split_months(args.oos_months)
    plan = {
        "schema": "alternative_representation_search_plan_v1",
        "stage": args.stage,
        "config_path": str(args.config),
        "config_sha256": _sha256(args.config),
        "downstream_label_contract": label_contract,
        "input_feature_count": len(features),
        "candidate_count": len(candidates),
        "candidate_counts_by_family": pd.Series([row.family for row in candidates]).value_counts().to_dict(),
        "periods": periods,
        "selection_contract": {
            "inner": "outcome_free_representation_and_density_proxy",
            "outer": "frozen_candidate_base_meta_economics",
            "final": "last_two_month_meta_oos",
            "representation_transductive": True,
            "base_parameter_contract": "frozen_production_params_no_hpo",
            "meta_parameter_contract": "frozen_side_params_no_hpo",
            "base_params_path": str(args.base_params),
            "base_params_sha256": _sha256(args.base_params),
            "meta_params_manifest_path": str(args.meta_params_manifest),
            "meta_params_manifest_sha256": _sha256(args.meta_params_manifest),
            "explicit_downstream_density_ids": list(args.downstream_density_ids),
        },
        "candidates": [candidate.to_dict() for candidate in candidates],
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_root / "plan.json", plan)
    if args.stage == "plan":
        print(json.dumps(_json_safe(plan), indent=2))
        return 0
    stages = (
        ["prepare", "encoders", "proxy", "idec_final", "density1", "density2", "base", "meta", "report"]
        if args.stage == "all"
        else [args.stage]
    )
    for stage in stages:
        if stage == "prepare":
            prepare(args, config, features)
        elif stage == "encoders":
            fit_encoders(args, config, candidates)
        elif stage == "proxy":
            run_proxy(args, config, candidates)
        elif stage == "idec_final":
            run_idec_final(args, config)
        elif stage == "density1":
            run_density_search(args, config, stage=1)
        elif stage == "density2":
            run_density_search(args, config, stage=2)
        elif stage == "overlap":
            run_overlap_refinement(args, config)
        elif stage == "iic":
            run_iic(args, config)
        elif stage == "base":
            run_base(args, config, features)
        elif stage == "meta":
            run_meta(args, config)
        elif stage == "report":
            report(args, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
