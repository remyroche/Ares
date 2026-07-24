"""Staged, outcome-free search helpers for alternatives to sequential DAE/GMM."""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

from .alternative_latent_encoders import (
    AlternativeLatentEncoder,
    EncoderConfig,
    NativeLatentOutput,
)
from .representation_proxy_metrics import (
    GmmPanelFit,
    GmmPanelSpec,
    align_diagonal_gmm_components,
    diagonal_gmm_statistics,
    entropy_distribution_diagnostics,
    fit_common_gmm_panel,
    normalized_entropy,
    occupancy_excess_instability,
    perturbation_consistency,
    reorder_posteriors_to_reference,
)


@dataclass(frozen=True)
class EncoderCandidate:
    candidate_id: str
    family: str
    config: dict[str, Any]
    stage: str = "proxy"
    output_mode: str = "embedding_gmm"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FittedRepresentation:
    candidate: EncoderCandidate
    encoder_path: str
    native_cache_path: str
    panel_path: str
    best_panel_id: str
    proxy_path: str


def _candidate_id(family: str, values: Mapping[str, Any]) -> str:
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]
    concise = [family]
    for key in ("kind", "latent_dim", "n_clusters", "objective", "policy", "output_mode"):
        if key in values:
            concise.append(f"{key}-{values[key]}")
    return "__".join(concise) + f"__{digest}"


def baseline_candidates(config: Mapping[str, Any]) -> list[EncoderCandidate]:
    candidates: list[EncoderCandidate] = []
    for row in config.get("baseline_learning_curve", []):
        values = {
            "kind": "dae",
            "latent_dim": 16,
            "dae_max_train_rows": int(row["dae_rows"]),
            "gmm_rows": int(row["gmm_rows"]),
            "epochs": 80,
            "conditional": row.get("conditional"),
        }
        candidates.append(
            EncoderCandidate(str(row["id"]), "incumbent", values, "saturation")
        )
    return candidates


def frozen_incumbent_candidates(config: Mapping[str, Any]) -> list[EncoderCandidate]:
    """Expose legacy fitted encoders as read-only comparison controls.

    The state is never fitted again. The runner only applies it to the active
    reference frame so its downstream comparison uses the current feature
    values and row universe.
    """

    candidates: list[EncoderCandidate] = []
    for row in config.get("frozen_incumbent_controls", []):
        state_path = str(row.get("encoder_state_path", ""))
        if not state_path:
            raise ValueError("Frozen incumbent control requires encoder_state_path")
        values = {
            "kind": "dae",
            "latent_dim": int(row.get("latent_dim", 16)),
            "dae_max_train_rows": int(row.get("dae_rows", 0)),
            "gmm_rows": int(row.get("gmm_rows", 100_000)),
            "frozen_encoder_state_path": state_path,
            "source_reference_contract_path": str(
                row.get("source_reference_contract_path", "")
            ),
        }
        candidates.append(
            EncoderCandidate(str(row["id"]), "legacy_incumbent", values, "frozen")
        )
    return candidates


def vade_candidates(config: Mapping[str, Any]) -> list[EncoderCandidate]:
    grid = config["vade"]
    keys = (
        "latent_dim",
        "components",
        "kl_weight",
        "kl_warmup_fraction",
        "initialization",
        "min_effective_occupancy",
        "reconstruction_objective",
    )
    output: list[EncoderCandidate] = []
    for combination in itertools.product(*(grid[key] for key in keys)):
        raw = dict(zip(keys, combination))
        values = {
            "kind": "vade",
            "latent_dim": int(raw.pop("latent_dim")),
            "n_clusters": int(raw.pop("components")),
            **raw,
        }
        output.append(
            EncoderCandidate(_candidate_id("vade", values), "vade", values)
        )
    return output


def idec_candidates(config: Mapping[str, Any], *, stage: str) -> list[EncoderCandidate]:
    grid = config["idec"][stage]
    keys = tuple(grid)
    output: list[EncoderCandidate] = []
    for combination in itertools.product(*(grid[key] for key in keys)):
        raw = dict(zip(keys, combination))
        values = {
            "kind": "idec",
            "latent_dim": int(raw.pop("latent_dim")),
            "target_update_frequency": int(raw.pop("target_update_frequency")),
            "student_t_df": float(raw.pop("student_t_df")),
            **raw,
        }
        output_mode = str(values.pop("output_mode", "direct"))
        candidate_values = {**values, "output_mode": output_mode}
        output.append(
            EncoderCandidate(
                _candidate_id("idec", candidate_values),
                "idec",
                values,
                stage,
                output_mode,
            )
        )
    return output


def idec_final_candidates(
    config: Mapping[str, Any],
    parents: Sequence[EncoderCandidate],
) -> list[EncoderCandidate]:
    """Expand only promoted IDEC proxy parents into the final two-stage grid."""

    final = idec_candidates(config, stage="final")
    output: list[EncoderCandidate] = []
    for parent in parents:
        for candidate in final:
            values = {
                **candidate.config,
                "n_clusters": int(parent.config.get("n_clusters", 6)),
                "pretraining_fraction": float(parent.config.get("pretraining_fraction", 0.66)),
                "proxy_parent_id": parent.candidate_id,
            }
            identity = {**values, "output_mode": candidate.output_mode}
            output.append(
                EncoderCandidate(
                    _candidate_id("idec_final", identity),
                    "idec",
                    values,
                    "final",
                    candidate.output_mode,
                )
            )
    return output


def ssl_candidates(config: Mapping[str, Any]) -> list[EncoderCandidate]:
    section = config["ssl"]
    kind_map = {
        "denoising_reconstruction": "masked",
        "masked_reconstruction": "masked",
        "scarf": "scarf",
        "vicreg": "vicreg",
    }
    output: list[EncoderCandidate] = []
    group_count = max(1, int(config.get("feature_group_count_hint", 8)))
    for objective, (policy_name, policy), view_pair in itertools.product(
        section["objectives"], section["policies"].items(), section["view_pairs"]
    ):
        if str(objective) in {"denoising_reconstruction", "masked_reconstruction"} and str(view_pair) != "weak_strong":
            continue
        values = {
            "kind": kind_map[str(objective)],
            "objective": str(objective),
            "ssl_objective": str(objective),
            "policy": str(policy_name),
            "view_pair": str(view_pair),
            "ssl_view_pair": str(view_pair),
            "latent_dim": 16,
            "element_mask_rate": float(policy.get("element_mask", 0.0)),
            "corruption_rate": 0.0,
            "additive_noise_std": float(policy.get("noise", 0.0)),
            # The encoder samples each non-side group independently. A one-group
            # policy therefore maps to an expected one group per view.
            "whole_feature_group_mask_rate": float(policy.get("group_masks", 0))
            / float(group_count),
            "group_donor_replacement_rate": (
                float(policy.get("element_replace", 0.0))
                if float(policy.get("element_replace", 0.0)) > 0.0
                else (
                    1.0 / float(group_count)
                    if str(policy.get("donor", "none")) != "none"
                    else 0.0
                )
            ),
            "donor_policy": str(policy.get("donor", "none")),
            "side_feature_group": "side",
        }
        output.append(
            EncoderCandidate(_candidate_id("ssl", values), "ssl", values)
        )
    return output


def all_encoder_candidates(config: Mapping[str, Any]) -> list[EncoderCandidate]:
    return [
        *baseline_candidates(config),
        *frozen_incumbent_candidates(config),
        *vade_candidates(config),
        *idec_candidates(config, stage="proxy"),
        *ssl_candidates(config),
    ]


def cap_candidates(
    candidates: Sequence[EncoderCandidate],
    *,
    max_per_family: int,
    seed: int,
) -> list[EncoderCandidate]:
    """Cap grids while preserving coverage of each configured search axis.

    A random cap can omit an entire objective, corruption policy, component
    count, or latent dimension. The greedy pass first covers rare axis values,
    then fills any remaining budget from a seeded deterministic order.
    """

    if int(max_per_family) <= 0:
        return list(candidates)
    output: list[EncoderCandidate] = []
    for family in sorted({candidate.family for candidate in candidates}):
        local = [candidate for candidate in candidates if candidate.family == family]
        if len(local) > int(max_per_family):
            rng = np.random.default_rng(int(seed) + sum(map(ord, str(family))))
            tie_order = rng.permutation(len(local)).tolist()
            tie_rank = {int(index): rank for rank, index in enumerate(tie_order)}

            def tokens(candidate: EncoderCandidate) -> set[tuple[str, str]]:
                values = {
                    **candidate.config,
                    "output_mode": candidate.output_mode,
                }
                return {
                    (str(key), json.dumps(value, sort_keys=True, default=str))
                    for key, value in values.items()
                    if value is not None and key not in {"kind", "proxy_parent_id"}
                }

            token_sets = [tokens(candidate) for candidate in local]
            frequency: dict[tuple[str, str], int] = {}
            for values in token_sets:
                for token in values:
                    frequency[token] = frequency.get(token, 0) + 1
            uncovered = set(frequency)
            selected_indices: list[int] = []
            remaining = set(range(len(local)))
            while remaining and len(selected_indices) < int(max_per_family):
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
                selected_indices.append(int(best))
                remaining.remove(best)
                uncovered.difference_update(token_sets[best])
            local = [local[index] for index in selected_indices]
        output.extend(local)
    return output


def encoder_config(candidate: EncoderCandidate, *, seed: int, device: str) -> EncoderConfig:
    accepted = {field.name for field in fields(EncoderConfig)}
    values = {
        key: value
        for key, value in candidate.config.items()
        if key in accepted and value is not None
    }
    values.setdefault("random_state", int(seed))
    values.setdefault("device", str(device))
    values.setdefault("epochs", 20)
    return EncoderConfig(**values)


def fit_encoder_candidate(
    candidate: EncoderCandidate,
    *,
    fit_values: np.ndarray,
    fit_sides: Sequence[Any],
    transform_values: np.ndarray,
    transform_sides: Sequence[Any],
    feature_group_indices: Mapping[str, Sequence[int]],
    fit_donor_regime_labels: Sequence[Any] | None = None,
    idec_pretraining_state: Mapping[str, Any] | None = None,
    output_dir: Path,
    seed: int,
    device: str = "auto",
) -> tuple[AlternativeLatentEncoder, NativeLatentOutput]:
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder = AlternativeLatentEncoder(encoder_config(candidate, seed=seed, device=device))
    fit_kwargs: dict[str, Any] = {
        "sides": np.asarray(fit_sides),
        "groups": None,
    }
    # Newer adapters consume feature-group geometry directly. Keep compatibility
    # with serialized v1 adapters while the runner remains resumable.
    if "feature_group_indices" in encoder.fit.__code__.co_varnames:
        fit_kwargs["feature_group_indices"] = feature_group_indices
        fit_kwargs["donor_regime_labels"] = fit_donor_regime_labels
    if idec_pretraining_state is not None:
        fit_kwargs["pretraining_state"] = idec_pretraining_state
    encoder.fit(np.asarray(fit_values, dtype=np.float32), **fit_kwargs)
    native = encoder.transform_native(
        np.asarray(transform_values, dtype=np.float32),
        sides=np.asarray(transform_sides),
    )
    encoder_path = output_dir / "encoder.joblib"
    native_path = output_dir / "native_outputs.npz"
    # The latent matrix is read repeatedly by the common GMM panel, density
    # stages, and downstream materialization.  Keep an uncompressed mmapable
    # copy so those stages do not repeatedly inflate an NPZ archive.
    np.save(output_dir / "native_latent.npy", np.asarray(native.latent, dtype=np.float32), allow_pickle=False)
    np.save(
        output_dir / "native_reconstruction_error.npy",
        _optional_array(native.reconstruction_error),
        allow_pickle=False,
    )
    np.save(
        output_dir / "native_cluster_probabilities.npy",
        _optional_array(native.cluster_probabilities),
        allow_pickle=False,
    )
    np.save(output_dir / "native_mean.npy", _optional_array(native.mean), allow_pickle=False)
    np.save(output_dir / "native_logvar.npy", _optional_array(native.logvar), allow_pickle=False)
    joblib.dump(encoder.to_state(), encoder_path, compress=3)
    np.savez(
        native_path,
        latent=np.asarray(native.latent, dtype=np.float32),
        reconstruction_error=_optional_array(native.reconstruction_error),
        cluster_probabilities=_optional_array(native.cluster_probabilities),
        mean=_optional_array(native.mean),
        logvar=_optional_array(native.logvar),
    )
    (output_dir / "candidate.json").write_text(
        json.dumps(candidate.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return encoder, native


def _optional_array(values: np.ndarray | None) -> np.ndarray:
    return np.empty((0,), dtype=np.float32) if values is None else np.asarray(values, dtype=np.float32)


def _posterior_margin(probabilities: np.ndarray) -> np.ndarray:
    if probabilities.shape[1] <= 1:
        return probabilities[:, 0].astype(np.float32, copy=False)
    partition = np.partition(probabilities, -2, axis=1)[:, -2:]
    return (partition[:, 1] - partition[:, 0]).astype(np.float32, copy=False)


def _empirical_percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    sorted_reference = np.sort(np.asarray(reference, dtype=np.float64))
    if not len(sorted_reference):
        return np.zeros(len(values), dtype=np.float32)
    positions = np.searchsorted(sorted_reference, np.asarray(values, dtype=np.float64), side="right")
    return (positions / float(len(sorted_reference))).astype(np.float32)


def panel_identifier(panel: GmmPanelFit) -> str:
    return (
        f"k{panel.spec.n_components}_{panel.spec.covariance_type}_"
        f"reg{panel.spec.reg_covar:g}_seed{panel.seed}"
    )


def evaluate_common_panel(
    latent: np.ndarray,
    *,
    fit_indices: np.ndarray,
    perturb_latent: np.ndarray,
    strata: Mapping[str, Sequence[Any]],
    seeds: Sequence[int],
    specs: Sequence[GmmPanelSpec],
    n_init: int = 3,
    nearby_fit_indices: Sequence[np.ndarray] = (),
    retry_reg_covars: Sequence[float] = (0.01, 0.03, 0.1),
) -> tuple[list[GmmPanelFit], pd.DataFrame]:
    """Evaluate each cached embedding against a small robust GMM panel."""

    z = np.asarray(latent, dtype=np.float32)
    fit_indices = np.asarray(fit_indices, dtype=np.int64)
    z_fit = z[fit_indices]
    heldout_mask = np.ones(len(z), dtype=bool)
    heldout_mask[fit_indices] = False
    heldout_indices = np.flatnonzero(heldout_mask)
    if len(heldout_indices) > 50_000:
        heldout_indices = heldout_indices[-50_000:]
    fit_failures: list[dict[str, Any]] = []
    fits = fit_common_gmm_panel(
        z_fit,
        seeds=seeds,
        specs=specs,
        n_init=int(n_init),
        retry_reg_covars=retry_reg_covars,
        failure_records=fit_failures,
    )
    for resample_number, resample_indices in enumerate(nearby_fit_indices):
        local = z[np.asarray(resample_indices, dtype=np.int64)]
        fits.extend(
            fit_common_gmm_panel(
                local,
                seeds=(int(seeds[0]) + 10_000 + resample_number,),
                specs=specs,
                n_init=int(n_init),
                retry_reg_covars=retry_reg_covars,
                failure_records=fit_failures,
            )
        )
    rows: list[dict[str, Any]] = []
    grouped: dict[tuple[int, str, float], list[GmmPanelFit]] = {}
    for fit in fits:
        grouped.setdefault(
            (fit.spec.n_components, fit.spec.covariance_type, fit.spec.reg_covar), []
        ).append(fit)
    for panel_key, panel_runs in grouped.items():
        reference = panel_runs[0]
        reference_stats = diagonal_gmm_statistics(z, reference.state)
        posterior_runs: dict[str, np.ndarray] = {}
        for run in panel_runs:
            stats = diagonal_gmm_statistics(z, run.state)
            probabilities = stats["posteriors"]
            if run is not reference:
                alignment = align_diagonal_gmm_components(reference.state, run.state)
                probabilities = reorder_posteriors_to_reference(probabilities, alignment)
            posterior_runs[panel_identifier(run)] = probabilities
        entropy_report = entropy_distribution_diagnostics(reference_stats["posteriors"])
        occupancy = occupancy_excess_instability(
            posterior_runs,
            expected_weights=reference.state["weights"],
            strata=strata,
        )
        perturbed_stats = diagonal_gmm_statistics(perturb_latent, reference.state)
        consistency = perturbation_consistency(
            z,
            perturb_latent,
            reference_stats["posteriors"],
            perturbed_stats["posteriors"],
            reference_stats["ood_score"],
            perturbed_stats["ood_score"],
        )
        entropy_values = normalized_entropy(reference_stats["posteriors"])
        calendar_values = np.asarray(strata.get("calendar_period", np.repeat("all", len(z))))
        temporal_entropy_means = [
            float(np.mean(entropy_values[calendar_values == value]))
            for value in np.unique(calendar_values)
        ]
        min_distance = np.min(reference_stats["mahalanobis"], axis=1)
        entropy_ood_corr = (
            float(np.corrcoef(entropy_values, reference_stats["ood_score"])[0, 1])
            if np.std(entropy_values) > 1e-12
            and np.std(reference_stats["ood_score"]) > 1e-12
            else 0.0
        )
        entropy_distance_corr = (
            float(np.corrcoef(entropy_values, min_distance)[0, 1])
            if np.std(entropy_values) > 1e-12 and np.std(min_distance) > 1e-12
            else 0.0
        )
        occupancy_overall = next(row for row in occupancy if row["stratum"] == "overall")
        record = {
                "panel_id": panel_identifier(reference),
                "n_components": int(panel_key[0]),
                "covariance_type": str(panel_key[1]),
                "reg_covar": float(panel_key[2]),
                "heldout_mean_nll": float(
                    np.mean(reference_stats["ood_score"][heldout_indices])
                    if len(heldout_indices)
                    else np.mean(reference_stats["ood_score"])
                ),
                "min_effective_occupancy": float(
                    np.min(reference.state["weights"]) * int(panel_key[0])
                ),
                "min_effective_occupancy_raw": float(
                    np.min(reference.state["weights"])
                ),
                "entropy_mass_near_zero": float(np.mean(entropy_values <= 0.05)),
                "entropy_mass_near_one": float(np.mean(entropy_values >= 0.95)),
                "entropy_boundary_mass": float(
                    np.mean(_posterior_margin(reference_stats["posteriors"]) <= 0.10)
                ),
                "entropy_temporal_variation": float(np.std(temporal_entropy_means)),
                "entropy_ood_correlation": entropy_ood_corr,
                "entropy_min_distance_correlation": entropy_distance_corr,
                "seed_excess_instability": float(
                    occupancy_overall["mean_pairwise_excess_l1"]
                ),
                "latent_perturb_cosine": float(consistency["latent_cosine_mean"]),
                "posterior_perturb_tv": float(consistency["posterior_tv_mean"]),
                "ood_rank_consistency": float(consistency["ood_rank_correlation"]),
                "entropy_mean_diagnostic_only": float(entropy_report["mean"]),
            }
        for stratum_name in ("symbol", "calendar_period", "side", "major_market_regime"):
            local_instability = [
                float(item["mean_pairwise_excess_l1"])
                for item in occupancy
                if item["stratum"] == stratum_name
            ]
            if local_instability:
                record[f"seed_excess_instability_{stratum_name}_mean"] = float(
                    np.mean(local_instability)
                )
                record[f"seed_excess_instability_{stratum_name}_max"] = float(
                    np.max(local_instability)
                )
        rows.append(record)
    report = pd.DataFrame(rows)
    if not report.empty:
        report["proxy_score"] = robust_panel_proxy_score(report)
    report.attrs["fit_failures"] = fit_failures
    return fits, report


def robust_panel_proxy_score(report: pd.DataFrame) -> np.ndarray:
    """Fixed outcome-free rank aggregation; no economic result may retune it."""

    def good_rank(column: str, *, ascending: bool) -> np.ndarray:
        values = pd.to_numeric(report[column], errors="coerce")
        return values.rank(pct=True, ascending=ascending, method="average").to_numpy(float)

    degeneracy = (
        pd.to_numeric(report["entropy_mass_near_zero"], errors="coerce").fillna(1.0)
        + pd.to_numeric(report["entropy_mass_near_one"], errors="coerce").fillna(1.0)
    ).to_numpy(float)
    degeneracy_rank = pd.Series(degeneracy).rank(pct=True, ascending=False).to_numpy(float)
    score = (
        0.25 * good_rank("seed_excess_instability", ascending=False)
        + 0.20 * good_rank("posterior_perturb_tv", ascending=False)
        + 0.15 * good_rank("latent_perturb_cosine", ascending=True)
        + 0.15 * good_rank("ood_rank_consistency", ascending=True)
        + 0.10 * degeneracy_rank
        + 0.10 * good_rank("min_effective_occupancy", ascending=True)
        + 0.05 * good_rank("heldout_mean_nll", ascending=False)
    )
    return np.asarray(score, dtype=np.float64)


def select_family_finalists(
    candidate_summary: pd.DataFrame,
    *,
    top_per_family: int = 3,
) -> pd.DataFrame:
    required = {"candidate_id", "family", "best_robust_panel_score"}
    missing = sorted(required.difference(candidate_summary.columns))
    if missing:
        raise ValueError(f"Candidate summary is missing columns: {missing}")
    ranked = candidate_summary.sort_values(
        ["family", "best_robust_panel_score", "candidate_id"],
        ascending=[True, False, True],
        kind="mergesort",
    ).copy()
    ranked["family_rank"] = ranked.groupby("family", observed=True).cumcount() + 1
    ranked["promoted"] = ranked["family_rank"].le(int(top_per_family))
    return ranked


def materialize_representation_features(
    *,
    keys: pd.DataFrame,
    native: NativeLatentOutput,
    panel: GmmPanelFit,
    reference_indices: np.ndarray,
    perturbation_consistency_score: float | np.ndarray | None = None,
    output_mode: str = "embedding_gmm",
    density_statistics: Mapping[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Create a common continuous output schema for base/meta sidecars."""

    latent = np.asarray(native.latent, dtype=np.float32)
    if len(keys) != len(latent):
        raise ValueError("Representation keys and latent rows are not aligned")
    output = keys.loc[:, ["__ts__", "__symbol__", "side"]].reset_index(drop=True).copy()
    for index in range(latent.shape[1]):
        output[f"repr_latent_{index:02d}"] = latent[:, index]
    if output_mode not in {"embedding_gmm", "direct", "embedding_only"}:
        raise ValueError(f"Unknown representation output mode: {output_mode}")
    if output_mode == "embedding_gmm":
        stats = (
            diagonal_gmm_statistics(latent, panel.state)
            if density_statistics is None
            else density_statistics
        )
        probabilities = np.asarray(stats["posteriors"], dtype=np.float32)
        mahal = np.asarray(stats["mahalanobis"], dtype=np.float32)
        for index in range(probabilities.shape[1]):
            output[f"repr_component_posterior_{index:02d}"] = probabilities[:, index]
            output[f"repr_component_mahalanobis_{index:02d}"] = mahal[:, index]
        output["repr_entropy_norm"] = normalized_entropy(probabilities)
        output["repr_posterior_margin"] = _posterior_margin(probabilities)
        output["repr_expected_mahalanobis"] = np.sum(probabilities * mahal, axis=1)
        output["repr_min_mahalanobis"] = np.min(mahal, axis=1)
        novelty = np.asarray(stats["ood_score"], dtype=np.float32)
        output["repr_novelty_raw"] = novelty
        output["repr_novelty_reference_pct"] = _empirical_percentile(
            novelty[np.asarray(reference_indices, dtype=np.int64)], novelty
        )
    if native.reconstruction_error is not None:
        output["repr_reconstruction_error"] = np.asarray(
            native.reconstruction_error, dtype=np.float32
        )
    if native.logvar is not None:
        variance = np.exp(np.clip(np.asarray(native.logvar, dtype=np.float32), -8.0, 6.0))
        output["repr_uncertainty_mean"] = variance.mean(axis=1)
        for index in range(variance.shape[1]):
            output[f"repr_posterior_variance_{index:02d}"] = variance[:, index]
        if native.reconstruction_error is not None:
            mean_sq = (
                np.square(np.asarray(native.mean, dtype=np.float32))
                if native.mean is not None
                else 0.0
            )
            elbo_novelty = np.asarray(native.reconstruction_error, dtype=np.float32) + 0.5 * np.mean(
                variance + mean_sq - np.asarray(native.logvar, dtype=np.float32) - 1.0,
                axis=1,
            )
            output["repr_elbo_novelty_raw"] = elbo_novelty
            output["repr_elbo_novelty_reference_pct"] = _empirical_percentile(
                elbo_novelty[np.asarray(reference_indices, dtype=np.int64)],
                elbo_novelty,
            )
    if native.mean is not None:
        for index in range(np.asarray(native.mean).shape[1]):
            output[f"repr_posterior_mean_{index:02d}"] = np.asarray(native.mean)[:, index]
    if native.cluster_probabilities is not None and output_mode != "embedding_only":
        native_prob = np.asarray(native.cluster_probabilities, dtype=np.float32)
        for index in range(native_prob.shape[1]):
            output[f"repr_native_posterior_{index:02d}"] = native_prob[:, index]
        output["repr_native_entropy_norm"] = normalized_entropy(native_prob)
        output["repr_native_posterior_margin"] = _posterior_margin(native_prob)
    if perturbation_consistency_score is not None:
        score = np.asarray(perturbation_consistency_score, dtype=np.float32)
        output["repr_perturbation_consistency"] = (
            np.full(len(output), float(score), dtype=np.float32)
            if score.ndim == 0
            else score
        )
    return output


__all__ = [
    "EncoderCandidate",
    "FittedRepresentation",
    "all_encoder_candidates",
    "baseline_candidates",
    "cap_candidates",
    "encoder_config",
    "evaluate_common_panel",
    "fit_encoder_candidate",
    "idec_candidates",
    "idec_final_candidates",
    "materialize_representation_features",
    "panel_identifier",
    "robust_panel_proxy_score",
    "select_family_finalists",
    "ssl_candidates",
    "vade_candidates",
]
