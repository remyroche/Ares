"""Helpers for optional generated model/context features.

These features are useful as representation diagnostics, but they should not
be hard live-data requirements. If they cannot be materialized at decision
time, inference should neutral-fill them while keeping core market/model
features strict.
"""

from __future__ import annotations


OPTIONAL_GENERATED_MODEL_FEATURE_EXACT = {
    "cluster_entropy",
    "cluster_entropy_norm",
    "cluster_entropy_delta_1",
    "cluster_entropy_accel_1",
    "gmm_cluster_id",
    "gmm_entropy",
    "mahalanobis_distance",
    "cluster_speed",
    "cluster_acceleration",
    "min_mahalanobis",
    "min_mahalanobis_delta_1",
    "expected_mahalanobis",
    "expected_mahalanobis_delta_1",
    "expected_mahalanobis_accel_1",
    "cluster_t",
    "time_since_cluster_change",
    "rolling_cluster_stability",
    "cluster_flip_count_20",
    "ae_reconstruction_error",
    "dae_reconstruction_error",
    "dae_reconstruction_error_zscore",
    "latent_mahalanobis_drift",
    "latent_speed",
    "latent_acceleration",
    "meta_sel_ood_abs_z_mean",
    "meta_sel_ood_abs_z_max",
    "meta_sel_ood_abs_z_p95",
    "meta_sel_ood_iqr_exceed_frac",
    "meta_sel_ood_missing_frac",
    "meta_sel_ood_centroid_l2",
}

OPTIONAL_GENERATED_MODEL_FEATURE_PREFIXES = (
    "dae_",
    "ae_",
    "gmm_",
    "meta_sel_ood_",
)


def is_optional_generated_model_feature_key(name: object) -> bool:
    """Return True for generated representation features that are optional live."""

    key = str(name or "").strip()
    if not key:
        return False
    lower = key.lower()
    if lower in OPTIONAL_GENERATED_MODEL_FEATURE_EXACT:
        return True
    if any(lower.endswith(f"_{exact}") for exact in OPTIONAL_GENERATED_MODEL_FEATURE_EXACT):
        return True
    if lower.startswith(OPTIONAL_GENERATED_MODEL_FEATURE_PREFIXES):
        return True
    if "_dae_" in lower or "_gmm_" in lower:
        return True
    if "_raw_state_svd_" in lower or lower.startswith("raw_state_svd_"):
        return True
    return False
