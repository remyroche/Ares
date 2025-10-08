"""Shared utilities for HTF feature generation."""

from typing import Dict, List, Set, Tuple

import pandas as pd

from feature_engineering_roadmap.feature_registry import FeatureFamily, FeatureRegistry
from feature_engineering_roadmap.transforms import TransformConfig, TransformType

try:
    from src.utils.tprint import tprint
except ImportError:  # pragma: no cover - fallback for environments without src package
    def tprint(*args, **kwargs):  # type: ignore
        """Fallback printer if the standard tprint utility is unavailable."""
        # Pop tprint-specific kwargs to avoid crashing the built-in print function.
        kwargs.pop("color", None)
        kwargs.pop("bold", None)
        print(*args, **kwargs)


# Mapping from Phase-1 family aliases to registry feature families
_FAMILY_TO_REGISTRY: Dict[str, Set[FeatureFamily]] = {
    'trend_level_vol': {FeatureFamily.PRICE_RETURNS, FeatureFamily.VOLATILITY},
    'oscillators': {FeatureFamily.MEAN_REVERSION},
    'anchors': {FeatureFamily.ANCHORS_TOD},
}


def build_htf_family_catalog(
    registry: FeatureRegistry,
) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """Build HTF family listings and reverse lookup from the registry."""
    tprint(
        "[HTF] Building high time-frame family catalog",
        f"families={len(_FAMILY_TO_REGISTRY)}",
    )
    family_to_features: Dict[str, List[str]] = {}
    feature_to_family: Dict[str, str] = {}

    for alias, registry_families in _FAMILY_TO_REGISTRY.items():
        features: List[str] = []
        for registry_family in registry_families:
            features.extend(registry.get_features_by_family(registry_family))

        # Preserve deterministic ordering for reproducibility
        unique_features = sorted(set(features))
        if not unique_features:
            continue

        family_to_features[alias] = unique_features
        for feature_name in unique_features:
            feature_to_family[feature_name] = alias

    tprint(
        "[HTF] Completed catalog build",
        f"aliases={len(family_to_features)}",
        f"features={len(feature_to_family)}",
    )
    return family_to_features, feature_to_family


def resample_htf_series(
    series: pd.Series, lookback_minutes: int, feature_family: FeatureFamily
) -> pd.Series:
    """Apply the standard HTF resampling strategy for a feature family."""
    if series.empty:
        tprint(
            "[HTF] Resample skipped due to empty series",
            f"lookback={lookback_minutes}",
            f"family={feature_family.name}",
        )
        return series

    rule = f"{lookback_minutes}min"
    tprint(
        "[HTF] Resampling series",
        f"rule={rule}",
        f"family={feature_family.name}",
    )

    if feature_family == FeatureFamily.MEAN_REVERSION:
        return series.resample(rule).mean()

    # Trend, volatility, anchors, and other context use the latest observation
    return series.resample(rule).last()


def format_transform_suffix(transform_config: TransformConfig) -> str:
    """Format the transform name with relevant parameterization."""
    suffix = transform_config.transform_type.value
    tprint(
        "[HTF] Formatting transform suffix",
        f"transform={transform_config.transform_type.name}",
    )

    if transform_config.transform_type == TransformType.EWZ:
        halflife = transform_config.params.get('halflife')
        if halflife is not None:
            tprint(
                "[HTF] Appending EWZ halflife",
                f"halflife={halflife}",
            )
            return f"{suffix}{halflife}"

    return suffix
