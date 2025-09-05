"""Feature engineering steps package."""

# Re-export for convenience if module users import from package
try:
    from .step06_advanced_features import AdvancedFeatureEngineeringStep  # noqa: F401
except Exception:
    # Keep package importable even if step module has issues
    AdvancedFeatureEngineeringStep = None  # type: ignore

