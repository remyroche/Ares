# Enhanced validation wrapper

from datetime import datetime
from typing import Any

import pandas as pd

from src.utils.data_quality_validator import validate_features


def enhanced_validate_features(
    data: pd.DataFrame,
    dataset_name: str = "features",
) -> dict[str, Any]:
    """Enhanced validation with detailed logging"""

    # Run original validation
    results = validate_features(data, dataset_name)

    # Enhanced logging
    detailed_report={
        "timestamp": datetime.now().isoformat(),
        "dataset_name": dataset_name, "data_shape": data.shape,
        "total_features": len(data.columns),
    }

    # Combine
    if isinstance(results, dict):
        detailed_report.update(results)

    return detailed_report


# Usage in step1_7_hmm_regime_discovery.py:
# Replace: validation_results=validate_features(features_df = f"features_{tf}")
# With: validation_results=enhanced_validate_features(features_df = f"features_{tf}")
