from ..standardized_parquet_handler import standardized_parquet_handler
"""Feature interaction engine for creating polynomial feature interactions."""

from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import numpy as np
from itertools import combinations

from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls

class FeatureInteractionEngine:
    """Engine for creating feature interactions and polynomial features."""

    @log_important_calls
    def __init__(self, degree: int = 2, max_features: int = 100) -> None:
        """Initialize the feature interaction engine.

        Args:
            degree: Maximum degree for polynomial interactions
            max_features: Maximum number of interaction features to generate
        """
        self.degree = degree
        self.max_features = max_features

    def create_interactions(self, data: pd.DataFrame, feature_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Create polynomial feature interactions.

        Args:
            data: Input DataFrame with features
            feature_columns: Specific columns to use for interactions (default: all numeric)

        Returns:
            DataFrame with interaction features
        """
        if feature_columns is None:
            # Use all numeric columns
            feature_columns = data.select_dtypes(include=[np.number]).columns.tolist()

        # Limit to reasonable number of features
        if len(feature_columns) > 20:
            feature_columns = feature_columns[:20]

        interaction_features = pd.DataFrame(index=data.index)

        # Create pairwise interactions (degree 2)
        if self.degree >= 2:
            for col1, col2 in combinations(feature_columns, 2):
                if len(interaction_features.columns) >= self.max_features:
                    break

                try:
                    # Product interaction
                    interaction_name = f"{col1}_{col2}_product"
                    interaction_features[interaction_name] = data[col1] * data[col2]

                    # Ratio interaction (avoid division by zero)
                    if data[col2].abs().min() > 1e-10:
                        ratio_name = f"{col1}_{col2}_ratio"
                        interaction_features[ratio_name] = data[col1] / data[col2]

                    # Difference interaction
                    diff_name = f"{col1}_{col2}_diff"
                    interaction_features[diff_name] = data[col1] - data[col2]

                except Exception as e:
                    # Skip problematic interactions
                    continue

        # Create higher-degree interactions if degree > 2
        if self.degree > 2:
            for d in range(3, self.degree + 1):
                for combo in combinations(feature_columns, d):
                    if len(interaction_features.columns) >= self.max_features:
                        break

                    try:
                        # Create polynomial interaction
                        combo_name = "_".join(combo) + f"_deg{d}"
                        interaction_value = data[combo[0]]

                        for col in combo[1:]:
                            interaction_value *= data[col]

                        interaction_features[combo_name] = interaction_value

                    except Exception as e:
                        continue

        return interaction_features

    def create_polynomial_features(self, data: pd.DataFrame, feature_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Create polynomial features (squares, cubes, etc.).

        Args:
            data: Input DataFrame with features
            feature_columns: Specific columns to use (default: all numeric)

        Returns:
            DataFrame with polynomial features
        """
        if feature_columns is None:
            feature_columns = data.select_dtypes(include=[np.number]).columns.tolist()

        poly_features = pd.DataFrame(index=data.index)

        for col in feature_columns:
            if len(poly_features.columns) >= self.max_features:
                break

            try:
                # Square
                poly_features[f"{col}_squared"] = data[col] ** 2

                # Cube (if degree >= 3)
                if self.degree >= 3:
                    poly_features[f"{col}_cubed"] = data[col] ** 3

                # Square root (for positive values)
                if (data[col] >= 0).all():
                    poly_features[f"{col}_sqrt"] = np.sqrt(data[col])

                # Log transformation (for positive values)
                if (data[col] > 0).all():
                    poly_features[f"{col}_log"] = np.log(data[col] + 1e-10)

            except Exception as e:
                continue

        return poly_features
