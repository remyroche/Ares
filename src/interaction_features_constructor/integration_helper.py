"""
Integration Helper

Provides helper functions to integrate the feature interaction constructor
with the existing training and live trading pipeline.
"""

import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path

try:
    import polars as pl  # type: ignore[import]
except Exception:  # pragma: no cover - optional dependency
    pl = None

from src.interaction_features_constructor.feature_calculator import FeatureCalculator
from src.interaction_features_constructor.feature_metadata_store import FeatureMetadataStore


class TrainingPipelineIntegration:
    """
    Integration helper for the training pipeline.

    Use this in feature_generation_final_feature_selection_step to save
    metadata about selected features.
    """

    @staticmethod
    def save_feature_metadata(
        selected_features_dict: Dict[str, List[str]],
        config: Dict[str, Any],
        artifacts_dir: str = 'artifacts'
    ) -> Dict[str, str]:
        """
        Save feature metadata for all feature sets (60, 50, 40).

        Args:
            selected_features_dict: Dict mapping size to feature list
                e.g., {'selected_features_60': [...], 'selected_features_50': [...]}
            config: Configuration dict with symbol, exchange, timeframe, etc.
            artifacts_dir: Directory to save metadata files

        Returns:
            Dict mapping feature set name to saved file path
        """
        saved_paths = {}

        for set_name, features in selected_features_dict.items():
            if set_name.startswith('selected_features_'):
                # Extract size (e.g., 60, 50, 40)
                size = set_name.split('_')[-1]

                # Create metadata store
                store = FeatureMetadataStore()
                store.create_from_selection(
                    selected_features=features,
                    symbol=config.get('symbol'),
                    exchange=config.get('exchange'),
                    timeframe=config.get('timeframe'),
                    direction=config.get('direction'),
                    model=config.get('execution_mode')
                )

                # Save metadata
                from datetime import datetime
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"feature_metadata_{size}_{config.get('symbol', 'UNKNOWN')}_{timestamp}.json"
                filepath = Path(artifacts_dir) / 'feature_metadata' / filename

                store.save(str(filepath))
                saved_paths[set_name] = str(filepath)

                print(f"Saved feature metadata for {set_name}: {filepath}")
                print(f"  - Total features: {len(features)}")
                print(f"  - Base features required: {len(store.get_base_features_required())}")

        return saved_paths

    @staticmethod
    def add_to_final_feature_selection_step(
        feature_sets: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Add this method to feature_generation_final_feature_selection_step._generate_artifacts().

        Example usage in feature_generation_final_feature_selection_step.py:

        ```python
        # In _generate_artifacts method, after creating feature_sets:

        from src.interaction_features_constructor.integration_helper import TrainingPipelineIntegration

        # Save feature metadata for live trading reconstruction
        metadata_paths = TrainingPipelineIntegration.add_to_final_feature_selection_step(
            feature_sets, config
        )
        ```

        Args:
            feature_sets: Dict containing selected_features_60, selected_features_50, etc.
            config: Configuration dict

        Returns:
            Dict mapping feature set name to metadata file path
        """
        # Extract only the feature lists
        feature_lists = {
            k: v for k, v in feature_sets.items()
            if k.startswith('selected_features_') and isinstance(v, list)
        }

        if not feature_lists:
            print("Warning: No selected features found in feature_sets")
            return {}

        return TrainingPipelineIntegration.save_feature_metadata(
            feature_lists,
            config
        )


class LiveTradingIntegration:
    """
    Integration helper for live trading system.

    Use this to load metadata and calculate features for live trading.
    """

    @staticmethod
    def load_feature_calculator(
        metadata_file: str
    ) -> FeatureCalculator:
        """
        Load feature calculator from saved metadata file.

        Args:
            metadata_file: Path to the metadata JSON file

        Returns:
            FeatureCalculator instance ready to use
        """
        return FeatureCalculator.from_metadata_file(metadata_file)

    @staticmethod
    def get_latest_metadata_file(
        artifacts_dir: str = 'artifacts',
        symbol: str = None,
        size: int = 60
    ) -> Optional[str]:
        """
        Get the latest metadata file for a symbol and feature set size.

        Args:
            artifacts_dir: Directory where metadata files are stored
            symbol: Trading symbol (optional filter)
            size: Feature set size (60, 50, or 40)

        Returns:
            Path to latest metadata file, or None if not found
        """
        metadata_dir = Path(artifacts_dir) / 'feature_metadata'

        if not metadata_dir.exists():
            return None

        # Find all metadata files matching criteria
        pattern = f"feature_metadata_{size}"
        if symbol:
            pattern += f"_{symbol}"
        pattern += "_*.json"

        matching_files = list(metadata_dir.glob(pattern))

        if not matching_files:
            return None

        # Return most recent file
        latest_file = max(matching_files, key=lambda p: p.stat().st_mtime)
        return str(latest_file)

    @staticmethod
    def calculate_features_for_live_trading(
        ohlcv_data: pd.DataFrame,
        feature_bank: pd.DataFrame,
        metadata_file: str
    ) -> pd.DataFrame:
        """
        Calculate features for live trading using saved metadata.

        Args:
            ohlcv_data: Current OHLCV data
            feature_bank: Base features from feature bank
            metadata_file: Path to metadata file

        Returns:
            DataFrame with calculated features ready for model prediction
        """
        # Load calculator
        calculator = FeatureCalculator.from_metadata_file(metadata_file)

        # If Polars is available, prefer passing Polars DataFrames into FeatureCalculator
        # so that upstream pipelines can remain Polars-first. FeatureCalculator will
        # internally normalize inputs back to pandas where necessary.
        if pl is not None:
            ohlcv_input: Any
            feature_bank_input: Any

            if isinstance(ohlcv_data, pd.DataFrame):
                ohlcv_input = pl.DataFrame(ohlcv_data)
            else:
                ohlcv_input = ohlcv_data

            if isinstance(feature_bank, pd.DataFrame):
                feature_bank_input = pl.DataFrame(feature_bank)
            else:
                feature_bank_input = feature_bank
        else:
            ohlcv_input = ohlcv_data
            feature_bank_input = feature_bank

        # Calculate features
        calculated_features = calculator.calculate(
            ohlcv_input,
            feature_bank_input,
            return_type='dataframe'
        )

        return calculated_features

    @staticmethod
    def example_live_trading_usage():
        """
        Example showing how to use in live trading system.

        ```python
        from src.interaction_features_constructor.integration_helper import LiveTradingIntegration

        # Get latest metadata file
        metadata_file = LiveTradingIntegration.get_latest_metadata_file(
            symbol='ETHUSDT',
            size=60
        )

        # Calculate features for current candle
        features = LiveTradingIntegration.calculate_features_for_live_trading(
            ohlcv_data=current_ohlcv,
            feature_bank=current_feature_bank,
            metadata_file=metadata_file
        )

        # Use features for prediction
        prediction = model.predict(features)
        ```
        """
        pass


def print_integration_instructions():
    """Print instructions for integrating the feature constructor."""
    instructions = """
==========================================================================
INTEGRATION INSTRUCTIONS FOR FEATURE INTERACTION CONSTRUCTOR
==========================================================================

## 1. TRAINING PIPELINE INTEGRATION

In: src/training/steps/pre_training/feature_generation_final_feature_selection_step.py

Add to the _generate_artifacts() method (around line 1600-1650):

```python
from src.interaction_features_constructor.integration_helper import TrainingPipelineIntegration

# In _generate_artifacts() method, after creating feature_sets dict:

# Save feature metadata for live trading reconstruction
try:
    metadata_paths = TrainingPipelineIntegration.add_to_final_feature_selection_step(
        feature_sets, config
    )
    tprint_info(f"✅ Saved feature metadata: {metadata_paths}")
except Exception as e:
    tprint_warning(f"⚠️ Failed to save feature metadata: {e}")
```

## 2. LIVE TRADING INTEGRATION

In your live trading system (e.g., src/tactician/ml_tactics_manager.py):

```python
from src.interaction_features_constructor.integration_helper import LiveTradingIntegration

# Load feature calculator once at initialization
metadata_file = LiveTradingIntegration.get_latest_metadata_file(
    symbol=self.symbol,
    size=60  # or 50, 40 depending on which model you're using
)

self.feature_calculator = LiveTradingIntegration.load_feature_calculator(metadata_file)

# When making predictions:
def get_features_for_prediction(self, current_ohlcv, feature_bank):
    # Calculate interaction features
    calculated_features = self.feature_calculator.calculate(
        current_ohlcv,
        feature_bank,
        return_type='dataframe'
    )

    return calculated_features
```

## 3. TESTING THE INTEGRATION

Run the test example:

```bash
python src/interaction_features_constructor/example_usage.py
```

## 4. VERIFICATION

After running a training pipeline with the integration:

1. Check that metadata files are created:
   ls artifacts/feature_metadata/

2. Verify metadata contains correct information:
   python -c "from src.interaction_features_constructor import FeatureMetadataStore; \
   store = FeatureMetadataStore.load('artifacts/feature_metadata/latest.json'); \
   print(store)"

3. Test feature calculation:
   python scripts/test_feature_calculation.py

==========================================================================
"""
    print(instructions)


if __name__ == '__main__':
    print_integration_instructions()
