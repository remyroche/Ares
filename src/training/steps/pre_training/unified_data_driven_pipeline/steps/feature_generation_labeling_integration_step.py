"""
Feature Generation Labeling Integration Step

This step integrates labeling for feature generation using the enhanced analyst labeler.
"""

import time
from datetime import datetime
from typing import Any, Dict, Optional
import pandas as pd
import numpy as np

from src.training.steps.models_training.unified_data_driven_pipeline.core.modular_architecture import ModularComponent
from src.training.steps.pre_training.utils.artifact_manager import get_pretraining_artifact_manager

try:  # Logging helpers
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error
    )
except Exception:  # pragma: no cover
    def tprint(*args, **kwargs): print(*args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)


from dataclasses import dataclass

@dataclass
class LabelingIntegrationResult:
    success: bool
    labeled_data: pd.DataFrame
    targets: pd.Series
    error_message: Optional[str] = None


class FeatureGenerationLabelingIntegrationStep(ModularComponent):
    def __init__(self, name: str = "labeling_integration_step",
                 config: Optional[Dict[str, Any]] = None,
                 logger: Optional[Any] = None) -> None:
        super().__init__(name, config, logger)

    def _initialize_resources(self) -> bool:
        try:
            self.set_state('initialized_at', time.time())
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize labeling integration: {e}")
            return False

    def _cleanup_resources(self) -> None:
        try:
            self.set_state('cleaned_up_at', time.time())
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    def _process_data(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Compute targets using the enhanced analyst labeler and save artifacts."""
        if data is None or not isinstance(data, pd.DataFrame) or data.empty:
            raise ValueError("Input data must be a non‑empty DataFrame")

        am = get_pretraining_artifact_manager()

        # Cache hit path
        cached_labeled = am.get_artifact('feature_generation_labeling_integration_step', 'labeled_dataframe')
        cached_targets = am.get_artifact('feature_generation_labeling_integration_step', 'targets')
        if isinstance(cached_labeled, pd.DataFrame) and isinstance(cached_targets, pd.Series):
            tprint_success("📦 Using cached labeling artifacts")
            return {
                'success': True,
                'integrated_labels': int(len(cached_targets)),
                'integration_metadata': {
                    'positive_rate': float((cached_targets > 0).mean()),
                    'target_std': float(cached_targets.std()),
                    'cache_hit': True
                },
                'artifacts': {
                    'labeled_dataframe': cached_labeled,
                    'targets': cached_targets,
                    'raw_dataframe': data
                }
            }

        # Validate required columns
        required_cols = ['open', 'high', 'low', 'close']
        missing = [c for c in required_cols if c not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns for labeling: {missing}")

        # Run multi‑horizon labeler
        try:
            from src.training.steps.pre_training.profit_labeling.volatility_aware_labeler import create_enhanced_analyst_labeler
            labeler = create_enhanced_analyst_labeler()
            lr = labeler.generate_labels(data)
            labels_df = getattr(lr, 'labels', pd.DataFrame())
            if labels_df is None or labels_df.empty:
                raise ValueError('Labeling produced no label columns')

            # Handle both Series and DataFrame cases
            if isinstance(labels_df, pd.Series):
                # Single target case - use the series directly
                targets = labels_df.dropna().astype(float)
                target_name = labels_df.name or 'target'
            else:
                # Multiple targets case - prefer columns that contain 'target'
                target_cols = [c for c in labels_df.columns if 'target' in str(c).lower()]
                target_col = target_cols[0] if target_cols else labels_df.select_dtypes(include=[np.number]).columns[0]
                targets = labels_df[target_col].dropna().astype(float)
                target_name = target_col

            # Align and build labeled DataFrame
            common_idx = data.index.intersection(targets.index)
            labeled = data.loc[common_idx].copy()
            targets = targets.loc[common_idx]
            labeled[target_name] = targets
            tprint_success(f"✅ Labeled {len(targets)} samples (var={targets.var():.6f})")
        except Exception as e:
            # Fallback to simple returns to keep the pipeline moving if labeler unavailable
            tprint_warning(f"⚠️ Multi‑Horizon labeler failed: {e}; falling back to simple returns")
            if 'close' not in data.columns:
                raise
            targets = data['close'].pct_change().shift(-1).fillna(0.0).astype(float)
            labeled = data.copy()
            labeled['target'] = targets

        # Persist artifacts
        try:
            tprint_info(f"🔍 [DEBUG] About to save artifacts: labeled={type(labeled)}, targets={type(targets)}")
            am.save(
                step_name='feature_generation_labeling_integration_step',
                artifacts={
                    'labeled_dataframe': labeled,
                    'targets': targets,
                    'raw_dataframe': data
                },
                metadata={
                    'step': 'feature_generation_labeling_integration_step',
                    'shape': labeled.shape,
                    'created_at': datetime.now().isoformat()
                }
            )
            tprint_success("✅ Saved labeling artifacts")
            
            # Verify artifacts were saved
            saved_labeled = am.get_artifact('feature_generation_labeling_integration_step', 'labeled_dataframe')
            saved_targets = am.get_artifact('feature_generation_labeling_integration_step', 'targets')
            tprint_info(f"🔍 [DEBUG] Verification - saved_labeled: {type(saved_labeled)}, saved_targets: {type(saved_targets)}")
        except Exception as e:
            tprint_warning(f"⚠️ Failed to save labeling artifacts: {e}")
            import traceback
            tprint_error(f"❌ Traceback: {traceback.format_exc()}")

        return {
            'success': True,
            'integrated_labels': int(len(targets)),
            'integration_metadata': {
                'positive_rate': float((targets > 0).mean()),
                'target_std': float(targets.std()),
                'cache_hit': False
            },
            'artifacts': {
                'labeled_dataframe': labeled,
                'targets': targets,
                'raw_dataframe': data
            }
        }

    async def execute(
        self,
        training_input: Optional[Dict[str, Any]] = None,
        pipeline_state: Optional[Dict[str, Any]] = None,
        data: Optional[pd.DataFrame] = None,
        **kwargs: Any
    ) -> LabelingIntegrationResult:
        """Async execute entry to integrate with ares_launcher sequential mode."""
        # Accept data from training_input or direct arg
        if data is None and isinstance(training_input, dict):
            data = training_input.get('data')
        if not isinstance(data, pd.DataFrame) or data.empty:
            raise ValueError("Labeling integration requires a non-empty DataFrame as 'data'.")

        result_dict = self._process_data(data, **(training_input or {}))
        artifacts = result_dict.get('artifacts', {})
        labeled_df = artifacts.get('labeled_dataframe', pd.DataFrame())
        targets = artifacts.get('targets', pd.Series(dtype=float))
        return LabelingIntegrationResult(
            success=bool(result_dict.get('success', False)),
            labeled_data=labeled_df,
            targets=targets,
            error_message=result_dict.get('error_message')
        )

    def _get_validation_rules(self) -> Dict[str, Any]:
        return {
            'data_types': ['pandas.DataFrame'],
            'required_attributes': ['open', 'high', 'low', 'close'],
            'min_rows': 100
        }

    def _validate_component_specific(self, data: Any) -> Dict[str, Any]:
        errors, warnings, metadata = [], [], {}
        if isinstance(data, pd.DataFrame):
            if len(data) < 100:
                errors.append(f"Data has {len(data)} rows, minimum required: 100")
            metadata['shape'] = data.shape
            metadata['columns'] = list(data.columns)
        return {'errors': errors, 'warnings': warnings, 'metadata': metadata}

    async def execute(self, data: pd.DataFrame, **kwargs) -> LabelingIntegrationResult:
        """Execute the labeling integration step."""
        result = self._process_data(data, **kwargs)
        return LabelingIntegrationResult(
            success=result.get('success', False),
            labeled_data=result.get('artifacts', {}).get('labeled_dataframe', pd.DataFrame()),
            targets=result.get('artifacts', {}).get('targets', pd.Series()),
            error_message=None
        )


# Handler for ares_launcher/sub_pipeline integration
async def handle_feature_generation_labeling_integration_step(
    symbol: str = "ETHUSDT",
    timeframe: str = "15m",
    direction: str = "longs",
    intensity: str = "blank",
    lookback_days: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    exchange: str = "binance",
    custom_overrides: Optional[Dict[str, Any]] = None,
    data: Optional[pd.DataFrame] = None,
    **kwargs: Any
) -> LabelingIntegrationResult:
    """Execute labeling integration and persist artifacts (launcher compatibility)."""
    step = FeatureGenerationLabelingIntegrationStep()

    # Attempt lazy load if data not provided
    if data is None or not isinstance(data, pd.DataFrame) or data.empty:
        try:
            from .feature_generation_data_validation_step import FeatureGenerationDataValidationStep  # type: ignore
            loader = FeatureGenerationDataValidationStep()
            loaded = await loader._load_data_for_validation(  # noqa: SLF001
                symbol, timeframe, exchange, start_date, end_date, lookback_days
            )
            data = loaded
        except Exception as e:
            tprint_error(f"❌ Failed to auto-load data for labeling integration: {e}")
            raise

    result_dict = step._process_data(data, symbol=symbol, timeframe=timeframe, direction=direction,
                                     intensity=intensity, lookback_days=lookback_days, start_date=start_date,
                                     end_date=end_date, exchange=exchange, custom_overrides=custom_overrides or {})

    artifacts = result_dict.get('artifacts', {})
    labeled_df = artifacts.get('labeled_dataframe', pd.DataFrame())
    targets = artifacts.get('targets', pd.Series(dtype=float))
    return LabelingIntegrationResult(
        success=bool(result_dict.get('success', False)),
        labeled_data=labeled_df,
        targets=targets,
        error_message=result_dict.get('error_message')
    )
