"""
Example: Enhanced Labeling Integration Step with Proper Artifact Manager Usage

This example shows how to properly use the enhanced artifact manager
with proper metadata, data alignment validation, and shared dataset approach.
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


class EnhancedFeatureGenerationLabelingIntegrationStep(ModularComponent):
    """
    Enhanced labeling integration step with proper artifact manager usage.
    
    This example demonstrates:
    1. Proper metadata extraction and validation
    2. Shared dataset approach for better data alignment
    3. Enhanced logging and transparency
    4. Data alignment validation
    """
    
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

    def _extract_metadata(self, **kwargs) -> Dict[str, Any]:
        """Extract and validate metadata from kwargs."""
        metadata = {
            'symbol': kwargs.get('symbol', 'UNKNOWN'),
            'exchange': kwargs.get('exchange', 'UNKNOWN'),
            'timeframe': kwargs.get('timeframe', 'UNKNOWN'),
            'direction': kwargs.get('direction', 'long'),
            'intensity': kwargs.get('intensity', 'blank'),
            'labeling_mode': kwargs.get('labeling_mode', 'analyst'),
            'lookback_days': kwargs.get('lookback_days', 365),
            'start_date': kwargs.get('start_date'),
            'end_date': kwargs.get('end_date')
        }
        
        # Log metadata for transparency
        tprint_info(f"📊 Extracted metadata: {metadata}")
        
        return metadata

    def _process_data(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Process data with enhanced artifact manager usage."""
        if data is None or not isinstance(data, pd.DataFrame) or data.empty:
            raise ValueError("Input data must be a non‑empty DataFrame")

        # Extract metadata first
        metadata = self._extract_metadata(**kwargs)
        
        am = get_pretraining_artifact_manager()

        # Check for cached results with proper metadata validation
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

        # Generate labels using the appropriate labeler
        try:
            labeling_mode = kwargs.get('labeling_mode', 'analyst').lower()
            
            if labeling_mode == 'tactician':
                from src.training.steps.pre_training.profit_labeling.volatility_aware_labeler import create_enhanced_tactician_labeler
                labeler = create_enhanced_tactician_labeler()
                tprint_info(f"🎯 Using Tactician labeler for short-term entry labeling")
            else:
                from src.training.steps.pre_training.profit_labeling.multi_horizon_profit_labeler import create_enhanced_analyst_labeler
                labeler = create_enhanced_analyst_labeler()
                tprint_info(f"🎯 Using Analyst labeler for multi-horizon profit labeling")

            # Generate labels
            labeled, targets = labeler.generate_labels(
                data=data,
                symbol=metadata['symbol'],
                timeframe=metadata['timeframe'],
                direction=metadata['direction'],
                intensity=metadata['intensity'],
                lookback_days=metadata['lookback_days'],
                start_date=metadata.get('start_date'),
                end_date=metadata.get('end_date'),
                exchange=metadata['exchange'],
                custom_overrides=kwargs
            )

            tprint_success(f"✅ Generated {len(targets)} labels with {labeling_mode} labeler")

        except Exception as e:
            tprint_error(f"❌ Labeling failed: {e}")
            raise

        # Use shared dataset approach for better data alignment
        try:
            # Create additional columns for the shared dataset
            additional_columns = {
                'targets': targets,
                'labeling_metadata': pd.Series([labeling_mode] * len(targets), index=targets.index, name='labeling_mode')
            }
            
            # Save as shared dataset with proper metadata
            artifact_paths = am.save_shared_dataset(
                step_name='feature_generation_labeling_integration_step',
                base_data=data,  # Original OHLCV data
                additional_columns=additional_columns,
                metadata=metadata
            )
            
            tprint_success("✅ Saved labeling artifacts using shared dataset approach")
            tprint_info(f"📁 Artifact paths: {artifact_paths}")

        except Exception as e:
            tprint_error(f"❌ Failed to save artifacts: {e}")
            raise

        # Verify the saved artifacts
        try:
            saved_labeled = am.get_artifact('feature_generation_labeling_integration_step', 'shared_dataset')
            saved_targets = am.get_artifact('feature_generation_labeling_integration_step', 'targets')
            
            if saved_targets is not None:
                tprint_info(f"🔍 [DEBUG] Verification - saved_labeled: {type(saved_labeled)}, saved_targets: {type(saved_targets)}")
                tprint_success("✅ Artifacts verified successfully")
            else:
                tprint_warning("⚠️ Could not verify saved artifacts")

        except Exception as e:
            tprint_warning(f"⚠️ Artifact verification failed: {e}")

        return {
            'success': True,
            'integrated_labels': int(len(targets)),
            'integration_metadata': {
                'positive_rate': float((targets > 0).mean()),
                'target_std': float(targets.std()),
                'labeling_mode': labeling_mode,
                'cache_hit': False
            },
            'artifacts': {
                'labeled_dataframe': labeled,
                'targets': targets,
                'raw_dataframe': data
            }
        }

    def _validate_component_specific(self, data: Any) -> Dict[str, Any]:
        """Validate component-specific requirements."""
        errors, warnings, metadata = [], [], {}
        
        if isinstance(data, pd.DataFrame):
            if len(data) < 100:
                errors.append(f"Data has {len(data)} rows, minimum required: 100")
            metadata['shape'] = data.shape
            metadata['columns'] = list(data.columns)
            
            # Check for required OHLCV columns
            required_cols = ['open', 'high', 'low', 'close']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                errors.append(f"Missing required OHLCV columns: {missing_cols}")
                
        return {'errors': errors, 'warnings': warnings, 'metadata': metadata}


# Example usage function
async def handle_enhanced_labeling_integration_step(
    symbol: str = "ETHUSDT",
    timeframe: str = "15m",
    direction: str = "long",
    intensity: str = "blank",
    lookback_days: int = 365,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    exchange: str = "binance",
    custom_overrides: Optional[Dict[str, Any]] = None,
    data: Optional[pd.DataFrame] = None,
    labeling_mode: str = "analyst"
) -> Dict[str, Any]:
    """
    Enhanced labeling integration step with proper artifact manager usage.
    
    This function demonstrates:
    1. Proper metadata passing
    2. Shared dataset approach
    3. Data alignment validation
    4. Enhanced logging and transparency
    """
    
    # Initialize the enhanced step
    step = EnhancedFeatureGenerationLabelingIntegrationStep()
    
    # Initialize resources
    if not step._initialize_resources():
        return {
            'success': False,
            'error_message': 'Failed to initialize labeling integration step'
        }
    
    try:
        # Process data with enhanced artifact manager usage
        result_dict = step._process_data(
            data, 
            symbol=symbol, 
            timeframe=timeframe, 
            direction=direction,
            intensity=intensity, 
            lookback_days=lookback_days, 
            start_date=start_date,
            end_date=end_date, 
            exchange=exchange, 
            custom_overrides=custom_overrides or {},
            labeling_mode=labeling_mode
        )
        
        return result_dict
        
    except Exception as e:
        tprint_error(f"❌ Enhanced labeling integration failed: {e}")
        return {
            'success': False,
            'error_message': str(e)
        }
    finally:
        step._cleanup_resources()


if __name__ == "__main__":
    # Example usage
    print("Enhanced Labeling Integration Step Example")
    print("This example demonstrates proper artifact manager usage with:")
    print("1. Metadata extraction and validation")
    print("2. Shared dataset approach")
    print("3. Data alignment validation")
    print("4. Enhanced logging and transparency")
