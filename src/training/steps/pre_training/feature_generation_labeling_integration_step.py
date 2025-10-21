"""
Feature Generation Labeling Integration Step

This step integrates labeling for feature generation using the enhanced analyst labeler.
"""

import time
from datetime import datetime
from typing import Any, Dict, Optional
import pandas as pd
import numpy as np

from src.training.steps.base_step import BaseStep


try:  # Logging helpers
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_data_preview
    )
except Exception:  # pragma: no cover
    def tprint(*args, **kwargs): print(*args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_data_preview(*args, **kwargs): pass  # No-op fallback


from dataclasses import dataclass

@dataclass
class LabelingIntegrationResult:
    success: bool
    labeled_data: pd.DataFrame
    targets: pd.Series
    error_message: Optional[str] = None


class FeatureGenerationLabelingIntegrationStep(BaseStep):
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__("feature_generation_labeling_integration_step", config)


    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the labeling integration step using BaseStep pattern."""
        self.logger.info("🔍 Starting labeling integration step")
        
        # Set context for enhanced file naming
        symbol = config.get('symbol', 'ETHUSDT')
        exchange = config.get('exchange', 'binance')
        direction = config.get('direction', 'long')
        model = config.get('model', 'Analyst')
        
        self._set_context(symbol=symbol, exchange=exchange, direction=direction, model=model)
        
        # Get data from config
        data = config.get('data')
        if data is None or not isinstance(data, pd.DataFrame) or data.empty:
            raise ValueError("Input data must be a non‑empty DataFrame")
        
        # Preview input data for troubleshooting
        tprint_data_preview(data, "input_data", level="INFO")

        # Cache hit path using BaseStep artifact methods
        cached_labeled = self._load_dataframe('labeled_dataframe')
        cached_targets = self._load_dataframe('targets')
        if isinstance(cached_labeled, pd.DataFrame) and isinstance(cached_targets, pd.Series):
            # Preview cached data for troubleshooting
            tprint_data_preview(cached_labeled, "cached_labeled_data", level="INFO")
            tprint_data_preview(cached_targets, "cached_targets", level="INFO")
            tprint_success("📦 Using cached labeling artifacts")
            return {
                'success': True,
                'artifacts': ['labeled_dataframe', 'targets'],
                'metrics': {
                    'integrated_labels': int(len(cached_targets)),
                    'integration_metadata': {
                        'positive_rate': float((cached_targets > 0).mean()),
                        'target_std': float(cached_targets.std()),
                        'cache_hit': True
                    }
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
                
                # Preview raw labels from labeler for troubleshooting
                tprint_data_preview(labels_df, "raw_labels_from_labeler", level="DEBUG")

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

                # Preview processed targets for troubleshooting
                tprint_data_preview(targets, f"processed_targets_{target_name}", level="INFO")

                # Align and build labeled DataFrame
                common_idx = data.index.intersection(targets.index)
                labeled = data.loc[common_idx].copy()
                targets = targets.loc[common_idx]
                labeled[target_name] = targets
                
                # Preview final labeled data for troubleshooting
                tprint_data_preview(labeled, "final_labeled_dataframe", level="INFO")
                tprint_data_preview(targets, "final_targets_series", level="INFO")
                tprint_success(f"✅ Labeled {len(targets)} samples (var={targets.var():.6f})")
            except Exception as e:
                # Fallback to simple returns to keep the pipeline moving if labeler unavailable
                tprint_warning(f"⚠️ Multi‑Horizon labeler failed: {e}; falling back to simple returns")
                if 'close' not in data.columns:
                    raise
                targets = data['close'].pct_change().shift(-1).fillna(0.0).astype(float)
                labeled = data.copy()
                labeled['target'] = targets
                
                # Preview fallback data for troubleshooting
                tprint_data_preview(targets, "fallback_targets", level="WARNING")
                tprint_data_preview(labeled, "fallback_labeled_data", level="WARNING")

        # Save artifacts using BaseStep methods
        try:
            tprint_info(f"🔍 [DEBUG] About to save artifacts: labeled={type(labeled)}, targets={type(targets)}")
            
            # Preview data before saving for troubleshooting
            tprint_data_preview(labeled, "pre_save_labeled_dataframe", level="DEBUG")
            tprint_data_preview(targets, "pre_save_targets", level="DEBUG")
            
            self._save_dataframe(labeled, 'labeled_dataframe')
            self._save_dataframe(targets, 'targets')
            self._save_dataframe(data, 'raw_dataframe')
            tprint_success("✅ Saved labeling artifacts")
        except Exception as e:
            tprint_warning(f"⚠️ Failed to save labeling artifacts: {e}")
            import traceback
            tprint_error(f"❌ Traceback: {traceback.format_exc()}")

        return {
            'success': True,
            'artifacts': ['labeled_dataframe', 'targets', 'raw_dataframe'],
            'metrics': {
                'integrated_labels': int(len(targets)),
                'integration_metadata': {
                    'positive_rate': float((targets > 0).mean()),
                    'target_std': float(targets.std()),
                    'cache_hit': False
                }
            }
        }




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

    # Create config for the step
    config = {
        'symbol': symbol,
        'timeframe': timeframe,
        'direction': direction,
        'intensity': intensity,
        'lookback_days': lookback_days,
        'start_date': start_date,
        'end_date': end_date,
        'exchange': exchange,
        'custom_overrides': custom_overrides or {},
        'data': data
    }

    result_dict = await step.execute(config)

    # Load artifacts using BaseStep methods
    labeled_df = step._load_dataframe('labeled_dataframe') or pd.DataFrame()
    targets = step._load_dataframe('targets') or pd.Series(dtype=float)
    
    return LabelingIntegrationResult(
        success=bool(result_dict.get('success', False)),
        labeled_data=labeled_df,
        targets=targets,
        error_message=result_dict.get('error')
    )
