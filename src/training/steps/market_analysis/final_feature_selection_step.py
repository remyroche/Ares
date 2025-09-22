#!/usr/bin/env python3
"""
Final Feature Selection Step

This module provides the integration step for the final feature selection pipeline
that runs at the end of the market analysis pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path
import asyncio

# Import the final feature selection pipeline
from .final_feature_selection_pipeline import (
    MultiStageFeatureSelector, FeatureSelectionConfig, 
    run_final_feature_selection, get_final_features
)

# Import system utilities
from src.utils.logger import get_logger
from src.utils.comprehensive_function_logger import log_all_calls
from src.core.decorators import handles_errors, traced, log_execution_time, validates

class FinalFeatureSelectionStep:
    """Final feature selection step for market analysis pipeline."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = get_logger("FinalFeatureSelectionStep")
        
        # Initialize feature selection configuration
        self.feature_config = FeatureSelectionConfig(
            initial_features=self.config.get('initial_features', 120),
            stage_1_target=self.config.get('stage_1_target', 100),
            stage_2_target=self.config.get('stage_2_target', 80),
            stage_3_target=self.config.get('stage_3_target', 60),
            rf_n_estimators=self.config.get('rf_n_estimators', 100),
            cv_folds=self.config.get('cv_folds', 5),
            save_analysis=self.config.get('save_analysis', True),
            output_directory=self.config.get('output_directory', "outcomes/market_analysis"),
            verbose=self.config.get('verbose', True)
        )
        
        self.logger.info("🚀 FinalFeatureSelectionStep initialized")
    
    @log_all_calls
    @handles_errors(Exception, fallback=False)
    @log_execution_time()
    async def execute_final_feature_selection(self, 
                                            symbol: str, 
                                            exchange: str, 
                                            timeframe: str, 
                                            data_dir: str,
                                            **kwargs) -> bool:
        """
        Execute final feature selection step.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Data timeframe
            data_dir: Data directory path
            **kwargs: Additional parameters
            
        Returns:
            bool: True if successful, False otherwise
        """
        
        self.logger.info("🔍 Starting final feature selection step")
        self.logger.info(f"   📊 Symbol: {symbol}")
        self.logger.info(f"   🏢 Exchange: {exchange}")
        self.logger.info(f"   ⏰ Timeframe: {timeframe}")
        self.logger.info(f"   📁 Data directory: {data_dir}")
        
        try:
            # Load feature data
            feature_data = await self._load_feature_data(symbol, exchange, timeframe, data_dir)
            if feature_data is None:
                self.logger.error("❌ Failed to load feature data")
                return False
            
            # Load target data (if available)
            target_data = await self._load_target_data(symbol, exchange, timeframe, data_dir)
            
            # Prepare data for feature selection
            X, y = self._prepare_data(feature_data, target_data)
            
            # Run feature selection
            selection_result = await self._run_feature_selection(X, y, symbol, exchange, timeframe)
            
            # Save results
            await self._save_selection_results(selection_result, symbol, exchange, timeframe, data_dir)
            
            # Generate summary report
            await self._generate_summary_report(selection_result, symbol, exchange, timeframe)
            
            self.logger.info("✅ Final feature selection completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Final feature selection failed: {e}")
            return False
    
    async def _load_feature_data(self, symbol: str, exchange: str, timeframe: str, data_dir: str) -> Optional[pd.DataFrame]:
        """Load feature data from previous pipeline steps."""
        
        try:
            # Try different possible file locations and formats
            possible_files = [
                f"{symbol.lower()}_{timeframe}_features.parquet",
                f"{symbol.lower()}_{timeframe}_engineered_features.parquet",
                f"{symbol.lower()}_{timeframe}_final_features.parquet",
                f"{symbol.lower()}_{timeframe}_matrix_features.parquet"
            ]
            
            data_path = Path(data_dir)
            
            for filename in possible_files:
                file_path = data_path / filename
                if file_path.exists():
                    self.logger.info(f"📂 Loading feature data from: {file_path}")
                    data = pd.read_parquet(file_path)

                    # 🔧 INTEGRATE DATA CLEANING UTILITY
                    # Clean corrupted data before final feature selection
                    try:
                        from src.utils.ml_common.data_processing.data_cleaning_utils import exclude_corrupted_periods

                        # Ensure datetime column exists
                        if 'timestamp' in data.columns and data['timestamp'].dtype == 'int64':
                            data['datetime'] = pd.to_datetime(data['timestamp'], unit='s')
                        elif 'datetime' not in data.columns:
                            # Try to infer datetime column
                            datetime_cols = [col for col in data.columns if 'time' in col.lower()]
                            if datetime_cols:
                                data['datetime'] = pd.to_datetime(data[datetime_cols[0]])
                            else:
                                data['datetime'] = data.index

                        # Apply data cleaning
                        original_count = len(data)
                        data = exclude_corrupted_periods(data)
                        cleaned_count = len(data)

                        if original_count != cleaned_count:
                            excluded_count = original_count - cleaned_count
                            self.logger.info(f"🧹 Final Feature Selection Data cleaning applied: Excluded {excluded_count:,} corrupted rows ({100*excluded_count/original_count:.4f}%)")

                    except ImportError as e:
                        self.logger.warning(f"⚠️ Data cleaning utility not available for final feature selection: {e}")
                    except Exception as e:
                        self.logger.warning(f"⚠️ Data cleaning failed for final feature selection, proceeding with original data: {e}")

                    self.logger.info(f"✅ Loaded {len(data)} samples with {len(data.columns)} features")
                    return data
            
            # If no specific feature file found, try to load from matrix operations
            matrix_file = data_path / f"{symbol.lower()}_{timeframe}_matrix_operations.parquet"
            if matrix_file.exists():
                self.logger.info(f"📂 Loading matrix operations data from: {matrix_file}")
                data = pd.read_parquet(matrix_file)
                self.logger.info(f"✅ Loaded {len(data)} samples with {len(data.columns)} features from matrix operations")
                return data
            
            self.logger.warning("⚠️ No feature data files found")
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load feature data: {e}")
            return None
    
    async def _load_target_data(self, symbol: str, exchange: str, timeframe: str, data_dir: str) -> Optional[pd.Series]:
        """Load target data if available."""
        
        try:
            # Try to load target data from labeling step
            possible_target_files = [
                f"{symbol.lower()}_{timeframe}_labels.parquet",
                f"{symbol.lower()}_{timeframe}_triple_barrier_labels.parquet",
                f"{symbol.lower()}_{timeframe}_target.parquet"
            ]
            
            data_path = Path(data_dir)
            
            for filename in possible_target_files:
                file_path = data_path / filename
                if file_path.exists():
                    self.logger.info(f"📂 Loading target data from: {file_path}")
                    data = pd.read_parquet(file_path)
                    
                    # Try to find target column
                    target_columns = ['target', 'label', 'y', 'return', 'triple_barrier_label']
                    target_col = None
                    
                    for col in target_columns:
                        if col in data.columns:
                            target_col = col
                            break
                    
                    if target_col:
                        target_data = data[target_col]
                        self.logger.info(f"✅ Loaded target data: {target_col} with {len(target_data)} samples")
                        return target_data
            
            self.logger.info("ℹ️ No target data found - will perform unsupervised feature selection")
            return None
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to load target data: {e}")
            return None
    
    def _prepare_data(self, feature_data: pd.DataFrame, target_data: Optional[pd.Series]) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Prepare data for feature selection."""
        
        # Clean feature data
        X = feature_data.copy()
        
        # Remove non-numeric columns
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_columns]
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Remove infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        self.logger.info(f"📊 Prepared {len(X)} samples with {len(X.columns)} numeric features")
        
        # Prepare target data if available
        y = None
        if target_data is not None:
            # Align target data with feature data
            common_indices = X.index.intersection(target_data.index)
            if len(common_indices) > 0:
                X = X.loc[common_indices]
                y = target_data.loc[common_indices]
                self.logger.info(f"✅ Aligned target data: {len(y)} samples")
            else:
                self.logger.warning("⚠️ No common indices between features and target")
        
        return X, y
    
    async def _run_feature_selection(self, X: pd.DataFrame, y: Optional[pd.Series],
                                   symbol: str, exchange: str, timeframe: str) -> Any:
        """Run the multi-stage feature selection."""
        
        self.logger.info("🔍 Running multi-stage feature selection")
        self.logger.info(f"   📊 Input: {len(X)} samples, {len(X.columns)} features")
        
        if y is not None:
            self.logger.info(f"   🎯 Target: {len(y)} samples (supervised)")
        else:
            self.logger.info("   🎯 No target data (unsupervised)")
        
        # Run feature selection in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        selection_result = await loop.run_in_executor(
            None, 
            self._run_selection_sync, 
            X, y
        )
        
        self.logger.info("✅ Feature selection completed")
        return selection_result
    
    def _run_selection_sync(self, X: pd.DataFrame, y: Optional[pd.Series]) -> Any:
        """Synchronous feature selection (to be run in thread pool)."""
        
        # Create feature selector
        selector = MultiStageFeatureSelector(self.feature_config)
        
        # Run selection
        if y is not None:
            result = selector.select_features(X, y)
        else:
            # For unsupervised selection, create a dummy target
            # This is a simplified approach - in practice, you might want to use
            # different unsupervised feature selection methods
            dummy_target = X.iloc[:, 0]  # Use first feature as proxy target
            result = selector.select_features(X, dummy_target)
            result.is_unsupervised = True
        
        return result
    
    async def _save_selection_results(self, selection_result: Any, symbol: str, exchange: str,
                                    timeframe: str, data_dir: str) -> None:
        """Save feature selection results."""
        
        try:
            output_dir = Path("generated/market_analysis") / "final_feature_selection"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save final selected features
            final_features_file = output_dir / f"{symbol.lower()}_{timeframe}_final_features.json"
            final_features = selection_result.final_features
            
            import json
            with open(final_features_file, 'w') as f:
                json.dump({
                    'symbol': symbol,
                    'exchange': exchange,
                    'timeframe': timeframe,
                    'final_features': final_features,
                    'feature_count': len(final_features),
                    'selection_method': 'multi_stage_rf_shap',
                    'stages': {
                        'stage_1': len(selection_result.stage_1_features),
                        'stage_2': len(selection_result.stage_2_features),
                        'stage_3': len(selection_result.stage_3_features),
                        'final': len(selection_result.final_features)
                    }
                }, f, indent=2)
            
            self.logger.info(f"💾 Final features saved to: {final_features_file}")
            
            # Save detailed results
            detailed_results_file = output_dir / f"{symbol.lower()}_{timeframe}_selection_results.json"
            
            results_dict = {
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'feature_counts': selection_result.feature_counts,
                'scores': {
                    'stage_1': selection_result.stage_1_scores,
                    'stage_2': selection_result.stage_2_scores,
                    'stage_3': selection_result.stage_3_scores,
                    'final': selection_result.final_scores
                },
                'selection_time': selection_result.selection_time,
                'is_unsupervised': getattr(selection_result, 'is_unsupervised', False)
            }
            
            with open(detailed_results_file, 'w') as f:
                json.dump(results_dict, f, indent=2, default=str)
            
            self.logger.info(f"💾 Detailed results saved to: {detailed_results_file}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save selection results: {e}")
    
    async def _generate_summary_report(self, selection_result: Any, symbol: str,
                                     exchange: str, timeframe: str) -> None:
        """Generate summary report of feature selection."""
        
        try:
            self.logger.info("📊 FEATURE SELECTION SUMMARY REPORT")
            self.logger.info("=" * 60)
            self.logger.info(f"🎯 Symbol: {symbol}")
            self.logger.info(f"🏢 Exchange: {exchange}")
            self.logger.info(f"⏰ Timeframe: {timeframe}")
            self.logger.info(f"⏱️ Selection Time: {selection_result.selection_time:.3f}s")
            self.logger.info("")
            
            self.logger.info("📈 FEATURE REDUCTION PIPELINE:")
            self.logger.info(f"   🔢 Initial Features: {selection_result.feature_counts.get('initial', 'N/A')}")
            self.logger.info(f"   📊 Stage 1 (120→100): {selection_result.feature_counts.get('stage_1', 'N/A')} features")
            self.logger.info(f"   📊 Stage 2 (100→80): {selection_result.feature_counts.get('stage_2', 'N/A')} features")
            self.logger.info(f"   📊 Stage 3 (80→60): {selection_result.feature_counts.get('stage_3', 'N/A')} features")
            self.logger.info(f"   ✅ Final Features: {selection_result.feature_counts.get('final', 'N/A')} features")
            self.logger.info("")
            
            self.logger.info("📊 STAGE SCORES:")
            if selection_result.stage_1_scores:
                self.logger.info(f"   🎯 Stage 1 Score: {selection_result.stage_1_scores.get('rf_importance_score', 'N/A'):.4f}")
            if selection_result.stage_2_scores:
                self.logger.info(f"   🎯 Stage 2 Score: {selection_result.stage_2_scores.get('enhanced_rf_score', selection_result.stage_2_scores.get('shap_importance_score', 'N/A')):.4f}")
            if selection_result.stage_3_scores:
                self.logger.info(f"   🎯 Stage 3 Score: {selection_result.stage_3_scores.get('combined_importance_score', 'N/A'):.4f}")
            if selection_result.final_scores:
                self.logger.info(f"   🎯 Final CV Score: {selection_result.final_scores.get('cv_mean', 'N/A'):.4f}")
            self.logger.info("")
            
            # Show top 10 final features
            if hasattr(selection_result, 'model_performance') and 'feature_importance' in selection_result.model_performance:
                top_features = sorted(
                    selection_result.model_performance['feature_importance'].items(),
                    key=lambda x: x[1], reverse=True
                )[:10]
                
                self.logger.info("🏆 TOP 10 FINAL FEATURES:")
                for i, (feature, importance) in enumerate(top_features, 1):
                    self.logger.info(f"   {i:2d}. {feature}: {importance:.4f}")
            
            self.logger.info("=" * 60)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate summary report: {e}")

# Convenience function for pipeline integration
async def run_final_feature_selection_step(symbol: str, 
                                         exchange: str, 
                                         timeframe: str = '1m', 
                                         data_dir: str = 'historical_data',
                                         config: Optional[Dict[str, Any]] = None) -> bool:
    """Run the final feature selection step."""
    
    step = FinalFeatureSelectionStep(config)
    return await step.execute_final_feature_selection(symbol, exchange, timeframe, data_dir)