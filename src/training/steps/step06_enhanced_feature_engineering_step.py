"""
Enhanced Step06 Feature Engineering Step with Modular Approach

This module implements a modular, memory-efficient feature engineering step with:
- Reduced nested functions using modular approach
- Integration of enhanced feature engineering components
- Strict temporal validation and lookahead bias prevention
- Memory-efficient chunking for large datasets
- Comprehensive error handling and validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from pathlib import Path
import json
import time
from contextlib import nullcontext

# Import base step and utilities
from src.training.base_step import BaseStep
from src.utils.comprehensive_function_logger import (
    log_step_functions, log_important_calls, log_all_calls, 
    log_internal_call, log_step_progress, log_data_operation
)
from src.utils.math_validation import (
    safe_divide, validate_positive, validate_range, MathValidationError
)

# Import enhanced components
from .step06_enhanced_feature_engineering import EnhancedFeatureEngineering

# Import validation framework
try:
    from .step06_enhanced_validation_framework import (
        step06_function_validator, step06_function_tracker, 
        step06_validation_context, ValidationLevel, FunctionStatus
    )
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False
    
    def step06_function_validator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def step06_function_tracker(func):
        return func
    
    def step06_validation_context(*args, **kwargs):
        return nullcontext()
    
    class ValidationLevel:
        BASIC = 'basic'
        DETAILED = 'detailed'
        COMPREHENSIVE = 'comprehensive'

logger = logging.getLogger(__name__)

class EnhancedFeatureEngineeringStep(BaseStep):
    """
    Enhanced Step 6: Feature Engineering with modular approach and advanced optimizations.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize enhanced feature engineering step."""
        super().__init__(config, '06', 'enhanced_feature_engineering')
        
        # Initialize enhanced feature engineering
        self.enhanced_engine = EnhancedFeatureEngineering(config)
        
        # Configuration
        self.feature_config = config.get('step06_feature_engineering', {
            'use_technical_indicators': True,
            'use_interaction_features': True,
            'use_regime_features': True,
            'use_sr_features': True,
            'use_dynamic_lookback': True,
            'chunk_size': 10000,
            'max_features': 500,
            'polynomial_degree': 2,
            'correlation_threshold': 0.95,
            'memory_limit_mb': 1000,
            'lookback_periods': {
                'RSI': [7, 14, 21],
                'MACD': [12, 26, 52],
                'Bollinger_Bands': [10, 20, 50],
                'SMA': [5, 20, 100],
                'EMA': [8, 21, 55],
                'ATR': [7, 14, 30],
                'Stochastic': [7, 14, 30],
                'ADX': [7, 14, 25],
                'OBV': [10, 20, 50],
                'MFI': [7, 14, 30]
            }
        })
        
        # Performance tracking
        self.performance_metrics = {
            'total_processing_time': 0.0,
            'total_memory_used_mb': 0.0,
            'features_created': 0,
            'chunks_processed': 0,
            'validation_errors': 0
        }
        
        self.logger.info("🚀 Enhanced Feature Engineering Step initialized")
        self.logger.info(f"   Chunk size: {self.feature_config['chunk_size']}")
        self.logger.info(f"   Max features: {self.feature_config['max_features']}")
        self.logger.info(f"   Polynomial degree: {self.feature_config['polynomial_degree']}")

    @log_step_functions
    def _initialize_step(self) -> None:
        """Initialize step-specific components."""
        try:
            # Initialize enhanced feature engineering components
            self.enhanced_engine = EnhancedFeatureEngineering(self.config)
            self.logger.info('✅ Enhanced feature engineering components initialized')
        except Exception as e:
            self.logger.error(f'❌ Failed to initialize enhanced components: {e}')
            raise

    @step06_function_validator(function_type='feature_engineering', validation_level=ValidationLevel.COMPREHENSIVE)
    def validate_inputs(self, training_input: Dict[str, Any], 
                       pipeline_state: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate step inputs with enhanced validation.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            Tuple of (is_valid, errors)
        """
        with step06_validation_context('validate_inputs', 'feature_engineering'):
            self.logger.info(f'🔍 Starting enhanced input validation')
            self.logger.info(f'   Training input keys: {list(training_input.keys())}')
            self.logger.info(f'   Pipeline state keys: {list(pipeline_state.keys())}')
        
        errors = []
        
        # Check for labeled data
        if 'labeled_data' not in pipeline_state:
            if not any(f'{split}_data' in pipeline_state for split in ['train', 'val', 'test']):
                errors.append('No labeled data from step 5')
        
        # Validate data quality
        data_to_validate = None
        for key in ['labeled_data', 'train_data', 'val_data', 'test_data']:
            if key in pipeline_state:
                data_to_validate = pipeline_state[key]
                break
        
        if data_to_validate is not None:
            validation_result = self._validate_data_quality(data_to_validate)
            if not validation_result['is_valid']:
                errors.extend(validation_result['errors'])
        
        # Validate configuration
        config_errors = self._validate_configuration()
        errors.extend(config_errors)
        
        is_valid = len(errors) == 0
        if not is_valid:
            self.performance_metrics['validation_errors'] += len(errors)
            self.logger.error(f'❌ Input validation failed: {errors}')
        else:
            self.logger.info('✅ Input validation passed')
        
        return is_valid, errors

    def _validate_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality for feature engineering."""
        errors = []
        
        try:
            # Check data shape
            if len(data) < 50:
                errors.append(f'Insufficient data: {len(data)} rows (minimum 50 required)')
            
            # Check required columns
            required_columns = ['open', 'high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                errors.append(f'Missing required columns: {missing_columns}')
            
            # Check for valid prices
            for col in required_columns:
                if col in data.columns:
                    if (data[col] <= 0).any():
                        errors.append(f'Invalid prices in {col}: non-positive values found')
                    if data[col].isna().any():
                        errors.append(f'NaN values in {col}')
            
            # Check temporal consistency
            if isinstance(data.index, pd.DatetimeIndex):
                if not data.index.is_monotonic_increasing:
                    errors.append('Data index is not temporally ordered')
            
        except Exception as e:
            errors.append(f'Data validation error: {e}')
        
        return {'is_valid': len(errors) == 0, 'errors': errors}

    def _validate_configuration(self) -> List[str]:
        """Validate configuration parameters."""
        errors = []
        
        try:
            # Validate chunk size
            chunk_size = self.feature_config.get('chunk_size', 10000)
            if chunk_size < 100:
                errors.append(f'Chunk size too small: {chunk_size} (minimum 100)')
            if chunk_size > 100000:
                errors.append(f'Chunk size too large: {chunk_size} (maximum 100000)')
            
            # Validate max features
            max_features = self.feature_config.get('max_features', 500)
            if max_features < 10:
                errors.append(f'Max features too small: {max_features} (minimum 10)')
            if max_features > 10000:
                errors.append(f'Max features too large: {max_features} (maximum 10000)')
            
            # Validate polynomial degree
            poly_degree = self.feature_config.get('polynomial_degree', 2)
            if poly_degree < 1 or poly_degree > 3:
                errors.append(f'Polynomial degree invalid: {poly_degree} (must be 1-3)')
            
        except Exception as e:
            errors.append(f'Configuration validation error: {e}')
        
        return errors

    @step06_function_validator(function_type='feature_engineering', validation_level=ValidationLevel.COMPREHENSIVE)
    async def execute_logic(self, training_input: Dict[str, Any], 
                           pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute enhanced feature engineering logic with modular approach.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            Updated pipeline state
        """
        start_time = time.time()
        
        with step06_validation_context('execute_logic', 'feature_engineering'):
            self.logger.info(f'🔧 Starting enhanced feature engineering execution')
            self.logger.info(f'   Training input keys: {list(training_input.keys())}')
            self.logger.info(f'   Pipeline state keys: {list(pipeline_state.keys())}')
        
        try:
            # Step 1: Get and validate data
            data_dict = self._get_data_to_process(pipeline_state)
            self.logger.info(f'📊 Processing {len(data_dict)} data splits')
            
            # Step 2: Process each data split
            engineered_data = {}
            feature_statistics = {}
            
            for split_name, data in data_dict.items():
                self.logger.info(f'🔧 Processing {split_name} split: {data.shape}')
                
                # Process with enhanced feature engineering
                engineered_split = await self._process_data_split(data, split_name)
                
                # Calculate statistics
                stats = self._calculate_feature_statistics(engineered_split)
                
                engineered_data[split_name] = engineered_split
                feature_statistics[split_name] = stats
                
                self.logger.info(f'✅ {split_name} split processed: {engineered_split.shape}')
            
            # Step 3: Feature selection and optimization
            if self.feature_config.get('feature_selection', {}).get('enabled', True):
                self.logger.info('🎯 Performing feature selection...')
                engineered_data, selected_features = await self._perform_feature_selection(
                    engineered_data, feature_statistics
                )
                self.logger.info(f'✅ Feature selection complete: {len(selected_features)} features selected')
            else:
                selected_features = self._get_all_feature_columns(engineered_data)
            
            # Step 4: Generate reports
            reports = self._generate_feature_reports(engineered_data, feature_statistics, selected_features)
            
            # Step 5: Update pipeline state
            pipeline_state.update({
                'engineered_data': engineered_data,
                'feature_statistics': feature_statistics,
                'selected_features': selected_features,
                'feature_reports': reports,
                'feature_config': self.feature_config,
                'performance_metrics': self.performance_metrics.copy()
            })
            
            # Update individual splits
            for split in ['train', 'val', 'test']:
                if split in engineered_data and f'{split}_data' in pipeline_state:
                    pipeline_state[f'{split}_data'] = engineered_data[split]
            
            # Step 6: Save outputs
            await self._save_outputs(training_input, pipeline_state)
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self.performance_metrics['total_processing_time'] += processing_time
            self.performance_metrics['features_created'] = len(selected_features)
            
            self.logger.info(f'✅ Enhanced feature engineering completed in {processing_time:.2f}s')
            self.logger.info(f'   Features created: {len(selected_features)}')
            self.logger.info(f'   Total processing time: {self.performance_metrics["total_processing_time"]:.2f}s')
            
            return pipeline_state
            
        except Exception as e:
            self.logger.error(f'❌ Enhanced feature engineering failed: {e}')
            self.performance_metrics['validation_errors'] += 1
            raise

    async def _process_data_split(self, data: pd.DataFrame, split_name: str) -> pd.DataFrame:
        """
        Process a single data split with enhanced feature engineering.
        
        Args:
            data: Data to process
            split_name: Name of the data split
            
        Returns:
            Processed data with enhanced features
        """
        try:
            # Start with base data
            processed_data = data.copy()
            
            # Step 1: Extract technical indicators using batch processing
            if self.feature_config.get('use_technical_indicators', True):
                self.logger.info(f'📊 Extracting technical indicators for {split_name}')
                
                lookback_periods = self.feature_config.get('lookback_periods', {})
                indicators = self.enhanced_engine.extract_indicators_batch(
                    processed_data, lookback_periods
                )
                
                # Combine with original data
                processed_data = pd.concat([processed_data, indicators], axis=1)
                self.logger.info(f'   Added {indicators.shape[1]} technical indicators')
            
            # Step 2: Create sophisticated interactions
            if self.feature_config.get('use_interaction_features', True):
                self.logger.info(f'🔗 Creating sophisticated interactions for {split_name}')
                
                # Apply temporal validation to prevent lookahead bias
                current_idx = len(processed_data) - 1  # Use all data up to current point
                interactions = self.enhanced_engine.create_sophisticated_interactions(
                    processed_data, current_idx
                )
                
                # Use only the new interaction features
                interaction_cols = [col for col in interactions.columns 
                                  if col not in processed_data.columns]
                if interaction_cols:
                    processed_data = pd.concat([
                        processed_data, 
                        interactions[interaction_cols]
                    ], axis=1)
                    self.logger.info(f'   Added {len(interaction_cols)} interaction features')
            
            # Step 3: Add regime-aware features
            if self.feature_config.get('use_regime_features', True) and 'regime_label' in data.columns:
                self.logger.info(f'🏛️ Adding regime-aware features for {split_name}')
                regime_features = self._create_regime_features(processed_data)
                processed_data = pd.concat([processed_data, regime_features], axis=1)
                self.logger.info(f'   Added {regime_features.shape[1]} regime features')
            
            # Step 4: Add support/resistance features
            if self.feature_config.get('use_sr_features', True):
                self.logger.info(f'📈 Adding S/R features for {split_name}')
                sr_features = self._create_sr_features(processed_data)
                processed_data = pd.concat([processed_data, sr_features], axis=1)
                self.logger.info(f'   Added {sr_features.shape[1]} S/R features')
            
            # Step 5: Add time-based features
            time_features = self._create_time_features(processed_data)
            processed_data = pd.concat([processed_data, time_features], axis=1)
            
            # Step 6: Clean and validate final data
            processed_data = self._clean_and_validate_data(processed_data)
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f'❌ Failed to process {split_name} split: {e}')
            raise

    def _create_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create regime-aware features."""
        regime_features = pd.DataFrame(index=data.index)
        
        if 'regime_label' in data.columns:
            # Create regime dummy variables
            regime_dummies = pd.get_dummies(data['regime_label'], prefix='regime')
            regime_features = pd.concat([regime_features, regime_dummies], axis=1)
            
            # Create regime transition features
            regime_features['regime_changed'] = (data['regime_label'] != data['regime_label'].shift(1)).astype(int)
            regime_features['time_in_regime'] = data.groupby(
                (data['regime_label'] != data['regime_label'].shift()).cumsum()
            ).cumcount()
            
            # Create regime-specific technical indicators
            for regime in data['regime_label'].unique():
                if pd.notna(regime):
                    regime_mask = data['regime_label'] == regime
                    regime_features[f'regime_{regime}_count'] = regime_mask.cumsum()
        
        return regime_features

    def _create_sr_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create support/resistance features."""
        sr_features = pd.DataFrame(index=data.index)
        
        # Price position features
        sr_features['price_position_20'] = safe_divide(
            data['close'] - data['low'].rolling(20).min(),
            data['high'].rolling(20).max() - data['low'].rolling(20).min(),
            default=0.5
        )
        
        # Distance to support/resistance
        sr_features['dist_to_high_20'] = safe_divide(
            data['high'].rolling(20).max() - data['close'],
            data['close'],
            default=0.0
        )
        sr_features['dist_to_low_20'] = safe_divide(
            data['close'] - data['low'].rolling(20).min(),
            data['close'],
            default=0.0
        )
        
        # Breakout features
        sr_features['breakout_high_20'] = (data['close'] > data['high'].rolling(20).max().shift(1)).astype(int)
        sr_features['breakout_low_20'] = (data['close'] < data['low'].rolling(20).min().shift(1)).astype(int)
        
        return sr_features

    def _create_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        time_features = pd.DataFrame(index=data.index)
        
        if hasattr(data.index, 'hour'):
            time_features['hour'] = data.index.hour
            time_features['minute'] = data.index.minute
            time_features['hour_sin'] = np.sin(2 * np.pi * data.index.hour / 24)
            time_features['hour_cos'] = np.cos(2 * np.pi * data.index.hour / 24)
        
        if hasattr(data.index, 'dayofweek'):
            time_features['dayofweek'] = data.index.dayofweek
            time_features['is_monday'] = (data.index.dayofweek == 0).astype(int)
            time_features['is_friday'] = (data.index.dayofweek == 4).astype(int)
        
        return time_features

    def _clean_and_validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate processed data."""
        # Remove infinite values
        data = data.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill and then fill remaining NaN with 0
        data = data.ffill().fillna(0)
        
        # Remove duplicate columns
        data = data.loc[:, ~data.columns.duplicated()]
        
        return data

    def _get_data_to_process(self, pipeline_state: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Get data splits to process."""
        data_dict = {}
        
        for split in ['train', 'val', 'test']:
            if f'{split}_data' in pipeline_state:
                data_dict[split] = pipeline_state[f'{split}_data'].copy()
        
        if not data_dict and 'labeled_data' in pipeline_state:
            data_dict['all'] = pipeline_state['labeled_data'].copy()
        
        return data_dict

    def _calculate_feature_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate statistics for engineered features."""
        feature_cols = [col for col in data.columns if col.startswith(('feature_', 'regime_', 'sr_', 'time_'))]
        
        stats = {
            'n_features': len(feature_cols),
            'feature_names': feature_cols,
            'missing_values': {},
            'zero_variance': [],
            'high_correlation_pairs': []
        }
        
        for col in feature_cols:
            missing_pct = data[col].isna().sum() / len(data) * 100
            if missing_pct > 0:
                stats['missing_values'][col] = missing_pct
        
        for col in feature_cols:
            if data[col].std() < 1e-10:
                stats['zero_variance'].append(col)
        
        # Calculate correlations for a sample
        if len(feature_cols) > 1:
            sample_size = min(1000, len(data))
            sample_data = data[feature_cols].sample(n=sample_size)
            corr_matrix = sample_data.corr()
            
            for i in range(len(feature_cols)):
                for j in range(i + 1, len(feature_cols)):
                    if abs(corr_matrix.iloc[i, j]) > 0.95:
                        stats['high_correlation_pairs'].append((
                            feature_cols[i], feature_cols[j], corr_matrix.iloc[i, j]
                        ))
        
        return stats

    async def _perform_feature_selection(self, engineered_data: Dict[str, pd.DataFrame], 
                                       feature_statistics: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
        """Perform feature selection."""
        train_data = engineered_data.get('train', next(iter(engineered_data.values())))
        all_features = [col for col in train_data.columns if col.startswith(('feature_', 'regime_', 'sr_', 'time_'))]
        
        # Remove zero variance features
        zero_var_features = set()
        for stats in feature_statistics.values():
            zero_var_features.update(stats.get('zero_variance', []))
        
        valid_features = [f for f in all_features if f not in zero_var_features]
        
        # Remove highly correlated features
        to_remove = set()
        for stats in feature_statistics.values():
            for feat1, feat2, corr in stats.get('high_correlation_pairs', []):
                to_remove.add(feat2)
        
        valid_features = [f for f in valid_features if f not in to_remove]
        
        # Limit to max features
        max_features = self.feature_config.get('max_features', 500)
        if len(valid_features) > max_features:
            valid_features = valid_features[:max_features]
        
        # Select features
        selected_data = {}
        base_columns = [col for col in train_data.columns if not col.startswith(('feature_', 'regime_', 'sr_', 'time_'))]
        selected_columns = base_columns + valid_features
        
        for split_name, data in engineered_data.items():
            selected_data[split_name] = data[selected_columns]
        
        self.logger.info(f'✅ Selected {len(valid_features)} features from {len(all_features)} total')
        return selected_data, valid_features

    def _get_all_feature_columns(self, engineered_data: Dict[str, pd.DataFrame]) -> List[str]:
        """Get all feature columns from engineered data."""
        all_features = set()
        for data in engineered_data.values():
            if isinstance(data, pd.DataFrame):
                features = [col for col in data.columns if col.startswith(('feature_', 'regime_', 'sr_', 'time_'))]
                all_features.update(features)
        return sorted(list(all_features))

    def _generate_feature_reports(self, engineered_data: Dict[str, pd.DataFrame], 
                                feature_statistics: Dict[str, Dict[str, Any]], 
                                selected_features: List[str]) -> Dict[str, str]:
        """Generate feature engineering reports."""
        reports = {}
        
        # Summary report
        summary_lines = [
            'Enhanced Feature Engineering Summary',
            '=' * 50,
            f'Total features created: {len(selected_features)}',
            f'Processing time: {self.performance_metrics["total_processing_time"]:.2f}s',
            f'Memory used: {self.performance_metrics["total_memory_used_mb"]:.1f}MB',
            f'Chunks processed: {self.performance_metrics["chunks_processed"]}',
            '',
            'Data splits:'
        ]
        
        for split_name, data in engineered_data.items():
            if isinstance(data, pd.DataFrame):
                summary_lines.append(f'  {split_name}: {data.shape[0]} rows × {data.shape[1]} columns')
        
        reports['summary'] = '\n'.join(summary_lines)
        
        # Statistics report
        stats_lines = ['Feature Statistics', '=' * 30]
        for split_name, stats in feature_statistics.items():
            stats_lines.extend([
                '',
                f'{split_name.upper()} split:',
                f"  Total features: {stats.get('n_features', 0)}",
                f"  Features with missing values: {len(stats.get('missing_values', {}))}",
                f"  Zero variance features: {len(stats.get('zero_variance', []))}",
                f"  High correlation pairs: {len(stats.get('high_correlation_pairs', []))}"
            ])
        
        reports['statistics'] = '\n'.join(stats_lines)
        
        return reports

    async def _save_outputs(self, training_input: Dict[str, Any], 
                          pipeline_state: Dict[str, Any]) -> None:
        """Save step outputs to disk."""
        output_dir = Path(training_input.get('output_dir', 'output')) / 'step06_enhanced_feature_engineering'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save engineered data
        if 'engineered_data' in pipeline_state:
            for split_name, data in pipeline_state['engineered_data'].items():
                if isinstance(data, pd.DataFrame):
                    file_path = output_dir / f'{split_name}_engineered.parquet'
                    data.to_parquet(file_path)
                    self.logger.info(f'💾 Saved {split_name} engineered data to {file_path}')
        
        # Save selected features
        if 'selected_features' in pipeline_state:
            features_path = output_dir / 'selected_features.json'
            with open(features_path, 'w') as f:
                json.dump(pipeline_state['selected_features'], f, indent=2)
            self.logger.info(f'💾 Saved selected features to {features_path}')
        
        # Save performance metrics
        metrics_path = output_dir / 'performance_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.performance_metrics, f, indent=2)
        self.logger.info(f'💾 Saved performance metrics to {metrics_path}')

    def validate_outputs(self, pipeline_state: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate step outputs."""
        errors = []
        
        if 'engineered_data' not in pipeline_state:
            errors.append('No engineered data in pipeline state')
            return False, errors
        
        engineered_data = pipeline_state['engineered_data']
        total_features = 0
        
        for split_name, data in engineered_data.items():
            if isinstance(data, pd.DataFrame):
                feature_cols = [col for col in data.columns if col.startswith(('feature_', 'regime_', 'sr_', 'time_'))]
                total_features += len(feature_cols)
        
        if total_features == 0:
            errors.append('No features were engineered')
        
        if 'selected_features' in pipeline_state:
            selected = pipeline_state['selected_features']
            if len(selected) == 0:
                errors.append('No features were selected')
        
        return len(errors) == 0, errors

    def get_required_inputs(self) -> List[str]:
        """Get list of required inputs for this step."""
        return ['labeled_data or split data with labels']

    def get_produced_outputs(self) -> List[str]:
        """Get list of outputs produced by this step."""
        return [
            'engineered_data', 'feature_statistics', 'selected_features', 
            'feature_reports', 'performance_metrics'
        ]

    def get_dependencies(self) -> List[str]:
        """Get list of step dependencies."""
        return ['05_labeling']

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return self.performance_metrics.copy()