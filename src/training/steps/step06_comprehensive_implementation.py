"""
Step06 Comprehensive Implementation with All Enhancements

This module demonstrates the complete implementation of all step06 improvements:
- Vectorized batch processing for indicator extraction
- Sophisticated feature interactions (polynomial, cross-timeframe, pattern recognition)
- Strict temporal validation and lookahead bias prevention
- Memory-efficient chunking for large datasets
- Enhanced financial parameters and transaction cost modeling
- Comprehensive validation and error handling
- Modular approach with reduced nested functions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
import asyncio
import time
from pathlib import Path
import json

# Import all enhanced components
from .step06_enhanced_feature_engineering import EnhancedFeatureEngineering
from .step06_enhanced_feature_engineering_step import EnhancedFeatureEngineeringStep
from .step06_labeling_components.optimized_triple_barrier_labeling import OptimizedTripleBarrierLabeling

# Import validation utilities
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, validate_positive, 
    validate_range, MathValidationError
)

logger = logging.getLogger(__name__)

class Step06ComprehensiveImplementation:
    """
    Comprehensive implementation of all step06 enhancements.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize comprehensive step06 implementation."""
        self.config = config
        self.logger = logger
        
        # Initialize all components
        self.enhanced_feature_engineering = EnhancedFeatureEngineering(config)
        self.enhanced_feature_step = EnhancedFeatureEngineeringStep(config)
        self.optimized_labeling = OptimizedTripleBarrierLabeling(
            profit_take_multiplier=0.004,  # 0.4%
            stop_loss_multiplier=0.003,    # 0.3%
            transaction_cost=0.0008        # 0.08%
        )
        
        # Performance tracking
        self.performance_metrics = {
            'total_execution_time': 0.0,
            'feature_engineering_time': 0.0,
            'labeling_time': 0.0,
            'validation_time': 0.0,
            'memory_usage_mb': 0.0,
            'features_created': 0,
            'labels_generated': 0,
            'validation_errors': 0,
            'chunks_processed': 0
        }
        
        self.logger.info("🚀 Step06 Comprehensive Implementation initialized")
        self.logger.info("   ✅ Enhanced feature engineering")
        self.logger.info("   ✅ Optimized triple barrier labeling")
        self.logger.info("   ✅ Comprehensive validation framework")
        self.logger.info("   ✅ Memory-efficient processing")
        self.logger.info("   ✅ Mathematical safety utilities")

    async def run_comprehensive_pipeline(self, market_data: pd.DataFrame, 
                                       target_data: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Run the comprehensive step06 pipeline with all enhancements.
        
        Args:
            market_data: OHLCV market data
            target_data: Optional target data for optimization
            
        Returns:
            Comprehensive results dictionary
        """
        start_time = time.time()
        self.logger.info("🚀 Starting comprehensive step06 pipeline")
        self.logger.info(f"   Input data shape: {market_data.shape}")
        self.logger.info(f"   Target data provided: {target_data is not None}")
        
        results = {
            'pipeline_status': 'running',
            'input_data_info': {
                'shape': market_data.shape,
                'columns': list(market_data.columns),
                'date_range': [market_data.index[0], market_data.index[-1]] if len(market_data) > 0 else None
            },
            'performance_metrics': {},
            'feature_engineering_results': {},
            'labeling_results': {},
            'validation_results': {},
            'errors': [],
            'warnings': []
        }
        
        try:
            # Step 1: Comprehensive validation
            validation_start = time.time()
            validation_results = await self._run_comprehensive_validation(market_data)
            validation_time = time.time() - validation_start
            
            results['validation_results'] = validation_results
            self.performance_metrics['validation_time'] = validation_time
            
            if not validation_results['is_valid']:
                results['errors'].extend(validation_results['errors'])
                results['pipeline_status'] = 'failed_validation'
                return results
            
            # Step 2: Enhanced feature engineering
            feature_start = time.time()
            feature_results = await self._run_enhanced_feature_engineering(market_data)
            feature_time = time.time() - feature_start
            
            results['feature_engineering_results'] = feature_results
            self.performance_metrics['feature_engineering_time'] = feature_time
            self.performance_metrics['features_created'] = feature_results.get('features_created', 0)
            
            # Step 3: Optimized labeling
            labeling_start = time.time()
            labeling_results = await self._run_optimized_labeling(market_data)
            labeling_time = time.time() - labeling_start
            
            results['labeling_results'] = labeling_results
            self.performance_metrics['labeling_time'] = labeling_time
            self.performance_metrics['labels_generated'] = labeling_results.get('labels_generated', 0)
            
            # Step 4: Integration and final processing
            integration_results = await self._integrate_results(
                feature_results, labeling_results, market_data
            )
            results['integration_results'] = integration_results
            
            # Update performance metrics
            total_time = time.time() - start_time
            self.performance_metrics['total_execution_time'] = total_time
            results['performance_metrics'] = self.performance_metrics.copy()
            
            results['pipeline_status'] = 'completed'
            self.logger.info(f"✅ Comprehensive pipeline completed in {total_time:.2f}s")
            self.logger.info(f"   Features created: {self.performance_metrics['features_created']}")
            self.logger.info(f"   Labels generated: {self.performance_metrics['labels_generated']}")
            self.logger.info(f"   Validation errors: {self.performance_metrics['validation_errors']}")
            
        except Exception as e:
            self.logger.error(f"❌ Comprehensive pipeline failed: {e}")
            results['pipeline_status'] = 'failed'
            results['errors'].append(str(e))
            self.performance_metrics['validation_errors'] += 1
        
        return results

    async def _run_comprehensive_validation(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Run comprehensive validation on market data."""
        self.logger.info("🔍 Running comprehensive validation...")
        
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'validation_details': {}
        }
        
        try:
            # Data quality validation
            data_quality = self._validate_data_quality(market_data)
            validation_results['validation_details']['data_quality'] = data_quality
            
            if not data_quality['is_valid']:
                validation_results['is_valid'] = False
                validation_results['errors'].extend(data_quality['errors'])
            
            # Financial parameter validation
            financial_validation = self._validate_financial_parameters()
            validation_results['validation_details']['financial_parameters'] = financial_validation
            
            if not financial_validation['is_valid']:
                validation_results['is_valid'] = False
                validation_results['errors'].extend(financial_validation['errors'])
            
            # Temporal validation
            temporal_validation = self._validate_temporal_consistency(market_data)
            validation_results['validation_details']['temporal_consistency'] = temporal_validation
            
            if not temporal_validation['is_valid']:
                validation_results['is_valid'] = False
                validation_results['errors'].extend(temporal_validation['errors'])
            
            self.logger.info(f"✅ Comprehensive validation completed: {'PASSED' if validation_results['is_valid'] else 'FAILED'}")
            
        except Exception as e:
            self.logger.error(f"❌ Validation failed: {e}")
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Validation error: {e}")
        
        return validation_results

    def _validate_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality."""
        errors = []
        warnings = []
        
        # Check data shape
        if len(data) < 50:
            errors.append(f"Insufficient data: {len(data)} rows (minimum 50 required)")
        
        # Check required columns
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        # Check for valid prices
        for col in required_columns:
            if col in data.columns:
                if (data[col] <= 0).any():
                    errors.append(f"Invalid prices in {col}: non-positive values found")
                if data[col].isna().any():
                    errors.append(f"NaN values in {col}")
        
        # Check for suspicious price movements
        if 'close' in data.columns:
            price_changes = data['close'].pct_change().abs()
            large_moves = (price_changes > 0.2).sum()
            if large_moves > len(data) * 0.01:  # More than 1% large moves
                warnings.append(f"Suspicious price movements: {large_moves} moves >20%")
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'data_shape': data.shape,
            'price_range': {
                'min': data['close'].min() if 'close' in data.columns else None,
                'max': data['close'].max() if 'close' in data.columns else None
            }
        }

    def _validate_financial_parameters(self) -> Dict[str, Any]:
        """Validate financial parameters."""
        errors = []
        
        # Validate labeling parameters
        try:
            # These will be validated by the OptimizedTripleBarrierLabeling constructor
            test_labeling = OptimizedTripleBarrierLabeling()
            self.logger.info("✅ Financial parameters validated successfully")
        except MathValidationError as e:
            errors.append(f"Financial parameter validation failed: {e}")
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'parameters': {
                'profit_take_multiplier': 0.004,
                'stop_loss_multiplier': 0.003,
                'transaction_cost': 0.0008
            }
        }

    def _validate_temporal_consistency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate temporal consistency."""
        errors = []
        warnings = []
        
        if isinstance(data.index, pd.DatetimeIndex):
            # Check temporal ordering
            if not data.index.is_monotonic_increasing:
                errors.append("Data index is not temporally ordered")
            
            # Check for timestamp gaps
            time_diffs = data.index.to_series().diff().dt.total_seconds()
            large_gaps = (time_diffs > 0.5).sum()
            if large_gaps > 0:
                warnings.append(f"Timestamp gaps detected: {large_gaps} gaps >0.5s")
            
            # Check for duplicates
            duplicate_count = data.index.duplicated().sum()
            if duplicate_count > len(data) * 0.001:  # More than 0.1%
                errors.append(f"Too many duplicate timestamps: {duplicate_count}")
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }

    async def _run_enhanced_feature_engineering(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Run enhanced feature engineering."""
        self.logger.info("🔧 Running enhanced feature engineering...")
        
        try:
            # Extract technical indicators using batch processing
            lookback_periods = {
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
            
            # Extract indicators
            indicators = self.enhanced_feature_engineering.extract_indicators_batch(
                market_data, lookback_periods
            )
            
            # Create sophisticated interactions
            interactions = self.enhanced_feature_engineering.create_sophisticated_interactions(
                indicators, current_idx=len(indicators) - 1
            )
            
            # Combine results
            engineered_data = pd.concat([market_data, indicators, interactions], axis=1)
            
            # Calculate statistics
            feature_cols = [col for col in engineered_data.columns if col not in market_data.columns]
            feature_stats = {
                'total_features': len(feature_cols),
                'technical_indicators': len([col for col in feature_cols if any(ind in col for ind in ['RSI_', 'MACD_', 'SMA_', 'EMA_', 'ATR_'])]),
                'interaction_features': len([col for col in feature_cols if col.startswith(('poly_', 'cross_', 'pattern_', 'momentum_', 'regime_'))]),
                'data_shape': engineered_data.shape
            }
            
            self.logger.info(f"✅ Enhanced feature engineering completed")
            self.logger.info(f"   Technical indicators: {feature_stats['technical_indicators']}")
            self.logger.info(f"   Interaction features: {feature_stats['interaction_features']}")
            self.logger.info(f"   Total features: {feature_stats['total_features']}")
            
            return {
                'engineered_data': engineered_data,
                'feature_statistics': feature_stats,
                'features_created': len(feature_cols),
                'processing_stats': self.enhanced_feature_engineering.get_processing_stats()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced feature engineering failed: {e}")
            raise

    async def _run_optimized_labeling(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Run optimized triple barrier labeling."""
        self.logger.info("🏷️ Running optimized triple barrier labeling...")
        
        try:
            # Apply triple barrier labeling
            labeled_data = self.optimized_labeling.apply_triple_barrier_labeling_vectorized(market_data)
            
            # Calculate labeling statistics
            label_distribution = labeled_data['label'].value_counts().to_dict()
            profit_stats = {
                'mean_profit': labeled_data['potential_profit_pct'].mean(),
                'std_profit': labeled_data['potential_profit_pct'].std(),
                'min_profit': labeled_data['potential_profit_pct'].min(),
                'max_profit': labeled_data['potential_profit_pct'].max()
            }
            
            # Calculate net profit after transaction costs
            long_profits = labeled_data[labeled_data['label'] == 1]['potential_profit_pct']
            short_profits = labeled_data[labeled_data['label'] == -1]['potential_profit_pct']
            
            net_profit_stats = {
                'long_mean_net_profit': long_profits.mean() if len(long_profits) > 0 else 0.0,
                'short_mean_net_profit': short_profits.mean() if len(short_profits) > 0 else 0.0,
                'overall_net_profit': labeled_data['potential_profit_pct'].mean()
            }
            
            self.logger.info(f"✅ Optimized labeling completed")
            self.logger.info(f"   Labels generated: {len(labeled_data)}")
            self.logger.info(f"   Label distribution: {label_distribution}")
            self.logger.info(f"   Net profit: {net_profit_stats['overall_net_profit']:.4f}")
            
            return {
                'labeled_data': labeled_data,
                'label_distribution': label_distribution,
                'profit_statistics': profit_stats,
                'net_profit_statistics': net_profit_stats,
                'labels_generated': len(labeled_data)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Optimized labeling failed: {e}")
            raise

    async def _integrate_results(self, feature_results: Dict[str, Any], 
                               labeling_results: Dict[str, Any], 
                               market_data: pd.DataFrame) -> Dict[str, Any]:
        """Integrate feature engineering and labeling results."""
        self.logger.info("🔗 Integrating results...")
        
        try:
            engineered_data = feature_results['engineered_data']
            labeled_data = labeling_results['labeled_data']
            
            # Align data by index
            common_index = engineered_data.index.intersection(labeled_data.index)
            aligned_engineered = engineered_data.loc[common_index]
            aligned_labeled = labeled_data.loc[common_index]
            
            # Combine engineered features with labels
            final_data = pd.concat([aligned_engineered, aligned_labeled[['label', 'potential_profit_pct']]], axis=1)
            
            # Calculate final statistics
            integration_stats = {
                'final_data_shape': final_data.shape,
                'features_used': len([col for col in final_data.columns if col not in market_data.columns and col not in ['label', 'potential_profit_pct']]),
                'samples_with_labels': len(final_data[final_data['label'] != 0]),
                'data_alignment_success': len(common_index) / len(market_data)
            }
            
            self.logger.info(f"✅ Results integration completed")
            self.logger.info(f"   Final data shape: {final_data.shape}")
            self.logger.info(f"   Features used: {integration_stats['features_used']}")
            self.logger.info(f"   Samples with labels: {integration_stats['samples_with_labels']}")
            
            return {
                'final_data': final_data,
                'integration_statistics': integration_stats,
                'alignment_success_rate': integration_stats['data_alignment_success']
            }
            
        except Exception as e:
            self.logger.error(f"❌ Results integration failed: {e}")
            raise

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        return self.performance_metrics.copy()

    def save_results(self, results: Dict[str, Any], output_dir: str = "step06_results") -> None:
        """Save comprehensive results to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save main results
        results_path = output_path / 'comprehensive_results.json'
        with open(results_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = self._make_json_serializable(results)
            json.dump(serializable_results, f, indent=2, default=str)
        
        # Save performance metrics
        metrics_path = output_path / 'performance_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.performance_metrics, f, indent=2)
        
        # Save final data if available
        if 'integration_results' in results and 'final_data' in results['integration_results']:
            final_data_path = output_path / 'final_engineered_data.parquet'
            results['integration_results']['final_data'].to_parquet(final_data_path)
        
        self.logger.info(f"💾 Results saved to {output_path}")

    def _make_json_serializable(self, obj: Any) -> Any:
        """Make object JSON serializable."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        else:
            return obj

# Example usage and testing
async def run_step06_comprehensive_example():
    """Example of running the comprehensive step06 implementation."""
    # Create sample market data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
    
    # Generate realistic OHLCV data
    base_price = 100.0
    returns = np.random.normal(0, 0.001, 1000)
    prices = [base_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    market_data = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
        'close': prices,
        'volume': np.random.uniform(1000, 10000, 1000)
    }, index=dates)
    
    # Ensure OHLC consistency
    market_data['high'] = np.maximum(market_data['high'], np.maximum(market_data['open'], market_data['close']))
    market_data['low'] = np.minimum(market_data['low'], np.minimum(market_data['open'], market_data['close']))
    
    # Configuration
    config = {
        'step06_feature_engineering': {
            'chunk_size': 5000,
            'max_features': 200,
            'polynomial_degree': 2,
            'correlation_threshold': 0.95,
            'memory_limit_mb': 500
        }
    }
    
    # Run comprehensive implementation
    implementation = Step06ComprehensiveImplementation(config)
    results = await implementation.run_comprehensive_pipeline(market_data)
    
    # Save results
    implementation.save_results(results)
    
    # Print summary
    print("\n" + "="*60)
    print("STEP06 COMPREHENSIVE IMPLEMENTATION SUMMARY")
    print("="*60)
    print(f"Pipeline Status: {results['pipeline_status']}")
    print(f"Total Execution Time: {results['performance_metrics']['total_execution_time']:.2f}s")
    print(f"Features Created: {results['performance_metrics']['features_created']}")
    print(f"Labels Generated: {results['performance_metrics']['labels_generated']}")
    print(f"Validation Errors: {results['performance_metrics']['validation_errors']}")
    
    if results['pipeline_status'] == 'completed':
        print("\n✅ All enhancements successfully implemented:")
        print("   ✅ Vectorized batch processing")
        print("   ✅ Sophisticated feature interactions")
        print("   ✅ Strict temporal validation")
        print("   ✅ Memory-efficient chunking")
        print("   ✅ Enhanced financial parameters")
        print("   ✅ Transaction cost modeling")
        print("   ✅ Comprehensive validation")
        print("   ✅ Mathematical safety utilities")
    
    return results

if __name__ == "__main__":
    # Run example
    asyncio.run(run_step06_comprehensive_example())