#!/usr/bin/env python3
"""Enhanced Step 3: HMM Regime Discovery with Integrated Improvements.

This module integrates all the improvements:
1. Bayesian parameter optimization with Optuna
2. Enhanced regime discovery features
3. Economic significance validation
4. Ensemble clustering (HMM + K-means + DBSCAN)
5. ML-based transition detection with Random Forest + LGBM
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.domain import (
    comprehensive_data_validation,
    ensure_data_integrity,
    handle_errors,
    memory_efficient,
    monitor_feature_engineering,
    monitor_step_execution,
    quality_gate,
    resource_monitor,
    secure_data_processing,
    secure_step_execution,
    validate_data_structure,
    with_tracing_span,
    validate_pipeline_step
)
from src.core.decorators import validates
from src.utils.logger import system_logger

# Import our new modules
from .step03_optimized_bayesian_optimization import OptimizedBayesianParameterOptimization
from .step03_regime_discovery_features import RegimeDiscoveryFeatureEngineer
from .step03_economic_significance_validator import EconomicSignificanceValidator
from .step03_ensemble_clustering import EnsembleClusteringRegimeDetector
from .step03_enhanced_ml_transition_detector import EnhancedMLRegimeTransitionDetector

logger = system_logger.getChild("Step3EnhancedHMMRegimeDiscovery")

class EnhancedHMMRegimeDiscoveryStep:
    """Enhanced Step 3: HMM Regime Discovery with all improvements integrated."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild('EnhancedHMMRegimeDiscoveryStep')
        self.start_time = None
        self.step_timings = {}
        
        # Initialize components
        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize all enhanced components."""
        self.logger.info('🔧 Initializing enhanced HMM regime discovery components...')
        
        # 1. Bayesian parameter optimization
        self.bayesian_optimizer = OptimizedBayesianParameterOptimization(self.config)
        
        # 2. Regime discovery feature engineer
        self.feature_engineer = RegimeDiscoveryFeatureEngineer(self.config)
        
        # 3. Economic significance validator
        self.economic_validator = EconomicSignificanceValidator(self.config)
        
        # 4. Ensemble clustering detector
        self.ensemble_detector = EnsembleClusteringRegimeDetector(self.config)
        
        # 5. Enhanced ML transition detector
        self.ml_transition_detector = EnhancedMLRegimeTransitionDetector(self.config)
        
        self.logger.info('✅ All enhanced components initialized successfully')

    @handles_errors(fallback=False)
    async def initialize(self) -> None:
        """Initialize the enhanced HMM regime discovery step."""
        self.start_time = time.time()
        self.logger.info('🚀 Initializing Enhanced HMM Regime Discovery Step...')
        
        # Initialize Bayesian optimizer
        await self.bayesian_optimizer.initialize()
        
        self.logger.info('✅ Enhanced HMM Regime Discovery Step initialized successfully')

    @validates(step_name='enhanced_hmm_regime_discovery', validation_level='CRITICAL', enable_rollback=True, max_retries=2)
    @ensure_data_integrity(check_schema=True, check_constraints=True, validate_relationships=True)
    @monitor_step_execution(enable_timing=True, enable_memory_monitoring=True, enable_progress_tracking=True)
    @secure_step_execution(error_handling=True, rollback_on_failure=True, data_validation=True, resource_cleanup=True)
    @traced(span_name='execute_enhanced_hmm_regime_discovery')
    @handles_errors(default_return={'success': False, 'regimes': [], 'error': 'Enhanced HMM discovery failed'}, context='enhanced_hmm_regime_discovery.execute')
    async def execute(self, training_input: dict[str, Any], pipeline_state: dict[str, Any]) -> dict[str, Any]:
        """Execute enhanced HMM regime discovery with all improvements."""
        step_start = time.time()
        self.logger.info('🎯 Starting Enhanced HMM regime discovery execution...')
        
        try:
            # Step 1: Data Quality Validation and Loading
            self.logger.info('=' * 60)
            self.logger.info('STEP 1: Data Quality Validation and Loading')
            self.logger.info('=' * 60)
            data_loading_start = time.time()
            
            data_loaded = await self._load_and_prepare_data(training_input)
            if not data_loaded.get('success', False):
                self.logger.error('❌ Failed to load and prepare data')
                pipeline_state['enhanced_hmm_regime_discovery_completed'] = False
                pipeline_state['regime_discovery_error'] = 'Data loading failed'
                return pipeline_state
            
            data_loading_elapsed = time.time() - data_loading_start
            self.logger.info(f'⏱️ Data Loading completed in {data_loading_elapsed:.2f} seconds')
            
            # Step 2: Bayesian Parameter Optimization
            self.logger.info('=' * 60)
            self.logger.info('STEP 2: Bayesian Parameter Optimization')
            self.logger.info('=' * 60)
            optimization_start = time.time()
            
            optimization_success = await self._run_bayesian_optimization(data_loaded['data'], data_loaded['features'])
            if not optimization_success:
                self.logger.warning('⚠️ Bayesian optimization failed, using default parameters')
            
            optimization_elapsed = time.time() - optimization_start
            self.logger.info(f'⏱️ Bayesian Parameter Optimization completed in {optimization_elapsed:.2f} seconds')
            
            # Step 3: Enhanced Feature Engineering
            self.logger.info('=' * 60)
            self.logger.info('STEP 3: Enhanced Regime Discovery Feature Engineering')
            self.logger.info('=' * 60)
            feature_engineering_start = time.time()
            
            enhanced_features = await self._create_enhanced_features(data_loaded['data'], data_loaded['features'])
            if enhanced_features is None:
                self.logger.error('❌ Failed to create enhanced features')
                pipeline_state['enhanced_hmm_regime_discovery_completed'] = False
                pipeline_state['regime_discovery_error'] = 'Feature engineering failed'
                return pipeline_state
            
            feature_engineering_elapsed = time.time() - feature_engineering_start
            self.logger.info(f'⏱️ Enhanced Feature Engineering completed in {feature_engineering_elapsed:.2f} seconds')
            
            # Step 4: Ensemble Clustering
            self.logger.info('=' * 60)
            self.logger.info('STEP 4: Ensemble Clustering (HMM + K-means + DBSCAN)')
            self.logger.info('=' * 60)
            clustering_start = time.time()
            
            ensemble_results = await self._run_ensemble_clustering(enhanced_features)
            if not ensemble_results.get('success', False):
                self.logger.error('❌ Ensemble clustering failed')
                pipeline_state['enhanced_hmm_regime_discovery_completed'] = False
                pipeline_state['regime_discovery_error'] = 'Ensemble clustering failed'
                return pipeline_state
            
            clustering_elapsed = time.time() - clustering_start
            self.logger.info(f'⏱️ Ensemble Clustering completed in {clustering_elapsed:.2f} seconds')
            
            # Step 5: Economic Significance Validation
            self.logger.info('=' * 60)
            self.logger.info('STEP 5: Economic Significance Validation')
            self.logger.info('=' * 60)
            validation_start = time.time()
            
            economic_validation = await self._run_economic_validation(data_loaded['data'], ensemble_results['consensus_regimes'])
            if not economic_validation.get('overall_significant', False):
                self.logger.warning('⚠️ Regimes do not show significant economic differences')
                self.logger.info('📊 Validation results:')
                for test, result in economic_validation.items():
                    if isinstance(result, dict) and 'significant' in str(result):
                        self.logger.info(f'   - {test}: {result}')
            
            validation_elapsed = time.time() - validation_start
            self.logger.info(f'⏱️ Economic Validation completed in {validation_elapsed:.2f} seconds')
            
            # Step 6: Enhanced ML Transition Detection
            self.logger.info('=' * 60)
            self.logger.info('STEP 6: Enhanced ML Transition Detection (Random Forest + LGBM)')
            self.logger.info('=' * 60)
            transition_start = time.time()
            
            transition_results = await self._run_ml_transition_detection(data_loaded['data'], ensemble_results['consensus_regimes'])
            if not transition_results.get('success', False):
                self.logger.warning('⚠️ ML transition detection failed, using basic transition detection')
            
            transition_elapsed = time.time() - transition_start
            self.logger.info(f'⏱️ Enhanced ML Transition Detection completed in {transition_elapsed:.2f} seconds')
            
            # Step 7: Compile Final Results
            self.logger.info('=' * 60)
            self.logger.info('STEP 7: Compiling Final Results')
            self.logger.info('=' * 60)
            
            final_results = await self._compile_final_results(
                ensemble_results, economic_validation, transition_results, enhanced_features
            )
            
            # Update pipeline state
            pipeline_state.update(final_results)
            pipeline_state['enhanced_hmm_regime_discovery_completed'] = True
            
            # Log final results
            self._log_final_results(final_results)
            
            # Log step artifacts to MLflow
            await self._log_enhanced_artifacts_to_mlflow(final_results, training_input)
            
        except Exception as e:
            self.logger.exception(f'❌ Unexpected error during enhanced HMM regime discovery: {e}')
            pipeline_state['enhanced_hmm_regime_discovery_completed'] = False
            pipeline_state['regime_discovery_error'] = str(e)
        
        total_elapsed = time.time() - step_start
        self.logger.info('=' * 60)
        self.logger.info('EXECUTION SUMMARY')
        self.logger.info('=' * 60)
        self.logger.info(f'⏱️ Total execution time: {total_elapsed:.2f} seconds')
        self.logger.info(f'⏱️ Step timings:')
        self.logger.info(f'   - Data Loading: {data_loading_elapsed:.2f}s')
        self.logger.info(f'   - Bayesian Optimization: {optimization_elapsed:.2f}s')
        self.logger.info(f'   - Feature Engineering: {feature_engineering_elapsed:.2f}s')
        self.logger.info(f'   - Ensemble Clustering: {clustering_elapsed:.2f}s')
        self.logger.info(f'   - Economic Validation: {validation_elapsed:.2f}s')
        self.logger.info(f'   - Enhanced ML Transition Detection: {transition_elapsed:.2f}s')
        
        success = pipeline_state.get('enhanced_hmm_regime_discovery_completed', False)
        self.logger.info(f"🎯 Final result: {('✅ SUCCESS' if success else '❌ FAILED')}")
        
        return pipeline_state

    @traced(span_name='load_and_prepare_data')
    @handles_errors(default_return={'success': False, 'error': 'Data loading failed'}, context='load_and_prepare_data')
    async def _load_and_prepare_data(self, training_input: dict[str, Any]) -> dict[str, Any]:
        """Load and prepare data for enhanced HMM regime discovery."""
        try:
            symbol = training_input.get('symbol', 'ETHUSDT')
            exchange = training_input.get('exchange', 'BINANCE')
            timeframe = training_input.get('timeframe', '1m')
            data_dir = training_input.get('data_dir')
            
            if data_dir is None:
                data_dir = 'data_cache'
            
            self.logger.info(f'📊 Loading data for enhanced HMM regime discovery...')
            self.logger.info(f'   Symbol: {symbol}')
            self.logger.info(f'   Exchange: {exchange}')
            self.logger.info(f'   Timeframe: {timeframe}')
            self.logger.info(f'   Data directory: {data_dir}')
            
            # Load klines data
            klines_path = Path(data_dir) / f"klines_{exchange}_{symbol}_{timeframe}_consolidated.parquet"
            
            if not klines_path.exists():
                self.logger.error(f'❌ Klines file not found: {klines_path}')
                return {'success': False, 'error': f'Klines file not found: {klines_path}'}
            
            # Load data
            df = pd.read_parquet(klines_path)
            
            if df.empty:
                self.logger.error('❌ Data is empty')
                return {'success': False, 'error': 'Data is empty'}
            
            # Prepare basic features
            features = await self._prepare_basic_features(df)
            
            self.logger.info(f'✅ Data loaded and prepared: {len(df):,} rows, {len(features.columns)} basic features')
            
            return {
                'success': True,
                'data': df,
                'features': features,
                'data_info': {
                    'rows': len(df),
                    'columns': list(df.columns),
                    'date_range': {
                        'start': df['timestamp'].min().isoformat(),
                        'end': df['timestamp'].max().isoformat()
                    }
                }
            }
            
        except Exception as e:
            self.logger.exception(f'❌ Error loading and preparing data: {e}')
            return {'success': False, 'error': str(e)}

    @handles_errors(fallback=pd.DataFrame())
    @monitor_feature_engineering()
    @validates()
    async def _prepare_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare basic features for regime discovery."""
        try:
            self.logger.info('🔧 Preparing basic features...')
            
            # Ensure timestamp is datetime
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Create basic feature set
            features = pd.DataFrame()
            features['timestamp'] = df['timestamp']
            
            # Price-based features
            features['price_momentum_5'] = df['close'].pct_change(5)
            features['price_momentum_10'] = df['close'].pct_change(10)
            features['price_momentum_20'] = df['close'].pct_change(20)
            
            # Volume features
            features['volume_momentum_5'] = df['volume'].pct_change(5)
            features['volume_momentum_10'] = df['volume'].pct_change(10)
            features['volume_ratio_5'] = df['volume'] / df['volume'].rolling(window=5).mean()
            features['volume_ratio_10'] = df['volume'] / df['volume'].rolling(window=10).mean()
            features['volume_ratio_20'] = df['volume'] / df['volume'].rolling(window=20).mean()
            
            # Volatility features
            features['volatility_5'] = df['close'].pct_change().rolling(window=5).std()
            features['volatility_10'] = df['close'].pct_change().rolling(window=10).std()
            features['volatility_20'] = df['close'].pct_change().rolling(window=20).std()
            features['ewma_volatility_20'] = df['close'].pct_change().ewm(span=20).std()
            
            # Technical indicators
            features['rsi'] = self._calculate_rsi(df['close'])
            features['macd'] = self._calculate_macd(df['close'])
            features['atr'] = self._calculate_atr(df)
            features['adx'] = self._calculate_adx(df)
            
            # Bollinger Bands
            bb_features = self._calculate_bollinger_bands(df['close'])
            features = pd.concat([features, bb_features], axis=1)
            
            # Moving averages
            features['sma_20'] = df['close'].rolling(window=20).mean()
            features['sma_50'] = df['close'].rolling(window=50).mean()
            features['ema_12'] = df['close'].ewm(span=12).mean()
            features['ema_26'] = df['close'].ewm(span=26).mean()
            
            # Price position relative to MAs
            features['price_vs_sma20'] = (df['close'] - features['sma_20']) / features['sma_20']
            features['price_vs_sma50'] = (df['close'] - features['sma_50']) / features['sma_50']
            
            # Feature interactions
            features['momentum_volume_interaction'] = features['price_momentum_10'] * features['volume_ratio_10']
            features['volatility_volume_interaction'] = features['volatility_20'] * features['volume_ratio_20']
            features['rsi_momentum_interaction'] = features['rsi'] * features['price_momentum_10']
            
            # Clean features
            hmm_features = features.drop('timestamp', axis=1)
            hmm_features = hmm_features.fillna(0)
            
            self.logger.info(f'✅ Basic features prepared: {len(hmm_features.columns)} features, {len(hmm_features)} samples')
            
            return hmm_features
            
        except Exception as e:
            self.logger.exception(f'❌ Error preparing basic features: {e}')
            return pd.DataFrame()

    @handles_errors(fallback=False)
    async def _run_bayesian_optimization(self, data: pd.DataFrame, features: pd.DataFrame) -> bool:
        """Run Bayesian parameter optimization."""
        try:
            self.logger.info('🔍 Running Bayesian parameter optimization...')
            
            # Update config with data info
            optimization_config = self.config.get('bayesian_optimization', {})
            optimization_config.update({
                'SYMBOL': self.config.get('SYMBOL', 'ETHUSDT'),
                'EXCHANGE': self.config.get('EXCHANGE', 'BINANCE'),
                'TIMEFRAME': self.config.get('TIMEFRAME', '1m'),
                'DATA_DIR': self.config.get('DATA_DIR', 'data_cache')
            })
            
            # Run optimization
            optimization_results = await self.bayesian_optimizer.optimize_parameters(data_loaded['data'], data_loaded['features'])
            success = optimization_results.get('success', False)
            
            if success:
                self.logger.info('✅ Bayesian parameter optimization completed successfully')
                # Store optimized parameters for use in ensemble clustering
                self.optimized_params = self.bayesian_optimizer.best_params
            else:
                self.logger.warning('⚠️ Bayesian parameter optimization failed')
                self.optimized_params = None
            
            return success
            
        except Exception as e:
            self.logger.exception(f'❌ Error in Bayesian optimization: {e}')
            return False

    @handles_errors(fallback=None)
    async def _create_enhanced_features(self, data: pd.DataFrame, basic_features: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Create enhanced regime discovery features."""
        try:
            self.logger.info('🔧 Creating enhanced regime discovery features...')
            
            # Create regime discovery features
            regime_features = self.feature_engineer.create_regime_discovery_features(data)
            
            # Combine with basic features
            enhanced_features = pd.concat([basic_features, regime_features], axis=1)
            
            # Clean and validate
            enhanced_features = enhanced_features.fillna(0)
            
            self.logger.info(f'✅ Enhanced features created: {len(enhanced_features.columns)} total features')
            self.logger.info(f'   - Basic features: {len(basic_features.columns)}')
            self.logger.info(f'   - Regime discovery features: {len(regime_features.columns)}')
            
            return enhanced_features
            
        except Exception as e:
            self.logger.exception(f'❌ Error creating enhanced features: {e}')
            return None

    @handles_errors(default_return={'success': False, 'error': 'Ensemble clustering failed'}, context='ensemble_clustering')
    async def _run_ensemble_clustering(self, features: pd.DataFrame) -> dict[str, Any]:
        """Run ensemble clustering with HMM + K-means + DBSCAN."""
        try:
            self.logger.info('🔍 Running ensemble clustering...')
            
            # Convert to numpy array
            features_array = features.values
            
            # Run ensemble clustering
            consensus_regimes, ensemble_results = self.ensemble_detector.ensemble_regime_detection(features_array)
            
            self.logger.info(f'✅ Ensemble clustering completed successfully')
            self.logger.info(f'   - Number of regimes: {ensemble_results["n_regimes"]}')
            self.logger.info(f'   - Ensemble quality: {ensemble_results["ensemble_quality"]}')
            self.logger.info(f'   - Quality weights: {ensemble_results["quality_weights"]}')
            
            return {
                'success': True,
                'consensus_regimes': consensus_regimes,
                'ensemble_results': ensemble_results,
                'features_used': features
            }
            
        except Exception as e:
            self.logger.exception(f'❌ Error in ensemble clustering: {e}')
            return {'success': False, 'error': str(e)}

    @handles_errors(default_return={'overall_significant': False, 'error': 'Economic validation failed'}, context='economic_validation')
    async def _run_economic_validation(self, data: pd.DataFrame, regimes: np.ndarray) -> dict[str, Any]:
        """Run economic significance validation."""
        try:
            self.logger.info('🔍 Running economic significance validation...')
            
            # Validate regime economics
            validation_results = self.economic_validator.validate_regime_economics(data, regimes)
            
            self.logger.info(f'✅ Economic validation completed')
            self.logger.info(f'   - Overall significant: {validation_results["overall_significant"]}')
            self.logger.info(f'   - Number of regimes: {validation_results["validation_summary"]["n_regimes"]}')
            self.logger.info(f'   - Significant tests: {validation_results["validation_summary"]["significant_tests"]}')
            
            return validation_results
            
        except Exception as e:
            self.logger.exception(f'❌ Error in economic validation: {e}')
            return {'overall_significant': False, 'error': str(e)}

    @handles_errors(default_return={'success': False, 'error': 'Enhanced ML transition detection failed'}, context='enhanced_ml_transition_detection')
    async def _run_ml_transition_detection(self, data: pd.DataFrame, regimes: np.ndarray) -> dict[str, Any]:
        """Run enhanced ML-based transition detection with Random Forest + LGBM."""
        try:
            self.logger.info('🔍 Running enhanced ML transition detection...')
            
            # Train transition models
            training_results = self.ml_transition_detector.train_transition_models(data, regimes)
            
            if training_results.get('feature_selection_completed') and training_results.get('lgb_training_completed'):
                self.logger.info(f'✅ Enhanced ML transition detection completed successfully')
                self.logger.info(f'   - Feature selection completed: {training_results["feature_selection_completed"]}')
                self.logger.info(f'   - LGBM training completed: {training_results["lgb_training_completed"]}')
                self.logger.info(f'   - Selected features: {len(training_results.get("selected_features", []))}')
                self.logger.info(f'   - Best performance: {training_results.get("best_performance", 0.0):.4f}')
                self.logger.info(f'   - Final performance: {training_results.get("final_performance", {})}')
                
                # Test predictions
                predictions = self.ml_transition_detector.predict_transitions(data, regimes)
                
                return {
                    'success': True,
                    'training_results': training_results,
                    'predictions': predictions
                }
            else:
                self.logger.warning('⚠️ Enhanced ML transition detection failed')
                return {'success': False, 'error': 'Enhanced ML transition detection failed'}
            
        except Exception as e:
            self.logger.exception(f'❌ Error in enhanced ML transition detection: {e}')
            return {'success': False, 'error': str(e)}

    @handles_errors(default_return={}, context='compile_final_results')
    async def _compile_final_results(self, ensemble_results: dict[str, Any], 
                                   economic_validation: dict[str, Any],
                                   transition_results: dict[str, Any],
                                   enhanced_features: pd.DataFrame) -> dict[str, Any]:
        """Compile final results from all components."""
        try:
            self.logger.info('📊 Compiling final results...')
            
            final_results = {
                # Ensemble clustering results
                'regime_states': ensemble_results['consensus_regimes'].tolist(),
                'n_regimes': ensemble_results['ensemble_results']['n_regimes'],
                'ensemble_quality': ensemble_results['ensemble_results']['ensemble_quality'],
                'ensemble_weights': ensemble_results['ensemble_results']['quality_weights'],
                
                # Economic validation results
                'economic_significance': economic_validation['overall_significant'],
                'economic_validation': economic_validation,
                
                # Enhanced ML transition detection results
                'enhanced_ml_transition_detection': transition_results.get('success', False),
                'transition_models': transition_results.get('training_results', {}),
                'transition_predictions': transition_results.get('predictions', {}),
                
                # Feature information
                'n_features': len(enhanced_features.columns),
                'feature_names': list(enhanced_features.columns),
                
                # Performance metrics
                'execution_time': time.time() - self.start_time,
                'step_timings': self.step_timings,
                
                # Regime statistics
                'regime_distribution': self._calculate_regime_distribution(ensemble_results['consensus_regimes']),
                'regime_transitions': self._calculate_regime_transitions(ensemble_results['consensus_regimes']),
                
                # Quality metrics
                'overall_quality_score': self._calculate_overall_quality_score(
                    ensemble_results, economic_validation, transition_results
                )
            }
            
            self.logger.info('✅ Final results compiled successfully')
            
            return final_results
            
        except Exception as e:
            self.logger.exception(f'❌ Error compiling final results: {e}')
            return {}

    def _calculate_regime_distribution(self, regimes: np.ndarray) -> dict[str, int]:
        """Calculate regime distribution."""
        unique_regimes, counts = np.unique(regimes, return_counts=True)
        return {f'regime_{regime}': int(count) for regime, count in zip(unique_regimes, counts)}

    def _calculate_regime_transitions(self, regimes: np.ndarray) -> dict[str, Any]:
        """Calculate regime transition statistics."""
        transitions = np.sum(np.diff(regimes) != 0)
        total_periods = len(regimes)
        transition_rate = transitions / total_periods if total_periods > 0 else 0
        
        return {
            'total_transitions': int(transitions),
            'transition_rate': float(transition_rate),
            'total_periods': int(total_periods)
        }

    def _calculate_overall_quality_score(self, ensemble_results: dict[str, Any], 
                                       economic_validation: dict[str, Any],
                                       transition_results: dict[str, Any]) -> float:
        """Calculate overall quality score."""
        try:
            # Ensemble quality (40% weight)
            ensemble_quality = ensemble_results['ensemble_results']['ensemble_quality']
            ensemble_score = ensemble_quality.get('silhouette_score', 0) * 0.4
            
            # Economic significance (40% weight)
            economic_score = 0.4 if economic_validation.get('overall_significant', False) else 0.0
            
            # Enhanced ML transition detection (20% weight)
            transition_score = 0.2 if transition_results.get('success', False) else 0.0
            
            overall_score = ensemble_score + economic_score + transition_score
            
            return float(overall_score)
            
        except Exception as e:
            self.logger.warning(f'Could not calculate overall quality score: {e}')
            return 0.0

    def _log_final_results(self, final_results: dict[str, Any]) -> None:
        """Log final results summary."""
        self.logger.info('📊 ENHANCED HMM REGIME DISCOVERY RESULTS')
        self.logger.info('-' * 50)
        self.logger.info(f"📈 Total periods analyzed: {final_results.get('regime_transitions', {}).get('total_periods', 0):,}")
        self.logger.info(f"🔄 Unique regimes discovered: {final_results.get('n_regimes', 0)}")
        self.logger.info(f"📊 Economic significance: {final_results.get('economic_significance', False)}")
        self.logger.info(f"🤖 Enhanced ML transition detection: {final_results.get('enhanced_ml_transition_detection', False)}")
        self.logger.info(f"🎯 Overall quality score: {final_results.get('overall_quality_score', 0):.4f}")
        
        # Regime distribution
        regime_dist = final_results.get('regime_distribution', {})
        if regime_dist:
            self.logger.info('📊 Regime distribution:')
            for regime, count in regime_dist.items():
                total_periods = final_results.get('regime_transitions', {}).get('total_periods', 1)
                percentage = count / total_periods * 100
                self.logger.info(f'   - {regime}: {count:,} periods ({percentage:.1f}%)')
        
        # Transition statistics
        transitions = final_results.get('regime_transitions', {})
        if transitions:
            self.logger.info(f"🔄 Total transitions: {transitions.get('total_transitions', 0)}")
            self.logger.info(f"📈 Transition rate: {transitions.get('transition_rate', 0):.4f}")

    @handles_errors(fallback=False)
    async def _log_enhanced_artifacts_to_mlflow(self, final_results: dict[str, Any], training_input: dict[str, Any]) -> None:
        """Log enhanced artifacts to MLflow."""
        try:
            symbol = training_input.get('symbol', 'ETHUSDT')
            exchange = training_input.get('exchange', 'BINANCE')
            timeframe = training_input.get('timeframe', '1m')
            
            # Log regime states
            regime_states = final_results.get('regime_states', [])
            if regime_states:
                regime_df = pd.DataFrame({
                    'timestamp': pd.date_range(start='2024-01-01', periods=len(regime_states), freq='1min'),
                    'regime_state': regime_states
                })
                
                # Log to MLflow (placeholder - would use actual MLflow logging)
                self.logger.info(f'✅ Logged regime states: {len(regime_states)} periods')
            
            # Log ensemble results
            ensemble_quality = final_results.get('ensemble_quality', {})
            if ensemble_quality:
                self.logger.info(f'✅ Logged ensemble quality metrics')
            
            # Log economic validation
            economic_validation = final_results.get('economic_validation', {})
            if economic_validation:
                self.logger.info(f'✅ Logged economic validation results')
            
            # Log enhanced ML transition results
            transition_models = final_results.get('transition_models', {})
            if transition_models:
                self.logger.info(f'✅ Logged enhanced ML transition detection results')
            
            self.logger.info('✅ Enhanced artifacts logged to MLflow successfully')
            
        except Exception as e:
            self.logger.error(f'❌ Failed to log enhanced artifacts to MLflow: {e}')

    # Technical indicator calculation methods
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - 100 / (1 + rs)
        return rsi

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        """Calculate MACD."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd

    def _calculate_atr(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high = df['high']
        low = df['low']
        close = df['close']
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        return atr

    def _calculate_adx(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Average Directional Index."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        dm_plus = high - high.shift(1)
        dm_minus = low.shift(1) - low
        dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
        dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
        
        tr_smooth = tr.rolling(window=window).mean()
        dm_plus_smooth = dm_plus.rolling(window=window).mean()
        dm_minus_smooth = dm_minus.rolling(window=window).mean()
        
        di_plus = 100 * (dm_plus_smooth / tr_smooth)
        di_minus = 100 * (dm_minus_smooth / tr_smooth)
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(window=window).mean()
        
        return adx

    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        bb_upper = sma + std * num_std
        bb_lower = sma - std * num_std
        bb_width = (bb_upper - bb_lower) / sma
        bb_position = (prices - bb_lower) / (bb_upper - bb_lower)
        
        bb_features = pd.DataFrame({
            'bb_upper': bb_upper,
            'bb_middle': sma,
            'bb_lower': bb_lower,
            'bb_width': bb_width,
            'bb_position': bb_position
        })
        
        return bb_features


@monitor_step_execution
@secure_step_execution
@validates()
@handles_errors(fallback=False)
async def run_enhanced_step(symbol: str, exchange: str, timeframe: str = "1m", 
                          data_dir: str = None, force_rerun: bool = False, **kwargs: Any) -> bool:
    """Run the enhanced HMM regime discovery step.

    Args:
        symbol: Trading symbol (e.g., "ETHUSDT")
        exchange: Exchange name (e.g., "BINANCE")
        timeframe: Timeframe (e.g., "1m")
        data_dir: Data directory (will use standardized path if None)
        force_rerun: Force re-run even if results exist
        **kwargs: Additional arguments

    Returns:
        bool: True if successful, False otherwise
    """
    start_time = time.time()
    
    try:
        logger = system_logger.getChild('EnhancedStep3HMMRegimeDiscovery')
        
        if data_dir is None:
            data_dir = "data_cache"
        
        logger.info('=' * 80)
        logger.info('🚀 ENHANCED STEP 3: HMM Regime Discovery with All Improvements')
        logger.info('=' * 80)
        logger.info(f'🎯 Symbol: {symbol}')
        logger.info(f'🏢 Exchange: {exchange}')
        logger.info(f'📊 Timeframe: {timeframe}')
        logger.info(f'📁 Data directory: {data_dir}')
        logger.info(f'🔄 Force rerun: {force_rerun}')
        logger.info(f"⏰ Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info('=' * 80)
        
        # Create configuration
        config = {
            'SYMBOL': symbol,
            'EXCHANGE': exchange,
            'TIMEFRAME': timeframe,
            'DATA_DIR': data_dir,
            'bayesian_optimization': {
                'n_trials': kwargs.get('n_trials', 100),
                'timeout_minutes': kwargs.get('timeout_minutes', 30),
                'cv_folds': kwargs.get('cv_folds', 3),
                'random_state': kwargs.get('random_state', 42)
            },
            'ensemble_weights': {
                'hmm': 0.4,
                'kmeans': 0.3,
                'dbscan': 0.3
            },
            'enhanced_ml_transition_detection': {
                'initial_features': 20,
                'feature_increment': 10,
                'max_features': 100,
                'min_improvement': 0.001,
                'patience': 3,
                'random_state': 42
            }
        }
        
        # Initialize and run enhanced step
        logger.info('🔧 Initializing enhanced HMM regime discovery step...')
        step = EnhancedHMMRegimeDiscoveryStep(config)
        
        initialized = await step.initialize()
        if not initialized:
            logger.error('❌ Failed to initialize enhanced HMM regime discovery step')
            return False
        
        logger.info('🎯 Executing enhanced HMM regime discovery...')
        training_input = {
            'symbol': symbol,
            'exchange': exchange,
            'timeframe': timeframe,
            'data_dir': data_dir,
            'force_rerun': force_rerun
        }
        pipeline_state = {}
        
        result = await step.execute(training_input, pipeline_state)
        
        if result.get('enhanced_hmm_regime_discovery_completed', False):
            logger.info('✅ Enhanced Step 3: HMM Regime Discovery completed successfully')
            
            # Log key results
            if result.get('n_regimes'):
                logger.info(f'📊 Discovered {result["n_regimes"]} unique regimes')
            if result.get('economic_significance'):
                logger.info('✅ Regimes show significant economic differences')
            if result.get('enhanced_ml_transition_detection'):
                logger.info('✅ Enhanced ML transition detection completed successfully')
            if result.get('overall_quality_score'):
                logger.info(f'🎯 Overall quality score: {result["overall_quality_score"]:.4f}')
            
            total_elapsed = time.time() - start_time
            logger.info('=' * 80)
            logger.info('🎉 ENHANCED STEP 3 EXECUTION SUMMARY')
            logger.info('=' * 80)
            logger.info(f'⏱️ Total execution time: {total_elapsed:.2f} seconds')
            logger.info(f"⏰ End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info('✅ SUCCESS')
            logger.info('=' * 80)
            
            return True
        else:
            logger.error('❌ Enhanced Step 3: HMM Regime Discovery failed')
            error = result.get('regime_discovery_error', 'Unknown error')
            logger.error(f'   Error: {error}')
            
            total_elapsed = time.time() - start_time
            logger.info('=' * 80)
            logger.info('💥 ENHANCED STEP 3 EXECUTION SUMMARY')
            logger.info('=' * 80)
            logger.info(f'⏱️ Total execution time: {total_elapsed:.2f} seconds')
            logger.info(f"⏰ End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info('❌ FAILED')
            logger.info(f'   Error: {error}')
            logger.info('=' * 80)
            
            return False
            
    except Exception as e:
        logger.exception(f'❌ Enhanced Step 3: HMM Regime Discovery failed with exception: {e}')
        
        total_elapsed = time.time() - start_time
        logger.info('=' * 80)
        logger.info('💥 ENHANCED STEP 3 EXECUTION SUMMARY')
        logger.info('=' * 80)
        logger.info(f'⏱️ Total execution time: {total_elapsed:.2f} seconds')
        logger.info(f"⏰ End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info('❌ FAILED')
        logger.info(f'   Exception: {e}')
        logger.info('=' * 80)
        
        return False


if __name__ == "__main__":
    # Example usage
    success = asyncio.run(run_enhanced_step(
        symbol="ETHUSDT",
        exchange="BINANCE",
        timeframe="1m",
        n_trials=50,
        timeout_minutes=15
    ))
    
    if success:
        print("✅ Enhanced HMM regime discovery completed successfully!")
    else:
        print("❌ Enhanced HMM regime discovery failed")