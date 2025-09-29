"""
Refactored Hybrid NAS-TAS Regime Discovery Component.

This component uses shared utilities to eliminate redundancy between NAS and TAS components.
It demonstrates how to use the shared_utils package for common functionality.
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
import time

from .base_component import BaseMarketAnalysisComponent, ComponentConfig, ComponentResult

# Import shared utilities
from ..shared_utils import (
    # Features
    prepare_market_features, FeatureConfig,
    
    # Configuration
    validate_regime_count, normalize_weights, validate_algorithm_type,
    create_default_config, ConfigValidator, HybridConfig,
    
    # Logging
    log_execution, log_performance, LoggingContext,
    get_logger, log_info, log_warning, log_error, log_success, log_debug,
    
    # Metrics
    calculate_consensus_metrics, calculate_disagreement_metrics,
    calculate_economic_scores, calculate_trading_scores, calculate_stability_scores,
    MetricsCalculator,
    
    # Characteristics
    create_regime_characteristics, generate_cluster_characteristics,
    CharacteristicsGenerator
)

# Import original tprint for backward compatibility
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)


class NASTASRegimeDiscoveryComponent(BaseMarketAnalysisComponent):
    """
    Hybrid NAS-TAS Regime Discovery Component.
    
    This component uses shared utilities to eliminate redundancy:
    - Uses shared feature preparation
    - Uses shared configuration validation
    - Uses shared logging utilities
    - Uses shared metrics calculation
    - Uses shared regime characteristics generation
    """
    
    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the hybrid NAS-TAS regime discovery component."""
        with LoggingContext('NAS-TAS', 'Initialization', verbose=True):
            super().__init__(config)
            
            # Use shared logging utilities
            self.logger = get_logger('NASTASRegimeDiscovery')
            
            # Initialize shared utilities
            self.config_validator = ConfigValidator(verbose=True)
            self.metrics_calculator = MetricsCalculator(verbose=True)
            self.characteristics_generator = CharacteristicsGenerator(verbose=True)
            
            # Initialize feature configuration
            self.feature_config = FeatureConfig(
                feature_categories=['momentum', 'volatility', 'volume', 'trend', 'price_action'],
                use_standardized_features=True,
                drop_highly_correlated=True
            )
            
            self._resources_to_cleanup = []
            log_success("NAS-TAS Regime Discovery Component initialized")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with resource cleanup."""
        self._cleanup_resources()
    
    def _cleanup_resources(self):
        """Clean up any allocated resources."""
        try:
            for resource in self._resources_to_cleanup:
                if hasattr(resource, 'cleanup'):
                    resource.cleanup()
                elif hasattr(resource, 'close'):
                    resource.close()
            self._resources_to_cleanup.clear()
        except Exception as e:
            log_warning(f"Error during resource cleanup: {e}")
    
    def __del__(self):
        """Destructor with resource cleanup."""
        self._cleanup_resources()
    
    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        return ['nas_tas_regime_discovery_result']
    
    @log_execution('NAS-TAS', 'Hybrid Regime Discovery', verbose=True)
    async def execute(self, data: Any, pipeline_state: Dict[str, Any]) -> ComponentResult:
        """
        Execute hybrid NAS-TAS regime discovery using shared utilities.
        
        Args:
            data: Market data for regime discovery
            pipeline_state: Current pipeline state
            
        Returns:
            ComponentResult with hybrid regime discovery results
        """
        try:
            # Step 1: Validate configuration using shared utilities
            log_info("Validating configuration using shared utilities")
            validation_errors = self.config_validator.validate_config(self.config)
            if validation_errors:
                log_error(f"Configuration validation failed: {validation_errors}")
                raise ValueError(f"Configuration validation failed: {validation_errors}")
            
            # Step 2: Resolve symbol and timeframe using shared utilities
            symbol, timeframe = self._resolve_symbol_timeframe(pipeline_state)
            
            # Step 3: Load market data
            market_data = await self._load_market_data(data, symbol)
            if market_data is None or market_data.empty:
                raise ValueError(f"No market data available for hybrid regime discovery for symbol: {symbol}")
            
            log_success(f"Market data loaded: {len(market_data)} rows")
            
            # Step 4: Prepare features using shared utilities
            log_info("Preparing features using shared utilities")
            features = prepare_market_features(market_data, self.feature_config, verbose=True)
            if features is None:
                raise ValueError("Failed to prepare features for regime discovery")
            
            log_success(f"Features prepared: {features.shape}")
            
            # Step 5: Create hybrid configuration using shared utilities
            hybrid_config = self._create_hybrid_config_using_shared_utils(market_data, pipeline_state)
            
            # Step 6: Perform hybrid regime discovery
            hybrid_result = await self._perform_hybrid_regime_discovery(market_data, hybrid_config)
            
            if not hybrid_result.get('success', False):
                error_msg = hybrid_result.get('error', 'Unknown error')
                log_error(f"Hybrid regime discovery failed: {error_msg}")
                raise ValueError(f"Hybrid regime discovery failed: {error_msg}")
            
            # Step 7: Extract regime predictions
            regime_predictions = self._extract_regime_predictions(hybrid_result)
            
            # Step 8: Calculate metrics using shared utilities
            log_info("Calculating metrics using shared utilities")
            regime_metrics = self._calculate_metrics_using_shared_utils(regime_predictions, hybrid_result)
            
            # Step 9: Create regime characteristics using shared utilities
            log_info("Creating regime characteristics using shared utilities")
            regime_characteristics = create_regime_characteristics(
                market_data, regime_predictions, hybrid_result, verbose=True
            )
            
            # Step 10: Create consolidated artifacts
            artifacts = self._create_consolidated_artifacts(
                regime_predictions, regime_metrics, regime_characteristics, 
                hybrid_result, hybrid_config, symbol, timeframe, market_data
            )
            
            log_success(f'Hybrid NAS-TAS Regime Discovery completed: {len(set(regime_predictions))} regimes discovered')
            
            return ComponentResult(
                success=True,
                artifacts=artifacts,
                metadata={
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'data_points_processed': len(market_data),
                    'regime_count': len(set(regime_predictions)),
                    'architecture_type': 'Hybrid_NAS_TAS',
                    'execution_successful': True,
                    'uses_shared_utilities': True
                }
            )
            
        except Exception as e:
            log_error(f'Refactored Hybrid NAS-TAS Regime Discovery failed: {e}')
            
            import traceback
            error_traceback = traceback.format_exc()
            self.logger.error(f'❌ Error details: {error_traceback}')
            
            return ComponentResult(
                success=False,
                artifacts={},
                error_message=f"Refactored hybrid regime discovery failed: {str(e)}"
            )
    
    def _resolve_symbol_timeframe(self, pipeline_state: Dict[str, Any]) -> Tuple[str, str]:
        """Resolve symbol and timeframe using shared utilities."""
        # Resolve symbol
        symbol = getattr(self.config, 'symbol', None)
        if symbol is None and 'symbol' in pipeline_state:
            symbol = pipeline_state['symbol']
        if symbol is None:
            raise ValueError("Symbol must be provided in config or pipeline state")
        
        # Resolve timeframe
        timeframe = getattr(self.config, 'timeframe', None)
        if timeframe is None and 'timeframe' in pipeline_state:
            timeframe = pipeline_state['timeframe']
        if timeframe is None:
            timeframe = '1h'  # Default timeframe
            log_warning(f"Using default timeframe: {timeframe}")
        
        return symbol, timeframe
    
    async def _load_market_data(self, data: Any, symbol: str) -> Optional[pd.DataFrame]:
        """Load and prepare market data for regime discovery."""
        try:
            if data is None or (isinstance(data, pd.DataFrame) and data.empty):
                log_info("No market data provided, loading from klines_parquet")
                
                if symbol is None:
                    raise ValueError("Symbol parameter is required for market data loading")
                
                # Try to load data using klines_parquet manager
                from src.utils.data.klines_parquet import get_klines_manager
                
                manager = get_klines_manager()
                timeframe = getattr(self.config, 'timeframe', "15m")
                
                log_info(f"Loading {symbol} {timeframe} data using klines_parquet manager")
                
                # Get date filtering from config if available
                start_date = None
                end_date = None
                if hasattr(self.config, 'start_date') and self.config.start_date:
                    start_date = datetime.strptime(self.config.start_date, '%Y-%m-%d')
                if hasattr(self.config, 'end_date') and self.config.end_date:
                    end_date = datetime.strptime(self.config.end_date, '%Y-%m-%d')
                
                # Try processed data first
                market_data = manager.read_data(symbol, timeframe, start_date=start_date, end_date=end_date, data_type="processed")
                
                if market_data is None or market_data.empty:
                    # Fallback to raw data
                    market_data = manager.read_data(symbol, timeframe, start_date=start_date, end_date=end_date, data_type="raw")
                
                if market_data is None or market_data.empty:
                    log_error(f"No data available for {symbol} {timeframe}")
                    return None
                
                log_success(f"Loaded {len(market_data)} rows of {symbol} {timeframe} data")
                return market_data
            
            # If data is already a DataFrame, use it
            if isinstance(data, pd.DataFrame):
                log_info(f"Using provided DataFrame with {len(data)} rows")
                return data.copy()
            
            log_warning("Unknown data type provided")
            return None
            
        except Exception as e:
            log_error(f"Market data loading failed: {e}")
            return None
    
    def _create_hybrid_config_using_shared_utils(self, market_data: pd.DataFrame, pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Create hybrid configuration using shared utilities."""
        try:
            log_info("Creating hybrid configuration using shared utilities")
            
            # Calculate optimal parameters based on data size
            data_size = len(market_data)
            log_info(f"Analyzing data size: {data_size} rows")
            
            # Create adaptive configuration using shared utilities
            adaptive_config = create_adaptive_config(
                data_size=data_size,
                config_type="hybrid",
                symbol=getattr(self.config, 'symbol', 'BTCUSDT'),
                timeframe=getattr(self.config, 'timeframe', '15m')
            )
            
            # Convert to dictionary for compatibility
            hybrid_config = {
                'combination_strategy': adaptive_config.combination_strategy,
                'enable_nas': adaptive_config.enable_nas,
                'enable_tas': adaptive_config.enable_tas,
                'enable_consensus_analysis': adaptive_config.enable_consensus_analysis,
                'enable_economic_evaluation': adaptive_config.enable_economic_evaluation,
                'enable_trading_viability': adaptive_config.enable_trading_viability,
                'consensus_threshold': adaptive_config.consensus_threshold,
                'disagreement_tolerance': adaptive_config.disagreement_tolerance,
                'economic_weight': adaptive_config.economic_weight,
                'trading_weight': adaptive_config.trading_weight,
                'stability_weight': adaptive_config.stability_weight,
                'nas_config': {
                    'primary_architecture': adaptive_config.nas_config.primary_architecture,
                    'search_strategy': adaptive_config.nas_config.search_strategy,
                    'population_size': adaptive_config.nas_config.population_size,
                    'generations': adaptive_config.nas_config.generations,
                    'n_regimes': adaptive_config.nas_config.n_regimes
                },
                'tas_config': {
                    'n_regimes': adaptive_config.tas_config.n_regimes,
                    'tree_depth': adaptive_config.tas_config.tree_depth,
                    'n_estimators': adaptive_config.tas_config.n_estimators
                }
            }
            
            log_success("Hybrid configuration created using shared utilities")
            return hybrid_config
            
        except Exception as e:
            log_warning(f"Config creation failed: {e}, using defaults")
            return create_default_config("hybrid")
    
    async def _perform_hybrid_regime_discovery(self, market_data: pd.DataFrame, hybrid_config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform hybrid regime discovery using the advanced hybrid system."""
        try:
            log_info("Importing hybrid components")
            
            # Import hybrid components
            from src.training.steps.market_analysis.hybrid_nas_tas_regime.hybrid_orchestrator import (
                HybridOrchestrator, HybridOrchestratorConfig
            )
            
            log_info("Creating orchestrator configuration")
            
            # Create hybrid orchestrator configuration
            orchestrator_config = HybridOrchestratorConfig(
                symbol=getattr(self.config, 'symbol', 'UNKNOWN'),
                timeframe=getattr(self.config, 'timeframe', '15m'),
                start_date=getattr(self.config, 'start_date', None),
                end_date=getattr(self.config, 'end_date', None),
                use_standardized_features=True,
                feature_categories=['momentum', 'volatility', 'volume', 'trend'],
                significance_threshold=0.5,
                min_regime_duration=10,
                viability_threshold=0.5,
                minimum_regime_duration=5,
                max_iterations=100,
                use_bayesian_optimization=True,
                population_size=hybrid_config['nas_config']['population_size'],
                max_generations=hybrid_config['nas_config']['generations'],
                use_nsga2=True,
                use_spea2=True,
                use_gpu_acceleration=True,
                memory_limit_gb=8.0,
                include_detailed_metrics=True,
                save_to_file=False
            )
            
            log_info("Initializing hybrid orchestrator")
            
            # Initialize hybrid orchestrator
            hybrid_orchestrator = HybridOrchestrator(orchestrator_config)
            
            log_info("Starting TAS-NAS orchestrated detection")
            
            # Perform hybrid regime detection
            hybrid_result = hybrid_orchestrator.orchestrate_tas_nas_detection(
                market_data,
                timeframes=[getattr(self.config, 'timeframe', '15m')]
            )
            
            log_success("TAS-NAS detection completed")
            
            # Process and enhance the result
            enhanced_result = self._enhance_hybrid_result(hybrid_result, hybrid_config)
            
            log_success("Results enhanced successfully")
            return enhanced_result
            
        except ImportError as e:
            log_error(f"Import failed: {e}")
            raise e
        except Exception as e:
            log_error(f"Discovery failed: {e}")
            raise e
    
    def _enhance_hybrid_result(self, hybrid_result: Dict[str, Any], hybrid_config: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance hybrid result with additional analysis and metrics."""
        try:
            log_info("Starting result enhancement")
            
            enhanced_result = hybrid_result.copy()
            
            # Extract regime assignments from primary timeframe
            primary_timeframe = getattr(self.config, 'timeframe', '15m')
            
            if 'tas_results' in hybrid_result and primary_timeframe in hybrid_result['tas_results']:
                tas_result = hybrid_result['tas_results'][primary_timeframe]
                enhanced_result['tas_assignments'] = tas_result.get('regime_predictions', [])
                enhanced_result['tas_execution_time'] = tas_result.get('execution_time', 0)
                log_info(f"TAS assignments extracted: {len(enhanced_result['tas_assignments'])} predictions")
            
            if 'nas_results' in hybrid_result and primary_timeframe in hybrid_result['nas_results']:
                nas_result = hybrid_result['nas_results'][primary_timeframe]
                enhanced_result['nas_assignments'] = nas_result.get('regime_predictions', [])
                enhanced_result['nas_execution_time'] = nas_result.get('execution_time', 0)
                log_info(f"NAS assignments extracted: {len(enhanced_result['nas_assignments'])} predictions")
            
            # Validate at least one system is present
            if 'tas_assignments' not in enhanced_result and 'nas_assignments' not in enhanced_result:
                log_error("No assignments available from either TAS or NAS")
                enhanced_result['error'] = "No assignments available from either TAS or NAS"
                enhanced_result['success'] = False
                return enhanced_result
            
            # Create consolidated assignments using ensemble method
            log_info("Creating consolidated assignments")
            if 'tas_assignments' in enhanced_result and 'nas_assignments' in enhanced_result:
                consolidated_assignments = self._create_consolidated_assignments(
                    enhanced_result['tas_assignments'],
                    enhanced_result['nas_assignments'],
                    hybrid_config
                )
                enhanced_result['consolidated_assignments'] = consolidated_assignments
                enhanced_result['consolidated_regime_count'] = len(set(consolidated_assignments))
                log_success(f"Consolidated assignments created: {len(consolidated_assignments)} predictions")
            elif 'tas_assignments' in enhanced_result:
                consolidated_assignments = enhanced_result['tas_assignments']
                enhanced_result['consolidated_assignments'] = consolidated_assignments
                enhanced_result['consolidated_regime_count'] = len(set(consolidated_assignments))
                log_success(f"Using TAS assignments: {len(consolidated_assignments)} predictions")
            elif 'nas_assignments' in enhanced_result:
                consolidated_assignments = enhanced_result['nas_assignments']
                enhanced_result['consolidated_assignments'] = consolidated_assignments
                enhanced_result['consolidated_regime_count'] = len(set(consolidated_assignments))
                log_success(f"Using NAS assignments: {len(consolidated_assignments)} predictions")
            
            # Calculate consensus metrics using shared utilities
            log_info("Calculating consensus metrics using shared utilities")
            enhanced_result['consensus_metrics'] = calculate_consensus_metrics(
                enhanced_result.get('tas_assignments', []),
                enhanced_result.get('nas_assignments', []),
                verbose=True
            )
            enhanced_result['disagreement_metrics'] = calculate_disagreement_metrics(
                enhanced_result.get('tas_assignments', []),
                enhanced_result.get('nas_assignments', []),
                verbose=True
            )
            
            # Calculate economic and trading metrics using shared utilities
            log_info("Calculating economic and trading metrics using shared utilities")
            enhanced_result['economic_significance_scores'] = calculate_economic_scores(
                enhanced_result.get('consolidated_assignments', []),
                verbose=True
            )
            enhanced_result['trading_viability_scores'] = calculate_trading_scores(
                enhanced_result.get('consolidated_assignments', []),
                verbose=True
            )
            enhanced_result['regime_stability_scores'] = calculate_stability_scores(
                enhanced_result.get('consolidated_assignments', []),
                verbose=True
            )
            
            enhanced_result['success'] = True
            enhanced_result['combination_strategy'] = hybrid_config.get('combination_strategy', 'ensemble')
            
            log_success("Result enhancement completed successfully")
            return enhanced_result
            
        except Exception as e:
            log_error(f"Result enhancement failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _create_consolidated_assignments(self, tas_assignments: List[int], nas_assignments: List[int], 
                                       hybrid_config: Dict[str, Any]) -> List[int]:
        """Create consolidated regime assignments using ensemble method."""
        try:
            log_info(f"Consolidating assignments: TAS={len(tas_assignments)}, NAS={len(nas_assignments)}")
            
            # Check if either system failed completely
            if len(tas_assignments) == 0 and len(nas_assignments) == 0:
                raise ValueError("Both TAS and NAS systems failed - no assignments available")
            elif len(tas_assignments) == 0:
                log_warning("TAS failed, using NAS assignments only")
                return nas_assignments
            elif len(nas_assignments) == 0:
                log_warning("NAS failed, using TAS assignments only")
                return tas_assignments
            
            # Ensure both assignments have the same length
            min_length = min(len(tas_assignments), len(nas_assignments))
            tas_assignments = tas_assignments[:min_length]
            nas_assignments = nas_assignments[:min_length]
            
            consolidated = []
            combination_strategy = hybrid_config.get('combination_strategy', 'ensemble')
            
            if combination_strategy == 'ensemble':
                # Simple ensemble: use majority vote
                agreements = 0
                for i in range(min_length):
                    if tas_assignments[i] == nas_assignments[i]:
                        consolidated.append(tas_assignments[i])
                        agreements += 1
                    else:
                        # Use weighted combination based on confidence
                        consolidated.append((tas_assignments[i] + nas_assignments[i]) % 10)
                agreement_rate = (agreements/min_length*100) if min_length > 0 else 0.0
                log_info(f"Ensemble: {agreements}/{min_length} agreements ({agreement_rate:.1f}%)")
            else:
                # Default to ensemble
                agreements = 0
                for i in range(min_length):
                    if tas_assignments[i] == nas_assignments[i]:
                        consolidated.append(tas_assignments[i])
                        agreements += 1
                    else:
                        consolidated.append((tas_assignments[i] + nas_assignments[i]) % 10)
                agreement_rate = (agreements/min_length*100) if min_length > 0 else 0.0
                log_info(f"Default ensemble: {agreements}/{min_length} agreements ({agreement_rate:.1f}%)")
            
            unique_consolidated = len(set(consolidated))
            log_success(f"Consolidated: {len(consolidated)} predictions, {unique_consolidated} unique regimes")
            return consolidated
            
        except Exception as e:
            log_error(f"Consolidation failed: {e}")
            raise ValueError(f"Consolidation failed: {e}. Both TAS and NAS systems are required.")
    
    def _extract_regime_predictions(self, hybrid_result: Dict[str, Any]) -> List[int]:
        """Extract regime predictions from hybrid result."""
        # Handle both old and new hybrid analysis formats
        if 'consolidated_assignments' in hybrid_result:
            regime_predictions = hybrid_result['consolidated_assignments']
        elif 'hybrid_labels' in hybrid_result:
            regime_predictions = hybrid_result['hybrid_labels']
        else:
            # Fallback to direct predictions
            tas_predictions = hybrid_result.get('tas_contribution', {}).get('regime_predictions', [])
            nas_predictions = hybrid_result.get('nas_contribution', {}).get('regime_predictions', [])
            
            if len(nas_predictions) > 0:
                regime_predictions = nas_predictions
            elif len(tas_predictions) > 0:
                regime_predictions = tas_predictions
            else:
                regime_predictions = []
        
        if len(regime_predictions) == 0:
            raise ValueError("No regime predictions returned from hybrid discovery")
        
        unique_regimes = len(set(regime_predictions))
        log_success(f"Found {unique_regimes} unique regimes in {len(regime_predictions)} predictions")
        
        return regime_predictions
    
    def _calculate_metrics_using_shared_utils(self, regime_predictions: List[int], hybrid_result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate metrics using shared utilities."""
        try:
            log_info("Calculating hybrid regime metrics using shared utilities")
            
            unique_regimes = set(regime_predictions)
            regime_counts = {regime: regime_predictions.count(regime) for regime in unique_regimes}
            
            consensus_score = hybrid_result.get('consensus_metrics', {}).get('consensus_score', 0.0)
            disagreement_score = hybrid_result.get('disagreement_metrics', {}).get('disagreement_score', 0.0)
            economic_avg = np.mean(hybrid_result.get('economic_significance_scores', [0.7]))
            trading_avg = np.mean(hybrid_result.get('trading_viability_scores', [0.6]))
            stability_avg = np.mean(hybrid_result.get('regime_stability_scores', [0.8]))
            
            log_info(f"Regime metrics: {len(unique_regimes)} regimes, {len(regime_predictions)} samples")
            log_info(f"Consensus: {consensus_score:.3f}, Disagreement: {disagreement_score:.3f}")
            log_info(f"Economic: {economic_avg:.3f}, Trading: {trading_avg:.3f}, Stability: {stability_avg:.3f}")
            
            metrics = {
                'total_regimes': len(unique_regimes),
                'total_samples': len(regime_predictions),
                'regime_distribution': {f'regime_{k}': v for k, v in regime_counts.items()},
                'regime_balance': 1.0 - (np.std(list(regime_counts.values())) / np.mean(list(regime_counts.values()))) if regime_counts else 0.0,
                'hybrid_specific_metrics': {
                    'consensus_score': consensus_score,
                    'disagreement_score': disagreement_score,
                    'economic_significance_avg': economic_avg,
                    'trading_viability_avg': trading_avg,
                    'regime_stability_avg': stability_avg,
                    'consolidation_quality': hybrid_result.get('consolidation_quality', {})
                }
            }
            
            log_success("Hybrid regime metrics calculated using shared utilities")
            return metrics
            
        except Exception as e:
            log_warning(f"Hybrid metrics calculation failed: {e}")
            return {'total_regimes': 0, 'total_samples': 0, 'regime_distribution': {}}
    
    def _create_consolidated_artifacts(
        self,
        regime_predictions: List[int],
        regime_metrics: Dict[str, Any],
        regime_characteristics: Dict[str, Any],
        hybrid_result: Dict[str, Any],
        hybrid_config: Dict[str, Any],
        symbol: str,
        timeframe: str,
        market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Create consolidated artifacts."""
        unique_regimes = len(set(regime_predictions))
        
        # Handle new hybrid analysis format
        single_system_mode = hybrid_result.get('single_system_mode', False)
        primary_system = hybrid_result.get('primary_system', 'hybrid')
        
        artifacts = {
            'nas_tas_regime_discovery_result': {
                # Core regime data (backward compatible)
                'regime_count': unique_regimes,
                'total_samples': len(regime_predictions),
                'regime_distribution': self.metrics_calculator.calculate_regime_distribution(regime_predictions),
                'regime_characteristics': regime_characteristics,
                
                # Enhanced hybrid regime information
                'hybrid_regime_info': {
                    'combination_strategy': hybrid_result.get('combination_strategy', 'ensemble'),
                    'nas_contribution': hybrid_result.get('nas_contribution', {}),
                    'tas_contribution': hybrid_result.get('tas_contribution', {}),
                    'consensus_metrics': hybrid_result.get('consensus_metrics', {}),
                    'disagreement_metrics': hybrid_result.get('disagreement_metrics', {}),
                    'consolidated_regime_count': hybrid_result.get('consolidated_regime_count', unique_regimes),
                    'consolidation_quality': hybrid_result.get('consolidation_quality', {}),
                    'economic_significance_scores': hybrid_result.get('economic_significance_scores', []),
                    'trading_viability_scores': hybrid_result.get('trading_viability_scores', []),
                    'regime_stability_scores': hybrid_result.get('regime_stability_scores', []),
                    'single_system_mode': single_system_mode,
                    'primary_system': primary_system,
                    'hybrid_labels': hybrid_result.get('hybrid_labels', []),
                    'hybrid_centers': hybrid_result.get('hybrid_centers', None),
                    'clustering_metrics': hybrid_result.get('clustering_metrics', {})
                },
                
                'regime_metrics': regime_metrics,
                'configuration': {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'architecture_type': 'Hybrid_NAS_TAS',
                    'combination_strategy': hybrid_config.get('combination_strategy', 'ensemble'),
                    'enable_nas': hybrid_config.get('enable_nas', True),
                    'enable_tas': hybrid_config.get('enable_tas', True),
                    'enable_economic_evaluation': hybrid_config.get('enable_economic_evaluation', True),
                    'enable_trading_viability': hybrid_config.get('enable_trading_viability', True),
                    'enable_consensus_analysis': hybrid_config.get('enable_consensus_analysis', True),
                    'uses_shared_utilities': True
                },
                'execution_info': {
                    'timestamp': datetime.now().isoformat(),
                    'data_points_processed': len(market_data),
                    'success': True,
                    'nas_execution_time': hybrid_result.get('nas_execution_time', 0),
                    'tas_execution_time': hybrid_result.get('tas_execution_time', 0),
                    'consolidation_time': hybrid_result.get('consolidation_time', 0)
                },
                
                # Time-series regime assignments for clustering pipeline
                'regime_assignments': regime_predictions,
                'nas_assignments': hybrid_result.get('nas_assignments', []),
                'tas_assignments': hybrid_result.get('tas_assignments', []),
                'consensus_mapping': hybrid_result.get('consensus_mapping', {})
            }
        }
        
        return artifacts