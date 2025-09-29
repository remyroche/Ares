"""
Hybrid NAS-TAS Regime Discovery Component.

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
    create_default_config, create_adaptive_config, ConfigValidator, HybridConfig,
    
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
        tprint("🚀 Initializing NAS-TAS Regime Discovery Component")
        with LoggingContext('NAS-TAS', 'Initialization', verbose=True):
            super().__init__(config)
            tprint("✅ Base component initialization completed")
            
            # Use shared logging utilities
            self.logger = get_logger('NASTASRegimeDiscovery')
            tprint("📝 Logger initialized")
            
            # Initialize shared utilities
            self.config_validator = ConfigValidator(verbose=True)
            self.metrics_calculator = MetricsCalculator(verbose=True)
            self.characteristics_generator = CharacteristicsGenerator(verbose=True)
            tprint("🔧 Shared utilities initialized (ConfigValidator, MetricsCalculator, CharacteristicsGenerator)")
            
            # Initialize feature configuration with all specified categories
            self.feature_config = FeatureConfig(
                feature_categories=[
                    'oscillator',      # RSI, MACD, Stochastic, etc.
                    'momentum',        # Price momentum indicators
                    'trend',           # Moving averages, trend lines
                    'price',           # Price action features
                    'volume',          # Volume-based indicators
                    'volatility',      # Volatility measures
                    'entropy',         # Market entropy and complexity
                    'returns'          # Price returns and derivatives
                ],
                use_standardized_features=True,
                drop_highly_correlated=True
            )
            tprint("⚙️ Feature configuration created with 8 categories")
            
            self._resources_to_cleanup = []
            log_success("NAS-TAS Regime Discovery Component initialized")
            tprint("🎯 NAS-TAS Regime Discovery Component initialization completed successfully")
    
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
        tprint("🎬 Starting NAS-TAS Hybrid Regime Discovery Execution")
        try:
            # Step 1: Resolve symbol and timeframe using shared utilities
            tprint("📍 Step 1: Resolving symbol and timeframe")
            symbol, timeframe = self._resolve_symbol_timeframe(pipeline_state)
            tprint(f"✅ Symbol: {symbol}, Timeframe: {timeframe}")
            
            # Step 2: Load market data
            tprint("📊 Step 2: Loading market data")
            market_data = await self._load_market_data(data, symbol)
            if market_data is None or market_data.empty:
                raise ValueError(f"No market data available for hybrid regime discovery for symbol: {symbol}")
            
            log_success(f"Market data loaded: {len(market_data)} rows")
            tprint(f"✅ Market data loaded: {len(market_data)} rows")
            
            # Step 3: Prepare features using shared utilities
            tprint("🔧 Step 3: Preparing features using shared utilities")
            log_info("Preparing features using shared utilities")
            features = prepare_market_features(market_data, self.feature_config, verbose=True)
            if features is None:
                raise ValueError("Failed to prepare features for regime discovery")
            
            log_success(f"Features prepared: {features.shape}")
            tprint(f"✅ Features prepared: {features.shape}")
            
            # Step 4: Create hybrid configuration using shared utilities
            tprint("⚙️ Step 4: Creating hybrid configuration")
            hybrid_config = self._create_hybrid_config_using_shared_utils(market_data, pipeline_state)
            tprint("✅ Hybrid configuration created")
            
            # Step 5: Validate the hybrid configuration using shared utilities
            tprint("🔍 Step 5: Validating hybrid configuration")
            log_info("Validating hybrid configuration using shared utilities")
            # Create a temporary config object for validation
            from ..shared_utils import HybridConfig
            temp_config = HybridConfig(
                symbol=symbol,
                timeframe=timeframe,
                n_regimes=hybrid_config.nas_config.n_regimes
            )
            validation_errors = self.config_validator.validate_config(temp_config)
            if validation_errors:
                log_error(f"Configuration validation failed: {validation_errors}")
                raise ValueError(f"Configuration validation failed: {validation_errors}")
            tprint("✅ Configuration validation passed")
            
            # Step 6: Perform regime discovery (TAS + NAS, no hybrid combination)
            tprint("🔬 Step 6: Performing regime discovery (TAS + NAS)")
            regime_result = await self._perform_regime_discovery(market_data, hybrid_config)
            
            if not regime_result.get('success', False):
                error_msg = regime_result.get('error', 'Unknown error')
                log_error(f"Regime discovery failed: {error_msg}")
                tprint(f"❌ Regime discovery failed: {error_msg}")
                raise ValueError(f"Regime discovery failed: {error_msg}")
            tprint("✅ Regime discovery completed successfully")
            
            # Step 7: Extract regime predictions (from both systems)
            tprint("📈 Step 7: Extracting regime predictions from both systems")
            regime_predictions = self._extract_regime_predictions(regime_result)
            tas_count = regime_predictions.get('tas_regime_count', 0)
            nas_count = regime_predictions.get('nas_regime_count', 0)
            tprint(f"✅ Extracted TAS: {tas_count} regimes, NAS: {nas_count} regimes")
            
            # Step 8: Calculate metrics using shared utilities
            tprint("📊 Step 8: Calculating metrics using shared utilities")
            log_info("Calculating metrics using shared utilities")
            regime_metrics = self._calculate_metrics_using_shared_utils(regime_predictions, regime_result)
            tprint("✅ Metrics calculation completed")
            
            # Step 9: Create regime characteristics using shared utilities
            tprint("🎯 Step 9: Creating regime characteristics")
            log_info("Creating regime characteristics using shared utilities")
            regime_characteristics = create_regime_characteristics(
                market_data, regime_predictions, regime_result, verbose=True
            )
            tprint("✅ Regime characteristics created")
            
            # Step 10: Create consolidated artifacts for clustering pipeline
            tprint("📦 Step 10: Creating consolidated artifacts for clustering pipeline")
            artifacts = self._create_consolidated_artifacts(
                regime_predictions, regime_metrics, regime_characteristics, 
                regime_result, hybrid_config, symbol, timeframe, market_data
            )
            tprint("✅ Consolidated artifacts created")
            
            total_regimes = tas_count + nas_count
            log_success(f'NAS-TAS Regime Discovery completed: TAS={tas_count} regimes, NAS={nas_count} regimes')
            tprint(f"🎉 NAS-TAS Regime Discovery completed successfully: TAS={tas_count} regimes, NAS={nas_count} regimes (Total diversity: {total_regimes})")
            
            return ComponentResult(
                success=True,
                artifacts=artifacts,
                metadata={
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'data_points_processed': len(market_data),
                    'tas_regime_count': tas_count,
                    'nas_regime_count': nas_count,
                    'combined_regime_diversity': total_regimes,
                    'architecture_type': 'NAS_TAS_Regime_Discovery',
                    'processing_strategy': 'both_systems_combined',
                    'execution_successful': True,
                    'uses_shared_utilities': True
                }
            )
            
        except Exception as e:
            tprint(f"❌ NAS-TAS Hybrid Regime Discovery failed: {e}")
            log_error(f'Refactored Hybrid NAS-TAS Regime Discovery failed: {e}')
            
            import traceback
            error_traceback = traceback.format_exc()
            self.logger.error(f'❌ Error details: {error_traceback}')
            tprint(f"🔍 Error traceback logged for debugging")
            
            return ComponentResult(
                success=False,
                artifacts={},
                error_message=f"Refactored hybrid regime discovery failed: {str(e)}"
            )
    
    def _resolve_symbol_timeframe(self, pipeline_state: Dict[str, Any]) -> Tuple[str, str]:
        """Resolve symbol and timeframe using shared utilities."""
        tprint("🔍 Resolving symbol and timeframe parameters")
        
        # Resolve symbol
        symbol = getattr(self.config, 'symbol', None)
        if symbol is None and 'symbol' in pipeline_state:
            symbol = pipeline_state['symbol']
        if symbol is None:
            raise ValueError("Symbol must be provided in config or pipeline state")
        tprint(f"📌 Symbol resolved: {symbol}")
        
        # Resolve timeframe
        timeframe = getattr(self.config, 'timeframe', None)
        if timeframe is None and 'timeframe' in pipeline_state:
            timeframe = pipeline_state['timeframe']
        if timeframe is None:
            timeframe = '1h'  # Default timeframe
            log_warning(f"Using default timeframe: {timeframe}")
            tprint(f"⚠️ Using default timeframe: {timeframe}")
        else:
            tprint(f"📌 Timeframe resolved: {timeframe}")
        
        return symbol, timeframe
    
    async def _load_market_data(self, data: Any, symbol: str) -> Optional[pd.DataFrame]:
        """Load and prepare market data for regime discovery."""
        tprint("📊 Starting market data loading process")
        try:
            if data is None or (isinstance(data, pd.DataFrame) and data.empty):
                tprint("📥 No market data provided, loading from klines_parquet")
                log_info("No market data provided, loading from klines_parquet")
                
                if symbol is None:
                    raise ValueError("Symbol parameter is required for market data loading")
                
                # Try to load data using klines_parquet manager
                from src.utils.data.klines_parquet import get_klines_manager
                
                manager = get_klines_manager()
                timeframe = getattr(self.config, 'timeframe', "15m")
                
                tprint(f"📊 Loading {symbol} {timeframe} data using klines_parquet manager")
                log_info(f"Loading {symbol} {timeframe} data using klines_parquet manager")
                
                # Get date filtering from config if available
                start_date = None
                end_date = None
                if hasattr(self.config, 'start_date') and self.config.start_date:
                    start_date = datetime.strptime(self.config.start_date, '%Y-%m-%d')
                    tprint(f"📅 Start date filter: {start_date}")
                if hasattr(self.config, 'end_date') and self.config.end_date:
                    end_date = datetime.strptime(self.config.end_date, '%Y-%m-%d')
                    tprint(f"📅 End date filter: {end_date}")
                
                # Try processed data first
                tprint("🔍 Attempting to load processed data first")
                market_data = manager.read_data(symbol, timeframe, start_date=start_date, end_date=end_date, data_type="processed")
                
                if market_data is None or market_data.empty:
                    # Fallback to raw data
                    tprint("⚠️ Processed data not available, falling back to raw data")
                    market_data = manager.read_data(symbol, timeframe, start_date=start_date, end_date=end_date, data_type="raw")
                
                if market_data is None or market_data.empty:
                    log_error(f"No data available for {symbol} {timeframe}")
                    tprint(f"❌ No data available for {symbol} {timeframe}")
                    return None
                
                log_success(f"Loaded {len(market_data)} rows of {symbol} {timeframe} data")
                tprint(f"✅ Loaded {len(market_data)} rows of {symbol} {timeframe} data")
                return market_data
            
            # If data is already a DataFrame, use it
            if isinstance(data, pd.DataFrame):
                tprint(f"📊 Using provided DataFrame with {len(data)} rows")
                log_info(f"Using provided DataFrame with {len(data)} rows")
                return data.copy()
            
            log_warning("Unknown data type provided")
            tprint("⚠️ Unknown data type provided")
            return None
            
        except Exception as e:
            log_error(f"Market data loading failed: {e}")
            tprint(f"❌ Market data loading failed: {e}")
            return None
    
    def _create_hybrid_config_using_shared_utils(self, market_data: pd.DataFrame, pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Create hybrid configuration using shared utilities."""
        tprint("⚙️ Creating hybrid configuration using shared utilities")
        try:
            log_info("Creating hybrid configuration using shared utilities")
            
            # Calculate optimal parameters based on data size
            data_size = len(market_data)
            tprint(f"📊 Analyzing data size: {data_size} rows")
            log_info(f"Analyzing data size: {data_size} rows")
            
            # Create adaptive configuration using shared utilities
            tprint("🔧 Creating adaptive configuration based on data size")
            adaptive_config = create_adaptive_config(
                data_size=data_size,
                config_type="hybrid",
                symbol=getattr(self.config, 'symbol', 'BTCUSDT'),
                timeframe=getattr(self.config, 'timeframe', '15m')
            )
            
            log_success("Hybrid configuration created using shared utilities")
            tprint("✅ Hybrid configuration created successfully")
            return adaptive_config
            
        except Exception as e:
            log_warning(f"Config creation failed: {e}, using defaults")
            tprint(f"⚠️ Config creation failed: {e}, using defaults")
            return create_default_config("hybrid")
    
    async def _perform_regime_discovery(self, market_data: pd.DataFrame, hybrid_config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform regime discovery using TAS and NAS systems with fast-fail logic."""
        tprint("🔬 Starting regime discovery process (TAS + NAS)")
        try:
            log_info("Importing regime discovery components")
            tprint("📦 Importing regime discovery components")
            
            # Import hybrid components
            from src.training.steps.market_analysis.hybrid_nas_tas_regime.hybrid_orchestrator import (
                HybridOrchestrator, HybridOrchestratorConfig
            )
            
            log_info("Creating orchestrator configuration")
            tprint("⚙️ Creating orchestrator configuration")
            
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
                population_size=hybrid_config.nas_config.population_size,
                max_generations=hybrid_config.nas_config.generations,
                use_nsga2=True,
                use_spea2=True,
                use_gpu_acceleration=True,
                memory_limit_gb=8.0,
                include_detailed_metrics=True,
                save_to_file=False
            )
            tprint("✅ Orchestrator configuration created")
            
            log_info("Initializing hybrid orchestrator")
            tprint("🚀 Initializing hybrid orchestrator")
            
            # Initialize hybrid orchestrator
            hybrid_orchestrator = HybridOrchestrator(orchestrator_config)
            tprint("✅ Hybrid orchestrator initialized")
            
            log_info("Starting TAS-NAS orchestrated detection")
            tprint("🎯 Starting TAS-NAS orchestrated detection")
            
            # Perform regime detection (TAS + NAS, no hybrid combination)
            regime_result = hybrid_orchestrator.orchestrate_tas_nas_detection(
                market_data,
                timeframes=[getattr(self.config, 'timeframe', '15m')]
            )
            
            log_success("TAS-NAS detection completed")
            tprint("✅ TAS-NAS detection completed")
            
            # Check for fast-fail conditions
            tprint("🔍 Checking for fast-fail conditions")
            primary_timeframe = getattr(self.config, 'timeframe', '15m')
            
            tas_success = (primary_timeframe in regime_result.get('tas_results', {}) and
                          regime_result['tas_results'][primary_timeframe].get('success', False))
            nas_success = (primary_timeframe in regime_result.get('nas_results', {}) and
                          regime_result['nas_results'][primary_timeframe].get('success', False))
            
            tprint(f"🔍 TAS Success: {tas_success}")
            tprint(f"🔍 NAS Success: {nas_success}")
            
            # Fast fail if either system fails
            if not tas_success or not nas_success:
                failed_systems = []
                if not tas_success:
                    failed_systems.append("TAS")
                if not nas_success:
                    failed_systems.append("NAS")
                
                error_msg = f"Fast fail: {' and '.join(failed_systems)} system(s) failed. Both TAS and NAS must succeed."
                log_error(error_msg)
                tprint(f"❌ {error_msg}")
                return {
                    'success': False,
                    'error': error_msg,
                    'fast_fail': True,
                    'failed_systems': failed_systems,
                    'tas_success': tas_success,
                    'nas_success': nas_success
                }
            
            # Process results (no hybrid combination - use best performing system)
            tprint("🔧 Processing results without hybrid combination")
            processed_result = self._process_regime_results(regime_result, hybrid_config)
            
            log_success("Results processed successfully")
            tprint("✅ Results processed successfully")
            return processed_result
            
        except ImportError as e:
            log_error(f"Import failed: {e}")
            tprint(f"❌ Import failed: {e}")
            raise e
        except Exception as e:
            log_error(f"Discovery failed: {e}")
            tprint(f"❌ Discovery failed: {e}")
            raise e
    
    def _process_regime_results(self, regime_result: Dict[str, Any], hybrid_config) -> Dict[str, Any]:
        """Process regime results without hybrid combination, with quality assessment."""
        tprint("🔧 Processing regime results with quality assessment")
        try:
            log_info("Processing regime results")
            
            processed_result = regime_result.copy()
            
            # Extract regime assignments from primary timeframe
            primary_timeframe = getattr(self.config, 'timeframe', '15m')
            tprint(f"📊 Extracting regime assignments for timeframe: {primary_timeframe}")
            
            if 'tas_results' in regime_result and primary_timeframe in regime_result['tas_results']:
                tas_result = regime_result['tas_results'][primary_timeframe]
                processed_result['tas_assignments'] = tas_result.get('regime_predictions', [])
                processed_result['tas_execution_time'] = tas_result.get('execution_time', 0)
                processed_result['tas_quality_metrics'] = tas_result.get('quality_metrics', {})
                log_info(f"TAS assignments extracted: {len(processed_result['tas_assignments'])} predictions")
                tprint(f"✅ TAS assignments extracted: {len(processed_result['tas_assignments'])} predictions")
            
            if 'nas_results' in regime_result and primary_timeframe in regime_result['nas_results']:
                nas_result = regime_result['nas_results'][primary_timeframe]
                processed_result['nas_assignments'] = nas_result.get('regime_predictions', [])
                processed_result['nas_execution_time'] = nas_result.get('execution_time', 0)
                processed_result['nas_quality_metrics'] = nas_result.get('quality_metrics', {})
                log_info(f"NAS assignments extracted: {len(processed_result['nas_assignments'])} predictions")
                tprint(f"✅ NAS assignments extracted: {len(processed_result['nas_assignments'])} predictions")
            
            # Use both TAS and NAS together for clustering
            tprint("🤝 Using both TAS and NAS systems together for clustering")
            
            # Ensure both systems have assignments
            tas_assignments = processed_result.get('tas_assignments', [])
            nas_assignments = processed_result.get('nas_assignments', [])
            
            if len(tas_assignments) == 0 or len(nas_assignments) == 0:
                raise ValueError("Both TAS and NAS assignments are required for clustering")
            
            # Create combined assignments structure for clustering
            combined_assignments = {
                'tas_assignments': tas_assignments,
                'nas_assignments': nas_assignments,
                'tas_regime_count': len(set(tas_assignments)),
                'nas_regime_count': len(set(nas_assignments))
            }
            
            processed_result['combined_assignments'] = combined_assignments
            processed_result['processing_strategy'] = 'both_systems_combined'
            
            tprint(f"✅ Combined assignments: TAS={len(tas_assignments)} ({len(set(tas_assignments))} regimes), NAS={len(nas_assignments)} ({len(set(nas_assignments))} regimes)")
            
            # Calculate comprehensive quality assessment
            log_info("Calculating comprehensive quality assessment")
            tprint("📊 Calculating comprehensive quality assessment")
            quality_assessment = self._calculate_quality_assessment(processed_result)
            processed_result['quality_assessment'] = quality_assessment
            tprint("✅ Quality assessment completed")
            
            # Calculate consensus metrics for validation
            log_info("Calculating consensus metrics for validation")
            tprint("🤝 Calculating consensus metrics for validation")
            processed_result['consensus_metrics'] = calculate_consensus_metrics(
                processed_result.get('tas_assignments', []),
                processed_result.get('nas_assignments', []),
                verbose=True
            )
            processed_result['disagreement_metrics'] = calculate_disagreement_metrics(
                processed_result.get('tas_assignments', []),
                processed_result.get('nas_assignments', []),
                verbose=True
            )
            tprint("✅ Consensus and disagreement metrics calculated")
            
            # Calculate economic and trading metrics (using combined data)
            log_info("Calculating economic and trading metrics")
            tprint("💰 Calculating economic and trading metrics")
            processed_result['economic_significance_scores'] = calculate_economic_scores(
                tas_assignments,  # Use TAS assignments as primary for metrics
                verbose=True
            )
            processed_result['trading_viability_scores'] = calculate_trading_scores(
                nas_assignments,  # Use NAS assignments as primary for metrics
                verbose=True
            )
            processed_result['regime_stability_scores'] = calculate_stability_scores(
                tas_assignments,  # Use TAS assignments for stability
                verbose=True
            )
            tprint("✅ Economic and trading metrics calculated")
            
            processed_result['success'] = True
            processed_result['processing_strategy'] = 'best_system_selection'
            
            log_success("Result processing completed successfully")
            tprint("🎉 Result processing completed successfully")
            return processed_result
            
        except Exception as e:
            log_error(f"Result processing failed: {e}")
            tprint(f"❌ Result processing failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _determine_best_system(self, processed_result: Dict[str, Any]) -> str:
        """Determine the best performing system based on quality metrics."""
        tprint("🎯 Determining best performing system based on quality metrics")
        
        tas_quality = processed_result.get('tas_quality_metrics', {})
        nas_quality = processed_result.get('nas_quality_metrics', {})
        
        # Extract quality scores (default to 0.5 if not available)
        tas_score = tas_quality.get('overall_score', 0.5)
        nas_score = nas_quality.get('overall_score', 0.5)
        
        # Additional factors
        tas_regime_count = len(set(processed_result.get('tas_assignments', [])))
        nas_regime_count = len(set(processed_result.get('nas_assignments', [])))
        
        # Prefer systems with reasonable regime count (3-8 regimes)
        tas_regime_penalty = 0 if 3 <= tas_regime_count <= 8 else abs(tas_regime_count - 5) * 0.1
        nas_regime_penalty = 0 if 3 <= nas_regime_count <= 8 else abs(nas_regime_count - 5) * 0.1
        
        tas_adjusted_score = tas_score - tas_regime_penalty
        nas_adjusted_score = nas_score - nas_regime_penalty
        
        best_system = 'TAS' if tas_adjusted_score >= nas_adjusted_score else 'NAS'
        
        tprint(f"📊 TAS score: {tas_score:.3f} (adjusted: {tas_adjusted_score:.3f}), regimes: {tas_regime_count}")
        tprint(f"📊 NAS score: {nas_score:.3f} (adjusted: {nas_adjusted_score:.3f}), regimes: {nas_regime_count}")
        tprint(f"🏆 Selected best system: {best_system}")
        
        return best_system
    
    def _calculate_quality_assessment(self, processed_result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive quality assessment for both systems."""
        tprint("📊 Calculating comprehensive quality assessment for both systems")
        
        # Get both systems' assignments
        tas_assignments = processed_result.get('tas_assignments', [])
        nas_assignments = processed_result.get('nas_assignments', [])
        combined_assignments = processed_result.get('combined_assignments', {})
        
        # Basic quality metrics for both systems
        tas_regime_count = len(set(tas_assignments))
        nas_regime_count = len(set(nas_assignments))
        tas_total_samples = len(tas_assignments)
        nas_total_samples = len(nas_assignments)
        
        # Regime distribution analysis for both systems
        tas_regime_counts = [np.sum(tas_assignments == i) for i in range(tas_regime_count)] if tas_regime_count > 0 else []
        nas_regime_counts = [np.sum(nas_assignments == i) for i in range(nas_regime_count)] if nas_regime_count > 0 else []
        
        tas_regime_balance = 1.0 - (np.std(tas_regime_counts) / np.mean(tas_regime_counts)) if tas_regime_counts else 0.0
        nas_regime_balance = 1.0 - (np.std(nas_regime_counts) / np.mean(nas_regime_counts)) if nas_regime_counts else 0.0
        
        # System-specific quality metrics
        tas_quality = processed_result.get('tas_quality_metrics', {})
        nas_quality = processed_result.get('nas_quality_metrics', {})
        
        # Economic and trading metrics (calculated on combined data)
        economic_avg = np.mean(processed_result.get('economic_significance_scores', [0.5]))
        trading_avg = np.mean(processed_result.get('trading_viability_scores', [0.5]))
        stability_avg = np.mean(processed_result.get('regime_stability_scores', [0.5]))
        
        # Consensus metrics between systems
        consensus_score = processed_result.get('consensus_metrics', {}).get('consensus_score', 0.5)
        disagreement_score = processed_result.get('disagreement_metrics', {}).get('disagreement_score', 0.3)
        
        # Calculate combined quality score
        combined_regime_balance = (tas_regime_balance + nas_regime_balance) / 2
        overall_score = (
            economic_avg * 0.25 +
            trading_avg * 0.25 +
            stability_avg * 0.2 +
            combined_regime_balance * 0.15 +
            consensus_score * 0.15
        )
        
        quality_assessment = {
            'overall_score': overall_score,
            'processing_strategy': 'both_systems_combined',
            
            # TAS system metrics
            'tas_metrics': {
                'regime_count': tas_regime_count,
                'total_samples': tas_total_samples,
                'regime_balance': tas_regime_balance,
                'quality_metrics': tas_quality,
                'system_score': tas_quality.get('overall_score', 0.5)
            },
            
            # NAS system metrics
            'nas_metrics': {
                'regime_count': nas_regime_count,
                'total_samples': nas_total_samples,
                'regime_balance': nas_regime_balance,
                'quality_metrics': nas_quality,
                'system_score': nas_quality.get('overall_score', 0.5)
            },
            
            # Combined metrics
            'combined_metrics': {
                'total_regime_diversity': tas_regime_count + nas_regime_count,
                'average_regime_count': (tas_regime_count + nas_regime_count) / 2,
                'combined_regime_balance': combined_regime_balance,
                'economic_significance_avg': economic_avg,
                'trading_viability_avg': trading_avg,
                'regime_stability_avg': stability_avg,
                'consensus_score': consensus_score,
                'disagreement_score': disagreement_score
            },
            
            'quality_grade': self._calculate_quality_grade(overall_score),
            'recommendations': self._generate_combined_quality_recommendations(
                overall_score, tas_regime_count, nas_regime_count, consensus_score, combined_regime_balance
            )
        }
        
        tprint(f"📊 Overall quality score: {overall_score:.3f} (Grade: {quality_assessment['quality_grade']})")
        tprint(f"📊 TAS: {tas_regime_count} regimes, NAS: {nas_regime_count} regimes")
        tprint(f"📊 Combined balance: {combined_regime_balance:.3f}, Consensus: {consensus_score:.3f}")
        
        return quality_assessment
    
    def _calculate_quality_grade(self, score: float) -> str:
        """Calculate quality grade based on score."""
        if score >= 0.8:
            return 'A'
        elif score >= 0.7:
            return 'B'
        elif score >= 0.6:
            return 'C'
        elif score >= 0.5:
            return 'D'
        else:
            return 'F'
    
    def _generate_quality_recommendations(self, score: float, regime_count: int, regime_balance: float) -> List[str]:
        """Generate quality improvement recommendations."""
        recommendations = []
        
        if score < 0.6:
            recommendations.append("Consider adjusting feature engineering parameters")
        
        if regime_count < 3:
            recommendations.append("Increase regime diversity - consider adjusting clustering parameters")
        elif regime_count > 8:
            recommendations.append("Reduce regime complexity - consider increasing minimum regime duration")
        
        if regime_balance < 0.3:
            recommendations.append("Improve regime balance - consider rebalancing clustering weights")
        
        if not recommendations:
            recommendations.append("Quality metrics are satisfactory")
        
        return recommendations
    
    def _generate_combined_quality_recommendations(self, score: float, tas_regime_count: int, nas_regime_count: int, 
                                                 consensus_score: float, combined_regime_balance: float) -> List[str]:
        """Generate quality improvement recommendations for combined systems."""
        recommendations = []
        
        if score < 0.6:
            recommendations.append("Consider adjusting feature engineering parameters for both systems")
        
        # Regime count recommendations
        if tas_regime_count < 3 or nas_regime_count < 3:
            recommendations.append("Increase regime diversity in one or both systems - consider adjusting clustering parameters")
        elif tas_regime_count > 8 or nas_regime_count > 8:
            recommendations.append("Reduce regime complexity in one or both systems - consider increasing minimum regime duration")
        
        # Consensus recommendations
        if consensus_score < 0.3:
            recommendations.append("Low consensus between TAS and NAS - consider aligning system parameters")
        elif consensus_score > 0.8:
            recommendations.append("Very high consensus - systems may be too similar, consider diversifying approaches")
        
        # Balance recommendations
        if combined_regime_balance < 0.3:
            recommendations.append("Improve regime balance in both systems - consider rebalancing clustering weights")
        
        # System diversity recommendations
        regime_diff = abs(tas_regime_count - nas_regime_count)
        if regime_diff > 4:
            recommendations.append("Large difference in regime counts between systems - consider parameter alignment")
        
        if not recommendations:
            recommendations.append("Combined system quality metrics are satisfactory")
        
        return recommendations
    
    def _create_consolidated_assignments(self, tas_assignments: List[int], nas_assignments: List[int], 
                                       hybrid_config: Dict[str, Any]) -> List[int]:
        """Create consolidated regime assignments using ensemble method."""
        tprint("🔄 Starting assignment consolidation process")
        try:
            log_info(f"Consolidating assignments: TAS={len(tas_assignments)}, NAS={len(nas_assignments)}")
            tprint(f"📊 Consolidating assignments: TAS={len(tas_assignments)}, NAS={len(nas_assignments)}")
            
            # Check if either system failed completely
            if len(tas_assignments) == 0 and len(nas_assignments) == 0:
                raise ValueError("Both TAS and NAS systems failed - no assignments available")
            elif len(tas_assignments) == 0:
                log_warning("TAS failed, using NAS assignments only")
                tprint("⚠️ TAS failed, using NAS assignments only")
                return nas_assignments
            elif len(nas_assignments) == 0:
                log_warning("NAS failed, using TAS assignments only")
                tprint("⚠️ NAS failed, using TAS assignments only")
                return tas_assignments
            
            # Ensure both assignments have the same length
            min_length = min(len(tas_assignments), len(nas_assignments))
            tas_assignments = tas_assignments[:min_length]
            nas_assignments = nas_assignments[:min_length]
            tprint(f"📏 Normalized to minimum length: {min_length}")
            
            consolidated = []
            combination_strategy = getattr(hybrid_config, 'combination_strategy', 'ensemble')
            tprint(f"🎯 Using combination strategy: {combination_strategy}")
            
            if combination_strategy == 'ensemble':
                # Simple ensemble: use majority vote
                agreements = 0
                for i in range(min_length):
                    tas_val = int(tas_assignments[i]) if hasattr(tas_assignments[i], '__len__') and len(tas_assignments[i]) == 1 else tas_assignments[i]
                    nas_val = int(nas_assignments[i]) if hasattr(nas_assignments[i], '__len__') and len(nas_assignments[i]) == 1 else nas_assignments[i]
                    
                    if tas_val == nas_val:
                        consolidated.append(tas_val)
                        agreements += 1
                    else:
                        # Use weighted combination based on confidence
                        consolidated.append((tas_val + nas_val) % 10)
                agreement_rate = (agreements/min_length*100) if min_length > 0 else 0.0
                log_info(f"Ensemble: {agreements}/{min_length} agreements ({agreement_rate:.1f}%)")
                tprint(f"🤝 Ensemble: {agreements}/{min_length} agreements ({agreement_rate:.1f}%)")
            else:
                # Default to ensemble
                agreements = 0
                for i in range(min_length):
                    tas_val = int(tas_assignments[i]) if hasattr(tas_assignments[i], '__len__') and len(tas_assignments[i]) == 1 else tas_assignments[i]
                    nas_val = int(nas_assignments[i]) if hasattr(nas_assignments[i], '__len__') and len(nas_assignments[i]) == 1 else nas_assignments[i]
                    
                    if tas_val == nas_val:
                        consolidated.append(tas_val)
                        agreements += 1
                    else:
                        consolidated.append((tas_val + nas_val) % 10)
                agreement_rate = (agreements/min_length*100) if min_length > 0 else 0.0
                log_info(f"Default ensemble: {agreements}/{min_length} agreements ({agreement_rate:.1f}%)")
                tprint(f"🤝 Default ensemble: {agreements}/{min_length} agreements ({agreement_rate:.1f}%)")
            
            unique_consolidated = len(set(consolidated))
            log_success(f"Consolidated: {len(consolidated)} predictions, {unique_consolidated} unique regimes")
            tprint(f"✅ Consolidated: {len(consolidated)} predictions, {unique_consolidated} unique regimes")
            return consolidated
            
        except Exception as e:
            log_error(f"Consolidation failed: {e}")
            tprint(f"❌ Consolidation failed: {e}")
            raise ValueError(f"Consolidation failed: {e}. Both TAS and NAS systems are required.")
    
    def _extract_regime_predictions(self, regime_result: Dict[str, Any]) -> Dict[str, List[int]]:
        """Extract regime predictions from both TAS and NAS systems."""
        tprint("📈 Starting regime predictions extraction for both systems")
        
        # Extract both TAS and NAS assignments
        tas_assignments = regime_result.get('tas_assignments', [])
        nas_assignments = regime_result.get('nas_assignments', [])
        
        if len(tas_assignments) == 0 and len(nas_assignments) == 0:
            # Fallback to old format
            if 'consolidated_assignments' in regime_result:
                tas_assignments = regime_result['consolidated_assignments']
                nas_assignments = regime_result['consolidated_assignments']
                tprint("✅ Using consolidated assignments for both systems")
            elif 'hybrid_labels' in regime_result:
                tas_assignments = regime_result['hybrid_labels']
                nas_assignments = regime_result['hybrid_labels']
                tprint("✅ Using hybrid labels for both systems")
            else:
                # Final fallback
                tas_predictions = regime_result.get('tas_contribution', {}).get('regime_predictions', [])
                nas_predictions = regime_result.get('nas_contribution', {}).get('regime_predictions', [])
                
                if tas_predictions:
                    tas_assignments = tas_predictions
                    tprint("✅ Using TAS predictions (fallback)")
                if nas_predictions:
                    nas_assignments = nas_predictions
                    tprint("✅ Using NAS predictions (fallback)")
        
        if len(tas_assignments) == 0 or len(nas_assignments) == 0:
            raise ValueError("Both TAS and NAS regime predictions are required")
        
        tas_regimes = len(set(tas_assignments))
        nas_regimes = len(set(nas_assignments))
        
        regime_predictions = {
            'tas_assignments': tas_assignments,
            'nas_assignments': nas_assignments,
            'tas_regime_count': tas_regimes,
            'nas_regime_count': nas_regimes
        }
        
        log_success(f"Found TAS: {tas_regimes} regimes in {len(tas_assignments)} predictions, NAS: {nas_regimes} regimes in {len(nas_assignments)} predictions")
        tprint(f"✅ Found TAS: {tas_regimes} regimes in {len(tas_assignments)} predictions")
        tprint(f"✅ Found NAS: {nas_regimes} regimes in {len(nas_assignments)} predictions")
        
        return regime_predictions
    
    def _calculate_metrics_using_shared_utils(self, regime_predictions: Dict[str, List[int]], regime_result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate metrics using shared utilities for both systems."""
        tprint("📊 Starting combined regime metrics calculation")
        try:
            log_info("Calculating combined regime metrics using shared utilities")
            
            # Extract both systems' predictions
            tas_assignments = regime_predictions.get('tas_assignments', [])
            nas_assignments = regime_predictions.get('nas_assignments', [])
            
            tas_unique_regimes = set(tas_assignments)
            nas_unique_regimes = set(nas_assignments)
            
            tas_regime_counts = {regime: np.sum(tas_assignments == regime) for regime in tas_unique_regimes}
            nas_regime_counts = {regime: np.sum(nas_assignments == regime) for regime in nas_unique_regimes}
            
            tprint(f"📈 TAS: {len(tas_unique_regimes)} unique regimes from {len(tas_assignments)} samples")
            tprint(f"📈 NAS: {len(nas_unique_regimes)} unique regimes from {len(nas_assignments)} samples")
            
            # Calculate consensus metrics
            consensus_score = regime_result.get('consensus_metrics', {}).get('consensus_score', 0.0)
            disagreement_score = regime_result.get('disagreement_metrics', {}).get('disagreement_score', 0.0)
            economic_avg = np.mean(regime_result.get('economic_significance_scores', [0.7]))
            trading_avg = np.mean(regime_result.get('trading_viability_scores', [0.6]))
            stability_avg = np.mean(regime_result.get('regime_stability_scores', [0.8]))
            
            # Calculate regime balance for both systems
            tas_balance = 1.0 - (np.std(list(tas_regime_counts.values())) / np.mean(list(tas_regime_counts.values()))) if tas_regime_counts else 0.0
            nas_balance = 1.0 - (np.std(list(nas_regime_counts.values())) / np.mean(list(nas_regime_counts.values()))) if nas_regime_counts else 0.0
            
            log_info(f"Combined regime metrics: TAS={len(tas_unique_regimes)} regimes, NAS={len(nas_unique_regimes)} regimes")
            log_info(f"Consensus: {consensus_score:.3f}, Disagreement: {disagreement_score:.3f}")
            log_info(f"Economic: {economic_avg:.3f}, Trading: {trading_avg:.3f}, Stability: {stability_avg:.3f}")
            
            tprint(f"📊 Combined regime metrics: TAS={len(tas_unique_regimes)} regimes, NAS={len(nas_unique_regimes)} regimes")
            tprint(f"🤝 Consensus: {consensus_score:.3f}, Disagreement: {disagreement_score:.3f}")
            tprint(f"💰 Economic: {economic_avg:.3f}, Trading: {trading_avg:.3f}, Stability: {stability_avg:.3f}")
            
            metrics = {
                'tas_metrics': {
                    'total_regimes': len(tas_unique_regimes),
                    'total_samples': len(tas_assignments),
                    'regime_distribution': {f'regime_{k}': v for k, v in tas_regime_counts.items()},
                    'regime_balance': tas_balance
                },
                'nas_metrics': {
                    'total_regimes': len(nas_unique_regimes),
                    'total_samples': len(nas_assignments),
                    'regime_distribution': {f'regime_{k}': v for k, v in nas_regime_counts.items()},
                    'regime_balance': nas_balance
                },
                'combined_metrics': {
                    'total_regime_diversity': len(tas_unique_regimes) + len(nas_unique_regimes),
                    'average_regime_count': (len(tas_unique_regimes) + len(nas_unique_regimes)) / 2,
                    'consensus_score': consensus_score,
                    'disagreement_score': disagreement_score,
                    'economic_significance_avg': economic_avg,
                    'trading_viability_avg': trading_avg,
                    'regime_stability_avg': stability_avg,
                    'combined_regime_balance': (tas_balance + nas_balance) / 2
                }
            }
            
            log_success("Combined regime metrics calculated using shared utilities")
            tprint("✅ Combined regime metrics calculated successfully")
            return metrics
            
        except Exception as e:
            log_warning(f"Combined metrics calculation failed: {e}")
            tprint(f"❌ Combined metrics calculation failed: {e}")
            return {
                'tas_metrics': {'total_regimes': 0, 'total_samples': 0, 'regime_distribution': {}},
                'nas_metrics': {'total_regimes': 0, 'total_samples': 0, 'regime_distribution': {}},
                'combined_metrics': {}
            }
    
    def _create_consolidated_artifacts(
        self,
        regime_predictions: Dict[str, List[int]],
        regime_metrics: Dict[str, Any],
        regime_characteristics: Dict[str, Any],
        regime_result: Dict[str, Any],
        hybrid_config: Dict[str, Any],
        symbol: str,
        timeframe: str,
        market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Create consolidated artifacts for clustering pipeline with both systems."""
        tprint("📦 Starting consolidated artifacts creation for clustering pipeline")
        
        # Extract both systems' predictions
        tas_assignments = regime_predictions.get('tas_assignments', [])
        nas_assignments = regime_predictions.get('nas_assignments', [])
        tas_regime_count = regime_predictions.get('tas_regime_count', 0)
        nas_regime_count = regime_predictions.get('nas_regime_count', 0)
        
        tprint(f"📊 Creating artifacts for TAS: {tas_regime_count} regimes, NAS: {nas_regime_count} regimes")
        
        # Extract system information
        quality_assessment = regime_result.get('quality_assessment', {})
        processing_strategy = regime_result.get('processing_strategy', 'both_systems_combined')
        combined_assignments = regime_result.get('combined_assignments', {})
        
        tprint(f"🔧 Strategy: {processing_strategy}")
        
        tprint("🏗️ Building core regime data structure for clustering pipeline")
        artifacts = {
            'nas_tas_regime_discovery_result': {
                # Core regime data for both systems
                'tas_regime_count': tas_regime_count,
                'nas_regime_count': nas_regime_count,
                'total_tas_samples': len(tas_assignments),
                'total_nas_samples': len(nas_assignments),
                'combined_regime_diversity': tas_regime_count + nas_regime_count,
                
                # Regime assignments for both systems
                'tas_assignments': tas_assignments,
                'nas_assignments': nas_assignments,
                'combined_assignments': combined_assignments,
                
                # Regime characteristics for both systems
                'regime_characteristics': regime_characteristics,
                
                # Quality assessment for both systems
                'quality_assessment': quality_assessment,
                
                # System information
                'system_info': {
                    'processing_strategy': processing_strategy,
                    'tas_success': regime_result.get('tas_success', False),
                    'nas_success': regime_result.get('nas_success', False),
                    'fast_fail': regime_result.get('fast_fail', False),
                    'both_systems_available': len(tas_assignments) > 0 and len(nas_assignments) > 0
                },
                
                # Individual system results
                'tas_results': {
                    'assignments': tas_assignments,
                    'regime_count': tas_regime_count,
                    'execution_time': regime_result.get('tas_execution_time', 0),
                    'quality_metrics': regime_result.get('tas_quality_metrics', {}),
                    'success': regime_result.get('tas_success', False)
                },
                'nas_results': {
                    'assignments': nas_assignments,
                    'regime_count': nas_regime_count,
                    'execution_time': regime_result.get('nas_execution_time', 0),
                    'quality_metrics': regime_result.get('nas_quality_metrics', {}),
                    'success': regime_result.get('nas_success', False)
                },
                
                # Consensus and disagreement metrics
                'consensus_metrics': regime_result.get('consensus_metrics', {}),
                'disagreement_metrics': regime_result.get('disagreement_metrics', {}),
                
                # Economic and trading metrics
                'economic_significance_scores': regime_result.get('economic_significance_scores', []),
                'trading_viability_scores': regime_result.get('trading_viability_scores', []),
                'regime_stability_scores': regime_result.get('regime_stability_scores', []),
                
                # Combined metrics
                'regime_metrics': regime_metrics,
                
                'configuration': {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'architecture_type': 'NAS_TAS_Regime_Discovery',
                    'processing_strategy': processing_strategy,
                    'enable_nas': getattr(hybrid_config, 'enable_nas', True),
                    'enable_tas': getattr(hybrid_config, 'enable_tas', True),
                    'enable_economic_evaluation': getattr(hybrid_config, 'enable_economic_evaluation', True),
                    'enable_trading_viability': getattr(hybrid_config, 'enable_trading_viability', True),
                    'enable_consensus_analysis': getattr(hybrid_config, 'enable_consensus_analysis', True),
                    'uses_shared_utilities': True
                },
                'execution_info': {
                    'timestamp': datetime.now().isoformat(),
                    'data_points_processed': len(market_data),
                    'success': True,
                    'nas_execution_time': regime_result.get('nas_execution_time', 0),
                    'tas_execution_time': regime_result.get('tas_execution_time', 0),
                    'total_execution_time': regime_result.get('tas_execution_time', 0) + regime_result.get('nas_execution_time', 0)
                },
                
                # Additional data for clustering analysis
                'market_data_shape': market_data.shape,
                'feature_count': market_data.shape[1] if len(market_data.shape) > 1 else 0,
                'clustering_readiness': {
                    'has_tas_assignments': len(tas_assignments) > 0,
                    'has_nas_assignments': len(nas_assignments) > 0,
                    'has_quality_metrics': bool(quality_assessment),
                    'has_characteristics': bool(regime_characteristics),
                    'quality_grade': quality_assessment.get('quality_grade', 'F'),
                    'overall_score': quality_assessment.get('overall_score', 0.0),
                    'both_systems_ready': len(tas_assignments) > 0 and len(nas_assignments) > 0
                }
            }
        }
        
        # Save thorough output to outcomes/ directory
        tprint("💾 Saving thorough output to outcomes/ directory")
        self._save_thorough_outcome(artifacts, symbol, timeframe)
        
        tprint("✅ Consolidated artifacts created successfully for clustering pipeline")
        tprint(f"📋 Artifacts include: both systems' regime data, quality assessment, metrics, clustering readiness")
        tprint(f"🎯 Quality Grade: {quality_assessment.get('quality_grade', 'F')}, Score: {quality_assessment.get('overall_score', 0.0):.3f}")
        
        return artifacts
    
    def _save_thorough_outcome(self, artifacts: Dict[str, Any], symbol: str, timeframe: str) -> None:
        """Save thorough outcome to outcomes/ directory."""
        try:
            tprint("💾 Saving thorough outcome to outcomes/ directory")
            
            # Create outcomes directory if it doesn't exist
            outcomes_dir = Path("outcomes")
            outcomes_dir.mkdir(exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"market_analysis_nas_tas_regime_discovery_outcome_{timestamp}.json"
            filepath = outcomes_dir / filename
            
            # Prepare outcome data
            outcome_data = {
                'metadata': {
                    'component': 'nas_tas_regime_discovery',
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'timestamp': datetime.now().isoformat(),
                    'version': '2.0',
                    'processing_strategy': 'both_systems_combined'
                },
                'artifacts': artifacts,
                'summary': {
                    'tas_regime_count': artifacts['nas_tas_regime_discovery_result'].get('tas_regime_count', 0),
                    'nas_regime_count': artifacts['nas_tas_regime_discovery_result'].get('nas_regime_count', 0),
                    'combined_regime_diversity': artifacts['nas_tas_regime_discovery_result'].get('combined_regime_diversity', 0),
                    'quality_grade': artifacts['nas_tas_regime_discovery_result']['quality_assessment'].get('quality_grade', 'F'),
                    'overall_score': artifacts['nas_tas_regime_discovery_result']['quality_assessment'].get('overall_score', 0.0),
                    'both_systems_successful': artifacts['nas_tas_regime_discovery_result']['system_info'].get('both_systems_available', False),
                    'clustering_ready': artifacts['nas_tas_regime_discovery_result']['clustering_readiness'].get('both_systems_ready', False)
                }
            }
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(outcome_data, f, indent=2, default=str)
            
            tprint(f"✅ Thorough outcome saved to: {filepath}")
            log_success(f"Thorough outcome saved to: {filepath}")
            
        except Exception as e:
            log_warning(f"Failed to save thorough outcome: {e}")
            tprint(f"⚠️ Failed to save thorough outcome: {e}")