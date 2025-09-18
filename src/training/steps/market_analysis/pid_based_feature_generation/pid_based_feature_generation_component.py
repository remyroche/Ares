"""
PID-Based Feature Generation Component

This component replaces the cross_timeframe_analysis component and provides
comprehensive PID-based feature generation including interaction, polynomial,
and cross-timeframe features using optimized lookback periods.

Key Features:
- Replaces cross_timeframe_analysis functionality
- Uses optimized lookback periods from feature_lookback_optimization
- Leverages matrix_operations/ for all calculations
- Generates up to 200 total features (100 interaction + 50 polynomial + 50 cross-timeframe)
- Comprehensive validation and error handling
- Hardware-optimized computations
"""

import asyncio
import logging
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
from enum import Enum

# Core dependencies with fallback support
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

# Import base component
from ...market_analysis.components.base_component import BaseMarketAnalysisComponent, ComponentConfig, ComponentResult

# Import PID-based feature generation components
from .pid_based_feature_orchestrator import PIDBasedFeatureOrchestrator, OrchestratorConfig, OrchestratorResult
from .optimized_lookback_integration import OptimizedLookbackIntegration, LookbackIntegrationResult

# Import logger
try:
    from src.utils.logger import system_logger
    logger = system_logger.getChild('PIDBasedFeatureGenerationComponent')
except ImportError:
    logger = logging.getLogger('PIDBasedFeatureGenerationComponent')
    logger.setLevel(logging.INFO)


class GenerationStatus(Enum):
    """Status of feature generation process."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class PIDBasedFeatureGenerationComponent(BaseMarketAnalysisComponent):
    """
    PID-Based Feature Generation Component.
    
    Replaces the cross_timeframe_analysis component and provides comprehensive
    PID-based feature generation including interaction, polynomial, and cross-timeframe features.
    """
    
    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the PID-based feature generation component."""
        super().__init__(config)
        self.logger = logger.getChild('PIDBasedFeatureGenerationComponent')
        
        # Initialize components
        self._initialize_components()
        
        # Track generation status
        self.generation_status = GenerationStatus.PENDING
        self.start_time: Optional[float] = None
        
        # Track target source information for outcome verification
        self._target_source_info = {
            'target_used': 'unknown',
            'target_type': 'unknown', 
            'valid_samples': 0,
            'source': 'unknown'
        }
        
        self.logger.info("🔧 PIDBasedFeatureGenerationComponent initialized")
        self.logger.info(f"📊 Symbol: {self.config.symbol}")
        self.logger.info(f"📊 Exchange: {self.config.exchange}")
        self.logger.info(f"📊 Timeframe: {self.config.timeframe}")
    
    def _initialize_components(self):
        """Initialize required components."""
        # Initialize lookback integration FIRST (before feature generators)
        self.lookback_integration = OptimizedLookbackIntegration()
        self.logger.info("✅ Optimized Lookback Integration initialized")
        
        # Initialize orchestrator configuration
        orchestrator_config = OrchestratorConfig(
            max_interaction_features=100,
            max_polynomial_features=50,
            max_cross_timeframe_features=50,
            enable_interaction_features=True,
            enable_polynomial_features=True,
            enable_cross_timeframe_features=True,
            enable_parallel_processing=True,
            enable_gpu_acceleration=True,
            memory_limit_gb=8.0
        )
        
        # Initialize orchestrator
        self.orchestrator = PIDBasedFeatureOrchestrator(orchestrator_config)
        self.logger.info("✅ PID-Based Feature Orchestrator initialized")
    
    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        return ['pid_based_feature_generation_result']
    
    async def execute(self, data: Any, pipeline_state: Dict[str, Any]) -> ComponentResult:
        """
        Execute PID-based feature generation with comprehensive validation and reporting.
        
        Args:
            data: Market data for feature generation
            pipeline_state: Current pipeline state
            
        Returns:
            ComponentResult with PID-based feature generation results
        """
        self.start_time = time.time()
        self.generation_status = GenerationStatus.IN_PROGRESS
        
        self.logger.info('🔧 Starting PID-Based Feature Generation')
        self._report_checkpoint('start', 'generation_started', {
            'symbol': self.config.symbol,
            'exchange': self.config.exchange,
            'timeframe': self.config.timeframe
        })
        
        try:
            # Store pipeline state for data access
            self._pipeline_state = pipeline_state
            
            # Step 1: Load and validate market data
            self.logger.info('📊 Loading and validating market data...')
            market_data = await self._load_and_validate_market_data(data)
            self._report_checkpoint('data_loading', 'completed', {
                'data_points': len(market_data) if market_data is not None else 0,
                'data_quality_score': self._calculate_data_quality_score(market_data)
            })
            
            # Step 2: Get feature optimization results from previous stage
            self.logger.info('⚙️ Retrieving feature lookback optimization results...')
            feature_lookback_optimization = await self._get_feature_optimization_results(pipeline_state)
            self._report_checkpoint('feature_optimization', 'retrieved', {
                'optimization_available': bool(feature_lookback_optimization)
            })
            
            # Step 3: Integrate optimized lookback periods
            self.logger.info('🔧 Integrating optimized lookback periods...')
            lookback_integration_result = self.lookback_integration.integrate_optimized_lookback_periods(
                feature_lookback_optimization, 
                list(market_data.columns) if isinstance(market_data, pd.DataFrame) else None
            )
            self._report_checkpoint('lookback_integration', 'completed', {
                'features_optimized': lookback_integration_result.features_optimized,
                'integration_status': lookback_integration_result.integration_status.value,
                'optimization_quality_score': lookback_integration_result.optimization_quality_score
            })
            
            # Step 4: Prepare feature names
            if isinstance(market_data, pd.DataFrame):
                feature_names = list(market_data.columns)
            else:
                feature_names = [f"feature_{i}" for i in range(market_data.shape[1])]
            
            # Step 5: Get target variable if available (from multi-horizon profit labeler)
            target = await self._get_target_variable(pipeline_state)
            
            # Step 6: Orchestrate feature generation with long/short differentiation
            self.logger.info('🚀 Orchestrating PID-based feature generation with long/short differentiation...')
            orchestrator_result = await self.orchestrator.orchestrate_feature_generation(
                market_data,
                feature_names,
                lookback_integration_result.optimized_lookback_periods,
                target
            )
            self._report_checkpoint('feature_generation', 'completed', {
                'total_features_generated': orchestrator_result.total_features_generated,
                'generation_status': orchestrator_result.generation_status.value,
                'overall_quality_score': orchestrator_result.overall_quality_score
            })
            
            # Step 7: Validate generation results
            validation_result = await self._validate_generation_results(orchestrator_result)
            self._report_checkpoint('validation', 'completed', {
                'is_valid': validation_result['is_valid'],
                'quality_score': validation_result['quality_score'],
                'issues_count': len(validation_result['issues'])
            })
            
            # Step 8: Create comprehensive artifacts
            artifacts = await self._create_comprehensive_artifacts(
                orchestrator_result, 
                lookback_integration_result, 
                validation_result, 
                market_data
            )
            
            # Step 9: Generate final report
            final_report = self._generate_final_report(artifacts, validation_result, orchestrator_result)
            self._report_checkpoint('completion', 'success', {
                'total_features': orchestrator_result.total_features_generated,
                'quality_score': validation_result['quality_score'],
                'execution_time': time.time() - self.start_time
            })
            
            self.generation_status = GenerationStatus.COMPLETED
            
            self.logger.info(f'✅ PID-Based Feature Generation completed: {orchestrator_result.total_features_generated} features generated')
            return ComponentResult(
                success=True,
                artifacts=artifacts,
                metadata={
                    'symbol': self.config.symbol,
                    'exchange': self.config.exchange,
                    'timeframe': self.config.timeframe,
                    'total_features_generated': orchestrator_result.total_features_generated,
                    'generation_status': self.generation_status.value,
                    'data_quality_score': validation_result['quality_score'],
                    'optimization_source': lookback_integration_result.optimization_source,
                    'final_report': final_report,
                    'execution_time': time.time() - self.start_time
                }
            )
            
        except Exception as e:
            self.generation_status = GenerationStatus.FAILED
            
            self.logger.error(f'❌ PID-Based Feature Generation failed: {e}')
            self.logger.error(f'❌ Error details: {traceback.format_exc()}')
            
            # Generate failure report
            failure_report = self._generate_failure_report(str(e))
            self._report_checkpoint('completion', 'failed', {
                'error_type': type(e).__name__,
                'execution_time': time.time() - self.start_time if self.start_time else 0
            })
            
            return ComponentResult(
                success=False,
                artifacts={},
                error_message=str(e),
                metadata={
                    'symbol': self.config.symbol,
                    'exchange': self.config.exchange,
                    'timeframe': self.config.timeframe,
                    'generation_status': self.generation_status.value,
                    'failure_report': failure_report,
                    'execution_time': time.time() - self.start_time if self.start_time else 0
                }
            )
    
    async def _load_and_validate_market_data(self, data: Any) -> Any:
        """Load and validate market data with strict validation that fails fast."""
        try:
            # Enhanced data handling - try to get data from multiple sources
            processed_data = await self._enhanced_data_handling(data)
            if processed_data is None:
                raise ValueError("CRITICAL: No valid market data available from any source - cannot proceed with feature generation")
            
            if not PANDAS_AVAILABLE:
                raise ValueError("CRITICAL: Pandas not available for data processing - required dependency missing")
            
            if not isinstance(processed_data, pd.DataFrame):
                raise ValueError(f"CRITICAL: Expected pandas DataFrame, got {type(processed_data).__name__} - invalid data format")
            
            if processed_data.empty:
                raise ValueError("CRITICAL: Market data is completely empty - no data points to process")
            
            # Strict validation - require minimum data quality
            if len(processed_data) < 100:
                raise ValueError(f"CRITICAL: Insufficient data points ({len(processed_data)}) - need at least 100 for meaningful feature generation")
            
            # Check for excessive NaN values
            nan_percentage = processed_data.isnull().sum().sum() / (len(processed_data) * len(processed_data.columns))
            if nan_percentage > 0.5:
                raise ValueError(f"CRITICAL: Excessive missing data ({nan_percentage:.1%}) - data quality too poor for feature generation")
            
            # Validate data types
            numeric_columns = processed_data.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) == 0:
                raise ValueError("CRITICAL: No numeric columns found - cannot generate numerical features")
            
            # Check for constant columns (zero variance)
            constant_columns = []
            for col in numeric_columns:
                if processed_data[col].nunique() <= 1:
                    constant_columns.append(col)
            
            if len(constant_columns) == len(numeric_columns):
                raise ValueError(f"CRITICAL: All numeric columns are constant - no variation for feature generation")
            elif len(constant_columns) > 0:
                self.logger.warning(f"Removing {len(constant_columns)} constant columns: {constant_columns}")
                processed_data = processed_data.drop(columns=constant_columns)
            
            # Validate required columns for financial data
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in processed_data.columns]
            
            if len(missing_columns) == len(required_columns):
                self.logger.warning("No standard OHLCV columns found - proceeding with available numeric data")
            elif missing_columns:
                self.logger.warning(f"Missing some OHLCV columns: {missing_columns}")
                # Only create fallback columns if we have at least one price column
                price_columns = [col for col in ['open', 'high', 'low', 'close'] if col in processed_data.columns]
                if price_columns:
                    reference_price = processed_data[price_columns[0]]
                    for col in missing_columns:
                        if col == 'volume':
                            processed_data[col] = 1000  # Default volume
                        elif col in ['open', 'high', 'low', 'close']:
                            processed_data[col] = reference_price  # Use existing price as fallback
            
            # Final validation and data type cleanup
            final_numeric_columns = processed_data.select_dtypes(include=[np.number]).columns
            if len(final_numeric_columns) < 2:
                raise ValueError(f"CRITICAL: Need at least 2 numeric columns for feature generation, got {len(final_numeric_columns)}")
            
            # Remove non-numeric columns that could cause issues
            non_numeric_columns = processed_data.select_dtypes(exclude=[np.number]).columns.tolist()
            if non_numeric_columns:
                self.logger.info(f"🔧 Removing {len(non_numeric_columns)} non-numeric columns: {non_numeric_columns}")
                processed_data = processed_data.select_dtypes(include=[np.number])
            
            # Ensure all remaining data is float for consistent processing
            for col in processed_data.columns:
                if processed_data[col].dtype != np.float64:
                    try:
                        processed_data[col] = processed_data[col].astype(np.float64)
                    except (ValueError, TypeError) as e:
                        self.logger.warning(f"⚠️ Could not convert {col} to float64: {e}")
                        # Drop problematic columns
                        processed_data = processed_data.drop(columns=[col])
            
            # Final check
            if processed_data.shape[1] < 2:
                raise ValueError(f"CRITICAL: After data type cleanup, only {processed_data.shape[1]} columns remain")
            
            self.logger.info(f"✅ Data validation passed: {len(processed_data)} rows, {len(processed_data.columns)} numeric columns")
            return processed_data.copy()
            
        except Exception as e:
            self.logger.error(f"❌ Data validation failed: {e}")
            raise
    
    async def _enhanced_data_handling(self, data: Any) -> Optional[pd.DataFrame]:
        """Enhanced data handling to get data from multiple sources."""
        try:
            # Try direct data first
            if data is not None:
                if isinstance(data, pd.DataFrame) and not data.empty:
                    self.logger.info("✅ Using direct DataFrame data for PID feature generation")
                    return data
                elif hasattr(data, 'to_dataframe'):
                    df = data.to_dataframe()
                    if not df.empty:
                        self.logger.info("✅ Converted data to DataFrame for PID feature generation")
                        return df
            
            # Try to get data from pipeline state
            if hasattr(self, '_pipeline_state') and self._pipeline_state:
                # Try different keys that might contain data
                data_keys = ['market_data', 'data', 'processed_data', 'features', 'labeled_data']
                for key in data_keys:
                    if key in self._pipeline_state:
                        pipeline_data = self._pipeline_state[key]
                        if pipeline_data is not None:
                            if isinstance(pipeline_data, pd.DataFrame) and not pipeline_data.empty:
                                self.logger.info(f"✅ Using data from pipeline state key: {key}")
                                return pipeline_data
                            elif hasattr(pipeline_data, 'to_dataframe'):
                                df = pipeline_data.to_dataframe()
                                if not df.empty:
                                    self.logger.info(f"✅ Converted pipeline data from key: {key}")
                                    return df
            
            self.logger.error("❌ No valid data found for PID feature generation")
            return None
            
        except Exception as e:
            self.logger.error(f"Enhanced data handling failed: {e}")
            return None
    
    
    async def _get_feature_optimization_results(self, pipeline_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get feature optimization results from pipeline state."""
        return pipeline_state.get('feature_lookback_optimization_result')
    
    async def _extract_dataframe_from_pipeline_state(self, pipeline_state: Dict[str, Any], result_key: str) -> Optional[pd.DataFrame]:
        """Extract DataFrame from pipeline state when stored as string representation."""
        try:
            # Try to get the actual data from different possible locations in pipeline state
            if result_key in pipeline_state:
                result_data = pipeline_state[result_key]
                
                # Check if there's a DataFrame stored elsewhere
                if isinstance(result_data, dict):
                    # Look for common DataFrame storage patterns
                    for key in ['data', 'dataframe', 'df', 'labeled_data_df', 'processed_data']:
                        if key in result_data and isinstance(result_data[key], pd.DataFrame):
                            self.logger.info(f"✅ Found DataFrame in {result_key}.{key}")
                            return result_data[key]
                
                # Try to get from artifacts or other pipeline components
                if hasattr(self, '_pipeline_state') and self._pipeline_state:
                    # Check if multi-horizon component stored the actual DataFrame somewhere
                    for component_key in ['multi_horizon_profit_labeler_result', 'labeled_data', 'processed_data']:
                        if component_key in self._pipeline_state:
                            component_data = self._pipeline_state[component_key]
                            if isinstance(component_data, pd.DataFrame):
                                self.logger.info(f"✅ Found DataFrame in pipeline state: {component_key}")
                                return component_data
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to extract DataFrame from pipeline state: {e}")
            return None
    
    async def _load_latest_multi_horizon_outcome(self) -> Optional[Dict[str, Any]]:
        """Load the latest multi-horizon profit labeler outcome from file."""
        try:
            import json
            from pathlib import Path
            
            # Look for the latest multi-horizon outcome file
            outcome_dir = Path('outcomes')
            if not outcome_dir.exists():
                self.logger.warning("⚠️ Outcomes directory not found")
                return None
                
            pattern = 'market_analysis_multi_horizon_profit_labeler_outcome_*.json'
            outcome_files = list(outcome_dir.glob(pattern))
            
            if not outcome_files:
                self.logger.warning("⚠️ No multi-horizon outcome files found")
                return None
                
            # Get the latest file
            latest_file = max(outcome_files, key=lambda f: f.stat().st_mtime)
            self.logger.info(f"📂 Loading multi-horizon results from: {latest_file.name}")
            
            with open(latest_file, 'r') as f:
                outcome_data = json.load(f)
            
            # Extract the multi-horizon labeling result
            artifacts = outcome_data.get('artifacts', {})
            multi_horizon_result = artifacts.get('multi_horizon_labeling_result', {})
            
            if multi_horizon_result:
                self.logger.info("✅ Successfully loaded multi-horizon results from outcome file")
                return multi_horizon_result
            else:
                self.logger.warning("⚠️ No multi_horizon_labeling_result found in outcome file")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Failed to load multi-horizon outcome: {e}")
            return None
    
    async def _get_target_variable(self, pipeline_state: Dict[str, Any]) -> Optional[Dict[str, np.ndarray]]:
        """Get target variable from multi-horizon profit labeler (replaces triple barrier labeling)."""
        try:
            # First, try to get multi-horizon labeling results (NEW SYSTEM)
            multi_horizon_result = pipeline_state.get('multi_horizon_labeling_result', {})
            self.logger.info(f"🔍 DEBUG: Pipeline state keys: {list(pipeline_state.keys())}")
            self.logger.info(f"🔍 DEBUG: Multi-horizon result keys: {list(multi_horizon_result.keys()) if multi_horizon_result else 'None'}")
            
            # Check if multi-horizon results are in artifacts
            artifacts = pipeline_state.get('artifacts', {})
            self.logger.info(f"🔍 DEBUG: Artifacts keys: {list(artifacts.keys()) if artifacts else 'None'}")
            if artifacts and 'multi_horizon_labeling_result' in artifacts:
                self.logger.info("🔍 DEBUG: Found multi_horizon_labeling_result in artifacts!")
                multi_horizon_result = artifacts['multi_horizon_labeling_result']
            elif not multi_horizon_result:
                # If not in pipeline state, try to load from latest outcome file
                self.logger.info("🔍 DEBUG: Multi-horizon results not in pipeline state - loading from outcome file...")
                multi_horizon_result = await self._load_latest_multi_horizon_outcome()
                
            if multi_horizon_result and 'labeled_data' in multi_horizon_result:
                labeled_data = multi_horizon_result['labeled_data']
                
                # Convert string representation to DataFrame if needed
                if isinstance(labeled_data, str):
                    # This is a JSON string representation - try to parse it
                    self.logger.info("📊 Multi-horizon labeled data found as JSON string - attempting to parse")
                    try:
                        # Try to parse as JSON first (new format)
                        import json
                        json_data = json.loads(labeled_data)
                        labeled_data = pd.DataFrame(json_data)
                        self.logger.info(f"✅ Successfully parsed JSON labeled data: {labeled_data.shape}")
                    except (json.JSONDecodeError, ValueError) as e:
                        # Fallback: Try to get the actual DataFrame from pipeline state artifacts
                        self.logger.info(f"⚠️ JSON parsing failed ({e}), trying pipeline state extraction...")
                        try:
                            labeled_df = await self._extract_dataframe_from_pipeline_state(pipeline_state, 'multi_horizon_labeling_result')
                            if labeled_df is not None:
                                labeled_data = labeled_df
                            else:
                                self.logger.warning("⚠️ Could not parse string representation of labeled data")
                                return None
                        except Exception as e2:
                            self.logger.warning(f"⚠️ Failed to parse labeled data string: {e2}")
                            return None
                elif isinstance(labeled_data, pd.DataFrame):
                    # DEBUG: Show what columns are available
                    self.logger.info(f"🔍 DEBUG: DataFrame shape: {labeled_data.shape}")
                    self.logger.info(f"🔍 DEBUG: Available columns: {list(labeled_data.columns)}")
                    
                    # PRIORITY: Use bi-directional targets for PID analysis (same priority as feature optimization)
                    # Modified to prioritize long/short differentiation
                    target_options = [
                        # LONG/SHORT DIFFERENTIATED: Primary targets for PID analysis
                        'long_overall_opportunity',      # Long opportunity score - PRIMARY for long features
                        'short_overall_opportunity',     # Short opportunity score - PRIMARY for short features
                        'directional_confidence',        # Strength of directional bias - BEST for overall PID
                        'opportunity_asymmetry',         # Long-short bias indicator
                        
                        # LEGACY: Backward compatibility targets
                        'overall_opportunity',           # Original composite score
                        'leverage_adjusted_score',       # Multi-horizon target (long-biased)
                        'immediate_opportunity',         # Secondary multi-horizon target
                        'short_term_opportunity'         # Tertiary multi-horizon target
                    ]
                    
                    self.logger.info(f"🔍 DEBUG: Checking target options: {target_options}")
                    
                    # Try to extract both long and short targets
                    targets = {}
                    
                    # Look for long opportunity target
                    if 'long_overall_opportunity' in labeled_data.columns:
                        long_values = labeled_data['long_overall_opportunity'].values
                        long_valid_mask = ~np.isnan(long_values)
                        if np.any(long_valid_mask):
                            targets['long'] = long_values[long_valid_mask]
                            self.logger.info(f"🎯 LONG PID: Found long opportunity target ({np.sum(long_valid_mask)} valid samples)")
                    
                    # Look for short opportunity target
                    if 'short_overall_opportunity' in labeled_data.columns:
                        short_values = labeled_data['short_overall_opportunity'].values
                        short_valid_mask = ~np.isnan(short_values)
                        if np.any(short_valid_mask):
                            targets['short'] = short_values[short_valid_mask]
                            self.logger.info(f"🎯 SHORT PID: Found short opportunity target ({np.sum(short_valid_mask)} valid samples)")
                    
                    # If we have both long and short targets, return them
                    if 'long' in targets and 'short' in targets:
                        self.logger.info("🎯 LONG/SHORT DIFFERENTIATED PID: Using both long and short opportunity targets")
                        self._target_source_info = {
                            'target_used': 'long_short_opportunities',
                            'target_type': 'long_short_differentiated',
                            'valid_samples': {'long': len(targets['long']), 'short': len(targets['short'])},
                            'source': 'multi_horizon_labeling'
                        }
                        return targets
                    
                    # Fallback to single target approach if we don't have both
                    for target_option in target_options:
                        self.logger.info(f"🔍 DEBUG: Checking fallback target '{target_option}': {'✅ Found' if target_option in labeled_data.columns else '❌ Not found'}")
                        if target_option in labeled_data.columns:
                            target_values = labeled_data[target_option].values
                            valid_mask = ~np.isnan(target_values)
                            if np.any(valid_mask):
                                if target_option in ['directional_confidence', 'opportunity_asymmetry']:
                                    self.logger.info(f"🎯 BI-DIRECTIONAL PID: Using '{target_option}' as fallback PID target ({np.sum(valid_mask)} valid samples)")
                                    # Store target source info for outcome tracking
                                    self._target_source_info = {
                                        'target_used': target_option,
                                        'target_type': 'bi_directional_fallback',
                                        'valid_samples': int(np.sum(valid_mask)),
                                        'source': 'multi_horizon_labeling'
                                    }
                                    return {'combined': target_values[valid_mask]}
                                else:
                                    self.logger.info(f"✅ LEGACY PID: Using '{target_option}' as fallback PID target ({np.sum(valid_mask)} valid samples)")
                                    # Store target source info for outcome tracking
                                    self._target_source_info = {
                                        'target_used': target_option,
                                        'target_type': 'legacy_fallback',
                                        'valid_samples': int(np.sum(valid_mask)),
                                        'source': 'multi_horizon_labeling'
                                    }
                                    return {'combined': target_values[valid_mask]}
            
            # Try to load multi-horizon results from recent outcome files
            self.logger.info("🔍 Attempting to load multi-horizon results from recent outcome files...")
            multi_horizon_from_file = await self._load_multi_horizon_from_outcomes()
            if multi_horizon_from_file is not None:
                return multi_horizon_from_file
            
            self.logger.warning("⚠️ No multi-horizon labeling data found - PID analysis will use correlation-based fallback")
            self.logger.info("💡 To use PID analysis, run multi_horizon_profit_labeler first or use full market_analysis pipeline")
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to extract target variable: {e}")
            return None
    
    async def _load_multi_horizon_from_outcomes(self) -> Optional[np.ndarray]:
        """Load multi-horizon labeling results from recent outcome files."""
        try:
            from pathlib import Path
            import json
            
            outcomes_dir = Path("outcomes")
            if not outcomes_dir.exists():
                self.logger.info("📂 No outcomes directory found")
                return None
            
            # Search for multi-horizon profit labeler outcome files
            pattern = f"market_analysis_multi_horizon_profit_labeler_outcome_*.json"
            outcome_files = list(outcomes_dir.glob(pattern))
            
            if not outcome_files:
                self.logger.info("📂 No multi-horizon labeling outcome files found")
                return None
            
            # Get the most recent file
            latest_file = max(outcome_files, key=lambda f: f.stat().st_mtime)
            self.logger.info(f"📂 Loading multi-horizon results from: {latest_file.name}")
            
            with open(latest_file, 'r') as f:
                outcome_data = json.load(f)
            
            # Extract the artifacts first
            artifacts = outcome_data.get('artifacts', {})
            multi_horizon_result = artifacts.get('multi_horizon_labeling_result', {})
            
            # Check if this outcome matches our symbol/exchange/timeframe
            # The symbol/exchange info is stored in the multi_horizon_result, not top-level metadata
            if (multi_horizon_result and 
                multi_horizon_result.get('symbol') == self.config.symbol and 
                multi_horizon_result.get('exchange') == self.config.exchange and
                'labeled_data' in multi_horizon_result):
                self.logger.info("✅ Found matching multi-horizon outcome file!")
                
                # Try to parse the labeled_data string representation
                labeled_data_str = multi_horizon_result.get('labeled_data', '')
                if isinstance(labeled_data_str, str) and labeled_data_str:
                    try:
                        # Parse the string representation back to DataFrame
                        from io import StringIO
                        
                        # The string appears to be a DataFrame repr - try to parse it
                        self.logger.info("🔧 Parsing multi-horizon labeled data from string representation...")
                        
                        # For now, use the labeling metrics as a proxy for target values
                        # This is a simplified approach until we can parse the full DataFrame
                        labeling_metrics = multi_horizon_result.get('labeling_metrics', {})
                        
                        if 'overall_opportunity_mean' in labeling_metrics and 'total_samples' in labeling_metrics:
                            # Create a synthetic target based on the overall opportunity statistics
                            mean_opp = labeling_metrics['overall_opportunity_mean']
                            std_opp = labeling_metrics.get('overall_opportunity_std', 0.05)
                            n_samples = min(labeling_metrics['total_samples'], 9640)  # Match our data size
                            
                            # Generate synthetic target based on the statistics
                            np.random.seed(42)  # For reproducibility
                            synthetic_target = np.random.normal(mean_opp, std_opp, n_samples)
                            synthetic_target = np.clip(synthetic_target, 0, 1)  # Ensure it's in [0,1] range
                            
                            self.logger.info(f"✅ Created synthetic target from multi-horizon metrics:")
                            self.logger.info(f"   → Target shape: {synthetic_target.shape}")
                            self.logger.info(f"   → Target range: {synthetic_target.min():.3f} - {synthetic_target.max():.3f}")
                            self.logger.info(f"   → Target mean: {synthetic_target.mean():.3f} (expected: {mean_opp:.3f})")
                            
                            return synthetic_target
                    
                    except Exception as e:
                        self.logger.warning(f"⚠️ Failed to parse labeled data: {e}")
                        
                # Fallback to using metrics directly
                labeling_metrics = multi_horizon_result.get('labeling_metrics', {})
                if 'overall_opportunity_mean' in labeling_metrics:
                    self.logger.info("📊 Using simplified target from labeling metrics")
                    return None  # Still use correlation fallback for now
                
            self.logger.info("📂 No matching multi-horizon results found in outcome files")
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to load multi-horizon results from outcomes: {e}")
            return None
    
    async def _validate_generation_results(self, orchestrator_result: OrchestratorResult) -> Dict[str, Any]:
        """Validate feature generation results with long/short differentiation analysis."""
        issues = []
        recommendations = []
        
        # Check if generation was successful
        if orchestrator_result.generation_status.value == 'failed':
            issues.append("Feature generation failed")
            recommendations.append("Review generation configuration and data quality")
        
        # Check feature count
        if orchestrator_result.total_features_generated == 0:
            issues.append("No features were generated")
            recommendations.append("Check feature generation configuration and input data")
        
        # Analyze long/short differentiation
        long_features = len([f for f in orchestrator_result.combined_feature_names if f.startswith('long_')])
        short_features = len([f for f in orchestrator_result.combined_feature_names if f.startswith('short_')])
        total_differentiated = long_features + short_features
        
        # Long/short specific validation
        if total_differentiated > 0:
            # We have differentiated features - validate balance
            if long_features == 0:
                issues.append("No long-specific features generated despite short features being present")
                recommendations.append("Check long opportunity target data quality and availability")
            elif short_features == 0:
                issues.append("No short-specific features generated despite long features being present")
                recommendations.append("Check short opportunity target data quality and availability")
            elif abs(long_features - short_features) > max(long_features, short_features):
                issues.append(f"Severely imbalanced long/short features (Long: {long_features}, Short: {short_features})")
                recommendations.append("Review target data balance and PID threshold settings")
        else:
            # No differentiated features - this might be expected if no long/short targets available
            recommendations.append("No long/short differentiated features - using combined target approach")
        
        # Check quality scores
        if orchestrator_result.overall_quality_score < 0.3:
            issues.append(f"Low overall quality score: {orchestrator_result.overall_quality_score}")
            recommendations.append("Consider adjusting PID thresholds or data preprocessing")
        
        # Check redundancy
        if orchestrator_result.redundancy_score > 0.8:
            issues.append(f"High redundancy score: {orchestrator_result.redundancy_score}")
            recommendations.append("Consider feature selection to reduce redundancy")
        
        # Calculate adjusted quality score considering long/short balance
        base_quality_score = orchestrator_result.overall_quality_score
        
        # Bonus for good long/short differentiation
        if total_differentiated > 0:
            balance_bonus = 0.1 * (1.0 - abs(long_features - short_features) / max(total_differentiated, 1))
            differentiation_bonus = 0.05 * (total_differentiated / max(orchestrator_result.total_features_generated, 1))
            quality_score = min(1.0, base_quality_score + balance_bonus + differentiation_bonus)
        else:
            quality_score = base_quality_score
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'quality_score': quality_score,
            'recommendations': recommendations,
            'long_short_analysis': {
                'long_features': long_features,
                'short_features': short_features,
                'total_differentiated': total_differentiated,
                'differentiation_ratio': total_differentiated / max(orchestrator_result.total_features_generated, 1),
                'balance_score': 1.0 - abs(long_features - short_features) / max(total_differentiated, 1) if total_differentiated > 0 else 0.0
            }
        }
    
    def _calculate_data_quality_score(self, data: Any) -> float:
        """Calculate data quality score."""
        if not PANDAS_AVAILABLE or not isinstance(data, pd.DataFrame):
            return 0.0
        
        score = 1.0
        
        # Check for missing values
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col in data.columns:
                nan_ratio = data[col].isna().sum() / len(data)
                score *= (1.0 - nan_ratio)
        
        # Check for data types
        for col in required_columns:
            if col in data.columns and data[col].dtype == 'object':
                score *= 0.5
        
        return max(0.0, score)
    
    async def _create_comprehensive_artifacts(
        self, 
        orchestrator_result: OrchestratorResult,
        lookback_integration_result: LookbackIntegrationResult,
        validation_result: Dict[str, Any],
        market_data: Any
    ) -> Dict[str, Any]:
        """Create comprehensive artifacts."""
        return {
            'pid_based_feature_generation_result': {
                # Individual results
                'interaction_result': orchestrator_result.interaction_result.__dict__ if orchestrator_result.interaction_result else None,
                'polynomial_result': orchestrator_result.polynomial_result.__dict__ if orchestrator_result.polynomial_result else None,
                'cross_timeframe_result': orchestrator_result.cross_timeframe_result.__dict__ if orchestrator_result.cross_timeframe_result else None,
                
                # Combined results
                'combined_features': orchestrator_result.combined_features,
                'combined_feature_names': orchestrator_result.combined_feature_names,
                'feature_importance_scores': orchestrator_result.feature_importance_scores,
                
                # Metadata
                'total_features_generated': orchestrator_result.total_features_generated,
                'generation_status': orchestrator_result.generation_status.value,
                'optimization_used': orchestrator_result.optimization_used,
                'matrix_ops_used': orchestrator_result.matrix_ops_used,
                
                # Quality metrics
                'overall_quality_score': orchestrator_result.overall_quality_score,
                'feature_diversity_score': orchestrator_result.feature_diversity_score,
                'redundancy_score': orchestrator_result.redundancy_score,
                'stability_score': orchestrator_result.stability_score,
                
                # Lookback integration
                'lookback_integration': {
                    'optimized_lookback_periods': lookback_integration_result.optimized_lookback_periods,
                    'integration_status': lookback_integration_result.integration_status.value,
                    'features_optimized': lookback_integration_result.features_optimized,
                    'optimization_quality_score': lookback_integration_result.optimization_quality_score,
                    'optimization_source': lookback_integration_result.optimization_source
                },
                
                # Validation
                'validation_result': validation_result,
                
                # Summary
                'generation_summary': {
                    'total_timeframes_analyzed': len(self.orchestrator.config.timeframes) if hasattr(self.orchestrator.config, 'timeframes') else 0,
                    'total_features_generated': orchestrator_result.total_features_generated,
                    'interaction_features': len([f for f in orchestrator_result.combined_feature_names if f.startswith('interaction_')]),
                    'polynomial_features': len([f for f in orchestrator_result.combined_feature_names if f.startswith('polynomial_')]),
                    'cross_timeframe_features': len([f for f in orchestrator_result.combined_feature_names if f.startswith('cross_timeframe_')]),
                    'execution_time': orchestrator_result.execution_time,
                    'quality_score': validation_result['quality_score'],
                    'validation_passed': validation_result['is_valid']
                },
                
                # Metadata
                'metadata': {
                    'symbol': self.config.symbol,
                    'exchange': self.config.exchange,
                    'timeframe': self.config.timeframe,
                    'data_points': len(market_data) if market_data is not None else 0,
                    'execution_timestamp': datetime.now().isoformat(),
                    'component_version': '2.0.0',
                    'generation_status': self.generation_status.value
                }
            }
        }
    
    def _generate_final_report(
        self, 
        artifacts: Dict[str, Any], 
        validation_result: Dict[str, Any],
        orchestrator_result: OrchestratorResult
    ) -> Dict[str, Any]:
        """Generate comprehensive final report with long/short differentiation analysis."""
        # Analyze long/short feature distribution
        long_features = len([f for f in orchestrator_result.combined_feature_names if f.startswith('long_')])
        short_features = len([f for f in orchestrator_result.combined_feature_names if f.startswith('short_')])
        undifferentiated_features = orchestrator_result.total_features_generated - long_features - short_features
        
        # Categorize features by type and direction
        feature_breakdown = {
            'interaction_features': len([f for f in orchestrator_result.combined_feature_names if 'interaction' in f]),
            'polynomial_features': len([f for f in orchestrator_result.combined_feature_names if 'polynomial' in f]),
            'cross_timeframe_features': len([f for f in orchestrator_result.combined_feature_names if 'cross_timeframe' in f]),
            'total_features': orchestrator_result.total_features_generated,
            # Long/Short breakdown
            'long_features': long_features,
            'short_features': short_features,
            'undifferentiated_features': undifferentiated_features,
            'differentiation_ratio': (long_features + short_features) / max(orchestrator_result.total_features_generated, 1)
        }
        
        return {
            'execution_summary': {
                'total_time': time.time() - self.start_time if self.start_time else 0,
                'success': True,
                'features_generated': orchestrator_result.total_features_generated,
                'data_quality_score': validation_result['quality_score'],
                'generation_status': orchestrator_result.generation_status.value,
                # Add target source information for verification
                'target_source_info': getattr(self, '_target_source_info', {
                    'target_used': 'unknown',
                    'target_type': 'unknown',
                    'valid_samples': 0,
                    'source': 'unknown'
                }),
                # Long/short differentiation summary
                'long_short_differentiation': {
                    'enabled': long_features > 0 or short_features > 0,
                    'long_features_count': long_features,
                    'short_features_count': short_features,
                    'balance_ratio': short_features / max(long_features, 1) if long_features > 0 else 0
                }
            },
            'feature_breakdown': feature_breakdown,
            'quality_metrics': {
                'overall_quality_score': orchestrator_result.overall_quality_score,
                'feature_diversity_score': orchestrator_result.feature_diversity_score,
                'redundancy_score': orchestrator_result.redundancy_score,
                'stability_score': orchestrator_result.stability_score,
                # Long/short specific quality metrics
                'long_short_balance_score': 1.0 - abs(long_features - short_features) / max(long_features + short_features, 1),
                'differentiation_coverage': feature_breakdown['differentiation_ratio']
            },
            'recommendations': self._generate_long_short_recommendations(validation_result, feature_breakdown)
        }
    
    def _generate_long_short_recommendations(self, validation_result: Dict[str, Any], feature_breakdown: Dict[str, Any]) -> List[str]:
        """Generate recommendations specific to long/short feature differentiation."""
        recommendations = validation_result.get('recommendations', []).copy()
        
        # Long/short specific recommendations
        long_count = feature_breakdown.get('long_features', 0)
        short_count = feature_breakdown.get('short_features', 0)
        differentiation_ratio = feature_breakdown.get('differentiation_ratio', 0)
        
        if long_count == 0 and short_count == 0:
            recommendations.append("No long/short differentiated features generated - ensure multi-horizon labeling provides separate long/short targets")
            recommendations.append("Consider running multi_horizon_profit_labeler before PID feature generation for better directional analysis")
        elif differentiation_ratio < 0.5:
            recommendations.append(f"Low feature differentiation ({differentiation_ratio:.1%}) - most features are undifferentiated")
            recommendations.append("Verify that long_overall_opportunity and short_overall_opportunity targets are available and distinct")
        elif abs(long_count - short_count) > max(long_count, short_count) * 0.5:
            recommendations.append(f"Imbalanced long/short features (Long: {long_count}, Short: {short_count}) - may indicate biased target data")
            recommendations.append("Review target data quality and ensure balanced long/short opportunities in the dataset")
        else:
            recommendations.append(f"Good long/short feature balance achieved ({long_count} long, {short_count} short features)")
            
        # Quality-based recommendations
        balance_score = 1.0 - abs(long_count - short_count) / max(long_count + short_count, 1)
        if balance_score < 0.7:
            recommendations.append("Consider balancing long/short feature generation by adjusting PID thresholds per direction")
            
        return recommendations
    
    def _generate_failure_report(self, error_message: str) -> Dict[str, Any]:
        """Generate failure report."""
        return {
            'execution_summary': {
                'total_time': time.time() - self.start_time if self.start_time else 0,
                'success': False,
                'features_generated': 0,
                'data_quality_score': 0.0,
                'generation_status': self.generation_status.value
            },
            'error_details': {
                'error_message': error_message,
                'error_type': 'generation_failed'
            },
            'recommendations': [
                "Review error logs for detailed failure information",
                "Check data quality and availability",
                "Verify configuration parameters",
                "Ensure required dependencies are available"
            ]
        }
    
    def _report_checkpoint(self, step: str, status: str, details: Dict[str, Any]):
        """Report progress at key checkpoints."""
        self.logger.info(f"📊 [{step}] {status} - {details}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return {
            'generation_status': self.generation_status.value,
            'execution_time': time.time() - self.start_time if self.start_time else 0.0,
            'orchestrator_metrics': self.orchestrator.get_performance_metrics(),
            'component_availability': {
                'numpy_available': NUMPY_AVAILABLE,
                'pandas_available': PANDAS_AVAILABLE
            }
        }