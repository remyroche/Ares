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
            
            # Step 5: Get target variable if available (from triple barrier labeling)
            target = await self._get_target_variable(pipeline_state)
            
            # Step 6: Orchestrate feature generation
            self.logger.info('🚀 Orchestrating PID-based feature generation...')
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
        """Load and validate market data with enhanced data handling."""
        try:
            # Enhanced data handling - try to get data from multiple sources
            processed_data = await self._enhanced_data_handling(data)
            if processed_data is None:
                raise ValueError("No valid market data available from any source")
            
            if not PANDAS_AVAILABLE:
                raise ValueError("Pandas not available for data processing")
            
            if not isinstance(processed_data, pd.DataFrame):
                raise ValueError(f"Expected pandas DataFrame, got {type(processed_data).__name__}")
            
            if processed_data.empty:
                raise ValueError("Market data is empty")
            
            # Validate required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in processed_data.columns]
            if missing_columns:
                self.logger.warning(f"Missing required columns: {missing_columns}")
                # Create fallback columns
                for col in missing_columns:
                    if col == 'volume':
                        processed_data[col] = 1000  # Default volume
                    else:
                        processed_data[col] = processed_data.get('close', 100.0)  # Use close price as fallback
            
            return processed_data.copy()
            
        except Exception as e:
            self.logger.error(f"Data loading failed: {e}")
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
    
    async def _get_target_variable(self, pipeline_state: Dict[str, Any]) -> Optional[np.ndarray]:
        """Get target variable from triple barrier labeling if available."""
        try:
            triple_barrier_result = pipeline_state.get('triple_barrier_labeling_result', {})
            if triple_barrier_result and 'labels' in triple_barrier_result:
                labels = triple_barrier_result['labels']
                if isinstance(labels, (list, np.ndarray)):
                    return np.array(labels)
            return None
        except Exception as e:
            self.logger.warning(f"Failed to extract target variable: {e}")
            return None
    
    async def _validate_generation_results(self, orchestrator_result: OrchestratorResult) -> Dict[str, Any]:
        """Validate feature generation results."""
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
        
        # Check quality scores
        if orchestrator_result.overall_quality_score < 0.3:
            issues.append(f"Low overall quality score: {orchestrator_result.overall_quality_score}")
            recommendations.append("Consider adjusting PID thresholds or data preprocessing")
        
        # Check redundancy
        if orchestrator_result.redundancy_score > 0.8:
            issues.append(f"High redundancy score: {orchestrator_result.redundancy_score}")
            recommendations.append("Consider feature selection to reduce redundancy")
        
        quality_score = orchestrator_result.overall_quality_score
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'quality_score': quality_score,
            'recommendations': recommendations
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
        """Generate comprehensive final report."""
        return {
            'execution_summary': {
                'total_time': time.time() - self.start_time if self.start_time else 0,
                'success': True,
                'features_generated': orchestrator_result.total_features_generated,
                'data_quality_score': validation_result['quality_score'],
                'generation_status': orchestrator_result.generation_status.value
            },
            'feature_breakdown': {
                'interaction_features': len([f for f in orchestrator_result.combined_feature_names if f.startswith('interaction_')]),
                'polynomial_features': len([f for f in orchestrator_result.combined_feature_names if f.startswith('polynomial_')]),
                'cross_timeframe_features': len([f for f in orchestrator_result.combined_feature_names if f.startswith('cross_timeframe_')]),
                'total_features': orchestrator_result.total_features_generated
            },
            'quality_metrics': {
                'overall_quality_score': orchestrator_result.overall_quality_score,
                'feature_diversity_score': orchestrator_result.feature_diversity_score,
                'redundancy_score': orchestrator_result.redundancy_score,
                'stability_score': orchestrator_result.stability_score
            },
            'recommendations': validation_result.get('recommendations', [])
        }
    
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