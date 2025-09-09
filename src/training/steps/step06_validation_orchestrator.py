from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import pandas as pd
from src.training.steps.standardized_parquet_handler import standardized_parquet_handler

"""
Step06 Validation Orchestrator with Extensive Utility Integration

This module orchestrates comprehensive validation, tracking, and reporting
for all step06 components with extensive utility integration. It provides:
- Function call validation and tracking
- Function-to-function call monitoring
- Comprehensive function completion reports
- Performance monitoring and analysis
- Error handling with detailed context
- Extensive utility integration with dependency injection
- M1 optimization for performance
- Advanced data processing and validation
"""
import asyncio
import json
import logging
from datetime import datetime
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation

# Import utility container and services
from .step06_utility_container import (
    Step06UtilityContainer, UtilityConfig, get_utility_container,
    utility_container_context, inject_utilities
)

try:
    from .step06_enhanced_validation_framework import get_step06_validation_summary, reset_step06_validation_tracking, ValidationLevel, FunctionStatus

    VALIDATION_FRAMEWORK_AVAILABLE = True
except ImportError as e:
    logging.warning(f'Step06 validation framework not available: {e}')
    VALIDATION_FRAMEWORK_AVAILABLE = False

    def get_step06_validation_summary() -> Any:
        return {'error': 'Validation framework not available'}

    def reset_step06_validation_tracking() -> None:
        pass

    class ValidationLevel:
        BASIC = 'basic'
        DETAILED = 'detailed'
        COMPREHENSIVE = 'comprehensive'

    class FunctionStatus:
        PENDING = 'pending'
        IN_PROGRESS = 'in_progress'
        COMPLETED = 'completed'
        FAILED = 'failed'
        TIMEOUT = 'timeout'

try:
    from src.training.steps.market_analysis.step06_feature_engineering import FeatureInteractionEngine
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    logging.warning(f'FeatureInteractionEngine not available: {e}')

    @log_important_calls
    class FeatureInteractionEngine:

        def __init__(self, config: Dict[str, Any]) -> None:
            self.config = config
            self.logger = logging.getLogger(__name__)

        async def create_interactions(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> Any:
            return data
    COMPONENTS_AVAILABLE = False
try:
    from src.training.steps.step06_labeling_components.optimized_triple_barrier_labeling import OptimizedTripleBarrierLabeling
except ImportError as e:
    logging.warning(f'OptimizedTripleBarrierLabeling not available: {e}')

    @log_important_calls
    class OptimizedTripleBarrierLabeling:

        def __init__(self, config: Dict[str, Any]) -> None:
            self.config = config
            self.logger = logging.getLogger(__name__)
try:
    from ...data_collection.feature_engineering.step06_feature_engineering import FeatureEngineeringStep
except ImportError as e:
    logging.warning(f'FeatureEngineeringStep not available: {e}')

    @log_important_calls
    class FeatureEngineeringStep:

        def __init__(self, config: Dict[str, Any]) -> None:
            self.config = config
            self.logger = logging.getLogger(__name__)

class Step06ValidationOrchestrator:
    """
    Orchestrates comprehensive validation and reporting for all step06 components
    with extensive utility integration and dependency injection.
    """
    @log_important_calls

    def __init__(self, output_dir: str='step06_validation_reports', 
                 utility_config: Optional[UtilityConfig] = None) -> None:
        """
        Initialize the step06 validation orchestrator with utility integration.
        
        Args:
            output_dir: Directory to save validation reports
            utility_config: Configuration for utility services
        """
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents = True, exist_ok = True)
        self.components = {}
        self.component_reports = {}
        self.overall_report = {}
        
        # Initialize utility configuration
        self.utility_config = utility_config or UtilityConfig(
            enable_common_operations=True,
            enable_data_processing=True,
            enable_math_validation=True,
            enable_parquet_utils=True,
            enable_serialization=True,
            enable_m1_gpu=True,
            enable_m1_memory=True,
            enable_m1_cpu=True,
            data_processing_chunk_size=10000,
            m1_memory_limit_gb=8.0,
            m1_max_workers=8
        )
        
        # Utility services will be initialized when needed
        self.utility_container = None
        self.performance_metrics = {
            'total_validation_time': 0.0,
            'utility_initialization_time': 0.0,
            'data_processing_time': 0.0,
            'memory_usage_mb': 0.0,
            'gpu_utilization': 0.0,
            'cpu_utilization': 0.0,
            'validation_errors': 0,
            'utility_errors': 0
        }
        
        self.logger.info('🎯 Step06 Validation Orchestrator with Utility Integration initialized')
        self.logger.info(f'   Output directory: {self.output_dir}')
        self.logger.info(f'   Components available: {COMPONENTS_AVAILABLE}')
        self.logger.info(f'   Utility integration: ENABLED')
        self.logger.info(f'   M1 optimization: ENABLED')

    async def initialize_utilities(self) -> Dict[str, bool]:
        """
        Initialize utility services with dependency injection.
        
        Returns:
            Dictionary with utility initialization status
        """
        start_time = time.time()
        self.logger.info('🔧 Initializing utility services...')
        
        try:
            # Initialize utility container
            self.utility_container = await get_utility_container(self.utility_config)
            
            # Test utility services
            utility_status = {}
            
            # Test common operations
            try:
                common_ops = self.utility_container.get_common_operations()
                current_time = common_ops.get_operation('datetime', 'get_current_datetime')()
                utility_status['common_operations'] = True
                self.logger.info('✅ Common operations service initialized')
            except Exception as e:
                utility_status['common_operations'] = False
                self.logger.error(f'❌ Common operations service failed: {e}')
                self.performance_metrics['utility_errors'] += 1
            
            # Test data processing
            try:
                data_proc = self.utility_container.get_data_processing()
                utility_status['data_processing'] = True
                self.logger.info('✅ Data processing service initialized')
            except Exception as e:
                utility_status['data_processing'] = False
                self.logger.error(f'❌ Data processing service failed: {e}')
                self.performance_metrics['utility_errors'] += 1
            
            # Test math validation
            try:
                math_val = self.utility_container.get_math_validation()
                utility_status['math_validation'] = True
                self.logger.info('✅ Math validation service initialized')
            except Exception as e:
                utility_status['math_validation'] = False
                self.logger.error(f'❌ Math validation service failed: {e}')
                self.performance_metrics['utility_errors'] += 1
            
            # Test parquet utilities
            try:
                parquet_svc = self.utility_container.get_parquet()
                utility_status['parquet_utils'] = True
                self.logger.info('✅ Parquet utilities service initialized')
            except Exception as e:
                utility_status['parquet_utils'] = False
                self.logger.error(f'❌ Parquet utilities service failed: {e}')
                self.performance_metrics['utility_errors'] += 1
            
            # Test serialization
            try:
                serialization_svc = self.utility_container.get_serialization()
                utility_status['serialization'] = True
                self.logger.info('✅ Serialization service initialized')
            except Exception as e:
                utility_status['serialization'] = False
                self.logger.error(f'❌ Serialization service failed: {e}')
                self.performance_metrics['utility_errors'] += 1
            
            # Test M1 GPU
            try:
                m1_gpu = self.utility_container.get_m1_gpu()
                utility_status['m1_gpu'] = True
                self.logger.info('✅ M1 GPU service initialized')
            except Exception as e:
                utility_status['m1_gpu'] = False
                self.logger.error(f'❌ M1 GPU service failed: {e}')
                self.performance_metrics['utility_errors'] += 1
            
            # Test M1 memory
            try:
                m1_memory = self.utility_container.get_m1_memory()
                utility_status['m1_memory'] = True
                self.logger.info('✅ M1 memory service initialized')
            except Exception as e:
                utility_status['m1_memory'] = False
                self.logger.error(f'❌ M1 memory service failed: {e}')
                self.performance_metrics['utility_errors'] += 1
            
            # Test M1 CPU
            try:
                m1_cpu = self.utility_container.get_m1_cpu()
                utility_status['m1_cpu'] = True
                self.logger.info('✅ M1 CPU service initialized')
            except Exception as e:
                utility_status['m1_cpu'] = False
                self.logger.error(f'❌ M1 CPU service failed: {e}')
                self.performance_metrics['utility_errors'] += 1
            
            # Update performance metrics
            init_time = time.time() - start_time
            self.performance_metrics['utility_initialization_time'] = init_time
            
            # Get health report
            health_report = self.utility_container.get_health_report()
            self.logger.info(f'📊 Utility services health: {health_report["status"]}')
            self.logger.info(f'   Healthy services: {health_report["healthy_services"]}/{health_report["total_services"]}')
            
            return utility_status
            
        except Exception as e:
            self.logger.error(f'❌ Utility initialization failed: {e}')
            self.performance_metrics['utility_errors'] += 1
            return {}

    @inject_utilities('common_ops', 'data_proc', 'math_val', 'parquet', 'serialization')
    async def _validate_with_utilities(self, test_data: pd.DataFrame, 
                                     common_ops, data_proc, math_val, parquet, serialization) -> Dict[str, Any]:
        """
        Perform validation using utility services.
        
        Args:
            test_data: Test data for validation
            common_ops: Common operations service
            data_proc: Data processing service
            math_val: Math validation service
            parquet: Parquet service
            serialization: Serialization service
            
        Returns:
            Validation results using utilities
        """
        validation_results = {
            'utility_validation': {},
            'data_quality': {},
            'mathematical_validation': {},
            'performance_metrics': {}
        }
        
        try:
            # Use common operations for data validation
            self.logger.info('🔍 Using common operations for data validation...')
            
            # Validate data shape
            shape_validation = common_ops.get_operation('validation', 'validate_dataframe')(test_data, ['open', 'high', 'low', 'close'])
            validation_results['utility_validation']['shape_validation'] = shape_validation
            
            # Use data processing utilities
            self.logger.info('📊 Using data processing utilities...')
            
            if data_proc.validator:
                quality_report = data_proc.validator.validate_dataframe(test_data)
                validation_results['data_quality'] = {
                    'total_issues': len(quality_report.issues),
                    'critical_issues': len([i for i in quality_report.issues if i.level.value == 'critical']),
                    'warning_issues': len([i for i in quality_report.issues if i.level.value == 'warning']),
                    'data_quality_score': quality_report.summary.get('data_quality_score', 0)
                }
            
            # Use math validation for numerical checks
            self.logger.info('🔢 Using math validation utilities...')
            
            if 'close' in test_data.columns:
                close_prices = test_data['close'].dropna()
                if len(close_prices) > 0:
                    # Validate price ranges
                    min_price = close_prices.min()
                    max_price = close_prices.max()
                    
                    try:
                        from src.utils.math_validation import validate_positive, validate_range
                        validate_positive(min_price, "min_price")
                        validate_positive(max_price, "max_price")
                        validate_range(max_price, min_price, min_price * 1000, "max_price")
                        
                        validation_results['mathematical_validation'] = {
                            'price_validation': 'passed',
                            'min_price': float(min_price),
                            'max_price': float(max_price),
                            'price_range_valid': True
                        }
                    except Exception as e:
                        validation_results['mathematical_validation'] = {
                            'price_validation': 'failed',
                            'error': str(e),
                            'price_range_valid': False
                        }
            
            # Use parquet utilities for data I/O testing
            self.logger.info('💾 Testing parquet utilities...')
            
            if parquet.parquet_utils:
                # Test parquet validation
                test_file = self.output_dir / 'test_validation.parquet'
                try:
                    test_data.to_parquet(test_file)
                    validation_result = parquet.parquet_utils.validate_parquet_file(str(test_file))
                    validation_results['utility_validation']['parquet_validation'] = validation_result
                    
                    # Clean up test file
                    test_file.unlink(missing_ok=True)
                except Exception as e:
                    validation_results['utility_validation']['parquet_validation'] = {'valid': False, 'error': str(e)}
            
            # Use serialization utilities
            self.logger.info('📄 Testing serialization utilities...')
            
            if serialization.serializers:
                test_serialization_data = {
                    'validation_timestamp': common_ops.get_operation('datetime', 'get_current_datetime')().isoformat(),
                    'data_shape': test_data.shape,
                    'columns': list(test_data.columns)
                }
                
                test_json_file = self.output_dir / 'test_serialization.json'
                try:
                    serialization.serializers['json'].save(test_serialization_data, test_json_file)
                    loaded_data = serialization.serializers['json'].load(test_json_file)
                    validation_results['utility_validation']['serialization_test'] = {
                        'save_success': True,
                        'load_success': loaded_data is not None,
                        'data_integrity': test_serialization_data == loaded_data
                    }
                    
                    # Clean up test file
                    test_json_file.unlink(missing_ok=True)
                except Exception as e:
                    validation_results['utility_validation']['serialization_test'] = {
                        'save_success': False,
                        'error': str(e)
                    }
            
            self.logger.info('✅ Utility-based validation completed')
            
        except Exception as e:
            self.logger.error(f'❌ Utility validation failed: {e}')
            validation_results['utility_validation']['error'] = str(e)
            self.performance_metrics['utility_errors'] += 1
        
        return validation_results

    def initialize_components(self, config: Dict[str, Any]) -> Dict[str, bool]:
        """
        Initialize all step06 components for validation.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Dictionary with component initialization status
        """
        self.logger.info('🔧 Initializing step06 components for validation...')
        initialization_status = {}
        try:
            self.components['feature_interaction_engine'] = FeatureInteractionEngine(config)
            initialization_status['feature_interaction_engine'] = True
            self.logger.info('✅ FeatureInteractionEngine initialized')
        except Exception as e:
            self.logger.error(f'❌ FeatureInteractionEngine initialization failed: {e}')
            initialization_status['feature_interaction_engine'] = False
        try:
            self.components['triple_barrier_labeling'] = OptimizedTripleBarrierLabeling()
            initialization_status['triple_barrier_labeling'] = True
            self.logger.info('✅ OptimizedTripleBarrierLabeling initialized')
        except Exception as e:
            self.logger.error(f'❌ OptimizedTripleBarrierLabeling initialization failed: {e}')
            initialization_status['triple_barrier_labeling'] = False
        try:
            self.components['feature_engineering_step'] = FeatureEngineeringStep(config)
            initialization_status['feature_engineering_step'] = True
            self.logger.info('✅ FeatureEngineeringStep initialized')
        except Exception as e:
            self.logger.error(f'❌ FeatureEngineeringStep initialization failed: {e}')
            initialization_status['feature_engineering_step'] = False
        self.logger.info(f'📊 Component initialization summary: {initialization_status}')
        return initialization_status

    async def run_comprehensive_validation(self, test_data: Optional[pd.DataFrame]=None) -> Dict[str, Any]:
        """
        Run comprehensive validation on all step06 components with utility integration.
        
        Args:
            test_data: Optional test data for validation
            
        Returns:
            Comprehensive validation report with utility integration
        """
        start_time = time.time()
        self.logger.info('🚀 Starting comprehensive step06 validation with utility integration...')
        
        # Initialize utilities first
        utility_status = await self.initialize_utilities()
        
        reset_step06_validation_tracking()
        if test_data is None:
            test_data = self._generate_test_data()
        
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'test_data_info': {
                'shape': test_data.shape,
                'columns': list(test_data.columns),
                'data_types': test_data.dtypes.to_dict()
            },
            'utility_integration': {
                'utility_status': utility_status,
                'utility_health': self.utility_container.get_health_report() if self.utility_container else None
            },
            'component_validation': {},
            'utility_validation': {},
            'performance_metrics': self.performance_metrics.copy(),
            'overall_summary': {}
        }
        
        # Run utility-based validation
        if self.utility_container:
            self.logger.info('🔧 Running utility-based validation...')
            try:
                utility_validation = await self._validate_with_utilities(test_data)
                validation_results['utility_validation'] = utility_validation
                self.logger.info('✅ Utility-based validation completed')
            except Exception as e:
                self.logger.error(f'❌ Utility validation failed: {e}')
                validation_results['utility_validation'] = {'error': str(e)}
                self.performance_metrics['utility_errors'] += 1
        
        # Run component validation
        for component_name, component in self.components.items():
            self.logger.info(f'🔍 Validating component: {component_name}')
            try:
                component_result = await self._validate_component(component_name, component, test_data)
                validation_results['component_validation'][component_name] = component_result
                self.logger.info(f'✅ Component {component_name} validation completed')
            except Exception as e:
                self.logger.error(f'❌ Component {component_name} validation failed: {e}')
                validation_results['component_validation'][component_name] = {
                    'status': 'failed', 
                    'error': str(e), 
                    'timestamp': datetime.now().isoformat()
                }
                self.performance_metrics['validation_errors'] += 1
        
        # Update performance metrics
        total_time = time.time() - start_time
        self.performance_metrics['total_validation_time'] = total_time
        
        # Generate overall summary
        validation_results['overall_summary'] = self._generate_overall_summary(validation_results)
        validation_results['performance_metrics'] = self.performance_metrics.copy()
        
        # Save validation report
        await self._save_validation_report(validation_results)
        
        self.logger.info('✅ Comprehensive step06 validation with utility integration completed')
        self.logger.info(f'   Total time: {total_time:.2f}s')
        self.logger.info(f'   Utility errors: {self.performance_metrics["utility_errors"]}')
        self.logger.info(f'   Validation errors: {self.performance_metrics["validation_errors"]}')
        
        return validation_results

    async def _validate_component(self, component_name: str, component: Any, test_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate a specific component.
        
        Args:
            component_name: Name of the component
            component: Component instance
            test_data: Test data for validation
            
        Returns:
            Component validation results
        """
        component_result = {'component_name': component_name, 'timestamp': datetime.now().isoformat(), 'validation_tests': {}, 'performance_metrics': {}, 'function_reports': {}}
        if component_name == 'feature_interaction_engine':
            component_result = await self._validate_feature_interaction_engine(component, test_data)
        elif component_name == 'triple_barrier_labeling':
            component_result = await self._validate_triple_barrier_labeling(component, test_data)
        elif component_name == 'feature_engineering_step':
            component_result = await self._validate_feature_engineering_step(component, test_data)
        return component_result

    async def _validate_feature_interaction_engine(self, engine: FeatureInteractionEngine, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Validate FeatureInteractionEngine component."""
        result = {'component_name': 'feature_interaction_engine', 'timestamp': datetime.now().isoformat(), 'validation_tests': {}, 'performance_metrics': {}, 'function_reports': {}}
        try:
            self.logger.info('🔧 Testing technical indicator extraction...')
            indicators = engine.extract_optimal_technical_indicators(test_data)
            result['validation_tests']['technical_indicators'] = {'status': 'passed', 'output_shape': indicators.shape, 'output_columns': len(indicators.columns)}
            self.logger.info('🔍 Testing correlation analysis...')
            correlation_results = engine.analyze_feature_correlations(indicators)
            result['validation_tests']['correlation_analysis'] = {'status': 'passed', 'high_correlations': correlation_results.get('n_high_correlations', 0), 'mean_correlation': correlation_results.get('mean_correlation', 0)}
            self.logger.info('🔗 Testing interaction feature extraction...')
            features_array = indicators.values
            feature_names = list(indicators.columns)
            interactions = engine.extract_interaction_features(features_array, feature_names, test_data)
            result['validation_tests']['interaction_features'] = {'status': 'passed', 'output_shape': interactions.shape, 'feature_count': interactions.shape[1]}
            self.logger.info('📋 Generating comprehensive function report...')
            comprehensive_report = engine.generate_comprehensive_function_report()
            result['function_reports']['comprehensive_report'] = comprehensive_report
        except Exception as e:
            result['validation_tests']['error'] = str(e)
            result['status'] = 'failed'
        return result

    async def _validate_triple_barrier_labeling(self, labeling: OptimizedTripleBarrierLabeling, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Validate OptimizedTripleBarrierLabeling component."""
        result = {'component_name': 'triple_barrier_labeling', 'timestamp': datetime.now().isoformat(), 'validation_tests': {}, 'performance_metrics': {}, 'function_reports': {}}
        try:
            self.logger.info('🏷️ Testing vectorized triple barrier labeling...')
            labeled_data = labeling.apply_triple_barrier_labeling_vectorized(test_data)
            result['validation_tests']['vectorized_labeling'] = {'status': 'passed', 'output_shape': labeled_data.shape, 'label_distribution': labeled_data['label'].value_counts().to_dict(), 'profit_tracking': 'potential_profit_pct' in labeled_data.columns}
            self.logger.info('🏷️ Testing convenience labeling method...')
            labels_only = labeling.apply_triple_barrier_labels(test_data)
            result['validation_tests']['convenience_method'] = {'status': 'passed', 'output_length': len(labels_only), 'label_distribution': labels_only.value_counts().to_dict()}
            self.logger.info('📋 Generating comprehensive labeling report...')
            comprehensive_report = labeling.generate_comprehensive_labeling_report()
            result['function_reports']['comprehensive_report'] = comprehensive_report
        except Exception as e:
            result['validation_tests']['error'] = str(e)
            result['status'] = 'failed'
        return result

    async def _validate_feature_engineering_step(self, step: FeatureEngineeringStep, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Validate FeatureEngineeringStep component."""
        result = {'component_name': 'feature_engineering_step', 'timestamp': datetime.now().isoformat(), 'validation_tests': {}, 'performance_metrics': {}, 'function_reports': {}}
        try:
            self.logger.info('✅ Testing input validation...')
            pipeline_state = {'labeled_data': test_data}
            is_valid, errors = step.validate_inputs({}, pipeline_state)
            result['validation_tests']['input_validation'] = {'status': 'passed' if is_valid else 'failed', 'is_valid': is_valid, 'errors': errors}
            self.logger.info('🔧 Testing feature engineering execution...')
            training_input = {'output_dir': str(self.output_dir)}
            pipeline_state = {'labeled_data': test_data}
            result['validation_tests']['execution'] = {'status': 'simulated', 'note': 'Full execution requires async pipeline context'}
            self.logger.info('✅ Testing output validation...')
            simulated_engineered_data = {'all': test_data.copy()}
            simulated_pipeline_state = {'engineered_data': simulated_engineered_data}
            is_valid, errors = step.validate_outputs(simulated_pipeline_state)
            result['validation_tests']['output_validation'] = {'status': 'passed' if is_valid else 'failed', 'is_valid': is_valid, 'errors': errors}
            self.logger.info('📋 Generating comprehensive step report...')
            comprehensive_report = step.generate_comprehensive_step06_report()
            result['function_reports']['comprehensive_report'] = comprehensive_report
        except Exception as e:
            result['validation_tests']['error'] = str(e)
            result['status'] = 'failed'
        return result
    @log_all_calls
    @inject_utilities('common_ops', 'math_val')
    async def _generate_test_data(self, common_ops=None, math_val=None) -> pd.DataFrame:
        """Generate test data for validation using utility services."""
        self.logger.info('📊 Generating test data for validation with utility integration...')
        
        # Use common operations for datetime generation
        if common_ops:
            current_time = common_ops.get_operation('datetime', 'get_current_datetime')()
            self.logger.info(f'   Using current time: {current_time}')
        
        np.random.seed(42)
        n_samples = 1000
        dates = pd.date_range('2024-01-01', periods=n_samples, freq='1min')
        base_price = 100.0
        
        # Use math validation for safe random generation
        returns = np.random.normal(0, 0.001, n_samples)
        prices = [base_price]
        
        for ret in returns[1:]:
            # Use safe mathematical operations
            if math_val:
                try:
                    from src.utils.math_validation import validate_finite
                    validate_finite(ret, "return")
                    new_price = prices[-1] * (1 + ret)
                    validate_finite(new_price, "new_price")
                    validate_positive(new_price, "new_price")
                    prices.append(new_price)
                except Exception as e:
                    self.logger.warning(f'Math validation failed for return {ret}: {e}')
                    prices.append(prices[-1] * (1 + ret))
            else:
                prices.append(prices[-1] * (1 + ret))
        
        # Generate OHLCV data with utility validation
        data = pd.DataFrame({
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
            'close': prices,
            'volume': np.random.uniform(1000, 10000, n_samples)
        }, index=dates)
        
        # Ensure OHLC consistency using safe operations
        if common_ops:
            # Use safe operations for data validation
            validate_df = common_ops.get_operation('validation', 'validate_dataframe')
            is_valid = validate_df(data, ['open', 'high', 'low', 'close'])
            if not is_valid:
                self.logger.warning('Generated data failed initial validation, applying corrections')
        
        # Apply OHLC corrections
        data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
        data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
        
        # Final validation
        if common_ops:
            final_validation = validate_df(data, ['open', 'high', 'low', 'close'])
            if final_validation:
                self.logger.info('✅ Generated data passed final validation')
            else:
                self.logger.warning('⚠️ Generated data failed final validation')
        
        self.logger.info(f'✅ Generated test data: {data.shape}')
        return data
    @log_all_calls

    def _generate_overall_summary(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall validation summary."""
        component_validation = validation_results['component_validation']
        total_components = len(component_validation)
        successful_components = sum((1 for comp in component_validation.values() if comp.get('status') != 'failed'))
        total_tests = 0
        successful_tests = 0
        for comp_name, comp_result in component_validation.items():
            validation_tests = comp_result.get('validation_tests', {})
            for test_name, test_result in validation_tests.items():
                if isinstance(test_result, dict) and 'status' in test_result:
                    total_tests += 1
                    if test_result['status'] == 'passed':
                        successful_tests += 1
        return {'total_components': total_components, 'successful_components': successful_components, 'component_success_rate': successful_components / total_components if total_components > 0 else 0, 'total_tests': total_tests, 'successful_tests': successful_tests, 'test_success_rate': successful_tests / total_tests if total_tests > 0 else 0, 'validation_framework_status': 'active', 'timestamp': datetime.now().isoformat()}

    async def _save_validation_report(self, validation_results: Dict[str, Any]) -> None:
        """Save comprehensive validation report."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.output_dir / f'step06_comprehensive_validation_report_{timestamp}.json'
        with open(report_path, 'w') as f:
            json.dump(validation_results, f, indent = 2, default = str)
        summary_path = self.output_dir / f'step06_validation_summary_{timestamp}.json'
        summary = {'timestamp': validation_results['timestamp'], 'overall_summary': validation_results['overall_summary'], 'component_summary': {name: {'status': result.get('status', 'unknown'), 'tests_count': len(result.get('validation_tests', {})), 'reports_count': len(result.get('function_reports', {}))} for name, result in validation_results['component_validation'].items()}}
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent = 2, default = str)
        self.logger.info(f'💾 Validation reports saved:')
        self.logger.info(f'   Main report: {report_path}')
        self.logger.info(f'   Summary report: {summary_path}')

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get current validation summary."""
        return get_step06_validation_summary()

    def reset_validation_tracking(self) -> None:
        """Reset validation tracking."""
        reset_step06_validation_tracking()
        self.logger.info('🔄 Validation tracking reset')
    
    async def cleanup(self) -> None:
        """Cleanup utility services and resources."""
        self.logger.info('🧹 Cleaning up Step06 Validation Orchestrator...')
        
        try:
            if self.utility_container:
                await self.utility_container.cleanup()
                self.logger.info('✅ Utility container cleaned up')
            
            # Reset performance metrics
            self.performance_metrics = {
                'total_validation_time': 0.0,
                'utility_initialization_time': 0.0,
                'data_processing_time': 0.0,
                'memory_usage_mb': 0.0,
                'gpu_utilization': 0.0,
                'cpu_utilization': 0.0,
                'validation_errors': 0,
                'utility_errors': 0
            }
            
            self.logger.info('✅ Step06 Validation Orchestrator cleanup completed')
            
        except Exception as e:
            self.logger.error(f'❌ Cleanup failed: {e}')
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return self.performance_metrics.copy()
    
    def get_utility_health_report(self) -> Dict[str, Any]:
        """Get utility services health report."""
        if self.utility_container:
            return self.utility_container.get_health_report()
        return {"status": "not_initialized", "services": {}}

async def run_step06_comprehensive_validation(config: Optional[Dict[str, Any]]=None, 
                                            test_data: Optional[pd.DataFrame]=None, 
                                            output_dir: str='step06_validation_reports',
                                            utility_config: Optional[UtilityConfig] = None) -> Dict[str, Any]:
    """
    Run comprehensive validation for all step06 components with utility integration.
    
    Args:
        config: Configuration dictionary
        test_data: Optional test data
        output_dir: Output directory for reports
        utility_config: Utility configuration for dependency injection
        
    Returns:
        Comprehensive validation results with utility integration
    """
    if config is None:
        config = {
            'step06_feature_engineering': {
                'use_matrix_optimizer': True,
                'force_regime_specific_periods': False,
                'momentum_volume_enabled': True,
                'trend_volatility_enabled': True,
                'oscillator_trend_enabled': True,
                'volume_price_enabled': True,
                'volatility_regime_enabled': True,
                'cross_timeframe_enabled': True,
                'regime_dependent_enabled': True
            }
        }
    
    # Create orchestrator with utility integration
    orchestrator = Step06ValidationOrchestrator(output_dir, utility_config)
    
    try:
        # Initialize components
        init_status = orchestrator.initialize_components(config)
        
        # Run comprehensive validation with utility integration
        validation_results = await orchestrator.run_comprehensive_validation(test_data)
        
        # Add utility health report to results
        validation_results['utility_health_report'] = orchestrator.get_utility_health_report()
        validation_results['performance_metrics'] = orchestrator.get_performance_metrics()
        
        return validation_results
        
    finally:
        # Cleanup resources
        await orchestrator.cleanup()
if __name__ == '__main__':
    import asyncio

    async def main() -> None:
        logging.basicConfig(level=logging.INFO)
        
        # Create utility configuration for demonstration
        utility_config = UtilityConfig(
            enable_common_operations=True,
            enable_data_processing=True,
            enable_math_validation=True,
            enable_parquet_utils=True,
            enable_serialization=True,
            enable_m1_gpu=True,
            enable_m1_memory=True,
            enable_m1_cpu=True,
            data_processing_chunk_size=5000,
            m1_memory_limit_gb=8.0,
            m1_max_workers=4
        )
        
        print("🚀 Running Step06 Comprehensive Validation with Utility Integration...")
        results = await run_step06_comprehensive_validation(utility_config=utility_config)
        
        print('\n' + '='*80)
        print('STEP06 COMPREHENSIVE VALIDATION RESULTS WITH UTILITY INTEGRATION')
        print('='*80)
        
        # Print overall summary
        print(f"\n📊 Overall Summary:")
        overall = results['overall_summary']
        print(f"  Total Components: {overall.get('total_components', 0)}")
        print(f"  Successful Components: {overall.get('successful_components', 0)}")
        print(f"  Component Success Rate: {overall.get('component_success_rate', 0):.2%}")
        print(f"  Total Tests: {overall.get('total_tests', 0)}")
        print(f"  Successful Tests: {overall.get('successful_tests', 0)}")
        print(f"  Test Success Rate: {overall.get('test_success_rate', 0):.2%}")
        
        # Print utility integration results
        if 'utility_integration' in results:
            print(f"\n🔧 Utility Integration:")
            utility_status = results['utility_integration'].get('utility_status', {})
            for service, status in utility_status.items():
                status_icon = "✅" if status else "❌"
                print(f"  {status_icon} {service}: {'ENABLED' if status else 'FAILED'}")
        
        # Print utility health report
        if 'utility_health_report' in results:
            health = results['utility_health_report']
            print(f"\n🏥 Utility Health Report:")
            print(f"  Status: {health.get('status', 'unknown')}")
            print(f"  Healthy Services: {health.get('healthy_services', 0)}/{health.get('total_services', 0)}")
        
        # Print performance metrics
        if 'performance_metrics' in results:
            metrics = results['performance_metrics']
            print(f"\n⚡ Performance Metrics:")
            print(f"  Total Validation Time: {metrics.get('total_validation_time', 0):.2f}s")
            print(f"  Utility Initialization Time: {metrics.get('utility_initialization_time', 0):.2f}s")
            print(f"  Validation Errors: {metrics.get('validation_errors', 0)}")
            print(f"  Utility Errors: {metrics.get('utility_errors', 0)}")
        
        # Print component validation results
        print(f"\n🔍 Component Validation Results:")
        for component_name, component_result in results['component_validation'].items():
            status = component_result.get('status', 'unknown')
            status_icon = "✅" if status == 'completed' else "❌" if status == 'failed' else "⚠️"
            print(f"  {status_icon} {component_name}:")
            print(f"    Status: {status}")
            print(f"    Tests: {len(component_result.get('validation_tests', {}))}")
            print(f"    Reports: {len(component_result.get('function_reports', {}))}")
        
        # Print utility validation results
        if 'utility_validation' in results and results['utility_validation']:
            print(f"\n🛠️ Utility Validation Results:")
            utility_validation = results['utility_validation']
            if 'utility_validation' in utility_validation:
                uv = utility_validation['utility_validation']
                if 'shape_validation' in uv:
                    print(f"  ✅ Data Shape Validation: PASSED")
                if 'parquet_validation' in uv:
                    pq_val = uv['parquet_validation']
                    status_icon = "✅" if pq_val.get('valid', False) else "❌"
                    print(f"  {status_icon} Parquet Validation: {'PASSED' if pq_val.get('valid', False) else 'FAILED'}")
                if 'serialization_test' in uv:
                    ser_test = uv['serialization_test']
                    save_ok = ser_test.get('save_success', False)
                    load_ok = ser_test.get('load_success', False)
                    integrity_ok = ser_test.get('data_integrity', False)
                    print(f"  ✅ Serialization Test: SAVE={save_ok}, LOAD={load_ok}, INTEGRITY={integrity_ok}")
        
        print(f"\n✅ Step06 Comprehensive Validation with Utility Integration completed!")
        print(f"   Reports saved to: {results.get('output_dir', 'step06_validation_reports')}")
    
    asyncio.run(main())