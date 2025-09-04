#!/usr/bin/env python3
"""Enhanced Optimization Package for Trading Pipeline.

This package contains all the enhanced components for optimization with comprehensive protection:
- Enhanced confidence calibration with validation and error handling
- Enhanced parameter optimization with monitoring and alerting
- Comprehensive pipeline orchestrator with state management
- Data protection and integrity validation
- Performance monitoring and alerting system
- Integration tests and validation framework
"""

# Legacy imports for backward compatibility
from .step16_confidence_calibration_per_regime import ConfidenceCalibrationPerRegimeStep
from .step16_confidence_calibration_validator import ConfidenceCalibrationValidator
from .step17_final_parameters_optimization_new import FinalParametersOptimizationStep
from .step17_final_parameters_optimization_per_regime import PerRegimeFinalParametersOptimizationStep
from .step17_final_parameters_optimization_validator import FinalParametersOptimizationValidator
from .step17_parameter_optimization_wrapper import ParameterOptimizationWrapper

# Enhanced components
from .enhanced_confidence_calibration import EnhancedConfidenceCalibrationStep
from .enhanced_parameter_optimization import EnhancedParameterOptimizationStep
from .optimisation_pipeline_orchestrator import (
    OptimisationPipelineOrchestrator,
    run_optimisation_pipeline
)
from .optimisation_pipeline_validator import (
    OptimisationPipelineValidator,
    ConfidenceCalibrationValidator as EnhancedConfidenceCalibrationValidator,
    ParameterOptimizationValidator as EnhancedParameterOptimizationValidator
)
from .optimisation_decorators import (
    protect_optimisation_operation,
    protect_data_operation,
    data_protection,
    error_handling,
    performance_monitoring,
    operation_logging
)
from .optimisation_utilities import (
    initialize_optimisation_utilities,
    get_data_formatting_utils,
    get_analysis_operations_utils,
    get_data_access_control,
    get_pipeline_state_manager,
    get_performance_optimizer
)
from .optimisation_monitoring_system import (
    initialize_monitoring_system,
    get_monitoring_system,
    AlertSeverity,
    MetricType
)

# Legacy pipeline function for backward compatibility
async def run_legacy_optimisation_pipeline(symbol, exchange, timeframe, data_dir, **config):
    """Run the legacy optimization pipeline (for backward compatibility)."""
    try:
        # Step 1: Confidence Calibration (if enabled)
        if config.get('confidence_calibration', True):
            confidence_calibrator = ConfidenceCalibrationPerRegimeStep()
            await confidence_calibrator.calibrate_confidence(symbol, exchange, timeframe, data_dir)
        
        # Step 2: Final Parameters Optimization (if enabled)
        if config.get('parameter_optimization', True):
            param_optimizer = FinalParametersOptimizationStep()
            await param_optimizer.optimize_parameters(symbol, exchange, timeframe, data_dir)
        
        return True
        
    except Exception as e:
        print(f"Legacy optimization pipeline failed: {e}")
        return False

# Enhanced pipeline function (recommended)
async def run_enhanced_optimisation_pipeline(symbol, exchange, timeframe, data_dir, **config):
    """Run the enhanced optimization pipeline with comprehensive protection."""
    try:
        # Initialize enhanced components
        enhanced_config = {
            "data_dir": data_dir,
            "symbol": symbol,
            "exchange": exchange,
            "timeframe": timeframe,
            **config
        }
        
        # Use the enhanced orchestrator
        result = await run_optimisation_pipeline(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            data_dir=data_dir,
            config=enhanced_config
        )
        
        return result.get("overall_success", False)
        
    except Exception as e:
        print(f"Enhanced optimization pipeline failed: {e}")
        return False

# Main pipeline function (uses enhanced version by default)
async def run_optimisation_pipeline(symbol, exchange, timeframe, data_dir, **config):
    """Run the complete optimization pipeline (enhanced version by default)."""
    # Use enhanced pipeline by default
    return await run_enhanced_optimisation_pipeline(symbol, exchange, timeframe, data_dir, **config)

__all__ = [
    # Legacy components
    'ConfidenceCalibrationPerRegimeStep',
    'ConfidenceCalibrationValidator',
    'FinalParametersOptimizationStep',
    'PerRegimeFinalParametersOptimizationStep',
    'FinalParametersOptimizationValidator',
    'ParameterOptimizationWrapper',
    'run_legacy_optimisation_pipeline',
    
    # Enhanced components
    'EnhancedConfidenceCalibrationStep',
    'EnhancedParameterOptimizationStep',
    'OptimisationPipelineOrchestrator',
    'OptimisationPipelineValidator',
    'EnhancedConfidenceCalibrationValidator',
    'EnhancedParameterOptimizationValidator',
    
    # Decorators
    'protect_optimisation_operation',
    'protect_data_operation',
    'data_protection',
    'error_handling',
    'performance_monitoring',
    'operation_logging',
    
    # Utilities
    'initialize_optimisation_utilities',
    'get_data_formatting_utils',
    'get_analysis_operations_utils',
    'get_data_access_control',
    'get_pipeline_state_manager',
    'get_performance_optimizer',
    
    # Monitoring
    'initialize_monitoring_system',
    'get_monitoring_system',
    'AlertSeverity',
    'MetricType',
    
    # Main functions
    'run_optimisation_pipeline',
    'run_enhanced_optimisation_pipeline'
]