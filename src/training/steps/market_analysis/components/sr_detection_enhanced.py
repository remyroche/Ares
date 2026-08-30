"""
Enhanced SR Detection Step with Advanced ML Integration

This module integrates the enhanced SR detection system with the existing training pipeline,
providing seamless integration with VectorBT, HPO, SHAP/LIME, and advanced validation.

Author: AI Assistant
Date: 2024
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

# Core imports
from src.core.base_step import BaseStep
from src.core.decorators import handles_errors, traced
from src.utils.logger import system_logger

# Enhanced SR detection imports
from src.tactician.sr_levels.enhanced_sr_detection_optimized import (
    EnhancedSROptimizedDetector, SROptimizationConfig, SRLevel
)

# VectorBT and optimization imports
try:
    from src.utils.ml_common.unified_vectorization_manager import (
        UnifiedVectorizationManager, OperationType, OptimizationStrategy
    )
    VECTORIZATION_AVAILABLE = True
except ImportError:
    VECTORIZATION_AVAILABLE = False

# Hardware optimization imports
try:
    from src.utils.hardware.unified_hardware_manager import (
        UnifiedHardwareManager, WorkloadType, OptimizationLevel
    )
    HARDWARE_OPTIMIZATION_AVAILABLE = True
except ImportError:
    HARDWARE_OPTIMIZATION_AVAILABLE = False

# ML explainability imports
try:
    from src.utils.ml_common.explainability.shap_lime_integration import (
        SHAPLIMEExplainer, ExplanationConfig
    )
    EXPLAINABILITY_AVAILABLE = True
except ImportError:
    EXPLAINABILITY_AVAILABLE = False

# Validation imports
try:
    from src.utils.ml_common.validation.temporal_cross_validation import temporal_cross_validation
    from src.utils.ml_common.validation.data_leakage_detector import DataLeakageDetector
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False

# HPO imports
try:
    from src.utils.ml_common.optimization.bayesian_tpe_optimizer import BayesianTPEOptimizer
    from src.utils.ml_common.optimization.hpo_utils import HPOConfig
    HPO_AVAILABLE = True
except ImportError:
    HPO_AVAILABLE = False

logger = system_logger.getChild('EnhancedSRDetectionStep')

class EnhancedSRDetectionStep(BaseStep):
    """
    Enhanced SR Detection Step with Advanced ML Integration.
    
    This step integrates:
    - VectorBT optimization for efficient time series operations
    - SHAP/LIME explainability for SR level significance
    - Advanced validation with temporal CV and data leakage detection
    - HPO integration for parameter optimization
    - Hardware optimization for M1 Mac performance
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the enhanced SR detection step."""
        super().__init__(config)
        self.logger = logger.getChild('EnhancedSRDetectionStep')
        
        # Initialize optimization components
        self._initialize_optimization_components()
        
        # Performance tracking
        self.performance_metrics = {
            'step_execution_time': 0.0,
            'sr_levels_detected': 0,
            'optimization_gains': {},
            'quality_metrics': {},
            'hardware_utilization': {}
        }
        
        self.logger.info("✅ Enhanced SR Detection Step initialized")
    
    def _initialize_optimization_components(self):
        """Initialize all optimization components."""
        # VectorBT optimization
        if VECTORIZATION_AVAILABLE:
            try:
                self.vectorization_manager = UnifiedVectorizationManager()
                self.logger.info("✅ VectorBT optimization initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ VectorBT initialization failed: {e}")
                self.vectorization_manager = None
        else:
            self.vectorization_manager = None
        
        # Hardware optimization
        if HARDWARE_OPTIMIZATION_AVAILABLE:
            try:
                self.hardware_manager = UnifiedHardwareManager()
                self.hardware_manager.initialize()
                self.logger.info("✅ Hardware optimization initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Hardware optimization failed: {e}")
                self.hardware_manager = None
        else:
            self.hardware_manager = None
        
        # ML explainability
        if EXPLAINABILITY_AVAILABLE:
            try:
                explanation_config = ExplanationConfig(
                    enable_shap=True,
                    enable_lime=True,
                    shap_sample_size=100,
                    lime_sample_size=1000,
                    parallel_explanations=True
                )
                self.explainer = SHAPLIMEExplainer(explanation_config)
                self.logger.info("✅ ML explainability initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Explainability initialization failed: {e}")
                self.explainer = None
        else:
            self.explainer = None
        
        # Validation components
        if VALIDATION_AVAILABLE:
            try:
                self.leakage_detector = DataLeakageDetector()
                self.logger.info("✅ Validation components initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Validation initialization failed: {e}")
                self.leakage_detector = None
        else:
            self.leakage_detector = None
        
        # HPO components
        if HPO_AVAILABLE:
            try:
                hpo_config = HPOConfig(
                    n_trials=50,
                    timeout=300,
                    direction='maximize'
                )
                self.hpo_optimizer = BayesianTPEOptimizer(hpo_config)
                self.logger.info("✅ HPO optimization initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ HPO initialization failed: {e}")
                self.hpo_optimizer = None
        else:
            self.hpo_optimizer = None
    
    @handles_errors(exceptions=(ValueError, AttributeError), default_return={}, context='execute enhanced SR detection step')
    async def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the enhanced SR detection step.
        
        Args:
            data: Input data containing market data and configuration
            
        Returns:
            Dictionary containing detected SR levels and metadata
        """
        self.logger.info("🚀 Starting enhanced SR detection step execution")
        start_time = time.time()
        
        try:
            # Extract market data
            market_data = self._extract_market_data(data)
            if market_data is None or market_data.empty:
                self.logger.warning("⚠️ No market data available for SR detection")
                return {'sr_levels': [], 'metadata': {}}
            
            # Optimize hardware for workload
            if self.hardware_manager:
                self.hardware_manager.optimize_for_workload(
                    WorkloadType.ML_TRAINING, 
                    OptimizationLevel.BALANCED
                )
            
            # Create SR detection configuration
            sr_config = self._create_sr_config(data)
            
            # Initialize enhanced SR detector
            sr_detector = EnhancedSROptimizedDetector(sr_config)
            
            # Detect SR levels
            sr_levels = sr_detector.detect_sr_levels(market_data)
            
            # Process and enhance results
            processed_levels = self._process_sr_levels(sr_levels, market_data)
            
            # Generate metadata and performance metrics
            metadata = self._generate_metadata(sr_detector, processed_levels, market_data)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            self.performance_metrics['step_execution_time'] = execution_time
            self.performance_metrics['sr_levels_detected'] = len(processed_levels)
            
            self.logger.info(f"✅ Enhanced SR detection completed: {len(processed_levels)} levels in {execution_time:.3f}s")
            
            return {
                'sr_levels': processed_levels,
                'metadata': metadata,
                'performance_metrics': self.performance_metrics
            }
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced SR detection step failed: {e}")
            return {'sr_levels': [], 'metadata': {}, 'error': str(e)}
    
    def _extract_market_data(self, data: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Extract market data from input data."""
        try:
            # Try different possible keys for market data
            market_data_keys = ['market_data', 'data', 'df', 'ohlcv', 'price_data']
            
            for key in market_data_keys:
                if key in data and isinstance(data[key], pd.DataFrame):
                    market_data = data[key]
                    
                    # Validate required columns
                    required_cols = ['open', 'high', 'low', 'close', 'volume']
                    if all(col in market_data.columns for col in required_cols):
                        return market_data
                    else:
                        self.logger.warning(f"Market data missing required columns: {required_cols}")
                        continue
            
            self.logger.warning("No valid market data found in input")
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to extract market data: {e}")
            return None
    
    def _create_sr_config(self, data: Dict[str, Any]) -> SROptimizationConfig:
        """Create SR detection configuration from input data."""
        try:
            # Extract configuration from input data
            config_data = data.get('config', {})
            sr_config_data = config_data.get('sr_detection', {})
            
            # Create configuration with defaults
            sr_config = SROptimizationConfig(
                min_touches=sr_config_data.get('min_touches', 2),
                tolerance_pct=sr_config_data.get('tolerance_pct', 0.5),
                lookback_periods=sr_config_data.get('lookback_periods', 100),
                min_r_squared=sr_config_data.get('min_r_squared', 0.7),
                min_quality_score=sr_config_data.get('min_quality_score', 0.6),
                min_consistency=sr_config_data.get('min_consistency', 0.5),
                enable_vectorbt=sr_config_data.get('enable_vectorbt', True),
                enable_hardware_optimization=sr_config_data.get('enable_hardware_optimization', True),
                enable_explainability=sr_config_data.get('enable_explainability', True),
                enable_validation=sr_config_data.get('enable_validation', True),
                enable_hpo=sr_config_data.get('enable_hpo', True),
                max_candidates=sr_config_data.get('max_candidates', 1000),
                batch_size=sr_config_data.get('batch_size', 100),
                parallel_workers=sr_config_data.get('parallel_workers', 4),
                hpo_trials=sr_config_data.get('hpo_trials', 50),
                hpo_timeout=sr_config_data.get('hpo_timeout', 300),
                cv_folds=sr_config_data.get('cv_folds', 5),
                gap_periods=sr_config_data.get('gap_periods', 10)
            )
            
            return sr_config
            
        except Exception as e:
            self.logger.warning(f"Failed to create SR config, using defaults: {e}")
            return SROptimizationConfig()
    
    def _process_sr_levels(self, sr_levels: List[SRLevel], market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Process SR levels for output format."""
        processed_levels = []
        
        for level in sr_levels:
            try:
                # Convert SRLevel to dictionary
                level_dict = {
                    'price': float(level.price),
                    'level_type': level.level_type,
                    'strength': float(level.strength),
                    'touches': int(level.touches),
                    'first_touch': level.first_touch.isoformat() if hasattr(level.first_touch, 'isoformat') else str(level.first_touch),
                    'last_touch': level.last_touch.isoformat() if hasattr(level.last_touch, 'isoformat') else str(level.last_touch),
                    'quality_score': float(level.quality_score),
                    'r_squared': float(level.r_squared),
                    'consistency': float(level.consistency),
                    'volatility': float(level.volatility),
                    'volume_profile': float(level.volume_profile),
                    'confidence': float(level.confidence),
                    'feature_importance': level.feature_importance,
                    'shap_values': level.shap_values,
                    'lime_explanation': level.lime_explanation,
                    'validation_score': float(level.validation_score),
                    'data_leakage_risk': float(level.data_leakage_risk),
                    'temporal_stability': float(level.temporal_stability)
                }
                
                processed_levels.append(level_dict)
                
            except Exception as e:
                self.logger.warning(f"Failed to process SR level {level.price}: {e}")
                continue
        
        return processed_levels
    
    def _generate_metadata(self, sr_detector: EnhancedSROptimizedDetector, 
                          sr_levels: List[Dict[str, Any]], 
                          market_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate metadata for the SR detection results."""
        try:
            # Get detector performance metrics
            detector_metrics = sr_detector.get_performance_metrics()
            optimization_status = sr_detector.get_optimization_status()
            
            # Calculate summary statistics
            if sr_levels:
                prices = [level['price'] for level in sr_levels]
                strengths = [level['strength'] for level in sr_levels]
                quality_scores = [level['quality_score'] for level in sr_levels]
                
                summary_stats = {
                    'total_levels': len(sr_levels),
                    'support_levels': len([l for l in sr_levels if l['level_type'] == 'support']),
                    'resistance_levels': len([l for l in sr_levels if l['level_type'] == 'resistance']),
                    'avg_strength': float(np.mean(strengths)),
                    'avg_quality_score': float(np.mean(quality_scores)),
                    'price_range': {
                        'min': float(np.min(prices)),
                        'max': float(np.max(prices)),
                        'mean': float(np.mean(prices))
                    }
                }
            else:
                summary_stats = {
                    'total_levels': 0,
                    'support_levels': 0,
                    'resistance_levels': 0,
                    'avg_strength': 0.0,
                    'avg_quality_score': 0.0,
                    'price_range': {'min': 0.0, 'max': 0.0, 'mean': 0.0}
                }
            
            # Generate metadata
            metadata = {
                'detection_timestamp': pd.Timestamp.now().isoformat(),
                'market_data_info': {
                    'rows': len(market_data),
                    'columns': list(market_data.columns),
                    'date_range': {
                        'start': market_data.index[0].isoformat() if len(market_data) > 0 else None,
                        'end': market_data.index[-1].isoformat() if len(market_data) > 0 else None
                    }
                },
                'summary_statistics': summary_stats,
                'detector_performance': detector_metrics,
                'optimization_status': optimization_status,
                'step_performance': self.performance_metrics
            }
            
            return metadata
            
        except Exception as e:
            self.logger.warning(f"Failed to generate metadata: {e}")
            return {
                'detection_timestamp': pd.Timestamp.now().isoformat(),
                'error': str(e)
            }
    
    def get_step_info(self) -> Dict[str, Any]:
        """Get information about this step."""
        return {
            'step_name': 'EnhancedSRDetectionStep',
            'description': 'Enhanced SR detection with advanced ML integration',
            'version': '2.0.0',
            'dependencies': [
                'VectorBT optimization',
                'Hardware optimization',
                'ML explainability (SHAP/LIME)',
                'Advanced validation',
                'HPO optimization'
            ],
            'optimization_available': {
                'vectorization': VECTORIZATION_AVAILABLE,
                'hardware': HARDWARE_OPTIMIZATION_AVAILABLE,
                'explainability': EXPLAINABILITY_AVAILABLE,
                'validation': VALIDATION_AVAILABLE,
                'hpo': HPO_AVAILABLE
            }
        }
    
    def validate_input(self, data: Dict[str, Any]) -> bool:
        """Validate input data for the step."""
        try:
            # Check if market data is present
            market_data = self._extract_market_data(data)
            if market_data is None or market_data.empty:
                return False
            
            # Check required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in market_data.columns for col in required_cols):
                return False
            
            # Check data types
            for col in required_cols:
                if not pd.api.types.is_numeric_dtype(market_data[col]):
                    return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Input validation failed: {e}")
            return False
    
    def get_required_inputs(self) -> List[str]:
        """Get list of required input keys."""
        return ['market_data', 'config']
    
    def get_output_keys(self) -> List[str]:
        """Get list of output keys."""
        return ['sr_levels', 'metadata', 'performance_metrics']

# Convenience functions
def create_enhanced_sr_detection_step(config: Optional[Dict[str, Any]] = None) -> EnhancedSRDetectionStep:
    """Create an enhanced SR detection step instance."""
    return EnhancedSRDetectionStep(config)

# Example usage
if __name__ == "__main__":
    # Create sample market data
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='15T')
    np.random.seed(42)
    
    base_price = 2000.0
    returns = np.random.normal(0, 0.001, len(dates))
    prices = base_price * np.exp(np.cumsum(returns))
    
    market_data = pd.DataFrame({
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, len(dates)))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, len(dates)))),
        'close': prices,
        'volume': np.random.uniform(1000, 10000, len(dates))
    }, index=dates)
    
    # Create step and execute
    step = create_enhanced_sr_detection_step()
    
    # Prepare input data
    input_data = {
        'market_data': market_data,
        'config': {
            'sr_detection': {
                'min_touches': 2,
                'tolerance_pct': 0.5,
                'enable_vectorbt': True,
                'enable_explainability': True
            }
        }
    }
    
    # Execute step
    result = asyncio.run(step.execute(input_data))
    
    print(f"Detected {len(result['sr_levels'])} SR levels")
    print(f"Metadata: {result['metadata']}")
    print(f"Performance: {result['performance_metrics']}")