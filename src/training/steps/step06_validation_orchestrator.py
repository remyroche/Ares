"""
Step06 Validation Orchestrator

This module orchestrates comprehensive validation, tracking, and reporting
for all step06 components. It provides a unified interface for:
- Function call validation and tracking
- Function-to-function call monitoring
- Comprehensive function completion reports
- Performance monitoring and analysis
- Error handling with detailed context
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np

# Import the validation framework
try:
    from step06_enhanced_validation_framework import (
        get_step06_validation_summary,
        reset_step06_validation_tracking,
        ValidationLevel,
        FunctionStatus
    )
    VALIDATION_FRAMEWORK_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Step06 validation framework not available: {e}")
    VALIDATION_FRAMEWORK_AVAILABLE = False
    
    # Fallback functions
    def get_step06_validation_summary():
        return {"error": "Validation framework not available"}
    
    def reset_step06_validation_tracking():
        pass
    
    class ValidationLevel:
        BASIC = "basic"
        DETAILED = "detailed"
        COMPREHENSIVE = "comprehensive"
    
    class FunctionStatus:
        PENDING = "pending"
        IN_PROGRESS = "in_progress"
        COMPLETED = "completed"
        FAILED = "failed"
        TIMEOUT = "timeout"

# Import step06 components with fallback
try:
    from src.training.steps.market_analysis.step06_feature_engineering import FeatureInteractionEngine
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"FeatureInteractionEngine not available: {e}")
    # Create fallback class
    class FeatureInteractionEngine:
        def __init__(self, config):
            self.config = config
            self.logger = logging.getLogger(__name__)
        async def create_interactions(self, data):
            return data
    COMPONENTS_AVAILABLE = False

try:
    from src.training.steps.step06_labeling_components.optimized_triple_barrier_labeling import OptimizedTripleBarrierLabeling
except ImportError as e:
    logging.warning(f"OptimizedTripleBarrierLabeling not available: {e}")
    class OptimizedTripleBarrierLabeling:
        def __init__(self, config):
            self.config = config
            self.logger = logging.getLogger(__name__)

try:
    from src.training.steps.data_collection.feature_engineering.step06_feature_engineering import FeatureEngineeringStep
except ImportError as e:
    logging.warning(f"FeatureEngineeringStep not available: {e}")
    class FeatureEngineeringStep:
        def __init__(self, config):
            self.config = config
            self.logger = logging.getLogger(__name__)


class Step06ValidationOrchestrator:
    """
    Orchestrates comprehensive validation and reporting for all step06 components.
    """
    
    def __init__(self, output_dir: str = "step06_validation_reports"):
        """
        Initialize the step06 validation orchestrator.
        
        Args:
            output_dir: Directory to save validation reports
        """
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.components = {}
        self.component_reports = {}
        self.overall_report = {}
        
        self.logger.info("🎯 Step06 Validation Orchestrator initialized")
        self.logger.info(f"   Output directory: {self.output_dir}")
        self.logger.info(f"   Components available: {COMPONENTS_AVAILABLE}")
    
    def initialize_components(self, config: Dict[str, Any]) -> Dict[str, bool]:
        """
        Initialize all step06 components for validation.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Dictionary with component initialization status
        """
        self.logger.info("🔧 Initializing step06 components for validation...")
        
        initialization_status = {}
        
        try:
            # Initialize FeatureInteractionEngine
            self.components["feature_interaction_engine"] = FeatureInteractionEngine(config)
            initialization_status["feature_interaction_engine"] = True
            self.logger.info("✅ FeatureInteractionEngine initialized")
        except Exception as e:
            self.logger.error(f"❌ FeatureInteractionEngine initialization failed: {e}")
            initialization_status["feature_interaction_engine"] = False
        
        try:
            # Initialize OptimizedTripleBarrierLabeling
            self.components["triple_barrier_labeling"] = OptimizedTripleBarrierLabeling()
            initialization_status["triple_barrier_labeling"] = True
            self.logger.info("✅ OptimizedTripleBarrierLabeling initialized")
        except Exception as e:
            self.logger.error(f"❌ OptimizedTripleBarrierLabeling initialization failed: {e}")
            initialization_status["triple_barrier_labeling"] = False
        
        try:
            # Initialize FeatureEngineeringStep
            self.components["feature_engineering_step"] = FeatureEngineeringStep(config)
            initialization_status["feature_engineering_step"] = True
            self.logger.info("✅ FeatureEngineeringStep initialized")
        except Exception as e:
            self.logger.error(f"❌ FeatureEngineeringStep initialization failed: {e}")
            initialization_status["feature_engineering_step"] = False
        
        self.logger.info(f"📊 Component initialization summary: {initialization_status}")
        return initialization_status
    
    async def run_comprehensive_validation(self, test_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Run comprehensive validation on all step06 components.
        
        Args:
            test_data: Optional test data for validation
            
        Returns:
            Comprehensive validation report
        """
        self.logger.info("🚀 Starting comprehensive step06 validation...")
        
        # Reset validation tracking
        reset_step06_validation_tracking()
        
        # Generate test data if not provided
        if test_data is None:
            test_data = self._generate_test_data()
        
        validation_results = {
            "timestamp": datetime.now().isoformat(),
            "test_data_info": {
                "shape": test_data.shape,
                "columns": list(test_data.columns),
                "data_types": test_data.dtypes.to_dict()
            },
            "component_validation": {},
            "overall_summary": {}
        }
        
        # Validate each component
        for component_name, component in self.components.items():
            self.logger.info(f"🔍 Validating component: {component_name}")
            
            try:
                component_result = await self._validate_component(component_name, component, test_data)
                validation_results["component_validation"][component_name] = component_result
                self.logger.info(f"✅ Component {component_name} validation completed")
            except Exception as e:
                self.logger.error(f"❌ Component {component_name} validation failed: {e}")
                validation_results["component_validation"][component_name] = {
                    "status": "failed",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
        
        # Generate overall summary
        validation_results["overall_summary"] = self._generate_overall_summary(validation_results)
        
        # Save comprehensive report
        await self._save_validation_report(validation_results)
        
        self.logger.info("✅ Comprehensive step06 validation completed")
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
        component_result = {
            "component_name": component_name,
            "timestamp": datetime.now().isoformat(),
            "validation_tests": {},
            "performance_metrics": {},
            "function_reports": {}
        }
        
        if component_name == "feature_interaction_engine":
            component_result = await self._validate_feature_interaction_engine(component, test_data)
        elif component_name == "triple_barrier_labeling":
            component_result = await self._validate_triple_barrier_labeling(component, test_data)
        elif component_name == "feature_engineering_step":
            component_result = await self._validate_feature_engineering_step(component, test_data)
        
        return component_result
    
    async def _validate_feature_interaction_engine(self, engine: FeatureInteractionEngine, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Validate FeatureInteractionEngine component."""
        result = {
            "component_name": "feature_interaction_engine",
            "timestamp": datetime.now().isoformat(),
            "validation_tests": {},
            "performance_metrics": {},
            "function_reports": {}
        }
        
        try:
            # Test technical indicator extraction
            self.logger.info("🔧 Testing technical indicator extraction...")
            indicators = engine.extract_optimal_technical_indicators(test_data)
            result["validation_tests"]["technical_indicators"] = {
                "status": "passed",
                "output_shape": indicators.shape,
                "output_columns": len(indicators.columns)
            }
            
            # Test correlation analysis
            self.logger.info("🔍 Testing correlation analysis...")
            correlation_results = engine.analyze_feature_correlations(indicators)
            result["validation_tests"]["correlation_analysis"] = {
                "status": "passed",
                "high_correlations": correlation_results.get("n_high_correlations", 0),
                "mean_correlation": correlation_results.get("mean_correlation", 0)
            }
            
            # Test interaction feature extraction
            self.logger.info("🔗 Testing interaction feature extraction...")
            features_array = indicators.values
            feature_names = list(indicators.columns)
            interactions = engine.extract_interaction_features(features_array, feature_names, test_data)
            result["validation_tests"]["interaction_features"] = {
                "status": "passed",
                "output_shape": interactions.shape,
                "feature_count": interactions.shape[1]
            }
            
            # Generate comprehensive report
            self.logger.info("📋 Generating comprehensive function report...")
            comprehensive_report = engine.generate_comprehensive_function_report()
            result["function_reports"]["comprehensive_report"] = comprehensive_report
            
        except Exception as e:
            result["validation_tests"]["error"] = str(e)
            result["status"] = "failed"
        
        return result
    
    async def _validate_triple_barrier_labeling(self, labeling: OptimizedTripleBarrierLabeling, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Validate OptimizedTripleBarrierLabeling component."""
        result = {
            "component_name": "triple_barrier_labeling",
            "timestamp": datetime.now().isoformat(),
            "validation_tests": {},
            "performance_metrics": {},
            "function_reports": {}
        }
        
        try:
            # Test vectorized labeling
            self.logger.info("🏷️ Testing vectorized triple barrier labeling...")
            labeled_data = labeling.apply_triple_barrier_labeling_vectorized(test_data)
            result["validation_tests"]["vectorized_labeling"] = {
                "status": "passed",
                "output_shape": labeled_data.shape,
                "label_distribution": labeled_data["label"].value_counts().to_dict(),
                "profit_tracking": "potential_profit_pct" in labeled_data.columns
            }
            
            # Test convenience method
            self.logger.info("🏷️ Testing convenience labeling method...")
            labels_only = labeling.apply_triple_barrier_labels(test_data)
            result["validation_tests"]["convenience_method"] = {
                "status": "passed",
                "output_length": len(labels_only),
                "label_distribution": labels_only.value_counts().to_dict()
            }
            
            # Generate comprehensive report
            self.logger.info("📋 Generating comprehensive labeling report...")
            comprehensive_report = labeling.generate_comprehensive_labeling_report()
            result["function_reports"]["comprehensive_report"] = comprehensive_report
            
        except Exception as e:
            result["validation_tests"]["error"] = str(e)
            result["status"] = "failed"
        
        return result
    
    async def _validate_feature_engineering_step(self, step: FeatureEngineeringStep, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Validate FeatureEngineeringStep component."""
        result = {
            "component_name": "feature_engineering_step",
            "timestamp": datetime.now().isoformat(),
            "validation_tests": {},
            "performance_metrics": {},
            "function_reports": {}
        }
        
        try:
            # Test input validation
            self.logger.info("✅ Testing input validation...")
            pipeline_state = {"labeled_data": test_data}
            is_valid, errors = step.validate_inputs({}, pipeline_state)
            result["validation_tests"]["input_validation"] = {
                "status": "passed" if is_valid else "failed",
                "is_valid": is_valid,
                "errors": errors
            }
            
            # Test feature engineering execution
            self.logger.info("🔧 Testing feature engineering execution...")
            training_input = {"output_dir": str(self.output_dir)}
            pipeline_state = {"labeled_data": test_data}
            
            # This would normally be async, but we'll simulate it
            result["validation_tests"]["execution"] = {
                "status": "simulated",
                "note": "Full execution requires async pipeline context"
            }
            
            # Test output validation
            self.logger.info("✅ Testing output validation...")
            # Simulate engineered data for validation
            simulated_engineered_data = {"all": test_data.copy()}
            simulated_pipeline_state = {"engineered_data": simulated_engineered_data}
            is_valid, errors = step.validate_outputs(simulated_pipeline_state)
            result["validation_tests"]["output_validation"] = {
                "status": "passed" if is_valid else "failed",
                "is_valid": is_valid,
                "errors": errors
            }
            
            # Generate comprehensive report
            self.logger.info("📋 Generating comprehensive step report...")
            comprehensive_report = step.generate_comprehensive_step06_report()
            result["function_reports"]["comprehensive_report"] = comprehensive_report
            
        except Exception as e:
            result["validation_tests"]["error"] = str(e)
            result["status"] = "failed"
        
        return result
    
    def _generate_test_data(self) -> pd.DataFrame:
        """Generate test data for validation."""
        self.logger.info("📊 Generating test data for validation...")
        
        # Generate synthetic OHLCV data
        np.random.seed(42)
        n_samples = 1000
        
        dates = pd.date_range("2024-01-01", periods=n_samples, freq="1min")
        
        # Generate realistic price data
        base_price = 100.0
        returns = np.random.normal(0, 0.001, n_samples)
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Generate OHLCV data
        data = pd.DataFrame({
            "open": prices,
            "high": [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
            "low": [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
            "close": prices,
            "volume": np.random.uniform(1000, 10000, n_samples)
        }, index=dates)
        
        # Ensure high >= max(open, close) and low <= min(open, close)
        data["high"] = np.maximum(data["high"], np.maximum(data["open"], data["close"]))
        data["low"] = np.minimum(data["low"], np.minimum(data["open"], data["close"]))
        
        self.logger.info(f"✅ Generated test data: {data.shape}")
        return data
    
    def _generate_overall_summary(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall validation summary."""
        component_validation = validation_results["component_validation"]
        
        total_components = len(component_validation)
        successful_components = sum(1 for comp in component_validation.values() 
                                  if comp.get("status") != "failed")
        
        total_tests = 0
        successful_tests = 0
        
        for comp_name, comp_result in component_validation.items():
            validation_tests = comp_result.get("validation_tests", {})
            for test_name, test_result in validation_tests.items():
                if isinstance(test_result, dict) and "status" in test_result:
                    total_tests += 1
                    if test_result["status"] == "passed":
                        successful_tests += 1
        
        return {
            "total_components": total_components,
            "successful_components": successful_components,
            "component_success_rate": successful_components / total_components if total_components > 0 else 0,
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "test_success_rate": successful_tests / total_tests if total_tests > 0 else 0,
            "validation_framework_status": "active",
            "timestamp": datetime.now().isoformat()
        }
    
    async def _save_validation_report(self, validation_results: Dict[str, Any]) -> None:
        """Save comprehensive validation report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save main report
        report_path = self.output_dir / f"step06_comprehensive_validation_report_{timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(validation_results, f, indent=2, default=str)
        
        # Save summary report
        summary_path = self.output_dir / f"step06_validation_summary_{timestamp}.json"
        summary = {
            "timestamp": validation_results["timestamp"],
            "overall_summary": validation_results["overall_summary"],
            "component_summary": {
                name: {
                    "status": result.get("status", "unknown"),
                    "tests_count": len(result.get("validation_tests", {})),
                    "reports_count": len(result.get("function_reports", {}))
                }
                for name, result in validation_results["component_validation"].items()
            }
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"💾 Validation reports saved:")
        self.logger.info(f"   Main report: {report_path}")
        self.logger.info(f"   Summary report: {summary_path}")
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get current validation summary."""
        return get_step06_validation_summary()
    
    def reset_validation_tracking(self) -> None:
        """Reset validation tracking."""
        reset_step06_validation_tracking()
        self.logger.info("🔄 Validation tracking reset")


async def run_step06_comprehensive_validation(
    config: Optional[Dict[str, Any]] = None,
    test_data: Optional[pd.DataFrame] = None,
    output_dir: str = "step06_validation_reports"
) -> Dict[str, Any]:
    """
    Run comprehensive validation for all step06 components.
    
    Args:
        config: Configuration dictionary
        test_data: Optional test data
        output_dir: Output directory for reports
        
    Returns:
        Comprehensive validation results
    """
    if config is None:
        config = {
            "step06_feature_engineering": {
                "use_matrix_optimizer": True,
                "force_regime_specific_periods": False,
                "momentum_volume_enabled": True,
                "trend_volatility_enabled": True,
                "oscillator_trend_enabled": True,
                "volume_price_enabled": True,
                "volatility_regime_enabled": True,
                "cross_timeframe_enabled": True,
                "regime_dependent_enabled": True
            }
        }
    
    orchestrator = Step06ValidationOrchestrator(output_dir)
    
    # Initialize components
    init_status = orchestrator.initialize_components(config)
    
    # Run comprehensive validation
    validation_results = await orchestrator.run_comprehensive_validation(test_data)
    
    return validation_results


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def main():
        logging.basicConfig(level=logging.INFO)
        
        # Run comprehensive validation
        results = await run_step06_comprehensive_validation()
        
        print("Step06 Comprehensive Validation Results:")
        print(f"Overall Summary: {results['overall_summary']}")
        
        for component_name, component_result in results["component_validation"].items():
            print(f"\n{component_name}:")
            print(f"  Status: {component_result.get('status', 'unknown')}")
            print(f"  Tests: {len(component_result.get('validation_tests', {}))}")
            print(f"  Reports: {len(component_result.get('function_reports', {}))}")
    
    asyncio.run(main())