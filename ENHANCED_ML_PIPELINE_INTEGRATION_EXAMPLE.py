#!/usr/bin/env python3
"""
Enhanced ML Pipeline Integration Example

This example demonstrates how to use the enhanced ML pipeline components
with comprehensive error detection, monitoring, and reporting capabilities.

The enhanced components work together to provide:
- Real-time error detection and classification
- HPO monitoring with convergence detection
- Comprehensive testing and validation
- Advanced reporting and alerting
- Health monitoring and trend analysis
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np

# Import enhanced components
from src.utils.ml_training_safeguards import MLTrainingSafeguards
from src.utils.ml_common.optimization.hpo_utils import HyperparameterOptimization
from src.training.model_interpretability.interpretability_reporter import InterpretabilityReporter
from src.training.core.training_manager import TrainingManager
from src.training.steps.model_training.model_validation import ModelValidationStep

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedMLPipelineExample:
    """Example of enhanced ML pipeline with comprehensive monitoring."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the enhanced ML pipeline example."""
        self.config = config or {}
        self.logger = logger.getChild('EnhancedMLPipelineExample')
        
        # Initialize enhanced components
        self.safeguards = MLTrainingSafeguards(self.config.get('safeguards', {}))
        self.hpo_optimizer = HyperparameterOptimization(self.config.get('hpo', {}))
        self.interpretability_reporter = InterpretabilityReporter(self.config.get('interpretability', {}))
        self.training_manager = TrainingManager(self.config.get('training', {}))
        self.model_validator = ModelValidationStep(self.config.get('validation', {}))
        
        # Pipeline state
        self.pipeline_id = f"enhanced_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"🚀 Enhanced ML Pipeline Example initialized with ID: {self.pipeline_id}")

    async def run_comprehensive_pipeline(self, symbol: str, exchange: str) -> Dict[str, Any]:
        """Run a comprehensive ML pipeline with enhanced monitoring."""
        try:
            self.logger.info(f"🚀 Starting comprehensive ML pipeline for {symbol} on {exchange}")
            
            # Initialize pipeline
            await self._initialize_pipeline()
            
            # Run data validation
            data_validation_result = await self._run_data_validation(symbol, exchange)
            
            # Run HPO with monitoring
            hpo_result = await self._run_hpo_with_monitoring(symbol, exchange)
            
            # Run model training
            training_result = await self._run_model_training(symbol, exchange)
            
            # Run model validation
            validation_result = await self._run_model_validation(symbol, exchange)
            
            # Run interpretability analysis
            interpretability_result = await self._run_interpretability_analysis(symbol, exchange)
            
            # Generate comprehensive report
            final_report = await self._generate_comprehensive_report({
                'data_validation': data_validation_result,
                'hpo': hpo_result,
                'training': training_result,
                'validation': validation_result,
                'interpretability': interpretability_result
            })
            
            self.logger.info("✅ Comprehensive ML pipeline completed successfully")
            return final_report
            
        except Exception as e:
            error_context = {
                'component': 'enhanced_ml_pipeline',
                'function': 'run_comprehensive_pipeline',
                'symbol': symbol,
                'exchange': exchange,
                'pipeline_id': self.pipeline_id
            }
            self.safeguards.detect_and_classify_error(e, error_context)
            self.logger.error(f"❌ Comprehensive pipeline failed: {e}")
            raise

    async def _initialize_pipeline(self):
        """Initialize the pipeline with enhanced monitoring."""
        try:
            self.logger.info("🔧 Initializing enhanced pipeline components...")
            
            # Initialize training manager
            if not await self.training_manager.initialize():
                raise Exception("Failed to initialize training manager")
            
            # Track initialization
            self.training_manager.track_training_execution(
                f"{self.pipeline_id}_init", 
                "completed", 
                {"initialization_time": datetime.now().isoformat()}
            )
            
            self.logger.info("✅ Pipeline initialization completed")
            
        except Exception as e:
            error_context = {
                'component': 'pipeline_initialization',
                'function': '_initialize_pipeline',
                'pipeline_id': self.pipeline_id
            }
            self.safeguards.detect_and_classify_error(e, error_context)
            raise

    async def _run_data_validation(self, symbol: str, exchange: str) -> Dict[str, Any]:
        """Run data validation with enhanced error detection."""
        try:
            self.logger.info("🔍 Running enhanced data validation...")
            
            # Simulate data loading and validation
            # In a real implementation, you would load actual data
            sample_data = self._generate_sample_data()
            
            # Use safeguards for data validation
            validation_result = {
                'is_valid': True,
                'data_quality_score': 0.95,
                'issues_detected': [],
                'recommendations': []
            }
            
            # Check for common data issues
            if sample_data.isnull().sum().sum() > 0:
                validation_result['issues_detected'].append("Missing values detected")
                validation_result['recommendations'].append("Handle missing values appropriately")
                validation_result['is_valid'] = False
            
            if np.isinf(sample_data.select_dtypes(include=np.number)).sum().sum() > 0:
                validation_result['issues_detected'].append("Infinite values detected")
                validation_result['recommendations'].append("Handle infinite values")
                validation_result['is_valid'] = False
            
            # Log validation result
            if validation_result['is_valid']:
                self.logger.info("✅ Data validation passed")
            else:
                self.logger.warning(f"⚠️ Data validation issues: {validation_result['issues_detected']}")
            
            return validation_result
            
        except Exception as e:
            error_context = {
                'component': 'data_validation',
                'function': '_run_data_validation',
                'symbol': symbol,
                'exchange': exchange
            }
            self.safeguards.detect_and_classify_error(e, error_context)
            raise

    async def _run_hpo_with_monitoring(self, symbol: str, exchange: str) -> Dict[str, Any]:
        """Run HPO with enhanced monitoring and convergence detection."""
        try:
            self.logger.info("🔍 Running HPO with enhanced monitoring...")
            
            # Start HPO study monitoring
            study_id = f"hpo_{symbol}_{exchange}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            study_info = self.hpo_optimizer.start_study_monitoring(study_id, f"HPO for {symbol}")
            
            # Simulate HPO trials
            hpo_result = {
                'study_id': study_id,
                'total_trials': 0,
                'best_score': None,
                'convergence_info': None,
                'monitoring_summary': {}
            }
            
            # Simulate trials with monitoring
            for trial_num in range(1, 21):  # 20 trials
                # Simulate trial parameters and results
                trial_params = {
                    'learning_rate': np.random.uniform(0.01, 0.3),
                    'max_depth': np.random.randint(3, 10),
                    'n_estimators': np.random.randint(50, 200)
                }
                
                # Simulate objective value with some noise
                base_score = 0.8 + np.random.normal(0, 0.05)
                objective_value = max(0.0, min(1.0, base_score))
                
                # Record trial with monitoring
                trial_result = self.hpo_optimizer.record_trial_with_monitoring(
                    study_id=study_id,
                    trial_number=trial_num,
                    parameters=trial_params,
                    objective_value=objective_value,
                    training_time=np.random.uniform(10, 60),
                    memory_usage=np.random.uniform(0.1, 0.8)
                )
                
                hpo_result['total_trials'] += 1
                if hpo_result['best_score'] is None or objective_value > hpo_result['best_score']:
                    hpo_result['best_score'] = objective_value
                
                # Check for convergence
                study_status = self.hpo_optimizer.get_study_status(study_id)
                if study_status and study_status.get('convergence_info'):
                    hpo_result['convergence_info'] = study_status['convergence_info']
                    self.logger.info(f"✅ HPO converged after {trial_num} trials")
                    break
            
            # Get monitoring summary
            hpo_result['monitoring_summary'] = self.hpo_optimizer.get_monitoring_summary()
            
            self.logger.info(f"✅ HPO completed: {hpo_result['total_trials']} trials, best score: {hpo_result['best_score']:.3f}")
            return hpo_result
            
        except Exception as e:
            error_context = {
                'component': 'hpo_optimization',
                'function': '_run_hpo_with_monitoring',
                'symbol': symbol,
                'exchange': exchange
            }
            self.safeguards.detect_and_classify_error(e, error_context)
            raise

    async def _run_model_training(self, symbol: str, exchange: str) -> Dict[str, Any]:
        """Run model training with enhanced monitoring."""
        try:
            self.logger.info("🏋️ Running model training with enhanced monitoring...")
            
            # Simulate model training
            training_result = {
                'model_id': f"model_{symbol}_{exchange}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'training_time': np.random.uniform(60, 300),
                'model_accuracy': np.random.uniform(0.7, 0.95),
                'training_loss': np.random.uniform(0.1, 0.5),
                'validation_loss': np.random.uniform(0.15, 0.6),
                'status': 'completed'
            }
            
            # Track training execution
            self.training_manager.track_training_execution(
                training_result['model_id'],
                training_result['status'],
                training_result
            )
            
            self.logger.info(f"✅ Model training completed: {training_result['model_id']}")
            return training_result
            
        except Exception as e:
            error_context = {
                'component': 'model_training',
                'function': '_run_model_training',
                'symbol': symbol,
                'exchange': exchange
            }
            self.safeguards.detect_and_classify_error(e, error_context)
            raise

    async def _run_model_validation(self, symbol: str, exchange: str) -> Dict[str, Any]:
        """Run model validation with enhanced monitoring."""
        try:
            self.logger.info("🔍 Running model validation with enhanced monitoring...")
            
            # Simulate model validation
            model_metrics = {
                'accuracy': np.random.uniform(0.7, 0.95),
                'precision': np.random.uniform(0.6, 0.9),
                'recall': np.random.uniform(0.6, 0.9),
                'f1_score': np.random.uniform(0.6, 0.9)
            }
            
            model_id = f"model_{symbol}_{exchange}"
            
            # Validate model performance
            validation_result = self.model_validator.validate_model_performance(model_metrics, model_id)
            
            # Track validation metrics
            self.model_validator.track_validation_metrics(model_id, model_metrics, validation_result)
            
            self.logger.info(f"✅ Model validation completed: {validation_result['validation_score']:.3f}")
            return {
                'model_id': model_id,
                'metrics': model_metrics,
                'validation_result': validation_result
            }
            
        except Exception as e:
            error_context = {
                'component': 'model_validation',
                'function': '_run_model_validation',
                'symbol': symbol,
                'exchange': exchange
            }
            self.safeguards.detect_and_classify_error(e, error_context)
            raise

    async def _run_interpretability_analysis(self, symbol: str, exchange: str) -> Dict[str, Any]:
        """Run interpretability analysis with enhanced monitoring."""
        try:
            self.logger.info("🔍 Running interpretability analysis with enhanced monitoring...")
            
            # Simulate interpretability results
            interpretability_results = {
                'feature_importance': {
                    'feature_1': np.random.uniform(0.1, 0.4),
                    'feature_2': np.random.uniform(0.1, 0.3),
                    'feature_3': np.random.uniform(0.05, 0.2),
                    'feature_4': np.random.uniform(0.05, 0.15),
                    'feature_5': np.random.uniform(0.02, 0.1)
                },
                'model_performance': {
                    'accuracy': np.random.uniform(0.7, 0.95),
                    'precision': np.random.uniform(0.6, 0.9),
                    'recall': np.random.uniform(0.6, 0.9)
                },
                'bias_analysis': {
                    'overall_bias': np.random.uniform(0.1, 0.3)
                }
            }
            
            model_id = f"model_{symbol}_{exchange}"
            
            # Analyze model health
            health_analysis = self.interpretability_reporter.analyze_model_health(interpretability_results)
            
            # Generate alert if needed
            self.interpretability_reporter.generate_alert_if_needed(health_analysis, model_id)
            
            # Track interpretability metrics
            self.interpretability_reporter.track_interpretability_metrics(interpretability_results, model_id)
            
            self.logger.info(f"✅ Interpretability analysis completed: {health_analysis['overall_health']}")
            return {
                'model_id': model_id,
                'results': interpretability_results,
                'health_analysis': health_analysis
            }
            
        except Exception as e:
            error_context = {
                'component': 'interpretability_analysis',
                'function': '_run_interpretability_analysis',
                'symbol': symbol,
                'exchange': exchange
            }
            self.safeguards.detect_and_classify_error(e, error_context)
            raise

    async def _generate_comprehensive_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive report with all monitoring data."""
        try:
            self.logger.info("📊 Generating comprehensive report...")
            
            # Get summaries from all components
            error_summary = self.safeguards.get_error_summary()
            training_summary = self.training_manager.get_training_summary()
            validation_summary = self.model_validator.get_validation_summary()
            interpretability_trends = self.interpretability_reporter.get_interpretability_trends()
            hpo_monitoring = self.hpo_optimizer.get_monitoring_summary()
            
            # Check health status
            training_health = self.training_manager.check_health_status()
            validation_health = self.model_validator.check_validation_health()
            
            comprehensive_report = {
                'pipeline_id': self.pipeline_id,
                'timestamp': datetime.now().isoformat(),
                'overall_status': 'completed',
                'results': results,
                'monitoring_summaries': {
                    'error_summary': error_summary,
                    'training_summary': training_summary,
                    'validation_summary': validation_summary,
                    'interpretability_trends': interpretability_trends,
                    'hpo_monitoring': hpo_monitoring
                },
                'health_status': {
                    'training_health': training_health,
                    'validation_health': validation_health
                },
                'recommendations': self._generate_recommendations(results, error_summary)
            }
            
            self.logger.info("✅ Comprehensive report generated")
            return comprehensive_report
            
        except Exception as e:
            error_context = {
                'component': 'report_generation',
                'function': '_generate_comprehensive_report',
                'pipeline_id': self.pipeline_id
            }
            self.safeguards.detect_and_classify_error(e, error_context)
            raise

    def _generate_recommendations(self, results: Dict[str, Any], error_summary: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on results and error summary."""
        recommendations = []
        
        # Check error rates
        if error_summary['recent_errors_1h'] > 5:
            recommendations.append("High error rate detected - investigate recent errors")
        
        # Check validation results
        if 'validation' in results:
            validation_result = results['validation']['validation_result']
            if not validation_result['is_valid']:
                recommendations.extend(validation_result['recommendations'])
        
        # Check interpretability results
        if 'interpretability' in results:
            health_analysis = results['interpretability']['health_analysis']
            if health_analysis['risk_level'] in ['high', 'critical']:
                recommendations.extend(health_analysis['recommendations'])
        
        # Check HPO results
        if 'hpo' in results:
            hpo_result = results['hpo']
            if hpo_result['best_score'] < 0.8:
                recommendations.append("Consider additional HPO trials or different search space")
        
        return recommendations

    def _generate_sample_data(self) -> pd.DataFrame:
        """Generate sample data for demonstration."""
        np.random.seed(42)
        n_samples = 1000
        
        data = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, n_samples),
            'feature_2': np.random.normal(0, 1, n_samples),
            'feature_3': np.random.normal(0, 1, n_samples),
            'feature_4': np.random.normal(0, 1, n_samples),
            'target': np.random.randint(0, 2, n_samples)
        })
        
        return data

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and monitoring data."""
        try:
            return {
                'pipeline_id': self.pipeline_id,
                'timestamp': datetime.now().isoformat(),
                'error_summary': self.safeguards.get_error_summary(),
                'training_summary': self.training_manager.get_training_summary(),
                'validation_summary': self.model_validator.get_validation_summary(),
                'interpretability_trends': self.interpretability_reporter.get_interpretability_trends(),
                'hpo_monitoring': self.hpo_optimizer.get_monitoring_summary()
            }
        except Exception as e:
            self.logger.error(f"❌ Failed to get pipeline status: {e}")
            return {'error': str(e)}


async def main():
    """Main function to demonstrate the enhanced ML pipeline."""
    try:
        # Configuration for enhanced components
        config = {
            'safeguards': {
                'enable_real_time_monitoring': True,
                'alert_thresholds': {
                    'critical_errors_per_hour': 3,
                    'high_errors_per_hour': 10,
                    'same_error_repetition': 5
                }
            },
            'hpo': {
                'enable_monitoring': True,
                'convergence': {
                    'improvement_threshold': 0.001,
                    'patience_trials': 15,
                    'variance_threshold': 0.01
                }
            },
            'interpretability': {
                'enable_real_time_monitoring': True,
                'alert_thresholds': {
                    'low_accuracy': 0.7,
                    'high_bias': 0.3
                }
            },
            'validation': {
                'validation_thresholds': {
                    'min_accuracy': 0.6,
                    'min_precision': 0.5,
                    'min_recall': 0.5,
                    'min_f1_score': 0.5
                }
            }
        }
        
        # Create enhanced pipeline example
        pipeline = EnhancedMLPipelineExample(config)
        
        # Run comprehensive pipeline
        result = await pipeline.run_comprehensive_pipeline("BTCUSDT", "binance")
        
        # Print results
        print("\n" + "="*80)
        print("ENHANCED ML PIPELINE RESULTS")
        print("="*80)
        print(f"Pipeline ID: {result['pipeline_id']}")
        print(f"Overall Status: {result['overall_status']}")
        print(f"Timestamp: {result['timestamp']}")
        
        print("\n📊 MONITORING SUMMARIES:")
        for component, summary in result['monitoring_summaries'].items():
            print(f"\n{component.upper()}:")
            if isinstance(summary, dict):
                for key, value in summary.items():
                    print(f"  {key}: {value}")
            else:
                print(f"  {summary}")
        
        print("\n🏥 HEALTH STATUS:")
        for component, health in result['health_status'].items():
            print(f"\n{component.upper()}:")
            print(f"  Overall Health: {health['overall_health']}")
            print(f"  Risk Level: {health['risk_level']}")
            if health['issues']:
                print(f"  Issues: {health['issues']}")
            if health['recommendations']:
                print(f"  Recommendations: {health['recommendations']}")
        
        print("\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(result['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        print("\n" + "="*80)
        print("ENHANCED ML PIPELINE COMPLETED SUCCESSFULLY")
        print("="*80)
        
    except Exception as e:
        logger.error(f"❌ Enhanced ML pipeline example failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())