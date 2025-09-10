"""
Comprehensive Data Flow Testing

This module provides comprehensive data flow testing and validation for the training pipeline,
ensuring that data flows correctly between all pipeline steps and maintains integrity.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from pathlib import Path

# Import common operations
from src.utils.common_operations import get_logger

logger = get_logger(__name__)


class ComprehensiveDataFlowTester:
    """
    Comprehensive Data Flow Tester for the training pipeline.
    
    This class provides complete data flow testing including:
    - Data structure validation
    - Data type checking
    - Data integrity verification
    - Step-to-step data flow testing
    - Mock data generation
    - Data flow visualization
    """
    
    def __init__(self):
        """Initialize comprehensive data flow tester."""
        self.logger = logger.getChild('ComprehensiveDataFlowTester')
        self.test_results = {}
        self.mock_data_generator = MockDataGenerator()
        self.logger.info("🔧 Comprehensive Data Flow Tester initialized")
    
    def generate_mock_pipeline_data(self) -> Dict[str, Any]:
        """Generate comprehensive mock data for pipeline testing."""
        try:
            self.logger.info("🔧 Generating mock pipeline data...")
            
            mock_data = {
                # Step 1: Data Collection & Qualification
                'raw_data': self.mock_data_generator.generate_market_data(),
                'data_quality_report': self.mock_data_generator.generate_data_quality_report(),
                'collection_metadata': self.mock_data_generator.generate_collection_metadata(),
                
                # Step 2: SR Levels Detection
                'sr_levels': self.mock_data_generator.generate_sr_levels(),
                'sr_metadata': self.mock_data_generator.generate_sr_metadata(),
                
                # Step 3: Cluster/HMM Regimes Definition
                'regimes': self.mock_data_generator.generate_regimes(),
                'regime_metadata': self.mock_data_generator.generate_regime_metadata(),
                
                # Step 4: Feature Engineering
                'engineered_features': self.mock_data_generator.generate_engineered_features(),
                'feature_metadata': self.mock_data_generator.generate_feature_metadata(),
                
                # Step 5: Feature Selection
                'selected_features': self.mock_data_generator.generate_selected_features(),
                'selection_metadata': self.mock_data_generator.generate_selection_metadata(),
                
                # Step 6: Analyst Training
                'analyst_models': self.mock_data_generator.generate_analyst_models(),
                'analyst_metadata': self.mock_data_generator.generate_analyst_metadata(),
                
                # Step 7: General Model Training
                'general_model': self.mock_data_generator.generate_general_model(),
                'general_model_metadata': self.mock_data_generator.generate_general_model_metadata(),
                
                # Step 8: Tactician Training
                'tactician_models': self.mock_data_generator.generate_tactician_models(),
                'tactician_metadata': self.mock_data_generator.generate_tactician_metadata(),
                
                # Step 9: Backtesting & Validation
                'backtesting_results': self.mock_data_generator.generate_backtesting_results(),
                'validation_results': self.mock_data_generator.generate_validation_results(),
                'validation_metadata': self.mock_data_generator.generate_validation_metadata()
            }
            
            self.logger.info("✅ Mock pipeline data generated successfully")
            return mock_data
            
        except Exception as e:
            self.logger.exception(f"Failed to generate mock pipeline data: {e}")
            raise
    
    def validate_data_structure(self, data: Any, expected_structure: Dict[str, Any], data_name: str) -> Dict[str, Any]:
        """Validate data structure against expected format."""
        try:
            self.logger.info(f"🔍 Validating data structure for: {data_name}")
            
            validation_result = {
                'data_name': data_name,
                'validation_timestamp': datetime.now().isoformat(),
                'passed': True,
                'errors': [],
                'warnings': [],
                'structure_info': {}
            }
            
            # Check if data exists
            if data is None:
                validation_result['passed'] = False
                validation_result['errors'].append("Data is None")
                return validation_result
            
            # Check data type
            expected_type = expected_structure.get('type')
            if expected_type and not isinstance(data, expected_type):
                validation_result['passed'] = False
                validation_result['errors'].append(f"Expected type {expected_type}, got {type(data)}")
            
            # Check required fields for dictionaries
            if isinstance(data, dict):
                required_fields = expected_structure.get('required_fields', [])
                for field in required_fields:
                    if field not in data:
                        validation_result['passed'] = False
                        validation_result['errors'].append(f"Missing required field: {field}")
                
                # Check field types
                field_types = expected_structure.get('field_types', {})
                for field, expected_field_type in field_types.items():
                    if field in data:
                        if not isinstance(data[field], expected_field_type):
                            validation_result['warnings'].append(f"Field '{field}' expected type {expected_field_type}, got {type(data[field])}")
                
                validation_result['structure_info']['field_count'] = len(data)
                validation_result['structure_info']['fields'] = list(data.keys())
            
            # Check list structure
            elif isinstance(data, list):
                expected_item_type = expected_structure.get('item_type')
                if expected_item_type:
                    for i, item in enumerate(data):
                        if not isinstance(item, expected_item_type):
                            validation_result['warnings'].append(f"Item {i} expected type {expected_item_type}, got {type(item)}")
                
                validation_result['structure_info']['item_count'] = len(data)
                if data:
                    validation_result['structure_info']['item_type'] = type(data[0]).__name__
            
            self.logger.info(f"✅ Data structure validation completed for: {data_name}")
            return validation_result
            
        except Exception as e:
            self.logger.exception(f"Data structure validation failed for {data_name}: {e}")
            return {
                'data_name': data_name,
                'validation_timestamp': datetime.now().isoformat(),
                'passed': False,
                'errors': [f"Validation error: {str(e)}"],
                'warnings': [],
                'structure_info': {}
            }
    
    def test_step_data_flow(self, step_name: str, input_data: Dict[str, Any], output_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test data flow for a specific pipeline step."""
        try:
            self.logger.info(f"🔍 Testing data flow for step: {step_name}")
            
            flow_test_result = {
                'step_name': step_name,
                'test_timestamp': datetime.now().isoformat(),
                'passed': True,
                'errors': [],
                'warnings': [],
                'data_flow_info': {
                    'input_data_keys': list(input_data.keys()) if isinstance(input_data, dict) else [],
                    'output_data_keys': list(output_data.keys()) if isinstance(output_data, dict) else [],
                    'data_preserved': True,
                    'new_data_added': True
                }
            }
            
            # Check if output contains expected data
            expected_output_keys = self._get_expected_output_keys(step_name)
            for expected_key in expected_output_keys:
                if expected_key not in output_data:
                    flow_test_result['passed'] = False
                    flow_test_result['errors'].append(f"Missing expected output key: {expected_key}")
            
            # Check if input data is preserved in output
            if isinstance(input_data, dict) and isinstance(output_data, dict):
                for key, value in input_data.items():
                    if key not in output_data:
                        flow_test_result['warnings'].append(f"Input data key '{key}' not preserved in output")
                        flow_test_result['data_flow_info']['data_preserved'] = False
            
            # Check for new data added
            if isinstance(input_data, dict) and isinstance(output_data, dict):
                new_keys = set(output_data.keys()) - set(input_data.keys())
                if not new_keys:
                    flow_test_result['warnings'].append("No new data added in this step")
                    flow_test_result['data_flow_info']['new_data_added'] = False
                else:
                    flow_test_result['data_flow_info']['new_keys_added'] = list(new_keys)
            
            self.logger.info(f"✅ Data flow test completed for step: {step_name}")
            return flow_test_result
            
        except Exception as e:
            self.logger.exception(f"Data flow test failed for step {step_name}: {e}")
            return {
                'step_name': step_name,
                'test_timestamp': datetime.now().isoformat(),
                'passed': False,
                'errors': [f"Test error: {str(e)}"],
                'warnings': [],
                'data_flow_info': {}
            }
    
    def test_complete_pipeline_data_flow(self, pipeline_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test complete pipeline data flow."""
        try:
            self.logger.info("🔍 Testing complete pipeline data flow...")
            
            pipeline_test_result = {
                'test_timestamp': datetime.now().isoformat(),
                'overall_passed': True,
                'step_results': {},
                'summary': {
                    'total_steps_tested': 0,
                    'steps_passed': 0,
                    'steps_failed': 0,
                    'total_errors': 0,
                    'total_warnings': 0
                }
            }
            
            # Define pipeline steps and their data flow
            pipeline_steps = [
                ('data_collection_qualification', ['raw_data'], ['data', 'data_quality_report', 'collection_metadata']),
                ('sr_levels_detection', ['data'], ['sr_levels', 'sr_metadata']),
                ('regimes_definition', ['data', 'sr_levels'], ['regimes', 'regime_metadata']),
                ('feature_engineering', ['regimes'], ['engineered_features', 'feature_metadata']),
                ('feature_selection', ['engineered_features'], ['selected_features', 'selection_metadata']),
                ('analyst_training', ['selected_features', 'regimes'], ['analyst_models', 'analyst_metadata']),
                ('general_model_training', ['selected_features', 'regimes'], ['general_model', 'general_model_metadata']),
                ('tactician_training', ['selected_features', 'regimes', 'analyst_models'], ['tactician_models', 'tactician_metadata']),
                ('backtesting_validation', ['analyst_models', 'general_model', 'tactician_models'], ['backtesting_results', 'validation_results', 'validation_metadata'])
            ]
            
            # Test each step
            for step_name, input_keys, output_keys in pipeline_steps:
                # Prepare input data for this step
                input_data = {key: pipeline_data.get(key) for key in input_keys if key in pipeline_data}
                output_data = {key: pipeline_data.get(key) for key in output_keys if key in pipeline_data}
                
                # Test step data flow
                step_result = self.test_step_data_flow(step_name, input_data, output_data)
                pipeline_test_result['step_results'][step_name] = step_result
                
                # Update summary
                pipeline_test_result['summary']['total_steps_tested'] += 1
                if step_result['passed']:
                    pipeline_test_result['summary']['steps_passed'] += 1
                else:
                    pipeline_test_result['summary']['steps_failed'] += 1
                    pipeline_test_result['overall_passed'] = False
                
                pipeline_test_result['summary']['total_errors'] += len(step_result['errors'])
                pipeline_test_result['summary']['total_warnings'] += len(step_result['warnings'])
            
            self.logger.info("✅ Complete pipeline data flow test completed")
            return pipeline_test_result
            
        except Exception as e:
            self.logger.exception(f"Complete pipeline data flow test failed: {e}")
            return {
                'test_timestamp': datetime.now().isoformat(),
                'overall_passed': False,
                'step_results': {},
                'summary': {
                    'total_steps_tested': 0,
                    'steps_passed': 0,
                    'steps_failed': 0,
                    'total_errors': 1,
                    'total_warnings': 0
                },
                'error': str(e)
            }
    
    def _get_expected_output_keys(self, step_name: str) -> List[str]:
        """Get expected output keys for a pipeline step."""
        expected_outputs = {
            'data_collection_qualification': ['data', 'data_quality_report', 'collection_metadata'],
            'sr_levels_detection': ['sr_levels', 'sr_metadata'],
            'regimes_definition': ['regimes', 'regime_metadata'],
            'feature_engineering': ['engineered_features', 'feature_metadata'],
            'feature_selection': ['selected_features', 'selection_metadata'],
            'analyst_training': ['analyst_models', 'analyst_metadata'],
            'general_model_training': ['general_model', 'general_model_metadata'],
            'tactician_training': ['tactician_models', 'tactician_metadata'],
            'backtesting_validation': ['backtesting_results', 'validation_results', 'validation_metadata']
        }
        return expected_outputs.get(step_name, [])
    
    def generate_data_flow_report(self, test_results: Dict[str, Any]) -> str:
        """Generate a comprehensive data flow report."""
        try:
            report = []
            report.append("=" * 80)
            report.append("COMPREHENSIVE DATA FLOW TEST REPORT")
            report.append("=" * 80)
            report.append(f"Test Timestamp: {test_results.get('test_timestamp', 'Unknown')}")
            report.append(f"Overall Status: {'✅ PASSED' if test_results.get('overall_passed', False) else '❌ FAILED'}")
            report.append("")
            
            # Summary
            summary = test_results.get('summary', {})
            report.append("📊 SUMMARY")
            report.append("-" * 20)
            report.append(f"Total Steps Tested: {summary.get('total_steps_tested', 0)}")
            report.append(f"Steps Passed: {summary.get('steps_passed', 0)}")
            report.append(f"Steps Failed: {summary.get('steps_failed', 0)}")
            report.append(f"Total Errors: {summary.get('total_errors', 0)}")
            report.append(f"Total Warnings: {summary.get('total_warnings', 0)}")
            report.append("")
            
            # Step-by-step results
            step_results = test_results.get('step_results', {})
            if step_results:
                report.append("🔍 STEP-BY-STEP RESULTS")
                report.append("-" * 30)
                
                for step_name, step_result in step_results.items():
                    status = "✅ PASSED" if step_result.get('passed', False) else "❌ FAILED"
                    report.append(f"{step_name}: {status}")
                    
                    if step_result.get('errors'):
                        for error in step_result['errors']:
                            report.append(f"  ❌ Error: {error}")
                    
                    if step_result.get('warnings'):
                        for warning in step_result['warnings']:
                            report.append(f"  ⚠️  Warning: {warning}")
                    
                    report.append("")
            
            # Data flow information
            if step_results:
                report.append("📈 DATA FLOW INFORMATION")
                report.append("-" * 30)
                
                for step_name, step_result in step_results.items():
                    data_flow_info = step_result.get('data_flow_info', {})
                    if data_flow_info:
                        report.append(f"{step_name}:")
                        report.append(f"  Input Keys: {data_flow_info.get('input_data_keys', [])}")
                        report.append(f"  Output Keys: {data_flow_info.get('output_data_keys', [])}")
                        report.append(f"  Data Preserved: {data_flow_info.get('data_preserved', 'Unknown')}")
                        report.append(f"  New Data Added: {data_flow_info.get('new_data_added', 'Unknown')}")
                        if 'new_keys_added' in data_flow_info:
                            report.append(f"  New Keys Added: {data_flow_info['new_keys_added']}")
                        report.append("")
            
            report.append("=" * 80)
            return "\n".join(report)
            
        except Exception as e:
            self.logger.exception(f"Failed to generate data flow report: {e}")
            return f"Error generating report: {str(e)}"


class MockDataGenerator:
    """Mock data generator for testing purposes."""
    
    def __init__(self):
        self.logger = logger.getChild('MockDataGenerator')
    
    def generate_market_data(self) -> Dict[str, Any]:
        """Generate mock market data."""
        return {
            'symbol': 'BTCUSDT',
            'exchange': 'binance',
            'timeframe': '1m',
            'data': [
                {'timestamp': '2024-01-01T00:00:00Z', 'open': 50000, 'high': 51000, 'low': 49000, 'close': 50500, 'volume': 1000},
                {'timestamp': '2024-01-01T00:01:00Z', 'open': 50500, 'high': 51500, 'low': 49500, 'close': 51000, 'volume': 1200},
                {'timestamp': '2024-01-01T00:02:00Z', 'open': 51000, 'high': 52000, 'low': 50000, 'close': 51500, 'volume': 1100}
            ],
            'metadata': {
                'data_points': 3,
                'time_range': '2024-01-01T00:00:00Z to 2024-01-01T00:02:00Z',
                'data_quality_score': 0.95
            }
        }
    
    def generate_data_quality_report(self) -> Dict[str, Any]:
        """Generate mock data quality report."""
        return {
            'overall_score': 0.95,
            'completeness': 0.98,
            'accuracy': 0.92,
            'consistency': 0.96,
            'timeliness': 0.94,
            'issues_found': [],
            'recommendations': ['Data quality is excellent']
        }
    
    def generate_collection_metadata(self) -> Dict[str, Any]:
        """Generate mock collection metadata."""
        return {
            'collection_timestamp': datetime.now().isoformat(),
            'data_source': 'binance_api',
            'collection_method': 'websocket_stream',
            'collection_duration': 30.5,
            'data_points_collected': 3
        }
    
    def generate_sr_levels(self) -> List[Dict[str, Any]]:
        """Generate mock SR levels."""
        return [
            {'level': 50000, 'type': 'resistance', 'strength': 0.8, 'confidence': 0.85},
            {'level': 48000, 'type': 'support', 'strength': 0.9, 'confidence': 0.90},
            {'level': 52000, 'type': 'resistance', 'strength': 0.7, 'confidence': 0.75}
        ]
    
    def generate_sr_metadata(self) -> Dict[str, Any]:
        """Generate mock SR metadata."""
        return {
            'detection_method': 'toolbox_enhanced',
            'levels_count': 3,
            'confidence_scores': [0.85, 0.90, 0.75],
            'detection_timestamp': datetime.now().isoformat()
        }
    
    def generate_regimes(self) -> Dict[str, Any]:
        """Generate mock regimes."""
        return {
            'regime_0': {
                'type': 'trending',
                'data': [{'price': 50000, 'volume': 1000}],
                'targets': [0.1, 0.2, 0.3],
                'confidence': 0.85
            },
            'regime_1': {
                'type': 'ranging',
                'data': [{'price': 50500, 'volume': 1200}],
                'targets': [0.2, 0.3, 0.4],
                'confidence': 0.90
            },
            'regime_2': {
                'type': 'volatile',
                'data': [{'price': 51000, 'volume': 1100}],
                'targets': [0.3, 0.4, 0.5],
                'confidence': 0.80
            }
        }
    
    def generate_regime_metadata(self) -> Dict[str, Any]:
        """Generate mock regime metadata."""
        return {
            'regime_count': 3,
            'regime_types': ['trending', 'ranging', 'volatile'],
            'regime_confidence': [0.85, 0.90, 0.80],
            'definition_timestamp': datetime.now().isoformat()
        }
    
    def generate_engineered_features(self) -> Dict[str, Any]:
        """Generate mock engineered features."""
        return {
            'technical_indicators': {'rsi': [50, 55, 60], 'macd': [0.1, 0.2, 0.3]},
            'statistical_features': {'mean': 50500, 'std': 1000, 'skew': 0.1},
            'lag_features': {'lag_1': [50000, 50500], 'lag_2': [49500, 50000]},
            'feature_count': 10
        }
    
    def generate_feature_metadata(self) -> Dict[str, Any]:
        """Generate mock feature metadata."""
        return {
            'feature_types': ['technical_indicators', 'statistical_features', 'lag_features'],
            'total_features': 10,
            'feature_engineering_timestamp': datetime.now().isoformat()
        }
    
    def generate_selected_features(self) -> List[str]:
        """Generate mock selected features."""
        return ['rsi', 'macd', 'mean', 'std', 'lag_1', 'lag_2']
    
    def generate_selection_metadata(self) -> Dict[str, Any]:
        """Generate mock selection metadata."""
        return {
            'selection_method': 'mutual_info',
            'selected_count': 6,
            'total_available': 10,
            'selection_timestamp': datetime.now().isoformat()
        }
    
    def generate_analyst_models(self) -> Dict[str, Any]:
        """Generate mock analyst models."""
        return {
            'regime_0': {
                'model_type': 'analyst_enhancement',
                'accuracy': 0.85,
                'multi_output_predictions': {
                    'price_prediction': [50500, 51000, 51500],
                    'probability': [0.8, 0.85, 0.9],
                    'risk': [0.1, 0.15, 0.2]
                }
            },
            'regime_1': {
                'model_type': 'analyst_enhancement',
                'accuracy': 0.88,
                'multi_output_predictions': {
                    'price_prediction': [50800, 51200, 51600],
                    'probability': [0.82, 0.87, 0.92],
                    'risk': [0.12, 0.17, 0.22]
                }
            }
        }
    
    def generate_analyst_metadata(self) -> Dict[str, Any]:
        """Generate mock analyst metadata."""
        return {
            'regimes_trained': ['regime_0', 'regime_1'],
            'multi_output_enabled': True,
            'multi_output_types': ['price_prediction', 'probability', 'risk'],
            'training_timestamp': datetime.now().isoformat()
        }
    
    def generate_general_model(self) -> Dict[str, Any]:
        """Generate mock general model."""
        return {
            'model_type': 'unified_regime_intelligence',
            'accuracy': 0.82,
            'regimes_used': ['regime_0', 'regime_1', 'regime_2'],
            'predictions': [0.15, 0.25, 0.35]
        }
    
    def generate_general_model_metadata(self) -> Dict[str, Any]:
        """Generate mock general model metadata."""
        return {
            'model_type': 'unified_regime_intelligence',
            'regimes_used': ['regime_0', 'regime_1', 'regime_2'],
            'training_timestamp': datetime.now().isoformat()
        }
    
    def generate_tactician_models(self) -> Dict[str, Any]:
        """Generate mock tactician models."""
        return {
            'regime_0': {
                'model_type': 'tactician_specialist',
                'accuracy': 0.90,
                'analyst_integration': True,
                'multi_output_predictions': {
                    'price_prediction': [50600, 51100, 51600],
                    'probability': [0.85, 0.90, 0.95],
                    'risk': [0.08, 0.12, 0.18]
                }
            },
            'regime_1': {
                'model_type': 'tactician_specialist',
                'accuracy': 0.92,
                'analyst_integration': True,
                'multi_output_predictions': {
                    'price_prediction': [50900, 51300, 51700],
                    'probability': [0.87, 0.92, 0.97],
                    'risk': [0.10, 0.14, 0.20]
                }
            }
        }
    
    def generate_tactician_metadata(self) -> Dict[str, Any]:
        """Generate mock tactician metadata."""
        return {
            'regimes_trained': ['regime_0', 'regime_1'],
            'analyst_integration': True,
            'multi_output_enabled': True,
            'multi_output_types': ['price_prediction', 'probability', 'risk'],
            'training_timestamp': datetime.now().isoformat()
        }
    
    def generate_backtesting_results(self) -> Dict[str, Any]:
        """Generate mock backtesting results."""
        return {
            'analyst_regime_0': {
                'sharpe_ratio': 1.5,
                'max_drawdown': 0.05,
                'total_return': 0.15,
                'win_rate': 0.65
            },
            'general_model': {
                'sharpe_ratio': 1.3,
                'max_drawdown': 0.08,
                'total_return': 0.12,
                'win_rate': 0.60
            },
            'tactician_regime_0': {
                'sharpe_ratio': 1.8,
                'max_drawdown': 0.04,
                'total_return': 0.18,
                'win_rate': 0.70
            }
        }
    
    def generate_validation_results(self) -> Dict[str, Any]:
        """Generate mock validation results."""
        return {
            'analyst_validation': {'passed': True, 'score': 0.85},
            'general_model_validation': {'passed': True, 'score': 0.80},
            'tactician_validation': {'passed': True, 'score': 0.88}
        }
    
    def generate_validation_metadata(self) -> Dict[str, Any]:
        """Generate mock validation metadata."""
        return {
            'models_validated': 3,
            'validation_timestamp': datetime.now().isoformat(),
            'overall_performance': {
                'average_sharpe': 1.4,
                'average_return': 0.12,
                'average_win_rate': 0.63
            }
        }


# Global instance for easy access
data_flow_tester = ComprehensiveDataFlowTester()


# Convenience functions
def test_pipeline_data_flow(pipeline_data: Dict[str, Any]) -> Dict[str, Any]:
    """Test complete pipeline data flow."""
    return data_flow_tester.test_complete_pipeline_data_flow(pipeline_data)


def generate_mock_pipeline_data() -> Dict[str, Any]:
    """Generate mock pipeline data for testing."""
    return data_flow_tester.generate_mock_pipeline_data()


def validate_data_structure(data: Any, expected_structure: Dict[str, Any], data_name: str) -> Dict[str, Any]:
    """Validate data structure."""
    return data_flow_tester.validate_data_structure(data, expected_structure, data_name)


# Example usage
if __name__ == "__main__":
    # Example: Test data flow
    print("🔧 Data Flow Testing Examples")
    print("=" * 50)
    
    # Generate mock data
    mock_data = generate_mock_pipeline_data()
    print(f"Generated mock data with {len(mock_data)} data sections")
    
    # Test complete pipeline data flow
    test_results = test_pipeline_data_flow(mock_data)
    print(f"Data flow test overall status: {'✅ PASSED' if test_results.get('overall_passed', False) else '❌ FAILED'}")
    
    # Generate report
    report = data_flow_tester.generate_data_flow_report(test_results)
    print("\n" + report)
    
    print("✅ Data flow testing examples completed")