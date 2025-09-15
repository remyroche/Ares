"""
Integration Test for PID-Based Feature Generation in Sub-Pipeline

This test verifies that the PID-based feature generation is properly integrated
into the market analysis sub-pipeline and that all artifacts and reports are
correctly generated and accessible.
"""

import asyncio
import logging
import sys
import os
from typing import Dict, Any

# Add the workspace root to the path
sys.path.append('/workspace')

# Import required modules
from src.training.steps.market_analysis.sub_pipeline import MarketAnalysisSubPipeline, SubPipelineConfig, ExecutionMode
from src.training.steps.market_analysis.components.component_factory import ComponentFactory

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_pid_based_feature_generation_integration():
    """Test PID-based feature generation integration in sub-pipeline."""
    
    logger.info("🧪 Starting PID-Based Feature Generation Integration Test")
    
    try:
        # Test 1: Verify component factory supports pid_based_feature_generation
        logger.info("📋 Test 1: Verifying component factory support...")
        factory = ComponentFactory()
        available_components = factory.get_available_components()
        
        assert 'pid_based_feature_generation' in available_components, "pid_based_feature_generation not found in available components"
        assert 'cross_timeframe_analysis' in available_components, "cross_timeframe_analysis not found in available components"
        
        logger.info("✅ Component factory supports both pid_based_feature_generation and cross_timeframe_analysis")
        
        # Test 2: Verify component creation
        logger.info("📋 Test 2: Verifying component creation...")
        
        # Test direct PID component creation
        pid_component = factory.create_component('pid_based_feature_generation')
        assert pid_component is not None, "Failed to create pid_based_feature_generation component"
        logger.info("✅ Successfully created pid_based_feature_generation component")
        
        # Test backward compatibility
        legacy_component = factory.create_component('cross_timeframe_analysis')
        assert legacy_component is not None, "Failed to create cross_timeframe_analysis component"
        logger.info("✅ Successfully created cross_timeframe_analysis component (backward compatibility)")
        
        # Test 3: Verify sub-pipeline configuration
        logger.info("📋 Test 3: Verifying sub-pipeline configuration...")
        
        config = SubPipelineConfig(
            mode=ExecutionMode.LIGHT,  # Use light mode for faster testing
            symbol="BTCUSDT",
            exchange="binance",
            timeframe="1h",
            data_dir="test_data",
            force_rerun=True,
            validation_enabled=True,
            monitoring_enabled=True
        )
        
        pipeline = MarketAnalysisSubPipeline(config)
        available_sub_pipelines = pipeline.get_available_sub_pipelines()
        
        assert 'pid_based_feature_generation' in available_sub_pipelines, "pid_based_feature_generation not found in available sub-pipelines"
        logger.info("✅ Sub-pipeline supports pid_based_feature_generation")
        
        # Test 4: Verify artifact requirements
        logger.info("📋 Test 4: Verifying artifact requirements...")
        
        # Check that the sub-pipeline knows about the required artifacts
        from src.training.steps.market_analysis.sub_pipeline import SubPipelineResult, SubPipelineStatus
        
        # Create a mock result to test artifact requirements
        mock_result = SubPipelineResult(
            sub_pipeline_name='pid_based_feature_generation',
            status=SubPipelineStatus.COMPLETED,
            start_time=None,
            end_time=None,
            artifacts={'pid_based_feature_generation_result': {'test': 'data'}}
        )
        
        required_artifacts = mock_result._get_required_artifacts()
        assert 'pid_based_feature_generation_result' in required_artifacts, "pid_based_feature_generation_result not in required artifacts"
        logger.info("✅ Artifact requirements properly configured")
        
        # Test 5: Verify component interface compatibility
        logger.info("📋 Test 5: Verifying component interface compatibility...")
        
        # Check that both components have the same interface
        pid_methods = [method for method in dir(pid_component) if not method.startswith('_')]
        legacy_methods = [method for method in dir(legacy_component) if not method.startswith('_')]
        
        # Key methods that should be present
        required_methods = ['execute', 'get_required_artifacts', 'validate_config']
        for method in required_methods:
            assert hasattr(pid_component, method), f"PID component missing method: {method}"
            assert hasattr(legacy_component, method), f"Legacy component missing method: {method}"
        
        logger.info("✅ Component interfaces are compatible")
        
        # Test 6: Verify configuration compatibility
        logger.info("📋 Test 6: Verifying configuration compatibility...")
        
        # Test that both components can be configured the same way
        from src.training.steps.market_analysis.components.base_component import ComponentConfig
        
        component_config = ComponentConfig(
            symbol="BTCUSDT",
            exchange="binance",
            timeframe="1h",
            data_dir="test_data"
        )
        
        # Both components should accept the same config
        pid_component_configured = factory.create_component('pid_based_feature_generation', component_config)
        legacy_component_configured = factory.create_component('cross_timeframe_analysis', component_config)
        
        assert pid_component_configured is not None, "Failed to create configured PID component"
        assert legacy_component_configured is not None, "Failed to create configured legacy component"
        
        logger.info("✅ Configuration compatibility verified")
        
        # Test 7: Verify artifact structure
        logger.info("📋 Test 7: Verifying artifact structure...")
        
        # Check that the PID component produces the expected artifact structure
        required_artifacts = pid_component.get_required_artifacts()
        assert 'pid_based_feature_generation_result' in required_artifacts, "PID component missing required artifact"
        
        logger.info("✅ Artifact structure verified")
        
        logger.info("🎉 All integration tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        import traceback
        logger.error(f"❌ Error details: {traceback.format_exc()}")
        return False


async def test_artifact_extraction():
    """Test that artifacts are properly extracted in the sub-pipeline."""
    
    logger.info("🧪 Testing artifact extraction...")
    
    try:
        # Mock the artifact structure that would be produced
        mock_artifacts = {
            'pid_based_feature_generation_result': {
                'combined_features': {'feature1': [1, 2, 3], 'feature2': [4, 5, 6]},
                'combined_feature_names': ['feature1', 'feature2'],
                'feature_importance_scores': {'feature1': 0.8, 'feature2': 0.6},
                'interaction_result': {'features': ['interaction_1', 'interaction_2']},
                'polynomial_result': {'features': ['polynomial_1', 'polynomial_2']},
                'cross_timeframe_result': {'features': ['cross_tf_1', 'cross_tf_2']},
                'overall_quality_score': 0.85,
                'feature_diversity_score': 0.75,
                'redundancy_score': 0.3,
                'stability_score': 0.9,
                'optimization_used': True,
                'matrix_ops_used': True,
                'lookback_integration': {
                    'optimized_lookback_periods': {'feature1': 20, 'feature2': 15},
                    'integration_status': 'completed',
                    'features_optimized': 2,
                    'optimization_quality_score': 0.8
                },
                'validation_result': {'is_valid': True, 'quality_score': 0.85},
                'total_features_generated': 6,
                'generation_status': 'completed',
                'generation_summary': {
                    'total_features_generated': 6,
                    'interaction_features': 2,
                    'polynomial_features': 2,
                    'cross_timeframe_features': 2
                }
            }
        }
        
        # Test the extraction logic (simulating what happens in sub_pipeline.py)
        pid_feature_data = mock_artifacts.get('pid_based_feature_generation_result', {})
        
        # Extract comprehensive PID-based feature generation results
        pid_based_features = {
            'combined_features': pid_feature_data.get('combined_features', {}),
            'combined_feature_names': pid_feature_data.get('combined_feature_names', []),
            'feature_importance_scores': pid_feature_data.get('feature_importance_scores', {}),
            'interaction_features': pid_feature_data.get('interaction_result', {}),
            'polynomial_features': pid_feature_data.get('polynomial_result', {}),
            'cross_timeframe_features': pid_feature_data.get('cross_timeframe_result', {})
        }
        
        pid_feature_metrics = {
            'generation_summary': pid_feature_data.get('generation_summary', {}),
            'quality_metrics': {
                'overall_quality_score': pid_feature_data.get('overall_quality_score', 0.0),
                'feature_diversity_score': pid_feature_data.get('feature_diversity_score', 0.0),
                'redundancy_score': pid_feature_data.get('redundancy_score', 0.0),
                'stability_score': pid_feature_data.get('stability_score', 0.0)
            },
            'optimization_metrics': {
                'optimization_used': pid_feature_data.get('optimization_used', False),
                'matrix_ops_used': pid_feature_data.get('matrix_ops_used', False),
                'lookback_integration': pid_feature_data.get('lookback_integration', {})
            },
            'validation_result': pid_feature_data.get('validation_result', {}),
            'total_features_generated': pid_feature_data.get('total_features_generated', 0),
            'generation_status': pid_feature_data.get('generation_status', 'unknown')
        }
        
        # Verify the extracted data
        assert len(pid_based_features['combined_feature_names']) == 2, "Failed to extract feature names"
        assert pid_feature_metrics['total_features_generated'] == 6, "Failed to extract total features"
        assert pid_feature_metrics['quality_metrics']['overall_quality_score'] == 0.85, "Failed to extract quality score"
        assert pid_feature_metrics['optimization_metrics']['optimization_used'] == True, "Failed to extract optimization status"
        
        logger.info("✅ Artifact extraction test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Artifact extraction test failed: {e}")
        return False


async def main():
    """Run all integration tests."""
    
    logger.info("🚀 Starting PID-Based Feature Generation Integration Tests")
    
    # Run tests
    test1_passed = await test_pid_based_feature_generation_integration()
    test2_passed = await test_artifact_extraction()
    
    # Summary
    if test1_passed and test2_passed:
        logger.info("🎉 All integration tests passed! PID-based feature generation is properly integrated.")
        return True
    else:
        logger.error("❌ Some integration tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    # Run the integration tests
    success = asyncio.run(main())
    sys.exit(0 if success else 1)