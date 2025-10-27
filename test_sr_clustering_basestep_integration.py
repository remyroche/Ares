#!/usr/bin/env python3
"""
Test script to verify SR Clustering Component integration with BaseStep.

This script tests:
1. SRClusteringComponent inheritance from BaseStep
2. Artifact saving and loading functionality
3. Integration validation
4. Required artifacts production
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.training.steps.market_analysis.components.sr_clustering import SRClusteringComponent
from src.training.steps.base_step import BaseStep


async def test_sr_clustering_basestep_integration():
    """Test SR Clustering Component integration with BaseStep."""
    print("🧪 Testing SR Clustering Component BaseStep Integration")
    print("=" * 60)
    
    # Test 1: Component instantiation and inheritance
    print("\n1. Testing component instantiation and inheritance...")
    try:
        component = SRClusteringComponent(step_name="test_sr_clustering")
        
        # Check inheritance
        is_basestep = isinstance(component, BaseStep)
        print(f"   ✅ Inherits from BaseStep: {is_basestep}")
        
        # Check required attributes
        has_step_name = hasattr(component, 'step_name') and component.step_name == "test_sr_clustering"
        has_artifact_manager = hasattr(component, 'artifact_manager') and component.artifact_manager is not None
        has_logger = hasattr(component, 'logger') and component.logger is not None
        
        print(f"   ✅ Has step_name: {has_step_name}")
        print(f"   ✅ Has artifact_manager: {has_artifact_manager}")
        print(f"   ✅ Has logger: {has_logger}")
        
        if not all([is_basestep, has_step_name, has_artifact_manager, has_logger]):
            print("   ❌ Component instantiation failed")
            return False
            
    except Exception as e:
        print(f"   ❌ Component instantiation failed: {e}")
        return False
    
    # Test 2: Integration validation
    print("\n2. Testing integration validation...")
    try:
        validation_results = component._validate_basestep_integration()
        
        print(f"   Integration validation results:")
        for key, value in validation_results.items():
            status = "✅" if value else "❌"
            print(f"   {status} {key}: {value}")
        
        if validation_results['integration_valid']:
            print("   ✅ Integration validation passed")
        else:
            print("   ❌ Integration validation failed")
            return False
            
    except Exception as e:
        print(f"   ❌ Integration validation failed: {e}")
        return False
    
    # Test 3: Required artifacts method
    print("\n3. Testing required artifacts method...")
    try:
        required_artifacts = component.get_required_artifacts()
        expected_artifacts = ['sr_clustering_result', 'sr_levels_dictionary']
        
        print(f"   Required artifacts: {required_artifacts}")
        print(f"   Expected artifacts: {expected_artifacts}")
        
        if set(required_artifacts) == set(expected_artifacts):
            print("   ✅ Required artifacts method works correctly")
        else:
            print("   ❌ Required artifacts method failed")
            return False
            
    except Exception as e:
        print(f"   ❌ Required artifacts method failed: {e}")
        return False
    
    # Test 4: Artifact saving functionality
    print("\n4. Testing artifact saving functionality...")
    try:
        # Test data
        test_data = {
            'test_clusters': [
                {'cluster_id': 1, 'levels': [1.2000, 1.2050], 'strength': 0.85},
                {'cluster_id': 2, 'levels': [1.2500, 1.2550], 'strength': 0.72}
            ],
            'total_clusters': 2,
            'clustering_efficiency': 0.6
        }
        
        # Save artifact
        artifact_path = component._save_artifact(
            data=test_data,
            artifact_name='test_sr_clustering_result',
            artifact_type='data',
            metadata={'test': True, 'created_by': 'integration_test'}
        )
        
        print(f"   ✅ Artifact saved to: {artifact_path}")
        
        # Verify artifact was saved
        if artifact_path and os.path.exists(artifact_path):
            print("   ✅ Artifact file exists")
        else:
            print("   ❌ Artifact file not found")
            return False
            
    except Exception as e:
        print(f"   ❌ Artifact saving failed: {e}")
        return False
    
    # Test 5: Artifact loading functionality
    print("\n5. Testing artifact loading functionality...")
    try:
        # Load artifact
        loaded_data = component._get_artifact(
            artifact_name='test_sr_clustering_result',
            artifact_type='data'
        )
        
        if loaded_data and loaded_data.get('total_clusters') == 2:
            print("   ✅ Artifact loaded successfully")
            print(f"   ✅ Loaded data matches: {loaded_data.get('total_clusters')} clusters")
        else:
            print("   ❌ Artifact loading failed or data mismatch")
            return False
            
    except Exception as e:
        print(f"   ❌ Artifact loading failed: {e}")
        return False
    
    # Test 6: SR levels loading functionality
    print("\n6. Testing SR levels loading functionality...")
    try:
        # Test _get_sr_levels method
        sr_levels = component._get_sr_levels(
            symbol='ETHUSDT',
            exchange='binance',
            timeframe='15m',
            direction='longs'
        )
        
        print(f"   ✅ SR levels method executed successfully")
        print(f"   ✅ SR levels structure: {type(sr_levels)}")
        
        if isinstance(sr_levels, dict) and 'levels' in sr_levels:
            print(f"   ✅ SR levels contains 'levels' key")
        else:
            print("   ⚠️  SR levels structure may be unexpected (this is OK for test)")
            
    except Exception as e:
        print(f"   ❌ SR levels loading failed: {e}")
        return False
    
    # Test 7: Component execution (simplified)
    print("\n7. Testing component execution...")
    try:
        # Create test config
        test_config = {
            'symbol': 'ETHUSDT',
            'exchange': 'binance',
            'timeframe': '15m',
            'direction': 'longs',
            'execution_mode': 'light',
            'enable_hardware_optimization': False,  # Disable for test
            'enable_vectorbt_optimization': False,  # Disable for test
            'clustering_algorithm': 'proximity'  # Use simple algorithm
        }
        
        # Execute component
        result = await component.execute(test_config)
        
        print(f"   ✅ Component execution completed")
        print(f"   ✅ Success: {result.get('success', False)}")
        print(f"   ✅ Artifacts created: {len(result.get('artifacts', []))}")
        print(f"   ✅ Metrics available: {len(result.get('metrics', {}))}")
        
        if result.get('success'):
            print("   ✅ Component execution successful")
        else:
            print(f"   ❌ Component execution failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"   ❌ Component execution failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 All tests passed! SR Clustering Component is fully integrated with BaseStep")
    return True


async def main():
    """Main test function."""
    try:
        success = await test_sr_clustering_basestep_integration()
        if success:
            print("\n✅ Integration test completed successfully")
            sys.exit(0)
        else:
            print("\n❌ Integration test failed")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())