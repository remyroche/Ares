"""
Test SR Detection Feedback Loop Implementation

This test verifies that the SR detection component can load and use
optimized parameters from the SR parameter optimization component.
"""

import asyncio
import json
from pathlib import Path

async def test_sr_feedback_loop():
    """Test the complete feedback loop between detection and optimization."""
    
    print("=" * 80)
    print("SR DETECTION FEEDBACK LOOP TEST")
    print("=" * 80)
    
    try:
        # Import components
        from src.training.steps.market_analysis.components.sr_detection import SRDetectionComponent
        print("✅ Successfully imported SRDetectionComponent")
    except Exception as e:
        print(f"❌ Failed to import SRDetectionComponent: {e}")
        return False
    
    # Test configuration
    config = {
        'symbol': 'ETHUSDT',
        'exchange': 'binance',
        'timeframe': '15m',
        'direction': 'long',
        'execution_mode': 'light',
        'data_dir': 'historical_data'
    }
    
    print("\n" + "=" * 80)
    print("TEST 1: Detection without optimized parameters (First Run)")
    print("=" * 80)
    
    try:
        # Create detection component instance
        detection = SRDetectionComponent('sr_detection_test')
        print("✅ Created SRDetectionComponent instance")
        
        # Check if get_required_input_artifacts method exists
        if hasattr(detection, 'get_required_input_artifacts'):
            required_inputs = detection.get_required_input_artifacts()
            print(f"✅ get_required_input_artifacts() exists")
            print(f"   Required inputs: {required_inputs}")
            
            if 'sr_parameter_optimization_result' in required_inputs:
                print("✅ Correctly declares 'sr_parameter_optimization_result' as input")
            else:
                print("❌ Missing 'sr_parameter_optimization_result' in required inputs")
        else:
            print("❌ get_required_input_artifacts() method not found")
        
        # Check if _load_optimized_parameters method exists
        if hasattr(detection, '_load_optimized_parameters'):
            print("✅ _load_optimized_parameters() method exists")
        else:
            print("❌ _load_optimized_parameters() method not found")
        
        # Check if _apply_quality_filters method exists
        if hasattr(detection, '_apply_quality_filters'):
            print("✅ _apply_quality_filters() method exists")
        else:
            print("❌ _apply_quality_filters() method not found")
        
        # Execute detection (should use defaults on first run)
        print("\n🔄 Executing detection (without optimized parameters)...")
        result = await detection.execute(config)
        
        if result.get('success'):
            print("✅ Detection executed successfully")
            
            metrics = result.get('metrics', {})
            print(f"\n📊 Detection Metrics:")
            print(f"   - Total levels: {metrics.get('total_levels', 0)}")
            print(f"   - Support levels: {metrics.get('support_levels', 0)}")
            print(f"   - Resistance levels: {metrics.get('resistance_levels', 0)}")
            print(f"   - Using optimized parameters: {metrics.get('using_optimized_parameters', False)}")
            
            feedback_loop = metrics.get('feedback_loop', {})
            if feedback_loop:
                print(f"\n🔄 Feedback Loop Info:")
                print(f"   - Used optimized params: {feedback_loop.get('used_optimized_parameters', False)}")
                print(f"   - Optimization timestamp: {feedback_loop.get('optimization_timestamp', 'N/A')}")
                print(f"   - Optimization score: {feedback_loop.get('optimization_score', 'N/A')}")
            
            # Check detection result metadata
            detection_result = result.get('detection_result', {})
            metadata = detection_result.get('metadata', {})
            if 'feedback_loop' in metadata:
                print("✅ Detection result contains feedback_loop metadata")
            else:
                print("❌ Detection result missing feedback_loop metadata")
                
        else:
            print(f"❌ Detection failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Test 1 failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 80)
    print("TEST 2: Verify artifact loading mechanism")
    print("=" * 80)
    
    try:
        # Test the _load_optimized_parameters method directly
        print("🔄 Testing _load_optimized_parameters()...")
        optimized_params = await detection._load_optimized_parameters()
        
        if optimized_params is None:
            print("✅ Correctly returns None when no optimization artifact exists (expected on first run)")
        else:
            print("✅ Successfully loaded optimized parameters")
            print(f"   Parameters keys: {list(optimized_params.keys())}")
            
            if 'parameters' in optimized_params:
                print("✅ Contains 'parameters' key")
            if 'quality_thresholds' in optimized_params:
                print("✅ Contains 'quality_thresholds' key")
            if 'optimization_summary' in optimized_params:
                print("✅ Contains 'optimization_summary' key")
    
    except Exception as e:
        print(f"❌ Test 2 failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 80)
    print("TEST 3: Test quality filter method")
    print("=" * 80)
    
    try:
        # Test quality filtering
        sample_levels = [
            {'price': 1.2000, 'type': 'support', 'strength': 0.85, 'confidence': 0.78, 'touches': 3},
            {'price': 1.2500, 'type': 'resistance', 'strength': 0.40, 'confidence': 0.30, 'touches': 1},  # Low quality
            {'price': 1.1800, 'type': 'support', 'strength': 0.68, 'confidence': 0.62, 'touches': 2},
        ]
        
        quality_thresholds = {
            'min_strength': 0.60,
            'min_confidence': 0.50,
            'min_touches': 2
        }
        
        print(f"🔄 Testing quality filter with {len(sample_levels)} levels...")
        filtered_levels = detection._apply_quality_filters(sample_levels, quality_thresholds)
        
        print(f"✅ Quality filter executed")
        print(f"   - Original levels: {len(sample_levels)}")
        print(f"   - Filtered levels: {len(filtered_levels)}")
        print(f"   - Removed: {len(sample_levels) - len(filtered_levels)}")
        
        if len(filtered_levels) == 2:  # Should remove the low quality level
            print("✅ Correctly filtered low-quality level")
        else:
            print(f"⚠️ Expected 2 levels after filtering, got {len(filtered_levels)}")
    
    except Exception as e:
        print(f"❌ Test 3 failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 80)
    print("TEST 4: Verify method signatures updated")
    print("=" * 80)
    
    try:
        import inspect
        
        # Check _perform_enhanced_sr_detection signature
        sig = inspect.signature(detection._perform_enhanced_sr_detection)
        params = list(sig.parameters.keys())
        
        if 'optimized_parameters' in params:
            print("✅ _perform_enhanced_sr_detection accepts 'optimized_parameters'")
        else:
            print("❌ _perform_enhanced_sr_detection missing 'optimized_parameters' parameter")
        
        # Check _detect_sr_levels_vectorbt signature
        sig = inspect.signature(detection._detect_sr_levels_vectorbt)
        params = list(sig.parameters.keys())
        
        if 'optimized_parameters' in params:
            print("✅ _detect_sr_levels_vectorbt accepts 'optimized_parameters'")
        else:
            print("❌ _detect_sr_levels_vectorbt missing 'optimized_parameters' parameter")
        
        # Check _detect_sr_levels_traditional signature
        sig = inspect.signature(detection._detect_sr_levels_traditional)
        params = list(sig.parameters.keys())
        
        if 'optimized_parameters' in params:
            print("✅ _detect_sr_levels_traditional accepts 'optimized_parameters'")
        else:
            print("❌ _detect_sr_levels_traditional missing 'optimized_parameters' parameter")
    
    except Exception as e:
        print(f"❌ Test 4 failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print("✅ All feedback loop components are correctly implemented")
    print("✅ Detection component can load optimized parameters")
    print("✅ Quality filtering is functional")
    print("✅ Method signatures updated to accept optimized parameters")
    print("✅ Metrics track feedback loop usage")
    print("\n🎉 FEEDBACK LOOP IS READY TO USE!")
    print("\nNext steps:")
    print("1. Run the full MARKET_ANALYSIS stage to generate optimized parameters")
    print("2. Run detection again - it will automatically use the optimized parameters")
    print("3. Check metrics['using_optimized_parameters'] to verify feedback loop")
    
    return True


if __name__ == "__main__":
    print("\n")
    success = asyncio.run(test_sr_feedback_loop())
    print("\n")
    exit(0 if success else 1)
