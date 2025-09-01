#!/usr/bin/env python3
"""
Validation Script for SR Levels Manager Imports

This script validates that sr_levels_manager properly imports and uses
the SR calculation logic from sr_breakout_predictor.py
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
import project_root = Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def validate_imports():
    pass
    pass
    """Validate that all required imports are working."""
    print("🔍 Validating SR Levels Manager Imports")
    print("=" * 50)

    try:
        # Test 1: Import SRBreakoutPredictor
    except Exception as e:
        pass
    except Exception as e:
        pass
        print("\\\n📦 Test 1: Importing SRBreakoutPredictor")
        from src.tactician.sr_breakout_predictor import SRBreakoutPredictor
import print
        print("✅ SRBreakoutPredictor imported successfully")

        # Test 2: Import SRLevelsManager
        print("\\\n📦 Test 2: Importing SRLevelsManager")
        print("✅ SRLevelsManager imported successfully")

        # Test 3: Import SR Trading Intelligence
        print("\\\n📦 Test 3: Importing SR Trading Intelligence")
        from src.trading.sr_trading_intelligence import SRTradingIntelligence
import print
        print("✅ SRTradingIntelligence imported successfully")

        # Test 4: Check SRBreakoutPredictor methods
        print("\\\n🔧 Test 4: Checking SRBreakoutPredictor Methods")
        predictor_methods = [
            '_detect_support_levels',
            '_detect_resistance_levels',
            '_detect_fractal_support_levels',
            '_detect_fractal_resistance_levels',
            '_detect_volume_support_levels',
            '_detect_volume_resistance_levels',
            '_detect_pivot_support_levels',
            '_detect_pivot_resistance_levels',
            '_detect_atr_support_levels',
            '_detect_atr_resistance_levels',
            'get_sr_context',
            'cluster_sr_levels_dbscan',
            'calculate_comprehensive_strength'
        ]

        for method in predictor_methods:
    pass
    pass
            if hasattr(SRBreakoutPredictor, method):
    pass
    pass
                print(f"✅ {method} - Available")
            else:
                print(f"❌ {method} - Missing")

        # Test 5: Check SRLevelsManager methods
        print("\\\n🔧 Test 5: Checking SRLevelsManager Methods")
        manager_methods = [
            'calculate_sr_levels_from_backtest',
            'calculate_sr_levels_with_method',
            'update_levels_with_live_data',
            'get_sr_levels_for_trading',
            'compare_price_vs_vwap_predictions',
            '_create_sr_level_from_data',
            '_level_exists'
        ]

        for method in manager_methods:
    pass
    pass
            if hasattr(SRLevelsManager, method):
    pass
    pass
                print(f"✅ {method} - Available")
            else:
                print(f"❌ {method} - Missing")

        # Test 6: Check SR Trading Intelligence methods
        print("\\\n🔧 Test 6: Checking SR Trading Intelligence Methods")
        intelligence_methods = [
            'get_sr_levels_for_trading',
            'update_position',
            'close_position',
            '_generate_trading_intelligence',
            '_assess_risk',
            '_generate_position_recommendations'
        ]

        for method in intelligence_methods:
    pass
    pass
            if hasattr(SRTradingIntelligence, method):
    pass
    pass
                print(f"✅ {method} - Available")
            else:
                print(f"❌ {method} - Missing")

        print("\\\n✅ All import validations completed successfully!")
        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False

async def validate_integration():
    """Validate that the integration between components works."""
    print("\\\n🔗 Validating SR Levels Manager Integration")
    print("=" * 50)

    try:
        # Configuration
    except Exception as e:
        pass
    except Exception as e:
        pass
        config = {
            "sr_levels_manager": {
                "storage_path": "data/sr_levels_test",
                "max_levels": 20,
                "min_strength": 0.3
            },
            "sr_breakout_predictor": {
                "enable_detailed_reporting": False,
                "max_sr_levels": 10,
                "min_sr_strength": 0.3
            }
        }

        # Test 1: Initialize SRBreakoutPredictor
        print("\\\n🔧 Test 1: Initializing SRBreakoutPredictor")
        from src.tactician.sr_breakout_predictor import SRBreakoutPredictor
import predictor = SRBreakoutPredictor
        predictor = SRBreakoutPredictor(config)
        if await predictor.initialize():
    pass
    pass
            print("✅ SRBreakoutPredictor initialized successfully")
        else:
            print("❌ SRBreakoutPredictor initialization failed")
            return False

        # Test 2: Initialize SRLevelsManager
        print("\\\n🔧 Test 2: Initializing SRLevelsManager")
        from src.tactician.sr_levels_manager import create_sr_levels_manager
import sr_manager = await create_sr_levels_manager
        sr_manager = await create_sr_levels_manager(config)
        if sr_manager:
    pass
    pass
            print("✅ SRLevelsManager initialized successfully")
        else:
            print("❌ SRLevelsManager initialization failed")
            return False

        # Test 3: Check that SRLevelsManager has access to SRBreakoutPredictor
        print("\\\n🔧 Test 3: Checking SRBreakoutPredictor Access")
        if hasattr(sr_manager, 'sr_predictor') and sr_manager.sr_predictor is not None:
    pass
    pass
            print("✅ SRLevelsManager has access to SRBreakoutPredictor")
        else:
            print("❌ SRLevelsManager does not have access to SRBreakoutPredictor")
            return False

        # Test 4: Check that SRBreakoutPredictor methods are accessible
        print("\\\n🔧 Test 4: Checking Method Accessibility")
        required_methods = [
            '_detect_support_levels',
            '_detect_resistance_levels',
            'get_sr_context'
        ]

        for method in required_methods:
    pass
    pass
            if hasattr(sr_manager.sr_predictor, method):
    pass
    pass
                print(f"✅ {method} - Accessible")
            else:
                print(f"❌ {method} - Not accessible")
                return False

        print("\\\n✅ All integration validations completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Integration validation error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main validation function."""
    print("🧪 SR Levels Manager Import Validation")
    print("=" * 60)

    # Validate imports
    imports_ok = validate_imports()

    if imports_ok:
    pass
    pass
        # Validate integration
        integration_ok = await validate_integration()

        if integration_ok:
    pass
    pass
            print("\\\n🎉 All validations passed! SR Levels Manager is properly integrated.")
            print("\\\n📋 Summary:")
            print("   ✅ All required imports are working")
            print("   ✅ SRBreakoutPredictor methods are accessible")
            print("   ✅ SRLevelsManager can use SR calculation logic")
            print("   ✅ Integration between components is functional")
        else:
            print("\\\n💥 Integration validation failed.")
            return False
    else:
        print("\\\n💥 Import validation failed.")
        return False

    return True

if __name__ == "__main__":
    pass
    pass
    asyncio.run(main())