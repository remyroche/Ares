#!/usr/bin/env python3
"""
Test script for the Sequential Fixer with proper import handling.
"""

import sys
import os
from pathlib import Path

# Add code_quality to path
sys.path.insert(0, str(Path(__file__).parent / "code_quality"))

# Now import and run the sequential fixer
try:
    from fixers.sequential_fixer import SequentialFixer
    from core.config import get_default_config
    
    print("✅ Successfully imported SequentialFixer")
    
    # Get default config
    config = get_default_config()
    print("✅ Successfully loaded default config")
    
    # Create fixer instance
    fixer = SequentialFixer(config)
    print("✅ Successfully created SequentialFixer instance")
    
    # Run on a small subset first
    test_target = "/workspace/src/utils"
    if os.path.exists(test_target):
        print(f"🚀 Running sequential fixer on {test_target}")
        results = fixer.run_pipeline(
            target=test_target,
            output_dir="/workspace/code_quality_reports",
            create_backups=False
        )
        print("✅ Sequential fixer completed successfully")
        print(f"Overall status: {results['summary']['overall_status']}")
    else:
        print(f"❌ Test target {test_target} does not exist")
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()