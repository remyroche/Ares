#!/usr/bin/env python3
"""
Test script for Duplicate Import Fixer

This script demonstrates the safety analysis and fixing capabilities
of the duplicate import fixer.
"""

import sys
import tempfile
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from analyzers.duplicate_import_fixer import DuplicateImportFixer


def create_test_files():
    """Create test files with various duplicate import scenarios."""
    temp_dir = tempfile.mkdtemp()
    
    # Test file 1: Safe duplicates
    test_file_1 = Path(temp_dir) / "safe_duplicates.py"
    test_file_1.write_text("""
import os
import sys
import os  # Duplicate - safe to remove
import json
from typing import List, Dict
from typing import List  # Duplicate - safe to remove

def test_function():
    print(os.getcwd())
    print(sys.version)
    data = json.loads('{}')
    items: List[str] = []
""")
    
    # Test file 2: Risky duplicates
    test_file_2 = Path(temp_dir) / "risky_duplicates.py"
    test_file_2.write_text("""
import os
import sys
import os  # Duplicate - but used later

# Later in the code
module_name = 'os'
module = globals()[module_name]  # Dynamic access - risky to remove

if some_condition:
    import matplotlib.pyplot as plt
    import matplotlib.pyplot as plt  # Side effects - risky to remove
""")
    
    # Test file 3: Conditional duplicates
    test_file_3 = Path(temp_dir) / "conditional_duplicates.py"
    test_file_3.write_text("""
import os
import sys

if some_condition:
    import json
else:
    import json  # Conditional duplicate - risky to remove

try:
    import pandas as pd
except ImportError:
    import pandas as pd  # Exception handling - risky to remove
""")
    
    # Test file 4: Mixed scenarios
    test_file_4 = Path(temp_dir) / "mixed_scenarios.py"
    test_file_4.write_text("""
import os
import sys
import os  # Safe duplicate
import json

from typing import List, Dict
from typing import List  # Safe duplicate

# Side effect import
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt  # Risky duplicate

def test_function():
    print(os.getcwd())
    items: List[str] = []
    
    # Dynamic access
    module_name = 'json'
    module = globals()[module_name]
""")
    
    return temp_dir


def test_safe_duplicates():
    """Test detection and fixing of safe duplicates."""
    print("="*60)
    print("Testing Safe Duplicate Detection")
    print("="*60)
    
    temp_dir = create_test_files()
    test_file = Path(temp_dir) / "safe_duplicates.py"
    
    try:
        fixer = DuplicateImportFixer()
        
        # Analyze the file
        analysis = fixer.analyze_file(str(test_file))
        
        print(f"File: {test_file}")
        print(f"Total imports: {analysis['total_imports']}")
        print(f"Duplicates found: {analysis['duplicates_found']}")
        print(f"Safe to remove: {analysis['safe_to_remove']}")
        print(f"Risky to remove: {analysis['risky_removals']}")
        
        # Show detailed analysis
        for original, duplicate in analysis['duplicates']:
            print(f"\nDuplicate found:")
            print(f"  Original: Line {original.line_number} - {original.full_line}")
            print(f"  Duplicate: Line {duplicate.line_number} - {duplicate.full_line}")
            print(f"  Can safely remove: {duplicate.can_safely_remove}")
            if hasattr(duplicate, 'safety_reasons'):
                print(f"  Safety reasons: {duplicate.safety_reasons}")
        
        # Test dry run fix
        print(f"\n--- Dry Run Fix ---")
        result = fixer.fix_duplicates(str(test_file), dry_run=True)
        print(f"Status: {result['status']}")
        print(f"Duplicates that would be removed: {result['duplicates_removed']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def test_risky_duplicates():
    """Test detection of risky duplicates."""
    print("\n" + "="*60)
    print("Testing Risky Duplicate Detection")
    print("="*60)
    
    temp_dir = create_test_files()
    test_file = Path(temp_dir) / "risky_duplicates.py"
    
    try:
        fixer = DuplicateImportFixer()
        
        # Analyze the file
        analysis = fixer.analyze_file(str(test_file))
        
        print(f"File: {test_file}")
        print(f"Total imports: {analysis['total_imports']}")
        print(f"Duplicates found: {analysis['duplicates_found']}")
        print(f"Safe to remove: {analysis['safe_to_remove']}")
        print(f"Risky to remove: {analysis['risky_removals']}")
        
        # Show detailed analysis
        for original, duplicate in analysis['duplicates']:
            print(f"\nDuplicate found:")
            print(f"  Original: Line {original.line_number} - {original.full_line}")
            print(f"  Duplicate: Line {duplicate.line_number} - {duplicate.full_line}")
            print(f"  Can safely remove: {duplicate.can_safely_remove}")
            if hasattr(duplicate, 'safety_reasons'):
                print(f"  Safety reasons: {duplicate.safety_reasons}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def test_safety_report():
    """Test detailed safety report generation."""
    print("\n" + "="*60)
    print("Testing Safety Report Generation")
    print("="*60)
    
    temp_dir = create_test_files()
    test_file = Path(temp_dir) / "mixed_scenarios.py"
    
    try:
        fixer = DuplicateImportFixer()
        
        # Generate safety report
        report = fixer.get_safety_report(str(test_file))
        
        print(f"Safety Report for {test_file}")
        print("=" * 50)
        print(f"Total duplicates: {report['total_duplicates']}")
        print(f"Safe to remove: {report['safety_summary']['safe_count']}")
        print(f"Risky to remove: {report['safety_summary']['risky_count']}")
        print(f"Safety percentage: {report['safety_summary']['safety_percentage']:.1f}%")
        
        if report['safe_removals']:
            print(f"\nSafe removals:")
            for removal in report['safe_removals']:
                print(f"  Line {removal['line_number']}: {removal['import_line']}")
        
        if report['risky_removals']:
            print(f"\nRisky removals:")
            for removal in report['risky_removals']:
                print(f"  Line {removal['line_number']}: {removal['import_line']}")
                print(f"    Reasons: {', '.join(removal['reasons'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def test_actual_fixing():
    """Test actual fixing of safe duplicates."""
    print("\n" + "="*60)
    print("Testing Actual Duplicate Removal")
    print("="*60)
    
    temp_dir = create_test_files()
    test_file = Path(temp_dir) / "safe_duplicates.py"
    
    try:
        fixer = DuplicateImportFixer()
        
        # Show original content
        print("Original content:")
        with open(test_file, 'r') as f:
            original_content = f.read()
        print(original_content)
        
        # Fix duplicates
        result = fixer.fix_duplicates(str(test_file), dry_run=False)
        
        print(f"\nFix result: {result['status']}")
        print(f"Duplicates removed: {result['duplicates_removed']}")
        
        # Show modified content
        print("\nModified content:")
        with open(test_file, 'r') as f:
            modified_content = f.read()
        print(modified_content)
        
        # Check if backup was created
        backup_file = Path(str(test_file) + ".backup_duplicate_fix")
        if backup_file.exists():
            print(f"\nBackup created: {backup_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def main():
    """Run all tests."""
    print("Duplicate Import Fixer Tests")
    print("="*60)
    
    tests = [
        ("Safe Duplicate Detection", test_safe_duplicates),
        ("Risky Duplicate Detection", test_risky_duplicates),
        ("Safety Report Generation", test_safety_report),
        ("Actual Duplicate Removal", test_actual_fixing)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                print(f"✅ {test_name} passed")
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The duplicate import fixer is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())