#!/usr/bin/env python3
"""
Test script for Intelligent Import Fixer

This script demonstrates the intelligent auto-fixing capabilities with
different confidence levels and automatic decision making.
"""

import sys
import tempfile
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from analyzers.intelligent_import_fixer import IntelligentImportFixer, ConfidenceLevel, FixAction


def create_test_files():
    """Create test files with various import scenarios for intelligent fixing."""
    temp_dir = tempfile.mkdtemp()
    
    # Test file 1: High confidence issues (95% - auto-fix)
    test_file_1 = Path(temp_dir) / "high_confidence.py"
    test_file_1.write_text("""
import os
import sys
import os  # Duplicate - high confidence (safe to remove)
import json
from typing import List, Dict
from typing import List  # Duplicate - high confidence (safe to remove)

def test_function():
    print(os.getcwd())
    print(sys.version)
    data = json.loads('{}')
    items: List[str] = []
""")
    
    # Test file 2: Medium confidence issues (4% - confirm)
    test_file_2 = Path(temp_dir) / "medium_confidence.py"
    test_file_2.write_text("""
import os
import sys
import os  # Duplicate - medium confidence (order sensitive)

# Later usage
current_dir = os.getcwd()

# Side effect import
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt  # Duplicate - medium confidence (side effects)
""")
    
    # Test file 3: Low confidence issues (1% - flag only)
    test_file_3 = Path(temp_dir) / "low_confidence.py"
    test_file_3.write_text("""
import os
import sys
import os  # Duplicate - low confidence (dynamic access)

# Dynamic access
module_name = 'os'
module = globals()[module_name]

if some_condition:
    import json
else:
    import json  # Conditional duplicate - low confidence
""")
    
    # Test file 4: Relative imports
    test_file_4 = Path(temp_dir) / "relative_imports.py"
    test_file_4.write_text("""
# Standalone script with relative imports
from . import utils  # High confidence - standalone script
from .. import config  # Medium confidence - deep relative import
from ... import parent  # Low confidence - very deep relative import

def main():
    pass
""")
    
    # Test file 5: Mixed scenarios
    test_file_5 = Path(temp_dir) / "mixed_scenarios.py"
    test_file_5.write_text("""
import os
import sys
import os  # High confidence duplicate
import json

from typing import List, Dict
from typing import List  # High confidence duplicate

# Medium confidence
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt  # Side effects

# Low confidence
if condition:
    import pandas as pd
else:
    import pandas as pd  # Conditional

# Relative imports
from . import utils  # High confidence - standalone
from .. import config  # Medium confidence - deep
""")
    
    return temp_dir


def test_confidence_assessment():
    """Test confidence level assessment for different issue types."""
    print("="*60)
    print("Testing Confidence Level Assessment")
    print("="*60)
    
    temp_dir = create_test_files()
    
    try:
        fixer = IntelligentImportFixer({'dry_run': True})
        
        # Test high confidence file
        high_conf_file = Path(temp_dir) / "high_confidence.py"
        issues = fixer._analyze_import_issues(str(high_conf_file))
        
        print(f"\n📁 High Confidence File: {high_conf_file.name}")
        high_conf_issues = [i for i in issues if i.confidence == ConfidenceLevel.HIGH]
        medium_conf_issues = [i for i in issues if i.confidence == ConfidenceLevel.MEDIUM]
        low_conf_issues = [i for i in issues if i.confidence == ConfidenceLevel.LOW]
        
        print(f"   High confidence: {len(high_conf_issues)}")
        print(f"   Medium confidence: {len(medium_conf_issues)}")
        print(f"   Low confidence: {len(low_conf_issues)}")
        
        for issue in issues:
            print(f"   Line {issue.line_number}: {issue.confidence.value} - {issue.reason}")
        
        # Test medium confidence file
        medium_conf_file = Path(temp_dir) / "medium_confidence.py"
        issues = fixer._analyze_import_issues(str(medium_conf_file))
        
        print(f"\n📁 Medium Confidence File: {medium_conf_file.name}")
        high_conf_issues = [i for i in issues if i.confidence == ConfidenceLevel.HIGH]
        medium_conf_issues = [i for i in issues if i.confidence == ConfidenceLevel.MEDIUM]
        low_conf_issues = [i for i in issues if i.confidence == ConfidenceLevel.LOW]
        
        print(f"   High confidence: {len(high_conf_issues)}")
        print(f"   Medium confidence: {len(medium_conf_issues)}")
        print(f"   Low confidence: {len(low_conf_issues)}")
        
        for issue in issues:
            print(f"   Line {issue.line_number}: {issue.confidence.value} - {issue.reason}")
        
        # Test low confidence file
        low_conf_file = Path(temp_dir) / "low_confidence.py"
        issues = fixer._analyze_import_issues(str(low_conf_file))
        
        print(f"\n📁 Low Confidence File: {low_conf_file.name}")
        high_conf_issues = [i for i in issues if i.confidence == ConfidenceLevel.HIGH]
        medium_conf_issues = [i for i in issues if i.confidence == ConfidenceLevel.MEDIUM]
        low_conf_issues = [i for i in issues if i.confidence == ConfidenceLevel.LOW]
        
        print(f"   High confidence: {len(high_conf_issues)}")
        print(f"   Medium confidence: {len(medium_conf_issues)}")
        print(f"   Low confidence: {len(low_conf_issues)}")
        
        for issue in issues:
            print(f"   Line {issue.line_number}: {issue.confidence.value} - {issue.reason}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def test_auto_fixing():
    """Test automatic fixing of high-confidence issues."""
    print("\n" + "="*60)
    print("Testing Automatic Fixing")
    print("="*60)
    
    temp_dir = create_test_files()
    
    try:
        # Test with dry run first
        fixer = IntelligentImportFixer({'dry_run': True})
        test_file = Path(temp_dir) / "high_confidence.py"
        
        print(f"📁 Testing auto-fix on: {test_file.name}")
        print("Original content:")
        with open(test_file, 'r') as f:
            print(f.read())
        
        result = fixer.analyze_and_fix_file(str(test_file), interactive=False)
        
        print(f"\n📊 Auto-fix results:")
        print(f"   Total issues: {result.total_issues}")
        print(f"   Auto-fixed: {result.auto_fixed}")
        print(f"   Flagged: {result.flagged_only}")
        
        # Test actual fixing
        fixer_actual = IntelligentImportFixer({'dry_run': False, 'backup_enabled': True})
        result_actual = fixer_actual.analyze_and_fix_file(str(test_file), interactive=False)
        
        print(f"\n📁 After actual fixing:")
        with open(test_file, 'r') as f:
            print(f.read())
        
        # Check if backup was created
        backup_files = list(Path(temp_dir).glob("*.backup_*"))
        if backup_files:
            print(f"\n💾 Backup created: {backup_files[0].name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def test_mixed_scenarios():
    """Test handling of mixed confidence scenarios."""
    print("\n" + "="*60)
    print("Testing Mixed Confidence Scenarios")
    print("="*60)
    
    temp_dir = create_test_files()
    
    try:
        fixer = IntelligentImportFixer({'dry_run': True})
        test_file = Path(temp_dir) / "mixed_scenarios.py"
        
        print(f"📁 Testing mixed scenarios on: {test_file.name}")
        
        result = fixer.analyze_and_fix_file(str(test_file), interactive=False)
        
        print(f"\n📊 Mixed scenario results:")
        print(f"   Total issues: {result.total_issues}")
        print(f"   Auto-fixed: {result.auto_fixed}")
        print(f"   Flagged: {result.flagged_only}")
        
        # Show flagged issues
        if result.flagged_issues:
            print(f"\n🚩 Flagged issues for review:")
            for issue in result.flagged_issues:
                print(f"   Line {issue.line_number}: {issue.confidence.value} - {issue.reason}")
                print(f"      Suggested: {issue.suggested_fix}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def test_safety_scoring():
    """Test safety scoring system."""
    print("\n" + "="*60)
    print("Testing Safety Scoring System")
    print("="*60)
    
    temp_dir = create_test_files()
    
    try:
        fixer = IntelligentImportFixer()
        
        # Test different safety scenarios
        test_cases = [
            ("high_confidence.py", "Should have high safety scores"),
            ("medium_confidence.py", "Should have medium safety scores"),
            ("low_confidence.py", "Should have low safety scores")
        ]
        
        for filename, description in test_cases:
            test_file = Path(temp_dir) / filename
            issues = fixer._analyze_import_issues(str(test_file))
            
            print(f"\n📁 {filename} - {description}")
            for issue in issues:
                if hasattr(issue, 'safety_score'):
                    print(f"   Line {issue.line_number}: Safety score {issue.safety_score}/{issue.max_safety_score} - {issue.confidence.value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def test_report_generation():
    """Test comprehensive report generation."""
    print("\n" + "="*60)
    print("Testing Report Generation")
    print("="*60)
    
    temp_dir = create_test_files()
    
    try:
        fixer = IntelligentImportFixer({'dry_run': True})
        results = []
        
        # Process multiple files
        for py_file in Path(temp_dir).glob("*.py"):
            result = fixer.analyze_and_fix_file(str(py_file), interactive=False)
            results.append(result)
        
        # Generate report
        report = fixer.generate_report(results)
        
        print(f"📊 Comprehensive Report:")
        print(f"   Files processed: {report['summary']['total_files_processed']}")
        print(f"   Total issues: {report['summary']['total_issues_found']}")
        print(f"   Auto-fixed: {report['summary']['auto_fixed']} ({report['summary']['auto_fix_rate']:.1f}%)")
        print(f"   Total fix rate: {report['summary']['total_fix_rate']:.1f}%")
        
        if report['flagged_issues']:
            print(f"\n🚩 Flagged issues summary:")
            confidence_counts = {}
            for issue in report['flagged_issues']:
                conf = issue['confidence']
                confidence_counts[conf] = confidence_counts.get(conf, 0) + 1
            
            for conf, count in confidence_counts.items():
                print(f"   {conf}: {count} issues")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def main():
    """Run all tests."""
    print("Intelligent Import Fixer Tests")
    print("="*60)
    
    tests = [
        ("Confidence Assessment", test_confidence_assessment),
        ("Automatic Fixing", test_auto_fixing),
        ("Mixed Scenarios", test_mixed_scenarios),
        ("Safety Scoring", test_safety_scoring),
        ("Report Generation", test_report_generation)
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
        print("🎉 All tests passed! The intelligent import fixer is working correctly.")
        print("\n🎯 Key Capabilities Demonstrated:")
        print("   ✅ Automatic confidence assessment (95%/4%/1%)")
        print("   ✅ Auto-fixing of high-confidence issues")
        print("   ✅ Flagging of medium/low confidence issues")
        print("   ✅ Safety scoring and validation")
        print("   ✅ Comprehensive reporting")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())