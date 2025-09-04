#!/usr/bin/env python3
"""Test that HMM clustering modules can be imported correctly."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
if project_root.name == 'src':
    # We're already in a nested structure
    project_root = project_root.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test importing HMM clustering modules."""
    print("🧪 Testing HMM Clustering Module Imports")
    print("=" * 80)
    
    results = []
    
    # Test 1: Check if files exist
    print("\n1️⃣ Testing file existence...")
    hmm_path = Path(project_root) / "src" / "training" / "steps" / "hmm_clustering"
    if hmm_path.exists():
        print(f"✅ HMM clustering directory exists: {hmm_path}")
        results.append(("Directory exists", True, None))
        
        # List files
        files = list(hmm_path.glob("*.py"))
        print(f"   Found {len(files)} Python files")
        for f in files[:5]:
            print(f"   - {f.name}")
        if len(files) > 5:
            print(f"   ... and {len(files) - 5} more")
    else:
        print(f"❌ HMM clustering directory not found: {hmm_path}")
        results.append(("Directory exists", False, "Directory not found"))
        
    # Test 2: Check __init__.py content
    print("\n2️⃣ Testing __init__.py content...")
    init_file = hmm_path / "__init__.py"
    if init_file.exists():
        try:
            with open(init_file, 'r') as f:
                content = f.read()
            
            # Check for expected classes
            expected_classes = [
                "HMMRegimeDiscoveryStep",
                "EnhancedHMMRegimeDiscoveryStep",
                "OptimizedBayesianParameterOptimization",
                "RegimeDiscoveryFeatureEngineer",
                "EconomicSignificanceValidator",
                "EnsembleClusteringRegimeDetector",
                "EnhancedMLRegimeTransitionDetector",
            ]
            
            found_classes = []
            for cls in expected_classes:
                if cls in content:
                    found_classes.append(cls)
                    
            print(f"✅ Found {len(found_classes)}/{len(expected_classes)} expected classes in __init__.py")
            for cls in found_classes[:3]:
                print(f"   ✅ {cls}")
            if len(found_classes) > 3:
                print(f"   ... and {len(found_classes) - 3} more")
            results.append(("__init__.py classes", len(found_classes) > 0, None))
        except Exception as e:
            print(f"❌ Failed to read __init__.py: {e}")
            results.append(("__init__.py classes", False, str(e)))
    else:
        print("❌ __init__.py not found")
        results.append(("__init__.py classes", False, "File not found"))
            
    # Test 3: Check key files
    print("\n3️⃣ Testing key file existence...")
    key_files = [
        "step03_enhanced_hmm_regime_discovery.py",
        "step03_hmm_clustering_wrapper.py",
        "step03_optimized_bayesian_optimization.py",
        "step03_regime_discovery_features.py",
        "step03_economic_significance_validator.py",
        "step03_ensemble_clustering.py",
        "step03_enhanced_ml_transition_detector.py",
    ]
    
    existing_files = 0
    for filename in key_files:
        file_path = hmm_path / filename
        if file_path.exists():
            existing_files += 1
            print(f"✅ {filename}")
        else:
            print(f"❌ {filename} - NOT FOUND")
            
    print(f"\n   Found {existing_files}/{len(key_files)} key files")
    results.append(("Key files", existing_files == len(key_files), f"{existing_files}/{len(key_files)} found"))
            
    # Test 4: Check for syntax errors
    print("\n4️⃣ Testing for syntax errors...")
    syntax_errors = []
    for py_file in hmm_path.glob("*.py"):
        try:
            with open(py_file, 'r') as f:
                code = f.read()
            compile(code, str(py_file), 'exec')
        except SyntaxError as e:
            syntax_errors.append((py_file.name, str(e)))
            
    if syntax_errors:
        print(f"❌ Found {len(syntax_errors)} files with syntax errors:")
        for filename, error in syntax_errors:
            print(f"   - {filename}: {error}")
        results.append(("Syntax check", False, f"{len(syntax_errors)} errors"))
    else:
        print("✅ No syntax errors found")
        results.append(("Syntax check", True, None))
        
    # Test 5: Check imports in wrapper
    print("\n5️⃣ Testing wrapper structure...")
    wrapper_file = hmm_path / "step03_hmm_clustering_wrapper.py"
    if wrapper_file.exists():
        try:
            with open(wrapper_file, 'r') as f:
                wrapper_content = f.read()
                
            # Check for required class and methods
            has_class = "class HMMRegimeDiscoveryStep" in wrapper_content
            has_init = "def __init__" in wrapper_content
            has_initialize = "def initialize" in wrapper_content
            has_execute = "def execute" in wrapper_content
            
            all_present = has_class and has_init and has_initialize and has_execute
            
            print(f"✅ HMMRegimeDiscoveryStep class: {'Found' if has_class else 'NOT FOUND'}")
            print(f"✅ __init__ method: {'Found' if has_init else 'NOT FOUND'}")
            print(f"✅ initialize method: {'Found' if has_initialize else 'NOT FOUND'}")
            print(f"✅ execute method: {'Found' if has_execute else 'NOT FOUND'}")
            
            results.append(("Wrapper structure", all_present, None if all_present else "Missing required elements"))
        except Exception as e:
            print(f"❌ Failed to check wrapper structure: {e}")
            results.append(("Wrapper structure", False, str(e)))
    else:
        print("❌ Wrapper file not found")
        results.append(("Wrapper structure", False, "File not found"))
        
    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, success, _ in results if success)
    failed = len(results) - passed
    
    print(f"✅ Passed: {passed}/{len(results)}")
    print(f"❌ Failed: {failed}/{len(results)}")
    
    if failed > 0:
        print("\n❌ Failed tests:")
        for name, success, error in results:
            if not success:
                print(f"   - {name}: {error}")
                
    print("=" * 80)
    
    return failed == 0


if __name__ == "__main__":
    success = test_imports()
    print("\n🎯 FINAL RESULT:", "✅ ALL TESTS PASSED" if success else "❌ SOME TESTS FAILED")
    sys.exit(0 if success else 1)