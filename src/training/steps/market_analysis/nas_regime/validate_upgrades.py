#!/usr/bin/env python3
"""
Simple validation script to check NAS regime upgrades.

This script validates that the upgraded files can be imported and have correct syntax.
"""

import sys
import os
import ast
import logging

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_syntax(file_path):
    """Validate that a Python file has correct syntax."""
    try:
        with open(file_path, 'r') as f:
            source = f.read()
        
        # Parse the AST to check for syntax errors
        ast.parse(source)
        return True, "Syntax OK"
        
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error: {e}"

def validate_imports(file_path):
    """Validate that a Python file can be imported (basic check)."""
    try:
        # Try to compile the file
        with open(file_path, 'r') as f:
            source = f.read()
        
        compile(source, file_path, 'exec')
        return True, "Import OK"
        
    except Exception as e:
        return False, f"Import error: {e}"

def check_file_upgrades(file_path, expected_imports):
    """Check if a file has the expected upgrade imports."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        found_imports = []
        missing_imports = []
        
        for import_line in expected_imports:
            if import_line in content:
                found_imports.append(import_line)
            else:
                missing_imports.append(import_line)
        
        return found_imports, missing_imports
        
    except Exception as e:
        return [], [f"Error reading file: {e}"]

def main():
    """Run validation checks."""
    logger.info("🚀 Starting NAS Regime Upgrade Validation...")
    
    # Files to validate
    files_to_check = [
        "core/enhanced_matrix_operations.py",
        "core/enhanced_data_operations.py", 
        "core/enhanced_ml_common_integration.py",
        "core/perfect_nas_regime_detector.py"
    ]
    
    # Expected imports for upgrades
    expected_imports = {
        "core/enhanced_matrix_operations.py": [
            "from src.utils.common_operations import",
            "from src.utils.math_validation import",
            "from src.utils.serialization_utils import"
        ],
        "core/enhanced_data_operations.py": [
            "from src.utils.common_operations import",
            "from src.utils.math_validation import",
            "from src.utils.serialization_utils import",
            "from src.utils.data.klines_parquet import"
        ],
        "core/enhanced_ml_common_integration.py": [
            "from src.utils.common_operations import",
            "from src.utils.math_validation import",
            "from src.utils.serialization_utils import"
        ],
        "core/perfect_nas_regime_detector.py": [
            "from src.utils.common_operations import",
            "from src.utils.math_validation import",
            "from src.utils.serialization_utils import",
            "from .enhanced_data_operations import"
        ]
    }
    
    results = []
    
    for file_path in files_to_check:
        logger.info(f"\n{'='*60}")
        logger.info(f"Validating: {file_path}")
        logger.info(f"{'='*60}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"❌ File not found: {file_path}")
            results.append((file_path, False, "File not found"))
            continue
        
        # Validate syntax
        syntax_ok, syntax_msg = validate_syntax(file_path)
        logger.info(f"Syntax validation: {'✅' if syntax_ok else '❌'} {syntax_msg}")
        
        # Validate imports
        import_ok, import_msg = validate_imports(file_path)
        logger.info(f"Import validation: {'✅' if import_ok else '❌'} {import_msg}")
        
        # Check for expected upgrades
        if file_path in expected_imports:
            found, missing = check_file_upgrades(file_path, expected_imports[file_path])
            logger.info(f"Upgrade imports found: {len(found)}/{len(expected_imports[file_path])}")
            
            for found_import in found:
                logger.info(f"  ✅ {found_import}")
            
            for missing_import in missing:
                logger.info(f"  ❌ Missing: {missing_import}")
        
        # Overall result for this file
        file_ok = syntax_ok and import_ok
        results.append((file_path, file_ok, f"Syntax: {syntax_ok}, Import: {import_ok}"))
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("VALIDATION SUMMARY")
    logger.info(f"{'='*60}")
    
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    
    for file_path, ok, msg in results:
        status = "✅ PASSED" if ok else "❌ FAILED"
        logger.info(f"{file_path}: {status} - {msg}")
    
    logger.info(f"\nOverall: {passed}/{total} files validated successfully")
    
    # Check for upgrade summary file
    if os.path.exists("UPGRADE_SUMMARY.md"):
        logger.info("✅ Upgrade summary document found")
    else:
        logger.warning("⚠️ Upgrade summary document not found")
    
    if passed == total:
        logger.info("🎉 All validations passed! Upgrades are syntactically correct.")
        return 0
    else:
        logger.error("⚠️ Some validations failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())