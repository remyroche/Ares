#!/usr/bin/env python3
"""
Intelligent Import Fixer Pipeline

This pipeline provides automatic fixing of import issues with different confidence levels:
- HIGH CONFIDENCE (95%): Auto-fix immediately
- MEDIUM CONFIDENCE (4%): Auto-fix with user confirmation
- LOW CONFIDENCE (1%): Flag for manual review only

This approach maximizes automation while maintaining safety.
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from analyzers.intelligent_import_fixer import IntelligentImportFixer, ConfidenceLevel, FixAction
from core.config import CodeQualityConfig
from plugins.base_plugin import FileProcessorPlugin, PluginMetadata, PluginCategory, PluginPriority


class IntelligentImportFixerPlugin(FileProcessorPlugin):
    """
    Intelligent import fixer plugin for automatic import issue resolution.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.metadata = PluginMetadata(
            name="IntelligentImportFixer",
            version="1.0.0",
            description="Automatic import issue fixing with confidence-based actions",
            author="Code Quality Team",
            category=PluginCategory.FIXER,
            priority=PluginPriority.HIGH
        )
        
        # Initialize the core intelligent import fixer
        self.fixer = IntelligentImportFixer()
        
        # Configuration
        self.auto_fix_high_confidence = config.get('auto_fix_high_confidence', True)
        self.auto_fix_medium_confidence = config.get('auto_fix_medium_confidence', False)
        self.dry_run = config.get('dry_run', False)
        
    def process_file(self, file_path: Path) -> Dict[str, Any]:
        """Process a single file for import issues."""
        try:
            # Analyze the file for import issues
            issues = self.fixer.analyze_file(file_path)
            
            # Apply fixes based on confidence levels
            fix_results = self.fixer.apply_fixes(
                file_path, 
                issues,
                auto_fix_high_confidence=self.auto_fix_high_confidence,
                auto_fix_medium_confidence=self.auto_fix_medium_confidence,
                dry_run=self.dry_run
            )
            
            return {
                "file_path": str(file_path),
                "issues_found": len(issues),
                "issues_fixed": fix_results.get('fixed_count', 0),
                "issues_flagged": fix_results.get('flagged_count', 0),
                "fix_rate": fix_results.get('fix_rate', 0.0),
                "details": fix_results
            }
            
        except Exception as e:
            return {
                "file_path": str(file_path),
                "error": str(e),
                "issues_found": 0,
                "issues_fixed": 0,
                "issues_flagged": 0,
                "fix_rate": 0.0
            }
    
    def process_directory(self, directory_path: Path) -> Dict[str, Any]:
        """Process all Python files in a directory."""
        results = {
            "directory": str(directory_path),
            "files_processed": 0,
            "total_issues_found": 0,
            "total_issues_fixed": 0,
            "total_issues_flagged": 0,
            "average_fix_rate": 0.0,
            "file_results": []
        }
        
        # Find all Python files
        python_files = list(directory_path.rglob("*.py"))
        
        for file_path in python_files:
            # Skip test files and __pycache__ directories
            if any(part in str(file_path) for part in ['__pycache__', 'test_', '_test.py']):
                continue
                
            file_result = self.process_file(file_path)
            results["file_results"].append(file_result)
            results["files_processed"] += 1
            results["total_issues_found"] += file_result.get("issues_found", 0)
            results["total_issues_fixed"] += file_result.get("issues_fixed", 0)
            results["total_issues_flagged"] += file_result.get("issues_flagged", 0)
        
        # Calculate average fix rate
        if results["files_processed"] > 0:
            total_fix_rate = sum(r.get("fix_rate", 0.0) for r in results["file_results"])
            results["average_fix_rate"] = total_fix_rate / results["files_processed"]
        
        return results


def main():
    """Main function for intelligent import fixer pipeline."""
    parser = argparse.ArgumentParser(
        description="Intelligent Import Fixer Pipeline"
    )
    parser.add_argument("--target", "-t", 
                       help="Path to Python file or directory to fix (default: current directory)")
    parser.add_argument("--output", "-o", 
                       help="Output file for JSON report")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be fixed without making changes")
    parser.add_argument("--auto-fix-high", action="store_true", default=True,
                       help="Auto-fix high confidence issues (default: True)")
    parser.add_argument("--auto-fix-medium", action="store_true",
                       help="Auto-fix medium confidence issues (default: False)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Determine target path
    target_path = Path(args.target) if args.target else Path.cwd()
    
    if not target_path.exists():
        print(f"Error: Target path {target_path} does not exist")
        return 1
    
    # Create plugin configuration
    config = {
        'auto_fix_high_confidence': args.auto_fix_high,
        'auto_fix_medium_confidence': args.auto_fix_medium,
        'dry_run': args.dry_run
    }
    
    # Initialize the plugin
    plugin = IntelligentImportFixerPlugin(config)
    
    print(f"🔧 Intelligent Import Fixer Pipeline")
    print(f"Target: {target_path}")
    print(f"Dry run: {args.dry_run}")
    print(f"Auto-fix high confidence: {args.auto_fix_high}")
    print(f"Auto-fix medium confidence: {args.auto_fix_medium}")
    print("-" * 50)
    
    # Process the target
    if target_path.is_file():
        result = plugin.process_file(target_path)
        print(f"📁 File: {result['file_path']}")
        print(f"   Issues found: {result['issues_found']}")
        print(f"   Issues fixed: {result['issues_fixed']}")
        print(f"   Issues flagged: {result['issues_flagged']}")
        print(f"   Fix rate: {result['fix_rate']:.1f}%")
        
        if 'error' in result:
            print(f"   Error: {result['error']}")
    else:
        result = plugin.process_directory(target_path)
        print(f"📁 Directory: {result['directory']}")
        print(f"   Files processed: {result['files_processed']}")
        print(f"   Total issues found: {result['total_issues_found']}")
        print(f"   Total issues fixed: {result['total_issues_fixed']}")
        print(f"   Total issues flagged: {result['total_issues_flagged']}")
        print(f"   Average fix rate: {result['average_fix_rate']:.1f}%")
    
    # Save report if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n💾 Report saved to: {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())