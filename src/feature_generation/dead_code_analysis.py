"""
Dead Code Analysis

This module identifies dead/deprecated code that can be safely removed
after implementing the new unified feature generation system.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Set, Any
import logging

logger = logging.getLogger(__name__)

class DeadCodeAnalyzer:
    """
    Analyzes the codebase to identify dead/deprecated code that can be safely removed.
    """
    
    def __init__(self, workspace_root: str = "/workspace"):
        self.workspace_root = Path(workspace_root)
        self.src_root = self.workspace_root / "src"
        self.logger = logger.getChild('DeadCodeAnalyzer')
        
        # Patterns for identifying dead code
        self.dead_code_patterns = {
            'backup_directories': [
                r'.*backup.*',
                r'.*_backup.*',
                r'.*backup_.*'
            ],
            'old_files': [
                r'.*_old\.py$',
                r'.*old_.*\.py$',
                r'.*_deprecated\.py$',
                r'.*deprecated_.*\.py$',
                r'.*_legacy\.py$',
                r'.*legacy_.*\.py$',
                r'.*_temp\.py$',
                r'.*temp_.*\.py$'
            ],
            'timestamped_files': [
                r'.*_\d{8}_\d{4}\.py$',  # _20250913_1422.py
                r'.*_\d{8}_\d{4}\.md$',  # _20250913_1422.md
                r'.*_\d{8}_\d{4}\.txt$'  # _20250913_1422.txt
            ],
            'test_files': [
                r'test_.*\.py$',
                r'.*_test\.py$',
                r'.*_tests\.py$'
            ],
            'duplicate_files': [
                r'.*_copy\.py$',
                r'.*copy_.*\.py$',
                r'.*_duplicate\.py$',
                r'.*duplicate_.*\.py$'
            ]
        }
        
        # Files that should be kept (not dead code)
        self.keep_patterns = [
            r'.*__init__\.py$',
            r'.*conftest\.py$',
            r'.*pytest\.py$',
            r'.*setup\.py$',
            r'.*requirements\.txt$',
            r'.*README\.md$',
            r'.*LICENSE$',
            r'.*\.gitignore$'
        ]
        
        # Directories to exclude from analysis
        self.exclude_directories = {
            '__pycache__',
            '.git',
            '.pytest_cache',
            'node_modules',
            '.venv',
            'venv',
            'env',
            '.env'
        }
    
    def analyze_dead_code(self) -> Dict[str, Any]:
        """Analyze the codebase for dead code."""
        
        self.logger.info("🔍 Starting dead code analysis...")
        
        analysis_result = {
            'backup_directories': [],
            'old_files': [],
            'timestamped_files': [],
            'test_files': [],
            'duplicate_files': [],
            'total_files_to_remove': 0,
            'total_size_to_remove': 0,
            'recommendations': []
        }
        
        # Analyze each pattern type
        for pattern_type, patterns in self.dead_code_patterns.items():
            files = self._find_files_matching_patterns(patterns)
            analysis_result[pattern_type] = files
            
            # Calculate total size
            total_size = 0
            for file_path in files:
                if file_path.exists():
                    total_size += file_path.stat().st_size
            
            analysis_result['total_size_to_remove'] += total_size
            analysis_result['total_files_to_remove'] += len(files)
        
        # Generate recommendations
        analysis_result['recommendations'] = self._generate_recommendations(analysis_result)
        
        return analysis_result
    
    def _find_files_matching_patterns(self, patterns: List[str]) -> List[Path]:
        """Find files matching the given patterns."""
        
        matching_files = []
        
        for root, dirs, files in os.walk(self.src_root):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in self.exclude_directories]
            
            for file in files:
                file_path = Path(root) / file
                
                # Check if file matches any pattern
                for pattern in patterns:
                    if re.match(pattern, file, re.IGNORECASE):
                        # Check if file should be kept
                        if not self._should_keep_file(file_path):
                            matching_files.append(file_path)
                        break
        
        return matching_files
    
    def _should_keep_file(self, file_path: Path) -> bool:
        """Check if a file should be kept (not considered dead code)."""
        
        for keep_pattern in self.keep_patterns:
            if re.match(keep_pattern, file_path.name, re.IGNORECASE):
                return True
        
        return False
    
    def _generate_recommendations(self, analysis_result: Dict[str, Any]) -> List[str]:
        """Generate recommendations for dead code removal."""
        
        recommendations = []
        
        # Backup directories
        if analysis_result['backup_directories']:
            recommendations.append(
                f"🗂️ Remove {len(analysis_result['backup_directories'])} backup directories: "
                f"{', '.join([str(p.relative_to(self.src_root)) for p in analysis_result['backup_directories']])}"
            )
        
        # Old files
        if analysis_result['old_files']:
            recommendations.append(
                f"📄 Remove {len(analysis_result['old_files'])} old/deprecated files: "
                f"{', '.join([str(p.relative_to(self.src_root)) for p in analysis_result['old_files'][:5]])}"
                + ("..." if len(analysis_result['old_files']) > 5 else "")
            )
        
        # Timestamped files
        if analysis_result['timestamped_files']:
            recommendations.append(
                f"⏰ Remove {len(analysis_result['timestamped_files'])} timestamped files: "
                f"{', '.join([str(p.relative_to(self.src_root)) for p in analysis_result['timestamped_files'][:5]])}"
                + ("..." if len(analysis_result['timestamped_files']) > 5 else "")
            )
        
        # Test files (if not needed)
        if analysis_result['test_files']:
            recommendations.append(
                f"🧪 Review {len(analysis_result['test_files'])} test files for removal: "
                f"{', '.join([str(p.relative_to(self.src_root)) for p in analysis_result['test_files'][:5]])}"
                + ("..." if len(analysis_result['test_files']) > 5 else "")
            )
        
        # Duplicate files
        if analysis_result['duplicate_files']:
            recommendations.append(
                f"📋 Remove {len(analysis_result['duplicate_files'])} duplicate files: "
                f"{', '.join([str(p.relative_to(self.src_root)) for p in analysis_result['duplicate_files'][:5]])}"
                + ("..." if len(analysis_result['duplicate_files']) > 5 else "")
            )
        
        # Size recommendation
        total_size_mb = analysis_result['total_size_to_remove'] / (1024 * 1024)
        if total_size_mb > 1:
            recommendations.append(
                f"💾 Total space to be freed: {total_size_mb:.1f} MB"
            )
        
        return recommendations
    
    def generate_removal_script(self, analysis_result: Dict[str, Any]) -> str:
        """Generate a script to remove dead code."""
        
        script_lines = [
            "#!/bin/bash",
            "# Dead Code Removal Script",
            "# Generated automatically - Review before running!",
            "",
            "set -e  # Exit on any error",
            "",
            "echo '🗑️ Starting dead code removal...'",
            ""
        ]
        
        # Remove backup directories
        for backup_dir in analysis_result['backup_directories']:
            relative_path = backup_dir.relative_to(self.workspace_root)
            script_lines.append(f"echo 'Removing backup directory: {relative_path}'")
            script_lines.append(f"rm -rf '{relative_path}'")
            script_lines.append("")
        
        # Remove old files
        for old_file in analysis_result['old_files']:
            relative_path = old_file.relative_to(self.workspace_root)
            script_lines.append(f"echo 'Removing old file: {relative_path}'")
            script_lines.append(f"rm -f '{relative_path}'")
            script_lines.append("")
        
        # Remove timestamped files
        for timestamped_file in analysis_result['timestamped_files']:
            relative_path = timestamped_file.relative_to(self.workspace_root)
            script_lines.append(f"echo 'Removing timestamped file: {relative_path}'")
            script_lines.append(f"rm -f '{relative_path}'")
            script_lines.append("")
        
        # Remove duplicate files
        for duplicate_file in analysis_result['duplicate_files']:
            relative_path = duplicate_file.relative_to(self.workspace_root)
            script_lines.append(f"echo 'Removing duplicate file: {relative_path}'")
            script_lines.append(f"rm -f '{relative_path}'")
            script_lines.append("")
        
        script_lines.extend([
            "echo '✅ Dead code removal completed!'",
            "echo '📊 Summary:'",
            f"echo '  - Files removed: {analysis_result['total_files_to_remove']}'",
            f"echo '  - Space freed: {analysis_result['total_size_to_remove'] / (1024 * 1024):.1f} MB'"
        ])
        
        return "\n".join(script_lines)
    
    def generate_report(self, analysis_result: Dict[str, Any]) -> str:
        """Generate a comprehensive dead code analysis report."""
        
        report_lines = [
            "=" * 80,
            "DEAD CODE ANALYSIS REPORT",
            "=" * 80,
            "",
            f"Total files to remove: {analysis_result['total_files_to_remove']}",
            f"Total size to free: {analysis_result['total_size_to_remove'] / (1024 * 1024):.1f} MB",
            "",
            "CATEGORIES:",
            "-" * 40
        ]
        
        # Backup directories
        if analysis_result['backup_directories']:
            report_lines.extend([
                "",
                "🗂️ BACKUP DIRECTORIES:",
                f"Count: {len(analysis_result['backup_directories'])}"
            ])
            for backup_dir in analysis_result['backup_directories']:
                relative_path = backup_dir.relative_to(self.src_root)
                report_lines.append(f"  - {relative_path}")
        
        # Old files
        if analysis_result['old_files']:
            report_lines.extend([
                "",
                "📄 OLD/DEPRECATED FILES:",
                f"Count: {len(analysis_result['old_files'])}"
            ])
            for old_file in analysis_result['old_files'][:10]:  # Show first 10
                relative_path = old_file.relative_to(self.src_root)
                report_lines.append(f"  - {relative_path}")
            if len(analysis_result['old_files']) > 10:
                report_lines.append(f"  ... and {len(analysis_result['old_files']) - 10} more")
        
        # Timestamped files
        if analysis_result['timestamped_files']:
            report_lines.extend([
                "",
                "⏰ TIMESTAMPED FILES:",
                f"Count: {len(analysis_result['timestamped_files'])}"
            ])
            for timestamped_file in analysis_result['timestamped_files'][:10]:  # Show first 10
                relative_path = timestamped_file.relative_to(self.src_root)
                report_lines.append(f"  - {relative_path}")
            if len(analysis_result['timestamped_files']) > 10:
                report_lines.append(f"  ... and {len(analysis_result['timestamped_files']) - 10} more")
        
        # Test files
        if analysis_result['test_files']:
            report_lines.extend([
                "",
                "🧪 TEST FILES (Review for removal):",
                f"Count: {len(analysis_result['test_files'])}"
            ])
            for test_file in analysis_result['test_files'][:10]:  # Show first 10
                relative_path = test_file.relative_to(self.src_root)
                report_lines.append(f"  - {relative_path}")
            if len(analysis_result['test_files']) > 10:
                report_lines.append(f"  ... and {len(analysis_result['test_files']) - 10} more")
        
        # Duplicate files
        if analysis_result['duplicate_files']:
            report_lines.extend([
                "",
                "📋 DUPLICATE FILES:",
                f"Count: {len(analysis_result['duplicate_files'])}"
            ])
            for duplicate_file in analysis_result['duplicate_files'][:10]:  # Show first 10
                relative_path = duplicate_file.relative_to(self.src_root)
                report_lines.append(f"  - {relative_path}")
            if len(analysis_result['duplicate_files']) > 10:
                report_lines.append(f"  ... and {len(analysis_result['duplicate_files']) - 10} more")
        
        # Recommendations
        if analysis_result['recommendations']:
            report_lines.extend([
                "",
                "RECOMMENDATIONS:",
                "-" * 40
            ])
            for recommendation in analysis_result['recommendations']:
                report_lines.append(f"  {recommendation}")
        
        report_lines.extend([
            "",
            "=" * 80,
            "⚠️  WARNING: Review all files before removal!",
            "⚠️  Some files might be needed for specific functionality.",
            "⚠️  Test the system after removal to ensure nothing breaks.",
            "=" * 80
        ])
        
        return "\n".join(report_lines)

def run_dead_code_analysis():
    """Run the complete dead code analysis."""
    
    print("🔍 Running Dead Code Analysis...")
    print("=" * 60)
    
    analyzer = DeadCodeAnalyzer()
    analysis_result = analyzer.analyze_dead_code()
    
    # Generate and print report
    report = analyzer.generate_report(analysis_result)
    print(report)
    
    # Generate removal script
    removal_script = analyzer.generate_removal_script(analysis_result)
    
    # Save removal script
    script_path = "/workspace/remove_dead_code.sh"
    with open(script_path, 'w') as f:
        f.write(removal_script)
    
    print(f"\n📝 Removal script saved to: {script_path}")
    print("⚠️  Review the script before running it!")
    
    return analyzer, analysis_result

if __name__ == "__main__":
    analyzer, analysis_result = run_dead_code_analysis()