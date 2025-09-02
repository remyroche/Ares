#!/usr/bin/env python3
"""
Systematic code quality improvement script for src/training/steps
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CodeQualityImprover:
    """Systematically improve code quality in src/training/steps"""
    
    def __init__(self, target_dir: str = "src/training/steps"):
        self.target_dir = Path(target_dir)
        self.python_files = []
        self.stats = {
            'files_processed': 0,
            'style_fixes': 0,
            'import_fixes': 0,
            'complexity_warnings': 0
        }
    
    def find_python_files(self) -> List[Path]:
        """Find all Python files in the target directory"""
        python_files = []
        for file_path in self.target_dir.rglob("*.py"):
            if not any(part.startswith('.') for part in file_path.parts):
                python_files.append(file_path)
        self.python_files = sorted(python_files)
        logger.info(f"Found {len(self.python_files)} Python files")
        return self.python_files
    
    def fix_style_issues(self, file_path: Path) -> bool:
        """Fix style issues using autopep8 and black"""
        try:
            # First, run autopep8 to fix basic PEP8 issues
            logger.info(f"Running autopep8 on {file_path}")
            subprocess.run([
                sys.executable, "-m", "autopep8",
                "--in-place",
                "--aggressive",
                "--aggressive",
                "--max-line-length", "120",
                str(file_path)
            ], check=True, capture_output=True, text=True)
            
            # Then run black for consistent formatting
            logger.info(f"Running black on {file_path}")
            subprocess.run([
                sys.executable, "-m", "black",
                "--line-length", "120",
                str(file_path)
            ], check=True, capture_output=True, text=True)
            
            self.stats['style_fixes'] += 1
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Error fixing style in {file_path}: {e}")
            return False
    
    def fix_imports(self, file_path: Path) -> bool:
        """Fix import issues using isort and autoflake"""
        try:
            # Remove unused imports
            logger.info(f"Running autoflake on {file_path}")
            subprocess.run([
                sys.executable, "-m", "autoflake",
                "--in-place",
                "--remove-all-unused-imports",
                "--remove-unused-variables",
                str(file_path)
            ], check=True, capture_output=True, text=True)
            
            # Sort imports
            logger.info(f"Running isort on {file_path}")
            subprocess.run([
                sys.executable, "-m", "isort",
                "--line-length", "120",
                "--profile", "black",
                str(file_path)
            ], check=True, capture_output=True, text=True)
            
            self.stats['import_fixes'] += 1
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Error fixing imports in {file_path}: {e}")
            return False
    
    def analyze_complexity(self, file_path: Path) -> List[Tuple[str, int]]:
        """Analyze code complexity and return high-complexity functions"""
        try:
            result = subprocess.run([
                sys.executable, "-m", "radon", "cc",
                str(file_path),
                "-s",  # Show complexity score
                "-n", "C"  # Show only C and above
            ], capture_output=True, text=True)
            
            if result.stdout:
                logger.warning(f"High complexity found in {file_path}:\n{result.stdout}")
                self.stats['complexity_warnings'] += 1
                return self._parse_complexity_output(result.stdout)
            return []
        except subprocess.CalledProcessError as e:
            logger.error(f"Error analyzing complexity in {file_path}: {e}")
            return []
    
    def _parse_complexity_output(self, output: str) -> List[Tuple[str, int]]:
        """Parse radon output to extract function names and complexity scores"""
        results = []
        for line in output.strip().split('\n'):
            if ' - ' in line and '(' in line:
                parts = line.split(' - ')
                if len(parts) >= 2:
                    func_info = parts[0].strip()
                    complexity_info = parts[1].strip()
                    if '(' in complexity_info:
                        score = complexity_info.split('(')[1].rstrip(')')
                        try:
                            results.append((func_info, int(score)))
                        except ValueError:
                            pass
        return results
    
    def process_file(self, file_path: Path) -> dict:
        """Process a single file with all improvements"""
        logger.info(f"\nProcessing: {file_path}")
        results = {
            'file': str(file_path),
            'style_fixed': False,
            'imports_fixed': False,
            'complexity_issues': []
        }
        
        # Fix imports first (removes unused imports)
        if self.fix_imports(file_path):
            results['imports_fixed'] = True
        
        # Fix style issues
        if self.fix_style_issues(file_path):
            results['style_fixed'] = True
        
        # Analyze complexity
        complexity_issues = self.analyze_complexity(file_path)
        if complexity_issues:
            results['complexity_issues'] = complexity_issues
        
        self.stats['files_processed'] += 1
        return results
    
    def generate_complexity_report(self, all_results: List[dict]) -> str:
        """Generate a report of high-complexity functions that need manual refactoring"""
        report_lines = ["# Code Complexity Report\n"]
        report_lines.append("## High Complexity Functions Requiring Manual Refactoring\n")
        
        for result in all_results:
            if result['complexity_issues']:
                report_lines.append(f"\n### {result['file']}\n")
                for func_name, score in result['complexity_issues']:
                    report_lines.append(f"- {func_name}: Complexity = {score}")
                    if score > 30:
                        report_lines.append("  **CRITICAL: Very high complexity, consider breaking into smaller functions**")
                    elif score > 20:
                        report_lines.append("  **HIGH: Consider refactoring for better maintainability**")
                    else:
                        report_lines.append("  **MODERATE: Could benefit from simplification**")
        
        return '\n'.join(report_lines)
    
    def run(self):
        """Run the complete code quality improvement process"""
        logger.info("Starting code quality improvement process...")
        
        # Find all Python files
        self.find_python_files()
        
        # Process each file
        all_results = []
        for file_path in self.python_files:
            result = self.process_file(file_path)
            all_results.append(result)
        
        # Generate complexity report
        complexity_report = self.generate_complexity_report(all_results)
        report_path = Path("code_quality_complexity_report.md")
        report_path.write_text(complexity_report)
        logger.info(f"Complexity report saved to: {report_path}")
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("CODE QUALITY IMPROVEMENT SUMMARY")
        logger.info("="*60)
        logger.info(f"Files processed: {self.stats['files_processed']}")
        logger.info(f"Style fixes applied: {self.stats['style_fixes']}")
        logger.info(f"Import fixes applied: {self.stats['import_fixes']}")
        logger.info(f"High complexity warnings: {self.stats['complexity_warnings']}")
        logger.info("="*60)
        
        return all_results


def main():
    """Main entry point"""
    improver = CodeQualityImprover()
    improver.run()


if __name__ == "__main__":
    main()