#!/usr/bin/env python3
"""
Simple Enhanced Dead Code Analysis Pipeline

This pipeline runs a simplified enhanced dead code analysis that shows all detected
issues with confidence levels and filtering reasons, without complex multi-tool
consensus that might cause errors.
"""

import sys
import time
import json
import ast
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set
from dataclasses import dataclass, field

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.config import AnalysisConfig
from utils.file_utils import find_python_files


@dataclass
class SimpleDeadCodeIssue:
    """Simple container for dead code analysis results."""
    file_path: str
    line_number: int
    issue_type: str
    description: str
    confidence: float
    severity: str
    code_snippet: str
    function_name: str = ""
    class_name: str = ""
    has_docstring: bool = False
    is_special_function: bool = False
    is_test_function: bool = False
    is_public_api: bool = False
    filtering_reasons: List[str] = field(default_factory=list)


class SimpleEnhancedDeadCodeAnalyzer:
    """Simple enhanced dead code analyzer with confidence scoring."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.issues = []
        
    def analyze_directory(self, directory: str | Path) -> List[SimpleDeadCodeIssue]:
        """Analyze directory for dead code."""
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        print(f"🔍 Analyzing {directory} for dead code...")
        
        # Find all Python files
        python_files = find_python_files(directory)
        print(f"📁 Found {len(python_files)} Python files to analyze")
        
        all_issues = []
        
        for file_path in python_files:
            try:
                issues = self._analyze_file(file_path)
                all_issues.extend(issues)
            except Exception as e:
                print(f"⚠️  Failed to analyze {file_path}: {e}")
        
        # Apply confidence scoring
        scored_issues = self._apply_confidence_scoring(all_issues)
        
        # Sort by confidence
        scored_issues.sort(key=lambda x: x.confidence, reverse=True)
        
        print(f"✅ Analysis complete. Found {len(scored_issues)} potential dead code issues")
        return scored_issues
    
    def _analyze_file(self, file_path: Path) -> List[SimpleDeadCodeIssue]:
        """Analyze a single file for dead code."""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content, filename=str(file_path))
            
            # Find all function definitions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if self._is_unused_function(node, tree, content):
                        issue = SimpleDeadCodeIssue(
                            file_path=str(file_path),
                            line_number=node.lineno,
                            issue_type="dead_code",
                            description=f"Function '{node.name}' is defined but never called",
                            confidence=70.0,  # Base confidence
                            severity="medium",
                            code_snippet=f"def {node.name}(...):",
                            function_name=node.name,
                            has_docstring=ast.get_docstring(node) is not None,
                            is_special_function=self._is_special_function(node.name),
                            is_test_function=node.name.startswith('test_'),
                            is_public_api=node.name[0].isupper() if node.name else False
                        )
                        issues.append(issue)
                
                elif isinstance(node, ast.ClassDef):
                    if self._is_unused_class(node, tree, content):
                        issue = SimpleDeadCodeIssue(
                            file_path=str(file_path),
                            line_number=node.lineno,
                            issue_type="dead_code",
                            description=f"Class '{node.name}' is defined but never used",
                            confidence=70.0,  # Base confidence
                            severity="medium",
                            code_snippet=f"class {node.name}:",
                            class_name=node.name,
                            has_docstring=ast.get_docstring(node) is not None,
                            is_public_api=node.name[0].isupper() if node.name else False
                        )
                        issues.append(issue)
        
        except Exception as e:
            # Skip files with syntax errors
            pass
        
        return issues
    
    def _is_unused_function(self, func: ast.FunctionDef, tree: ast.AST, content: str) -> bool:
        """Check if a function is unused."""
        func_name = func.name
        
        # Skip special functions
        if self._is_special_function(func_name):
            return False
        
        # Check for direct calls
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == func_name:
                    return False
                elif isinstance(node.func, ast.Attribute) and node.func.attr == func_name:
                    return False
        
        # Check for string references (simple check)
        if f'"{func_name}"' in content or f"'{func_name}'" in content:
            return False
        
        return True
    
    def _is_unused_class(self, cls: ast.ClassDef, tree: ast.AST, content: str) -> bool:
        """Check if a class is unused."""
        cls_name = cls.name
        
        # Skip special classes
        if cls_name.startswith('_'):
            return False
        
        # Check for direct usage
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == cls_name:
                return False
        
        return True
    
    def _is_special_function(self, func_name: str) -> bool:
        """Check if function is special (should not be considered dead code)."""
        special_patterns = [
            r'^__\w+__$',  # Special methods
            r'^test_',     # Test functions
            r'^setup_',    # Setup functions
            r'^teardown_', # Teardown functions
            r'^main$',     # Main function
        ]
        
        import re
        for pattern in special_patterns:
            if re.match(pattern, func_name):
                return True
        
        return False
    
    def _apply_confidence_scoring(self, issues: List[SimpleDeadCodeIssue]) -> List[SimpleDeadCodeIssue]:
        """Apply confidence scoring based on various factors."""
        for issue in issues:
            original_confidence = issue.confidence
            
            # Adjust confidence based on function characteristics
            if issue.is_special_function:
                issue.confidence *= 0.3  # Very low confidence for special functions
                issue.filtering_reasons.append("Special function (likely not dead code)")
            
            # REMOVED: Test function confidence adjustment
            # if issue.is_test_function:
            #     issue.confidence *= 0.4  # Low confidence for test functions
            #     issue.filtering_reasons.append("Test function (may be unused but not dead)")
            
            # REMOVED: Docstring confidence adjustment
            # if issue.has_docstring:
            #     issue.confidence *= 0.7  # Lower confidence for documented functions
            #     issue.filtering_reasons.append("Has docstring (likely important)")
            
            if issue.is_public_api:
                issue.confidence *= 0.6  # Lower confidence for public APIs
                issue.filtering_reasons.append("Public API (likely used externally)")
            
            # Check for dynamic usage patterns
            if self._has_dynamic_usage(issue):
                issue.confidence *= 0.5  # Very low confidence for dynamic usage
                issue.filtering_reasons.append("Dynamic usage detected")
            
            # Ensure confidence is between 0 and 100
            issue.confidence = max(0.0, min(100.0, issue.confidence))
        
        return issues
    
    def _has_dynamic_usage(self, issue: SimpleDeadCodeIssue) -> bool:
        """Check if a function has dynamic usage patterns."""
        if not issue.function_name:
            return False
        
        file_path = Path(issue.file_path)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Simple dynamic usage patterns
            patterns = [
                f'"{issue.function_name}"',
                f"'{issue.function_name}'",
                f'getattr(.*{issue.function_name}',
                f'setattr(.*{issue.function_name}',
                f'@.*{issue.function_name}',
            ]
            
            for pattern in patterns:
                if pattern in content:
                    return True
            
            return False
        except Exception:
            return False


class SimpleEnhancedDeadCodePipeline:
    """Pipeline for simple enhanced dead code analysis."""
    
    def __init__(self, project_root: str = "/workspace"):
        self.project_root = Path(project_root)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create reports directory
        self.reports_dir = Path("code_quality/reports/dead_code")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize the analyzer
        self.config = AnalysisConfig()
        self.analyzer = SimpleEnhancedDeadCodeAnalyzer(self.config)
        
        print(f"✅ Initialized Simple Enhanced Dead Code Analyzer")
        print(f"📊 Confidence scoring enabled")
        print(f"🎯 Filtering reasons will be provided")
    
    def run_analysis(self) -> Dict[str, Any]:
        """Run the simple enhanced dead code analysis."""
        print("\n" + "="*80)
        print("SIMPLE ENHANCED DEAD CODE ANALYSIS PIPELINE")
        print("="*80)
        print(f"Project root: {self.project_root}")
        print(f"Timestamp: {self.timestamp}")
        print()
        
        start_time = time.time()
        
        try:
            # Run the analysis
            print("🔍 Running Simple Enhanced Dead Code Analysis...")
            print("   - AST-based function and class detection")
            print("   - Dynamic usage pattern detection")
            print("   - Confidence scoring with filtering reasons")
            print()
            
            issues = self.analyzer.analyze_directory(str(self.project_root))
            
            # Save the report
            report_path = self.reports_dir / f"simple_enhanced_dead_code_{self.timestamp}.json"
            self._save_report(issues, report_path)
            
            # Generate summary
            self._print_analysis_summary(issues)
            
            execution_time = time.time() - start_time
            
            return {
                "status": "completed",
                "execution_time": execution_time,
                "report_path": str(report_path),
                "total_issues": len(issues),
                "high_confidence_issues": len([i for i in issues if i.confidence > 80]),
                "medium_confidence_issues": len([i for i in issues if 60 <= i.confidence <= 80]),
                "low_confidence_issues": len([i for i in issues if i.confidence < 60]),
                "issues": issues
            }
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "execution_time": time.time() - start_time
            }
    
    def _save_report(self, issues: List[SimpleDeadCodeIssue], report_path: Path) -> None:
        """Save the report to JSON."""
        report_dict = {
            "timestamp": self.timestamp,
            "analysis_type": "simple_enhanced_dead_code",
            "project_root": str(self.project_root),
            "total_issues": len(issues),
            "results": {
                "issues": [
                    {
                        "file": issue.file_path,
                        "line": issue.line_number,
                        "type": issue.issue_type,
                        "description": issue.description,
                        "confidence": issue.confidence,
                        "severity": issue.severity,
                        "code_snippet": issue.code_snippet,
                        "function_name": issue.function_name,
                        "class_name": issue.class_name,
                        "has_docstring": issue.has_docstring,
                        "is_special_function": issue.is_special_function,
                        "is_test_function": issue.is_test_function,
                        "is_public_api": issue.is_public_api,
                        "filtering_reasons": issue.filtering_reasons
                    }
                    for issue in issues
                ]
            }
        }
        
        with open(report_path, "w") as f:
            json.dump(report_dict, f, indent=2)
        
        print(f"📄 Report saved to: {report_path}")
    
    def _print_analysis_summary(self, issues: List[SimpleDeadCodeIssue]) -> None:
        """Print a comprehensive analysis summary."""
        print("\n" + "="*80)
        print("SIMPLE ENHANCED DEAD CODE ANALYSIS RESULTS")
        print("="*80)
        
        # Overall statistics
        total_issues = len(issues)
        high_confidence = len([i for i in issues if i.confidence > 80])
        medium_confidence = len([i for i in issues if 60 <= i.confidence <= 80])
        low_confidence = len([i for i in issues if i.confidence < 60])
        
        print(f"📊 Total Issues Found: {total_issues}")
        print(f"🎯 High Confidence Issues (>80%): {high_confidence}")
        print(f"⚖️  Medium Confidence Issues (60-80%): {medium_confidence}")
        print(f"⚠️  Low Confidence Issues (<60%): {low_confidence}")
        print()
        
        # Show confidence distribution
        if total_issues > 0:
            print("📈 CONFIDENCE DISTRIBUTION:")
            high_pct = (high_confidence / total_issues) * 100
            medium_pct = (medium_confidence / total_issues) * 100
            low_pct = (low_confidence / total_issues) * 100
            print(f"   High Confidence:   {high_confidence:3d} ({high_pct:5.1f}%)")
            print(f"   Medium Confidence: {medium_confidence:3d} ({medium_pct:5.1f}%)")
            print(f"   Low Confidence:    {low_confidence:3d} ({low_pct:5.1f}%)")
            print()
        
        # Show some example issues with confidence levels
        if issues:
            print("🔍 EXAMPLE ISSUES BY CONFIDENCE LEVEL:")
            
            # Show top 10 high confidence issues
            high_conf_issues = [i for i in issues if i.confidence > 80][:10]
            if high_conf_issues:
                print("   🎯 HIGH CONFIDENCE ISSUES (>80%):")
                for i, issue in enumerate(high_conf_issues, 1):
                    print(f"      {i:2d}. {issue.confidence:5.1f}% - {issue.function_name or issue.class_name or 'Unknown'} in {Path(issue.file_path).name}:{issue.line_number}")
                    if issue.filtering_reasons:
                        print(f"         Reasons: {', '.join(issue.filtering_reasons[:2])}")
                print()
            
            # Show top 10 medium confidence issues
            medium_conf_issues = [i for i in issues if 60 <= i.confidence <= 80][:10]
            if medium_conf_issues:
                print("   ⚖️  MEDIUM CONFIDENCE ISSUES (60-80%):")
                for i, issue in enumerate(medium_conf_issues, 1):
                    print(f"      {i:2d}. {issue.confidence:5.1f}% - {issue.function_name or issue.class_name or 'Unknown'} in {Path(issue.file_path).name}:{issue.line_number}")
                    if issue.filtering_reasons:
                        print(f"         Reasons: {', '.join(issue.filtering_reasons[:2])}")
                print()
            
            # Show top 10 low confidence issues
            low_conf_issues = [i for i in issues if i.confidence < 60][:10]
            if low_conf_issues:
                print("   ⚠️  LOW CONFIDENCE ISSUES (<60%):")
                for i, issue in enumerate(low_conf_issues, 1):
                    print(f"      {i:2d}. {issue.confidence:5.1f}% - {issue.function_name or issue.class_name or 'Unknown'} in {Path(issue.file_path).name}:{issue.line_number}")
                    if issue.filtering_reasons:
                        print(f"         Reasons: {', '.join(issue.filtering_reasons[:2])}")
                print()
        
        # Top files with issues
        if issues:
            print("📁 TOP 10 FILES WITH ISSUES:")
            file_issue_counts = {}
            for issue in issues:
                file_path = issue.file_path
                file_issue_counts[file_path] = file_issue_counts.get(file_path, 0) + 1
            
            sorted_files = sorted(file_issue_counts.items(), key=lambda x: x[1], reverse=True)
            for i, (file_path, count) in enumerate(sorted_files[:10], 1):
                print(f"   {i:2d}. {count:3d} issues: {Path(file_path).name}")
            print()
        
        print("="*80)


def main():
    """Main entry point for the pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple Enhanced Dead Code Analysis Pipeline")
    parser.add_argument("--project-root", default="/workspace", 
                       help="Root directory of the project to analyze")
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = SimpleEnhancedDeadCodePipeline(project_root=args.project_root)
    
    results = pipeline.run_analysis()
    
    if results["status"] == "completed":
        print(f"\n✅ Simple Enhanced Dead Code Analysis completed successfully!")
        print(f"⏱️  Execution time: {results['execution_time']:.2f} seconds")
        print(f"📊 Found {results['total_issues']} potential dead code issues")
        print(f"🎯 High confidence issues: {results['high_confidence_issues']}")
        print(f"⚖️  Medium confidence issues: {results['medium_confidence_issues']}")
        print(f"⚠️  Low confidence issues: {results['low_confidence_issues']}")
    else:
        print(f"\n❌ Analysis failed: {results.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()