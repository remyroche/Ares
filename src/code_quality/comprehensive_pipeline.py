#!/usr/bin/env python3
"""
Comprehensive Code Quality Pipeline

This module orchestrates all code quality checks including the new attribute access scanner.
It provides a unified interface for running comprehensive code analysis.
"""

import os
import sys
import subprocess
import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.warning_symbols import error, warning, failed, critical
from src.utils.logger import system_logger

logger = system_logger.getChild('ComprehensivePipeline')


class ComprehensivePipeline:
    """Comprehensive code quality pipeline orchestrating all analysis tools."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the comprehensive pipeline."""
        self.config = config or {}
        self.logger = logger
        self.results = {}
        self.report_dir = Path("src/code_quality/reports/comprehensive")
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
        # Pipeline configuration
        self.enabled_tools = self.config.get('enabled_tools', [
            'formatting', 'linting', 'type_checking', 'complexity', 
            'maintainability', 'dead_code', 'circular_imports', 
            'attribute_access'
        ])
        self.scan_path = self.config.get('scan_path', 'src')
        self.verbose = self.config.get('verbose', False)
        self.parallel_execution = self.config.get('parallel_execution', False)
        
    async def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Run comprehensive code quality analysis."""
        self.logger.info("🚀 Starting comprehensive code quality analysis...")
        
        analysis_results = {
            'timestamp': datetime.now().isoformat(),
            'scan_path': self.scan_path,
            'enabled_tools': self.enabled_tools,
            'tool_results': {},
            'overall_summary': {}
        }
        
        # Define tool configurations
        tools = self._get_tool_configurations()
        
        if self.parallel_execution:
            # Run tools in parallel
            tasks = []
            for tool_name in self.enabled_tools:
                if tool_name in tools:
                    task = self._run_tool_async(tool_name, tools[tool_name])
                    tasks.append(task)
            
            tool_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, tool_name in enumerate(self.enabled_tools):
                if tool_name in tools:
                    result = tool_results[i]
                    if isinstance(result, Exception):
                        analysis_results['tool_results'][tool_name] = {
                            'success': False,
                            'error': str(result)
                        }
                    else:
                        analysis_results['tool_results'][tool_name] = result
        else:
            # Run tools sequentially
            for tool_name in self.enabled_tools:
                if tool_name in tools:
                    result = await self._run_tool_async(tool_name, tools[tool_name])
                    analysis_results['tool_results'][tool_name] = result
        
        # Generate overall summary
        analysis_results['overall_summary'] = self._generate_overall_summary(analysis_results['tool_results'])
        
        # Save results
        self._save_results(analysis_results)
        
        return analysis_results
    
    def _get_tool_configurations(self) -> Dict[str, Dict[str, Any]]:
        """Get configurations for all available tools."""
        return {
            'formatting': {
                'name': 'Code Formatting',
                'command': 'poetry run ruff format .',
                'description': 'Format code using ruff formatter',
                'category': 'style'
            },
            'linting': {
                'name': 'Code Linting',
                'command': 'poetry run ruff check . --fix',
                'description': 'Check code style and potential issues using ruff',
                'category': 'style'
            },
            'type_checking': {
                'name': 'Static Type Checking',
                'command': 'poetry run mypy --ignore-missing-imports --package src',
                'description': 'Perform static type checking using mypy',
                'category': 'correctness'
            },
            'complexity': {
                'name': 'Cyclomatic Complexity Analysis',
                'command': 'poetry run radon cc src/ -s -nc',
                'description': 'Analyze code complexity using radon',
                'category': 'maintainability'
            },
            'maintainability': {
                'name': 'Maintainability Index',
                'command': 'poetry run radon mi src/ -s -nc',
                'description': 'Calculate maintainability index using radon',
                'category': 'maintainability'
            },
            'dead_code': {
                'name': 'Dead Code Detection',
                'command': 'poetry run vulture src/',
                'description': 'Find unused code using vulture',
                'category': 'correctness'
            },
            'circular_imports': {
                'name': 'Circular Import Detection',
                'command': 'poetry run pylint --disable=all --enable=cyclic-import src/',
                'description': 'Detect circular imports using pylint',
                'category': 'correctness'
            },
            'attribute_access': {
                'name': 'Attribute Access Pattern Analysis',
                'command': f'python3 src/utils/quick_attribute_checker.py {self.scan_path}',
                'description': 'Scan for unsafe attribute access patterns',
                'category': 'correctness',
                'custom_tool': True
            },
            'method_references': {
                'name': 'Method Reference Checking',
                'command': f'python3 src/utils/method_reference_checker.py {self.scan_path}',
                'description': 'Check for missing or invalid method references',
                'category': 'correctness',
                'custom_tool': True
            }
        }
    
    async def _run_tool_async(self, tool_name: str, tool_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single tool asynchronously."""
        self.logger.info(f"🔧 Running {tool_config['name']}...")
        
        try:
            if tool_config.get('custom_tool'):
                # Run custom tool (like our attribute access scanner)
                result = await self._run_custom_tool(tool_name, tool_config)
            else:
                # Run external tool
                result = await self._run_external_tool(tool_name, tool_config)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Tool {tool_name} failed: {e}")
            return {
                'tool': tool_name,
                'success': False,
                'error': str(e),
                'category': tool_config.get('category', 'unknown')
            }
    
    async def _run_external_tool(self, tool_name: str, tool_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run an external tool."""
        try:
            process = await asyncio.create_subprocess_shell(
                tool_config['command'],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            return {
                'tool': tool_name,
                'success': process.returncode == 0,
                'return_code': process.returncode,
                'stdout': stdout.decode() if stdout else '',
                'stderr': stderr.decode() if stderr else '',
                'category': tool_config.get('category', 'unknown')
            }
            
        except Exception as e:
            return {
                'tool': tool_name,
                'success': False,
                'error': str(e),
                'category': tool_config.get('category', 'unknown')
            }
    
    async def _run_custom_tool(self, tool_name: str, tool_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a custom tool (like our attribute access scanner)."""
        try:
            if tool_name == 'attribute_access':
                # Import and run our attribute access pipeline
                from src.code_quality.attribute_access_pipeline import AttributeAccessPipeline
                
                pipeline_config = {
                    'scan_path': self.scan_path,
                    'verbose': self.verbose
                }
                
                pipeline = AttributeAccessPipeline(pipeline_config)
                results = pipeline.run_comprehensive_scan()
                
                return {
                    'tool': tool_name,
                    'success': results['summary']['total_issues'] == 0,
                    'issues_found': results['summary']['total_issues'],
                    'results': results,
                    'category': tool_config.get('category', 'unknown')
                }
            
            return {
                'tool': tool_name,
                'success': False,
                'error': f'Unknown custom tool: {tool_name}',
                'category': tool_config.get('category', 'unknown')
            }
            
        except Exception as e:
            return {
                'tool': tool_name,
                'success': False,
                'error': str(e),
                'category': tool_config.get('category', 'unknown')
            }
    
    def _generate_overall_summary(self, tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an overall summary of all tool results."""
        total_tools = len(tool_results)
        successful_tools = sum(1 for result in tool_results.values() if result.get('success', False))
        failed_tools = total_tools - successful_tools
        
        # Categorize results
        categories = {}
        for tool_name, result in tool_results.items():
            category = result.get('category', 'unknown')
            if category not in categories:
                categories[category] = {'total': 0, 'successful': 0, 'failed': 0}
            
            categories[category]['total'] += 1
            if result.get('success', False):
                categories[category]['successful'] += 1
            else:
                categories[category]['failed'] += 1
        
        # Count issues
        total_issues = 0
        for result in tool_results.values():
            if 'issues_found' in result:
                total_issues += result['issues_found']
        
        return {
            'total_tools': total_tools,
            'successful_tools': successful_tools,
            'failed_tools': failed_tools,
            'success_rate': (successful_tools / total_tools * 100) if total_tools > 0 else 0,
            'total_issues': total_issues,
            'categories': categories
        }
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save the comprehensive analysis results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comprehensive_analysis_{timestamp}.json"
        filepath = self.report_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)
            
            self.logger.info(f"📄 Comprehensive results saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save comprehensive results: {e}")
    
    def generate_comprehensive_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive human-readable report."""
        report = []
        report.append("🚀 Comprehensive Code Quality Analysis Report")
        report.append("=" * 60)
        report.append(f"Analysis Time: {results['timestamp']}")
        report.append(f"Scan Path: {results['scan_path']}")
        report.append(f"Tools Enabled: {', '.join(results['enabled_tools'])}")
        report.append("")
        
        # Overall summary
        summary = results['overall_summary']
        report.append("📊 OVERALL SUMMARY:")
        report.append(f"  Total Tools: {summary['total_tools']}")
        report.append(f"  Successful: {summary['successful_tools']}")
        report.append(f"  Failed: {summary['failed_tools']}")
        report.append(f"  Success Rate: {summary['success_rate']:.1f}%")
        report.append(f"  Total Issues Found: {summary['total_issues']}")
        report.append("")
        
        # Category breakdown
        if summary['categories']:
            report.append("📋 CATEGORY BREAKDOWN:")
            for category, stats in summary['categories'].items():
                report.append(f"  {category.title()}:")
                report.append(f"    Tools: {stats['total']}")
                report.append(f"    Successful: {stats['successful']}")
                report.append(f"    Failed: {stats['failed']}")
            report.append("")
        
        # Individual tool results
        report.append("🔧 TOOL RESULTS:")
        for tool_name, result in results['tool_results'].items():
            status = "✅" if result.get('success', False) else "❌"
            report.append(f"  {status} {tool_name}")
            
            if not result.get('success', False):
                if 'error' in result:
                    report.append(f"    Error: {result['error']}")
                if 'issues_found' in result:
                    report.append(f"    Issues: {result['issues_found']}")
        
        return "\n".join(report)


async def main():
    """Main entry point for the comprehensive pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Code Quality Analysis Pipeline")
    parser.add_argument("--path", default="src", help="Path to scan")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--parallel", "-p", action="store_true", help="Run tools in parallel")
    parser.add_argument("--tools", nargs="+", help="Specific tools to run")
    parser.add_argument("--output", "-o", help="Output file for report")
    
    args = parser.parse_args()
    
    # Configure pipeline
    config = {
        'scan_path': args.path,
        'verbose': args.verbose,
        'parallel_execution': args.parallel,
        'enabled_tools': args.tools or None  # Use all tools if none specified
    }
    
    # Run comprehensive analysis
    pipeline = ComprehensivePipeline(config)
    results = await pipeline.run_comprehensive_analysis()
    
    # Generate and display report
    report = pipeline.generate_comprehensive_report(results)
    print(report)
    
    # Save report if requested
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"\n📄 Report saved to {args.output}")
    
    # Exit with appropriate code
    if results['overall_summary']['failed_tools'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
