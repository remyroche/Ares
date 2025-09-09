#!/usr/bin/env python3
"""
Unified Enhanced Pipeline

This pipeline integrates multiple code quality analysis tools including:
- Attribute Access Pattern Analysis
- Method Reference Checking
- Import-Free Analysis
- Comprehensive Code Quality Checks

Usage:
    python src/code_quality/pipelines/pipeline_unified_enhanced.py [--tools TOOLS] [--path PATH]
"""

import os
import sys
import subprocess
import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import argparse
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

class UnifiedEnhancedPipeline:
    """Unified pipeline for comprehensive code quality analysis."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the unified enhanced pipeline."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.results = {}
        self.report_dir = Path("src/code_quality/reports/unified_enhanced")
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
        # Pipeline configuration
        self.enabled_tools = self.config.get('enabled_tools', [
            'attribute_access', 'method_references', 'import_free_analysis',
            'comprehensive_analysis', 'quick_checks'
        ])
        self.scan_path = self.config.get('scan_path', 'src')
        self.verbose = self.config.get('verbose', False)
        self.parallel_execution = self.config.get('parallel_execution', False)
        
    async def run_unified_analysis(self) -> Dict[str, Any]:
        """Run unified enhanced analysis."""
        self.logger.info("🚀 Starting unified enhanced analysis...")
        
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
            'attribute_access': {
                'name': 'Enhanced Attribute Access Analysis',
                'command': f'python3 data_quality/attribute_access_analyzer.py --path {self.scan_path}',
                'description': 'Enhanced scan for unsafe attribute access patterns with intelligent filtering',
                'category': 'correctness',
                'custom_tool': True
            },
            'method_references': {
                'name': 'Enhanced Method Reference Analysis',
                'command': f'python3 data_quality/method_reference_analyzer.py --path {self.scan_path}',
                'description': 'Enhanced check for missing or invalid method references with false positive reduction',
                'category': 'correctness',
                'custom_tool': True
            },
            'import_free_analysis': {
                'name': 'Import-Free Analysis',
                'command': f'python3 src/code_quality/pipelines/import_free_analysis_pipeline.py --path {self.scan_path} --analysis-type all',
                'description': 'Comprehensive analysis without external dependencies',
                'category': 'analysis',
                'custom_tool': True
            },
            'comprehensive_analysis': {
                'name': 'Comprehensive Analysis',
                'command': f'python3 src/code_quality/comprehensive_pipeline.py --path {self.scan_path}',
                'description': 'Full comprehensive code quality analysis',
                'category': 'comprehensive',
                'custom_tool': True
            },
            'quick_checks': {
                'name': 'Quick Code Quality Checks',
                'command': f'python3 src/code_quality/attribute_access_pipeline.py --path {self.scan_path}',
                'description': 'Quick attribute access and method checks',
                'category': 'quick',
                'custom_tool': True
            }
        }
    
    async def _run_tool_async(self, tool_name: str, tool_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single tool asynchronously."""
        self.logger.info(f"🔧 Running {tool_config['name']}...")
        
        try:
            if tool_config.get('custom_tool'):
                # Run custom tool
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
        """Run a custom tool."""
        try:
            if tool_name == 'attribute_access':
                # Import and run our attribute access checker
                from src.utils.quick_attribute_checker import QuickAttributeChecker
                
                checker = QuickAttributeChecker()
                scan_path = Path(self.scan_path)
                
                if scan_path.is_file():
                    issues = checker.check_file(scan_path)
                else:
                    issues = checker.check_directory(scan_path)
                
                return {
                    'tool': tool_name,
                    'success': len(issues) == 0,
                    'issues_found': len(issues),
                    'issues': issues,
                    'category': tool_config.get('category', 'unknown')
                }
            
            elif tool_name == 'method_references':
                # Import and run our method reference checker
                from src.utils.method_reference_checker import MethodReferenceChecker
                
                checker = MethodReferenceChecker(verbose=self.verbose)
                scan_path = Path(self.scan_path)
                
                if scan_path.is_file():
                    issues = checker.scan_file(scan_path)
                else:
                    issues = checker.scan_directory(scan_path)
                
                return {
                    'tool': tool_name,
                    'success': len(issues) == 0,
                    'issues_found': len(issues),
                    'issues': issues,
                    'category': tool_config.get('category', 'unknown')
                }
            
            elif tool_name == 'import_free_analysis':
                # Import and run our import-free analyzer
                from src.code_quality.pipelines.import_free_analysis_pipeline import ImportFreeAnalyzer
                
                analyzer = ImportFreeAnalyzer(verbose=self.verbose)
                scan_path = Path(self.scan_path)
                
                analysis_types = ["method_references", "attribute_access", "import_analysis", 
                                "class_structure", "function_complexity"]
                results = analyzer.analyze_directory(scan_path, analysis_types)
                
                return {
                    'tool': tool_name,
                    'success': results['summary']['total_issues'] == 0,
                    'issues_found': results['summary']['total_issues'],
                    'results': results,
                    'category': tool_config.get('category', 'unknown')
                }
            
            elif tool_name == 'comprehensive_analysis':
                # Import and run our comprehensive pipeline
                from src.code_quality.comprehensive_pipeline import ComprehensivePipeline
                
                pipeline_config = {
                    'scan_path': self.scan_path,
                    'verbose': self.verbose,
                    'parallel_execution': False  # Avoid nested parallel execution
                }
                
                pipeline = ComprehensivePipeline(pipeline_config)
                results = await pipeline.run_comprehensive_analysis()
                
                return {
                    'tool': tool_name,
                    'success': results['overall_summary']['failed_tools'] == 0,
                    'issues_found': results['overall_summary']['total_issues'],
                    'results': results,
                    'category': tool_config.get('category', 'unknown')
                }
            
            elif tool_name == 'quick_checks':
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
        """Save the unified analysis results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"unified_enhanced_analysis_{timestamp}.json"
        filepath = self.report_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)
            
            self.logger.info(f"📄 Unified results saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save unified results: {e}")
    
    def generate_unified_report(self, results: Dict[str, Any]) -> str:
        """Generate a unified comprehensive report."""
        report = []
        report.append("🚀 Unified Enhanced Analysis Report")
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
            else:
                if 'issues_found' in result:
                    report.append(f"    Issues: {result['issues_found']}")
        
        return "\n".join(report)


async def main():
    """Main entry point for the unified enhanced pipeline."""
    parser = argparse.ArgumentParser(description="Unified Enhanced Code Quality Analysis Pipeline")
    parser.add_argument("--path", default="src", help="Path to analyze")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--parallel", "-p", action="store_true", help="Run tools in parallel")
    parser.add_argument("--tools", nargs="+", 
                       choices=["attribute_access", "method_references", "import_free_analysis",
                               "comprehensive_analysis", "quick_checks"],
                       help="Specific tools to run")
    parser.add_argument("--output", "-o", help="Output file for report")
    
    args = parser.parse_args()
    
    # Configure pipeline
    config = {
        'scan_path': args.path,
        'verbose': args.verbose,
        'parallel_execution': args.parallel,
        'enabled_tools': args.tools or None  # Use all tools if none specified
    }
    
    # Run unified analysis
    pipeline = UnifiedEnhancedPipeline(config)
    results = await pipeline.run_unified_analysis()
    
    # Generate and display report
    report = pipeline.generate_unified_report(results)
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
