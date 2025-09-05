#!/usr/bin/env python3
"""
Script Integration Manager

This script ensures all scripts in the code_quality directory and its sub-folders
are properly integrated into the pipeline system. It provides:

1. Script discovery and categorization
2. Integration status checking
3. Automatic integration suggestions
4. Pipeline organization management
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Any
from datetime import datetime
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class ScriptIntegrationManager:
    """Manages integration of all scripts into the pipeline system."""
    
    def __init__(self, code_quality_root: str = "/workspace/code_quality"):
        self.code_quality_root = Path(code_quality_root)
        self.pipelines_dir = self.code_quality_root / "pipelines"
        self.scripts_dir = self.code_quality_root / "scripts"
        
        # Script categories
        self.categories = {
            "analyzers": [],
            "fixers": [],
            "validators": [],
            "reporters": [],
            "visualizers": [],
            "utilities": [],
            "standalone": [],
            "integration": [],
            "testing": []
        }
        
        # Integration status
        self.integration_status = {
            "integrated": [],
            "partially_integrated": [],
            "not_integrated": [],
            "needs_review": []
        }
    
    def discover_all_scripts(self) -> Dict[str, List[Path]]:
        """Discover all Python scripts in the code_quality directory."""
        scripts = {}
        
        # Find all Python files
        for py_file in self.code_quality_root.rglob("*.py"):
            # Skip backup files, test files, and reports
            if (py_file.name.endswith(".backup") or 
                "test" in py_file.name.lower() or
                "reports" in str(py_file) or
                "tests" in str(py_file)):
                continue
            
            # Categorize the script
            category = self._categorize_script(py_file)
            if category not in scripts:
                scripts[category] = []
            scripts[category].append(py_file)
        
        return scripts
    
    def _categorize_script(self, script_path: Path) -> str:
        """Categorize a script based on its name and location."""
        name = script_path.name.lower()
        path_str = str(script_path)
        
        # Check directory-based categorization
        if "analyzers" in path_str:
            return "analyzers"
        elif "fixers" in path_str:
            return "fixers"
        elif "reporters" in path_str:
            return "reporters"
        elif "visualizers" in path_str:
            return "visualizers"
        elif "scripts" in path_str:
            return "utilities"
        elif "plugins" in path_str:
            return "utilities"
        
        # Check name-based categorization
        if any(keyword in name for keyword in ["analyzer", "analysis"]):
            return "analyzers"
        elif any(keyword in name for keyword in ["fix", "fixer", "repair"]):
            return "fixers"
        elif any(keyword in name for keyword in ["validate", "validator", "check"]):
            return "validators"
        elif any(keyword in name for keyword in ["report", "reporter", "summary"]):
            return "reporters"
        elif any(keyword in name for keyword in ["visualize", "visualizer", "graph", "plot"]):
            return "visualizers"
        elif any(keyword in name for keyword in ["run", "execute", "main"]):
            return "standalone"
        elif any(keyword in name for keyword in ["test", "mock"]):
            return "testing"
        elif any(keyword in name for keyword in ["integrate", "pipeline", "unified"]):
            return "integration"
        else:
            return "utilities"
    
    def check_integration_status(self, scripts: Dict[str, List[Path]]) -> Dict[str, Any]:
        """Check the integration status of all scripts."""
        status = {
            "total_scripts": 0,
            "integrated": 0,
            "partially_integrated": 0,
            "not_integrated": 0,
            "needs_review": 0,
            "details": {}
        }
        
        # Check each script
        for category, script_list in scripts.items():
            status["details"][category] = {
                "total": len(script_list),
                "integrated": [],
                "partially_integrated": [],
                "not_integrated": [],
                "needs_review": []
            }
            
            for script in script_list:
                status["total_scripts"] += 1
                integration_level = self._check_script_integration(script)
                
                if integration_level == "integrated":
                    status["integrated"] += 1
                    status["details"][category]["integrated"].append(str(script))
                elif integration_level == "partially_integrated":
                    status["partially_integrated"] += 1
                    status["details"][category]["partially_integrated"].append(str(script))
                elif integration_level == "needs_review":
                    status["needs_review"] += 1
                    status["details"][category]["needs_review"].append(str(script))
                else:
                    status["not_integrated"] += 1
                    status["details"][category]["not_integrated"].append(str(script))
        
        return status
    
    def _check_script_integration(self, script_path: Path) -> str:
        """Check if a script is integrated into the pipeline system."""
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for pipeline integration indicators
            pipeline_indicators = [
                "pipeline",
                "unified",
                "enhanced",
                "sequential",
                "plugin",
                "analyzer",
                "fixer"
            ]
            
            # Check if script is imported in pipeline files
            pipeline_files = list(self.pipelines_dir.glob("*.py"))
            is_imported = False
            
            for pipeline_file in pipeline_files:
                try:
                    with open(pipeline_file, 'r', encoding='utf-8') as pf:
                        pipeline_content = pf.read()
                    
                    script_name = script_path.stem
                    if script_name in pipeline_content:
                        is_imported = True
                        break
                except:
                    continue
            
            # Determine integration level
            if is_imported and any(indicator in content.lower() for indicator in pipeline_indicators):
                return "integrated"
            elif is_imported or any(indicator in content.lower() for indicator in pipeline_indicators):
                return "partially_integrated"
            elif "main" in content and "__name__" in content:
                return "needs_review"
            else:
                return "not_integrated"
                
        except Exception as e:
            return "needs_review"
    
    def generate_integration_report(self, scripts: Dict[str, List[Path]], status: Dict[str, Any]) -> str:
        """Generate a comprehensive integration report."""
        report = []
        report.append("=" * 80)
        report.append("SCRIPT INTEGRATION REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary
        report.append("SUMMARY:")
        report.append(f"  Total scripts: {status['total_scripts']}")
        report.append(f"  Integrated: {status['integrated']} ({status['integrated']/status['total_scripts']*100:.1f}%)")
        report.append(f"  Partially integrated: {status['partially_integrated']} ({status['partially_integrated']/status['total_scripts']*100:.1f}%)")
        report.append(f"  Not integrated: {status['not_integrated']} ({status['not_integrated']/status['total_scripts']*100:.1f}%)")
        report.append(f"  Needs review: {status['needs_review']} ({status['needs_review']/status['total_scripts']*100:.1f}%)")
        report.append("")
        
        # Category breakdown
        report.append("CATEGORY BREAKDOWN:")
        for category, details in status["details"].items():
            if details["total"] > 0:
                report.append(f"\n{category.upper()}:")
                report.append(f"  Total: {details['total']}")
                report.append(f"  Integrated: {len(details['integrated'])}")
                report.append(f"  Partially integrated: {len(details['partially_integrated'])}")
                report.append(f"  Not integrated: {len(details['not_integrated'])}")
                report.append(f"  Needs review: {len(details['needs_review'])}")
        
        # Detailed lists
        report.append("\n" + "=" * 80)
        report.append("DETAILED INTEGRATION STATUS")
        report.append("=" * 80)
        
        for category, details in status["details"].items():
            if details["total"] > 0:
                report.append(f"\n{category.upper()}:")
                
                if details["integrated"]:
                    report.append("  ✅ INTEGRATED:")
                    for script in details["integrated"]:
                        report.append(f"    - {script}")
                
                if details["partially_integrated"]:
                    report.append("  ⚠️  PARTIALLY INTEGRATED:")
                    for script in details["partially_integrated"]:
                        report.append(f"    - {script}")
                
                if details["not_integrated"]:
                    report.append("  ❌ NOT INTEGRATED:")
                    for script in details["not_integrated"]:
                        report.append(f"    - {script}")
                
                if details["needs_review"]:
                    report.append("  🔍 NEEDS REVIEW:")
                    for script in details["needs_review"]:
                        report.append(f"    - {script}")
        
        # Recommendations
        report.append("\n" + "=" * 80)
        report.append("RECOMMENDATIONS")
        report.append("=" * 80)
        
        if status["not_integrated"] > 0:
            report.append(f"\n1. Integrate {status['not_integrated']} scripts that are not integrated:")
            for category, details in status["details"].items():
                if details["not_integrated"]:
                    report.append(f"   - {category}: {len(details['not_integrated'])} scripts")
        
        if status["partially_integrated"] > 0:
            report.append(f"\n2. Complete integration for {status['partially_integrated']} partially integrated scripts")
        
        if status["needs_review"] > 0:
            report.append(f"\n3. Review {status['needs_review']} scripts that need manual review")
        
        report.append("\n4. Consider creating specialized pipelines for:")
        report.append("   - Standalone utilities")
        report.append("   - Testing and validation scripts")
        report.append("   - Reporting and visualization tools")
        
        return "\n".join(report)
    
    def create_pipeline_organization_plan(self, scripts: Dict[str, List[Path]]) -> Dict[str, Any]:
        """Create a plan for organizing scripts into pipelines."""
        plan = {
            "core_pipelines": {
                "unified_enhanced_pipeline": {
                    "description": "Comprehensive analysis and fixing pipeline",
                    "scripts": []
                },
                "sequential_code_fixer": {
                    "description": "Sequential code fixing pipeline",
                    "scripts": []
                },
                "code_interaction_mapper": {
                    "description": "Code interaction and dependency mapping",
                    "scripts": []
                },
                "dead_code_analyzer": {
                    "description": "Dead code detection and removal",
                    "scripts": []
                },
                "complexity_cli": {
                    "description": "Code complexity analysis",
                    "scripts": []
                }
            },
            "specialized_pipelines": {
                "validation_pipeline": {
                    "description": "Code validation and testing",
                    "scripts": []
                },
                "reporting_pipeline": {
                    "description": "Report generation and visualization",
                    "scripts": []
                },
                "utility_pipeline": {
                    "description": "Utility scripts and tools",
                    "scripts": []
                }
            },
            "recommendations": []
        }
        
        # Categorize scripts for pipeline assignment
        for category, script_list in scripts.items():
            for script in script_list:
                script_name = script.stem
                
                # Assign to appropriate pipeline
                if category in ["analyzers", "fixers"]:
                    plan["core_pipelines"]["unified_enhanced_pipeline"]["scripts"].append(str(script))
                elif "sequential" in script_name or "fix" in script_name:
                    plan["core_pipelines"]["sequential_code_fixer"]["scripts"].append(str(script))
                elif "interaction" in script_name or "map" in script_name:
                    plan["core_pipelines"]["code_interaction_mapper"]["scripts"].append(str(script))
                elif "dead" in script_name or "unused" in script_name:
                    plan["core_pipelines"]["dead_code_analyzer"]["scripts"].append(str(script))
                elif "complexity" in script_name:
                    plan["core_pipelines"]["complexity_cli"]["scripts"].append(str(script))
                elif category in ["validators", "testing"]:
                    plan["specialized_pipelines"]["validation_pipeline"]["scripts"].append(str(script))
                elif category in ["reporters", "visualizers"]:
                    plan["specialized_pipelines"]["reporting_pipeline"]["scripts"].append(str(script))
                else:
                    plan["specialized_pipelines"]["utility_pipeline"]["scripts"].append(str(script))
        
        # Generate recommendations
        plan["recommendations"] = [
            "Create a master pipeline orchestrator that can run all pipelines",
            "Implement pipeline dependency management",
            "Add pipeline configuration management",
            "Create pipeline execution monitoring and reporting",
            "Implement pipeline result aggregation"
        ]
        
        return plan
    
    def run_full_analysis(self) -> Dict[str, Any]:
        """Run a full analysis of script integration."""
        print("🔍 Discovering all scripts...")
        scripts = self.discover_all_scripts()
        
        print("📊 Checking integration status...")
        status = self.check_integration_status(scripts)
        
        print("📋 Generating integration report...")
        report = self.generate_integration_report(scripts, status)
        
        print("🎯 Creating pipeline organization plan...")
        plan = self.create_pipeline_organization_plan(scripts)
        
        return {
            "scripts": scripts,
            "status": status,
            "report": report,
            "plan": plan
        }


def main():
    """Main entry point for the script integration manager."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Script Integration Manager")
    parser.add_argument("--output", "-o", help="Output file for report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Initialize the manager
    manager = ScriptIntegrationManager()
    
    # Run full analysis
    results = manager.run_full_analysis()
    
    # Display report
    print(results["report"])
    
    # Save report if output file specified
    if args.output:
        with open(args.output, 'w') as f:
            f.write(results["report"])
        print(f"\nReport saved to {args.output}")
    
    # Save detailed results as JSON
    json_output = args.output.replace('.txt', '.json') if args.output else 'script_integration_results.json'
    with open(json_output, 'w') as f:
        # Convert Path objects to strings for JSON serialization
        json_results = {
            "status": results["status"],
            "plan": results["plan"],
            "scripts": {k: [str(p) for p in v] for k, v in results["scripts"].items()}
        }
        json.dump(json_results, f, indent=2)
    print(f"Detailed results saved to {json_output}")


if __name__ == "__main__":
    main()