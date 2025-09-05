#!/usr/bin/env python3
"""
Script Integration Analysis

This script analyzes all scripts in the code_quality directory and its subdirectories
to determine which ones are integrated into pipelines and which ones need integration.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
from datetime import datetime


class ScriptIntegrationAnalyzer:
    """Analyzes script integration status across the code_quality directory."""
    
    def __init__(self, code_quality_root: Path):
        self.code_quality_root = code_quality_root
        self.pipelines_dir = code_quality_root / "pipelines"
        self.analyzers_dir = code_quality_root / "analyzers"
        self.scripts_dir = code_quality_root / "scripts"
        self.fixers_dir = code_quality_root / "fixers"
        
        # Scripts that are already integrated into pipelines
        self.integrated_scripts = {
            "enhanced_import_analysis.py": "pipelines/enhanced_import_analysis.py",
            "intelligent_import_fixer.py": "pipelines/intelligent_import_fixer.py",
            "run_enhanced_import_analysis.py": "integrated via enhanced_import_analysis.py",
            "dead_code_analyzer.py": "pipelines/dead_code_analyzer.py",
            "code_interaction_mapper.py": "pipelines/code_interaction_mapper.py",
            "sequential_code_fixer.py": "pipelines/sequential_code_fixer.py",
            "complexity_cli.py": "pipelines/complexity_cli.py",
        }
        
        # Scripts that should be integrated but aren't yet
        self.scripts_needing_integration = []
        
        # Scripts that are standalone utilities (don't need pipeline integration)
        self.standalone_scripts = {
            "test_*.py": "Test files",
            "verify_*.py": "Verification utilities",
            "example_*.py": "Example scripts",
            "debug_*.py": "Debug utilities",
            "quick_start.py": "Quick start utility",
            "cli.py": "Main CLI interface",
            "__init__.py": "Package initialization",
        }
    
    def analyze_scripts(self) -> Dict[str, any]:
        """Analyze all scripts and their integration status."""
        results = {
            "analysis_timestamp": datetime.now().isoformat(),
            "total_scripts": 0,
            "integrated_scripts": [],
            "scripts_needing_integration": [],
            "standalone_scripts": [],
            "unknown_scripts": [],
            "integration_recommendations": []
        }
        
        # Find all Python files in the main directory
        main_scripts = list(self.code_quality_root.glob("*.py"))
        
        # Find all Python files in subdirectories
        subdir_scripts = []
        for subdir in ["analyzers", "scripts", "fixers", "utils", "reporters", "visualizers"]:
            subdir_path = self.code_quality_root / subdir
            if subdir_path.exists():
                subdir_scripts.extend(list(subdir_path.glob("*.py")))
        
        all_scripts = main_scripts + subdir_scripts
        results["total_scripts"] = len(all_scripts)
        
        for script_path in all_scripts:
            script_name = script_path.name
            relative_path = script_path.relative_to(self.code_quality_root)
            
            # Check if it's already integrated
            if script_name in self.integrated_scripts:
                results["integrated_scripts"].append({
                    "name": script_name,
                    "path": str(relative_path),
                    "integration_status": self.integrated_scripts[script_name]
                })
                continue
            
            # Check if it's a standalone script
            is_standalone = False
            for pattern, description in self.standalone_scripts.items():
                if pattern.replace("*", "") in script_name or script_name.startswith(pattern.replace("*", "")):
                    results["standalone_scripts"].append({
                        "name": script_name,
                        "path": str(relative_path),
                        "type": description
                    })
                    is_standalone = True
                    break
            
            if is_standalone:
                continue
            
            # Check if it has a main function (indicating it's a runnable script)
            has_main = self._has_main_function(script_path)
            
            if has_main:
                results["scripts_needing_integration"].append({
                    "name": script_name,
                    "path": str(relative_path),
                    "has_main": True,
                    "recommended_pipeline": self._recommend_pipeline(script_name, str(relative_path))
                })
            else:
                results["unknown_scripts"].append({
                    "name": script_name,
                    "path": str(relative_path),
                    "has_main": False
                })
        
        # Generate integration recommendations
        results["integration_recommendations"] = self._generate_recommendations(results)
        
        return results
    
    def _has_main_function(self, script_path: Path) -> bool:
        """Check if a script has a main function."""
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
                return 'def main(' in content and 'if __name__ == "__main__"' in content
        except Exception:
            return False
    
    def _recommend_pipeline(self, script_name: str, relative_path: str) -> str:
        """Recommend which pipeline a script should be integrated into."""
        if "import" in script_name.lower():
            return "enhanced_import_analysis or intelligent_import_fixer"
        elif "dead_code" in script_name.lower():
            return "dead_code_analyzer"
        elif "complexity" in script_name.lower():
            return "complexity_cli"
        elif "fix" in script_name.lower():
            return "sequential_code_fixer or auto_fixer_pipeline"
        elif "analyze" in script_name.lower():
            return "unified_enhanced_pipeline"
        elif "map" in script_name.lower() or "interaction" in script_name.lower():
            return "code_interaction_mapper"
        else:
            return "unified_enhanced_pipeline"
    
    def _generate_recommendations(self, results: Dict) -> List[Dict]:
        """Generate specific recommendations for script integration."""
        recommendations = []
        
        for script in results["scripts_needing_integration"]:
            recommendations.append({
                "script": script["name"],
                "action": "Create pipeline wrapper",
                "target_pipeline": script["recommended_pipeline"],
                "priority": "High" if "import" in script["name"].lower() or "fix" in script["name"].lower() else "Medium"
            })
        
        return recommendations
    
    def generate_report(self, results: Dict) -> str:
        """Generate a comprehensive integration report."""
        report = []
        report.append("# Script Integration Analysis Report")
        report.append(f"Generated: {results['analysis_timestamp']}")
        report.append("")
        
        report.append("## Summary")
        report.append(f"- Total scripts analyzed: {results['total_scripts']}")
        report.append(f"- Already integrated: {len(results['integrated_scripts'])}")
        report.append(f"- Need integration: {len(results['scripts_needing_integration'])}")
        report.append(f"- Standalone utilities: {len(results['standalone_scripts'])}")
        report.append(f"- Unknown/other: {len(results['unknown_scripts'])}")
        report.append("")
        
        if results['integrated_scripts']:
            report.append("## ✅ Integrated Scripts")
            for script in results['integrated_scripts']:
                report.append(f"- **{script['name']}** → {script['integration_status']}")
            report.append("")
        
        if results['scripts_needing_integration']:
            report.append("## ⚠️ Scripts Needing Integration")
            for script in results['scripts_needing_integration']:
                report.append(f"- **{script['name']}** ({script['path']})")
                report.append(f"  - Recommended pipeline: {script['recommended_pipeline']}")
            report.append("")
        
        if results['standalone_scripts']:
            report.append("## 🔧 Standalone Utilities")
            for script in results['standalone_scripts']:
                report.append(f"- **{script['name']}** - {script['type']}")
            report.append("")
        
        if results['integration_recommendations']:
            report.append("## 📋 Integration Recommendations")
            for rec in results['integration_recommendations']:
                report.append(f"- **{rec['script']}**: {rec['action']} → {rec['target_pipeline']} (Priority: {rec['priority']})")
            report.append("")
        
        return "\n".join(report)


def main():
    """Main function for script integration analysis."""
    code_quality_root = Path(__file__).parent
    analyzer = ScriptIntegrationAnalyzer(code_quality_root)
    
    print("🔍 Analyzing script integration status...")
    results = analyzer.analyze_scripts()
    
    print("📊 Generating integration report...")
    report = analyzer.generate_report(results)
    
    # Save report
    report_path = code_quality_root / "script_integration_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ Report saved to: {report_path}")
    print("\n" + "="*50)
    print(report)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())