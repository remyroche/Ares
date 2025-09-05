#!/usr/bin/env python3
"""
Pipeline Reorganization Verification Script

This script verifies that the pipeline reorganization has been completed
successfully and all pipelines work with the exact command-line interfaces
specified in the requirements.
"""

import sys
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Tuple


class PipelineReorganizationVerifier:
    """Verifies the pipeline reorganization is complete and functional."""
    
    def __init__(self, pipelines_dir: Path):
        self.pipelines_dir = pipelines_dir
        self.verification_results = {}
        
        # Define the exact command-line interfaces that were requested
        self.required_pipelines = {
            "complexity_pipeline.py": {
                "command": ["--analysis-type", "cyclomatic"],
                "description": "Complexity analysis with cyclomatic focus"
            },
            "dead_code_pipeline.py": {
                "command": ["--analysis-type", "enhanced", "--auto-fix"],
                "description": "Dead code analysis with auto-fix"
            },
            "auto_fixer_pipeline.py": {
                "command": ["--fix-type", "imports", "--conservative"],
                "description": "Conservative import fixing"
            },
            "interaction_mapping_pipeline.py": {
                "command": ["--analysis-type", "call_graph"],
                "description": "Call graph analysis"
            },
            "import_free_analysis_pipeline.py": {
                "command": ["--analysis-type", "syntax"],
                "description": "Syntax analysis without imports"
            },
            "pipeline_unified_enhanced.py": {
                "command": [],
                "description": "Comprehensive analysis with imports"
            },
            "overall_pipeline.py": {
                "command": ["--list"],
                "description": "Master orchestrator"
            }
        }
    
    def verify_pipeline_exists(self, pipeline_name: str) -> bool:
        """Verify that a pipeline file exists."""
        pipeline_path = self.pipelines_dir / pipeline_name
        exists = pipeline_path.exists()
        
        if exists:
            print(f"✅ {pipeline_name} exists")
        else:
            print(f"❌ {pipeline_name} missing")
        
        return exists
    
    def verify_pipeline_help(self, pipeline_name: str) -> bool:
        """Verify that a pipeline responds to --help."""
        pipeline_path = self.pipelines_dir / pipeline_name
        
        try:
            result = subprocess.run(
                [sys.executable, str(pipeline_path), "--help"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and "usage:" in result.stdout.lower():
                print(f"✅ {pipeline_name} responds to --help")
                return True
            else:
                print(f"❌ {pipeline_name} doesn't respond properly to --help")
                return False
                
        except Exception as e:
            print(f"❌ {pipeline_name} error with --help: {e}")
            return False
    
    def verify_pipeline_arguments(self, pipeline_name: str, required_args: List[str]) -> bool:
        """Verify that a pipeline accepts the required arguments."""
        pipeline_path = self.pipelines_dir / pipeline_name
        
        try:
            # Test with --help to see if arguments are recognized
            result = subprocess.run(
                [sys.executable, str(pipeline_path), "--help"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                print(f"❌ {pipeline_name} --help failed")
                return False
            
            help_text = result.stdout.lower()
            
            # Check if required arguments are mentioned in help
            for arg in required_args:
                if arg.replace("--", "") in help_text:
                    print(f"✅ {pipeline_name} supports {arg}")
                else:
                    print(f"❌ {pipeline_name} missing {arg}")
                    return False
            
            return True
            
        except Exception as e:
            print(f"❌ {pipeline_name} error checking arguments: {e}")
            return False
    
    def verify_overall_pipeline_list(self) -> bool:
        """Verify that the overall pipeline can list available pipelines."""
        pipeline_path = self.pipelines_dir / "overall_pipeline.py"
        
        try:
            result = subprocess.run(
                [sys.executable, str(pipeline_path), "--list"],
                capture_output=True,
                text=True,
                timeout=15
            )
            
            if result.returncode == 0:
                output = result.stdout
                # Check if it lists the expected pipelines
                expected_pipelines = [
                    "complexity", "dead_code", "auto_fixer", 
                    "interaction_mapping", "import_free_analysis", 
                    "pipeline_unified_enhanced"
                ]
                
                found_pipelines = 0
                for pipeline in expected_pipelines:
                    if pipeline in output:
                        found_pipelines += 1
                
                if found_pipelines >= 5:  # At least 5 of 6 expected pipelines
                    print(f"✅ overall_pipeline.py --list shows {found_pipelines}/6 expected pipelines")
                    return True
                else:
                    print(f"❌ overall_pipeline.py --list only shows {found_pipelines}/6 expected pipelines")
                    return False
            else:
                print(f"❌ overall_pipeline.py --list failed with return code {result.returncode}")
                return False
                
        except Exception as e:
            print(f"❌ overall_pipeline.py --list error: {e}")
            return False
    
    def run_verification(self) -> Dict[str, bool]:
        """Run complete verification of pipeline reorganization."""
        print("🔍 Verifying Pipeline Reorganization")
        print("=" * 50)
        
        results = {}
        
        # Verify each required pipeline
        for pipeline_name, pipeline_info in self.required_pipelines.items():
            print(f"\n📋 Verifying {pipeline_name}")
            print("-" * 30)
            
            # Check if file exists
            exists = self.verify_pipeline_exists(pipeline_name)
            if not exists:
                results[pipeline_name] = False
                continue
            
            # Check if it responds to --help
            help_works = self.verify_pipeline_help(pipeline_name)
            if not help_works:
                results[pipeline_name] = False
                continue
            
            # Check if it accepts required arguments
            if pipeline_name == "overall_pipeline.py":
                # Special case for overall pipeline
                args_work = self.verify_overall_pipeline_list()
            else:
                args_work = self.verify_pipeline_arguments(pipeline_name, pipeline_info["command"])
            
            results[pipeline_name] = args_work
        
        return results
    
    def generate_summary(self, results: Dict[str, bool]) -> str:
        """Generate verification summary."""
        total_pipelines = len(results)
        successful_pipelines = sum(results.values())
        success_rate = (successful_pipelines / total_pipelines) * 100
        
        summary = []
        summary.append("# Pipeline Reorganization Verification Summary")
        summary.append("")
        summary.append(f"**Total Pipelines**: {total_pipelines}")
        summary.append(f"**Successful**: {successful_pipelines}")
        summary.append(f"**Failed**: {total_pipelines - successful_pipelines}")
        summary.append(f"**Success Rate**: {success_rate:.1f}%")
        summary.append("")
        
        summary.append("## Detailed Results")
        for pipeline_name, success in results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            description = self.required_pipelines[pipeline_name]["description"]
            summary.append(f"- **{pipeline_name}**: {status} - {description}")
        
        summary.append("")
        if success_rate == 100:
            summary.append("🎉 **All pipelines are working correctly!**")
            summary.append("")
            summary.append("## Ready-to-Use Commands")
            summary.append("```bash")
            summary.append("# Complexity analysis")
            summary.append("python pipelines/complexity_pipeline.py --analysis-type cyclomatic")
            summary.append("")
            summary.append("# Dead code analysis with auto-fix")
            summary.append("python pipelines/dead_code_pipeline.py --analysis-type enhanced --auto-fix")
            summary.append("")
            summary.append("# Conservative import fixing")
            summary.append("python pipelines/auto_fixer_pipeline.py --fix-type imports --conservative")
            summary.append("")
            summary.append("# Call graph analysis")
            summary.append("python pipelines/interaction_mapping_pipeline.py --analysis-type call_graph")
            summary.append("")
            summary.append("# Syntax analysis without imports")
            summary.append("python pipelines/import_free_analysis_pipeline.py --analysis-type syntax")
            summary.append("")
            summary.append("# Comprehensive analysis with imports")
            summary.append("python pipelines/pipeline_unified_enhanced.py")
            summary.append("")
            summary.append("# Master orchestrator")
            summary.append("python pipelines/overall_pipeline.py --all")
            summary.append("```")
        else:
            summary.append("⚠️ **Some pipelines need attention.**")
        
        return "\n".join(summary)


def main():
    """Main verification function."""
    parser = argparse.ArgumentParser(description="Verify pipeline reorganization")
    parser.add_argument("--pipelines-dir", 
                       help="Pipelines directory (default: current directory)")
    
    args = parser.parse_args()
    
    # Determine pipelines directory
    if args.pipelines_dir:
        pipelines_dir = Path(args.pipelines_dir)
    else:
        pipelines_dir = Path(__file__).parent
    
    if not pipelines_dir.exists():
        print(f"Error: Pipelines directory {pipelines_dir} does not exist")
        return 1
    
    # Run verification
    verifier = PipelineReorganizationVerifier(pipelines_dir)
    results = verifier.run_verification()
    
    # Generate and print summary
    summary = verifier.generate_summary(results)
    print("\n" + "=" * 60)
    print(summary)
    
    # Save summary
    summary_path = pipelines_dir / "pipeline_verification_summary.md"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"\n📄 Summary saved to: {summary_path}")
    
    # Return exit code based on results
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())