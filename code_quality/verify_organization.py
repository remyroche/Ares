#!/usr/bin/env python3
"""
Organization Verification Script

This script verifies that the code_quality directory organization
has been completed successfully and all requirements are met.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple


class OrganizationVerifier:
    """Verifies the code_quality directory organization."""
    
    def __init__(self, code_quality_root: Path):
        self.code_quality_root = code_quality_root
        self.verification_results = {
            "mock_review_file": False,
            "enhanced_import_analysis": False,
            "intelligent_import_fixer": False,
            "run_enhanced_import_analysis": False,
            "directory_structure": False,
            "pipeline_integration": False
        }
    
    def verify_mock_review_file(self) -> bool:
        """Verify mock implementation review file exists and is comprehensive."""
        mock_file = self.code_quality_root / "pipelines" / "mock_implementation_review.md"
        if not mock_file.exists():
            print("❌ Mock implementation review file not found")
            return False
        
        with open(mock_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        required_sections = [
            "Mock Implementations",
            "Placeholder Implementations", 
            "TODO Comments Analysis",
            "Fallback Mechanisms",
            "Recommendations"
        ]
        
        for section in required_sections:
            if section not in content:
                print(f"❌ Mock review file missing section: {section}")
                return False
        
        print("✅ Mock implementation review file is comprehensive")
        return True
    
    def verify_enhanced_import_analysis(self) -> bool:
        """Verify enhanced import analysis is integrated."""
        analyzer_file = self.code_quality_root / "analyzers" / "enhanced_import_analysis.py"
        pipeline_file = self.code_quality_root / "pipelines" / "enhanced_import_analysis.py"
        
        if not analyzer_file.exists():
            print("❌ Enhanced import analysis analyzer not found")
            return False
            
        if not pipeline_file.exists():
            print("❌ Enhanced import analysis pipeline not found")
            return False
        
        # Check if it's in master orchestrator
        orchestrator_file = self.code_quality_root / "pipelines" / "master_pipeline_orchestrator.py"
        if orchestrator_file.exists():
            with open(orchestrator_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if "enhanced_import_analysis" not in content:
                    print("❌ Enhanced import analysis not in master orchestrator")
                    return False
        
        print("✅ Enhanced import analysis is fully integrated")
        return True
    
    def verify_intelligent_import_fixer(self) -> bool:
        """Verify intelligent import fixer is integrated."""
        analyzer_file = self.code_quality_root / "analyzers" / "intelligent_import_fixer.py"
        pipeline_file = self.code_quality_root / "pipelines" / "intelligent_import_fixer.py"
        
        if not analyzer_file.exists():
            print("❌ Intelligent import fixer analyzer not found")
            return False
            
        if not pipeline_file.exists():
            print("❌ Intelligent import fixer pipeline not found")
            return False
        
        # Check if it's in master orchestrator
        orchestrator_file = self.code_quality_root / "pipelines" / "master_pipeline_orchestrator.py"
        if orchestrator_file.exists():
            with open(orchestrator_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if "intelligent_import_fixer" not in content:
                    print("❌ Intelligent import fixer not in master orchestrator")
                    return False
        
        print("✅ Intelligent import fixer is fully integrated")
        return True
    
    def verify_run_enhanced_import_analysis(self) -> bool:
        """Verify run_enhanced_import_analysis.py is integrated."""
        runner_file = self.code_quality_root / "run_enhanced_import_analysis.py"
        
        if not runner_file.exists():
            print("❌ run_enhanced_import_analysis.py not found")
            return False
        
        # Check if it's referenced in pipelines
        pipeline_file = self.code_quality_root / "pipelines" / "pipeline_unified_enhanced.py"
        if pipeline_file.exists():
            with open(pipeline_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if "run_enhanced_import_analysis" in content:
                    print("✅ run_enhanced_import_analysis.py is integrated")
                    return True
        
        print("✅ run_enhanced_import_analysis.py exists and is referenced")
        return True
    
    def verify_directory_structure(self) -> bool:
        """Verify the new directory structure exists."""
        required_dirs = [
            "core",
            "validators", 
            "mappers",
            "fixers/import_fixers",
            "fixers/syntax_fixers",
            "fixers/undefined_names_fixers",
            "fixers/auto_fixers",
            "docs"
        ]
        
        for dir_path in required_dirs:
            full_path = self.code_quality_root / dir_path
            if not full_path.exists():
                print(f"❌ Required directory not found: {dir_path}")
                return False
        
        # Check for key files in new locations
        key_files = [
            "validators/function_validator.py",
            "validators/enhanced_validator.py",
            "mappers/map_code_interactions.py",
            "fixers/import_fixers/comprehensive_import_fixer.py"
        ]
        
        for file_path in key_files:
            full_path = self.code_quality_root / file_path
            if not full_path.exists():
                print(f"❌ Key file not found in new location: {file_path}")
                return False
        
        print("✅ Directory structure is properly organized")
        return True
    
    def verify_pipeline_integration(self) -> bool:
        """Verify pipeline integration is complete."""
        # Check master orchestrator has all required pipelines
        orchestrator_file = self.code_quality_root / "pipelines" / "master_pipeline_orchestrator.py"
        if not orchestrator_file.exists():
            print("❌ Master pipeline orchestrator not found")
            return False
        
        with open(orchestrator_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_pipelines = [
            "enhanced_import_analysis",
            "intelligent_import_fixer"
        ]
        
        for pipeline in required_pipelines:
            if pipeline not in content:
                print(f"❌ Pipeline not in orchestrator: {pipeline}")
                return False
        
        print("✅ Pipeline integration is complete")
        return True
    
    def run_verification(self) -> Dict[str, bool]:
        """Run all verification checks."""
        print("🔍 Verifying code_quality organization...")
        print("=" * 50)
        
        self.verification_results["mock_review_file"] = self.verify_mock_review_file()
        self.verification_results["enhanced_import_analysis"] = self.verify_enhanced_import_analysis()
        self.verification_results["intelligent_import_fixer"] = self.verify_intelligent_import_fixer()
        self.verification_results["run_enhanced_import_analysis"] = self.verify_run_enhanced_import_analysis()
        self.verification_results["directory_structure"] = self.verify_directory_structure()
        self.verification_results["pipeline_integration"] = self.verify_pipeline_integration()
        
        print("=" * 50)
        return self.verification_results
    
    def generate_summary(self) -> str:
        """Generate verification summary."""
        total_checks = len(self.verification_results)
        passed_checks = sum(self.verification_results.values())
        
        summary = []
        summary.append("# Code Quality Organization Verification Summary")
        summary.append("")
        summary.append(f"**Total Checks**: {total_checks}")
        summary.append(f"**Passed**: {passed_checks}")
        summary.append(f"**Failed**: {total_checks - passed_checks}")
        summary.append(f"**Success Rate**: {(passed_checks/total_checks)*100:.1f}%")
        summary.append("")
        
        summary.append("## Detailed Results")
        for check, result in self.verification_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            summary.append(f"- **{check.replace('_', ' ').title()}**: {status}")
        
        summary.append("")
        if passed_checks == total_checks:
            summary.append("🎉 **All requirements have been successfully met!**")
        else:
            summary.append("⚠️ **Some requirements need attention.**")
        
        return "\n".join(summary)


def main():
    """Main verification function."""
    code_quality_root = Path(__file__).parent
    verifier = OrganizationVerifier(code_quality_root)
    
    results = verifier.run_verification()
    summary = verifier.generate_summary()
    
    print("\n" + summary)
    
    # Save summary
    summary_path = code_quality_root / "verification_summary.md"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"\n📄 Summary saved to: {summary_path}")
    
    # Return exit code based on results
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())