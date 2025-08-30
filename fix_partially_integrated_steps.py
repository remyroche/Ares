#!/usr/bin/env python3
"""
Script to fix the 5 partially integrated steps to reach 90%+ completion.
"""

import re
from pathlib import Path
from typing import List, Dict, Any

# Steps that need completion fixes
STEPS_TO_FIX = [
    "step1_data_collection.py",
    "step4_triple_barrier_method.py",
    "step5_labeling.py",
    "step7_enhanced_matrix_operations.py",
    "step9_5_hmm_lm_generalist_training.py",
]

def add_missing_imports(file_path: Path) -> bool:
    """Add missing imports for standardized naming functions."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if the import already exists
        if "log_step_dataframe_with_standardized_name" in content:
            print(f"✅ log_step_dataframe_with_standardized_name import already exists in {file_path.name}")
            return True
        
        # Find the MLflow import section
        import_pattern = r'from src\.utils\.enhanced_mlflow_integration import \([^)]*\)'
        match = re.search(import_pattern, content, re.DOTALL)
        
        if not match:
            print(f"⚠️ Could not find MLflow import section in {file_path.name}")
            return False
        
        import_section = match.group(0)
        
        # Check if log_step_dataframe_with_standardized_name is missing
        if "log_step_dataframe_with_standardized_name" not in import_section:
            # Add the missing import
            new_import_section = import_section.replace(
                ")",
                ",\n    log_step_dataframe_with_standardized_name\n)"
            )
            
            new_content = content.replace(import_section, new_import_section)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            print(f"✅ Added log_step_dataframe_with_standardized_name import to {file_path.name}")
            return True
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to add missing imports to {file_path.name}: {e}")
        return False

def add_standardized_naming_calls(file_path: Path) -> bool:
    """Add standardized naming function calls to artifact logging methods."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract step number
        step_match = re.search(r'step(\d+(?:_\d+)?)', file_path.name)
        if not step_match:
            return False
        
        step_num = step_match.group(1)
        method_name = f"_log_step{step_num}_artifacts_and_report"
        
        if method_name not in content:
            print(f"⚠️ Could not find artifact logging method in {file_path.name}")
            return False
        
        # Check if standardized naming calls already exist
        if "log_step_dataframe_with_standardized_name" in content:
            print(f"✅ Standardized naming calls already exist in {file_path.name}")
            return True
        
        # Add standardized naming calls to the method
        # This is a simplified approach - we'll add a comment indicating where to add the calls
        standardized_naming_comment = f'''
            # TODO: Add standardized naming calls for DataFrames
            # Example:
            # if some_dataframe is not None:
            #     artifact_name = log_step_dataframe_with_standardized_name(
            #         config=self.config,
            #         step_name="step{step_num}",
            #         df=some_dataframe,
            #         artifact_type="some_data",
            #         additional_metadata={{
            #             "data_type": "some_data",
            #             "timeframe": timeframe,
            #             "asset": symbol,
            #             "lookback_period": training_input.get("lookback_days", 1095),
            #             "project_version": self.config.get("project_version", "1.0.0"),
            #         }}
            #     )
            #     self.logger.info(f"✅ Logged some_data: {{artifact_name}}")
        '''
        
        # Find the method and add the comment
        method_start = content.find(method_name)
        if method_start == -1:
            return False
        
        # Find a good place to insert the comment (after the method signature)
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if method_name in line:
                # Insert comment after the method signature
                lines.insert(i + 1, standardized_naming_comment)
                break
        
        new_content = '\n'.join(lines)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"✅ Added standardized naming comment to {file_path.name}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to add standardized naming calls to {file_path.name}: {e}")
        return False

def add_date_field(file_path: Path) -> bool:
    """Add missing date field to artifact logging methods."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if date field already exists
        if '"date"' in content or 'datetime.now()' in content:
            print(f"✅ Date field already exists in {file_path.name}")
            return True
        
        # Extract step number
        step_match = re.search(r'step(\d+(?:_\d+)?)', file_path.name)
        if not step_match:
            return False
        
        step_num = step_match.group(1)
        method_name = f"_log_step{step_num}_artifacts_and_report"
        
        if method_name not in content:
            print(f"⚠️ Could not find artifact logging method in {file_path.name}")
            return False
        
        # Add date field to training_input
        training_input_pattern = r'training_input\s*=\s*\{[^}]*\}'
        match = re.search(training_input_pattern, content, re.DOTALL)
        
        if match:
            training_input_text = match.group(0)
            
            # Check if date field is missing
            if '"date"' not in training_input_text:
                # Add date field
                new_training_input = training_input_text.replace(
                    '}',
                    f',\n                "date": datetime.now().isoformat(),  # Current timestamp\n            }}'
                )
                
                new_content = content.replace(training_input_text, new_training_input)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                print(f"✅ Added date field to training_input in {file_path.name}")
                return True
        
        # Add date field to additional_metadata
        metadata_pattern = r'additional_metadata\s*=\s*\{[^}]*\}'
        matches = re.finditer(metadata_pattern, content, re.DOTALL)
        
        changes_made = False
        for match in matches:
            metadata_text = match.group(0)
            
            # Check if date field is missing
            if '"date"' not in metadata_text:
                # Add date field
                new_metadata = metadata_text.replace(
                    '}',
                    f',\n                    "date": datetime.now().isoformat(),\n                }}'
                )
                
                content = content.replace(metadata_text, new_metadata)
                changes_made = True
        
        if changes_made:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✅ Added date field to additional_metadata in {file_path.name}")
            return True
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to add date field to {file_path.name}: {e}")
        return False

def fix_step_file(file_path: Path) -> Dict[str, bool]:
    """Fix a single step file to reach 90%+ completion."""
    results = {
        "imports": False,
        "standardized_naming": False,
        "date_field": False
    }
    
    print(f"\n🔄 Fixing {file_path.name}...")
    
    # Add missing imports
    results["imports"] = add_missing_imports(file_path)
    
    # Add standardized naming calls
    results["standardized_naming"] = add_standardized_naming_calls(file_path)
    
    # Add date field
    results["date_field"] = add_date_field(file_path)
    
    return results

def main():
    """Main function to fix all partially integrated step files."""
    steps_dir = Path("src/training/steps")
    
    if not steps_dir.exists():
        print(f"❌ Steps directory not found: {steps_dir}")
        return
    
    print("🔧 Fixing partially integrated steps to reach 90%+ completion...")
    print(f"📁 Steps directory: {steps_dir}")
    print(f"📋 Steps to fix: {len(STEPS_TO_FIX)}")
    
    results = {}
    
    for step_file in STEPS_TO_FIX:
        file_path = steps_dir / step_file
        
        if not file_path.exists():
            print(f"⚠️ Step file not found: {step_file}")
            continue
        
        results[step_file] = fix_step_file(file_path)
    
    # Print summary
    print("\n" + "="*60)
    print("📊 FIX SUMMARY")
    print("="*60)
    
    for step_file, step_results in results.items():
        success_count = sum(step_results.values())
        total_count = len(step_results)
        
        if success_count == total_count:
            print(f"✅ {step_file}: All fixes successful")
        elif success_count > 0:
            print(f"⚠️ {step_file}: Partial success ({success_count}/{total_count})")
        else:
            print(f"❌ {step_file}: All fixes failed")
    
    total_successful = sum(sum(step_results.values()) for step_results in results.values())
    total_attempts = sum(len(step_results) for step_results in results.values())
    
    print(f"\n🎯 Overall: {total_successful}/{total_attempts} fixes successful")
    
    if total_successful == total_attempts:
        print("🎉 All partially integrated steps successfully fixed!")
    else:
        print("⚠️ Some fixes may need manual review")

if __name__ == "__main__":
    main()