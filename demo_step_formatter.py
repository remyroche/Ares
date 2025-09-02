#!/usr/bin/env python3
"""
Demonstration script showing how to use the StepFormatter class programmatically.
"""

from step_formatter import StepFormatter
import tempfile
import os
from pathlib import Path

def demo_programmatic_usage():
    """Demonstrate using the StepFormatter class directly in Python code."""
    
    print("=" * 60)
    print("STEP FORMATTER - PROGRAMMATIC USAGE DEMO")
    print("=" * 60)
    
    # Create a temporary directory for demonstration
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create sample files with step mentions
        sample_files = {
            'workflow_step1.py': '''#!/usr/bin/env python3
"""
Sample workflow with step mentions.
"""

def step1_initialize():
    """Initialize the system."""
    print("Running step1")
    return True

def step2_process():
    """Process data."""
    print("Running step2")
    return True

def step3_cleanup():
    """Clean up resources."""
    print("Running step3")
    return True

def main():
    """Main workflow."""
    step1_initialize()
    step2_process()
    step3_cleanup()
    print("Workflow completed!")
''',
            'config_step1.json': '''{
  "workflow": {
    "steps": [
      {"id": "step1", "name": "Initialize"},
      {"id": "step2", "name": "Process"},
      {"id": "step3", "name": "Cleanup"}
    ]
  }
}''',
            'README_step1.md': '''# Workflow Documentation

## Steps

1. **step1**: Initialize the system
2. **step2**: Process the data
3. **step3**: Clean up resources

## Usage

```python
step1_initialize()
step2_process()
step3_cleanup()
```
'''
        }
        
        print(f"Creating sample files in: {temp_path}")
        
        # Create the sample files
        for filename, content in sample_files.items():
            file_path = temp_path / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  Created: {filename}")
        
        print("\n" + "-" * 40)
        print("BEFORE FORMATTING:")
        print("-" * 40)
        
        # Show content before formatting
        for filename in sample_files.keys():
            file_path = temp_path / filename
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                step_mentions = content.count('step1') + content.count('step2') + content.count('step3')
                print(f"  {filename}: {step_mentions} step mentions")
        
        print("\n" + "-" * 40)
        print("RUNNING STEP FORMATTER:")
        print("-" * 40)
        
        # Create formatter instance
        formatter = StepFormatter(dry_run=False, backup=True)
        
        # Process the temporary directory
        stats = formatter.process_directory(temp_path, recursive=False)
        
        print(f"  Files processed: {stats['files_processed']}")
        print(f"  Content changes: {stats['content_changes']}")
        print(f"  Filename changes: {stats['filename_changes']}")
        print(f"  Total changes: {stats['total_changes']}")
        
        print("\n" + "-" * 40)
        print("AFTER FORMATTING:")
        print("-" * 40)
        
        # Show content after formatting
        for filename in sample_files.keys():
            file_path = temp_path / filename
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                step_mentions = content.count('step01') + content.count('step02') + content.count('step03')
                print(f"  {filename}: {step_mentions} formatted step mentions")
        
        print("\n" + "-" * 40)
        print("SAMPLE OF FORMATTED CONTENT:")
        print("-" * 40)
        
        # Show a sample of the formatted content
        sample_file = temp_path / 'workflow_step1.py'
        with open(sample_file, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
            for i, line in enumerate(lines[:15], 1):
                if 'step' in line:
                    print(f"  Line {i:2d}: {line}")
        
        print("\n" + "-" * 40)
        print("BACKUP FILES CREATED:")
        print("-" * 40)
        
        # Show backup files
        backup_files = list(temp_path.glob('*.backup'))
        for backup_file in backup_files:
            print(f"  {backup_file.name}")
        
        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETED!")
        print("=" * 60)

def demo_content_formatting():
    """Demonstrate content formatting directly."""
    
    print("\n" + "=" * 60)
    print("CONTENT FORMATTING DEMO")
    print("=" * 60)
    
    # Sample content with step mentions
    sample_content = """
    This is a sample workflow:
    1. step1: Initialize
    2. step2: Process
    3. step3: Report
    4. step4: Cleanup
    
    The steps should be executed in order:
    - step1 must complete before step2
    - step2 and step3 can run in parallel
    - step4 depends on step2 and step3
    """
    
    print("Original content:")
    print(sample_content)
    
    # Create formatter and format content
    formatter = StepFormatter()
    formatted_content, changes = formatter.format_step_content(sample_content)
    
    print(f"\nFormatted content ({changes} changes made):")
    print(formatted_content)
    
    print("\n" + "=" * 60)

def demo_filename_formatting():
    """Demonstrate filename formatting directly."""
    
    print("\n" + "=" * 60)
    print("FILENAME FORMATTING DEMO")
    print("=" * 60)
    
    # Sample filenames with step mentions
    sample_filenames = [
        "workflow_step1.py",
        "config_step2.json",
        "report_step3.md",
        "data_step4.csv",
        "script_step5.sh"
    ]
    
    print("Original filenames:")
    for filename in sample_filenames:
        print(f"  {filename}")
    
    # Create formatter and format filenames
    formatter = StepFormatter()
    
    print(f"\nFormatted filenames:")
    for filename in sample_filenames:
        formatted_name, changes = formatter.format_filename(filename)
        if changes > 0:
            print(f"  {filename} -> {formatted_name}")
        else:
            print(f"  {filename} (no changes)")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    try:
        # Run all demonstrations
        demo_programmatic_usage()
        demo_content_formatting()
        demo_filename_formatting()
        
        print("\n🎉 All demonstrations completed successfully!")
        print("\nTo use the step formatter in your own code:")
        print("  from step_formatter import StepFormatter")
        print("  formatter = StepFormatter(dry_run=False, backup=True)")
        print("  stats = formatter.process_directory('/path/to/directory')")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()