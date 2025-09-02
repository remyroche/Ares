#!/usr/bin/env python3
"""
Test script for the step formatter.
Creates sample files with step mentions to test the formatter.
"""

import os
from pathlib import Path

def create_test_files():
    """Create test files with step mentions to demonstrate the formatter."""
    
    # Test file 1: Python file with step mentions
    test_py_content = '''#!/usr/bin/env python3
"""
Test script with step mentions.
"""

def step1_function():
    """This is step01 function."""
    print("Executing step01")
    return "step01 completed"

def step2_function():
    """This is step02 function."""
    print("Executing step02")
    return "step02 completed"

def step3_function():
    """This is step03 function."""
    print("Executing step03")
    return "step03 completed"

def main():
    """Main function that calls all steps."""
    result1 = step1_function()
    result2 = step2_function()
    result3 = step3_function()
    
    print(f"Results: {result1}, {result2}, {result3}")
    
    # Check if step04 exists
    if hasattr(step4_function, '__call__'):
        step4_function()
    else:
        print("step04 not implemented yet")

if __name__ == "__main__":
    main()
'''
    
    # Test file 2: Markdown file with step mentions
    test_md_content = '''# Test Documentation

This document contains various step mentions.

## Process Steps

1. **step01**: Initialize the system
2. **step02**: Load configuration
3. **step03**: Process data
4. **step04**: Generate report
5. **step05**: Clean up resources

## Implementation Notes

- step01 should be called first
- step02 depends on step01 completion
- step03 can run in parallel with step02
- step04 requires both step02 and step03
- step05 is optional but recommended

## Code Example

```python
# Execute steps in order
step01()
step02()
step03()
step04()
step05()
```

## Troubleshooting

If step01 fails, check the configuration.
If step02 fails, verify step01 completed successfully.
If step03 fails, check data integrity.
If step04 fails, ensure step02 and step03 completed.
If step05 fails, it's not critical but should be investigated.
'''
    
    # Test file 3: JSON file with step mentions
    test_json_content = '''{
  "workflow": {
    "name": "Test Workflow",
    "steps": [
      {
        "id": "step01",
        "name": "Initialize",
        "description": "Initialize the system"
      },
      {
        "id": "step02", 
        "name": "Configure",
        "description": "Load configuration"
      },
      {
        "id": "step03",
        "name": "Process",
        "description": "Process data"
      },
      {
        "id": "step04",
        "name": "Report",
        "description": "Generate report"
      },
      {
        "id": "step05",
        "name": "Cleanup",
        "description": "Clean up resources"
      }
    ],
    "dependencies": {
      "step02": ["step01"],
      "step03": ["step01"],
      "step04": ["step02", "step03"],
      "step05": ["step04"]
    }
  }
}
'''
    
    # Create test files
    test_files = {
        'test_step1_script.py': test_py_content,
        'test_step2_documentation.md': test_md_content,
        'test_step3_config.json': test_json_content
    }
    
    print("Creating test files with step mentions...")
    
    for filename, content in test_files.items():
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Created: {filename}")
    
    print("\nTest files created successfully!")
    print("You can now run the step formatter to see how it works:")
    print("  python step_formatter.py --dry-run")
    print("  python step_formatter.py --backup")

def cleanup_test_files():
    """Remove test files created for testing."""
    test_files = [
        'test_step1_script.py',
        'test_step2_documentation.md', 
        'test_step3_config.json'
    ]
    
    print("Cleaning up test files...")
    
    for filename in test_files:
        if os.path.exists(filename):
            os.remove(filename)
            print(f"Removed: {filename}")
    
    print("Cleanup completed!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--cleanup":
        cleanup_test_files()
    else:
        create_test_files()
        print("\nTo clean up test files, run: python test_step_formatter.py --cleanup")