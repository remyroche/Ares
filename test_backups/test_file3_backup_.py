#!/usr/bin/env python3
"""
Test file 3 - Mixed content with some existing tprint
"""

from src.utils.tprint import tprint

def main():
    print("This should be converted to tprint")
    tprint("This is already tprint")
    print("Another print statement")
    
    # Print with complex expressions
    result = 42
    print("Result:", result, "Type:", type(result))
    
    tprint("End of test file 3")

if __name__ == "__main__":
    main()