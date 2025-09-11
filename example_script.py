#!/usr/bin/env python3
from src.utils.tprint import tprint

"""
Example script with print statements to demonstrate migration.
"""

import os
import sys

def main():
    tprint("Starting application...")
    
    # Simple print statements
    tprint("Hello, world!")
    tprint("This is a test script")
    
    # Print with variables
    name = "Alice"
    age = 30
    tprint(f"Name: {name}, Age: {age}")
    
    # Print with multiple arguments
    tprint("Debug info:", "variable1", 42, [1, 2, 3])
    
    # Print in loops
    for i in range(3):
        tprint(f"Loop iteration: {i}")
    
    # Print with conditions
    if True:
        tprint("Condition is true")
    else:
        tprint("Condition is false")
    
    # Print in functions
    def helper_function():
        tprint("Inside helper function")
        return "result"
    
    result = helper_function()
    tprint("Function result:", result)
    
    tprint("Application completed!")

if __name__ == "__main__":
    main()