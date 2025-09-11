#!/usr/bin/env python3
"""
Example script with print statements to demonstrate migration.
"""

import os
import sys

def main():
    print("Starting application...")
    
    # Simple print statements
    print("Hello, world!")
    print("This is a test script")
    
    # Print with variables
    name = "Alice"
    age = 30
    print(f"Name: {name}, Age: {age}")
    
    # Print with multiple arguments
    print("Debug info:", "variable1", 42, [1, 2, 3])
    
    # Print in loops
    for i in range(3):
        print(f"Loop iteration: {i}")
    
    # Print with conditions
    if True:
        print("Condition is true")
    else:
        print("Condition is false")
    
    # Print in functions
    def helper_function():
        print("Inside helper function")
        return "result"
    
    result = helper_function()
    print("Function result:", result)
    
    print("Application completed!")

if __name__ == "__main__":
    main()