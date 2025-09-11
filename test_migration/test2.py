#!/usr/bin/env python3

"""
Test script 2 with print statements.
"""

import os
import sys

def main():
    print("Starting test 2...")
    
    # Print with multiple arguments
    print("Debug info:", "variable1", 42, [1, 2, 3])
    
    # Print in loops
    for i in range(2):
        print(f"Loop iteration: {i}")
    
    print("Test 2 completed!")

if __name__ == "__main__":
    main()