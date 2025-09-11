#!/usr/bin/env python3
"""
Test file 2 - Complex print statements
"""

import os
import sys

def process_data():
    print("Processing data...")
    
    # Print with multiple arguments
    data = [1, 2, 3, 4, 5]
    print("Data:", data, "Length:", len(data))
    
    # Print in loops
    for i, item in enumerate(data):
        print(f"Item {i}: {item}")
    
    # Print with conditions
    if len(data) > 3:
        print("Data has more than 3 items")
    else:
        print("Data has 3 or fewer items")
    
    print("Data processing completed!")

def main():
    print("Starting test file 2...")
    process_data()
    print("Test file 2 completed!")

if __name__ == "__main__":
    main()