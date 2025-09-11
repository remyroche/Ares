#!/usr/bin/env python3
from src.utils.tprint import tprint

"""
Test file 2 - Complex print statements
"""

import os
import sys

def process_data():
    tprint("Processing data...")
    
    # Print with multiple arguments
    data = [1, 2, 3, 4, 5]
    tprint("Data:", data, "Length:", len(data))
    
    # Print in loops
    for i, item in enumerate(data):
        tprint(f"Item {i}: {item}")
    
    # Print with conditions
    if len(data) > 3:
        tprint("Data has more than 3 items")
    else:
        tprint("Data has 3 or fewer items")
    
    tprint("Data processing completed!")

def main():
    tprint("Starting test file 2...")
    process_data()
    tprint("Test file 2 completed!")

if __name__ == "__main__":
    main()