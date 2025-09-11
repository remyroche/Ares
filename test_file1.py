#!/usr/bin/env python3
from src.utils.tprint import tprint

"""
Test file 1 - Simple print statements
"""

def main():
    tprint("Starting test file 1...")
    tprint("Hello from test file 1!")
    
    # Print with variables
    name = "Test User"
    count = 5
    tprint(f"Name: {name}, Count: {count}")
    
    tprint("Test file 1 completed!")

if __name__ == "__main__":
    main()