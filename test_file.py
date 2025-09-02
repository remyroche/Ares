#!/usr/bin/env python3
"""
Test file for code quality tools.
"""

import os
import sys
from pathlib import Path


def badly_formatted_function(x, y, z):
    """This function has bad formatting."""
    if x > 0:
        print("x is positive")
    else:
        print("x is not positive")

    return x + y + z


class BadlyFormattedClass:
    def __init__(self, name):
        self.name = name

    def get_name(self):
        return self.name


if __name__ == "__main__":
    obj = BadlyFormattedClass("test")
    result = badly_formatted_function(1, 2, 3)
    print(f"Result: {result}")
