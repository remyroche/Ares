#!/usr/bin/env python3
"""
Simple test file for comprehensive analysis testing.
"""


def simple_function(x: int) -> int:
    """A simple function that returns x + 1."""
    return x + 1

def complex_function(data: list[int]) -> dict[str, int]:
    """A more complex function with multiple operations."""
    if not data:
        return {}

    result = {
        "sum": sum(data),
        "count": len(data),
        "max": max(data),
        "min": min(data),
    }

    # Calculate average
    result["average"] = result["sum"] / result["count"]

    return result

class TestClass:
    """A simple test class."""

    def __init__(self, name: str):
        self.name = name
        self.values = []

    def add_value(self, value: int) -> None:
        """Add a value to the list."""
        self.values.append(value)

    def get_stats(self) -> dict[str, int]:
        """Get statistics about the values."""
        return complex_function(self.values)

def main():
    """Main function."""
    test_obj = TestClass("test")
    test_obj.add_value(10)
    test_obj.add_value(20)
    test_obj.add_value(30)

    stats = test_obj.get_stats()
    print(f"Stats: {stats}")

    # Test simple function
    result = simple_function(5)
    print(f"Simple function result: {result}")

if __name__ == "__main__":
    main()
