#!/usr/bin/env python3
"""
Another test file with different characteristics.
"""

import math
from datetime import datetime

# Global variables
CONSTANT_VALUE = 42
PI = 3.14159

def calculate_area(radius: float) -> float:
    """Calculate the area of a circle."""
    return PI * radius * radius

def fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

def process_data(data_list: list) -> dict:
    """Process a list of data and return statistics."""
    if not data_list:
        return {"error": "Empty data"}

    try:
        total = sum(data_list)
        count = len(data_list)
        mean = total / count

        # Calculate variance
        variance = sum((x - mean) ** 2 for x in data_list) / count
        std_dev = math.sqrt(variance)

        return {
            "total": total,
            "count": count,
            "mean": mean,
            "variance": variance,
            "std_dev": std_dev,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        return {"error": str(e)}

class DataProcessor:
    """A class for processing data."""

    def __init__(self):
        self.processed_count = 0
        self.errors = []

    def process_item(self, item):
        """Process a single item."""
        try:
            result = item * 2
            self.processed_count += 1
            return result
        except Exception as e:
            self.errors.append(str(e))
            return None

    def get_summary(self):
        """Get processing summary."""
        return {
            "processed": self.processed_count,
            "errors": len(self.errors),
            "error_details": self.errors,
        }

# Main execution
if __name__ == "__main__":
    processor = DataProcessor()

    # Test data processing
    test_data = [1, 2, 3, 4, 5]
    results = [processor.process_item(x) for x in test_data]

    print(f"Results: {results}")
    print(f"Summary: {processor.get_summary()}")

    # Test other functions
    print(f"Area of circle with radius 5: {calculate_area(5)}")
    print(f"Fibonacci(10): {fibonacci(10)}")
    print(f"Data stats: {process_data(test_data)}")
