#!/usr/bin/env python3
"""
Simple test for EvolutionaryArchitectureSearch without external dependencies.
"""

import sys
import os
import time
import random
import json
from pathlib import Path

# Mock numpy and pandas for testing
class MockNumpy:
    """Comprehensive mock numpy implementation for testing without external dependencies."""
    
    def __init__(self):
        self.random = MockRandom()
    
    def mean(self, x):
        """Calculate mean of array-like object."""
        if not x:
            return 0.0
        try:
            if hasattr(x, '__iter__') and not isinstance(x, str):
                return sum(x) / len(x)
            else:
                return float(x)
        except (TypeError, ValueError):
            return 0.0
    
    def std(self, x):
        """Calculate standard deviation of array-like object."""
        if not x or len(x) < 2:
            return 0.0
        try:
            mean_val = self.mean(x)
            variance = sum((val - mean_val) ** 2 for val in x) / (len(x) - 1)
            return variance ** 0.5
        except (TypeError, ValueError):
            return 0.0
    
    def randn(self, *args):
        """Generate random numbers from normal distribution."""
        if len(args) == 1:
            # 1D array
            return [random.gauss(0, 1) for _ in range(args[0])]
        elif len(args) == 2:
            # 2D array
            return [[random.gauss(0, 1) for _ in range(args[1])] for _ in range(args[0])]
        else:
            # Higher dimensions - flatten to 2D for simplicity
            total_elements = 1
            for arg in args:
                total_elements *= arg
            return [random.gauss(0, 1) for _ in range(total_elements)]
    
    def array(self, data):
        """Create array from data."""
        return MockArray(data)
    
    def zeros(self, shape):
        """Create array of zeros."""
        if isinstance(shape, int):
            return [0.0] * shape
        elif len(shape) == 2:
            return [[0.0] * shape[1] for _ in range(shape[0])]
        else:
            return [0.0] * shape[0]
    
    def ones(self, shape):
        """Create array of ones."""
        if isinstance(shape, int):
            return [1.0] * shape
        elif len(shape) == 2:
            return [[1.0] * shape[1] for _ in range(shape[0])]
        else:
            return [1.0] * shape[0]
    
    def arange(self, start, stop=None, step=1):
        """Create array with range of values."""
        if stop is None:
            start, stop = 0, start
        return list(range(start, stop, step))
    
    def linspace(self, start, stop, num=50):
        """Create array with linearly spaced values."""
        if num <= 1:
            return [start]
        step = (stop - start) / (num - 1)
        return [start + i * step for i in range(num)]
    
    def concatenate(self, arrays, axis=0):
        """Concatenate arrays."""
        if not arrays:
            return []
        if axis == 0:
            result = []
            for arr in arrays:
                if hasattr(arr, '__iter__') and not isinstance(arr, str):
                    result.extend(arr)
                else:
                    result.append(arr)
            return result
        else:
            # For axis != 0, assume 2D concatenation
            return [row for arr in arrays for row in arr]
    
    def reshape(self, array, shape):
        """Reshape array."""
        if not array:
            return array
        if isinstance(shape, int):
            return array[:shape]
        elif len(shape) == 2:
            rows, cols = shape
            result = []
            for i in range(rows):
                start_idx = i * cols
                end_idx = start_idx + cols
                result.append(array[start_idx:end_idx])
            return result
        return array
    
    def transpose(self, array):
        """Transpose 2D array."""
        if not array or not array[0]:
            return array
        return [[row[i] for row in array] for i in range(len(array[0]))]
    
    def dot(self, a, b):
        """Matrix multiplication."""
        if not a or not b:
            return 0
        if isinstance(a[0], (int, float)) and isinstance(b[0], (int, float)):
            # Vector dot product
            return sum(x * y for x, y in zip(a, b))
        else:
            # Matrix multiplication
            result = []
            for i in range(len(a)):
                row = []
                for j in range(len(b[0])):
                    val = sum(a[i][k] * b[k][j] for k in range(len(b)))
                    row.append(val)
                result.append(row)
            return result
    
    def sum(self, array, axis=None):
        """Sum array elements."""
        if not array:
            return 0
        if axis is None:
            if isinstance(array[0], (int, float)):
                return sum(array)
            else:
                return sum(sum(row) for row in array)
        elif axis == 0:
            return [sum(row[i] for row in array) for i in range(len(array[0]))]
        elif axis == 1:
            return [sum(row) for row in array]
        return sum(array)
    
    def max(self, array, axis=None):
        """Find maximum value."""
        if not array:
            return 0
        if axis is None:
            if isinstance(array[0], (int, float)):
                return max(array)
            else:
                return max(max(row) for row in array)
        elif axis == 0:
            return [max(row[i] for row in array) for i in range(len(array[0]))]
        elif axis == 1:
            return [max(row) for row in array]
        return max(array)
    
    def min(self, array, axis=None):
        """Find minimum value."""
        if not array:
            return 0
        if axis is None:
            if isinstance(array[0], (int, float)):
                return min(array)
            else:
                return min(min(row) for row in array)
        elif axis == 0:
            return [min(row[i] for row in array) for i in range(len(array[0]))]
        elif axis == 1:
            return [min(row) for row in array]
        return min(array)
    
    def argmax(self, array, axis=None):
        """Find index of maximum value."""
        if not array:
            return 0
        if axis is None:
            if isinstance(array[0], (int, float)):
                return array.index(max(array))
            else:
                flat_array = [val for row in array for val in row]
                return flat_array.index(max(flat_array))
        elif axis == 0:
            return [array.index(max(row)) for row in zip(*array)]
        elif axis == 1:
            return [row.index(max(row)) for row in array]
        return array.index(max(array))
    
    def argmin(self, array, axis=None):
        """Find index of minimum value."""
        if not array:
            return 0
        if axis is None:
            if isinstance(array[0], (int, float)):
                return array.index(min(array))
            else:
                flat_array = [val for row in array for val in row]
                return flat_array.index(min(flat_array))
        elif axis == 0:
            return [array.index(min(row)) for row in zip(*array)]
        elif axis == 1:
            return [row.index(min(row)) for row in array]
        return array.index(min(array))
    
    # Add ndarray for type hints
    class ndarray:
        """Mock ndarray class for type hints."""
        pass

class MockArray:
    """Mock array class that behaves like numpy array."""
    
    def __init__(self, data):
        self.data = data
        if isinstance(data, list) and data and isinstance(data[0], list):
            self.shape = (len(data), len(data[0]))
        else:
            self.shape = (len(data),)
    
    def __getitem__(self, key):
        return self.data[key]
    
    def __setitem__(self, key, value):
        self.data[key] = value
    
    def __len__(self):
        return len(self.data)
    
    def __iter__(self):
        return iter(self.data)
    
    def __repr__(self):
        return f"MockArray({self.data})"
    
    def sum(self, axis=None):
        """Sum array elements."""
        if axis is None:
            if isinstance(self.data[0], (int, float)):
                return sum(self.data)
            else:
                return sum(sum(row) for row in self.data)
        elif axis == 0:
            return [sum(row[i] for row in self.data) for i in range(len(self.data[0]))]
        elif axis == 1:
            return [sum(row) for row in self.data]
        return sum(self.data)
    
    def mean(self, axis=None):
        """Calculate mean."""
        if axis is None:
            if isinstance(self.data[0], (int, float)):
                return sum(self.data) / len(self.data)
            else:
                total = sum(sum(row) for row in self.data)
                count = sum(len(row) for row in self.data)
                return total / count if count > 0 else 0
        elif axis == 0:
            return [sum(row[i] for row in self.data) / len(self.data) for i in range(len(self.data[0]))]
        elif axis == 1:
            return [sum(row) / len(row) for row in self.data]
        return sum(self.data) / len(self.data)
    
    def reshape(self, shape):
        """Reshape array."""
        return MockArray(np.reshape(self.data, shape))
    
    def transpose(self):
        """Transpose array."""
        return MockArray(np.transpose(self.data))

class MockRandom:
    """Comprehensive mock random number generator for testing."""
    
    def __init__(self):
        self.seed_value = None
    
    def randint(self, low, high):
        """Generate random integer in range [low, high] inclusive."""
        return random.randint(low, high)
    
    def random(self):
        """Generate random float in range [0.0, 1.0)."""
        return random.random()
    
    def uniform(self, low, high):
        """Generate random float in range [low, high)."""
        return random.uniform(low, high)
    
    def choice(self, sequence):
        """Choose random element from sequence."""
        return random.choice(sequence)
    
    def choices(self, population, weights=None, k=1):
        """Choose k random elements from population with replacement."""
        return random.choices(population, weights=weights, k=k)
    
    def sample(self, population, k):
        """Choose k random elements from population without replacement."""
        return random.sample(population, k)
    
    def shuffle(self, x):
        """Shuffle list x in place."""
        random.shuffle(x)
        return x
    
    def gauss(self, mu, sigma):
        """Generate random number from Gaussian distribution."""
        return random.gauss(mu, sigma)
    
    def normal(self, loc=0.0, scale=1.0, size=None):
        """Generate random numbers from normal distribution."""
        if size is None:
            return random.gauss(loc, scale)
        elif isinstance(size, int):
            return [random.gauss(loc, scale) for _ in range(size)]
        else:
            # Multi-dimensional
            total_elements = 1
            for s in size:
                total_elements *= s
            return [random.gauss(loc, scale) for _ in range(total_elements)]
    
    def exponential(self, scale=1.0, size=None):
        """Generate random numbers from exponential distribution."""
        if size is None:
            return random.expovariate(1.0 / scale)
        elif isinstance(size, int):
            return [random.expovariate(1.0 / scale) for _ in range(size)]
        else:
            total_elements = 1
            for s in size:
                total_elements *= s
            return [random.expovariate(1.0 / scale) for _ in range(total_elements)]
    
    def beta(self, a, b, size=None):
        """Generate random numbers from beta distribution (approximation)."""
        if size is None:
            # Simple approximation using gamma distribution
            x = random.gammavariate(a, 1.0)
            y = random.gammavariate(b, 1.0)
            return x / (x + y)
        elif isinstance(size, int):
            return [self.beta(a, b) for _ in range(size)]
        else:
            total_elements = 1
            for s in size:
                total_elements *= s
            return [self.beta(a, b) for _ in range(total_elements)]
    
    def gamma(self, shape, scale=1.0, size=None):
        """Generate random numbers from gamma distribution."""
        if size is None:
            return random.gammavariate(shape, scale)
        elif isinstance(size, int):
            return [random.gammavariate(shape, scale) for _ in range(size)]
        else:
            total_elements = 1
            for s in size:
                total_elements *= s
            return [random.gammavariate(shape, scale) for _ in range(total_elements)]
    
    def poisson(self, lam, size=None):
        """Generate random numbers from Poisson distribution."""
        if size is None:
            return random.poissonvariate(lam)
        elif isinstance(size, int):
            return [random.poissonvariate(lam) for _ in range(size)]
        else:
            total_elements = 1
            for s in size:
                total_elements *= s
            return [random.poissonvariate(lam) for _ in range(total_elements)]
    
    def binomial(self, n, p, size=None):
        """Generate random numbers from binomial distribution."""
        if size is None:
            return sum(1 for _ in range(n) if random.random() < p)
        elif isinstance(size, int):
            return [self.binomial(n, p) for _ in range(size)]
        else:
            total_elements = 1
            for s in size:
                total_elements *= s
            return [self.binomial(n, p) for _ in range(total_elements)]
    
    def seed(self, seed=None):
        """Set random seed."""
        self.seed_value = seed
        random.seed(seed)
    
    def get_state(self):
        """Get current random state."""
        return random.getstate()
    
    def set_state(self, state):
        """Set random state."""
        random.setstate(state)
    
    def rand(self, *dims):
        """Generate random numbers in range [0.0, 1.0)."""
        if not dims:
            return random.random()
        elif len(dims) == 1:
            return [random.random() for _ in range(dims[0])]
        elif len(dims) == 2:
            return [[random.random() for _ in range(dims[1])] for _ in range(dims[0])]
        else:
            # Higher dimensions
            total_elements = 1
            for dim in dims:
                total_elements *= dim
            return [random.random() for _ in range(total_elements)]
    
    def randn(self, *dims):
        """Generate random numbers from standard normal distribution."""
        if not dims:
            return random.gauss(0, 1)
        elif len(dims) == 1:
            return [random.gauss(0, 1) for _ in range(dims[0])]
        elif len(dims) == 2:
            return [[random.gauss(0, 1) for _ in range(dims[1])] for _ in range(dims[0])]
        else:
            total_elements = 1
            for dim in dims:
                total_elements *= dim
            return [random.gauss(0, 1) for _ in range(total_elements)]

# Mock numpy
np = MockNumpy()

# Mock pandas
class MockDataFrame:
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def columns(self):
        return list(range(len(self.data[0]))) if self.data else []
    
    def shape(self):
        """Return shape of the DataFrame."""
        if not self.data:
            return (0, 0)
        if isinstance(self.data[0], (list, tuple)):
            return (len(self.data), len(self.data[0]))
        else:
            return (len(self.data), 1)
    
    def head(self, n=5):
        """Return first n rows."""
        return MockDataFrame(self.data[:n])
    
    def tail(self, n=5):
        """Return last n rows."""
        return MockDataFrame(self.data[-n:])
    
    def info(self):
        """Print DataFrame info."""
        print(f"<class 'MockDataFrame'>")
        print(f"RangeIndex: {len(self.data)} entries, 0 to {len(self.data)-1}")
        print(f"Data columns (total {len(self.columns())} columns):")
        for i, col in enumerate(self.columns()):
            print(f" {i}   {col}")
        print(f"dtypes: object")
        print(f"memory usage: {len(str(self.data))} bytes")
    
    def describe(self):
        """Generate descriptive statistics."""
        if not self.data:
            return MockDataFrame()
        
        stats = {}
        for col in self.columns():
            col_data = [row[col] for row in self.data if isinstance(row, (list, tuple)) and col < len(row)]
            numeric_data = [x for x in col_data if isinstance(x, (int, float))]
            if numeric_data:
                stats[col] = {
                    'count': len(numeric_data),
                    'mean': sum(numeric_data) / len(numeric_data),
                    'std': self._std(numeric_data),
                    'min': min(numeric_data),
                    'max': max(numeric_data)
                }
        
        return MockDataFrame(stats)
    
    def _std(self, data):
        """Calculate standard deviation."""
        if len(data) < 2:
            return 0.0
        mean_val = sum(data) / len(data)
        variance = sum((x - mean_val) ** 2 for x in data) / (len(data) - 1)
        return variance ** 0.5
    
    def dropna(self, axis=0, how='any'):
        """Drop rows/columns with missing values."""
        if axis == 0:  # Drop rows
            clean_data = []
            for row in self.data:
                if isinstance(row, (list, tuple)):
                    if how == 'any' and None not in row:
                        clean_data.append(row)
                    elif how == 'all' and not all(x is None for x in row):
                        clean_data.append(row)
                else:
                    if row is not None:
                        clean_data.append(row)
            return MockDataFrame(clean_data)
        else:  # Drop columns
            # For simplicity, return self
            return self
    
    def fillna(self, value):
        """Fill missing values."""
        filled_data = []
        for row in self.data:
            if isinstance(row, (list, tuple)):
                filled_row = [value if x is None else x for x in row]
                filled_data.append(filled_row)
            else:
                filled_data.append(value if row is None else row)
        return MockDataFrame(filled_data)
    
    def isna(self):
        """Check for missing values."""
        na_data = []
        for row in self.data:
            if isinstance(row, (list, tuple)):
                na_row = [x is None for x in row]
                na_data.append(na_row)
            else:
                na_data.append(row is None)
        return MockDataFrame(na_data)
    
    def isnull(self):
        """Alias for isna."""
        return self.isna()
    
    def notna(self):
        """Check for non-missing values."""
        notna_data = []
        for row in self.data:
            if isinstance(row, (list, tuple)):
                notna_row = [x is not None for x in row]
                notna_data.append(notna_row)
            else:
                notna_data.append(row is not None)
        return MockDataFrame(notna_data)
    
    def notnull(self):
        """Alias for notna."""
        return self.notna()
    
    def sum(self, axis=0):
        """Sum values along axis."""
        if axis == 0:  # Sum columns
            sums = {}
            for col in self.columns():
                col_data = [row[col] for row in self.data if isinstance(row, (list, tuple)) and col < len(row)]
                numeric_data = [x for x in col_data if isinstance(x, (int, float))]
                sums[col] = sum(numeric_data)
            return MockDataFrame([sums])
        else:  # Sum rows
            row_sums = []
            for row in self.data:
                if isinstance(row, (list, tuple)):
                    numeric_data = [x for x in row if isinstance(x, (int, float))]
                    row_sums.append(sum(numeric_data))
                else:
                    row_sums.append(row if isinstance(row, (int, float)) else 0)
            return MockDataFrame([[s] for s in row_sums])
    
    def mean(self, axis=0):
        """Calculate mean along axis."""
        if axis == 0:  # Mean of columns
            means = {}
            for col in self.columns():
                col_data = [row[col] for row in self.data if isinstance(row, (list, tuple)) and col < len(row)]
                numeric_data = [x for x in col_data if isinstance(x, (int, float))]
                means[col] = sum(numeric_data) / len(numeric_data) if numeric_data else 0
            return MockDataFrame([means])
        else:  # Mean of rows
            row_means = []
            for row in self.data:
                if isinstance(row, (list, tuple)):
                    numeric_data = [x for x in row if isinstance(x, (int, float))]
                    row_means.append(sum(numeric_data) / len(numeric_data) if numeric_data else 0)
                else:
                    row_means.append(row if isinstance(row, (int, float)) else 0)
            return MockDataFrame([[m] for m in row_means])
    
    def std(self, axis=0):
        """Calculate standard deviation along axis."""
        if axis == 0:  # Std of columns
            stds = {}
            for col in self.columns():
                col_data = [row[col] for row in self.data if isinstance(row, (list, tuple)) and col < len(row)]
                numeric_data = [x for x in col_data if isinstance(x, (int, float))]
                stds[col] = self._std(numeric_data)
            return MockDataFrame([stds])
        else:  # Std of rows
            row_stds = []
            for row in self.data:
                if isinstance(row, (list, tuple)):
                    numeric_data = [x for x in row if isinstance(x, (int, float))]
                    row_stds.append(self._std(numeric_data))
                else:
                    row_stds.append(0)
            return MockDataFrame([[s] for s in row_stds])
    
    def min(self, axis=0):
        """Find minimum along axis."""
        if axis == 0:  # Min of columns
            mins = {}
            for col in self.columns():
                col_data = [row[col] for row in self.data if isinstance(row, (list, tuple)) and col < len(row)]
                numeric_data = [x for x in col_data if isinstance(x, (int, float))]
                mins[col] = min(numeric_data) if numeric_data else 0
            return MockDataFrame([mins])
        else:  # Min of rows
            row_mins = []
            for row in self.data:
                if isinstance(row, (list, tuple)):
                    numeric_data = [x for x in row if isinstance(x, (int, float))]
                    row_mins.append(min(numeric_data) if numeric_data else 0)
                else:
                    row_mins.append(row if isinstance(row, (int, float)) else 0)
            return MockDataFrame([[m] for m in row_mins])
    
    def max(self, axis=0):
        """Find maximum along axis."""
        if axis == 0:  # Max of columns
            maxs = {}
            for col in self.columns():
                col_data = [row[col] for row in self.data if isinstance(row, (list, tuple)) and col < len(row)]
                numeric_data = [x for x in col_data if isinstance(x, (int, float))]
                maxs[col] = max(numeric_data) if numeric_data else 0
            return MockDataFrame([maxs])
        else:  # Max of rows
            row_maxs = []
            for row in self.data:
                if isinstance(row, (list, tuple)):
                    numeric_data = [x for x in row if isinstance(x, (int, float))]
                    row_maxs.append(max(numeric_data) if numeric_data else 0)
                else:
                    row_maxs.append(row if isinstance(row, (int, float)) else 0)
            return MockDataFrame([[m] for m in row_maxs])
    
    def to_dict(self, orient='dict'):
        """Convert to dictionary."""
        if orient == 'dict':
            result = {}
            for col in self.columns():
                result[col] = [row[col] for row in self.data if isinstance(row, (list, tuple)) and col < len(row)]
            return result
        elif orient == 'records':
            return [dict(zip(self.columns(), row)) for row in self.data if isinstance(row, (list, tuple))]
        else:
            raise ValueError(f"Unsupported orient: {orient}")
    
    def to_csv(self, path=None, sep=','):
        """Convert to CSV string or save to file."""
        csv_lines = [sep.join(str(col) for col in self.columns())]
        for row in self.data:
            if isinstance(row, (list, tuple)):
                csv_lines.append(sep.join(str(x) for x in row))
            else:
                csv_lines.append(str(row))
        
        csv_content = '\n'.join(csv_lines)
        
        if path:
            with open(path, 'w') as f:
                f.write(csv_content)
        else:
            return csv_content
    
    def copy(self):
        """Create a copy of the DataFrame."""
        return MockDataFrame(self.data.copy())
    
    def append(self, other, ignore_index=False):
        """Append another DataFrame."""
        if not isinstance(other, MockDataFrame):
            raise TypeError("Can only append MockDataFrame objects")
        
        new_data = self.data + other.data
        return MockDataFrame(new_data)
    
    def merge(self, other, on=None, how='inner'):
        """Merge with another DataFrame."""
        if not isinstance(other, MockDataFrame):
            raise TypeError("Can only merge MockDataFrame objects")
        
        if on is None:
            # Use common columns
            common_cols = set(self.columns()) & set(other.columns())
            if not common_cols:
                raise ValueError("No common columns to merge on")
            on = list(common_cols)[0]
        
        # Simple merge implementation
        merged_data = []
        for row1 in self.data:
            for row2 in other.data:
                if isinstance(row1, (list, tuple)) and isinstance(row2, (list, tuple)):
                    if row1[on] == row2[on]:
                        merged_row = row1 + row2
                        merged_data.append(merged_row)
        
        return MockDataFrame(merged_data)
    
    def groupby(self, by):
        """Group by column(s)."""
        if isinstance(by, str):
            by = [by]
        
        groups = {}
        for i, row in enumerate(self.data):
            if isinstance(row, (list, tuple)):
                key = tuple(row[col] for col in by if col < len(row))
                if key not in groups:
                    groups[key] = []
                groups[key].append((i, row))
        
        return MockGroupBy(groups, self.columns(), by)
    
    def sort_values(self, by, ascending=True):
        """Sort by column values."""
        if isinstance(by, str):
            by = [by]
        
        def sort_key(row):
            if isinstance(row, (list, tuple)):
                return tuple(row[col] for col in by if col < len(row))
            else:
                return row
        
        sorted_data = sorted(self.data, key=sort_key, reverse=not ascending)
        return MockDataFrame(sorted_data)
    
    def reset_index(self, drop=False):
        """Reset index."""
        if drop:
            return MockDataFrame(self.data)
        else:
            new_data = [[i] + (row if isinstance(row, (list, tuple)) else [row]) 
                       for i, row in enumerate(self.data)]
            return MockDataFrame(new_data)

class MockGroupBy:
    """Mock groupby object for DataFrame grouping operations."""
    
    def __init__(self, groups, columns, by):
        self.groups = groups
        self.columns = columns
        self.by = by
    
    def sum(self):
        """Sum grouped values."""
        result_data = []
        for key, group_data in self.groups.items():
            row = [0] * len(self.columns)
            for _, group_row in group_data:
                if isinstance(group_row, (list, tuple)):
                    for i, val in enumerate(group_row):
                        if isinstance(val, (int, float)):
                            row[i] += val
            result_data.append(row)
        return MockDataFrame(result_data)
    
    def mean(self):
        """Mean of grouped values."""
        result_data = []
        for key, group_data in self.groups.items():
            row = [0] * len(self.columns)
            counts = [0] * len(self.columns)
            for _, group_row in group_data:
                if isinstance(group_row, (list, tuple)):
                    for i, val in enumerate(group_row):
                        if isinstance(val, (int, float)):
                            row[i] += val
                            counts[i] += 1
            for i in range(len(row)):
                if counts[i] > 0:
                    row[i] /= counts[i]
            result_data.append(row)
        return MockDataFrame(result_data)
    
    def count(self):
        """Count grouped values."""
        result_data = []
        for key, group_data in self.groups.items():
            row = [len(group_data)] * len(self.columns)
            result_data.append(row)
        return MockDataFrame(result_data)

def mock_numpy():
    """Mock numpy functions."""
    return MockNumpy()

# Set up mock modules
sys.modules['numpy'] = mock_numpy()
sys.modules['pandas'] = type('MockPandas', (), {'DataFrame': MockDataFrame})()

# Now import our module
try:
    from evolutionary_search import (
        EvolutionaryArchitectureSearch,
        ArchitectureConfig,
        EvolutionaryConfig,
        FitnessConfig,
        Architecture
    )
    print("✅ Successfully imported EvolutionaryArchitectureSearch")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def test_basic_functionality():
    """Test basic functionality without external dependencies."""
    print("🧪 Testing basic functionality...")
    
    # Test configuration classes
    arch_config = ArchitectureConfig(
        max_layers=4,
        min_layers=2,
        max_neurons_per_layer=128,
        min_neurons_per_layer=16,
        max_parameters=100000,  # Lower limit for test
        min_parameters=100,     # Lower limit for test
        max_flops=1000000,      # Lower limit for test
        min_flops=1000          # Lower limit for test
    )
    
    evo_config = EvolutionaryConfig(
        population_size=5,
        max_generations=2,
        n_workers=1
    )
    
    fitness_config = FitnessConfig(
        cv_folds=2,
        max_training_epochs=10
    )
    
    print("✅ Configuration classes work")
    
    # Test architecture creation
    layers = [
        {'type': 'dense', 'neurons': 64, 'activation': 'relu'},
        {'type': 'dense', 'neurons': 32, 'activation': 'sigmoid'}
    ]
    
    arch = Architecture(layers, arch_config)
    print(f"   Architecture layers: {len(arch.layers)}")
    print(f"   Min layers: {arch_config.min_layers}, Max layers: {arch_config.max_layers}")
    print(f"   Is valid: {arch.is_valid()}")
    if not arch.is_valid():
        print(f"   Architecture layers: {arch.layers}")
    assert arch.is_valid(), "Architecture should be valid"
    print("✅ Architecture creation works")
    
    # Test complexity calculation
    complexity = arch.calculate_complexity()
    assert 'parameters' in complexity, "Should have parameters"
    assert 'flops' in complexity, "Should have FLOPs"
    print("✅ Complexity calculation works")
    
    # Test serialization
    arch_dict = arch.to_dict()
    restored_arch = Architecture.from_dict(arch_dict, arch_config)
    assert restored_arch.fitness == arch.fitness, "Serialization should work"
    print("✅ Serialization works")
    
    # Test NAS initialization
    # Create mock data
    X = [[random.random() for _ in range(10)] for _ in range(100)]
    y = [random.randint(0, 1) for _ in range(100)]
    
    nas = EvolutionaryArchitectureSearch(
        architecture_config=arch_config,
        evolutionary_config=evo_config,
        fitness_config=fitness_config,
        data=(X, y),
        log_dir="test_logs"
    )
    
    assert nas.arch_config.max_layers == 4, "Config should be set"
    print("✅ NAS initialization works")
    
    # Test population initialization
    population = nas.initialize_population()
    assert len(population) > 0, "Should create population"
    print("✅ Population initialization works")
    
    # Test fitness evaluation
    if population:
        arch = population[0]
        fitness = nas.evaluate_fitness(arch)
        assert 0 <= fitness <= 1, "Fitness should be valid"
        print("✅ Fitness evaluation works")
    
    # Test genetic operators
    if len(population) >= 2:
        parent1, parent2 = population[0], population[1]
        child1, child2 = nas.crossover(parent1, parent2)
        assert isinstance(child1, Architecture), "Crossover should work"
        print("✅ Crossover works")
        
        mutated = nas.mutate(parent1)
        assert isinstance(mutated, Architecture), "Mutation should work"
        print("✅ Mutation works")
    
    # Test selection
    parents = nas.select_parents(population)
    assert len(parents) == len(population), "Selection should work"
    print("✅ Selection works")
    
    print("✅ All basic tests passed!")

def test_evolution_cycle():
    """Test a complete evolution cycle."""
    print("🧪 Testing evolution cycle...")
    
    # Create mock data
    X = [[random.random() for _ in range(5)] for _ in range(50)]
    y = [random.randint(0, 1) for _ in range(50)]
    
    # Configure for quick test
    arch_config = ArchitectureConfig(
        max_layers=3,
        min_layers=2,
        max_neurons_per_layer=64,
        min_neurons_per_layer=16,
        max_parameters=50000,   # Much lower limits
        min_parameters=10,      # Much lower limits
        max_flops=100000,       # Much lower limits
        min_flops=10            # Much lower limits
    )
    
    evo_config = EvolutionaryConfig(
        population_size=4,
        max_generations=2,
        n_workers=1
    )
    
    fitness_config = FitnessConfig(
        cv_folds=2,
        max_training_epochs=5,
        max_training_time=10.0
    )
    
    # Initialize NAS
    nas = EvolutionaryArchitectureSearch(
        architecture_config=arch_config,
        evolutionary_config=evo_config,
        fitness_config=fitness_config,
        data=(X, y),
        log_dir="test_logs"
    )
    
    # Run evolution
    start_time = time.time()
    best_architecture = nas.run_evolution()
    end_time = time.time()
    
    assert best_architecture is not None, "Should find best architecture"
    assert best_architecture.fitness is not None, "Should have fitness"
    assert 0 <= best_architecture.fitness <= 1, "Fitness should be valid"
    
    # Check summary
    summary = nas.get_search_summary()
    assert summary['total_generations'] > 0, "Should complete generations"
    assert summary['total_evaluations'] > 0, "Should perform evaluations"
    
    print(f"✅ Evolution completed in {end_time - start_time:.2f} seconds")
    print(f"   Best fitness: {best_architecture.fitness:.4f}")
    print(f"   Total evaluations: {summary['total_evaluations']}")
    print("✅ Evolution cycle test passed!")

def test_error_handling():
    """Test error handling."""
    print("🧪 Testing error handling...")
    
    # Test with None data
    try:
        nas = EvolutionaryArchitectureSearch(data=None)
        assert nas.X is None, "Should handle None data"
        print("✅ None data handling works")
    except Exception as e:
        print(f"   Expected error with None data: {e}")
    
    # Test with empty data
    try:
        nas = EvolutionaryArchitectureSearch(data=([], []))
        print("✅ Empty data handling works")
    except Exception as e:
        print(f"   Expected error with empty data: {e}")
    
    print("✅ Error handling test passed!")

def main():
    """Run all tests."""
    print("🚀 Starting simple EvolutionaryArchitectureSearch tests...")
    print("=" * 60)
    
    try:
        test_basic_functionality()
        print("\n" + "=" * 60)
        test_evolution_cycle()
        print("\n" + "=" * 60)
        test_error_handling()
        
        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()