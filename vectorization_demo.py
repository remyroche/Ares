#!/usr/bin/env python3
"""
Vectorization Optimization Demonstration

This script demonstrates the vectorization optimizations implemented in Ares.
"""

import numpy as np
import pandas as pd
import time

def print_colored(text):
    """Simple print function."""
    print(text)

def demo_vectorized_vs_loop():
    """Demonstrate vectorized vs loop performance."""
    print_colored("\n🚀 VECTORIZATION PERFORMANCE DEMONSTRATION")
    print_colored("="*60)

    # Create sample data
    np.random.seed(42)
    n_samples = 100000
    data = np.random.randn(n_samples, 3)

    print_colored(f"📊 Processing {n_samples} samples with 3 features")

    # Traditional loop approach
    print_colored("\n⏱️  TRADITIONAL LOOP APPROACH")
    start_time = time.time()

    result_loop = np.zeros(n_samples)
    for i in range(n_samples):
        result_loop[i] = np.sum(data[i] ** 2) + np.mean(data[i])

    loop_time = time.time() - start_time
    print_colored(".4f")

    # Vectorized approach
    print_colored("\n⚡ VECTORIZED APPROACH")
    start_time = time.time()

    result_vectorized = np.sum(data ** 2, axis=1) + np.mean(data, axis=1)

    vectorized_time = time.time() - start_time
    print_colored(".4f")

    # Calculate speedup
    speedup = loop_time / vectorized_time if vectorized_time > 0 else float('inf')
    print_colored(".1f")

    # Verify results are equivalent
    if np.allclose(result_loop, result_vectorized):
        print_colored("   • ✅ Results verification: PASSED")
    else:
        print_colored("   • ❌ Results verification: FAILED")

    return speedup

def demo_matrix_operations():
    """Demonstrate matrix operations."""
    print_colored("\n" + "="*60)
    print_colored("🧮 MATRIX OPERATIONS OPTIMIZATION")
    print_colored("="*60)

    # Create matrices
    size = 500
    A = np.random.randn(size, size)
    B = np.random.randn(size, size)

    print_colored(f"📊 Matrix multiplication: {size}x{size}")

    # Traditional approach
    start_time = time.time()
    C1 = A @ B
    traditional_time = time.time() - start_time

    # NumPy optimized approach
    start_time = time.time()
    C2 = np.dot(A, B)
    numpy_time = time.time() - start_time

    speedup = traditional_time / numpy_time if numpy_time > 0 else 1.0

    print_colored(".4f")
    print_colored(".4f")
    print_colored(".1f")

    return speedup

def demo_pandas_vectorization():
    """Demonstrate pandas vectorization."""
    print_colored("\n" + "="*60)
    print_colored("🐼 PANDAS VECTORIZATION")
    print_colored("="*60)

    # Create sample DataFrame
    np.random.seed(42)
    n_rows = 50000
    df = pd.DataFrame({
        'A': np.random.randn(n_rows),
        'B': np.random.randn(n_rows),
        'C': np.random.randn(n_rows)
    })

    print_colored(f"📊 Processing {n_rows} rows with 3 columns")

    # Traditional apply approach
    start_time = time.time()
    result_apply = df.apply(lambda row: row['A'] * row['B'] + row['C'], axis=1)
    apply_time = time.time() - start_time

    # Vectorized approach
    start_time = time.time()
    result_vectorized = df['A'] * df['B'] + df['C']
    vectorized_time = time.time() - start_time

    speedup = apply_time / vectorized_time if vectorized_time > 0 else float('inf')

    print_colored(".4f")
    print_colored(".4f")
    print_colored(".1f")

    return speedup

def main():
    """Run all demonstrations."""
    print_colored("🎯 ARES VECTORIZATION OPTIMIZATION DEMONSTRATION")
    print_colored("="*60)

    # Run demonstrations
    speedup1 = demo_vectorized_vs_loop()
    speedup2 = demo_matrix_operations()
    speedup3 = demo_pandas_vectorization()

    # Summary
    print_colored("\n" + "="*60)
    print_colored("🎉 OPTIMIZATION RESULTS")
    print_colored("="*60)

    print_colored("📊 Performance Improvements:")
    print_colored(".1f")
    print_colored(".1f")
    print_colored(".1f")

    avg_speedup = (speedup1 + speedup2 + speedup3) / 3
    print_colored(".1f")

    print_colored("\n🚀 Key Benefits of Vectorization:")
    print_colored("   • Eliminates Python loops for numerical operations")
    print_colored("   • Leverages optimized C/C++/Fortran libraries")
    print_colored("   • Reduces function call overhead")
    print_colored("   • Enables SIMD (Single Instruction, Multiple Data)")
    print_colored("   • Better memory access patterns")
    print_colored("   • Automatic parallelization where possible")

    print_colored("\n✨ Vectorization is a fundamental optimization technique!")
    print_colored("   Apply it to all numerical computations in your trading system.")

if __name__ == "__main__":
    main()
