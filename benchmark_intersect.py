
import numpy as np
import time
from numba import njit
from scipy import sparse

@njit
def numba_intersect_size(a, b):
    # a and b must be sorted
    i = 0
    j = 0
    count = 0
    na = len(a)
    nb = len(b)
    while i < na and j < nb:
        if a[i] < b[j]:
            i += 1
        elif a[i] > b[j]:
            j += 1
        else:
            count += 1
            i += 1
            j += 1
    return count

def test_benchmark():
    # Create random sorted arrays
    N = 2000
    a = np.sort(np.random.choice(10000, 500, replace=False))
    b = np.sort(np.random.choice(10000, 500, replace=False))

    # Warmup
    numba_intersect_size(a, b)

    t0 = time.time()
    for _ in range(1000):
        len(np.intersect1d(a, b, assume_unique=True))
    t1 = time.time()
    print(f"numpy intersect1d: {t1-t0:.4f}s")

    t0 = time.time()
    for _ in range(1000):
        numba_intersect_size(a, b)
    t1 = time.time()
    print(f"numba intersect:   {t1-t0:.4f}s")

if __name__ == "__main__":
    test_benchmark()
