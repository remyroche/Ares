import numpy as np

def calculate_entropy_statistics_numba(values, window):
    # Quick implementation to see what the test expects vs what is produced
    from src.utils.entropy_optimized import calculate_entropy_statistics_numba
    return calculate_entropy_statistics_numba(values, window)

np.random.seed(42)
N = 1000
window = 10
values = np.random.random(N)

expected_ma = np.full(N, np.nan)
expected_std = np.full(N, np.nan)
expected_zscore = np.full(N, np.nan)

for i in range(window - 1, N):
    window_data = values[i - window + 1:i + 1]
    ma = np.mean(window_data)
    std = np.std(window_data)
    expected_ma[i] = ma
    expected_std[i] = std
    if std > 0:
        expected_zscore[i] = (values[i] - ma) / std
    else:
        expected_zscore[i] = 0.0

# Run optimized function
ma, std, zscore = calculate_entropy_statistics_numba(values, window)

valid_mask = np.isfinite(expected_zscore) & np.isfinite(zscore)
diffs = np.abs(zscore[valid_mask] - expected_zscore[valid_mask])
print(f"Max abs diff: {np.max(diffs)}")
