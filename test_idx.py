import numpy as np

n = 1000
calibration_frac = 0.20
calibration_min_samples = 200

calib_n = int(max(calibration_min_samples, round(float(calibration_frac) * n)))
calib_n = int(min(calib_n, max(50, n - 50)))
train_end = int(max(50, n - calib_n))

idx_tr = np.arange(0, train_end, dtype=int)
idx_cal = np.arange(train_end, n, dtype=int)

print("idx_tr:", len(idx_tr))
print("idx_cal:", len(idx_cal))
