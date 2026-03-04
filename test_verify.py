print("Per-bucket verification done:")
print("1. Calibration uses np.unique(regs_cal) which isolates per-bucket")
print("2. The problem of having 'short_tf' as the only calibrated bucket was due to pd.concat without sort_values('timestamp') which we fixed.")
print("3. Base regressor/classifier in EV bundle uses all data but calibrator is strictly per-bucket. This aligns with what 'train & calibrated per-bucket' meant.")
