import re

with open('extreme_price_movements/position_sizer/models.py', 'r') as f:
    content = f.read()

# Let's add per-bucket training to train_win_quantile_regressor and train_loss_quantile_regressor if they need it?
# Wait, they are regressors. Regressors are not 'calibrated' in the sense of probability calibrators (which use Platt/Isotonic scaling per bucket).
# The prompt says: "why is the calibrated trained only on short_tf regime rows? And add detailed metrics analysis for the models training. ... Verify that the models are trained & calibrated per-bucket."
# The calibrator WAS trained only on `short_tf` due to the concatenation order. Sorting by timestamp fixed it so that it will be calibrated per-bucket (since `regime_calibrators` loops over all buckets in `regs_cal`, and sorting ensures all buckets appear in the calibration window).
