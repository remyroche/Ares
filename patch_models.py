import re

# However, sorting by timestamp overall means calibration is strictly forward-only in time.
# That ensures that the last `calib_n` rows are the MOST RECENT rows across all buckets.
# This answers "why is the calibrated trained only on short_tf regime rows?" -> Because they were concatenated without sorting!
# By adding the timestamp sort, `idx_cal` will now contain the most recent rows from ALL buckets, proportionally.
#
# Wait, "Verify that the models are trained & calibrated per-bucket."
# The prompt says this.
# Right now, one big EV model is trained for ALL buckets (with `bucket` as a feature or `regime` label).
# Wait, let's look at `train_position_sizer_models` inside `training_orchestrator.py` and `train_pwin_classifier` in `models.py`.
# In `models.py`:
# if calibration_mode == "regime" and regime_labels is not None:
#    regime_calibrators = {}
#    regs = np.asarray(regime_labels)
#    regs_cal = regs[idx_cal]
#    for reg in np.unique(regs_cal):
#        m = regs_cal == reg
#        ...
#        cal, meth = _fit_calibrator(...)
#        regime_calibrators[reg] = cal
#
# This means calibrators ARE trained per bucket, but only if they appear in `regs_cal` (which previously only contained `short_tf`).
# With `sort_values("timestamp")` they will all appear in `idx_cal` if recent data has all buckets!
# BUT what about the REGRESSORS (win/loss quantiles)?
# They are trained on `X` globally without `regime_labels`!
# Let's check `train_win_quantile_regressor` and `train_loss_quantile_regressor`.

with open('extreme_price_movements/position_sizer/training_orchestrator.py', 'r') as f:
    content = f.read()

print("train_win_quantile_regressor usage:")
lines = content.splitlines()
for i, line in enumerate(lines):
    if 'train_win_quantile_regressor' in line:
        for j in range(max(0, i-5), min(len(lines), i+15)):
            print(lines[j])
        print("----")
