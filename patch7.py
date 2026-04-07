import re

with open("extreme_price_movements/simple_position_sizer.py", "r") as f:
    content = f.read()

# Update the calls to clean_and_standardize to use the new keyword arguments
old_call_train = "X_tr_clean, medians, scaler, mean_1d, std_1d = clean_and_standardize(X_tr)"
new_call_train = "X_tr_clean, medians, scaler, center_1d, scale_1d = clean_and_standardize(X_tr)"
content = content.replace(old_call_train, new_call_train)

old_call_test = "X_te_clean, _, _, _, _ = clean_and_standardize(X_te, fit_medians=medians, scaler=scaler, mean_1d=mean_1d, std_1d=std_1d)"
new_call_test = "X_te_clean, _, _, _, _ = clean_and_standardize(X_te, fit_medians=medians, scaler=scaler, center_1d=center_1d, scale_1d=scale_1d)"
content = content.replace(old_call_test, new_call_test)

with open("extreme_price_movements/simple_position_sizer.py", "w") as f:
    f.write(content)
