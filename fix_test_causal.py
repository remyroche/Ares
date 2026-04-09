with open("tests/test_causal_surprise_robustness.py", "r") as f:
    content = f.read()

# Replace range index with DatetimeIndex in the setup or tests
content = content.replace('pd.Series(base_data)', 'pd.Series(base_data, index=pd.date_range("2020-01-01", periods=len(base_data)))')

with open("tests/test_causal_surprise_robustness.py", "w") as f:
    f.write(content)
