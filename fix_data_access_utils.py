#!/usr/bin/env python3
"""
Fix syntax issues in data_access_utils.py
"""

import re

def fix_data_access_utils():
    """Fix syntax issues in data_access_utils.py"""
    
    with open("src/training/data_access_utils.py", 'r') as f:
        content = f.read()
    
    # Fix all the comma assignment issues
    fixes = [
        # Function signatures
        (r'= exchange: str = "BINANCE"', ', exchange: str = "BINANCE"'),
        (r'= None = \)', ', None)'),
        (r'= \*\*kwargs = \)', ', **kwargs)'),
        (r'= \*\*kwargs = \)\s*->\s*pd\.DataFrame:', ', **kwargs) -> pd.DataFrame:'),
        
        # Return types
        (r'->\s*tuple\[pd\.DataFrame = pd\.Series\]:', '-> tuple[pd.DataFrame, pd.Series]:'),
        (r'->\s*tuple\[np\.ndarray = np\.ndarray\]:', '-> tuple[np.ndarray, np.ndarray]:'),
        (r'->\s*dict\[str = Any\]:', '-> dict[str, Any]:'),
        
        # Function calls
        (r'get_data_manager\(data_dir = symbol', 'get_data_manager(data_dir, symbol'),
        (r'get_data_manager\(data_dir = symbol = exchange', 'get_data_manager(data_dir, symbol, exchange'),
        (r'get_data_manager\(data_dir = symbol = exchange\)', 'get_data_manager(data_dir, symbol, exchange)'),
        
        # Variable assignments
        (r'data_dir = symbol', 'data_dir, symbol'),
        (r'data_dir = symbol = exchange', 'data_dir, symbol, exchange'),
        (r'X_val = y_val = load_training_data', 'X_val, y_val = load_training_data'),
        (r'X_val = y_val = load_training_data\(', 'X_val, y_val = load_training_data('),
        (r'X_val_np = X_val\.fillna\(0\)\.values', 'X_val_np = X_val.fillna(0).values'),
        (r'y_val_np = y_val\.fillna\(0\)\.astype\(int\)\.values', 'y_val_np = y_val.fillna(0).astype(int).values'),
        (r'y_val_np = np\.clip\(y_val_np = -1 = 1\)', 'y_val_np = np.clip(y_val_np, -1, 1)'),
        (r'np\.unique\(y_val_np = return_counts=True\)', 'np.unique(y_val_np, return_counts=True)'),
        (r'return X_val_np = y_val_np', 'return X_val_np, y_val_np'),
        
        # Other assignments
        (r'split_type = "train" = \*\*kwargs', 'split_type="train", **kwargs'),
        (r'split_type = "validation", \*\*kwargs', 'split_type="validation", **kwargs'),
        (r'split_type = "test" = \*\*kwargs', 'split_type="test", **kwargs'),
        (r'updated_data: pd\.DataFrame = split_type', 'updated_data: pd.DataFrame, split_type'),
        (r'data_manager\.update_data_split\(split_type, updated_data\)', 'data_manager.update_data_split(split_type, updated_data)'),
        (r'return metadata\.get\("splits", \{\}\)', 'return metadata.get("splits", {})'),
        
        # Constructor calls
        (r'data_dir=data_dir = symbol=symbol', 'data_dir=data_dir, symbol=symbol'),
        (r'lookback_days=lookback_days or 730 = \)', 'lookback_days=lookback_days or 730)'),
        
        # Function calls with parameters
        (r'get_features_and_labels\(split_type = label_column\)', 'get_features_and_labels(split_type, label_column)'),
        (r'load_training_data\(data_dir, symbol = exchange', 'load_training_data(data_dir, symbol, exchange'),
        (r'load_training_data\(data_dir, split_type="train" = \*\*kwargs\)', 'load_training_data(data_dir, split_type="train", **kwargs)'),
        (r'load_training_data\(data_dir, split_type="validation", \*\*kwargs\)', 'load_training_data(data_dir, split_type="validation", **kwargs)'),
        (r'load_training_data\(data_dir, split_type="test" = \*\*kwargs\)', 'load_training_data(data_dir, split_type="test", **kwargs)'),
    ]
    
    for pattern, replacement in fixes:
        content = re.sub(pattern, replacement, content)
    
    # Write back to file
    with open("src/training/data_access_utils.py", 'w') as f:
        f.write(content)
    
    print("✅ Fixed syntax issues in data_access_utils.py")

if __name__ == "__main__":
    fix_data_access_utils()