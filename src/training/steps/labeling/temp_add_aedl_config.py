import re

# Read the file
with open('label_based_layer_2.py', 'r') as f:
    content = f.read()

# Find the __init__ method and add AEDL configuration
init_pattern = r'(        self\.transaction_cost = float\(transaction_cost\)\s+\s+        self\.n_trials = kwargs\.get\("n_trials", 60)\s+\s+self\.n_splits = kwargs\.get\("n_splits", 2)\s+\s+self\.random_state = kwargs\.get\("random_state", 42)\s+\s+self\.verbose = kwargs\.get\("verbose", True)\s+\s+self\.force_hpo = kwargs\.get\("force_hpo", False))'

# Enhanced replacement with AEDL configuration
init_replacement = r'''        self.transaction_cost = float(transaction_cost)
        self.n_trials = kwargs.get("n_trials", 60)
        self.n_splits = kwargs.get("n_splits", 2)
        self.random_state = kwargs.get("random_state", 42)
        self.verbose = kwargs.get("verbose", True)
        self.force_hpo = kwargs.get("force_hpo", False)
        
        # AEDL Framework Parameters
        self.enable_aedl = kwargs.get("enable_aedl", True)
        self.aedl_spectral_vision = kwargs.get("aedl_spectral_vision", True)
        self.aedl_causal_compression = kwargs.get("aedl_causal_compression", True)
        self.aedl_resonance_detection = kwargs.get("aedl_resonance_detection", True)
        
        # Spectral Chaser Parameters
        self.spectral_chaser_enabled = kwargs.get("spectral_chaser_enabled", True)
        self.spectral_chaser_models = kwargs.get("spectral_chaser_models", ['xgb', 'catboost', 'rf', 'linear'])
        self.spectral_chaser_cv_folds = kwargs.get("spectral_chaser_cv_folds", 5)
        
        # RSV Integration Parameters
        self.rsv_integration_enabled = kwargs.get("rsv_integration_enabled", True)
        self.rsv_position_sizing = kwargs.get("rsv_position_sizing", True)
        self.rsv_regime_aware = kwargs.get("rsv_regime_aware", True)'''

# Apply the replacement
content = re.sub(init_pattern, init_replacement, content)

# Write back to file
with open('label_based_layer_2.py', 'w') as f:
    f.write(content)

print("Added AEDL configuration to LabelBasedLayer2.__init__")
