import re

# Read the file
with open('label_based_layer_2.py', 'r') as f:
    content = f.read()

# Add AEDL configuration to __init__ method
init_pattern = r'(        # Causal Discovery Parameters\s+self\.significance_level = kwargs\.get\("significance_level", 0\.05)\s+self\.max_conditioning_set = kwargs\.get\("max_conditioning_set", 3)\s+self\.use_lingam = kwargs\.get\("use_lingam", True)\s+self\.use_bayesian_discovery = kwargs\.get\("use_bayesian_discovery", True)\s+self\.bayesian_n_bootstrap = kwargs\.get\("bayesian_n_bootstrap", 50)\s+self\.use_bayesian_discovery = kwargs\.get\("use_bayesian_discovery", True)\s+self\.bayesian_n_bootstrap = kwargs\.get\("bayesian_n_bootstrap", 50))'

init_replacement = r'''        # Causal Discovery Parameters
        self.significance_level = kwargs.get("significance_level", 0.05)
        self.max_conditioning_set = kwargs.get("max_conditioning_set", 3)
        self.use_lingam = kwargs.get("use_lingam", True)
        self.use_bayesian_discovery = kwargs.get("use_bayesian_discovery", True)
        self.bayesian_n_bootstrap = kwargs.get("bayesian_n_bootstrap", 50)
        
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
content = re.sub(init_pattern, init_replacement, content, flags=re.MULTILINE | re.DOTALL)

# Write back to file
with open('label_based_layer_2.py', 'w') as f:
    f.write(content)

print("Enhanced Layer 2 __init__ with AEDL configuration")
