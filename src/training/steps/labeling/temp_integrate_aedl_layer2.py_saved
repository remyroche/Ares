import re

# Read the file
with open('label_based_layer_2.py', 'r') as f:
    content = f.read()

# Add AEDL imports
import_pattern = r'(from \.causal_uncertainty_quantification import BayesianCausalDiscovery, quick_bayesian_causal_discovery)'
import_replacement = r'''from .causal_uncertainty_quantification import BayesianCausalDiscovery, quick_bayesian_causal_discovery
from .adaptive_event_driven_labeling import AdaptiveEventDrivenLabeling
from .spectral_chaser import SpectralChaser'''

# Apply the replacement
content = re.sub(import_pattern, import_replacement, content)

# Add AEDL configuration to __init__
config_pattern = r'(        # Causal Discovery Parameters\s+self\.significance_level = kwargs\.get\("significance_level", 0\.05)\s+self\.max_conditioning_set = kwargs\.get\("max_conditioning_set", 3)\s+self\.use_lingam = kwargs\.get\("use_lingam", True)\s+self\.use_bayesian_discovery = kwargs\.get\("use_bayesian_discovery", True)\s+self\.bayesian_n_bootstrap = kwargs\.get\("bayesian_n_bootstrap", 50))'

config_replacement = r'''        # Causal Discovery Parameters
        self.significance_level = kwargs.get("significance_level", 0.05)
        self.max_conditioning_set = kwargs.get("max_conditioning_set", 3)
        self.use_lingam = kwargs.get("use_lingam", True)
        self.use_bayesian_discovery = kwargs.get("use_bayesian_discovery", True)
        self.bayesian_n_bootstrap = kwargs.get("bayesian_n_bootstrap", 50)
        
        # AEDL Framework Parameters
        self.enable_aedl = kwargs.get("enable_aedl", True)
        self.aedl_spectral_vision = kwargs.get("aedl_spectral_vision", True)
        self.aedl_causal_compression = kwargs.get("aedl_causal_compression", True)
        self.aedl_resonance_detection = kwargs.get("aedl_resonance_detection", True)'''

# Apply the replacement
content = re.sub(config_pattern, config_replacement, content, flags=re.MULTILINE | re.DOTALL)

# Write back to file
with open('label_based_layer_2.py', 'w') as f:
    f.write(content)

print("Added AEDL imports and configuration to label_based_layer_2.py")
