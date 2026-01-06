import re

# Read the file
with open('comprehensive_de_prado_config.py', 'r') as f:
    content = f.read()

# Find the ENHANCED_CONFIG dictionary and add AEDL options
pattern = r'(    # =============================================================================
    # COMPREHENSIVE METRICS
    =============================================================================
    'comprehensive_metrics_enabled': True,  # Enable comprehensive metrics reporting
    'layer2_metrics_enabled': True,        # Layer 2 discovery and engineering metrics
    'layer25_metrics_enabled': True,       # Layer 2.5 Chaser metrics
    'layer3_metrics_enabled': True,        # Layer 3 meta-learner metrics)'

# Enhanced configuration with AEDL options
replacement = r'''    # =============================================================================
    # AEDL FRAMEWORK (NEW)
    =============================================================================
    'enable_aedl': True,                    # Enable Adaptive Event-Driven Labeling
    'aedl_spectral_vision': True,           # Enable spectral vision for Chaser
    'aedl_causal_compression': True,        # Enable causal compression (20→4 features)
    'aedl_resonance_detection': True,        # Enable cross-scale resonance detection
    
    # Wavelet Parameters
    'wavelet_family': 'db4',                # Wavelet family for decomposition
    'wavelet_levels': 5,                    # Number of decomposition levels
    'wavelet_scales': ['d1', 'd2', 'd3', 'd4', 's4'],  # Scale names
    
    # Spectral Specialists
    'priority_specialists': [
        'inventory_specialist',              # Priority 1: Dealer exhaustion
        'volume_specialist',                  # Priority 2: Micro-surge vs macro-trend
        'volatility_specialist',             # Priority 3: Dynamic wavelet thresholding
        'information_specialist'              # Priority 4: PIN/VPIN informed flow
    ],
    
    # Resonance Detection
    'coherence_threshold': 0.7,            # Minimum coherence for resonance
    'phase_threshold': 0.1,                 # Phase difference threshold for lead/lag
    'resonance_pairs': [('d1', 'd3'), ('d2', 'd4')],  # Micro-macro scale pairs
    
    # Causal Compression
    'pca_components': 2,                    # Number of PCA components to keep
    'variance_threshold': 0.95,             # Minimum variance to preserve
    'min_samples': 100,                     # Minimum samples for compression
    
    # Spectral Chaser
    'spectral_chaser_enabled': True,         # Enable Spectral Chaser
    'spectral_chaser_models': ['xgb', 'catboost', 'rf', 'linear'],  # Model types
    'spectral_chaser_cv_folds': 5,          # Cross-validation folds
    
    # RSV Integration
    'rsv_integration_enabled': True,         # Enable RSV integration in Layer 3
    'rsv_position_sizing': True,             # Enable RSV-based position sizing
    'rsv_regime_aware': True,                # Enable regime-aware trading
    
    # =============================================================================
    # COMPREHENSIVE METRICS
    =============================================================================
    'comprehensive_metrics_enabled': True,  # Enable comprehensive metrics reporting
    'layer2_metrics_enabled': True,        # Layer 2 discovery and engineering metrics
    'layer25_metrics_enabled': True,       # Layer 2.5 Chaser metrics
    'layer3_metrics_enabled': True,        # Layer 3 meta-learner metrics'''

# Apply the replacement
content = re.sub(pattern, replacement, content)

# Write back to file
with open('comprehensive_de_prado_config.py', 'w') as f:
    f.write(content)

print("Enhanced comprehensive_de_prado_config.py with AEDL configuration")
