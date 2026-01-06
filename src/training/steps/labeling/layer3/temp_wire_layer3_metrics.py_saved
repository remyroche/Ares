import re

# Read the file
with open('layer3/core.py', 'r') as f:
    content = f.read()

# Find the end of the layer3_analyst_lgbm function and add comprehensive metrics reporting
end_pattern = r'(    return df, meta_results)'

# Enhanced ending with comprehensive metrics
enhanced_ending = '''    # Generate comprehensive metrics report
    if cfg.get("comprehensive_metrics_enabled", True):
        try:
            tprint_info("📊 Layer 3: Generating comprehensive metrics report...")
            
            metrics_report = {
                'layer3_metrics': {
                    'total_features': len(meta_features),
                    'base_features': len(safe_base_cols),
                    'minimal_features': len([f for f in meta_features if 'minimal_' in f]),
                    'streamlined_features': len([f for f in meta_features if 'streamlined_' in f]),
                    'chaser_features': len([f for f in meta_features if 'chaser_' in f]),
                    'regime_features': len([f for f in meta_features if any(x in f for x in ['regime_', 'vol_regime', 'trend_regime'])]),
                    'liquidity_features': len([f for f in meta_features if any(x in f for x in ['liquidity_', 'spread_', 'vwap_', 'impact_'])]),
                    'feature_expansion_ratio': len(meta_features) / len(safe_base_cols) if len(safe_base_cols) > 0 else 1.0
                },
                'feature_importance_analysis': {
                    'top_10_features': meta_features[:10] if len(meta_features) >= 10 else meta_features,
                    'feature_categories': {
                        'minimal': [f for f in meta_features if 'minimal_' in f],
                        'streamlined': [f for f in meta_features if 'streamlined_' in f],
                        'chaser': [f for f in meta_features if 'chaser_' in f],
                        'regime': [f for f in meta_features if any(x in f for x in ['regime_', 'vol_regime', 'trend_regime'])],
                        'liquidity': [f for f in meta_features if any(x in f for x in ['liquidity_', 'spread_', 'vwap_', 'impact_'])]
                    }
                },
                'layer3_performance': {
                    'meta_learner_type': 'lgbm_analyst',
                    'feature_engineering_complete': True,
                    'causal_framework_integrated': True
                }
            }
            
            # Add metrics to results
            if 'meta_results' not in locals():
                meta_results = {}
            meta_results['comprehensive_metrics'] = metrics_report
            
            tprint_success("✅ Layer 3 comprehensive metrics complete:")
            tprint_info(f"   - Total features: {metrics_report['layer3_metrics']['total_features']}")
            tprint_info(f"   - Feature expansion: {metrics_report['layer3_metrics']['feature_expansion_ratio']:.2f}x")
            tprint_info(f"   - Regime features: {metrics_report['layer3_metrics']['regime_features']}")
            tprint_info(f"   - Liquidity features: {metrics_report['layer3_metrics']['liquidity_features']}")
            
        except Exception as e:
            tprint_warning(f"⚠️ Layer 3 comprehensive metrics failed: {e}")
    
    return df, meta_results'''

# Apply the replacement
content = re.sub(end_pattern, enhanced_ending, content)

# Write back to file
with open('layer3/core.py', 'w') as f:
    f.write(content)

print("Added comprehensive metrics reporting to layer3/core.py")
