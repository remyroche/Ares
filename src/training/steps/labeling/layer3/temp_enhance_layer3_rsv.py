import re

# Read the file
with open('layer3/core.py', 'r') as f:
    content = f.read()

# Add RSV integration before the final return
end_pattern = r'(    return df, meta_results)'

# Enhanced ending with RSV integration
enhanced_ending = '''    # Integrate RSV features if available
    if cfg.get("rsv_integration_enabled", True):
        try:
            tprint_info("🔢 Layer 3: Integrating RSV features...")
            
            # Check if RSV data is available from Layer 2
            rsv_data = None
            if hasattr(self, 'layer2_rsv_data'):
                rsv_data = self.layer2_rsv_data
            
            if rsv_data is not None:
                # Add RSV eigenvalue as primary feature
                rsv_eigenvalue = rsv_data.get('rsv_eigenvalue', 0.0)
                df['rsv_eigenvalue'] = rsv_eigenvalue
                
                # Add RSV regime as categorical feature
                rsv_regime = rsv_data.get('position_sizing_guidance', {}).get('resonance_regime', 'UNKNOWN')
                df['rsv_regime'] = rsv_regime
                
                # Add position sizing guidance
                recommended_size = rsv_data.get('position_sizing_guidance', {}).get('recommended_position_size', 0.10)
                df['rsv_position_size'] = recommended_size
                
                # Add leverage multiplier
                leverage_mult = rsv_data.get('position_sizing_guidance', {}).get('leverage_multiplier', 1.0)
                df['rsv_leverage_multiplier'] = leverage_mult
                
                # Add confidence level
                confidence_level = rsv_data.get('position_sizing_guidance', {}).get('confidence_level', 0.5)
                df['rsv_confidence_level'] = confidence_level
                
                # Add to meta-features if they exist
                if 'meta_features' in locals():
                    meta_features.extend([
                        'rsv_eigenvalue', 'rsv_regime', 'rsv_position_size', 
                        'rsv_leverage_multiplier', 'rsv_confidence_level'
                    ])
                
                if self.verbose:
                    tprint_success(f"   ✅ RSV features integrated:")
                    tprint_info(f"      - RSV eigenvalue: {rsv_eigenvalue:.3f}")
                    tprint_info(f"      - RSV regime: {rsv_regime}")
                    tprint_info(f"      - Position size: {recommended_size:.1%}")
                    tprint_info(f"      - Leverage multiplier: {leverage_mult:.1f}")
                    tprint_info(f"      - Confidence level: {confidence_level:.2f}")
            else:
                if self.verbose:
                    tprint_warning("   ⚠️ No RSV data available from Layer 2")
                
        except Exception as e:
            if self.verbose:
                tprint_warning(f"   ⚠️ RSV integration failed: {e}")
    
    return df, meta_results'''

# Apply the replacement
content = re.sub(end_pattern, enhanced_ending, content)

# Write back to file
with open('layer3/core.py', 'w') as f:
    f.write(content)

print("Enhanced layer3/core.py with RSV integration")
