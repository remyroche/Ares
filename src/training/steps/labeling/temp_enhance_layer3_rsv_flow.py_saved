import re

# Read the file
with open('layer3/core.py', 'r') as f:
    content = f.read()

# Find the layer3_analyst_lgbm function and enhance RSV integration
pattern = r'(    # Integrate RSV features if available\s+if cfg\.get\("rsv_integration_enabled", True):)'

# Enhanced RSV integration with proper data flow
replacement = r'''    # Integrate RSV features if available (ENHANCED)
    if cfg.get("rsv_integration_enabled", True):
        try:
            tprint_info("🔢 Layer 3: Integrating RSV features from Layer 2...")
            
            # Get RSV data from Layer 2 with multiple fallback methods
            rsv_data = None
            
            # Method 1: Check if RSV data is passed through layer3_analyst_lgbm
            if hasattr(layer3_analyst_lgbm, 'layer2_rsv_data'):
                rsv_data = layer3_analyst_lgbm.layer2_rsv_data
                if cfg.get('verbose', True):
                    tprint_info("   📊 Found RSV data in layer3_analyst_lgbm.layer2_rsv_data")
            
            # Method 2: Check if RSV data is in cfg
            if rsv_data is None and 'layer2_rsv_data' in cfg:
                rsv_data = cfg['layer2_rsv_data']
                if cfg.get('verbose', True):
                    tprint_info("   📊 Found RSV data in cfg['layer2_rsv_data']")
            
            # Method 3: Check if RSV data is in global context
            if rsv_data is None:
                try:
                    # Try to get from the calling context
                    import inspect
                    frame = inspect.currentframe()
                    while frame:
                        if 'layer2_rsv_data' in frame.f_locals:
                            rsv_data = frame.f_locals['layer2_rsv_data']
                            if cfg.get('verbose', True):
                                tprint_info("   📊 Found RSV data in calling context")
                            break
                        frame = frame.f_back
                except Exception:
                    pass
            
            # Process RSV data if found
            if rsv_data is not None and isinstance(rsv_data, dict):
                # Add RSV eigenvalue as primary feature
                rsv_eigenvalue = rsv_data.get('eigenvalue', rsv_data.get('rsv_eigenvalue', 0.0))
                df['rsv_eigenvalue'] = rsv_eigenvalue
                
                # Add RSV regime as categorical feature
                rsv_regime = rsv_data.get('resonance_regime', rsv_data.get('rsv_info', {}).get('resonance_regime', 'UNKNOWN'))
                df['rsv_regime'] = rsv_regime
                
                # Add position sizing guidance
                position_guidance = rsv_data.get('position_guidance', rsv_data.get('position_sizing_guidance', {}))
                if not position_guidance:
                    position_guidance = rsv_data.get('rsv_info', {}).get('position_sizing_guidance', {})
                
                df['rsv_position_size'] = position_guidance.get('recommended_position_size', 0.10)
                df['rsv_leverage_multiplier'] = position_guidance.get('leverage_multiplier', 1.0)
                df['rsv_confidence_level'] = position_guidance.get('confidence_level', 0.5)
                
                # Add additional RSV features if available
                if 'harmonic_entries' in rsv_data:
                    harmonic_entries = rsv_data['harmonic_entries']
                    df['rsv_harmonic_signal'] = harmonic_entries.get('entry_signal', 0)
                    df['rsv_entry_quality'] = harmonic_entries.get('entry_quality', 0)
                
                if 'structural_breakouts' in rsv_data:
                    breakouts = rsv_data['structural_breakouts']
                    df['rsv_breakout_signal'] = breakouts.get('dominant_breakout_specialist', 0)
                    df['rsv_breakout_strength'] = breakouts.get('breakout_strength', 0)
                
                # Add to meta-features if they exist
                rsv_features = [
                    'rsv_eigenvalue', 'rsv_regime', 'rsv_position_size', 
                    'rsv_leverage_multiplier', 'rsv_confidence_level',
                    'rsv_harmonic_signal', 'rsv_entry_quality', 
                    'rsv_breakout_signal', 'rsv_breakout_strength'
                ]
                
                # Add to meta_features if they exist
                if 'meta_features' in locals():
                    meta_features.extend(rsv_features)
                
                if cfg.get('verbose', True):
                    tprint_success(f"   ✅ RSV features integrated:")
                    tprint_info(f"      - RSV eigenvalue: {rsv_eigenvalue:.3f}")
                    tprint_info(f"      - RSV regime: {rsv_regime}")
                    tprint_info(f"      - Position size: {position_guidance.get('recommended_position_size', 0.10):.1%}")
                    tprint_info(f"      - Leverage multiplier: {position_guidance.get('leverage_multiplier', 1.0):.1f}")
                    tprint_info(f"      - Confidence level: {position_guidance.get('confidence_level', 0.5):.2f}")
                    tprint_info(f"      - Total RSV features: {len([f for f in rsv_features if f in df.columns])}")
            else:
                if cfg.get('verbose', True):
                    tprint_warning("   ⚠️ No RSV data available from Layer 2")
                    tprint_info("      - Layer 2.5 Spectral Chaser may not be enabled")
                    tprint_info("      - AEDL framework may not be enabled")
                
        except Exception as e:
            if cfg.get('verbose', True):
                tprint_warning(f"   ⚠️ RSV integration failed: {e}")
                import traceback
                tprint_info(f"      - Error details: {traceback.format_exc()}")'''

# Apply the replacement
content = re.sub(pattern, replacement, content)

# Write back to file
with open('layer3/core.py', 'w') as f:
    f.write(content)

print("Enhanced Layer 3 RSV integration with proper data flow")
