#!/usr/bin/env python3
"""
Test script to demonstrate S/R analysis report generation functionality.
"""

import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import numpy as np
import time

def create_sample_sr_data():
    """Create sample S/R data to demonstrate the functionality."""
    
    # Sample S/R context data (what would come from SRBreakoutPredictor)
    sr_context = {
        'current_price': 3250.50,
        'support_levels': [
            {'price': 3200.00, 'strength': 0.85, 'touches': 5},
            {'price': 3150.00, 'strength': 0.72, 'touches': 3},
            {'price': 3100.00, 'strength': 0.68, 'touches': 4},
            {'price': 3050.00, 'strength': 0.61, 'touches': 2},
            {'price': 3000.00, 'strength': 0.55, 'touches': 3}
        ],
        'resistance_levels': [
            {'price': 3300.00, 'strength': 0.78, 'touches': 4},
            {'price': 3350.00, 'strength': 0.71, 'touches': 3},
            {'price': 3400.00, 'strength': 0.65, 'touches': 2},
            {'price': 3450.00, 'strength': 0.58, 'touches': 2},
            {'price': 3500.00, 'strength': 0.52, 'touches': 1}
        ],
        'nearest_support': {'price': 3200.00, 'strength': 0.85, 'touches': 5},
        'nearest_resistance': {'price': 3300.00, 'strength': 0.78, 'touches': 4},
        'market_structure': 'bullish',
        'breakout_probability': 0.65,
        'volume_confirmation': True
    }
    
    # Sample S/R regime features
    sr_regime_features = {
        'sr_proximity_by_regime': {
            'regime_0': {'support_proximity': 0.85, 'resistance_proximity': 0.75},
            'regime_1': {'support_proximity': 0.65, 'resistance_proximity': 0.45},
            'regime_2': {'support_proximity': 0.45, 'resistance_proximity': 0.85}
        },
        'sr_strength_by_regime': {
            'regime_0': {'support_strength': 0.78, 'resistance_strength': 0.65},
            'regime_1': {'support_strength': 0.55, 'resistance_strength': 0.45},
            'regime_2': {'support_strength': 0.45, 'resistance_strength': 0.72}
        }
    }
    
    return sr_context, sr_regime_features

def generate_sr_analysis_report(composite_analysis):
    """Generate Support/Resistance analysis report (copied from step03)."""
    try:
        report = []
        report.append('# Support/Resistance Analysis Report')
        report.append('')
        
        # Extract S/R data from composite analysis
        sr_features = composite_analysis.get('sr_regime_features', {})
        sr_context = composite_analysis.get('sr_context', {})
        
        # Current S/R Levels
        report.append('## Current Support/Resistance Levels')
        report.append('')
        
        # Support levels
        support_levels = sr_context.get('support_levels', [])
        if support_levels:
            report.append('### Support Levels')
            for i, level in enumerate(support_levels[:5]):  # Top 5 support levels
                strength = level.get('strength', 0.5)
                touches = level.get('touches', 0)
                report.append(f'- **Support {i+1}**: ${level.get("price", 0):.2f} (Strength: {strength:.2f}, Touches: {touches})')
            report.append('')
        else:
            report.append('### Support Levels')
            report.append('- No significant support levels identified')
            report.append('')
        
        # Resistance levels
        resistance_levels = sr_context.get('resistance_levels', [])
        if resistance_levels:
            report.append('### Resistance Levels')
            for i, level in enumerate(resistance_levels[:5]):  # Top 5 resistance levels
                strength = level.get('strength', 0.5)
                touches = level.get('touches', 0)
                report.append(f'- **Resistance {i+1}**: ${level.get("price", 0):.2f} (Strength: {strength:.2f}, Touches: {touches})')
            report.append('')
        else:
            report.append('### Resistance Levels')
            report.append('- No significant resistance levels identified')
            report.append('')
        
        # Current Price Analysis
        current_price = sr_context.get('current_price', 0)
        if current_price > 0:
            report.append('## Current Price Analysis')
            report.append('')
            report.append(f'**Current Price**: ${current_price:.2f}')
            
            # Distance to nearest levels
            nearest_support = sr_context.get('nearest_support', {})
            nearest_resistance = sr_context.get('nearest_resistance', {})
            
            if nearest_support:
                support_price = nearest_support.get('price', 0)
                support_distance = ((current_price - support_price) / current_price) * 100
                report.append(f'**Distance to Nearest Support**: {support_distance:.2f}% (${support_price:.2f})')
            
            if nearest_resistance:
                resistance_price = nearest_resistance.get('price', 0)
                resistance_distance = ((resistance_price - current_price) / current_price) * 100
                report.append(f'**Distance to Nearest Resistance**: {resistance_distance:.2f}% (${resistance_price:.2f})')
            
            report.append('')
        
        # S/R Strength Analysis
        report.append('## Support/Resistance Strength Analysis')
        report.append('')
        
        # Overall market structure
        market_structure = sr_context.get('market_structure', 'neutral')
        report.append(f'**Market Structure**: {market_structure.title()}')
        
        # Breakout probability
        breakout_probability = sr_context.get('breakout_probability', 0.5)
        report.append(f'**Breakout Probability**: {breakout_probability:.1%}')
        
        # Volume confirmation
        volume_confirmation = sr_context.get('volume_confirmation', False)
        report.append(f'**Volume Confirmation**: {"Yes" if volume_confirmation else "No"}')
        report.append('')
        
        # Regime-Specific S/R Analysis
        sr_proximity = sr_features.get('sr_proximity_by_regime', {})
        sr_strength = sr_features.get('sr_strength_by_regime', {})
        
        if sr_proximity or sr_strength:
            report.append('## Regime-Specific S/R Analysis')
            report.append('')
            
            for regime_key in sr_proximity.keys():
                regime_num = regime_key.split('_')[1]
                proximity_data = sr_proximity.get(regime_key, {})
                strength_data = sr_strength.get(regime_key, {})
                
                report.append(f'### Regime {regime_num}')
                report.append(f'- **Support Proximity**: {proximity_data.get("support_proximity", 1.0):.2f}')
                report.append(f'- **Resistance Proximity**: {proximity_data.get("resistance_proximity", 1.0):.2f}')
                report.append(f'- **Support Strength**: {strength_data.get("support_strength", 0.5):.2f}')
                report.append(f'- **Resistance Strength**: {strength_data.get("resistance_strength", 0.5):.2f}')
                report.append('')
        
        # Trading Implications
        report.append('## Trading Implications')
        report.append('')
        
        # Breakout scenarios
        if breakout_probability > 0.7:
            report.append('### High Breakout Probability')
            report.append('- Monitor for volume confirmation on breakouts')
            report.append('- Set stop losses below support levels')
            report.append('- Target next resistance level on breakouts')
        elif breakout_probability < 0.3:
            report.append('### Low Breakout Probability')
            report.append('- Expect range-bound trading between S/R levels')
            report.append('- Use mean reversion strategies')
            report.append('- Avoid breakout trades without strong confirmation')
        else:
            report.append('### Moderate Breakout Probability')
            report.append('- Wait for clear directional bias')
            report.append('- Use tight stop losses')
            report.append('- Monitor volume for confirmation')
        
        report.append('')
        
        # Key Levels to Watch
        report.append('## Key Levels to Watch')
        report.append('')
        
        if support_levels:
            strongest_support = max(support_levels, key=lambda x: x.get('strength', 0))
            report.append(f'**Key Support**: ${strongest_support.get("price", 0):.2f} (Strength: {strongest_support.get("strength", 0):.2f})')
        
        if resistance_levels:
            strongest_resistance = max(resistance_levels, key=lambda x: x.get('strength', 0))
            report.append(f'**Key Resistance**: ${strongest_resistance.get("price", 0):.2f} (Strength: {strongest_resistance.get("strength", 0):.2f})')
        
        return '\n'.join(report)
        
    except Exception as e:
        return f'Error generating S/R analysis report: {e}'

def main():
    """Main function to demonstrate S/R report generation."""
    
    print("🚀 Support/Resistance Analysis Report Generation Demo")
    print("=" * 60)
    
    # Parameters
    symbol = "ETHUSDT"
    exchange = "BINANCE"
    timeframe = "1m"
    data_dir = "data_cache"
    
    print(f"📊 Symbol: {symbol}")
    print(f"🏢 Exchange: {exchange}")
    print(f"⏰ Timeframe: {timeframe}")
    print()
    
    # Create sample S/R data
    print("📝 Generating sample S/R data...")
    sr_context, sr_regime_features = create_sample_sr_data()
    
    # Create composite analysis structure
    composite_analysis = {
        'sr_context': sr_context,
        'sr_regime_features': sr_regime_features
    }
    
    # Generate S/R analysis report
    print("📊 Generating S/R analysis report...")
    sr_report = generate_sr_analysis_report(composite_analysis)
    
    # Create reports directory
    reports_dir = Path(data_dir) / 'reports' / 'step03_hmm_regime_discovery'
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for file naming
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save S/R analysis report
    sr_report_file = reports_dir / f'{exchange}_{symbol}_{timeframe}_sr_analysis_{timestamp}.md'
    with open(sr_report_file, 'w', encoding='utf-8') as f:
        f.write(sr_report)
    
    print(f'📄 Saved S/R analysis report: {sr_report_file}')
    
    # Save S/R context as JSON
    sr_context_file = reports_dir / f'{exchange}_{symbol}_{timeframe}_sr_context_{timestamp}.json'
    with open(sr_context_file, 'w', encoding='utf-8') as f:
        json.dump(sr_context, f, indent=2, default=str)
    
    print(f'📊 Saved S/R context: {sr_context_file}')
    
    print()
    print("✅ S/R analysis report generation completed successfully!")
    print(f"📁 Report saved to: {reports_dir}")
    print()
    print("📋 Generated files:")
    print(f"   - {sr_report_file.name}")
    print(f"   - {sr_context_file.name}")
    
    print()
    print("🎯 This demonstrates the S/R analysis report functionality that is now")
    print("   integrated into step03_hmm_regime_discovery when it runs successfully.")
    
    # Show a preview of the report
    print()
    print("📖 Report Preview:")
    print("-" * 40)
    print(sr_report[:500] + "..." if len(sr_report) > 500 else sr_report)

if __name__ == "__main__":
    main()
