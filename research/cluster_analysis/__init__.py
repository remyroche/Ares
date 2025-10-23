"""
Cluster Analysis Research Framework

A systematic 4-step approach to market analysis:
1. Price Patterns: Discover and define price movement patterns
2. Market Factor Analysis: Transform features into market dimensions  
3. Clustering: Define market states from dimensions
4. Economic Relevance: Analyze dimension-pattern relationships

Usage:
    from research.cluster_analysis import (
        PricePatternOrchestrator,
        MarketFactorAnalyzer,
        MarketStateClusterer, 
        EconomicRelevanceAnalyzer
    )
"""

# Main orchestrators for each research phase
from .price_patterns import PricePatternOrchestrator
from .market_factor_analysis import MarketFactorAnalyzer  
from .clustering import MarketStateClusterer
from .economic_relevance import EconomicRelevanceAnalyzer

# Core framework version
__version__ = "1.0.0"

# Research workflow components
__all__ = [
    "PricePatternOrchestrator",
    "MarketFactorAnalyzer", 
    "MarketStateClusterer",
    "EconomicRelevanceAnalyzer"
]

# Framework workflow
def run_complete_analysis(price_data, feature_data):
    """
    Run complete 4-step cluster analysis workflow.
    
    Args:
        price_data: OHLCV price data
        feature_data: Engineered features
        
    Returns:
        Complete analysis results from all 4 steps
    """
    
    # Step 1: Discover price patterns
    pattern_orchestrator = PricePatternOrchestrator()
    patterns = pattern_orchestrator.discover_all_patterns(price_data)
    
    # Step 2: Extract market dimensions
    factor_analyzer = MarketFactorAnalyzer()
    dimensions = factor_analyzer.discover_market_dimensions(feature_data)
    
    # Step 3: Cluster market states
    clusterer = MarketStateClusterer()
    market_states = clusterer.discover_market_states(dimensions)
    
    # Step 4: Analyze economic relevance
    relevance_analyzer = EconomicRelevanceAnalyzer()
    relevance = relevance_analyzer.analyze_pattern_dimension_relevance(
        patterns, dimensions, market_states
    )
    
    return {
        'patterns': patterns,
        'dimensions': dimensions, 
        'market_states': market_states,
        'economic_relevance': relevance
    }