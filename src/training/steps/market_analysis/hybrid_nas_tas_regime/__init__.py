"""
Hybrid NAS TAS Regime Module

This module combines the outputs from NAS regime detection and TAS regime detection
to create a coherent regime modeling system with economic and financial relevance.

Key Features:
- Integrates TAS and NAS regime detection outputs
- Creates coherent regime modeling with economic significance
- Performs clustering based on combined TAS & NAS inputs
- Tags existing data with regime information
- Replaces hmm_clustering functionality

Architecture:
- Core: Main hybrid regime detector and modeling components
- Integration: TAS and NAS integration components
- Clustering: Advanced clustering algorithms for regime detection
- Modeling: Economic and financial regime modeling
- Tagging: Data tagging and labeling functionality
- Config: Configuration management
- Utils: Utility functions and helpers
"""

from .core.hybrid_regime_detector import HybridRegimeDetector
from .core.hybrid_regime_modeler import HybridRegimeModeler
from .core.economic_regime_analyzer import EconomicRegimeAnalyzer
from .core.financial_regime_analyzer import FinancialRegimeAnalyzer

from .integration.tas_integration import TASIntegration
from .integration.nas_integration import NASIntegration
from .integration.hybrid_integration import HybridIntegration

from .clustering.hybrid_clusterer import HybridClusterer
from .clustering.economic_clusterer import EconomicClusterer
from .clustering.financial_clusterer import FinancialClusterer

from .modeling.regime_modeler import RegimeModeler
from .modeling.economic_modeler import EconomicModeler
from .modeling.financial_modeler import FinancialModeler

from .tagging.regime_tagger import RegimeTagger
from .tagging.economic_tagger import EconomicTagger
from .tagging.financial_tagger import FinancialTagger

from .config.hybrid_config import HybridNASConfig, HybridTASConfig, HybridRegimeConfig
from .utils.regime_utils import RegimeUtils
from .utils.economic_utils import EconomicUtils
from .utils.financial_utils import FinancialUtils

__version__ = "1.0.0"
__author__ = "Hybrid NAS TAS Regime System"
__description__ = "Hybrid regime detection combining NAS and TAS with economic and financial relevance"

__all__ = [
    # Core components
    'HybridRegimeDetector',
    'HybridRegimeModeler', 
    'EconomicRegimeAnalyzer',
    'FinancialRegimeAnalyzer',
    
    # Integration components
    'TASIntegration',
    'NASIntegration',
    'HybridIntegration',
    
    # Clustering components
    'HybridClusterer',
    'EconomicClusterer',
    'FinancialClusterer',
    
    # Modeling components
    'RegimeModeler',
    'EconomicModeler',
    'FinancialModeler',
    
    # Tagging components
    'RegimeTagger',
    'EconomicTagger',
    'FinancialTagger',
    
    # Configuration
    'HybridNASConfig',
    'HybridTASConfig', 
    'HybridRegimeConfig',
    
    # Utilities
    'RegimeUtils',
    'EconomicUtils',
    'FinancialUtils'
]