"""
Feature Generation Adapter

Adapter class that integrates the existing src/feature_generation/ system
with the new unified data-driven pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
import logging

try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)

logger = logging.getLogger(__name__)


class FeatureGenerationAdapter:
    """
    Adapter that integrates existing feature generation with unified pipeline.
    
    This class provides a seamless interface between the existing
    src/feature_generation/ system and the new unified data-driven pipeline.
    """
    
    def __init__(self, 
                 feature_generation_config: Optional[Dict[str, Any]] = None,
                 enable_existing_features: bool = True):
        """
        Initialize the feature generation adapter.
        
        Args:
            feature_generation_config: Configuration for feature generation
            enable_existing_features: Whether to use existing feature generation system
        """
        self.feature_generation_config = feature_generation_config or {}
        self.enable_existing_features = enable_existing_features
        self.feature_bank = None
        self.generated_features = {}
        
        # Try to import existing feature generation system
        if self.enable_existing_features:
            self._initialize_existing_features()
        
        tprint_info("FeatureGenerationAdapter initialized")
    
    def _initialize_existing_features(self):
        """Initialize connection to existing feature generation system."""
        try:
            from src.feature_generation.core.factory import get_feature_bank
            self.feature_bank = get_feature_bank(self.feature_generation_config)
            tprint_success("✓ Connected to existing feature generation system")
        except ImportError as e:
            tprint_warning(f"⚠️ Existing feature generation not available: {e}")
            self.enable_existing_features = False
        except Exception as e:
            tprint_error(f"❌ Failed to initialize feature generation: {e}")
            self.enable_existing_features = False
    
    def generate_features(self, 
                         data: pd.DataFrame, 
                         categories: Optional[List[str]] = None,
                         max_features_per_category: int = 10) -> pd.DataFrame:
        """
        Generate features using existing or synthetic methods.
        
        Args:
            data: Input market data
            categories: List of feature categories to generate
            max_features_per_category: Maximum features per category
            
        Returns:
            DataFrame with generated features
        """
        if categories is None:
            categories = ['momentum', 'volatility', 'volume', 'trend', 'oscillator']
        
        tprint_info(f"Generating features for categories: {categories}")
        
        if self.enable_existing_features and self.feature_bank:
            return self._generate_with_existing_system(data, categories, max_features_per_category)
        else:
            return self._generate_synthetic_features(data, categories, max_features_per_category)
    
    def _generate_with_existing_system(self, 
                                     data: pd.DataFrame, 
                                     categories: List[str],
                                     max_features_per_category: int) -> pd.DataFrame:
        """Generate features using existing feature generation system."""
        tprint_debug("Using existing feature generation system")
        
        all_features = {}
        
        for category in categories:
            try:
                tprint_debug(f"Generating {category} features...")
                
                # Get generators for this category
                generators = self.feature_bank.get_generators_by_category(category)
                
                if not generators:
                    tprint_warning(f"No generators found for category: {category}")
                    continue
                
                # Limit number of generators
                generators = generators[:max_features_per_category]
                
                category_features = {}
                for generator in generators:
                    try:
                        result = generator.generate(data)
                        if hasattr(result, 'features') and result.features:
                            category_features.update(result.features)
                    except Exception as e:
                        tprint_debug(f"Generator {generator.name} failed: {e}")
                        continue
                
                all_features.update(category_features)
                tprint_debug(f"Generated {len(category_features)} {category} features")
                
            except Exception as e:
                tprint_warning(f"Category {category} generation failed: {e}")
                continue
        
        if not all_features:
            tprint_warning("No features generated, falling back to synthetic features")
            return self._generate_synthetic_features(data, categories, max_features_per_category)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(all_features, index=data.index)
        features_df = features_df.fillna(method='ffill').fillna(0)
        
        tprint_success(f"Generated {len(features_df.columns)} features using existing system")
        return features_df
    
    def _generate_synthetic_features(self, 
                                   data: pd.DataFrame, 
                                   categories: List[str],
                                   max_features_per_category: int) -> pd.DataFrame:
        """Generate synthetic features for demonstration."""
        tprint_debug("Using synthetic feature generation")
        
        all_features = {}
        
        for category in categories:
            category_features = {}
            
            if category == 'momentum':
                # Momentum features
                for i in range(min(5, max_features_per_category)):
                    category_features[f'momentum_{i+1}'] = data['close'].pct_change(i+1)
                    category_features[f'momentum_ma_{i+1}'] = data['close'].pct_change(i+1).rolling(5).mean()
            
            elif category == 'volatility':
                # Volatility features
                for i in range(min(5, max_features_per_category)):
                    category_features[f'volatility_{i+1}'] = data['close'].rolling(i+1).std()
                    category_features[f'volatility_ma_{i+1}'] = data['close'].rolling(i+1).std().rolling(5).mean()
            
            elif category == 'volume':
                # Volume features
                for i in range(min(5, max_features_per_category)):
                    category_features[f'volume_ma_{i+1}'] = data['volume'].rolling(i+1).mean()
                    category_features[f'volume_ratio_{i+1}'] = data['volume'] / data['volume'].rolling(i+1).mean()
            
            elif category == 'trend':
                # Trend features
                for i in range(min(5, max_features_per_category)):
                    category_features[f'sma_{i+1}'] = data['close'].rolling(i+1).mean()
                    category_features[f'ema_{i+1}'] = data['close'].ewm(span=i+1).mean()
            
            elif category == 'oscillator':
                # Oscillator features
                for i in range(min(5, max_features_per_category)):
                    high = data['high'].rolling(i+1).max()
                    low = data['low'].rolling(i+1).min()
                    category_features[f'rsi_{i+1}'] = self._calculate_rsi(data['close'], i+1)
                    category_features[f'stoch_{i+1}'] = (data['close'] - low) / (high - low)
            
            all_features.update(category_features)
            tprint_debug(f"Generated {len(category_features)} synthetic {category} features")
        
        # Convert to DataFrame
        features_df = pd.DataFrame(all_features, index=data.index)
        features_df = features_df.fillna(method='ffill').fillna(0)
        
        tprint_success(f"Generated {len(features_df.columns)} synthetic features")
        return features_df
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI for demonstration."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def get_available_categories(self) -> List[str]:
        """Get list of available feature categories."""
        if self.enable_existing_features and self.feature_bank:
            try:
                return self.feature_bank.get_available_categories()
            except Exception:
                pass
        
        return ['momentum', 'volatility', 'volume', 'trend', 'oscillator']
    
    def get_category_info(self, category: str) -> Dict[str, Any]:
        """Get information about a specific category."""
        if self.enable_existing_features and self.feature_bank:
            try:
                generators = self.feature_bank.get_generators_by_category(category)
                return {
                    'name': category,
                    'generator_count': len(generators),
                    'generators': [g.name for g in generators[:5]]  # First 5
                }
            except Exception:
                pass
        
        return {
            'name': category,
            'generator_count': 5,  # Synthetic features
            'generators': [f'{category}_1', f'{category}_2', f'{category}_3', f'{category}_4', f'{category}_5']
        }


class IntegratedFeaturePipeline:
    """
    Integrated pipeline that combines feature generation and selection.
    
    This class provides a complete feature engineering solution by combining
    the existing feature generation system with the new unified pipeline.
    """
    
    def __init__(self, 
                 pipeline_config=None,
                 feature_generation_config=None,
                 enable_existing_features: bool = True):
        """
        Initialize the integrated pipeline.
        
        Args:
            pipeline_config: Configuration for unified pipeline
            feature_generation_config: Configuration for feature generation
            enable_existing_features: Whether to use existing feature generation
        """
        # Import unified pipeline
        from .unified_pipeline import create_unified_pipeline, create_default_config
        
        # Initialize unified pipeline
        self.pipeline_config = pipeline_config or create_default_config()
        self.pipeline = create_unified_pipeline(self.pipeline_config)
        
        # Initialize feature generation adapter
        self.feature_adapter = FeatureGenerationAdapter(
            feature_generation_config=feature_generation_config,
            enable_existing_features=enable_existing_features
        )
        
        tprint_info("IntegratedFeaturePipeline initialized")
    
    def process(self, 
                data: pd.DataFrame, 
                targets: Optional[pd.Series] = None,
                feature_categories: Optional[List[str]] = None,
                max_features_per_category: int = 10) -> Any:
        """
        Process data through complete feature engineering pipeline.
        
        Args:
            data: Input market data
            targets: Target variable for feature selection
            feature_categories: Categories of features to generate
            max_features_per_category: Maximum features per category
            
        Returns:
            Pipeline result with selected features
        """
        tprint_info("Starting integrated feature processing")
        
        # Step 1: Generate features
        tprint_info("Step 1: Generating features")
        features_df = self.feature_adapter.generate_features(
            data, 
            categories=feature_categories,
            max_features_per_category=max_features_per_category
        )
        
        if features_df.empty:
            tprint_error("No features generated")
            return None
        
        # Step 2: Select optimal features
        if targets is not None:
            tprint_info("Step 2: Selecting optimal features")
            result = self.pipeline.process(features_df, targets)
            
            # Add generation metadata
            result.generation_metadata = {
                'total_features_generated': len(features_df.columns),
                'categories_used': feature_categories or self.feature_adapter.get_available_categories(),
                'max_features_per_category': max_features_per_category,
                'using_existing_system': self.feature_adapter.enable_existing_features
            }
            
            return result
        else:
            tprint_info("No targets provided, returning generated features")
            return features_df
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the integrated pipeline."""
        return {
            'pipeline_config': str(self.pipeline_config),
            'feature_generation_available': self.feature_adapter.enable_existing_features,
            'available_categories': self.feature_adapter.get_available_categories(),
            'pipeline_stats': self.pipeline.get_performance_stats()
        }


# Convenience functions
def create_integrated_pipeline(pipeline_config=None, 
                             feature_generation_config=None,
                             enable_existing_features: bool = True) -> IntegratedFeaturePipeline:
    """Create an integrated feature pipeline."""
    return IntegratedFeaturePipeline(
        pipeline_config=pipeline_config,
        feature_generation_config=feature_generation_config,
        enable_existing_features=enable_existing_features
    )


def process_with_integrated_pipeline(data: pd.DataFrame, 
                                   targets: Optional[pd.Series] = None,
                                   feature_categories: Optional[List[str]] = None,
                                   pipeline_config=None,
                                   feature_generation_config=None) -> Any:
    """Process data using integrated pipeline."""
    pipeline = create_integrated_pipeline(
        pipeline_config=pipeline_config,
        feature_generation_config=feature_generation_config
    )
    
    return pipeline.process(data, targets, feature_categories)