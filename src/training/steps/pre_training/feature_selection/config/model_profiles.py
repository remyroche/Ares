"""
Model profile management for feature selection.

This module provides model-specific configuration profiles and management
capabilities for different machine learning model types.
"""

from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum

from src.utils.tprint import tprint_debug, tprint_info, tprint_warning, tprint_success


class ModelType(Enum):
    """Supported model types for feature selection."""
    NEURAL_NETWORK = "neural_network"
    LINEAR_MODEL = "linear_model"
    ENSEMBLE_MODEL = "ensemble_model"
    TIME_SERIES = "time_series"
    REGIME_DETECTION = "regime_detection"
    ADVANCED_MAMBA_HYBRID = "AdvancedMambaHybrid"
    FINANCIAL_RESNET = "FinancialResNet"
    DEEP_SCALER = "DeepScaler"
    NBEATS = "NBEATS"
    DEFAULT = "default"


@dataclass
class ModelProfile:
    """Model-specific configuration profile."""
    model_type: ModelType
    target_features: int
    min_features: int
    max_features: int
    
    # Feature selection thresholds
    vif_threshold: float = 10.0
    correlation_threshold: float = 0.95
    mutual_info_threshold: float = 0.001
    variance_threshold: float = 0.01
    
    # Selection preferences
    priority_categories: List[str] = field(default_factory=lambda: ['momentum', 'volatility'])
    prefer_uncorrelated_features: bool = True
    prefer_interpretable_features: bool = False
    prefer_interaction_features: bool = False
    
    # Performance settings
    enable_parallel_processing: bool = True
    memory_efficient: bool = True
    chunk_size: int = 1000
    
    # Advanced settings
    enable_regime_aware_selection: bool = False
    enable_temporal_filtering: bool = False
    enable_precision_filtering: bool = False
    
    # Model-specific parameters
    model_specific_params: Dict[str, Any] = field(default_factory=dict)


class ModelProfileManager:
    """Manager for model-specific configuration profiles."""
    
    def __init__(self):
        self.logger = get_logger("ModelProfileManager")
        self.profiles: Dict[ModelType, ModelProfile] = {}
        
        # Initialize default profiles
        self._initialize_default_profiles()
    
    def _initialize_default_profiles(self):
        """Initialize default model profiles."""
        tprint_info("🔧 Initializing default model profiles")
        
        # Neural Network Profile
        self.profiles[ModelType.NEURAL_NETWORK] = ModelProfile(
            model_type=ModelType.NEURAL_NETWORK,
            target_features=80,
            min_features=60,
            max_features=100,
            vif_threshold=8.0,
            correlation_threshold=0.90,
            priority_categories=['momentum', 'volatility', 'microstructure'],
            prefer_interaction_features=True,
            enable_parallel_processing=True,
            memory_efficient=True
        )
        
        # Linear Model Profile
        self.profiles[ModelType.LINEAR_MODEL] = ModelProfile(
            model_type=ModelType.LINEAR_MODEL,
            target_features=60,
            min_features=40,
            max_features=80,
            vif_threshold=6.0,
            correlation_threshold=0.85,
            priority_categories=['statistical', 'momentum'],
            prefer_uncorrelated_features=True,
            prefer_interpretable_features=True,
            enable_parallel_processing=True,
            memory_efficient=True
        )
        
        # Ensemble Model Profile
        self.profiles[ModelType.ENSEMBLE_MODEL] = ModelProfile(
            model_type=ModelType.ENSEMBLE_MODEL,
            target_features=90,
            min_features=70,
            max_features=120,
            vif_threshold=12.0,
            correlation_threshold=0.88,
            priority_categories=['momentum', 'volatility', 'microstructure', 'temporal'],
            prefer_interaction_features=True,
            enable_parallel_processing=True,
            memory_efficient=True
        )
        
        # Time Series Profile
        self.profiles[ModelType.TIME_SERIES] = ModelProfile(
            model_type=ModelType.TIME_SERIES,
            target_features=70,
            min_features=50,
            max_features=90,
            vif_threshold=8.0,
            correlation_threshold=0.90,
            priority_categories=['temporal', 'trend', 'seasonality', 'volatility'],
            enable_temporal_filtering=True,
            enable_parallel_processing=True,
            memory_efficient=True
        )
        
        # Regime Detection Profile
        self.profiles[ModelType.REGIME_DETECTION] = ModelProfile(
            model_type=ModelType.REGIME_DETECTION,
            target_features=80,
            min_features=60,
            max_features=100,
            vif_threshold=10.0,
            correlation_threshold=0.95,
            priority_categories=['volatility', 'structural', 'volume_regime', 'statistical'],
            enable_regime_aware_selection=True,
            enable_parallel_processing=True,
            memory_efficient=True
        )
        
        # Advanced Mamba Hybrid Profile
        self.profiles[ModelType.ADVANCED_MAMBA_HYBRID] = ModelProfile(
            model_type=ModelType.ADVANCED_MAMBA_HYBRID,
            target_features=100,
            min_features=80,
            max_features=120,
            vif_threshold=8.0,
            correlation_threshold=0.88,
            priority_categories=['momentum', 'interaction', 'microstructure', 'temporal'],
            prefer_interaction_features=True,
            enable_parallel_processing=True,
            memory_efficient=True,
            model_specific_params={
                'attention_mechanism': True,
                'multi_timeframe': True,
                'gate_protection': True
            }
        )
        
        # Financial ResNet Profile
        self.profiles[ModelType.FINANCIAL_RESNET] = ModelProfile(
            model_type=ModelType.FINANCIAL_RESNET,
            target_features=120,
            min_features=100,
            max_features=150,
            vif_threshold=12.0,
            correlation_threshold=0.95,
            priority_categories=['regime', 'temporal', 'volatility', 'microstructure'],
            enable_regime_features=True,
            enable_parallel_processing=True,
            memory_efficient=True,
            model_specific_params={
                'residual_connections': True,
                'regime_classification': True,
                'deep_architecture': True
            }
        )
        
        # Deep Scaler Profile
        self.profiles[ModelType.DEEP_SCALER] = ModelProfile(
            model_type=ModelType.DEEP_SCALER,
            target_features=80,
            min_features=60,
            max_features=100,
            vif_threshold=6.0,
            correlation_threshold=0.85,
            priority_categories=['statistical', 'momentum', 'volatility'],
            enable_precision_filtering=True,
            enable_parallel_processing=True,
            memory_efficient=True,
            model_specific_params={
                'precision_focus': True,
                'scaling_aware': True,
                'quality_optimized': True
            }
        )
        
        # NBEATS Profile
        self.profiles[ModelType.NBEATS] = ModelProfile(
            model_type=ModelType.NBEATS,
            target_features=70,
            min_features=50,
            max_features=80,
            vif_threshold=8.0,
            correlation_threshold=0.90,
            priority_categories=['temporal', 'trend', 'seasonality', 'volatility'],
            enable_temporal_filtering=True,
            enable_parallel_processing=True,
            memory_efficient=True,
            model_specific_params={
                'temporal_modeling': True,
                'trend_seasonality': True,
                'backcast_forecast': True
            }
        )
        
        # Default Profile
        self.profiles[ModelType.DEFAULT] = ModelProfile(
            model_type=ModelType.DEFAULT,
            target_features=80,
            min_features=60,
            max_features=100,
            vif_threshold=10.0,
            correlation_threshold=0.95,
            priority_categories=['momentum', 'volatility', 'microstructure'],
            enable_parallel_processing=True,
            memory_efficient=True
        )
        
        tprint_success(f"   ✅ Initialized {len(self.profiles)} model profiles")
    
    def get_profile(self, model_type: Union[ModelType, str]) -> Optional[ModelProfile]:
        """Get model profile by type."""
        tprint_debug(f"🔍 Getting profile for model type: {model_type}")
        
        try:
            # Convert string to ModelType if needed
            if isinstance(model_type, str):
                model_type = self._parse_model_type(model_type)
            
            if model_type in self.profiles:
                profile = self.profiles[model_type]
                tprint_debug(f"   ✅ Found profile: {profile.target_features} features")
                return profile
            else:
                tprint_warning(f"   ⚠️ Profile not found for {model_type}, using default")
                return self.profiles[ModelType.DEFAULT]
                
        except Exception as e:
            tprint_warning(f"   ⚠️ Error getting profile: {e}")
            return self.profiles[ModelType.DEFAULT]
    
    def _parse_model_type(self, model_type_str: str) -> ModelType:
        """Parse string model type to ModelType enum."""
        # Try exact match first
        for model_type in ModelType:
            if model_type.value == model_type_str:
                return model_type
        
        # Try case-insensitive match
        for model_type in ModelType:
            if model_type.value.lower() == model_type_str.lower():
                return model_type
        
        # Try partial matches
        model_type_str_lower = model_type_str.lower()
        if 'neural' in model_type_str_lower or 'nn' in model_type_str_lower:
            return ModelType.NEURAL_NETWORK
        elif 'linear' in model_type_str_lower:
            return ModelType.LINEAR_MODEL
        elif 'ensemble' in model_type_str_lower:
            return ModelType.ENSEMBLE_MODEL
        elif 'time' in model_type_str_lower or 'series' in model_type_str_lower:
            return ModelType.TIME_SERIES
        elif 'regime' in model_type_str_lower:
            return ModelType.REGIME_DETECTION
        elif 'mamba' in model_type_str_lower:
            return ModelType.ADVANCED_MAMBA_HYBRID
        elif 'resnet' in model_type_str_lower:
            return ModelType.FINANCIAL_RESNET
        elif 'scaler' in model_type_str_lower:
            return ModelType.DEEP_SCALER
        elif 'nbeats' in model_type_str_lower:
            return ModelType.NBEATS
        
        # Default fallback
        return ModelType.DEFAULT
    
    def create_custom_profile(
        self,
        model_type: str,
        target_features: int,
        min_features: int,
        max_features: int,
        **kwargs
    ) -> ModelProfile:
        """Create a custom model profile."""
        tprint_info(f"🔧 Creating custom profile for {model_type}")
        
        try:
            # Create custom model type
            custom_model_type = ModelType(f"custom_{model_type}")
            
            # Create profile with provided parameters
            profile = ModelProfile(
                model_type=custom_model_type,
                target_features=target_features,
                min_features=min_features,
                max_features=max_features,
                **kwargs
            )
            
            # Store the profile
            self.profiles[custom_model_type] = profile
            
            tprint_success(f"   ✅ Custom profile created: {target_features} features")
            return profile
            
        except Exception as e:
            tprint_warning(f"   ⚠️ Failed to create custom profile: {e}")
            return self.profiles[ModelType.DEFAULT]
    
    def update_profile(self, model_type: Union[ModelType, str], **kwargs) -> bool:
        """Update an existing model profile."""
        tprint_info(f"🔧 Updating profile for {model_type}")
        
        try:
            profile = self.get_profile(model_type)
            if not profile:
                return False
            
            # Update profile attributes
            for key, value in kwargs.items():
                if hasattr(profile, key):
                    setattr(profile, key, value)
                    tprint_debug(f"   🔧 Updated {key} = {value}")
            
            tprint_success(f"   ✅ Profile updated for {model_type}")
            return True
            
        except Exception as e:
            tprint_warning(f"   ⚠️ Failed to update profile: {e}")
            return False
    
    def get_available_model_types(self) -> List[str]:
        """Get list of available model types."""
        return [model_type.value for model_type in self.profiles.keys()]
    
    def get_profile_summary(self, model_type: Union[ModelType, str]) -> Dict[str, Any]:
        """Get summary of a model profile."""
        profile = self.get_profile(model_type)
        if not profile:
            return {}
        
        return {
            'model_type': profile.model_type.value,
            'target_features': profile.target_features,
            'min_features': profile.min_features,
            'max_features': profile.max_features,
            'vif_threshold': profile.vif_threshold,
            'correlation_threshold': profile.correlation_threshold,
            'priority_categories': profile.priority_categories,
            'prefer_uncorrelated_features': profile.prefer_uncorrelated_features,
            'prefer_interpretable_features': profile.prefer_interpretable_features,
            'prefer_interaction_features': profile.prefer_interaction_features,
            'enable_parallel_processing': profile.enable_parallel_processing,
            'memory_efficient': profile.memory_efficient,
            'chunk_size': profile.chunk_size,
            'enable_regime_aware_selection': profile.enable_regime_aware_selection,
            'enable_temporal_filtering': profile.enable_temporal_filtering,
            'enable_precision_filtering': profile.enable_precision_filtering,
            'model_specific_params': profile.model_specific_params
        }
    
    def export_profiles(self, output_path: str):
        """Export all profiles to a file."""
        tprint_info(f"📁 Exporting profiles to {output_path}")
        
        try:
            import json
            
            profiles_data = {}
            for model_type, profile in self.profiles.items():
                profiles_data[model_type.value] = self.get_profile_summary(model_type)
            
            with open(output_path, 'w') as f:
                json.dump(profiles_data, f, indent=2)
            
            tprint_success(f"   ✅ Exported {len(profiles_data)} profiles")
            
        except Exception as e:
            tprint_warning(f"   ⚠️ Failed to export profiles: {e}")
    
    def import_profiles(self, input_path: str):
        """Import profiles from a file."""
        tprint_info(f"📁 Importing profiles from {input_path}")
        
        try:
            import json
            
            with open(input_path, 'r') as f:
                profiles_data = json.load(f)
            
            imported_count = 0
            for model_type_str, profile_data in profiles_data.items():
                try:
                    # Create custom profile
                    profile = ModelProfile(
                        model_type=ModelType(f"imported_{model_type_str}"),
                        target_features=profile_data['target_features'],
                        min_features=profile_data['min_features'],
                        max_features=profile_data['max_features'],
                        vif_threshold=profile_data.get('vif_threshold', 10.0),
                        correlation_threshold=profile_data.get('correlation_threshold', 0.95),
                        priority_categories=profile_data.get('priority_categories', ['momentum', 'volatility']),
                        prefer_uncorrelated_features=profile_data.get('prefer_uncorrelated_features', True),
                        prefer_interpretable_features=profile_data.get('prefer_interpretable_features', False),
                        prefer_interaction_features=profile_data.get('prefer_interaction_features', False),
                        enable_parallel_processing=profile_data.get('enable_parallel_processing', True),
                        memory_efficient=profile_data.get('memory_efficient', True),
                        chunk_size=profile_data.get('chunk_size', 1000),
                        enable_regime_aware_selection=profile_data.get('enable_regime_aware_selection', False),
                        enable_temporal_filtering=profile_data.get('enable_temporal_filtering', False),
                        enable_precision_filtering=profile_data.get('enable_precision_filtering', False),
                        model_specific_params=profile_data.get('model_specific_params', {})
                    )
                    
                    self.profiles[profile.model_type] = profile
                    imported_count += 1
                    
                except Exception as e:
                    tprint_warning(f"   ⚠️ Failed to import profile {model_type_str}: {e}")
            
            tprint_success(f"   ✅ Imported {imported_count} profiles")
            
        except Exception as e:
            tprint_warning(f"   ⚠️ Failed to import profiles: {e}")