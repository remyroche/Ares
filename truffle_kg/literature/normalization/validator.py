"""
Data Normalization and Validation Pipeline
Normalizes units, validates data quality, and performs sanity checks
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import json
import numpy as np
import pandas as pd
from datetime import datetime
import unicodedata

# Scientific units and validation
import pint
from pint import UnitRegistry
import scipy.stats as stats

# Data validation
from cerberus import Validator
import jsonschema

logger = logging.getLogger(__name__)

# Initialize unit registry
ureg = UnitRegistry()

@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    confidence: float
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class NormalizedParameter:
    """A normalized parameter with validation"""
    parameter_name: str
    original_value: Union[float, str]
    original_unit: str
    normalized_value: float
    normalized_unit: str
    confidence: float
    validation_result: ValidationResult
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class NormalizedExperiment:
    """A normalized experiment with validation"""
    experiment_id: str
    paper_id: str
    fungus_taxon_id: Optional[str]
    fungus_name: str
    host_taxon_id: Optional[str]
    host_name: str
    inoculum_form: str
    plant_age_d: Optional[int]
    chamber_type: str
    flow_regime: str
    volume_L: Optional[float]
    duration_d: Optional[int]
    replicates: Optional[int]
    colonization_pct: Optional[float]
    time_to_colonization_d: Optional[int]
    fruiting: Optional[bool]
    yield_g: Optional[float]
    notes: str
    confidence_0_1: float
    parameters: List[NormalizedParameter] = field(default_factory=list)
    environment: Dict[str, Any] = field(default_factory=dict)
    nutrient_recipe: Dict[str, Any] = field(default_factory=dict)
    validation_result: ValidationResult = field(default_factory=lambda: ValidationResult(True, 1.0))

class UnitNormalizer:
    """Normalize units to standard forms"""
    
    def __init__(self):
        self.unit_mappings = {
            'ph': {
                'target_unit': 'dimensionless',
                'conversion_factors': {
                    '': 1.0,
                    'pH': 1.0,
                    'ph': 1.0
                }
            },
            'ec': {
                'target_unit': 'mS/cm',
                'conversion_factors': {
                    'mS/cm': 1.0,
                    'mS cm-1': 1.0,
                    'mS/cm': 1.0,
                    'µS/cm': 0.001,
                    'µS cm-1': 0.001,
                    'dS/m': 0.1,
                    'dS m-1': 0.1
                }
            },
            'temperature': {
                'target_unit': '°C',
                'conversion_factors': {
                    '°C': 1.0,
                    'C': 1.0,
                    '°F': lambda x: (x - 32) * 5/9,
                    'F': lambda x: (x - 32) * 5/9,
                    'K': lambda x: x - 273.15
                }
            },
            'dissolved_oxygen': {
                'target_unit': 'mg/L',
                'conversion_factors': {
                    'mg/L': 1.0,
                    'mg L-1': 1.0,
                    'ppm': 1.0,  # Assume mg/L for DO
                    'µM': 0.032,  # Approximate conversion
                    'mM': 32.0
                }
            },
            'co2': {
                'target_unit': 'ppm',
                'conversion_factors': {
                    'ppm': 1.0,
                    'ppb': 0.001,
                    'µL/L': 1.0,
                    'mL/L': 1000.0
                }
            },
            'photoperiod': {
                'target_unit': 'h',
                'conversion_factors': {
                    'h': 1.0,
                    'hours': 1.0,
                    'hrs': 1.0,
                    'min': 1/60,
                    'minutes': 1/60
                }
            },
            'ppfd': {
                'target_unit': 'µmol m⁻² s⁻¹',
                'conversion_factors': {
                    'µmol m⁻² s⁻¹': 1.0,
                    'µmol/m²/s': 1.0,
                    'µmol m-2 s-1': 1.0,
                    'µE m⁻² s⁻¹': 1.0,  # Approximate
                    'µE/m²/s': 1.0
                }
            },
            'volume': {
                'target_unit': 'L',
                'conversion_factors': {
                    'L': 1.0,
                    'liters': 1.0,
                    'mL': 0.001,
                    'milliliters': 0.001,
                    'µL': 0.000001,
                    'microliters': 0.000001
                }
            },
            'duration': {
                'target_unit': 'days',
                'conversion_factors': {
                    'days': 1.0,
                    'weeks': 7.0,
                    'months': 30.0,
                    'years': 365.0,
                    'h': 1/24,
                    'hours': 1/24
                }
            },
            'colonization': {
                'target_unit': '%',
                'conversion_factors': {
                    '%': 1.0,
                    'percent': 1.0,
                    'fraction': 100.0,
                    'ratio': 100.0
                }
            },
            'yield': {
                'target_unit': 'g',
                'conversion_factors': {
                    'g': 1.0,
                    'grams': 1.0,
                    'mg': 0.001,
                    'milligrams': 0.001,
                    'kg': 1000.0,
                    'kilograms': 1000.0
                }
            }
        }
        
        # Salt to ion conversion factors
        self.salt_to_ion = {
            'KNO3': {'K': 0.387, 'NO3-N': 0.139},
            'NH4NO3': {'NH4-N': 0.35, 'NO3-N': 0.35},
            'KH2PO4': {'K': 0.287, 'PO4-P': 0.228},
            'K2HPO4': {'K': 0.449, 'PO4-P': 0.184},
            'Ca(NO3)2': {'Ca': 0.244, 'NO3-N': 0.118},
            'MgSO4': {'Mg': 0.201, 'SO4-S': 0.133},
            'Fe-EDTA': {'Fe': 0.131},
            'FeSO4': {'Fe': 0.368, 'SO4-S': 0.211},
            'MnSO4': {'Mn': 0.325, 'SO4-S': 0.171},
            'ZnSO4': {'Zn': 0.405, 'SO4-S': 0.197},
            'CuSO4': {'Cu': 0.398, 'SO4-S': 0.201},
            'H3BO3': {'B': 0.175},
            'Na2MoO4': {'Mo': 0.396}
        }
    
    def normalize_parameter(self, param_name: str, value: float, unit: str) -> Tuple[float, str, float]:
        """Normalize a parameter to standard units"""
        param_name_lower = param_name.lower()
        
        if param_name_lower not in self.unit_mappings:
            return value, unit, 0.5  # Unknown parameter, low confidence
        
        mapping = self.unit_mappings[param_name_lower]
        target_unit = mapping['target_unit']
        conversion_factors = mapping['conversion_factors']
        
        # Find best matching unit
        best_match = None
        best_confidence = 0.0
        
        for unit_key, factor in conversion_factors.items():
            if self._units_match(unit, unit_key):
                best_match = unit_key
                best_confidence = 0.9
                break
            elif self._units_similar(unit, unit_key):
                best_match = unit_key
                best_confidence = 0.7
                break
        
        if best_match is None:
            # No match found, return original with low confidence
            return value, unit, 0.3
        
        # Apply conversion
        if callable(conversion_factors[best_match]):
            normalized_value = conversion_factors[best_match](value)
        else:
            normalized_value = value * conversion_factors[best_match]
        
        return normalized_value, target_unit, best_confidence
    
    def _units_match(self, unit1: str, unit2: str) -> bool:
        """Check if units match exactly"""
        return unit1.lower().strip() == unit2.lower().strip()
    
    def _units_similar(self, unit1: str, unit2: str) -> bool:
        """Check if units are similar (fuzzy matching)"""
        unit1_clean = re.sub(r'[^\w]', '', unit1.lower())
        unit2_clean = re.sub(r'[^\w]', '', unit2.lower())
        
        # Check if one contains the other
        return unit1_clean in unit2_clean or unit2_clean in unit1_clean
    
    def convert_salt_to_ions(self, salt_name: str, concentration: float) -> Dict[str, float]:
        """Convert salt concentration to ion concentrations"""
        salt_name_clean = salt_name.replace(' ', '').replace('-', '')
        
        if salt_name_clean in self.salt_to_ion:
            ions = {}
            for ion, factor in self.salt_to_ion[salt_name_clean].items():
                ions[ion] = concentration * factor
            return ions
        else:
            logger.warning(f"Unknown salt: {salt_name}")
            return {}

class DataValidator:
    """Validate data quality and perform sanity checks"""
    
    def __init__(self):
        self.validator = Validator()
        self._setup_validation_rules()
    
    def _setup_validation_rules(self):
        """Setup validation rules for different parameter types"""
        self.validation_rules = {
            'ph': {
                'min': 3.0,
                'max': 9.0,
                'typical_range': (5.0, 7.0),
                'warning_range': (4.0, 8.0)
            },
            'ec': {
                'min': 0.1,
                'max': 10.0,
                'typical_range': (0.5, 3.0),
                'warning_range': (0.2, 5.0)
            },
            'temperature': {
                'min': 0.0,
                'max': 50.0,
                'typical_range': (15.0, 30.0),
                'warning_range': (5.0, 40.0)
            },
            'dissolved_oxygen': {
                'min': 0.0,
                'max': 20.0,
                'typical_range': (5.0, 12.0),
                'warning_range': (2.0, 15.0)
            },
            'co2': {
                'min': 0.0,
                'max': 10000.0,
                'typical_range': (300.0, 1000.0),
                'warning_range': (200.0, 2000.0)
            },
            'photoperiod': {
                'min': 0.0,
                'max': 24.0,
                'typical_range': (8.0, 16.0),
                'warning_range': (4.0, 20.0)
            },
            'ppfd': {
                'min': 0.0,
                'max': 2000.0,
                'typical_range': (100.0, 800.0),
                'warning_range': (50.0, 1200.0)
            },
            'volume': {
                'min': 0.001,
                'max': 1000.0,
                'typical_range': (0.1, 10.0),
                'warning_range': (0.01, 100.0)
            },
            'duration': {
                'min': 1.0,
                'max': 3650.0,  # 10 years
                'typical_range': (7.0, 365.0),
                'warning_range': (1.0, 730.0)
            },
            'colonization': {
                'min': 0.0,
                'max': 100.0,
                'typical_range': (20.0, 90.0),
                'warning_range': (0.0, 100.0)
            },
            'yield': {
                'min': 0.0,
                'max': 1000.0,
                'typical_range': (0.1, 100.0),
                'warning_range': (0.0, 500.0)
            }
        }
    
    def validate_parameter(self, param_name: str, value: float, unit: str) -> ValidationResult:
        """Validate a single parameter"""
        param_name_lower = param_name.lower()
        
        if param_name_lower not in self.validation_rules:
            return ValidationResult(True, 0.5, warnings=[f"Unknown parameter: {param_name}"])
        
        rules = self.validation_rules[param_name_lower]
        errors = []
        warnings = []
        suggestions = []
        
        # Check if value is within acceptable range
        if value < rules['min'] or value > rules['max']:
            errors.append(f"{param_name} value {value} {unit} is outside acceptable range ({rules['min']}-{rules['max']})")
            return ValidationResult(False, 0.0, errors=errors)
        
        # Check if value is within typical range
        if not (rules['typical_range'][0] <= value <= rules['typical_range'][1]):
            if rules['warning_range'][0] <= value <= rules['warning_range'][1]:
                warnings.append(f"{param_name} value {value} {unit} is outside typical range ({rules['typical_range'][0]}-{rules['typical_range'][1]})")
            else:
                errors.append(f"{param_name} value {value} {unit} is outside warning range ({rules['warning_range'][0]}-{rules['warning_range'][1]})")
                return ValidationResult(False, 0.3, errors=errors)
        
        # Calculate confidence based on how close to typical range
        typical_min, typical_max = rules['typical_range']
        if typical_min <= value <= typical_max:
            confidence = 1.0
        else:
            # Calculate distance from typical range
            if value < typical_min:
                distance = typical_min - value
                max_distance = typical_min - rules['warning_range'][0]
            else:
                distance = value - typical_max
                max_distance = rules['warning_range'][1] - typical_max
            
            confidence = max(0.5, 1.0 - (distance / max_distance))
        
        return ValidationResult(True, confidence, warnings=warnings, suggestions=suggestions)
    
    def validate_experiment(self, experiment) -> ValidationResult:
        """Validate a complete experiment"""
        errors = []
        warnings = []
        suggestions = []
        
        # Check required fields
        if not experiment.fungus_name or experiment.fungus_name == 'Unknown':
            errors.append("Fungus name is required")
        
        if not experiment.host_name or experiment.host_name == 'Unknown':
            errors.append("Host name is required")
        
        # Check parameter consistency
        if experiment.parameters:
            # Check for pH and EC consistency
            ph_params = [p for p in experiment.parameters if p.parameter_name.lower() == 'ph']
            ec_params = [p for p in experiment.parameters if p.parameter_name.lower() == 'ec']
            
            if ph_params and ec_params:
                ph_value = ph_params[0].normalized_value
                ec_value = ec_params[0].normalized_value
                
                # Check if EC is reasonable for pH
                if ph_value < 6.0 and ec_value > 2.0:
                    warnings.append("High EC ({} mS/cm) with low pH ({}) may indicate nutrient imbalance".format(ec_value, ph_value))
        
        # Check for missing key parameters
        param_names = [p.parameter_name.lower() for p in experiment.parameters]
        key_params = ['ph', 'ec', 'temperature']
        missing_params = [p for p in key_params if p not in param_names]
        
        if missing_params:
            warnings.append("Missing key parameters: {}".format(', '.join(missing_params)))
        
        # Check experiment duration vs colonization
        if experiment.duration_d and experiment.colonization_pct:
            if experiment.duration_d < 30 and experiment.colonization_pct > 80:
                warnings.append("High colonization rate ({}) in short duration ({}) may be unrealistic".format(
                    experiment.colonization_pct, experiment.duration_d))
        
        # Calculate overall confidence
        if errors:
            confidence = 0.0
        elif warnings:
            confidence = 0.7
        else:
            confidence = 1.0
        
        return ValidationResult(len(errors) == 0, confidence, errors=errors, warnings=warnings, suggestions=suggestions)
    
    def validate_nutrient_recipe(self, recipe: Dict[str, Any]) -> ValidationResult:
        """Validate nutrient recipe"""
        errors = []
        warnings = []
        
        # Check if recipe has required nutrients
        required_nutrients = ['NO3-N', 'NH4-N', 'PO4-P', 'K', 'Ca', 'Mg']
        present_nutrients = [k for k in recipe.keys() if k in required_nutrients]
        
        if len(present_nutrients) < 3:
            errors.append("Nutrient recipe should contain at least 3 major nutrients")
        
        # Check for reasonable nutrient ratios
        if 'NO3-N' in recipe and 'NH4-N' in recipe:
            no3_nh4_ratio = recipe['NO3-N'] / recipe['NH4-N'] if recipe['NH4-N'] > 0 else float('inf')
            if no3_nh4_ratio < 0.5 or no3_nh4_ratio > 10.0:
                warnings.append("NO3-N/NH4-N ratio ({:.2f}) is outside typical range (0.5-10.0)".format(no3_nh4_ratio))
        
        # Check for micronutrients
        micronutrients = ['Fe', 'Mn', 'Zn', 'Cu', 'B', 'Mo']
        present_micronutrients = [k for k in recipe.keys() if k in micronutrients]
        
        if len(present_micronutrients) < 2:
            warnings.append("Nutrient recipe should contain micronutrients")
        
        confidence = 1.0 if not errors else 0.5
        return ValidationResult(len(errors) == 0, confidence, errors=errors, warnings=warnings)

class DataNormalizer:
    """Main data normalization and validation class"""
    
    def __init__(self):
        self.unit_normalizer = UnitNormalizer()
        self.validator = DataValidator()
    
    def normalize_parameter(self, param_name: str, value: float, unit: str) -> NormalizedParameter:
        """Normalize a parameter and validate it"""
        # Normalize units
        normalized_value, normalized_unit, unit_confidence = self.unit_normalizer.normalize_parameter(
            param_name, value, unit
        )
        
        # Validate parameter
        validation_result = self.validator.validate_parameter(param_name, normalized_value, normalized_unit)
        
        # Calculate overall confidence
        overall_confidence = (unit_confidence + validation_result.confidence) / 2
        
        return NormalizedParameter(
            parameter_name=param_name,
            original_value=value,
            original_unit=unit,
            normalized_value=normalized_value,
            normalized_unit=normalized_unit,
            confidence=overall_confidence,
            validation_result=validation_result,
            metadata={
                'unit_confidence': unit_confidence,
                'validation_confidence': validation_result.confidence
            }
        )
    
    def normalize_experiment(self, experiment) -> NormalizedExperiment:
        """Normalize a complete experiment"""
        # Normalize parameters
        normalized_params = []
        for param in experiment.parameters:
            normalized_param = self.normalize_parameter(
                param.parameter_name,
                param.value,
                param.unit
            )
            normalized_params.append(normalized_param)
        
        # Normalize basic experiment fields
        normalized_volume = None
        if experiment.volume_L:
            normalized_volume = self.unit_normalizer.normalize_parameter('volume', experiment.volume_L, 'L')[0]
        
        normalized_duration = None
        if experiment.duration_d:
            normalized_duration = self.unit_normalizer.normalize_parameter('duration', experiment.duration_d, 'days')[0]
        
        normalized_yield = None
        if experiment.yield_g:
            normalized_yield = self.unit_normalizer.normalize_parameter('yield', experiment.yield_g, 'g')[0]
        
        # Create normalized experiment
        normalized_experiment = NormalizedExperiment(
            experiment_id=experiment.experiment_id,
            paper_id=experiment.paper_id,
            fungus_taxon_id=experiment.fungus_taxon_id,
            fungus_name=experiment.fungus_name,
            host_taxon_id=experiment.host_taxon_id,
            host_name=experiment.host_name,
            inoculum_form=experiment.inoculum_form,
            plant_age_d=experiment.plant_age_d,
            chamber_type=experiment.chamber_type,
            flow_regime=experiment.flow_regime,
            volume_L=normalized_volume,
            duration_d=normalized_duration,
            replicates=experiment.replicates,
            colonization_pct=experiment.colonization_pct,
            time_to_colonization_d=experiment.time_to_colonization_d,
            fruiting=experiment.fruiting,
            yield_g=normalized_yield,
            notes=experiment.notes,
            confidence_0_1=experiment.confidence_0_1,
            parameters=normalized_params
        )
        
        # Validate experiment
        validation_result = self.validator.validate_experiment(normalized_experiment)
        normalized_experiment.validation_result = validation_result
        
        # Update confidence based on validation
        if validation_result.confidence < normalized_experiment.confidence_0_1:
            normalized_experiment.confidence_0_1 = validation_result.confidence
        
        return normalized_experiment
    
    def deduplicate_experiments(self, experiments: List[NormalizedExperiment]) -> List[NormalizedExperiment]:
        """Remove duplicate experiments"""
        unique_experiments = []
        seen_experiments = set()
        
        for exp in experiments:
            # Create a signature for the experiment
            signature = (
                exp.fungus_name,
                exp.host_name,
                exp.chamber_type,
                exp.flow_regime,
                exp.volume_L,
                exp.duration_d
            )
            
            if signature not in seen_experiments:
                seen_experiments.add(signature)
                unique_experiments.append(exp)
            else:
                # Keep the one with higher confidence
                for i, existing_exp in enumerate(unique_experiments):
                    if (existing_exp.fungus_name, existing_exp.host_name, existing_exp.chamber_type,
                        existing_exp.flow_regime, existing_exp.volume_L, existing_exp.duration_d) == signature:
                        if exp.confidence_0_1 > existing_exp.confidence_0_1:
                            unique_experiments[i] = exp
                        break
        
        return unique_experiments
    
    def generate_quality_report(self, experiments: List[NormalizedExperiment]) -> Dict[str, Any]:
        """Generate a quality report for normalized experiments"""
        if not experiments:
            return {'error': 'No experiments to analyze'}
        
        # Basic statistics
        total_experiments = len(experiments)
        valid_experiments = sum(1 for exp in experiments if exp.validation_result.is_valid)
        high_confidence_experiments = sum(1 for exp in experiments if exp.confidence_0_1 > 0.8)
        
        # Parameter coverage
        all_params = set()
        for exp in experiments:
            for param in exp.parameters:
                all_params.add(param.parameter_name)
        
        param_coverage = {}
        for param_name in all_params:
            param_experiments = sum(1 for exp in experiments 
                                  if any(p.parameter_name == param_name for p in exp.parameters))
            param_coverage[param_name] = {
                'count': param_experiments,
                'percentage': (param_experiments / total_experiments) * 100
            }
        
        # Confidence distribution
        confidences = [exp.confidence_0_1 for exp in experiments]
        confidence_stats = {
            'mean': np.mean(confidences),
            'std': np.std(confidences),
            'min': np.min(confidences),
            'max': np.max(confidences),
            'median': np.median(confidences)
        }
        
        # Validation issues
        validation_issues = {
            'errors': [],
            'warnings': [],
            'suggestions': []
        }
        
        for exp in experiments:
            validation_issues['errors'].extend(exp.validation_result.errors)
            validation_issues['warnings'].extend(exp.validation_result.warnings)
            validation_issues['suggestions'].extend(exp.validation_result.suggestions)
        
        # Count unique issues
        validation_issues['error_counts'] = {error: validation_issues['errors'].count(error) 
                                           for error in set(validation_issues['errors'])}
        validation_issues['warning_counts'] = {warning: validation_issues['warnings'].count(warning) 
                                             for warning in set(validation_issues['warnings'])}
        
        return {
            'total_experiments': total_experiments,
            'valid_experiments': valid_experiments,
            'high_confidence_experiments': high_confidence_experiments,
            'validation_rate': (valid_experiments / total_experiments) * 100,
            'high_confidence_rate': (high_confidence_experiments / total_experiments) * 100,
            'parameter_coverage': param_coverage,
            'confidence_statistics': confidence_stats,
            'validation_issues': validation_issues
        }

def main():
    """Example usage of the data normalizer"""
    normalizer = DataNormalizer()
    
    # Example parameter normalization
    ph_param = normalizer.normalize_parameter('ph', 6.2, 'pH')
    print(f"pH: {ph_param.original_value} {ph_param.original_unit} -> {ph_param.normalized_value} {ph_param.normalized_unit}")
    print(f"Confidence: {ph_param.confidence:.2f}")
    print(f"Valid: {ph_param.validation_result.is_valid}")
    
    # Example EC normalization
    ec_param = normalizer.normalize_parameter('ec', 1.5, 'mS/cm')
    print(f"EC: {ec_param.original_value} {ec_param.original_unit} -> {ec_param.normalized_value} {ec_param.normalized_unit}")
    print(f"Confidence: {ec_param.confidence:.2f}")
    print(f"Valid: {ec_param.validation_result.is_valid}")

if __name__ == "__main__":
    main()