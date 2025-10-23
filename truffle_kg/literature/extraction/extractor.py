"""
Data Extraction Pipeline
Hybrid rules + ML for extracting parameters from literature
"""

import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import json
import numpy as np
import pandas as pd
from datetime import datetime
import unicodedata

# ML and NLP
import spacy
from spacy.matcher import Matcher
from spacy.tokens import Span
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Table processing
import camelot
import pdfplumber

# Scientific units
import pint
from pint import UnitRegistry

logger = logging.getLogger(__name__)

# Initialize unit registry
ureg = UnitRegistry()

@dataclass
class ExtractedParameter:
    """A parameter extracted from literature"""
    parameter_name: str
    value: Union[float, str]
    unit: str
    confidence: float
    source_type: str  # table, text, caption
    source_location: str  # page, section, table_id
    text_span: str
    normalized_value: Optional[float] = None
    normalized_unit: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExtractedExperiment:
    """An experiment extracted from literature"""
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
    parameters: List[ExtractedParameter] = field(default_factory=list)
    environment: Dict[str, Any] = field(default_factory=dict)
    nutrient_recipe: Dict[str, Any] = field(default_factory=dict)

class ScientificNERExtractor:
    """Scientific Named Entity Recognition for truffle cultivation parameters"""
    
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found, installing...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
        # Add custom patterns
        self._setup_patterns()
        
        # Initialize matcher
        self.matcher = Matcher(self.nlp.vocab)
        self._setup_matcher()
        
        # Chemical and species dictionaries
        self._load_dictionaries()
    
    def _setup_patterns(self):
        """Setup regex patterns for parameter extraction"""
        self.patterns = {
            'ph': [
                r'\bpH\s*(?:=|of|was)?\s*(\d\.\d|\d)',
                r'pH\s*(\d\.\d|\d)',
                r'(\d\.\d|\d)\s*pH'
            ],
            'ec': [
                r'\b(?:EC|electrical conductivity)\s*(?:=|of)?\s*([0-9]\.?[0-9]?)\s*(?:mS/?cm|mS cm-1)',
                r'EC\s*([0-9]\.?[0-9]?)\s*(?:mS/?cm|mS cm-1)',
                r'([0-9]\.?[0-9]?)\s*(?:mS/?cm|mS cm-1)'
            ],
            'temperature': [
                r'(\d{1,2}\.?\d?)\s*°\s?C',
                r'temperature\s*(?:=|of)?\s*(\d{1,2}\.?\d?)\s*°\s?C',
                r'(\d{1,2}\.?\d?)\s*°C'
            ],
            'dissolved_oxygen': [
                r'(?:dissolved oxygen|DO)\s*(?:=|of)?\s*(\d{1,2}\.?\d?)\s*mg/?L',
                r'DO\s*(\d{1,2}\.?\d?)\s*mg/?L',
                r'(\d{1,2}\.?\d?)\s*mg/?L\s*(?:dissolved oxygen|DO)'
            ],
            'co2': [
                r'CO₂?\s*(?:=|of)?\s*(\d{1,4}\.?\d?)\s*ppm',
                r'(\d{1,4}\.?\d?)\s*ppm\s*CO₂?',
                r'carbon dioxide\s*(?:=|of)?\s*(\d{1,4}\.?\d?)\s*ppm'
            ],
            'photoperiod': [
                r'photoperiod\s*(?:=|of)?\s*(\d{1,2}\.?\d?)\s*h(?:ours?)?',
                r'(\d{1,2}\.?\d?)\s*h(?:ours?)?\s*photoperiod',
                r'light\s*(?:period|cycle)\s*(?:=|of)?\s*(\d{1,2}\.?\d?)\s*h(?:ours?)?'
            ],
            'ppfd': [
                r'PPFD\s*(?:=|of)?\s*(\d{1,4}\.?\d?)\s*µmol\s*m⁻²\s*s⁻¹',
                r'(\d{1,4}\.?\d?)\s*µmol\s*m⁻²\s*s⁻¹\s*PPFD',
                r'photosynthetic\s*photon\s*flux\s*density\s*(?:=|of)?\s*(\d{1,4}\.?\d?)\s*µmol\s*m⁻²\s*s⁻¹'
            ],
            'nutrient': [
                r'(NO3|NH4|PO4|K|Ca|Mg|Fe(?:-EDTA)?)\s*[:=]?\s*([0-9]+\.?[0-9]*)\s*(mg/L|µM|mM)',
                r'([0-9]+\.?[0-9]*)\s*(mg/L|µM|mM)\s*(NO3|NH4|PO4|K|Ca|Mg|Fe(?:-EDTA)?)',
                r'(nitrate|ammonium|phosphate|potassium|calcium|magnesium|iron)\s*(?:=|of)?\s*([0-9]+\.?[0-9]*)\s*(mg/L|µM|mM)'
            ],
            'volume': [
                r'volume\s*(?:=|of)?\s*([0-9]+\.?[0-9]*)\s*(?:L|mL|liters?|milliliters?)',
                r'([0-9]+\.?[0-9]*)\s*(?:L|mL|liters?|milliliters?)\s*volume',
                r'([0-9]+\.?[0-9]*)\s*(?:L|mL|liters?|milliliters?)'
            ],
            'duration': [
                r'duration\s*(?:=|of)?\s*([0-9]+\.?[0-9]*)\s*(?:days?|weeks?|months?)',
                r'([0-9]+\.?[0-9]*)\s*(?:days?|weeks?|months?)\s*duration',
                r'incubated?\s*(?:for)?\s*([0-9]+\.?[0-9]*)\s*(?:days?|weeks?|months?)'
            ],
            'replicates': [
                r'replicates?\s*(?:=|of)?\s*([0-9]+)',
                r'([0-9]+)\s*replicates?',
                r'n\s*=\s*([0-9]+)'
            ],
            'colonization': [
                r'colonization\s*(?:rate|percentage|%)\s*(?:=|of)?\s*([0-9]+\.?[0-9]*)\s*%',
                r'([0-9]+\.?[0-9]*)\s*%\s*colonization',
                r'colonized\s*(?:at|by)?\s*([0-9]+\.?[0-9]*)\s*%'
            ],
            'yield': [
                r'yield\s*(?:=|of)?\s*([0-9]+\.?[0-9]*)\s*(?:g|mg|kg)',
                r'([0-9]+\.?[0-9]*)\s*(?:g|mg|kg)\s*yield',
                r'produced?\s*(?:a)?\s*yield\s*(?:of)?\s*([0-9]+\.?[0-9]*)\s*(?:g|mg|kg)'
            ]
        }
    
    def _setup_matcher(self):
        """Setup spaCy matcher for entity recognition"""
        # Add patterns for different entity types
        patterns = {
            'FUNGUS': [
                [{'LOWER': 'tuber'}, {'LOWER': 'melanosporum'}],
                [{'LOWER': 'tuber'}, {'LOWER': 'magnatum'}],
                [{'LOWER': 'tuber'}, {'LOWER': 'borchii'}],
                [{'LOWER': 'tuber'}, {'LOWER': 'aestivum'}],
                [{'LOWER': 'truffle'}]
            ],
            'HOST': [
                [{'LOWER': 'quercus'}, {'LOWER': 'ilex'}],
                [{'LOWER': 'quercus'}, {'LOWER': 'petraea'}],
                [{'LOWER': 'corylus'}, {'LOWER': 'avellana'}],
                [{'LOWER': 'oak'}],
                [{'LOWER': 'hazel'}]
            ],
            'NUTRIENT': [
                [{'LOWER': 'nitrate'}],
                [{'LOWER': 'ammonium'}],
                [{'LOWER': 'phosphate'}],
                [{'LOWER': 'potassium'}],
                [{'LOWER': 'calcium'}],
                [{'LOWER': 'magnesium'}],
                [{'LOWER': 'iron'}],
                [{'LOWER': 'hoagland'}]
            ],
            'CHEMICAL': [
                [{'LOWER': 'ph'}],
                [{'LOWER': 'ec'}],
                [{'LOWER': 'temperature'}],
                [{'LOWER': 'oxygen'}],
                [{'LOWER': 'co2'}],
                [{'LOWER': 'co₂'}]
            ]
        }
        
        for label, pattern_list in patterns.items():
            self.matcher.add(label, pattern_list)
    
    def _load_dictionaries(self):
        """Load chemical and species dictionaries"""
        self.chemical_dict = {
            'NO3': 'nitrate',
            'NH4': 'ammonium',
            'PO4': 'phosphate',
            'K': 'potassium',
            'Ca': 'calcium',
            'Mg': 'magnesium',
            'Fe': 'iron',
            'Fe-EDTA': 'iron-EDTA'
        }
        
        self.species_dict = {
            'tuber melanosporum': 'Tuber melanosporum',
            'tuber magnatum': 'Tuber magnatum',
            'tuber borchii': 'Tuber borchii',
            'tuber aestivum': 'Tuber aestivum',
            'quercus ilex': 'Quercus ilex',
            'quercus petraea': 'Quercus petraea',
            'corylus avellana': 'Corylus avellana'
        }
    
    def extract_parameters(self, text: str, source_type: str = 'text', source_location: str = '') -> List[ExtractedParameter]:
        """Extract parameters from text using pattern matching"""
        parameters = []
        
        for param_name, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    try:
                        value = float(match.group(1))
                        unit = self._extract_unit(text, match.start(), match.end())
                        confidence = self._calculate_confidence(param_name, value, unit, text, match.start())
                        
                        parameter = ExtractedParameter(
                            parameter_name=param_name,
                            value=value,
                            unit=unit,
                            confidence=confidence,
                            source_type=source_type,
                            source_location=source_location,
                            text_span=match.group(0),
                            normalized_value=self._normalize_value(param_name, value, unit),
                            normalized_unit=self._normalize_unit(param_name, unit),
                            metadata={'pattern': pattern, 'match_start': match.start(), 'match_end': match.end()}
                        )
                        parameters.append(parameter)
                        
                    except (ValueError, IndexError) as e:
                        logger.debug(f"Error parsing parameter {param_name}: {e}")
                        continue
        
        return parameters
    
    def _extract_unit(self, text: str, start: int, end: int) -> str:
        """Extract unit from text around the match"""
        # Look for unit in a window around the match
        window_start = max(0, start - 20)
        window_end = min(len(text), end + 20)
        window_text = text[window_start:window_end]
        
        # Common unit patterns
        unit_patterns = [
            r'(mS/?cm|mS cm-1)',
            r'(mg/?L|mg L-1)',
            r'(µM|mM|M)',
            r'(°C|°F)',
            r'(ppm|ppb)',
            r'(h|hours?|hrs?)',
            r'(L|mL|liters?|milliliters?)',
            r'(days?|weeks?|months?)',
            r'(µmol m⁻² s⁻¹|µmol/m²/s)',
            r'(%)'
        ]
        
        for pattern in unit_patterns:
            match = re.search(pattern, window_text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return ''
    
    def _calculate_confidence(self, param_name: str, value: float, unit: str, text: str, position: int) -> float:
        """Calculate confidence score for extracted parameter"""
        confidence = 0.5  # Base confidence
        
        # Boost confidence for reasonable values
        if param_name == 'ph' and 3.0 <= value <= 9.0:
            confidence += 0.3
        elif param_name == 'ec' and 0.1 <= value <= 10.0:
            confidence += 0.3
        elif param_name == 'temperature' and 0 <= value <= 50:
            confidence += 0.3
        elif param_name == 'dissolved_oxygen' and 0 <= value <= 20:
            confidence += 0.3
        
        # Boost confidence if unit is present and correct
        if unit:
            confidence += 0.2
        
        # Boost confidence if in methods section
        if 'method' in text.lower()[:position]:
            confidence += 0.1
        
        # Boost confidence if near other parameters
        context = text[max(0, position-100):min(len(text), position+100)]
        param_count = sum(1 for p in self.patterns.keys() if re.search(self.patterns[p][0], context, re.IGNORECASE))
        if param_count > 1:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _normalize_value(self, param_name: str, value: float, unit: str) -> Optional[float]:
        """Normalize value to standard units"""
        try:
            if param_name == 'ph':
                return value  # pH is unitless
            elif param_name == 'ec':
                if unit in ['mS/cm', 'mS cm-1']:
                    return value
                elif unit in ['µS/cm', 'µS cm-1']:
                    return value / 1000
                else:
                    return value
            elif param_name == 'temperature':
                if unit in ['°C', 'C']:
                    return value
                elif unit in ['°F', 'F']:
                    return (value - 32) * 5/9
                else:
                    return value
            elif param_name == 'dissolved_oxygen':
                if unit in ['mg/L', 'mg L-1']:
                    return value
                elif unit in ['ppm']:
                    return value  # Assume mg/L for DO
                else:
                    return value
            elif param_name == 'co2':
                if unit in ['ppm', 'ppb']:
                    return value
                else:
                    return value
            elif param_name == 'photoperiod':
                if unit in ['h', 'hours', 'hrs']:
                    return value
                else:
                    return value
            elif param_name == 'ppfd':
                if unit in ['µmol m⁻² s⁻¹', 'µmol/m²/s']:
                    return value
                else:
                    return value
            elif param_name == 'volume':
                if unit in ['L', 'liters']:
                    return value
                elif unit in ['mL', 'milliliters']:
                    return value / 1000
                else:
                    return value
            elif param_name == 'duration':
                if unit in ['days']:
                    return value
                elif unit in ['weeks']:
                    return value * 7
                elif unit in ['months']:
                    return value * 30
                else:
                    return value
            elif param_name == 'colonization':
                if unit in ['%']:
                    return value
                else:
                    return value
            elif param_name == 'yield':
                if unit in ['g']:
                    return value
                elif unit in ['mg']:
                    return value / 1000
                elif unit in ['kg']:
                    return value * 1000
                else:
                    return value
            else:
                return value
        except Exception as e:
            logger.debug(f"Error normalizing value: {e}")
            return value
    
    def _normalize_unit(self, param_name: str, unit: str) -> str:
        """Normalize unit to standard form"""
        unit_mapping = {
            'ph': '',
            'ec': 'mS/cm',
            'temperature': '°C',
            'dissolved_oxygen': 'mg/L',
            'co2': 'ppm',
            'photoperiod': 'h',
            'ppfd': 'µmol m⁻² s⁻¹',
            'volume': 'L',
            'duration': 'days',
            'colonization': '%',
            'yield': 'g'
        }
        
        if param_name in unit_mapping:
            return unit_mapping[param_name]
        else:
            return unit
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities using spaCy"""
        doc = self.nlp(text)
        entities = []
        
        # Extract standard entities
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'confidence': 0.8  # Default confidence for spaCy entities
            })
        
        # Extract custom entities using matcher
        matches = self.matcher(doc)
        for match_id, start, end in matches:
            span = doc[start:end]
            entities.append({
                'text': span.text,
                'label': self.nlp.vocab.strings[match_id],
                'start': span.start_char,
                'end': span.end_char,
                'confidence': 0.9  # Higher confidence for custom patterns
            })
        
        return entities

class TableExtractor:
    """Extract parameters from tables"""
    
    def __init__(self):
        self.ner_extractor = ScientificNERExtractor()
    
    def extract_from_table(self, table_data: List[List[str]], table_id: str = '') -> List[ExtractedParameter]:
        """Extract parameters from table data"""
        parameters = []
        
        if not table_data or len(table_data) < 2:
            return parameters
        
        # Find parameter rows
        for row_idx, row in enumerate(table_data):
            for col_idx, cell in enumerate(row):
                if not cell or not isinstance(cell, str):
                    continue
                
                # Extract parameters from cell text
                cell_params = self.ner_extractor.extract_parameters(
                    cell, 
                    source_type='table', 
                    source_location=f"{table_id}_r{row_idx}_c{col_idx}"
                )
                
                # Add table context to parameters
                for param in cell_params:
                    param.metadata.update({
                        'table_id': table_id,
                        'row': row_idx,
                        'column': col_idx,
                        'cell_text': cell
                    })
                
                parameters.extend(cell_params)
        
        return parameters
    
    def identify_parameter_tables(self, table_data: List[List[str]]) -> bool:
        """Identify if table contains parameter data"""
        if not table_data or len(table_data) < 2:
            return False
        
        # Look for parameter keywords in headers
        header_keywords = [
            'ph', 'ec', 'temp', '°c', 'mg/l', 'hoagland', 'modified',
            'no3', 'nh4', 'po4', 'ca', 'mg', 'fe-edta', 'micronutrients',
            'photoperiod', 'par', 'ppfd', 'volume', 'duration', 'replicates'
        ]
        
        # Check first row (usually headers)
        header_text = ' '.join([str(cell).lower() for cell in table_data[0] if cell])
        
        keyword_count = sum(1 for keyword in header_keywords if keyword in header_text)
        return keyword_count >= 2  # At least 2 parameter keywords

class ExperimentExtractor:
    """Extract complete experiments from parsed documents"""
    
    def __init__(self):
        self.ner_extractor = ScientificNERExtractor()
        self.table_extractor = TableExtractor()
    
    def extract_experiments(self, parsed_doc) -> List[ExtractedExperiment]:
        """Extract experiments from parsed document"""
        experiments = []
        
        # Extract from methods section
        methods_sections = [s for s in parsed_doc.sections if s.section_type == 'methods']
        for section in methods_sections:
            exp = self._extract_experiment_from_section(section, parsed_doc.document_id)
            if exp:
                experiments.append(exp)
        
        # Extract from tables
        for table in parsed_doc.tables:
            if self.table_extractor.identify_parameter_tables(table['data']):
                exp = self._extract_experiment_from_table(table, parsed_doc.document_id)
                if exp:
                    experiments.append(exp)
        
        return experiments
    
    def _extract_experiment_from_section(self, section: DocumentSection, paper_id: str) -> Optional[ExtractedExperiment]:
        """Extract experiment from methods section"""
        # Extract parameters
        parameters = self.ner_extractor.extract_parameters(
            section.content, 
            source_type='text', 
            source_location=f"section_{section.section_type}"
        )
        
        if not parameters:
            return None
        
        # Extract basic experiment info
        fungus_name = self._extract_fungus_name(section.content)
        host_name = self._extract_host_name(section.content)
        inoculum_form = self._extract_inoculum_form(section.content)
        
        # Create experiment
        experiment = ExtractedExperiment(
            experiment_id=f"{paper_id}_exp_{len(parameters)}",
            paper_id=paper_id,
            fungus_taxon_id=None,
            fungus_name=fungus_name,
            host_taxon_id=None,
            host_name=host_name,
            inoculum_form=inoculum_form,
            plant_age_d=self._extract_plant_age(section.content),
            chamber_type=self._extract_chamber_type(section.content),
            flow_regime=self._extract_flow_regime(section.content),
            volume_L=self._extract_volume(section.content),
            duration_d=self._extract_duration(section.content),
            replicates=self._extract_replicates(section.content),
            colonization_pct=self._extract_colonization(section.content),
            time_to_colonization_d=self._extract_time_to_colonization(section.content),
            fruiting=self._extract_fruiting(section.content),
            yield_g=self._extract_yield(section.content),
            notes=section.content[:200] + "..." if len(section.content) > 200 else section.content,
            confidence_0_1=self._calculate_experiment_confidence(parameters),
            parameters=parameters
        )
        
        return experiment
    
    def _extract_experiment_from_table(self, table: Dict[str, Any], paper_id: str) -> Optional[ExtractedExperiment]:
        """Extract experiment from parameter table"""
        # Extract parameters from table
        parameters = self.table_extractor.extract_from_table(
            table['data'], 
            table.get('table_id', 'unknown')
        )
        
        if not parameters:
            return None
        
        # Extract basic info from table caption
        caption = table.get('caption', '')
        fungus_name = self._extract_fungus_name(caption)
        host_name = self._extract_host_name(caption)
        
        # Create experiment
        experiment = ExtractedExperiment(
            experiment_id=f"{paper_id}_table_{table.get('table_id', 'unknown')}",
            paper_id=paper_id,
            fungus_taxon_id=None,
            fungus_name=fungus_name,
            host_taxon_id=None,
            host_name=host_name,
            inoculum_form='unknown',
            plant_age_d=None,
            chamber_type='unknown',
            flow_regime='unknown',
            volume_L=None,
            duration_d=None,
            replicates=None,
            colonization_pct=None,
            time_to_colonization_d=None,
            fruiting=None,
            yield_g=None,
            notes=caption,
            confidence_0_1=self._calculate_experiment_confidence(parameters),
            parameters=parameters
        )
        
        return experiment
    
    def _extract_fungus_name(self, text: str) -> str:
        """Extract fungus name from text"""
        # Look for Tuber species
        tuber_pattern = r'(Tuber\s+\w+)'
        match = re.search(tuber_pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
        
        # Look for truffle
        if 'truffle' in text.lower():
            return 'Truffle'
        
        return 'Unknown'
    
    def _extract_host_name(self, text: str) -> str:
        """Extract host name from text"""
        # Look for Quercus species
        quercus_pattern = r'(Quercus\s+\w+)'
        match = re.search(quercus_pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
        
        # Look for Corylus species
        corylus_pattern = r'(Corylus\s+\w+)'
        match = re.search(corylus_pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
        
        # Look for common names
        if 'oak' in text.lower():
            return 'Oak'
        elif 'hazel' in text.lower():
            return 'Hazel'
        
        return 'Unknown'
    
    def _extract_inoculum_form(self, text: str) -> str:
        """Extract inoculum form from text"""
        text_lower = text.lower()
        
        if 'spore' in text_lower:
            return 'spore'
        elif 'mycelium' in text_lower:
            return 'mycelium'
        elif 'root' in text_lower and 'ectomycorrhizal' in text_lower:
            return 'ectomycorrhizal_rootlets'
        else:
            return 'unknown'
    
    def _extract_plant_age(self, text: str) -> Optional[int]:
        """Extract plant age in days"""
        age_patterns = [
            r'(\d+)\s*days?\s*old',
            r'(\d+)\s*weeks?\s*old',
            r'(\d+)\s*months?\s*old'
        ]
        
        for pattern in age_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                age = int(match.group(1))
                if 'week' in pattern:
                    return age * 7
                elif 'month' in pattern:
                    return age * 30
                else:
                    return age
        
        return None
    
    def _extract_chamber_type(self, text: str) -> str:
        """Extract chamber type from text"""
        text_lower = text.lower()
        
        if 'hydroponic' in text_lower:
            return 'hydroponic'
        elif 'aeroponic' in text_lower:
            return 'aeroponic'
        elif 'bioreactor' in text_lower:
            return 'bioreactor'
        elif 'petri' in text_lower:
            return 'petri_dish'
        else:
            return 'unknown'
    
    def _extract_flow_regime(self, text: str) -> str:
        """Extract flow regime from text"""
        text_lower = text.lower()
        
        if 'static' in text_lower:
            return 'static'
        elif 'recirculat' in text_lower:
            return 'recirculating'
        elif 'aeroponic' in text_lower:
            return 'aeroponic'
        elif 'mist' in text_lower:
            return 'mist'
        else:
            return 'unknown'
    
    def _extract_volume(self, text: str) -> Optional[float]:
        """Extract volume in liters"""
        volume_patterns = [
            r'(\d+\.?\d*)\s*L',
            r'(\d+\.?\d*)\s*liters?',
            r'(\d+\.?\d*)\s*mL'
        ]
        
        for pattern in volume_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                volume = float(match.group(1))
                if 'mL' in pattern:
                    return volume / 1000
                else:
                    return volume
        
        return None
    
    def _extract_duration(self, text: str) -> Optional[int]:
        """Extract duration in days"""
        duration_patterns = [
            r'(\d+)\s*days?',
            r'(\d+)\s*weeks?',
            r'(\d+)\s*months?'
        ]
        
        for pattern in duration_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                duration = int(match.group(1))
                if 'week' in pattern:
                    return duration * 7
                elif 'month' in pattern:
                    return duration * 30
                else:
                    return duration
        
        return None
    
    def _extract_replicates(self, text: str) -> Optional[int]:
        """Extract number of replicates"""
        replicate_patterns = [
            r'(\d+)\s*replicates?',
            r'n\s*=\s*(\d+)',
            r'(\d+)\s*samples?'
        ]
        
        for pattern in replicate_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        return None
    
    def _extract_colonization(self, text: str) -> Optional[float]:
        """Extract colonization percentage"""
        colonization_patterns = [
            r'(\d+\.?\d*)\s*%\s*colonization',
            r'colonization\s*(\d+\.?\d*)\s*%',
            r'(\d+\.?\d*)\s*%\s*colonized'
        ]
        
        for pattern in colonization_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return float(match.group(1))
        
        return None
    
    def _extract_time_to_colonization(self, text: str) -> Optional[int]:
        """Extract time to colonization in days"""
        time_patterns = [
            r'(\d+)\s*days?\s*to\s*colonization',
            r'colonization\s*after\s*(\d+)\s*days?',
            r'(\d+)\s*days?\s*colonization'
        ]
        
        for pattern in time_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        return None
    
    def _extract_fruiting(self, text: str) -> Optional[bool]:
        """Extract fruiting information"""
        text_lower = text.lower()
        
        if 'fruiting' in text_lower or 'fruit' in text_lower:
            return True
        elif 'no fruiting' in text_lower or 'no fruit' in text_lower:
            return False
        
        return None
    
    def _extract_yield(self, text: str) -> Optional[float]:
        """Extract yield in grams"""
        yield_patterns = [
            r'(\d+\.?\d*)\s*g\s*yield',
            r'yield\s*(\d+\.?\d*)\s*g',
            r'(\d+\.?\d*)\s*g\s*produced'
        ]
        
        for pattern in yield_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return float(match.group(1))
        
        return None
    
    def _calculate_experiment_confidence(self, parameters: List[ExtractedParameter]) -> float:
        """Calculate confidence for extracted experiment"""
        if not parameters:
            return 0.0
        
        # Average confidence of parameters
        avg_confidence = sum(p.confidence for p in parameters) / len(parameters)
        
        # Boost confidence if we have key parameters
        key_params = ['ph', 'ec', 'temperature', 'fungus_name', 'host_name']
        key_param_count = sum(1 for p in parameters if p.parameter_name in key_params)
        
        if key_param_count >= 3:
            avg_confidence += 0.2
        
        return min(1.0, avg_confidence)

def main():
    """Example usage of the data extractor"""
    extractor = ScientificNERExtractor()
    
    # Example text
    text = """
    The experiment was conducted using Tuber melanosporum with Quercus ilex seedlings.
    The pH was maintained at 6.2, EC at 1.5 mS/cm, and temperature at 22°C.
    The nutrient solution contained 100 mg/L NO3-N, 50 mg/L NH4-N, and 30 mg/L PO4-P.
    The experiment lasted for 90 days with 5 replicates.
    Colonization was 85% after 60 days.
    """
    
    # Extract parameters
    parameters = extractor.extract_parameters(text)
    
    print("Extracted parameters:")
    for param in parameters:
        print(f"- {param.parameter_name}: {param.value} {param.unit} (confidence: {param.confidence:.2f})")
    
    # Extract entities
    entities = extractor.extract_entities(text)
    
    print("\nExtracted entities:")
    for entity in entities:
        print(f"- {entity['text']}: {entity['label']} (confidence: {entity['confidence']:.2f})")

if __name__ == "__main__":
    main()