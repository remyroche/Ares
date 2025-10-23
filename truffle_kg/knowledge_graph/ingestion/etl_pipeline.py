"""
ETL Pipeline for Truffle Knowledge Graph
Handles ingestion from papers (PDF), lab CSVs, sensor time-series, and images
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path
import requests
from dataclasses import dataclass, asdict
import hashlib

# NLP and document processing
import spacy
from transformers import AutoTokenizer, AutoModel
import PyPDF2
import fitz  # PyMuPDF
from PIL import Image
import cv2

# RDF and graph processing
from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, RDFS, OWL, XSD
import neo4j
from neo4j import GraphDatabase

# Data validation
import jsonschema
from jsonschema import validate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Measurement:
    """Represents a quantitative measurement with uncertainty"""
    value: float
    unit: str
    uncertainty: Optional[float] = None
    confidence: Optional[float] = None
    measurement_date: Optional[datetime] = None
    method: Optional[str] = None
    device: Optional[str] = None

@dataclass
class Evidence:
    """Represents evidence supporting measurements"""
    evidence_code: str  # in_vitro, in_planta, field_trial, etc.
    method: str
    device: str
    calibration: str
    operator: str
    date: datetime
    confidence: float

@dataclass
class Outcome:
    """Represents experimental outcomes"""
    colonization_percent: Optional[Measurement] = None
    hyphal_density: Optional[Measurement] = None
    yield: Optional[Measurement] = None
    primordia_count: Optional[int] = None
    success: bool = False
    evidence: List[Evidence] = None

class TruffleKGIngestion:
    """Main ETL pipeline for truffle knowledge graph"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rdf_graph = Graph()
        self.neo4j_driver = None
        self.nlp_model = None
        self.bert_model = None
        self.bert_tokenizer = None
        
        # Initialize namespaces
        self.ex = Namespace("http://example.org/truffle/kg#")
        self.prov = Namespace("http://www.w3.org/ns/prov#")
        self.qudt = Namespace("http://qudt.org/schema/qudt/")
        
        self._setup_models()
        self._setup_databases()
    
    def _setup_models(self):
        """Initialize NLP and ML models"""
        try:
            # Load spaCy model for entity recognition
            self.nlp_model = spacy.load("en_core_sci_sm")
            logger.info("Loaded spaCy model")
        except OSError:
            logger.warning("spaCy model not found, using basic tokenization")
            self.nlp_model = None
        
        try:
            # Load SciBERT for scientific text processing
            model_name = "allenai/scibert_scivocab_uncased"
            self.bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.bert_model = AutoModel.from_pretrained(model_name)
            logger.info("Loaded SciBERT model")
        except Exception as e:
            logger.warning(f"Could not load SciBERT: {e}")
            self.bert_model = None
            self.bert_tokenizer = None
    
    def _setup_databases(self):
        """Initialize database connections"""
        # Neo4j connection
        try:
            self.neo4j_driver = GraphDatabase.driver(
                self.config["neo4j"]["uri"],
                auth=(self.config["neo4j"]["user"], self.config["neo4j"]["password"])
            )
            logger.info("Connected to Neo4j")
        except Exception as e:
            logger.error(f"Could not connect to Neo4j: {e}")
    
    def ingest_pdf_papers(self, pdf_directory: str) -> List[Dict[str, Any]]:
        """Extract structured data from PDF papers"""
        logger.info(f"Ingesting PDFs from {pdf_directory}")
        extracted_data = []
        
        for pdf_file in Path(pdf_directory).glob("*.pdf"):
            try:
                data = self._extract_pdf_data(str(pdf_file))
                extracted_data.append(data)
                logger.info(f"Processed {pdf_file.name}")
            except Exception as e:
                logger.error(f"Error processing {pdf_file.name}: {e}")
        
        return extracted_data
    
    def _extract_pdf_data(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text and metadata from PDF"""
        data = {
            "source_file": pdf_path,
            "extraction_date": datetime.now().isoformat(),
            "text": "",
            "entities": [],
            "measurements": [],
            "references": []
        }
        
        # Extract text using PyMuPDF
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            data["text"] = text
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return data
        
        # Extract entities and measurements using NLP
        if self.nlp_model:
            data["entities"] = self._extract_entities(text)
            data["measurements"] = self._extract_measurements(text)
        
        return data
    
    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text"""
        if not self.nlp_model:
            return []
        
        doc = self.nlp_model(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "confidence": 0.8  # Placeholder
            })
        
        return entities
    
    def _extract_measurements(self, text: str) -> List[Dict[str, Any]]:
        """Extract measurements from text using regex patterns"""
        import re
        
        measurements = []
        
        # Pattern for pH measurements
        ph_pattern = r'pH\s*[=:]\s*(\d+\.?\d*)'
        for match in re.finditer(ph_pattern, text, re.IGNORECASE):
            measurements.append({
                "type": "pH",
                "value": float(match.group(1)),
                "unit": "pH",
                "context": text[max(0, match.start()-50):match.end()+50]
            })
        
        # Pattern for EC measurements
        ec_pattern = r'EC\s*[=:]\s*(\d+\.?\d*)\s*(mS/cm|dS/m)'
        for match in re.finditer(ec_pattern, text, re.IGNORECASE):
            measurements.append({
                "type": "EC",
                "value": float(match.group(1)),
                "unit": match.group(2),
                "context": text[max(0, match.start()-50):match.end()+50]
            })
        
        # Pattern for temperature measurements
        temp_pattern = r'(\d+\.?\d*)\s*°?C'
        for match in re.finditer(temp_pattern, text):
            measurements.append({
                "type": "temperature",
                "value": float(match.group(1)),
                "unit": "°C",
                "context": text[max(0, match.start()-50):match.end()+50]
            })
        
        return measurements
    
    def ingest_lab_csv(self, csv_path: str) -> List[Dict[str, Any]]:
        """Ingest laboratory CSV data"""
        logger.info(f"Ingesting lab CSV from {csv_path}")
        
        try:
            df = pd.read_csv(csv_path)
            data = []
            
            for _, row in df.iterrows():
                record = {
                    "source_file": csv_path,
                    "ingestion_date": datetime.now().isoformat(),
                    "data": row.to_dict()
                }
                data.append(record)
            
            return data
        except Exception as e:
            logger.error(f"Error ingesting CSV {csv_path}: {e}")
            return []
    
    def ingest_sensor_data(self, sensor_directory: str) -> List[Dict[str, Any]]:
        """Ingest time-series sensor data"""
        logger.info(f"Ingesting sensor data from {sensor_directory}")
        
        sensor_data = []
        
        for sensor_file in Path(sensor_directory).glob("*.json"):
            try:
                with open(sensor_file, 'r') as f:
                    data = json.load(f)
                
                # Process time-series data
                processed_data = self._process_sensor_data(data, str(sensor_file))
                sensor_data.append(processed_data)
                
            except Exception as e:
                logger.error(f"Error processing sensor file {sensor_file}: {e}")
        
        return sensor_data
    
    def _process_sensor_data(self, data: Dict[str, Any], source_file: str) -> Dict[str, Any]:
        """Process raw sensor data into structured format"""
        processed = {
            "source_file": source_file,
            "ingestion_date": datetime.now().isoformat(),
            "sensor_type": data.get("sensor_type", "unknown"),
            "location": data.get("location", "unknown"),
            "time_series": []
        }
        
        # Process time series data
        if "readings" in data:
            for reading in data["readings"]:
                processed["time_series"].append({
                    "timestamp": reading.get("timestamp"),
                    "values": reading.get("values", {}),
                    "quality": reading.get("quality", "unknown")
                })
        
        return processed
    
    def normalize_data(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize units, taxon IDs, and synonyms"""
        logger.info("Normalizing data")
        
        normalized_data = []
        
        for record in raw_data:
            try:
                normalized = self._normalize_record(record)
                normalized_data.append(normalized)
            except Exception as e:
                logger.error(f"Error normalizing record: {e}")
        
        return normalized_data
    
    def _normalize_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a single record"""
        normalized = record.copy()
        
        # Normalize units
        if "measurements" in record:
            for measurement in record["measurements"]:
                if "unit" in measurement:
                    measurement["unit"] = self._normalize_unit(measurement["unit"])
        
        # Normalize species names
        if "species" in record:
            record["species"] = self._normalize_species_name(record["species"])
        
        return normalized
    
    def _normalize_unit(self, unit: str) -> str:
        """Normalize unit strings to standard forms"""
        unit_mapping = {
            "mS/cm": "mS/cm",
            "dS/m": "mS/cm",  # Convert dS/m to mS/cm
            "°C": "°C",
            "C": "°C",
            "pH": "pH",
            "mg/L": "mg/L",
            "ppm": "mg/L"  # Convert ppm to mg/L
        }
        
        return unit_mapping.get(unit, unit)
    
    def _normalize_species_name(self, species: str) -> str:
        """Normalize species names to standard format"""
        # Basic normalization - in practice, this would use taxonomic databases
        return species.strip().title()
    
    def validate_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate data using SHACL shapes"""
        logger.info("Validating data")
        
        validated_data = []
        
        for record in data:
            try:
                if self._validate_record(record):
                    validated_data.append(record)
                else:
                    logger.warning(f"Record failed validation: {record.get('id', 'unknown')}")
            except Exception as e:
                logger.error(f"Error validating record: {e}")
        
        return validated_data
    
    def _validate_record(self, record: Dict[str, Any]) -> bool:
        """Validate a single record"""
        # Basic validation - in practice, this would use SHACL
        required_fields = ["id", "type"]
        
        for field in required_fields:
            if field not in record:
                return False
        
        return True
    
    def store_rdf(self, data: List[Dict[str, Any]]):
        """Store data in RDF format"""
        logger.info("Storing data in RDF format")
        
        for record in data:
            self._add_record_to_rdf(record)
        
        # Save RDF graph
        output_file = self.config["output"]["rdf_file"]
        self.rdf_graph.serialize(destination=output_file, format="turtle")
        logger.info(f"RDF data saved to {output_file}")
    
    def _add_record_to_rdf(self, record: Dict[str, Any]):
        """Add a single record to the RDF graph"""
        record_id = record.get("id", f"record_{hashlib.md5(str(record).encode()).hexdigest()}")
        record_uri = self.ex[record_id]
        
        # Add basic properties
        if "species" in record:
            self.rdf_graph.add((record_uri, self.ex.species, Literal(record["species"])))
        
        if "type" in record:
            type_uri = self.ex[record["type"]]
            self.rdf_graph.add((record_uri, RDF.type, type_uri))
    
    def store_neo4j(self, data: List[Dict[str, Any]]):
        """Store data in Neo4j property graph"""
        logger.info("Storing data in Neo4j")
        
        with self.neo4j_driver.session() as session:
            for record in data:
                self._add_record_to_neo4j(session, record)
        
        logger.info("Neo4j data storage completed")
    
    def _add_record_to_neo4j(self, session, record: Dict[str, Any]):
        """Add a single record to Neo4j"""
        record_type = record.get("type", "Record")
        record_id = record.get("id", f"record_{hashlib.md5(str(record).encode()).hexdigest()}")
        
        # Create node properties
        properties = {k: v for k, v in record.items() if k not in ["id", "type"]}
        properties["id"] = record_id
        
        # Create node
        query = f"CREATE (n:{record_type} $props)"
        session.run(query, props=properties)
    
    def run_pipeline(self, input_directory: str):
        """Run the complete ETL pipeline"""
        logger.info("Starting ETL pipeline")
        
        # Ingest data from various sources
        all_data = []
        
        # Ingest PDFs
        pdf_dir = os.path.join(input_directory, "papers")
        if os.path.exists(pdf_dir):
            pdf_data = self.ingest_pdf_papers(pdf_dir)
            all_data.extend(pdf_data)
        
        # Ingest lab CSVs
        csv_dir = os.path.join(input_directory, "lab_data")
        if os.path.exists(csv_dir):
            for csv_file in Path(csv_dir).glob("*.csv"):
                csv_data = self.ingest_lab_csv(str(csv_file))
                all_data.extend(csv_data)
        
        # Ingest sensor data
        sensor_dir = os.path.join(input_directory, "sensor_data")
        if os.path.exists(sensor_dir):
            sensor_data = self.ingest_sensor_data(sensor_dir)
            all_data.extend(sensor_data)
        
        # Normalize data
        normalized_data = self.normalize_data(all_data)
        
        # Validate data
        validated_data = self.validate_data(normalized_data)
        
        # Store data
        self.store_rdf(validated_data)
        self.store_neo4j(validated_data)
        
        logger.info(f"ETL pipeline completed. Processed {len(validated_data)} records")

def main():
    """Main function to run the ETL pipeline"""
    config = {
        "neo4j": {
            "uri": "bolt://localhost:7687",
            "user": "neo4j",
            "password": "password"
        },
        "output": {
            "rdf_file": "/workspace/truffle_kg/data/truffle_kg.ttl"
        }
    }
    
    pipeline = TruffleKGIngestion(config)
    pipeline.run_pipeline("/workspace/truffle_kg/data/input")

if __name__ == "__main__":
    main()