# Truffle Literature Processing System

A comprehensive pipeline for discovering, extracting, and analyzing scientific literature on truffle cultivation, with a focus on hydroponic and soilless growing methods.

## 🎯 Overview

This system implements a complete literature processing workflow that:

1. **Discovers** relevant papers from multiple academic databases
2. **Parses** PDFs and XML documents to extract structured content
3. **Extracts** experimental parameters using hybrid ML and rule-based methods
4. **Normalizes** data to standard units and validates quality
5. **Reviews** data through human-in-the-loop interface
6. **Publishes** comprehensive literature reviews and loads data into knowledge graphs

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Discovery     │    │   Parsing       │    │   Extraction    │
│                 │    │                 │    │                 │
│ • OpenAlex      │───▶│ • GROBID        │───▶│ • ML NER        │
│ • Crossref      │    │ • PDF parsing   │    │ • Pattern match │
│ • PubMed        │    │ • XML parsing   │    │ • Table extract │
│ • arXiv         │    │ • Segmentation  │    │ • Relation assy │
│ • Patents       │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Publishing    │    │   Human Review  │    │   Normalization │
│                 │    │                 │    │                 │
│ • HTML reports  │◀───│ • Streamlit UI  │◀───│ • Unit convert  │
│ • PDF reports   │    │ • Data correct  │    │ • Validation    │
│ • Data dumps    │    │ • Quality check │    │ • Deduplication │
│ • KG loading    │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📁 Directory Structure

```
literature/
├── discovery/
│   └── downloader.py          # Multi-source literature discovery
├── parsing/
│   └── parser.py              # Document parsing and segmentation
├── extraction/
│   └── extractor.py           # Parameter extraction with ML NER
├── normalization/
│   └── validator.py           # Data normalization and validation
├── review/
│   └── review_app.py          # Human-in-the-loop review interface
├── publishing/
│   ├── review_publisher.py    # Report generation and KG loading
│   └── templates/
│       └── literature_review.html
├── main_pipeline.py           # Main orchestration script
├── queries.json               # Search queries configuration
└── README.md                  # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install GROBID (optional, for advanced PDF parsing)
# Download from https://github.com/kermitt2/grobid
# Extract to /opt/grobid or update config

# Install spaCy model
python -m spacy download en_core_web_sm
```

### 2. Configuration

Edit `config/config.yaml` to set up:

```yaml
literature:
  output_dir: "data/literature_processing"
  enable_human_review: true
  max_results_per_query: 1000
  
  # API configurations
  openalex:
    rate_limit: 10
  crossref:
    rate_limit: 50
  pubmed:
    rate_limit: 3
    
  # GROBID configuration
  grobid:
    path: "/opt/grobid"
    server: "http://localhost:8070"
```

### 3. Run Pipeline

```bash
# Run full pipeline
python literature/main_pipeline.py --config config/config.yaml

# Run specific step
python literature/main_pipeline.py --step discover_download

# Enable human review
python literature/main_pipeline.py --enable-review
```

## 🔧 Components

### 1. Discovery & Download (`discovery/downloader.py`)

**Purpose**: Discovers and downloads literature from multiple sources

**Features**:
- Multi-source API integration (OpenAlex, Crossref, PubMed, arXiv, Unpaywall)
- Rate limiting and error handling
- PDF download for open access papers
- Metadata extraction and deduplication

**Usage**:
```python
from literature.discovery.downloader import LiteratureDownloader

downloader = LiteratureDownloader(config)
papers, patents = downloader.run_discovery(queries, max_results=1000)
```

### 2. Parsing & Segmentation (`parsing/parser.py`)

**Purpose**: Parses documents into structured sections

**Features**:
- GROBID integration for academic papers
- PDF parsing with PyMuPDF and pdfplumber
- XML parsing for patents
- Section classification and table extraction

**Usage**:
```python
from literature.parsing.parser import DocumentParser

parser = DocumentParser(config)
parsed_doc = parser.parse_document(pdf_path)
```

### 3. Data Extraction (`extraction/extractor.py`)

**Purpose**: Extracts experimental parameters using hybrid ML and rule-based methods

**Features**:
- Scientific NER with spaCy
- Pattern matching for parameters (pH, EC, temperature, etc.)
- Table extraction and parameter identification
- Experiment assembly and confidence scoring

**Usage**:
```python
from literature.extraction.extractor import ExperimentExtractor

extractor = ExperimentExtractor()
experiments = extractor.extract_experiments(parsed_doc)
```

### 4. Normalization & Validation (`normalization/validator.py`)

**Purpose**: Normalizes units and validates data quality

**Features**:
- Unit conversion to standard forms
- Data validation with sanity checks
- Confidence scoring and quality metrics
- Deduplication of similar experiments

**Usage**:
```python
from literature.normalization.validator import DataNormalizer

normalizer = DataNormalizer()
normalized_exp = normalizer.normalize_experiment(experiment)
```

### 5. Human Review Interface (`review/review_app.py`)

**Purpose**: Provides interactive interface for data review and correction

**Features**:
- Streamlit-based web interface
- Experiment browsing and filtering
- Parameter correction and validation
- Progress tracking and statistics

**Usage**:
```bash
streamlit run literature/review/review_app.py
```

### 6. Publishing (`publishing/review_publisher.py`)

**Purpose**: Generates reports and loads data into knowledge graph

**Features**:
- HTML/PDF report generation
- Data dumps (CSV format)
- Visualization generation
- Knowledge graph loading

**Usage**:
```python
from literature.publishing.review_publisher import LiteratureReviewPublisher

publisher = LiteratureReviewPublisher(config)
results = publisher.publish_review(experiments, papers_metadata)
```

## 📊 Data Model

### Core Entities

**Paper**: Academic paper or patent
- `paper_id`, `doi`, `title`, `year`, `venue`, `authors`, `abstract`

**Experiment**: Individual experimental setup
- `experiment_id`, `fungus_name`, `host_name`, `inoculum_form`, `chamber_type`

**Parameter**: Measured experimental parameter
- `parameter_name`, `value`, `unit`, `confidence`, `source_location`

**Outcome**: Experimental results
- `colonization_pct`, `yield_g`, `fruiting`, `time_to_colonization_d`

### Parameter Types

- **Environmental**: pH, EC, temperature, dissolved oxygen, CO₂
- **Nutrient**: NO₃-N, NH₄-N, PO₄-P, K, Ca, Mg, micronutrients
- **Biological**: Fungus species, host species, inoculum form
- **Experimental**: Volume, duration, replicates, flow regime

## 🔍 Search Queries

The system uses predefined queries to discover relevant literature:

```json
{
  "papers": {
    "truffle_mycorrhiza_hydroponic": "Tuber OR truffle* AND (mycorrhiz* OR ectomycorrhiz*) AND (hydropon* OR soilles* OR aeropon*)",
    "ph_ec_controls": "(\"pH\" OR \"EC\" OR \"dissolved oxygen\") AND (mycorrhiza* OR Tuber*)",
    "truffle_cultivation_methods": "truffle AND (cultivation OR inoculation) AND (hydroponic OR soilless)"
  },
  "patents": {
    "soilless_cultivation": "truffle AND (hydroponic OR soilless OR aeroponic)",
    "mycorrhiza_methods": "mycorrhiza AND (inoculation OR cultivation OR method)"
  }
}
```

## 📈 Quality Metrics

The system tracks several quality metrics:

- **Validation Rate**: Percentage of experiments passing validation
- **Confidence Scores**: Per-parameter and per-experiment confidence
- **Parameter Coverage**: Completeness of parameter measurements
- **Species Diversity**: Number of unique fungus/host combinations
- **Success Rate**: Percentage of experiments with >50% colonization

## 🎯 Example Workflows

### 1. Literature Review Generation

```bash
# Run full pipeline
python literature/main_pipeline.py --config config/config.yaml

# Results in:
# - data/literature_processing/experiments.csv
# - data/literature_processing/literature_review.html
# - data/literature_processing/quality_report.json
```

### 2. Human Review Process

```bash
# Launch review interface
streamlit run literature/review/review_app.py

# Review and correct data
# Export corrected data
```

### 3. Knowledge Graph Integration

```python
# Load experiments into KG
from literature.publishing.review_publisher import LiteratureReviewPublisher

publisher = LiteratureReviewPublisher(config)
results = publisher.load_into_knowledge_graph(experiments)
```

## 🔧 Configuration

### API Rate Limits

```yaml
literature:
  apis:
    openalex:
      rate_limit: 10  # requests per second
    crossref:
      rate_limit: 50
    pubmed:
      rate_limit: 3
```

### Extraction Parameters

```yaml
literature:
  extraction:
    confidence_threshold: 0.5
    parameter_patterns:
      ph: ["pH", "ph"]
      ec: ["EC", "electrical conductivity"]
      temperature: ["temperature", "temp", "°C"]
```

### Validation Rules

```yaml
literature:
  validation:
    ph:
      min: 3.0
      max: 9.0
      typical_range: [5.0, 7.0]
    ec:
      min: 0.1
      max: 10.0
      typical_range: [0.5, 3.0]
```

## 🚨 Troubleshooting

### Common Issues

1. **GROBID Server Not Running**
   ```bash
   # Start GROBID server
   cd /opt/grobid
   ./gradlew run
   ```

2. **API Rate Limiting**
   - Check rate limits in configuration
   - Implement exponential backoff
   - Use multiple API keys if available

3. **PDF Parsing Errors**
   - Ensure PyMuPDF is installed
   - Check PDF file integrity
   - Try alternative parsing methods

4. **Memory Issues**
   - Process documents in batches
   - Increase available memory
   - Use streaming processing

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python literature/main_pipeline.py --config config/config.yaml
```

## 📚 Dependencies

### Core Dependencies
- `requests`: HTTP requests for APIs
- `pandas`: Data manipulation
- `numpy`: Numerical computing
- `spacy`: Natural language processing
- `streamlit`: Web interface
- `jinja2`: Template rendering

### Optional Dependencies
- `grobid`: Advanced PDF parsing
- `weasyprint`: PDF generation
- `camelot`: Table extraction
- `pdfplumber`: PDF text extraction

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- GROBID team for PDF parsing
- spaCy team for NLP capabilities
- Streamlit team for web interface
- OpenAlex, Crossref, and other API providers

## 📞 Support

For questions or issues:
- Create an issue on GitHub
- Contact the development team
- Check the documentation wiki

---

**Built with ❤️ for the future of truffle cultivation research**