"""
Literature Parsing and Segmentation Pipeline
Parses papers and patents into structured sections
"""

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import xml.etree.ElementTree as ET
import json
import re
from datetime import datetime

# PDF processing
import fitz  # PyMuPDF
import camelot
import pdfplumber

# Text processing
import spacy
from spacy.matcher import Matcher
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

logger = logging.getLogger(__name__)

@dataclass
class DocumentSection:
    """A section of a document"""
    section_type: str  # title, abstract, methods, results, discussion, references, tables, figures
    content: str
    page_number: int
    start_char: int
    end_char: int
    confidence: float
    metadata: Dict[str, Any]

@dataclass
class ParsedDocument:
    """A parsed document with sections"""
    document_id: str
    document_type: str  # paper, patent
    title: str
    authors: List[str]
    abstract: str
    sections: List[DocumentSection]
    tables: List[Dict[str, Any]]
    figures: List[Dict[str, Any]]
    references: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    parsed_at: datetime

class GROBIDParser:
    """Parser using GROBID for academic papers"""
    
    def __init__(self, grobid_path: str = None):
        self.grobid_path = grobid_path or self._find_grobid()
        self.grobid_server = "http://localhost:8070"
        
    def _find_grobid(self) -> Optional[str]:
        """Find GROBID installation"""
        # Try common installation paths
        possible_paths = [
            "/opt/grobid",
            "/usr/local/grobid",
            "~/grobid",
            "./grobid"
        ]
        
        for path in possible_paths:
            grobid_path = Path(path).expanduser()
            if grobid_path.exists() and (grobid_path / "grobid-server").exists():
                return str(grobid_path)
        
        return None
    
    def start_server(self):
        """Start GROBID server"""
        if not self.grobid_path:
            raise RuntimeError("GROBID not found. Please install GROBID or provide path.")
        
        try:
            # Start GROBID server
            subprocess.Popen([
                str(Path(self.grobid_path) / "grobid-server" / "bin" / "grobid-server"),
                "-Xmx4g"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Wait for server to start
            import time
            time.sleep(10)
            
            logger.info("GROBID server started")
            
        except Exception as e:
            logger.error(f"Failed to start GROBID server: {e}")
            raise
    
    def parse_pdf(self, pdf_path: Path) -> Optional[ParsedDocument]:
        """Parse PDF using GROBID"""
        try:
            import requests
            
            # Send PDF to GROBID
            with open(pdf_path, 'rb') as f:
                files = {'input': f}
                data = {
                    'generateIDs': '1',
                    'includeRawCitations': '1',
                    'includeRawAffiliations': '1'
                }
                
                response = requests.post(
                    f"{self.grobid_server}/api/processFulltextDocument",
                    files=files,
                    data=data,
                    timeout=60
                )
            
            if response.status_code != 200:
                logger.error(f"GROBID parsing failed: {response.status_code}")
                return None
            
            # Parse TEI XML response
            return self._parse_tei_xml(response.text, pdf_path)
            
        except Exception as e:
            logger.error(f"Error parsing PDF with GROBID: {e}")
            return None
    
    def _parse_tei_xml(self, tei_xml: str, pdf_path: Path) -> Optional[ParsedDocument]:
        """Parse TEI XML from GROBID"""
        try:
            root = ET.fromstring(tei_xml)
            
            # Extract basic info
            title_elem = root.find('.//{http://www.tei-c.org/ns/1.0}titleStmt/{http://www.tei-c.org/ns/1.0}title')
            title = title_elem.text if title_elem is not None else ""
            
            # Extract authors
            authors = []
            for author in root.findall('.//{http://www.tei-c.org/ns/1.0}sourceDesc/{http://www.tei-c.org/ns/1.0}biblStruct/{http://www.tei-c.org/ns/1.0}analytic/{http://www.tei-c.org/ns/1.0}author'):
                pers_name = author.find('.//{http://www.tei-c.org/ns/1.0}persName')
                if pers_name is not None:
                    forename = pers_name.find('.//{http://www.tei-c.org/ns/1.0}forename')
                    surname = pers_name.find('.//{http://www.tei-c.org/ns/1.0}surname')
                    if forename is not None and surname is not None:
                        authors.append(f"{forename.text} {surname.text}")
            
            # Extract abstract
            abstract_elem = root.find('.//{http://www.tei-c.org/ns/1.0}abstract')
            abstract = abstract_elem.text if abstract_elem is not None else ""
            
            # Extract sections
            sections = []
            for div in root.findall('.//{http://www.tei-c.org/ns/1.0}text/{http://www.tei-c.org/ns/1.0}body/{http://www.tei-c.org/ns/1.0}div'):
                head = div.find('.//{http://www.tei-c.org/ns/1.0}head')
                if head is not None:
                    section_type = self._classify_section(head.text)
                    content = self._extract_text(div)
                    
                    if content.strip():
                        section = DocumentSection(
                            section_type=section_type,
                            content=content,
                            page_number=0,  # GROBID doesn't provide page numbers
                            start_char=0,
                            end_char=len(content),
                            confidence=0.8,
                            metadata={'head': head.text}
                        )
                        sections.append(section)
            
            # Extract tables
            tables = self._extract_tables(root)
            
            # Extract figures
            figures = self._extract_figures(root)
            
            # Extract references
            references = self._extract_references(root)
            
            return ParsedDocument(
                document_id=pdf_path.stem,
                document_type='paper',
                title=title,
                authors=authors,
                abstract=abstract,
                sections=sections,
                tables=tables,
                figures=figures,
                references=references,
                metadata={'pdf_path': str(pdf_path)},
                parsed_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error parsing TEI XML: {e}")
            return None
    
    def _classify_section(self, head_text: str) -> str:
        """Classify section based on heading text"""
        head_lower = head_text.lower()
        
        if any(word in head_lower for word in ['abstract', 'summary']):
            return 'abstract'
        elif any(word in head_lower for word in ['introduction', 'background']):
            return 'introduction'
        elif any(word in head_lower for word in ['method', 'materials', 'experimental']):
            return 'methods'
        elif any(word in head_lower for word in ['result', 'finding']):
            return 'results'
        elif any(word in head_lower for word in ['discussion', 'conclusion']):
            return 'discussion'
        elif any(word in head_lower for word in ['reference', 'bibliography']):
            return 'references'
        else:
            return 'other'
    
    def _extract_text(self, element) -> str:
        """Extract text from XML element"""
        text_parts = []
        for elem in element.iter():
            if elem.text:
                text_parts.append(elem.text)
        return ' '.join(text_parts)
    
    def _extract_tables(self, root) -> List[Dict[str, Any]]:
        """Extract tables from TEI XML"""
        tables = []
        for table in root.findall('.//{http://www.tei-c.org/ns/1.0}table'):
            table_data = {
                'caption': '',
                'data': [],
                'metadata': {}
            }
            
            # Extract caption
            caption = table.find('.//{http://www.tei-c.org/ns/1.0}head')
            if caption is not None:
                table_data['caption'] = caption.text
            
            # Extract table data
            rows = table.findall('.//{http://www.tei-c.org/ns/1.0}row')
            for row in rows:
                row_data = []
                cells = row.findall('.//{http://www.tei-c.org/ns/1.0}cell')
                for cell in cells:
                    cell_text = self._extract_text(cell)
                    row_data.append(cell_text)
                table_data['data'].append(row_data)
            
            tables.append(table_data)
        
        return tables
    
    def _extract_figures(self, root) -> List[Dict[str, Any]]:
        """Extract figures from TEI XML"""
        figures = []
        for figure in root.findall('.//{http://www.tei-c.org/ns/1.0}figure'):
            fig_data = {
                'caption': '',
                'label': '',
                'metadata': {}
            }
            
            # Extract caption
            caption = figure.find('.//{http://www.tei-c.org/ns/1.0}figDesc')
            if caption is not None:
                fig_data['caption'] = caption.text
            
            # Extract label
            label = figure.find('.//{http://www.tei-c.org/ns/1.0}label')
            if label is not None:
                fig_data['label'] = label.text
            
            figures.append(fig_data)
        
        return figures
    
    def _extract_references(self, root) -> List[Dict[str, Any]]:
        """Extract references from TEI XML"""
        references = []
        for ref in root.findall('.//{http://www.tei-c.org/ns/1.0}listBibl/{http://www.tei-c.org/ns/1.0}biblStruct'):
            ref_data = {
                'title': '',
                'authors': [],
                'year': '',
                'venue': '',
                'doi': ''
            }
            
            # Extract title
            title = ref.find('.//{http://www.tei-c.org/ns/1.0}title')
            if title is not None:
                ref_data['title'] = title.text
            
            # Extract authors
            for author in ref.findall('.//{http://www.tei-c.org/ns/1.0}author'):
                pers_name = author.find('.//{http://www.tei-c.org/ns/1.0}persName')
                if pers_name is not None:
                    forename = pers_name.find('.//{http://www.tei-c.org/ns/1.0}forename')
                    surname = pers_name.find('.//{http://www.tei-c.org/ns/1.0}surname')
                    if forename is not None and surname is not None:
                        ref_data['authors'].append(f"{forename.text} {surname.text}")
            
            # Extract year
            date = ref.find('.//{http://www.tei-c.org/ns/1.0}date')
            if date is not None:
                ref_data['year'] = date.get('when', '')
            
            # Extract venue
            venue = ref.find('.//{http://www.tei-c.org/ns/1.0}monogr/{http://www.tei-c.org/ns/1.0}title')
            if venue is not None:
                ref_data['venue'] = venue.text
            
            # Extract DOI
            doi = ref.find('.//{http://www.tei-c.org/ns/1.0}idno[@type="DOI"]')
            if doi is not None:
                ref_data['doi'] = doi.text
            
            references.append(ref_data)
        
        return references

class PatentParser:
    """Parser for patent documents"""
    
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
    
    def parse_patent_xml(self, xml_path: Path) -> Optional[ParsedDocument]:
        """Parse patent XML file"""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Extract basic info
            title = self._extract_patent_title(root)
            inventors = self._extract_patent_inventors(root)
            abstract = self._extract_patent_abstract(root)
            
            # Extract sections
            sections = self._extract_patent_sections(root)
            
            # Extract claims
            claims = self._extract_patent_claims(root)
            
            # Extract description
            description = self._extract_patent_description(root)
            
            return ParsedDocument(
                document_id=xml_path.stem,
                document_type='patent',
                title=title,
                authors=inventors,
                abstract=abstract,
                sections=sections,
                tables=[],
                figures=[],
                references=[],
                metadata={'xml_path': str(xml_path)},
                parsed_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error parsing patent XML: {e}")
            return None
    
    def _extract_patent_title(self, root) -> str:
        """Extract patent title"""
        title_elem = root.find('.//invention-title')
        return title_elem.text if title_elem is not None else ""
    
    def _extract_patent_inventors(self, root) -> List[str]:
        """Extract patent inventors"""
        inventors = []
        for inventor in root.findall('.//inventor'):
            first_name = inventor.find('.//first-name')
            last_name = inventor.find('.//last-name')
            if first_name is not None and last_name is not None:
                inventors.append(f"{first_name.text} {last_name.text}")
        return inventors
    
    def _extract_patent_abstract(self, root) -> str:
        """Extract patent abstract"""
        abstract_elem = root.find('.//abstract')
        if abstract_elem is not None:
            return self._extract_text(abstract_elem)
        return ""
    
    def _extract_patent_sections(self, root) -> List[DocumentSection]:
        """Extract patent sections"""
        sections = []
        
        # Extract description sections
        for section in root.findall('.//description//section'):
            title_elem = section.find('.//heading')
            title = title_elem.text if title_elem is not None else "Unknown"
            
            content = self._extract_text(section)
            if content.strip():
                section_type = self._classify_patent_section(title)
                doc_section = DocumentSection(
                    section_type=section_type,
                    content=content,
                    page_number=0,
                    start_char=0,
                    end_char=len(content),
                    confidence=0.8,
                    metadata={'title': title}
                )
                sections.append(doc_section)
        
        return sections
    
    def _classify_patent_section(self, title: str) -> str:
        """Classify patent section"""
        title_lower = title.lower()
        
        if 'field' in title_lower:
            return 'field'
        elif 'background' in title_lower:
            return 'background'
        elif 'summary' in title_lower:
            return 'summary'
        elif 'description' in title_lower:
            return 'description'
        elif 'example' in title_lower:
            return 'examples'
        else:
            return 'other'
    
    def _extract_patent_claims(self, root) -> List[str]:
        """Extract patent claims"""
        claims = []
        for claim in root.findall('.//claim'):
            claim_text = self._extract_text(claim)
            if claim_text.strip():
                claims.append(claim_text)
        return claims
    
    def _extract_patent_description(self, root) -> str:
        """Extract patent description"""
        description_elem = root.find('.//description')
        if description_elem is not None:
            return self._extract_text(description_elem)
        return ""
    
    def _extract_text(self, element) -> str:
        """Extract text from XML element"""
        text_parts = []
        for elem in element.iter():
            if elem.text:
                text_parts.append(elem.text)
        return ' '.join(text_parts)

class PDFTableExtractor:
    """Extract tables from PDF files"""
    
    def __init__(self):
        self.camelot_params = {
            'flavor': 'lattice',
            'line_scale': 40,
            'copy_text': ['v']
        }
    
    def extract_tables(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Extract tables from PDF using Camelot"""
        tables = []
        
        try:
            # Extract tables using Camelot
            camelot_tables = camelot.read_pdf(str(pdf_path), **self.camelot_params)
            
            for i, table in enumerate(camelot_tables):
                table_data = {
                    'table_id': f"table_{i+1}",
                    'page_number': table.page,
                    'data': table.df.values.tolist(),
                    'accuracy': table.accuracy,
                    'whitespace': table.whitespace,
                    'metadata': {
                        'method': 'camelot',
                        'flavor': self.camelot_params['flavor']
                    }
                }
                tables.append(table_data)
                
        except Exception as e:
            logger.error(f"Error extracting tables with Camelot: {e}")
            
            # Fallback to pdfplumber
            try:
                tables = self._extract_tables_pdfplumber(pdf_path)
            except Exception as e2:
                logger.error(f"Error extracting tables with pdfplumber: {e2}")
        
        return tables
    
    def _extract_tables_pdfplumber(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Extract tables using pdfplumber as fallback"""
        tables = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_tables = page.extract_tables()
                
                for i, table in enumerate(page_tables):
                    table_data = {
                        'table_id': f"table_{page_num+1}_{i+1}",
                        'page_number': page_num + 1,
                        'data': table,
                        'accuracy': 0.5,  # Lower confidence for pdfplumber
                        'whitespace': 0,
                        'metadata': {
                            'method': 'pdfplumber'
                        }
                    }
                    tables.append(table_data)
        
        return tables

class DocumentParser:
    """Main document parser that coordinates different parsing methods"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.grobid_parser = GROBIDParser(config.get('grobid_path'))
        self.patent_parser = PatentParser()
        self.table_extractor = PDFTableExtractor()
        
        # Initialize NLP
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found, installing...")
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
    
    def parse_document(self, file_path: Path) -> Optional[ParsedDocument]:
        """Parse a document based on its type"""
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.pdf':
            return self._parse_pdf(file_path)
        elif file_path.suffix.lower() == '.xml':
            return self._parse_xml(file_path)
        else:
            logger.error(f"Unsupported file type: {file_path.suffix}")
            return None
    
    def _parse_pdf(self, pdf_path: Path) -> Optional[ParsedDocument]:
        """Parse PDF document"""
        logger.info(f"Parsing PDF: {pdf_path}")
        
        # Try GROBID first
        if self.grobid_parser.grobid_path:
            try:
                parsed_doc = self.grobid_parser.parse_pdf(pdf_path)
                if parsed_doc:
                    # Extract additional tables
                    tables = self.table_extractor.extract_tables(pdf_path)
                    parsed_doc.tables.extend(tables)
                    return parsed_doc
            except Exception as e:
                logger.error(f"GROBID parsing failed: {e}")
        
        # Fallback to basic PDF parsing
        return self._parse_pdf_basic(pdf_path)
    
    def _parse_pdf_basic(self, pdf_path: Path) -> Optional[ParsedDocument]:
        """Basic PDF parsing using PyMuPDF"""
        try:
            doc = fitz.open(pdf_path)
            
            # Extract text
            full_text = ""
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                full_text += page.get_text()
            
            # Basic section extraction
            sections = self._extract_sections_basic(full_text)
            
            # Extract tables
            tables = self.table_extractor.extract_tables(pdf_path)
            
            doc.close()
            
            return ParsedDocument(
                document_id=pdf_path.stem,
                document_type='paper',
                title=pdf_path.stem,
                authors=[],
                abstract="",
                sections=sections,
                tables=tables,
                figures=[],
                references=[],
                metadata={'pdf_path': str(pdf_path)},
                parsed_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error parsing PDF: {e}")
            return None
    
    def _parse_xml(self, xml_path: Path) -> Optional[ParsedDocument]:
        """Parse XML document (patent)"""
        logger.info(f"Parsing XML: {xml_path}")
        return self.patent_parser.parse_patent_xml(xml_path)
    
    def _extract_sections_basic(self, text: str) -> List[DocumentSection]:
        """Basic section extraction using regex patterns"""
        sections = []
        
        # Define section patterns
        section_patterns = [
            (r'(?i)^\s*(abstract|summary)\s*$', 'abstract'),
            (r'(?i)^\s*(introduction|background)\s*$', 'introduction'),
            (r'(?i)^\s*(method|materials|experimental|procedure)\s*$', 'methods'),
            (r'(?i)^\s*(result|finding|observation)\s*$', 'results'),
            (r'(?i)^\s*(discussion|conclusion)\s*$', 'discussion'),
            (r'(?i)^\s*(reference|bibliography)\s*$', 'references')
        ]
        
        lines = text.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line matches a section header
            section_found = False
            for pattern, section_type in section_patterns:
                if re.match(pattern, line):
                    # Save previous section
                    if current_section and current_content:
                        content = '\n'.join(current_content)
                        section = DocumentSection(
                            section_type=current_section,
                            content=content,
                            page_number=0,
                            start_char=0,
                            end_char=len(content),
                            confidence=0.6,
                            metadata={}
                        )
                        sections.append(section)
                    
                    # Start new section
                    current_section = section_type
                    current_content = []
                    section_found = True
                    break
            
            if not section_found and current_section:
                current_content.append(line)
        
        # Save last section
        if current_section and current_content:
            content = '\n'.join(current_content)
            section = DocumentSection(
                section_type=current_section,
                content=content,
                page_number=0,
                start_char=0,
                end_char=len(content),
                confidence=0.6,
                metadata={}
            )
            sections.append(section)
        
        return sections

def main():
    """Example usage of the document parser"""
    config = {
        'grobid_path': None  # Will auto-detect
    }
    
    parser = DocumentParser(config)
    
    # Example: Parse a PDF
    pdf_path = Path("example.pdf")
    if pdf_path.exists():
        parsed_doc = parser.parse_document(pdf_path)
        if parsed_doc:
            print(f"Parsed document: {parsed_doc.title}")
            print(f"Sections: {[s.section_type for s in parsed_doc.sections]}")
            print(f"Tables: {len(parsed_doc.tables)}")

if __name__ == "__main__":
    main()