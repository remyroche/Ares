"""
Literature Discovery and Download System
Downloads papers, patents, and preprints from multiple APIs
"""

import requests
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import hashlib
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
import backoff
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

@dataclass
class PaperMetadata:
    """Metadata for a paper"""
    paper_id: str
    doi: Optional[str]
    title: str
    year: int
    venue: str
    url_pdf: Optional[str]
    source: str
    license: Optional[str]
    authors: List[str]
    keywords: List[str]
    abstract: Optional[str]
    sections_json: Dict[str, Any]
    fetched_at: datetime
    url_source: str
    citation_count: Optional[int] = None
    open_access: bool = False

@dataclass
class PatentMetadata:
    """Metadata for a patent"""
    patent_id: str
    title: str
    inventors: List[str]
    assignee: str
    filing_date: str
    publication_date: str
    url_pdf: Optional[str]
    source: str
    cpc_codes: List[str]
    ipc_codes: List[str]
    abstract: Optional[str]
    claims: List[str]
    description: str
    fetched_at: datetime
    url_source: str

class LiteratureDownloader:
    """Main class for downloading literature from multiple sources"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = Path(config.get('output_dir', 'data/literature'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # API configurations
        self.apis = {
            'openalex': {
                'base_url': 'https://api.openalex.org',
                'rate_limit': 10,  # requests per second
                'headers': {'User-Agent': 'TruffleKG/1.0 (mailto:research@example.com)'}
            },
            'crossref': {
                'base_url': 'https://api.crossref.org',
                'rate_limit': 50,
                'headers': {'User-Agent': 'TruffleKG/1.0 (mailto:research@example.com)'}
            },
            'pubmed': {
                'base_url': 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils',
                'rate_limit': 3,
                'headers': {}
            },
            'arxiv': {
                'base_url': 'https://export.arxiv.org/api/query',
                'rate_limit': 1,
                'headers': {}
            },
            'unpaywall': {
                'base_url': 'https://api.unpaywall.org/v2',
                'rate_limit': 1,
                'headers': {'User-Agent': 'TruffleKG/1.0 (mailto:research@example.com)'}
            }
        }
        
        # Setup session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Rate limiting
        self.last_request = {api: datetime.min for api in self.apis}
        
    def _rate_limit(self, api_name: str):
        """Apply rate limiting for API requests"""
        now = datetime.now()
        last = self.last_request[api_name]
        min_interval = 1.0 / self.apis[api_name]['rate_limit']
        
        if (now - last).total_seconds() < min_interval:
            time.sleep(min_interval - (now - last).total_seconds())
        
        self.last_request[api_name] = datetime.now()
    
    @backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_tries=3)
    def _make_request(self, api_name: str, url: str, params: Dict = None) -> requests.Response:
        """Make a rate-limited API request"""
        self._rate_limit(api_name)
        
        headers = self.apis[api_name]['headers'].copy()
        response = self.session.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        return response
    
    def search_openalex(self, query: str, max_results: int = 1000) -> List[PaperMetadata]:
        """Search OpenAlex for papers"""
        logger.info(f"Searching OpenAlex for: {query}")
        papers = []
        
        page_size = 200
        cursor = '*'
        
        while len(papers) < max_results:
            params = {
                'search': query,
                'per_page': min(page_size, max_results - len(papers)),
                'cursor': cursor,
                'filter': 'type:journal-article,type:preprint',
                'sort': 'publication_date:desc'
            }
            
            try:
                response = self._make_request('openalex', f"{self.apis['openalex']['base_url']}/works", params)
                data = response.json()
                
                for work in data.get('results', []):
                    paper = self._parse_openalex_work(work)
                    if paper:
                        papers.append(paper)
                
                # Check if there are more results
                if 'meta' in data and 'next_cursor' in data['meta']:
                    cursor = data['meta']['next_cursor']
                else:
                    break
                    
            except Exception as e:
                logger.error(f"Error searching OpenAlex: {e}")
                break
        
        logger.info(f"Found {len(papers)} papers from OpenAlex")
        return papers
    
    def _parse_openalex_work(self, work: Dict) -> Optional[PaperMetadata]:
        """Parse OpenAlex work into PaperMetadata"""
        try:
            # Extract basic info
            paper_id = work.get('id', '').split('/')[-1]
            title = work.get('title', '')
            year = work.get('publication_year', 0)
            
            # Extract DOI
            doi = None
            if 'doi' in work:
                doi = work['doi'].replace('https://doi.org/', '')
            
            # Extract venue
            venue = 'Unknown'
            if 'primary_location' in work and work['primary_location']:
                venue = work['primary_location'].get('source', {}).get('display_name', 'Unknown')
            
            # Extract authors
            authors = []
            if 'authorships' in work:
                for authorship in work['authorships']:
                    if 'author' in authorship and 'display_name' in authorship['author']:
                        authors.append(authorship['author']['display_name'])
            
            # Extract keywords
            keywords = []
            if 'concepts' in work:
                for concept in work['concepts'][:10]:  # Top 10 concepts
                    if 'display_name' in concept:
                        keywords.append(concept['display_name'])
            
            # Extract abstract
            abstract = work.get('abstract_inverted_index', {})
            if abstract:
                # Reconstruct abstract from inverted index
                abstract_text = self._reconstruct_abstract(abstract)
            else:
                abstract_text = None
            
            # Extract PDF URL
            url_pdf = None
            if 'open_access' in work and work['open_access']:
                if 'primary_location' in work and work['primary_location']:
                    url_pdf = work['primary_location'].get('pdf_url')
            
            # Extract license
            license_info = None
            if 'open_access' in work and work['open_access']:
                if 'primary_location' in work and work['primary_location']:
                    license_info = work['primary_location'].get('license')
            
            # Extract citation count
            citation_count = work.get('cited_by_count', 0)
            
            return PaperMetadata(
                paper_id=paper_id,
                doi=doi,
                title=title,
                year=year,
                venue=venue,
                url_pdf=url_pdf,
                source='openalex',
                license=license_info,
                authors=authors,
                keywords=keywords,
                abstract=abstract_text,
                sections_json={},
                fetched_at=datetime.now(),
                url_source=work.get('id', ''),
                citation_count=citation_count,
                open_access=work.get('open_access', {}).get('is_oa', False)
            )
            
        except Exception as e:
            logger.error(f"Error parsing OpenAlex work: {e}")
            return None
    
    def _reconstruct_abstract(self, abstract_index: Dict) -> str:
        """Reconstruct abstract from inverted index"""
        try:
            # Create a list of (position, word) tuples
            words = []
            for word, positions in abstract_index.items():
                for pos in positions:
                    words.append((pos, word))
            
            # Sort by position and join
            words.sort(key=lambda x: x[0])
            return ' '.join([word for pos, word in words])
        except:
            return ""
    
    def search_crossref(self, query: str, max_results: int = 1000) -> List[PaperMetadata]:
        """Search Crossref for papers"""
        logger.info(f"Searching Crossref for: {query}")
        papers = []
        
        params = {
            'query': query,
            'rows': min(1000, max_results),
            'sort': 'published',
            'order': 'desc'
        }
        
        try:
            response = self._make_request('crossref', f"{self.apis['crossref']['base_url']}/works", params)
            data = response.json()
            
            for item in data.get('message', {}).get('items', []):
                paper = self._parse_crossref_item(item)
                if paper:
                    papers.append(paper)
                    
        except Exception as e:
            logger.error(f"Error searching Crossref: {e}")
        
        logger.info(f"Found {len(papers)} papers from Crossref")
        return papers
    
    def _parse_crossref_item(self, item: Dict) -> Optional[PaperMetadata]:
        """Parse Crossref item into PaperMetadata"""
        try:
            # Extract basic info
            paper_id = item.get('DOI', '').replace('https://doi.org/', '')
            title = item.get('title', [''])[0] if item.get('title') else ''
            year = item.get('published-print', {}).get('date-parts', [[0]])[0][0]
            
            # Extract venue
            venue = 'Unknown'
            if 'container-title' in item and item['container-title']:
                venue = item['container-title'][0]
            
            # Extract authors
            authors = []
            if 'author' in item:
                for author in item['author']:
                    if 'given' in author and 'family' in author:
                        authors.append(f"{author['given']} {author['family']}")
                    elif 'family' in author:
                        authors.append(author['family'])
            
            # Extract abstract
            abstract = item.get('abstract', '')
            
            # Extract PDF URL (if available)
            url_pdf = None
            if 'link' in item:
                for link in item['link']:
                    if link.get('content-type') == 'application/pdf':
                        url_pdf = link.get('URL')
                        break
            
            return PaperMetadata(
                paper_id=paper_id,
                doi=paper_id,
                title=title,
                year=year,
                venue=venue,
                url_pdf=url_pdf,
                source='crossref',
                license=None,
                authors=authors,
                keywords=[],
                abstract=abstract,
                sections_json={},
                fetched_at=datetime.now(),
                url_source=f"https://doi.org/{paper_id}",
                citation_count=item.get('is-referenced-by-count', 0)
            )
            
        except Exception as e:
            logger.error(f"Error parsing Crossref item: {e}")
            return None
    
    def search_pubmed(self, query: str, max_results: int = 1000) -> List[PaperMetadata]:
        """Search PubMed for biomedical papers"""
        logger.info(f"Searching PubMed for: {query}")
        papers = []
        
        try:
            # Search for PMIDs
            search_params = {
                'db': 'pubmed',
                'term': query,
                'retmax': min(10000, max_results),
                'retmode': 'json'
            }
            
            search_response = self._make_request('pubmed', f"{self.apis['pubmed']['base_url']}/esearch.fcgi", search_params)
            search_data = search_response.json()
            
            pmids = search_data.get('esearchresult', {}).get('idlist', [])
            
            if not pmids:
                return papers
            
            # Fetch details for each PMID
            fetch_params = {
                'db': 'pubmed',
                'id': ','.join(pmids[:max_results]),
                'retmode': 'xml'
            }
            
            fetch_response = self._make_request('pubmed', f"{self.apis['pubmed']['base_url']}/efetch.fcgi", fetch_params)
            
            # Parse XML response (simplified)
            papers = self._parse_pubmed_xml(fetch_response.text)
            
        except Exception as e:
            logger.error(f"Error searching PubMed: {e}")
        
        logger.info(f"Found {len(papers)} papers from PubMed")
        return papers
    
    def _parse_pubmed_xml(self, xml_content: str) -> List[PaperMetadata]:
        """Parse PubMed XML response"""
        papers = []
        # This is a simplified parser - in production, use proper XML parsing
        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(xml_content)
            
            for article in root.findall('.//PubmedArticle'):
                try:
                    # Extract basic info
                    pmid = article.find('.//PMID').text if article.find('.//PMID') is not None else ''
                    title = article.find('.//ArticleTitle').text if article.find('.//ArticleTitle') is not None else ''
                    
                    # Extract year
                    year = 0
                    pub_date = article.find('.//PubDate')
                    if pub_date is not None:
                        year_elem = pub_date.find('Year')
                        if year_elem is not None:
                            year = int(year_elem.text)
                    
                    # Extract venue
                    venue = 'Unknown'
                    journal = article.find('.//Journal/Title')
                    if journal is not None:
                        venue = journal.text
                    
                    # Extract authors
                    authors = []
                    for author in article.findall('.//Author'):
                        last_name = author.find('LastName')
                        first_name = author.find('ForeName')
                        if last_name is not None:
                            if first_name is not None:
                                authors.append(f"{first_name.text} {last_name.text}")
                            else:
                                authors.append(last_name.text)
                    
                    # Extract abstract
                    abstract = ''
                    abstract_elem = article.find('.//Abstract/AbstractText')
                    if abstract_elem is not None:
                        abstract = abstract_elem.text or ''
                    
                    paper = PaperMetadata(
                        paper_id=f"pmid_{pmid}",
                        doi=None,
                        title=title,
                        year=year,
                        venue=venue,
                        url_pdf=None,
                        source='pubmed',
                        license=None,
                        authors=authors,
                        keywords=[],
                        abstract=abstract,
                        sections_json={},
                        fetched_at=datetime.now(),
                        url_source=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                        citation_count=0
                    )
                    papers.append(paper)
                    
                except Exception as e:
                    logger.error(f"Error parsing PubMed article: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error parsing PubMed XML: {e}")
        
        return papers
    
    def search_arxiv(self, query: str, max_results: int = 1000) -> List[PaperMetadata]:
        """Search arXiv for preprints"""
        logger.info(f"Searching arXiv for: {query}")
        papers = []
        
        params = {
            'search_query': query,
            'start': 0,
            'max_results': min(1000, max_results),
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }
        
        try:
            response = self._make_request('arxiv', self.apis['arxiv']['base_url'], params)
            
            # Parse arXiv XML response
            papers = self._parse_arxiv_xml(response.text)
            
        except Exception as e:
            logger.error(f"Error searching arXiv: {e}")
        
        logger.info(f"Found {len(papers)} papers from arXiv")
        return papers
    
    def _parse_arxiv_xml(self, xml_content: str) -> List[PaperMetadata]:
        """Parse arXiv XML response"""
        papers = []
        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(xml_content)
            
            for entry in root.findall('.//{http://www.w3.org/2005/Atom}entry'):
                try:
                    # Extract basic info
                    arxiv_id = entry.find('.//{http://arxiv.org/schemas/atom}id').text.split('/')[-1]
                    title = entry.find('.//{http://www.w3.org/2005/Atom}title').text
                    
                    # Extract year
                    year = 0
                    published = entry.find('.//{http://www.w3.org/2005/Atom}published')
                    if published is not None:
                        year = int(published.text[:4])
                    
                    # Extract authors
                    authors = []
                    for author in entry.findall('.//{http://www.w3.org/2005/Atom}author'):
                        name = author.find('.//{http://www.w3.org/2005/Atom}name')
                        if name is not None:
                            authors.append(name.text)
                    
                    # Extract abstract
                    abstract = ''
                    summary = entry.find('.//{http://www.w3.org/2005/Atom}summary')
                    if summary is not None:
                        abstract = summary.text
                    
                    # Extract PDF URL
                    url_pdf = None
                    for link in entry.findall('.//{http://www.w3.org/2005/Atom}link'):
                        if link.get('type') == 'application/pdf':
                            url_pdf = link.get('href')
                            break
                    
                    paper = PaperMetadata(
                        paper_id=f"arxiv_{arxiv_id}",
                        doi=None,
                        title=title,
                        year=year,
                        venue='arXiv',
                        url_pdf=url_pdf,
                        source='arxiv',
                        license='arXiv',
                        authors=authors,
                        keywords=[],
                        abstract=abstract,
                        sections_json={},
                        fetched_at=datetime.now(),
                        url_source=f"https://arxiv.org/abs/{arxiv_id}",
                        citation_count=0
                    )
                    papers.append(paper)
                    
                except Exception as e:
                    logger.error(f"Error parsing arXiv entry: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error parsing arXiv XML: {e}")
        
        return papers
    
    def search_patents(self, query: str, max_results: int = 1000) -> List[PatentMetadata]:
        """Search patent databases (simplified implementation)"""
        logger.info(f"Searching patents for: {query}")
        patents = []
        
        # This is a simplified implementation
        # In production, you would integrate with EPO OPS and WIPO PATENTSCOPE APIs
        
        logger.info(f"Found {len(patents)} patents")
        return patents
    
    def download_pdf(self, paper: PaperMetadata) -> Optional[Path]:
        """Download PDF for a paper"""
        if not paper.url_pdf:
            return None
        
        try:
            # Create filename
            filename = f"{paper.paper_id}.pdf"
            filepath = self.output_dir / "pdfs" / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Download PDF
            response = self.session.get(paper.url_pdf, timeout=30)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Downloaded PDF: {filename}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error downloading PDF for {paper.paper_id}: {e}")
            return None
    
    def save_metadata(self, papers: List[PaperMetadata], patents: List[PatentMetadata]):
        """Save metadata to JSON files"""
        # Save papers metadata
        papers_data = [asdict(paper) for paper in papers]
        papers_file = self.output_dir / "papers_metadata.json"
        
        # Convert datetime objects to strings
        for paper_data in papers_data:
            if 'fetched_at' in paper_data:
                paper_data['fetched_at'] = paper_data['fetched_at'].isoformat()
        
        with open(papers_file, 'w', encoding='utf-8') as f:
            json.dump(papers_data, f, indent=2, ensure_ascii=False)
        
        # Save patents metadata
        patents_data = [asdict(patent) for patent in patents]
        patents_file = self.output_dir / "patents_metadata.json"
        
        for patent_data in patents_data:
            if 'fetched_at' in patent_data:
                patent_data['fetched_at'] = patent_data['fetched_at'].isoformat()
        
        with open(patents_file, 'w', encoding='utf-8') as f:
            json.dump(patents_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved metadata: {len(papers)} papers, {len(patents)} patents")
    
    def run_discovery(self, queries: Dict[str, str], max_results_per_query: int = 1000):
        """Run the complete discovery process"""
        logger.info("Starting literature discovery process...")
        
        all_papers = []
        all_patents = []
        
        # Search papers
        paper_queries = queries.get('papers', {})
        for query_name, query in paper_queries.items():
            logger.info(f"Searching papers with query: {query_name}")
            
            # Search multiple sources
            papers = []
            papers.extend(self.search_openalex(query, max_results_per_query // 4))
            papers.extend(self.search_crossref(query, max_results_per_query // 4))
            papers.extend(self.search_pubmed(query, max_results_per_query // 4))
            papers.extend(self.search_arxiv(query, max_results_per_query // 4))
            
            all_papers.extend(papers)
        
        # Search patents
        patent_queries = queries.get('patents', {})
        for query_name, query in patent_queries.items():
            logger.info(f"Searching patents with query: {query_name}")
            patents = self.search_patents(query, max_results_per_query)
            all_patents.extend(patents)
        
        # Remove duplicates based on DOI or title
        unique_papers = self._deduplicate_papers(all_papers)
        
        # Download PDFs for papers with open access
        logger.info("Downloading PDFs...")
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_paper = {
                executor.submit(self.download_pdf, paper): paper 
                for paper in unique_papers if paper.url_pdf
            }
            
            for future in as_completed(future_to_paper):
                paper = future_to_paper[future]
                try:
                    pdf_path = future.result()
                    if pdf_path:
                        paper.sections_json['pdf_path'] = str(pdf_path)
                except Exception as e:
                    logger.error(f"Error downloading PDF for {paper.paper_id}: {e}")
        
        # Save metadata
        self.save_metadata(unique_papers, all_patents)
        
        logger.info(f"Discovery complete: {len(unique_papers)} papers, {len(all_patents)} patents")
        return unique_papers, all_patents
    
    def _deduplicate_papers(self, papers: List[PaperMetadata]) -> List[PaperMetadata]:
        """Remove duplicate papers based on DOI or title similarity"""
        seen_dois = set()
        seen_titles = set()
        unique_papers = []
        
        for paper in papers:
            # Check DOI first
            if paper.doi and paper.doi in seen_dois:
                continue
            
            # Check title similarity (simple approach)
            title_lower = paper.title.lower().strip()
            if title_lower in seen_titles:
                continue
            
            # Add to seen sets
            if paper.doi:
                seen_dois.add(paper.doi)
            seen_titles.add(title_lower)
            unique_papers.append(paper)
        
        return unique_papers

def main():
    """Example usage of the literature downloader"""
    config = {
        'output_dir': 'data/literature',
        'max_results_per_query': 100
    }
    
    downloader = LiteratureDownloader(config)
    
    queries = {
        'papers': {
            'truffle_mycorrhiza': 'Tuber OR truffle* AND (mycorrhiz* OR ectomycorrhiz*) AND (hydropon* OR soilles* OR aeropon* OR bioreactor OR "nutrient solution")',
            'ph_ec_controls': '("pH" OR "EC" OR "dissolved oxygen" OR "nutrient solution" OR "Hoagland" OR "phosphate") AND (mycorrhiza* OR Tuber*)'
        },
        'patents': {
            'soilless_cultivation': 'truffle AND (hydroponic OR soilless OR aeroponic)',
            'mycorrhiza_methods': 'mycorrhiza AND (inoculation OR cultivation OR method)'
        }
    }
    
    papers, patents = downloader.run_discovery(queries)
    print(f"Found {len(papers)} papers and {len(patents)} patents")

if __name__ == "__main__":
    main()