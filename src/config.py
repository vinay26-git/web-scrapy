"""
Configuration settings for the Web Scraper pipeline.
"""
import os
from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path

@dataclass
class ScrapingConfig:
    """Configuration for web scraping settings."""
    # Selenium settings
    headless: bool = True
    page_load_timeout: int = 30
    implicit_wait: int = 10
    window_size: tuple = (1920, 1080)
    
    # Request settings
    max_retries: int = 3
    retry_delay: int = 2
    request_timeout: int = 30
    
    # User agents for different browsers
    user_agents: List[str] = None
    
    def __post_init__(self):
        if self.user_agents is None:
            self.user_agents = [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ]

@dataclass
class ChunkingConfig:
    """Configuration for text chunking settings."""
    # Recursive chunking settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: List[str] = None
    
    # Content-aware settings
    min_chunk_size: int = 100
    max_chunk_size: int = 2000
    preserve_paragraphs: bool = True
    preserve_sentences: bool = True
    
    def __post_init__(self):
        if self.separators is None:
            self.separators = [
                "\n\n",  # Paragraphs
                "\n",    # Lines
                ". ",    # Sentences
                "! ",    # Exclamations
                "? ",    # Questions
                "; ",    # Semicolons
                ", ",    # Commas
                " ",     # Words
                ""       # Characters
            ]

@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32
    max_length: int = 512
    normalize_embeddings: bool = True
    device: str = "auto"  # "auto", "cpu", "cuda"
    
    # Cache settings
    cache_dir: str = "./embeddings_cache"
    save_embeddings: bool = True

@dataclass
class WebsiteConfig:
    """Configuration for different website types."""
    business_selectors: Dict[str, List[str]] = None
    news_selectors: Dict[str, List[str]] = None
    wikipedia_selectors: Dict[str, List[str]] = None
    
    def __post_init__(self):
        if self.business_selectors is None:
            self.business_selectors = {
                "title": ["h1", ".title", ".page-title", "title"],
                "content": ["main", ".content", ".main-content", "article", ".post-content"],
                "description": [".description", ".summary", ".excerpt", "meta[name='description']"],
                "contact": [".contact", ".contact-info", ".footer", ".address"],
                "services": [".services", ".products", ".offerings", ".features"]
            }
        
        if self.news_selectors is None:
            self.news_selectors = {
                "title": ["h1", ".headline", ".title", ".article-title"],
                "content": ["article", ".article-content", ".story-content", ".post-content"],
                "author": [".author", ".byline", ".writer", "meta[name='author']"],
                "date": [".date", ".published", ".timestamp", "time"],
                "summary": [".summary", ".excerpt", ".lead", ".intro"]
            }
        
        if self.wikipedia_selectors is None:
            self.wikipedia_selectors = {
                "title": ["h1.firstHeading", ".mw-page-title-main"],
                "content": ["#mw-content-text", ".mw-parser-output"],
                "toc": ["#toc", ".toc"],
                "infobox": [".infobox", ".wikitable"],
                "references": ["#References", ".reflist"]
            }

@dataclass
class PipelineConfig:
    """Main configuration for the entire pipeline."""
    scraping: ScrapingConfig = None
    chunking: ChunkingConfig = None
    embedding: EmbeddingConfig = None
    website: WebsiteConfig = None
    
    # Output settings
    output_dir: str = "./output"
    log_level: str = "INFO"
    save_raw_html: bool = True
    save_cleaned_text: bool = True
    save_chunks: bool = True
    
    def __post_init__(self):
        if self.scraping is None:
            self.scraping = ScrapingConfig()
        if self.chunking is None:
            self.chunking = ChunkingConfig()
        if self.embedding is None:
            self.embedding = EmbeddingConfig()
        if self.website is None:
            self.website = WebsiteConfig()
        
        # Create output directories
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.embedding.cache_dir).mkdir(parents=True, exist_ok=True)

# Default configuration
DEFAULT_CONFIG = PipelineConfig()