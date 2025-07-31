"""
Configuration settings for the Web Scraper project.
"""

from pydantic import BaseModel
from typing import Dict, List, Optional
import os

class ScrapingConfig(BaseModel):
    """Configuration for web scraping."""
    
    # Selenium settings
    headless: bool = True
    implicit_wait: int = 10
    page_load_timeout: int = 30
    window_size: tuple = (1920, 1080)
    
    # User agents for different site types
    user_agents: Dict[str, str] = {
        "business": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "news": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "wikipedia": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    # Site-specific selectors
    content_selectors: Dict[str, Dict[str, List[str]]] = {
        "business": {
            "main_content": ["main", ".main-content", ".content", "#content", "article", ".post-content"],
            "remove": [".advertisement", ".ads", ".social-share", ".cookie-banner", "nav", "footer", ".sidebar"]
        },
        "news": {
            "main_content": ["article", ".article-content", ".story-content", ".post-content", "main", ".content"],
            "remove": [".advertisement", ".ads", ".social-share", ".related-articles", "nav", "footer", ".comments"]
        },
        "wikipedia": {
            "main_content": ["#mw-content-text", ".mw-parser-output", "#content"],
            "remove": [".navbox", ".metadata", ".ambox", ".infobox", ".thumb", "sup", ".reference"]
        }
    }

class ChunkingConfig(BaseModel):
    """Configuration for text chunking."""
    
    # Recursive chunking settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: List[str] = ["\n\n", "\n", ".", "!", "?", ";", ",", " ", ""]
    
    # Content-aware chunking settings
    min_chunk_size: int = 100
    max_chunk_size: int = 1500
    
    # Semantic chunking thresholds
    similarity_threshold: float = 0.5
    
class EmbeddingConfig(BaseModel):
    """Configuration for embedding generation."""
    
    # Model settings
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32
    max_seq_length: int = 512
    
    # Processing settings
    normalize_embeddings: bool = True
    device: str = "cuda" if os.getenv("KAGGLE_KERNEL_RUN_TYPE") else "cpu"

class Config:
    """Main configuration class."""
    
    def __init__(self):
        self.scraping = ScrapingConfig()
        self.chunking = ChunkingConfig()
        self.embedding = EmbeddingConfig()
        
    def get_site_type(self, url: str) -> str:
        """Determine site type based on URL."""
        url_lower = url.lower()
        
        if "wikipedia.org" in url_lower:
            return "wikipedia"
        elif any(news_domain in url_lower for news_domain in [
            "cnn.com", "bbc.com", "reuters.com", "npr.org", "nytimes.com",
            "washingtonpost.com", "theguardian.com", "bloomberg.com", "wsj.com"
        ]):
            return "news"
        else:
            return "business"

# Global config instance
config = Config()