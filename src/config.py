"""
Configuration settings for the Web Scraper project.
"""
import os
from typing import Dict, List, Optional
from pydantic import BaseSettings, Field
from enum import Enum


class WebsiteType(str, Enum):
    """Supported website types for scraping."""
    BUSINESS = "business"
    NEWS = "news"
    WIKIPEDIA = "wikipedia"


class ScrapingConfig(BaseSettings):
    """Configuration for web scraping settings."""
    
    # Selenium Settings
    headless: bool = Field(True, description="Run browser in headless mode")
    page_load_timeout: int = Field(30, description="Page load timeout in seconds")
    implicit_wait: int = Field(10, description="Implicit wait timeout in seconds")
    user_agent: str = Field(
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        description="User agent string"
    )
    
    # Rate Limiting
    request_delay: float = Field(1.0, description="Delay between requests in seconds")
    max_retries: int = Field(3, description="Maximum number of retries for failed requests")
    
    # Content Filtering
    min_content_length: int = Field(100, description="Minimum content length to process")
    max_content_length: int = Field(50000, description="Maximum content length to process")
    
    class Config:
        env_prefix = "SCRAPER_"


class ChunkingConfig(BaseSettings):
    """Configuration for text chunking."""
    
    # Recursive Chunking Settings
    chunk_size: int = Field(1000, description="Default chunk size in characters")
    chunk_overlap: int = Field(200, description="Overlap between chunks")
    
    # Content-Aware Settings
    use_semantic_splitting: bool = Field(True, description="Enable semantic splitting")
    sentence_similarity_threshold: float = Field(0.7, description="Threshold for sentence similarity")
    
    # Separators for recursive chunking
    separators: List[str] = Field(
        default=["\n\n", "\n", ". ", "! ", "? ", "; ", ": ", ", ", " ", ""],
        description="Text separators in order of preference"
    )
    
    class Config:
        env_prefix = "CHUNK_"


class EmbeddingConfig(BaseSettings):
    """Configuration for embedding generation."""
    
    # Model Settings
    model_name: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence transformer model name"
    )
    batch_size: int = Field(32, description="Batch size for embedding generation")
    max_seq_length: int = Field(512, description="Maximum sequence length")
    
    # Processing Settings
    normalize_embeddings: bool = Field(True, description="Normalize embeddings to unit vectors")
    device: str = Field("cuda", description="Device for model inference (cuda/cpu)")
    
    class Config:
        env_prefix = "EMBED_"


class Config:
    """Main configuration class."""
    
    def __init__(self):
        self.scraping = ScrapingConfig()
        self.chunking = ChunkingConfig()
        self.embedding = EmbeddingConfig()
    
    # Website-specific selectors
    WEBSITE_SELECTORS: Dict[WebsiteType, Dict[str, str]] = {
        WebsiteType.BUSINESS: {
            "title": "h1, .page-title, .entry-title, .post-title, .article-title",
            "content": ".content, .main-content, .entry-content, .post-content, .article-content, main, article",
            "remove": "nav, footer, header, .nav, .footer, .header, .sidebar, .ads, .advertisement, .social-share"
        },
        WebsiteType.NEWS: {
            "title": "h1, .headline, .article-headline, .entry-title, .post-title",
            "content": ".article-body, .entry-content, .post-content, .story-body, .article-content, main article",
            "remove": "nav, footer, header, .nav, .footer, .header, .sidebar, .ads, .advertisement, .social-share, .related-articles"
        },
        WebsiteType.WIKIPEDIA: {
            "title": "h1.firstHeading, .mw-page-title-main",
            "content": "#mw-content-text .mw-parser-output",
            "remove": ".navbox, .infobox, .reference, .reflist, .navbar, .sistersitebox, .mw-editsection"
        }
    }
    
    # Common elements to remove across all website types
    COMMON_REMOVE_SELECTORS = [
        "script", "style", "noscript", "iframe", "object", "embed",
        ".cookie-banner", ".popup", ".modal", ".overlay", ".advertisement"
    ]