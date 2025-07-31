"""
Utility functions for web scraping pipeline.
"""
import re
from enum import Enum
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import spacy
from loguru import logger

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class WebsiteType(Enum):
    """Enumeration of supported website types."""
    BUSINESS = "business"
    NEWS = "news"
    WIKIPEDIA = "wikipedia"
    UNKNOWN = "unknown"


def detect_website_type(url: str, soup: BeautifulSoup) -> WebsiteType:
    """
    Detect the type of website based on URL and HTML content.
    
    Args:
        url: The URL of the website
        soup: BeautifulSoup object of the HTML content
        
    Returns:
        WebsiteType enum value
    """
    # Check URL patterns first
    url_lower = url.lower()
    
    # Wikipedia detection
    if 'wikipedia.org' in url_lower or 'wiki' in url_lower:
        return WebsiteType.WIKIPEDIA
    
    # News website patterns
    news_domains = [
        'news', 'bbc', 'cnn', 'reuters', 'ap', 'bloomberg', 'nytimes',
        'washingtonpost', 'theguardian', 'forbes', 'techcrunch', 'ars',
        'wired', 'verge', 'engadget', 'mashable', 'huffpost', 'vox'
    ]
    
    if any(domain in url_lower for domain in news_domains):
        return WebsiteType.NEWS
    
    # Check HTML content for indicators
    html_text = soup.get_text().lower()
    
    # Wikipedia indicators
    wiki_indicators = [
        'mw-content-text', 'mw-parser-output', 'firstHeading',
        'edit this page', 'view history', 'talk page'
    ]
    if any(indicator in html_text for indicator in wiki_indicators):
        return WebsiteType.WIKIPEDIA
    
    # News indicators
    news_indicators = [
        'breaking news', 'latest news', 'headlines', 'byline',
        'published on', 'article', 'story', 'reporter', 'journalist'
    ]
    if any(indicator in html_text for indicator in news_indicators):
        return WebsiteType.NEWS
    
    # Business indicators
    business_indicators = [
        'about us', 'contact us', 'services', 'products',
        'company', 'business', 'enterprise', 'solutions',
        'get quote', 'request demo', 'pricing'
    ]
    if any(indicator in html_text for indicator in business_indicators):
        return WebsiteType.BUSINESS
    
    # Default to business if no clear indicators
    return WebsiteType.BUSINESS


def clean_text(text: str) -> str:
    """
    Clean and preprocess text content.
    
    Args:
        text: Raw text content
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}]', '', text)
    
    # Remove multiple periods
    text = re.sub(r'\.{2,}', '.', text)
    
    # Remove multiple spaces
    text = re.sub(r' +', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def extract_sentences(text: str) -> List[str]:
    """
    Extract sentences from text using NLTK.
    
    Args:
        text: Input text
        
    Returns:
        List of sentences
    """
    try:
        sentences = sent_tokenize(text)
        return [s.strip() for s in sentences if s.strip()]
    except Exception as e:
        logger.warning(f"Error tokenizing sentences: {e}")
        # Fallback to simple sentence splitting
        return [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]


def extract_paragraphs(text: str) -> List[str]:
    """
    Extract paragraphs from text.
    
    Args:
        text: Input text
        
    Returns:
        List of paragraphs
    """
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    return paragraphs


def get_text_statistics(text: str) -> Dict[str, Any]:
    """
    Get statistics about the text content.
    
    Args:
        text: Input text
        
    Returns:
        Dictionary with text statistics
    """
    if not text:
        return {
            'char_count': 0,
            'word_count': 0,
            'sentence_count': 0,
            'paragraph_count': 0,
            'avg_sentence_length': 0,
            'avg_paragraph_length': 0
        }
    
    sentences = extract_sentences(text)
    paragraphs = extract_paragraphs(text)
    words = word_tokenize(text)
    
    return {
        'char_count': len(text),
        'word_count': len(words),
        'sentence_count': len(sentences),
        'paragraph_count': len(paragraphs),
        'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
        'avg_paragraph_length': len(words) / len(paragraphs) if paragraphs else 0
    }


def is_meaningful_text(text: str, min_length: int = 50) -> bool:
    """
    Check if text contains meaningful content.
    
    Args:
        text: Input text
        min_length: Minimum length for meaningful text
        
    Returns:
        True if text is meaningful
    """
    if not text or len(text.strip()) < min_length:
        return False
    
    # Check if text contains mostly special characters or numbers
    alpha_ratio = len(re.findall(r'[a-zA-Z]', text)) / len(text)
    if alpha_ratio < 0.3:
        return False
    
    # Check for repetitive content
    words = word_tokenize(text.lower())
    if len(set(words)) / len(words) < 0.3:
        return False
    
    return True


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    Extract keywords from text using simple frequency analysis.
    
    Args:
        text: Input text
        max_keywords: Maximum number of keywords to extract
        
    Returns:
        List of keywords
    """
    try:
        # Tokenize and clean words
        words = word_tokenize(text.lower())
        
        # Remove stopwords and short words
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word.isalpha() and word not in stop_words and len(word) > 2]
        
        # Count word frequencies
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and return top keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:max_keywords]]
        
    except Exception as e:
        logger.warning(f"Error extracting keywords: {e}")
        return []


def normalize_url(url: str) -> str:
    """
    Normalize URL for consistent processing.
    
    Args:
        url: Input URL
        
    Returns:
        Normalized URL
    """
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    # Remove trailing slash
    url = url.rstrip('/')
    
    return url


def validate_url(url: str) -> bool:
    """
    Validate if URL is properly formatted.
    
    Args:
        url: Input URL
        
    Returns:
        True if URL is valid
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False