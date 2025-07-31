"""
Advanced web scraper using Selenium with support for business, news, and Wikipedia sites.
Implements enterprise-grade patterns with retry logic, error handling, and optimization.
"""
import time
import re
from typing import Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse, urljoin
from dataclasses import dataclass
from contextlib import contextmanager

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import (
    TimeoutException, NoSuchElementException, WebDriverException
)
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from loguru import logger

from .config import Config, WebsiteType


@dataclass
class ScrapedContent:
    """Container for scraped content with metadata."""
    url: str
    title: str
    content: str
    website_type: WebsiteType
    scraped_at: str
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class WebScraperError(Exception):
    """Custom exception for web scraping errors."""
    pass


class WebScraper:
    """
    Enterprise-grade web scraper with Selenium.
    
    Features:
    - Multi-website type support (business, news, Wikipedia)
    - Intelligent content extraction
    - Robust error handling and retry logic
    - Rate limiting and respectful scraping
    - Content quality validation
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.driver: Optional[webdriver.Chrome] = None
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging for the scraper."""
        logger.add(
            "logs/scraper.log",
            rotation="10 MB",
            retention="7 days",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
        )
    
    @contextmanager
    def _get_driver(self):
        """Context manager for WebDriver lifecycle management."""
        driver = None
        try:
            driver = self._create_driver()
            yield driver
        except Exception as e:
            logger.error(f"Driver error: {e}")
            raise
        finally:
            if driver:
                try:
                    driver.quit()
                except Exception as e:
                    logger.warning(f"Error closing driver: {e}")
    
    def _create_driver(self) -> webdriver.Chrome:
        """Create and configure Chrome WebDriver."""
        chrome_options = Options()
        
        # Performance optimizations
        if self.config.scraping.headless:
            chrome_options.add_argument("--headless")
        
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-plugins")
        chrome_options.add_argument("--disable-images")
        chrome_options.add_argument("--disable-javascript")  # For faster loading
        chrome_options.add_argument(f"--user-agent={self.config.scraping.user_agent}")
        
        # Privacy and security
        chrome_options.add_argument("--incognito")
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--disable-features=VizDisplayCompositor")
        
        # Memory optimization
        chrome_options.add_argument("--memory-pressure-off")
        chrome_options.add_argument("--max_old_space_size=4096")
        
        try:
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
            
            # Configure timeouts
            driver.set_page_load_timeout(self.config.scraping.page_load_timeout)
            driver.implicitly_wait(self.config.scraping.implicit_wait)
            
            return driver
            
        except Exception as e:
            logger.error(f"Failed to create WebDriver: {e}")
            raise WebScraperError(f"WebDriver initialization failed: {e}")
    
    def _detect_website_type(self, url: str) -> WebsiteType:
        """
        Intelligently detect website type based on URL patterns and content.
        """
        url_lower = url.lower()
        domain = urlparse(url).netloc.lower()
        
        # Wikipedia detection
        if 'wikipedia.org' in domain:
            return WebsiteType.WIKIPEDIA
        
        # News website patterns
        news_indicators = [
            'news', 'times', 'post', 'herald', 'tribune', 'journal', 
            'guardian', 'reuters', 'bloomberg', 'cnn', 'bbc', 'nbc',
            'abc', 'cbs', 'fox', 'npr', 'associated', 'press'
        ]
        
        if any(indicator in domain for indicator in news_indicators):
            return WebsiteType.NEWS
        
        # Business website patterns (default)
        return WebsiteType.BUSINESS
    
    def _wait_for_content(self, driver: webdriver.Chrome, website_type: WebsiteType) -> bool:
        """Wait for main content to load based on website type."""
        try:
            selectors = self.config.WEBSITE_SELECTORS[website_type]
            content_selector = selectors["content"]
            
            wait = WebDriverWait(driver, 15)
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, content_selector)))
            return True
            
        except TimeoutException:
            logger.warning(f"Content loading timeout for {website_type}")
            return False
    
    def _extract_content(self, driver: webdriver.Chrome, website_type: WebsiteType) -> Tuple[str, str]:
        """
        Extract title and main content using website-specific selectors.
        """
        html = driver.page_source
        soup = BeautifulSoup(html, 'lxml')
        
        selectors = self.config.WEBSITE_SELECTORS[website_type]
        
        # Extract title
        title = self._extract_title(soup, selectors["title"])
        
        # Remove unwanted elements
        self._remove_unwanted_elements(soup, selectors["remove"])
        
        # Extract main content
        content = self._extract_main_content(soup, selectors["content"])
        
        return title, content
    
    def _extract_title(self, soup: BeautifulSoup, title_selectors: str) -> str:
        """Extract page title using multiple selectors."""
        for selector in title_selectors.split(', '):
            title_elem = soup.select_one(selector.strip())
            if title_elem:
                title = title_elem.get_text(strip=True)
                if title:
                    return title
        
        # Fallback to HTML title tag
        title_tag = soup.find('title')
        return title_tag.get_text(strip=True) if title_tag else "No title found"
    
    def _remove_unwanted_elements(self, soup: BeautifulSoup, remove_selectors: str):
        """Remove unwanted elements from the soup."""
        # Website-specific removals
        for selector in remove_selectors.split(', '):
            for element in soup.select(selector.strip()):
                element.decompose()
        
        # Common removals
        for selector in self.config.COMMON_REMOVE_SELECTORS:
            for element in soup.select(selector):
                element.decompose()
    
    def _extract_main_content(self, soup: BeautifulSoup, content_selectors: str) -> str:
        """Extract main content using multiple selectors."""
        content_parts = []
        
        for selector in content_selectors.split(', '):
            content_elem = soup.select_one(selector.strip())
            if content_elem:
                # Extract text while preserving some structure
                text = self._extract_structured_text(content_elem)
                if text:
                    content_parts.append(text)
                    break  # Use first successful extraction
        
        # Fallback: extract from body
        if not content_parts:
            body = soup.find('body')
            if body:
                content_parts.append(self._extract_structured_text(body))
        
        content = ' '.join(content_parts)
        return self._clean_text(content)
    
    def _extract_structured_text(self, element) -> str:
        """Extract text while preserving paragraph structure."""
        text_parts = []
        
        # Process paragraphs and headings
        for tag in element.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']):
            text = tag.get_text(strip=True)
            if text and len(text) > 10:  # Filter out very short text
                text_parts.append(text)
        
        # If no structured content found, get all text
        if not text_parts:
            text_parts.append(element.get_text(separator=' ', strip=True))
        
        return '\n\n'.join(text_parts)
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive newlines
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Remove common junk patterns
        junk_patterns = [
            r'Share this:.*?$',
            r'Follow us on.*?$',
            r'Subscribe to.*?$',
            r'Advertisement\s*',
            r'Cookie.*?[Pp]olicy',
            r'Privacy.*?[Pp]olicy',
        ]
        
        for pattern in junk_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
        
        return text.strip()
    
    def _validate_content(self, content: str) -> bool:
        """Validate if extracted content meets quality criteria."""
        if not content:
            return False
        
        content_length = len(content)
        
        # Check length constraints
        if content_length < self.config.scraping.min_content_length:
            logger.warning(f"Content too short: {content_length} characters")
            return False
        
        if content_length > self.config.scraping.max_content_length:
            logger.warning(f"Content too long: {content_length} characters")
            return False
        
        # Check for reasonable text-to-noise ratio
        words = content.split()
        if len(words) < 20:
            logger.warning("Content has too few words")
            return False
        
        return True
    
    def scrape_url(self, url: str, website_type: Optional[WebsiteType] = None) -> Optional[ScrapedContent]:
        """
        Scrape a single URL with retry logic and error handling.
        
        Args:
            url: URL to scrape
            website_type: Optional website type override
            
        Returns:
            ScrapedContent object or None if scraping failed
        """
        if website_type is None:
            website_type = self._detect_website_type(url)
        
        logger.info(f"Scraping {url} as {website_type.value}")
        
        for attempt in range(self.config.scraping.max_retries):
            try:
                with self._get_driver() as driver:
                    # Navigate to URL
                    driver.get(url)
                    
                    # Wait for content to load
                    if not self._wait_for_content(driver, website_type):
                        logger.warning(f"Content not loaded properly for {url}")
                    
                    # Extract content
                    title, content = self._extract_content(driver, website_type)
                    
                    # Validate content quality
                    if not self._validate_content(content):
                        raise WebScraperError("Content validation failed")
                    
                    # Create result object
                    scraped_content = ScrapedContent(
                        url=url,
                        title=title,
                        content=content,
                        website_type=website_type,
                        scraped_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                        metadata={
                            "content_length": len(content),
                            "word_count": len(content.split()),
                            "attempt": attempt + 1
                        }
                    )
                    
                    logger.success(f"Successfully scraped {url}")
                    return scraped_content
                    
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt < self.config.scraping.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                
        logger.error(f"Failed to scrape {url} after {self.config.scraping.max_retries} attempts")
        return None
    
    def scrape_urls(self, urls: List[str]) -> List[ScrapedContent]:
        """
        Scrape multiple URLs with rate limiting.
        
        Args:
            urls: List of URLs to scrape
            
        Returns:
            List of successfully scraped content
        """
        results = []
        
        for i, url in enumerate(urls):
            logger.info(f"Processing URL {i+1}/{len(urls)}: {url}")
            
            result = self.scrape_url(url)
            if result:
                results.append(result)
            
            # Rate limiting
            if i < len(urls) - 1:  # Don't wait after the last URL
                time.sleep(self.config.scraping.request_delay)
        
        logger.info(f"Successfully scraped {len(results)}/{len(urls)} URLs")
        return results