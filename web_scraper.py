"""
Web Scraper Pipeline for Business, News, and Wikipedia Websites
Author: Full-Stack Developer (Google, Amazon, Microsoft experience)
"""

import time
import re
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from urllib.parse import urlparse
import logging

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import requests
from loguru import logger


@dataclass
class ScrapedContent:
    """Data class to store scraped content with metadata."""
    url: str
    title: str
    content: str
    website_type: str
    metadata: Dict
    raw_html: str
    timestamp: float


class WebScraper:
    """
    Advanced web scraper with support for business, news, and Wikipedia websites.
    Uses Selenium for dynamic content and handles different website structures.
    """
    
    def __init__(self, headless: bool = True, timeout: int = 30):
        """
        Initialize the web scraper.
        
        Args:
            headless: Run browser in headless mode
            timeout: Page load timeout in seconds
        """
        self.timeout = timeout
        self.driver = None
        self.headless = headless
        self._setup_driver()
        
        # Website type patterns
        self.website_patterns = {
            'business': [
                r'\.com$', r'\.org$', r'\.net$', r'\.io$', r'\.co$',
                r'company', r'corp', r'inc', r'llc', r'ltd'
            ],
            'news': [
                r'news', r'media', r'press', r'journal', r'times',
                r'post', r'tribune', r'herald', r'gazette', r'chronicle'
            ],
            'wikipedia': [
                r'wikipedia\.org', r'wiki\.', r'\.wikipedia\.'
            ]
        }
        
        logger.info("WebScraper initialized successfully")
    
    def _setup_driver(self):
        """Setup Chrome driver with optimized settings."""
        try:
            chrome_options = Options()
            
            if self.headless:
                chrome_options.add_argument("--headless")
            
            # Performance optimizations
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--disable-extensions")
            chrome_options.add_argument("--disable-plugins")
            chrome_options.add_argument("--disable-images")
            chrome_options.add_argument("--disable-javascript")
            chrome_options.add_argument("--disable-css")
            chrome_options.add_argument("--disable-web-security")
            chrome_options.add_argument("--disable-features=VizDisplayCompositor")
            
            # User agent
            chrome_options.add_argument("--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
            
            # Memory optimizations
            chrome_options.add_argument("--memory-pressure-off")
            chrome_options.add_argument("--max_old_space_size=4096")
            
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            self.driver.set_page_load_timeout(self.timeout)
            
            logger.info("Chrome driver setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup Chrome driver: {e}")
            raise
    
    def _detect_website_type(self, url: str) -> str:
        """
        Detect website type based on URL patterns.
        
        Args:
            url: URL to analyze
            
        Returns:
            Website type: 'business', 'news', 'wikipedia', or 'unknown'
        """
        url_lower = url.lower()
        
        for website_type, patterns in self.website_patterns.items():
            for pattern in patterns:
                if re.search(pattern, url_lower):
                    return website_type
        
        return 'unknown'
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.
        
        Args:
            text: Raw text to clean
            
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
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _extract_title(self, soup: BeautifulSoup, url: str) -> str:
        """
        Extract page title using multiple strategies.
        
        Args:
            soup: BeautifulSoup object
            url: URL of the page
            
        Returns:
            Extracted title
        """
        # Try multiple title selectors
        title_selectors = [
            'h1',
            'title',
            '[class*="title"]',
            '[id*="title"]',
            '.headline',
            '.article-title',
            '.post-title'
        ]
        
        for selector in title_selectors:
            title_elem = soup.select_one(selector)
            if title_elem and title_elem.get_text().strip():
                title = self._clean_text(title_elem.get_text())
                if len(title) > 10:  # Minimum title length
                    return title
        
        # Fallback to URL-based title
        parsed_url = urlparse(url)
        return parsed_url.path.replace('/', ' ').replace('-', ' ').strip()
    
    def _extract_content_business(self, soup: BeautifulSoup) -> str:
        """
        Extract content from business websites.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            Extracted content
        """
        # Common business website content selectors
        content_selectors = [
            'main',
            'article',
            '.content',
            '.main-content',
            '.post-content',
            '.entry-content',
            '.page-content',
            '#content',
            '#main'
        ]
        
        content = ""
        
        for selector in content_selectors:
            elements = soup.select(selector)
            for element in elements:
                # Remove navigation, footer, sidebar
                for unwanted in element.select('nav, footer, .sidebar, .navigation, .menu'):
                    unwanted.decompose()
                
                text = element.get_text()
                if len(text) > 100:  # Minimum content length
                    content += text + " "
        
        return self._clean_text(content)
    
    def _extract_content_news(self, soup: BeautifulSoup) -> str:
        """
        Extract content from news websites.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            Extracted content
        """
        # News-specific content selectors
        content_selectors = [
            '.article-body',
            '.story-body',
            '.post-content',
            '.entry-content',
            '.article-content',
            '.news-content',
            '.content-body',
            'article',
            '.article-text'
        ]
        
        content = ""
        
        for selector in content_selectors:
            elements = soup.select(selector)
            for element in elements:
                # Remove ads, social media buttons, related articles
                for unwanted in element.select('.ad, .advertisement, .social-share, .related-articles, .comments'):
                    unwanted.decompose()
                
                text = element.get_text()
                if len(text) > 200:  # News articles should be longer
                    content += text + " "
        
        return self._clean_text(content)
    
    def _extract_content_wikipedia(self, soup: BeautifulSoup) -> str:
        """
        Extract content from Wikipedia pages.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            Extracted content
        """
        # Wikipedia-specific content extraction
        content_selectors = [
            '#mw-content-text',
            '.mw-parser-output'
        ]
        
        content = ""
        
        for selector in content_selectors:
            element = soup.select_one(selector)
            if element:
                # Remove navigation, references, external links
                for unwanted in element.select('.navbox, .reflist, .external, .mw-editsection, .toc'):
                    unwanted.decompose()
                
                # Get paragraphs
                paragraphs = element.find_all('p')
                for p in paragraphs:
                    text = p.get_text()
                    if len(text) > 50:  # Minimum paragraph length
                        content += text + " "
        
        return self._clean_text(content)
    
    def scrape_url(self, url: str) -> Optional[ScrapedContent]:
        """
        Scrape content from a given URL.
        
        Args:
            url: URL to scrape
            
        Returns:
            ScrapedContent object or None if failed
        """
        try:
            logger.info(f"Starting to scrape: {url}")
            
            # Detect website type
            website_type = self._detect_website_type(url)
            logger.info(f"Detected website type: {website_type}")
            
            # Load page
            self.driver.get(url)
            
            # Wait for page to load
            WebDriverWait(self.driver, self.timeout).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Get page source
            html_content = self.driver.page_source
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract title
            title = self._extract_title(soup, url)
            
            # Extract content based on website type
            if website_type == 'business':
                content = self._extract_content_business(soup)
            elif website_type == 'news':
                content = self._extract_content_news(soup)
            elif website_type == 'wikipedia':
                content = self._extract_content_wikipedia(soup)
            else:
                # Generic extraction for unknown types
                content = self._extract_content_business(soup)
            
            # Create metadata
            metadata = {
                'website_type': website_type,
                'url': url,
                'title': title,
                'content_length': len(content),
                'word_count': len(content.split()),
                'scraped_at': time.time()
            }
            
            scraped_content = ScrapedContent(
                url=url,
                title=title,
                content=content,
                website_type=website_type,
                metadata=metadata,
                raw_html=html_content,
                timestamp=time.time()
            )
            
            logger.info(f"Successfully scraped {url}. Content length: {len(content)} characters")
            return scraped_content
            
        except TimeoutException:
            logger.error(f"Timeout while scraping {url}")
            return None
        except WebDriverException as e:
            logger.error(f"WebDriver error while scraping {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error while scraping {url}: {e}")
            return None
    
    def scrape_multiple_urls(self, urls: List[str]) -> List[ScrapedContent]:
        """
        Scrape multiple URLs.
        
        Args:
            urls: List of URLs to scrape
            
        Returns:
            List of ScrapedContent objects
        """
        results = []
        
        for url in urls:
            try:
                content = self.scrape_url(url)
                if content:
                    results.append(content)
                time.sleep(2)  # Be respectful to servers
            except Exception as e:
                logger.error(f"Error scraping {url}: {e}")
                continue
        
        return results
    
    def close(self):
        """Close the web driver."""
        if self.driver:
            self.driver.quit()
            logger.info("Web driver closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    # Example usage
    urls = [
        "https://www.example.com",
        "https://www.bbc.com/news",
        "https://en.wikipedia.org/wiki/Python_(programming_language)"
    ]
    
    with WebScraper(headless=True) as scraper:
        results = scraper.scrape_multiple_urls(urls)
        for result in results:
            print(f"URL: {result.url}")
            print(f"Title: {result.title}")
            print(f"Content length: {len(result.content)}")
            print(f"Website type: {result.website_type}")
            print("-" * 50)