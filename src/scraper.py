"""
Web Scraper module using Selenium with advanced features.
"""
import time
import random
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from urllib.parse import urlparse, urljoin
from pathlib import Path
import json

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException, 
    NoSuchElementException, 
    WebDriverException,
    StaleElementReferenceException
)
from webdriver_manager.chrome import ChromeDriverManager
import undetected_chromedriver as uc
from bs4 import BeautifulSoup
import requests
from loguru import logger

from .config import ScrapingConfig, WebsiteConfig
from .utils import WebsiteType, detect_website_type


class WebScraper:
    """
    Advanced web scraper with Selenium, anti-detection, and website-specific extraction.
    """
    
    def __init__(self, config: ScrapingConfig, website_config: WebsiteConfig):
        self.config = config
        self.website_config = website_config
        self.driver = None
        self.session = requests.Session()
        self._setup_session()
    
    def _setup_session(self):
        """Setup requests session with headers."""
        self.session.headers.update({
            'User-Agent': random.choice(self.config.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
    
    def _setup_driver(self, use_undetected: bool = True):
        """Setup Chrome driver with anti-detection features."""
        try:
            if use_undetected:
                options = uc.ChromeOptions()
            else:
                options = Options()
            
            # Basic options
            if self.config.headless:
                options.add_argument('--headless')
            
            options.add_argument(f'--window-size={self.config.window_size[0]},{self.config.window_size[1]}')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-gpu')
            options.add_argument('--disable-extensions')
            options.add_argument('--disable-plugins')
            options.add_argument('--disable-images')
            options.add_argument('--disable-javascript')  # We'll enable it selectively
            
            # Anti-detection options
            options.add_argument('--disable-blink-features=AutomationControlled')
            options.add_experimental_option("excludeSwitches", ["enable-automation"])
            options.add_experimental_option('useAutomationExtension', False)
            
            # Random user agent
            user_agent = random.choice(self.config.user_agents)
            options.add_argument(f'--user-agent={user_agent}')
            
            # Performance options
            options.add_argument('--disable-logging')
            options.add_argument('--disable-default-apps')
            options.add_argument('--disable-sync')
            options.add_argument('--disable-translate')
            
            if use_undetected:
                self.driver = uc.Chrome(options=options)
            else:
                service = Service(ChromeDriverManager().install())
                self.driver = webdriver.Chrome(service=service, options=options)
            
            # Set timeouts
            self.driver.set_page_load_timeout(self.config.page_load_timeout)
            self.driver.implicitly_wait(self.config.implicit_wait)
            
            # Execute script to remove webdriver property
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            logger.info("Chrome driver setup completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup Chrome driver: {e}")
            raise
    
    def _get_page_with_retry(self, url: str, max_retries: int = None) -> Optional[str]:
        """Get page content with retry mechanism."""
        if max_retries is None:
            max_retries = self.config.max_retries
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempting to fetch {url} (attempt {attempt + 1}/{max_retries})")
                
                # Try with requests first (faster)
                response = self.session.get(url, timeout=self.config.request_timeout)
                response.raise_for_status()
                
                # Check if JavaScript is needed
                if self._needs_javascript(response.text):
                    logger.info("JavaScript required, switching to Selenium")
                    return self._get_page_with_selenium(url)
                
                return response.text
                
            except requests.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    logger.error(f"All request attempts failed for {url}")
                    return None
    
    def _needs_javascript(self, html_content: str) -> bool:
        """Check if the page requires JavaScript to render content."""
        # Simple heuristics to detect if JavaScript is needed
        js_indicators = [
            'window.__INITIAL_STATE__',
            'window.__PRELOADED_STATE__',
            'data-reactroot',
            'ng-app',
            'v-app',
            'id="app"',
            'class="app"',
            'noscript',
            'script src=',
            'window.',
            'document.',
        ]
        
        return any(indicator in html_content for indicator in js_indicators)
    
    def _get_page_with_selenium(self, url: str) -> Optional[str]:
        """Get page content using Selenium."""
        if self.driver is None:
            self._setup_driver()
        
        try:
            logger.info(f"Fetching {url} with Selenium")
            self.driver.get(url)
            
            # Wait for page to load
            WebDriverWait(self.driver, self.config.page_load_timeout).until(
                lambda driver: driver.execute_script("return document.readyState") == "complete"
            )
            
            # Additional wait for dynamic content
            time.sleep(random.uniform(2, 5))
            
            return self.driver.page_source
            
        except TimeoutException:
            logger.error(f"Timeout while loading {url}")
            return None
        except WebDriverException as e:
            logger.error(f"Selenium error for {url}: {e}")
            return None
    
    def _extract_content_by_type(self, soup: BeautifulSoup, website_type: WebsiteType) -> Dict[str, Any]:
        """Extract content based on website type."""
        content = {
            'url': '',
            'title': '',
            'content': '',
            'metadata': {},
            'website_type': website_type.value
        }
        
        selectors = self._get_selectors_for_type(website_type)
        
        # Extract title
        for selector in selectors.get('title', []):
            element = soup.select_one(selector)
            if element:
                content['title'] = element.get_text(strip=True)
                break
        
        # Extract main content
        for selector in selectors.get('content', []):
            element = soup.select_one(selector)
            if element:
                content['content'] = element.get_text(separator='\n', strip=True)
                break
        
        # Extract additional metadata
        for key, selector_list in selectors.items():
            if key not in ['title', 'content']:
                for selector in selector_list:
                    element = soup.select_one(selector)
                    if element:
                        content['metadata'][key] = element.get_text(strip=True)
                        break
        
        return content
    
    def _get_selectors_for_type(self, website_type: WebsiteType) -> Dict[str, List[str]]:
        """Get CSS selectors for specific website type."""
        if website_type == WebsiteType.BUSINESS:
            return self.website_config.business_selectors
        elif website_type == WebsiteType.NEWS:
            return self.website_config.news_selectors
        elif website_type == WebsiteType.WIKIPEDIA:
            return self.website_config.wikipedia_selectors
        else:
            return {}
    
    def _clean_html(self, html_content: str) -> str:
        """Clean HTML content and extract text."""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
            script.decompose()
        
        # Remove comments
        for comment in soup.find_all(string=lambda text: isinstance(text, str) and text.strip().startswith('<!--')):
            comment.extract()
        
        return soup.get_text(separator='\n', strip=True)
    
    def scrape_url(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Scrape a single URL and return structured content.
        
        Args:
            url: The URL to scrape
            
        Returns:
            Dictionary containing scraped content and metadata
        """
        try:
            logger.info(f"Starting to scrape: {url}")
            
            # Get page content
            html_content = self._get_page_with_retry(url)
            if not html_content:
                return None
            
            # Parse HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Detect website type
            website_type = detect_website_type(url, soup)
            logger.info(f"Detected website type: {website_type.value}")
            
            # Extract content based on type
            content = self._extract_content_by_type(soup, website_type)
            content['url'] = url
            
            # If no content was extracted, fall back to general extraction
            if not content['content']:
                content['content'] = self._clean_html(html_content)
            
            # Add metadata
            content['metadata']['html_length'] = len(html_content)
            content['metadata']['text_length'] = len(content['content'])
            content['metadata']['scraped_at'] = time.time()
            
            logger.info(f"Successfully scraped {url} - Content length: {len(content['content'])}")
            return content
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return None
    
    def scrape_urls(self, urls: List[str]) -> List[Dict[str, Any]]:
        """
        Scrape multiple URLs.
        
        Args:
            urls: List of URLs to scrape
            
        Returns:
            List of scraped content dictionaries
        """
        results = []
        
        for url in urls:
            try:
                result = self.scrape_url(url)
                if result:
                    results.append(result)
                
                # Random delay between requests
                time.sleep(random.uniform(1, 3))
                
            except Exception as e:
                logger.error(f"Error processing {url}: {e}")
                continue
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], output_dir: str = "./output"):
        """Save scraping results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for i, result in enumerate(results):
            # Create filename from URL
            url_hash = hashlib.md5(result['url'].encode()).hexdigest()[:8]
            filename = f"{url_hash}_{i}.json"
            filepath = output_path / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved result to {filepath}")
    
    def close(self):
        """Close the web driver and clean up resources."""
        if self.driver:
            self.driver.quit()
            logger.info("Web driver closed")