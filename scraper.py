"""
Advanced Web Scraper using Selenium with site-specific strategies.
Optimized for business websites, news sites, and Wikipedia pages.
"""

import time
import re
from typing import Optional, Dict, List, Tuple
from urllib.parse import urljoin, urlparse

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager

from bs4 import BeautifulSoup
from loguru import logger

from config import config


class WebScraper:
    """Advanced web scraper with site-specific optimization."""
    
    def __init__(self):
        self.driver: Optional[webdriver.Chrome] = None
        self.wait: Optional[WebDriverWait] = None
        
    def _setup_driver(self, site_type: str) -> webdriver.Chrome:
        """Set up Chrome driver with optimized options for the site type."""
        chrome_options = Options()
        
        # Basic options
        if config.scraping.headless:
            chrome_options.add_argument("--headless")
        
        chrome_options.add_argument(f"--window-size={config.scraping.window_size[0]},{config.scraping.window_size[1]}")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        
        # Site-specific user agent
        user_agent = config.scraping.user_agents.get(site_type, config.scraping.user_agents["business"])
        chrome_options.add_argument(f"--user-agent={user_agent}")
        
        # Performance optimizations
        chrome_options.add_argument("--disable-images")
        chrome_options.add_argument("--disable-javascript")  # We'll enable selectively
        chrome_options.add_argument("--disable-plugins")
        chrome_options.add_argument("--disable-extensions")
        
        # Privacy settings
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        # Initialize driver
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # Additional stealth settings
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        driver.execute_cdp_cmd('Network.setUserAgentOverride', {"userAgent": user_agent})
        
        # Set timeouts
        driver.implicitly_wait(config.scraping.implicit_wait)
        driver.set_page_load_timeout(config.scraping.page_load_timeout)
        
        return driver
    
    def _handle_consent_banners(self, site_type: str) -> None:
        """Handle cookie consent banners and popups."""
        consent_selectors = [
            # Common consent button texts and selectors
            "button[contains(text(), 'Accept')]",
            "button[contains(text(), 'Accept All')]",
            "button[contains(text(), 'I Accept')]",
            "button[contains(text(), 'OK')]",
            "button[contains(text(), 'Continue')]",
            ".cookie-accept",
            ".consent-accept",
            "#accept-cookies",
            ".gdpr-accept",
            "[data-testid*='accept']",
            "[data-cy*='accept']"
        ]
        
        for selector in consent_selectors:
            try:
                if "contains(text()" in selector:
                    # XPath selector
                    elements = self.driver.find_elements(By.XPATH, f"//{selector}")
                else:
                    # CSS selector
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                
                if elements:
                    elements[0].click()
                    time.sleep(1)
                    logger.info(f"Clicked consent button: {selector}")
                    break
            except Exception:
                continue
    
    def _wait_for_content_load(self, site_type: str) -> None:
        """Wait for main content to load based on site type."""
        main_selectors = config.scraping.content_selectors[site_type]["main_content"]
        
        for selector in main_selectors:
            try:
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                )
                logger.info(f"Content loaded for selector: {selector}")
                return
            except TimeoutException:
                continue
        
        logger.warning(f"No main content selector found for {site_type}")
    
    def _extract_metadata(self, soup: BeautifulSoup, url: str) -> Dict[str, str]:
        """Extract metadata from the page."""
        metadata = {
            "url": url,
            "title": "",
            "description": "",
            "author": "",
            "published_date": "",
            "language": "en"
        }
        
        # Title
        title_tag = soup.find("title")
        if title_tag:
            metadata["title"] = title_tag.get_text().strip()
        
        # Meta tags
        meta_tags = soup.find_all("meta")
        for tag in meta_tags:
            name = tag.get("name", "").lower()
            property_attr = tag.get("property", "").lower()
            content = tag.get("content", "")
            
            if name in ["description", "og:description"] or property_attr == "og:description":
                metadata["description"] = content
            elif name in ["author", "article:author"] or property_attr == "article:author":
                metadata["author"] = content
            elif name in ["date", "published_time", "article:published_time"] or property_attr == "article:published_time":
                metadata["published_date"] = content
            elif name == "language" or property_attr == "og:locale":
                metadata["language"] = content.split("-")[0] if content else "en"
        
        # JSON-LD structured data
        json_ld_scripts = soup.find_all("script", {"type": "application/ld+json"})
        for script in json_ld_scripts:
            try:
                import json
                data = json.loads(script.string)
                if isinstance(data, dict):
                    if "author" in data and not metadata["author"]:
                        author = data["author"]
                        if isinstance(author, dict):
                            metadata["author"] = author.get("name", "")
                        elif isinstance(author, str):
                            metadata["author"] = author
                    
                    if "datePublished" in data and not metadata["published_date"]:
                        metadata["published_date"] = data["datePublished"]
            except:
                continue
        
        return metadata
    
    def _clean_html_content(self, soup: BeautifulSoup, site_type: str) -> BeautifulSoup:
        """Clean HTML content by removing unwanted elements."""
        remove_selectors = config.scraping.content_selectors[site_type]["remove"]
        
        # Remove unwanted elements
        for selector in remove_selectors:
            elements = soup.select(selector)
            for element in elements:
                element.decompose()
        
        # Remove script and style tags
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        
        # Remove comments
        for comment in soup.find_all(string=lambda text: isinstance(text, str) and text.strip().startswith("<!--")):
            comment.extract()
        
        return soup
    
    def _extract_main_content(self, soup: BeautifulSoup, site_type: str) -> str:
        """Extract main content based on site type."""
        main_selectors = config.scraping.content_selectors[site_type]["main_content"]
        
        content_parts = []
        
        for selector in main_selectors:
            elements = soup.select(selector)
            if elements:
                for element in elements:
                    text = element.get_text(separator="\n", strip=True)
                    if text and len(text) > 100:  # Only include substantial content
                        content_parts.append(text)
                break
        
        if not content_parts:
            # Fallback: extract all paragraph text
            paragraphs = soup.find_all("p")
            content_parts = [p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 50]
        
        return "\n\n".join(content_parts)
    
    def scrape_url(self, url: str) -> Optional[Dict[str, any]]:
        """
        Scrape a single URL and return structured data.
        
        Args:
            url: URL to scrape
            
        Returns:
            Dictionary containing scraped data or None if failed
        """
        site_type = config.get_site_type(url)
        logger.info(f"Scraping {url} as {site_type} site")
        
        try:
            # Setup driver for this site type
            if not self.driver:
                self.driver = self._setup_driver(site_type)
                self.wait = WebDriverWait(self.driver, 10)
            
            # Navigate to URL
            self.driver.get(url)
            
            # Handle consent banners
            time.sleep(2)  # Allow page to load
            self._handle_consent_banners(site_type)
            
            # Wait for content to load
            self._wait_for_content_load(site_type)
            
            # Additional wait for dynamic content
            time.sleep(3)
            
            # Get page source and create soup
            page_source = self.driver.page_source
            soup = BeautifulSoup(page_source, "lxml")
            
            # Extract metadata
            metadata = self._extract_metadata(soup, url)
            
            # Clean HTML content
            cleaned_soup = self._clean_html_content(soup, site_type)
            
            # Extract main content
            main_content = self._extract_main_content(cleaned_soup, site_type)
            
            if not main_content or len(main_content) < 200:
                logger.warning(f"Insufficient content extracted from {url}")
                return None
            
            result = {
                "url": url,
                "site_type": site_type,
                "metadata": metadata,
                "content": main_content,
                "raw_html": str(cleaned_soup),
                "content_length": len(main_content),
                "scraped_at": time.time()
            }
            
            logger.success(f"Successfully scraped {url} - {len(main_content)} characters")
            return result
            
        except TimeoutException:
            logger.error(f"Timeout while scraping {url}")
        except WebDriverException as e:
            logger.error(f"WebDriver error while scraping {url}: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error while scraping {url}: {str(e)}")
        
        return None
    
    def scrape_urls(self, urls: List[str]) -> List[Dict[str, any]]:
        """
        Scrape multiple URLs.
        
        Args:
            urls: List of URLs to scrape
            
        Returns:
            List of scraped data dictionaries
        """
        results = []
        
        for i, url in enumerate(urls):
            logger.info(f"Scraping URL {i+1}/{len(urls)}: {url}")
            
            result = self.scrape_url(url)
            if result:
                results.append(result)
            
            # Add delay between requests to be respectful
            if i < len(urls) - 1:
                time.sleep(2)
        
        logger.info(f"Completed scraping {len(results)}/{len(urls)} URLs successfully")
        return results
    
    def close(self):
        """Close the browser driver."""
        if self.driver:
            self.driver.quit()
            self.driver = None
            self.wait = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Example usage and testing
if __name__ == "__main__":
    # Test URLs for different site types
    test_urls = [
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://www.bbc.com/news",
        "https://www.microsoft.com/en-us/ai"
    ]
    
    with WebScraper() as scraper:
        results = scraper.scrape_urls(test_urls)
        
        for result in results:
            print(f"\nURL: {result['url']}")
            print(f"Site Type: {result['site_type']}")
            print(f"Title: {result['metadata']['title']}")
            print(f"Content Length: {result['content_length']} characters")
            print(f"Content Preview: {result['content'][:200]}...")