import re
import time
from typing import List, Tuple, Dict

# Selenium & browser automation
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

# HTML parsing
from bs4 import BeautifulSoup, NavigableString, Tag

# Embeddings
from sentence_transformers import SentenceTransformer
import torch

# -----------------------------------------------------------------------------
# Selenium Scraper
# -----------------------------------------------------------------------------

class SeleniumScraper:
    """Lightweight wrapper around a headless Chrome WebDriver."""

    def __init__(self, driver: webdriver.Chrome = None, timeout: int = 20):
        self.timeout = timeout
        if driver is None:
            self.driver = self._start_driver()
        else:
            self.driver = driver

    @staticmethod
    def _start_driver() -> webdriver.Chrome:
        """Spin up a headless Chrome driver with Kaggle-friendly flags."""
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        driver = webdriver.Chrome(options=chrome_options)
        return driver

    def fetch(self, url: str) -> str:
        """Load `url` and return page source after DOM is ready."""
        self.driver.get(url)
        # naive wait – replace with WebDriverWait if finer control is needed
        time.sleep(3)
        html = self.driver.page_source
        return html

    def close(self):
        self.driver.quit()


# -----------------------------------------------------------------------------
# Cleaning & Pre-Processing
# -----------------------------------------------------------------------------

class HTMLCleaner:
    """Cleans raw HTML and extracts readable text + heading structure."""

    def __init__(self):
        # Pre-compiled regexes for speed
        self._multi_ws_re = re.compile(r"\s+")

    def _strip_unwanted_tags(self, soup: BeautifulSoup):
        for tag in soup(["script", "style", "noscript", "header", "footer", "svg", "img", "audio", "video"]):
            tag.decompose()

    def _get_text(self, soup: BeautifulSoup) -> str:
        """Return body text with normalized whitespace."""
        text = soup.get_text(separator=" ")
        text = self._multi_ws_re.sub(" ", text)
        return text.strip()

    def clean(self, html: str) -> Tuple[str, BeautifulSoup]:
        """Return clean text and the parsed BeautifulSoup document."""
        soup = BeautifulSoup(html, "html.parser")
        self._strip_unwanted_tags(soup)
        text = self._get_text(soup)
        return text, soup


# -----------------------------------------------------------------------------
# Hybrid Chunking (Content-Aware + Recursive)
# -----------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Very naive whitespace tokeniser (replace with tiktoken for prod)."""
    return text.split()


def _detokenize(tokens: List[str]) -> str:
    return " ".join(tokens)


def recursive_split(tokens: List[str], max_tokens: int, overlap: int) -> List[str]:
    """Recursively split tokens into <= `max_tokens` chunks with `overlap`."""
    if len(tokens) <= max_tokens:
        return [_detokenize(tokens)]

    half = len(tokens) // 2
    left = tokens[: half + overlap]
    right = tokens[half - overlap :]
    return recursive_split(left, max_tokens, overlap) + recursive_split(right, max_tokens, overlap)


def content_aware_chunk(html_soup: BeautifulSoup, max_tokens: int = 512, overlap: int = 50) -> List[str]:
    """Chunk document based on heading hierarchy first, then recursively."""
    chunks: List[str] = []
    current_segment_tokens: List[str] = []

    def flush_segment():
        if current_segment_tokens:
            chunks.extend(recursive_split(current_segment_tokens, max_tokens, overlap))
            current_segment_tokens.clear()

    for element in html_soup.body.descendants:
        if isinstance(element, Tag) and element.name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            # Start a new segment at each heading
            flush_segment()
            heading_text = element.get_text(separator=" ").strip()
            current_segment_tokens.extend(_tokenize(heading_text))
        elif isinstance(element, NavigableString):
            tokens = _tokenize(element.strip())
            current_segment_tokens.extend(tokens)
            if len(current_segment_tokens) >= max_tokens:
                # segment overflow – flush early
                flush_segment()

    flush_segment()  # final segment
    return chunks


# -----------------------------------------------------------------------------
# Embedding Generation
# -----------------------------------------------------------------------------

class EmbeddingGenerator:
    """Generates vector embeddings for text chunks using Sentence-Transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)

    def embed(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True).tolist()


# -----------------------------------------------------------------------------
# End-to-End Pipeline
# -----------------------------------------------------------------------------

class WebScraperPipeline:
    """Combines scraping, cleaning, chunking, and embedding into a single flow."""

    def __init__(self, max_tokens: int = 512, overlap: int = 50):
        self.scraper = SeleniumScraper()
        self.cleaner = HTMLCleaner()
        self.embedder = EmbeddingGenerator()
        self.max_tokens = max_tokens
        self.overlap = overlap

    def process_url(self, url: str) -> Dict[str, List]:
        """Return dict with clean text, chunks, and embeddings for a single URL."""
        html = self.scraper.fetch(url)
        clean_text, soup = self.cleaner.clean(html)
        chunks = content_aware_chunk(soup, self.max_tokens, self.overlap)
        embeddings = self.embedder.embed(chunks)
        return {
            "url": url,
            "clean_text": clean_text,
            "chunks": chunks,
            "embeddings": embeddings,
        }

    def close(self):
        self.scraper.close()


# -----------------------------------------------------------------------------
# Example Execution (uncomment for standalone run)
# -----------------------------------------------------------------------------
# if __name__ == "__main__":
#     urls = [
#         "https://en.wikipedia.org/wiki/Web_scraping",  # Wikipedia page
#         "https://www.bbc.com/news/business-66231792",   # News article
#         "https://www.microsoft.com/en-us/",            # Business site
#     ]
#     pipeline = WebScraperPipeline()
#     try:
#         for url in urls:
#             result = pipeline.process_url(url)
#             print(f"URL: {result['url']}")
#             print(f"Total chunks: {len(result['chunks'])}")
#             print(f"Embedding[0][:5]: {result['embeddings'][0][:5]}")
#             print("---\n")
#     finally:
#         pipeline.close()