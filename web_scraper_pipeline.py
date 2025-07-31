#!/usr/bin/env python
"""
Web Scraper Pipeline
--------------------
Scrape HTML from business, news, and Wikipedia pages using Selenium,
clean & preprocess the text, apply a hybrid (recursive + content-aware)
chunking strategy, and finally generate sentence-level embeddings.

Author: ChatGPT-Assistant
"""

import os
import re
import time
from typing import List, Dict, Tuple

from bs4 import BeautifulSoup
from nltk import download as nltk_download
from nltk.tokenize import sent_tokenize
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# NLTK setup (download tokenizer data the first time this script is executed)
# ---------------------------------------------------------------------------
try:
    nltk_download("punkt", quiet=True)
except Exception as _:
    # Fallback: in constrained environments, ignore download errors
    pass

# ---------------------------------------------------------------------------
# Selenium utilities
# ---------------------------------------------------------------------------

def init_driver(headless: bool = True, timeout: int = 30) -> webdriver.Chrome:
    """Initialise and return a Selenium Chrome WebDriver."""
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(
        service=ChromeService(ChromeDriverManager().install()),
        options=chrome_options,
    )
    driver.set_page_load_timeout(timeout)
    return driver


def fetch_html(url: str, driver: webdriver.Chrome, wait_time: int = 3) -> str:
    """Navigate to *url* using *driver* and return page source."""
    try:
        driver.get(url)
        time.sleep(wait_time)  # give JS some time to render
        html = driver.page_source
    except Exception as e:
        raise RuntimeError(f"Failed to fetch {url}: {e}")
    return html

# ---------------------------------------------------------------------------
# Cleaning & preprocessing
# ---------------------------------------------------------------------------

def clean_html(raw_html: str) -> str:
    """Strip scripts/styles & return raw visible HTML for text extraction."""
    soup = BeautifulSoup(raw_html, "lxml")

    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()

    return str(soup)


def extract_readable_text(html: str) -> str:
    """Extract visible text from cleaned HTML."""
    soup = BeautifulSoup(html, "lxml")

    # Use headings & paragraphs to preserve document structure
    texts: List[str] = []
    for element in soup.find_all([f"h{i}" for i in range(1, 7)] + ["p", "li"]):
        text = element.get_text(strip=True, separator=" ")
        if text:
            texts.append(text)
    return "\n".join(texts)


def preprocess_text(text: str) -> str:
    """Light text normalisation: collapse spaces, fix linebreaks, etc."""
    text = re.sub(r"[ \t]+", " ", text)  # collapse consecutive spaces/tabs
    text = re.sub(r"\n{2,}", "\n", text)  # collapse multiple newlines
    return text.strip()

# ---------------------------------------------------------------------------
# Hybrid chunking (content-aware + recursive)
# ---------------------------------------------------------------------------

MAX_WORDS = 512            # ~ 750 tokens in many LLMs
OVERLAP_WORDS = 50         # sliding window overlap to preserve context


def _words(text: str) -> List[str]:
    return text.split()


def _split_recursive(text: str, max_words: int) -> List[str]:
    """Recursively split text at sentence boundaries until <= *max_words*."""
    if len(_words(text)) <= max_words:
        return [text]

    # sentence split
    sentences = sent_tokenize(text)
    mid = len(sentences) // 2
    left = " ".join(sentences[:mid])
    right = " ".join(sentences[mid:])
    return _split_recursive(left, max_words) + _split_recursive(right, max_words)


def hybrid_chunk(text: str, max_words: int = MAX_WORDS, overlap: int = OVERLAP_WORDS) -> List[str]:
    """Hybrid (content-aware + recursive) chunking strategy."""
    paragraphs = [p for p in text.split("\n") if p.strip()]

    chunks: List[str] = []
    current: List[str] = []
    current_size = 0

    def flush_current():
        nonlocal current, current_size
        if current:
            chunk = " ".join(current).strip()
            # Recursive refinement if chunk too large
            if len(_words(chunk)) > max_words:
                chunks.extend(_split_recursive(chunk, max_words))
            else:
                chunks.append(chunk)
            current = []
            current_size = 0

    for para in paragraphs:
        words_in_para = len(_words(para))
        if current_size + words_in_para > max_words:
            flush_current()
        current.append(para)
        current_size += words_in_para
    flush_current()

    # Add overlap
    if overlap > 0 and len(chunks) > 1:
        overlapped: List[str] = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped.append(chunk)
                continue
            prev_words = _words(chunks[i - 1])
            overlap_text = " ".join(prev_words[-overlap:])
            overlapped.append(overlap_text + " " + chunk)
        chunks = overlapped

    return chunks

# ---------------------------------------------------------------------------
# Embedding generation
# ---------------------------------------------------------------------------

class Embedder:
    """Sentence-Transformers wrapper for chunk embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_gpu: bool = True):
        device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: List[str], batch_size: int = 32, **kwargs) -> List[List[float]]:
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=True, **kwargs)

# ---------------------------------------------------------------------------
# Orchestration helpers
# ---------------------------------------------------------------------------

def scrape_and_prepare(url: str, driver: webdriver.Chrome) -> Tuple[str, List[str]]:
    """Fetch *url*, return cleaned full text and hybrid chunks."""
    raw_html = fetch_html(url, driver)
    cleaned_html = clean_html(raw_html)
    raw_text = extract_readable_text(cleaned_html)
    processed_text = preprocess_text(raw_text)
    chunks = hybrid_chunk(processed_text)
    return processed_text, chunks


def process_urls(urls: List[str], model_name: str = "all-MiniLM-L6-v2") -> Dict[str, Dict[str, List]]:
    """Complete pipeline: scrape, chunk, and embed a list of URLs."""
    results: Dict[str, Dict[str, List]] = {}
    driver = init_driver()
    embedder = Embedder(model_name)
    try:
        for url in tqdm(urls, desc="Processing URLs"):
            full_text, chunks = scrape_and_prepare(url, driver)
            embeddings = embedder.encode(chunks)
            results[url] = {
                "full_text": full_text,
                "chunks": chunks,
                "embeddings": embeddings,
            }
    finally:
        driver.quit()
    return results

# ---------------------------------------------------------------------------
# CLI entry point (example usage)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="Web Scraper Pipeline")
    parser.add_argument("url", nargs="*", help="One or more URLs to process.")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Sentence-Transformers model.")
    parser.add_argument("--out", default="results.json", help="Path to output JSON file.")
    args = parser.parse_args()

    if not args.url:
        parser.error("Please provide at least one URL.")

    data = process_urls(args.url, model_name=args.model)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"Saved results to {args.out}")