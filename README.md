# Web Scraper Pipeline

This project scrapes web pages (business sites, news sites, and Wikipedia articles), cleans & preprocesses the text, chunks it using a hybrid strategy (content-aware + recursive), and produces sentence-level embeddings. The pipeline stops **before** any LLM inference.

---

## Installation

Create a fresh Python environment (Python 3.8+) and install dependencies:

```bash
pip install -r requirements.txt
```

On Kaggle, most packages are pre-installed; you may only need:

```bash
pip install selenium webdriver-manager sentence-transformers
```

> **GPU:** If a GPU is available (e.g. Kaggle P100), the script automatically uses it for embedding generation via PyTorch.

---

## Usage

```bash
python web_scraper_pipeline.py https://en.wikipedia.org/wiki/Natural_language_processing \
                               https://www.bbc.com/news/world-us-canada-66578299
```

Options:

* `--model` Sentence-Transformers model name (default: `all-MiniLM-L6-v2`).
* `--out` Path for the output JSON (default: `results.json`).

The output JSON structure for each URL is:

```json
{
  "<url>": {
    "full_text": "<entire article text>",
    "chunks": ["chunk 1", "chunk 2", ...],
    "embeddings": [[...], [...], ...]  // 2-D list of floats
  }
}
```

---

## Notes

1. The hybrid chunker prioritises HTML headings and paragraph boundaries, then recursively splits oversized chunks on sentence boundaries to meet the target word count (512 words by default).
2. Overlap of 50 words between consecutive chunks preserves context for downstream models.
3. Selenium runs in **headless** Chrome mode using `webdriver-manager` to fetch the correct driver.

---

## License

MIT