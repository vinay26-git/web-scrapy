# Advanced Web Scraper Pipeline

A comprehensive web scraping pipeline designed for extracting, processing, and generating embeddings from web content. Built with enterprise-grade features including anti-detection mechanisms, advanced text chunking, and optimized embedding generation.

## 🚀 Features

### Web Scraping
- **Selenium-based scraping** with undetected-chromedriver for anti-detection
- **Smart fallback** between requests and Selenium based on JavaScript requirements
- **Website type detection** (Business, News, Wikipedia) with specialized selectors
- **Retry mechanisms** and error handling
- **Respectful crawling** with configurable delays

### Text Processing
- **Hybrid chunking approach** combining recursive chunking with content-aware techniques
- **Semantic boundary preservation** (paragraphs, sentences)
- **Intelligent chunk optimization** with size constraints
- **Similarity-based chunk merging** using TF-IDF and cosine similarity

### Embedding Generation
- **Multiple model support** (SentenceTransformers, HuggingFace)
- **GPU acceleration** with automatic device detection
- **Embedding caching** for performance optimization
- **Batch processing** for efficient generation
- **Similarity computation** and search capabilities

### Pipeline Features
- **Modular architecture** with configurable components
- **Comprehensive logging** with rotation and retention
- **Result persistence** in JSON format
- **Statistics and analytics** for processing insights
- **Command-line interface** for easy usage

## 📋 Requirements

- Python 3.8+
- Chrome browser (for Selenium)
- CUDA-compatible GPU (optional, for acceleration)

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd web-scraper
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download NLTK data:**
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

4. **Download spaCy model (optional):**
```bash
python -m spacy download en_core_web_sm
```

## 🚀 Quick Start

### Basic Usage

```python
from src.pipeline import WebScrapingPipeline

# Create pipeline with default configuration
pipeline = WebScrapingPipeline()

# Process a single URL
result = pipeline.process_url("https://en.wikipedia.org/wiki/Artificial_intelligence")

# Process multiple URLs
urls = [
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://www.bbc.com/news/technology",
    "https://www.microsoft.com/en-us/about"
]
results = pipeline.process_urls(urls)

# Save results
pipeline.save_results(results)
```

### Command Line Usage

```bash
# Run with example URLs
python main.py --example

# Process specific URLs
python main.py --urls "https://example.com" "https://another-example.com"

# Process URLs from file
python main.py --url-file urls.txt

# Use custom configuration
python main.py --config example_config.json --urls "https://example.com"

# Customize parameters
python main.py --chunk-size 500 --model "sentence-transformers/all-mpnet-base-v2" --urls "https://example.com"
```

## 📖 Detailed Usage

### Configuration

The pipeline is highly configurable through the `PipelineConfig` class:

```python
from src.config import PipelineConfig, ScrapingConfig, ChunkingConfig, EmbeddingConfig

config = PipelineConfig(
    scraping=ScrapingConfig(
        headless=True,
        page_load_timeout=30,
        max_retries=3
    ),
    chunking=ChunkingConfig(
        chunk_size=1000,
        chunk_overlap=200,
        preserve_paragraphs=True
    ),
    embedding=EmbeddingConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=32,
        device="auto"
    ),
    output_dir="./output"
)

pipeline = WebScrapingPipeline(config)
```

### Custom Website Selectors

You can customize selectors for different website types:

```python
from src.config import WebsiteConfig

website_config = WebsiteConfig(
    business_selectors={
        "title": ["h1", ".title", ".page-title"],
        "content": ["main", ".content", ".main-content"],
        "description": [".description", ".summary"]
    },
    news_selectors={
        "title": ["h1", ".headline", ".article-title"],
        "content": ["article", ".article-content"],
        "author": [".author", ".byline"]
    }
)
```

### Advanced Chunking

The hybrid chunker provides advanced features:

```python
from src.chunker import HybridChunker

chunker = HybridChunker(config.chunking)

# Chunk text with metadata
chunks = chunker.chunk_text(
    text="Your text here...",
    metadata={"source": "example.com", "category": "technology"}
)

# Merge similar chunks
merged_chunks = chunker.merge_similar_chunks(chunks, similarity_threshold=0.8)

# Get chunk statistics
stats = chunker.get_chunk_statistics(chunks)
```

### Embedding Generation

```python
from src.embeddings import EmbeddingGenerator

embedding_gen = EmbeddingGenerator(config.embedding)

# Generate embeddings for chunks
chunks_with_embeddings = embedding_gen.generate_chunk_embeddings(chunks)

# Find similar chunks
query_embedding = embedding_gen.generate_embeddings(["your query text"])[0]
similar_chunks = embedding_gen.find_similar_chunks(
    query_embedding, 
    [chunk.embedding for chunk in chunks_with_embeddings],
    top_k=5
)
```

## 📊 Output Format

The pipeline generates structured JSON output:

```json
{
  "url": "https://example.com",
  "scraped_content": {
    "title": "Page Title",
    "content": "Main content...",
    "website_type": "business",
    "metadata": {
      "html_length": 15000,
      "text_length": 5000,
      "scraped_at": 1640995200.0
    }
  },
  "chunks": [
    {
      "text": "Chunk text...",
      "chunk_id": "chunk_0_1234",
      "start_index": 0,
      "end_index": 1000,
      "metadata": {
        "url": "https://example.com",
        "title": "Page Title",
        "website_type": "business"
      },
      "embedding": [0.1, 0.2, 0.3, ...]
    }
  ],
  "statistics": {
    "text_statistics": {
      "char_count": 5000,
      "word_count": 800,
      "sentence_count": 50,
      "paragraph_count": 20
    },
    "chunk_statistics": {
      "total_chunks": 5,
      "avg_chunk_size": 1000,
      "min_chunk_size": 800,
      "max_chunk_size": 1200
    },
    "embedding_statistics": {
      "count": 5,
      "dimension": 384,
      "mean_norm": 1.0
    }
  }
}
```

## 🔧 Configuration Options

### Scraping Configuration
- `headless`: Run browser in headless mode
- `page_load_timeout`: Maximum time to wait for page load
- `max_retries`: Number of retry attempts
- `user_agents`: List of user agents for rotation

### Chunking Configuration
- `chunk_size`: Target size for chunks
- `chunk_overlap`: Overlap between consecutive chunks
- `min_chunk_size`: Minimum acceptable chunk size
- `max_chunk_size`: Maximum acceptable chunk size
- `preserve_paragraphs`: Maintain paragraph boundaries
- `preserve_sentences`: Maintain sentence boundaries

### Embedding Configuration
- `model_name`: Embedding model to use
- `batch_size`: Batch size for processing
- `max_length`: Maximum sequence length
- `device`: Device for computation (auto/cpu/cuda)
- `normalize_embeddings`: Normalize embeddings to unit vectors

## 🎯 Supported Website Types

### Business Websites
- Company information pages
- Service/product pages
- About us pages
- Contact pages

### News Websites
- Article pages
- News stories
- Blog posts
- Editorial content

### Wikipedia Pages
- Article content
- Infoboxes
- References
- Table of contents

## 🚀 Performance Optimization

### GPU Acceleration
The pipeline automatically detects and uses CUDA-compatible GPUs:

```python
# Force CPU usage
config.embedding.device = "cpu"

# Force CUDA usage
config.embedding.device = "cuda"

# Auto-detect (recommended)
config.embedding.device = "auto"
```

### Caching
Embeddings are automatically cached to avoid recomputation:

```python
# Clear cache if needed
embedding_gen.clear_cache()

# Disable caching
config.embedding.save_embeddings = False
```

### Batch Processing
Optimize batch sizes based on your hardware:

```python
# For GPU with more memory
config.embedding.batch_size = 64

# For CPU or limited GPU memory
config.embedding.batch_size = 16
```

## 🔍 Troubleshooting

### Common Issues

1. **Chrome driver issues:**
   - Ensure Chrome browser is installed
   - Update Chrome to latest version
   - Try using undetected-chromedriver

2. **Memory issues:**
   - Reduce batch size
   - Use smaller chunk sizes
   - Process URLs in smaller batches

3. **CUDA out of memory:**
   - Reduce batch size
   - Use CPU instead of GPU
   - Process smaller chunks

### Debug Mode

Enable debug logging for detailed information:

```python
config.log_level = "DEBUG"
```

## 📈 Monitoring and Analytics

The pipeline provides comprehensive statistics:

```python
# Get pipeline information
info = pipeline.get_pipeline_info()

# Get processing statistics
for result in results:
    stats = result['statistics']
    print(f"Text length: {stats['text_statistics']['char_count']}")
    print(f"Chunks: {stats['chunk_statistics']['total_chunks']}")
    print(f"Embedding dimension: {stats['embedding_statistics']['dimension']}")
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Selenium](https://selenium-python.readthedocs.io/) for web automation
- [SentenceTransformers](https://www.sbert.net/) for embedding models
- [LangChain](https://langchain.com/) for text splitting
- [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) for HTML parsing
- [NLTK](https://www.nltk.org/) for text processing
- [spaCy](https://spacy.io/) for NLP capabilities

## 📞 Support

For questions, issues, or contributions, please open an issue on GitHub or contact the maintainers.