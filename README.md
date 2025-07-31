# Web Scraper: Enterprise-Grade Text Processing Pipeline

A comprehensive web scraping and text processing pipeline designed for Kaggle's GPU environment (P100). This project implements a sophisticated system that scrapes web content, processes it using advanced techniques, and generates high-quality embeddings for downstream ML applications.

## 🚀 Features

### Core Capabilities
- **Multi-Website Support**: Specialized scrapers for Business, News, and Wikipedia websites
- **Hybrid Chunking**: Combines recursive chunking with content-aware semantic techniques
- **Advanced Preprocessing**: Unicode normalization, noise removal, and quality assessment
- **Enterprise Embeddings**: GPU-accelerated embedding generation with caching
- **Quality Control**: Comprehensive quality scoring and filtering at each stage

### Technical Highlights
- **Selenium-Based Scraping**: Robust web scraping with retry logic and rate limiting
- **Content-Aware Chunking**: Semantic similarity-based text segmentation
- **Embedding Optimization**: Batch processing with GPU acceleration
- **Comprehensive Logging**: Detailed tracking and error reporting
- **Modular Architecture**: Easy to extend and customize

## 📁 Project Structure

```
web-scraper/
├── src/
│   ├── __init__.py
│   ├── config.py          # Configuration management
│   ├── scraper.py         # Selenium-based web scraper
│   ├── preprocessor.py    # Text cleaning and preprocessing
│   ├── chunker.py         # Hybrid chunking system
│   ├── embeddings.py      # Embedding generation pipeline
│   └── pipeline.py        # Main orchestrator
├── main.py                # Entry point with examples
├── example_usage.py       # Comprehensive usage examples
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## 🛠 Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd web-scraper
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download NLTK Data (if needed)
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### 4. Setup Chrome for Selenium
The scraper will automatically download ChromeDriver, but ensure Chrome is installed on your system.

## 🚦 Quick Start

### Simple Usage
```python
from src.pipeline import run_web_scraping_pipeline

# Define URLs to scrape
urls = [
    "https://en.wikipedia.org/wiki/Machine_learning",
    "https://techcrunch.com/category/artificial-intelligence/",
    "https://blog.hubspot.com/marketing/what-is-content-marketing"
]

# Run the complete pipeline
result = run_web_scraping_pipeline(urls, output_dir="output/my_run")

print(f"Generated {len(result.embeddings)} embeddings")
print(f"Processing time: {result.execution_time:.2f} seconds")
```

### Custom Configuration
```python
from src.config import Config
from src.pipeline import WebScrapingPipeline

# Create custom configuration
config = Config()
config.chunking.chunk_size = 800
config.chunking.chunk_overlap = 150
config.embedding.model_name = "sentence-transformers/all-MiniLM-L6-v2"
config.embedding.batch_size = 32

# Initialize pipeline with custom config
pipeline = WebScrapingPipeline(config)
result = pipeline.run_pipeline(urls, min_quality_score=60.0)
```

## 🔧 Configuration Options

### Scraping Configuration
```python
config.scraping.headless = True              # Run browser in headless mode
config.scraping.page_load_timeout = 30       # Page load timeout (seconds)
config.scraping.request_delay = 1.0          # Delay between requests
config.scraping.max_retries = 3              # Maximum retry attempts
config.scraping.min_content_length = 100     # Minimum content length
```

### Chunking Configuration
```python
config.chunking.chunk_size = 1000            # Target chunk size (tokens)
config.chunking.chunk_overlap = 200          # Overlap between chunks
config.chunking.use_semantic_splitting = True # Enable semantic chunking
config.chunking.sentence_similarity_threshold = 0.7 # Similarity threshold
```

### Embedding Configuration
```python
config.embedding.model_name = "sentence-transformers/all-MiniLM-L6-v2"
config.embedding.batch_size = 32             # Batch size for embedding generation
config.embedding.max_seq_length = 512        # Maximum sequence length
config.embedding.normalize_embeddings = True # Normalize embeddings
config.embedding.device = "cuda"             # Device for inference
```

## 📊 Pipeline Stages

### 1. Web Scraping
- **Selenium-based scraping** with website-specific optimizations
- **Intelligent content extraction** using CSS selectors
- **Robust error handling** with retry logic and rate limiting
- **Content validation** and quality checks

### 2. Text Preprocessing
- **Unicode normalization** and encoding issue resolution
- **Noise removal** (ads, navigation, social media elements)
- **Language detection** and content filtering
- **Quality assessment** with comprehensive scoring

### 3. Hybrid Chunking
- **Recursive chunking** with hierarchical text separators
- **Content-aware chunking** using semantic similarity
- **Intelligent chunk selection** based on quality metrics
- **Token-aware splitting** for LLM compatibility

### 4. Embedding Generation
- **GPU-accelerated processing** with batch optimization
- **Multiple model support** (SentenceTransformers, Transformers)
- **Embedding validation** and quality control
- **Automatic caching** for efficiency

## 🎯 Supported Website Types

### Business Websites
- Optimized selectors for business content
- Focus on main articles and product descriptions
- Removal of commercial elements and ads

### News Websites
- Specialized extraction for news articles
- Handling of dynamic content and multimedia
- Filtering of related articles and navigation

### Wikipedia Pages
- Precise content extraction from Wikipedia structure
- Removal of infoboxes, references, and navigation
- Preservation of article hierarchy

## 📈 Performance Optimization

### GPU Acceleration
- Automatic GPU detection and utilization
- Batch processing for maximum throughput
- Memory-efficient embedding generation

### Caching System
- File-based embedding cache
- Automatic cache key generation
- Reduces redundant processing

### Quality Control
- Multi-stage quality assessment
- Configurable quality thresholds
- Automatic filtering of low-quality content

## 📝 Usage Examples

### Example 1: Basic Pipeline
```python
# Run the demo
python main.py
```

### Example 2: Step-by-Step Processing
```python
from src.pipeline import WebScrapingPipeline
from src.config import Config

config = Config()
pipeline = WebScrapingPipeline(config)

# Step 1: Scrape
scraped = pipeline.scrape_urls(["https://example.com"])

# Step 2: Process
processed = pipeline.process_contents(scraped)

# Step 3: Chunk
chunks = pipeline.chunk_contents(processed)

# Step 4: Embed
embeddings = pipeline.generate_embeddings(chunks)
```

### Example 3: Comprehensive Examples
```python
# Run all usage examples
python example_usage.py
```

## 🔍 Output Structure

The pipeline generates several output files:

```
output/
├── embeddings.pkl          # Main embedding output
├── embeddings.json         # Embedding metadata
├── pipeline_summary.json   # Execution summary
├── detailed_statistics.json # Comprehensive stats
└── scraped_metadata.json   # Scraped content metadata
```

### Embedding Data Structure
```python
@dataclass
class ChunkEmbedding:
    chunk_id: str              # Unique chunk identifier
    embedding: np.ndarray      # Vector embedding
    model_name: str           # Embedding model used
    embedding_dim: int        # Embedding dimension
    source_url: str           # Original URL
    chunk_text: str           # Text content (truncated)
    token_count: int          # Number of tokens
    metadata: Dict            # Additional metadata
```

## 🐛 Debugging and Logging

### Log Files
- `logs/scraper.log` - Scraping activities and errors
- `logs/pipeline.log` - Pipeline execution details

### Debug Mode
```python
from loguru import logger
logger.add(sys.stdout, level="DEBUG")
```

## ⚡ Performance Tips

### For Kaggle P100 Environment
1. **Use GPU acceleration**: Ensure `config.embedding.device = "cuda"`
2. **Optimize batch sizes**: Set `config.embedding.batch_size = 32` or higher
3. **Enable caching**: Let the system cache embeddings for repeated runs
4. **Use efficient models**: Consider smaller models for faster processing

### Memory Management
```python
# For large datasets
config.embedding.batch_size = 16  # Reduce if memory issues
config.chunking.chunk_size = 500  # Smaller chunks for memory efficiency
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **LangChain** for recursive chunking inspiration
- **Sentence Transformers** for embedding models
- **Selenium** for robust web scraping
- **HuggingFace** for transformer models

## 📞 Support

For questions or issues:
1. Check the example scripts in `example_usage.py`
2. Review the configuration options in `src/config.py`
3. Check the logs in the `logs/` directory
4. Open an issue on GitHub

---

**Built for enterprise-scale text processing with focus on quality, performance, and reliability.**