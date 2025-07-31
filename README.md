# 🌐 Advanced Web Scraping Pipeline

A comprehensive web scraping pipeline designed for business, news, and Wikipedia websites with advanced chunking and embedding generation capabilities.

## 🚀 Features

- **Multi-Website Support**: Optimized scraping for business, news, and Wikipedia websites
- **Advanced Chunking**: Hybrid approach combining recursive and content-aware chunking
- **Multiple Embedding Models**: Support for various sentence transformer models
- **GPU Acceleration**: Optimized for Kaggle P100 GPU environments
- **Comprehensive Pipeline**: End-to-end processing from scraping to embeddings
- **Domain-Specific Optimizations**: Tailored processing for different content types
- **Robust Error Handling**: Graceful failure handling and recovery
- **Progress Tracking**: Detailed logging and progress monitoring
- **Result Persistence**: Save and load intermediate and final results

## 📋 Requirements

- Python 3.8+
- CUDA-compatible GPU (P100 recommended for optimal performance)
- Chrome browser (for Selenium)

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd web-scrapy
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Install spaCy model (optional but recommended):**
```bash
python -m spacy download en_core_web_sm
```

4. **Install Chrome WebDriver (automatically handled by webdriver-manager)**

## 🏗️ Architecture

The pipeline consists of four main components:

### 1. Web Scraper (`web_scraper.py`)
- **Selenium-based scraping** with optimized Chrome settings
- **Website type detection** using URL patterns
- **Domain-specific content extraction** for business, news, and Wikipedia
- **Robust error handling** and timeout management

### 2. Hybrid Chunker (`chunking.py`)
- **Recursive chunking** based on semantic boundaries
- **Content-aware chunking** using NLP techniques
- **Domain-specific optimizations** for different content types
- **Semantic coherence scoring** for quality assessment

### 3. Embedding Generator (`embeddings.py`)
- **Multiple model support** (all-MiniLM-L6-v2, all-mpnet-base-v2, etc.)
- **Batch processing** for efficiency
- **Domain-specific preprocessing** for better embedding quality
- **Similarity search** capabilities

### 4. Main Pipeline (`pipeline.py`)
- **Orchestrates all components** in a unified workflow
- **Result persistence** and loading
- **Statistics and monitoring**
- **Error recovery** and retry logic

## 🚀 Quick Start

### Basic Usage

```python
from pipeline import WebScrapingPipeline

# Initialize pipeline
pipeline = WebScrapingPipeline(
    output_dir="output",
    chunk_config={
        'max_chunk_size': 800,
        'min_chunk_size': 100,
        'overlap_size': 50,
        'semantic_threshold': 0.6
    },
    embedding_config={
        'model_name': 'all-MiniLM-L6-v2',
        'batch_size': 32,
        'max_length': 512
    }
)

# Process a single URL
url = "https://en.wikipedia.org/wiki/Python_(programming_language)"
result = pipeline.process_single_url(url)

if result:
    print(f"Content length: {len(result.scraped_content.content)}")
    print(f"Chunks created: {len(result.chunks)}")
    print(f"Embeddings generated: {len(result.embeddings)}")
    print(f"Processing time: {result.processing_time:.2f}s")

# Cleanup
pipeline.cleanup()
```

### Batch Processing

```python
urls = [
    "https://www.apple.com",
    "https://www.bbc.com/news",
    "https://en.wikipedia.org/wiki/Artificial_intelligence"
]

# Process multiple URLs
results = pipeline.process_multiple_urls(urls)

# Get statistics
stats = pipeline.get_pipeline_statistics()
print(f"Total results: {stats['total_results']}")
print(f"Total chunks: {stats['total_chunks']}")
print(f"Total embeddings: {stats['total_embeddings']}")
```

### Advanced Configuration

```python
# High-quality configuration
pipeline = WebScrapingPipeline(
    output_dir="high_quality_output",
    chunk_config={
        'max_chunk_size': 1000,
        'min_chunk_size': 150,
        'overlap_size': 100,
        'semantic_threshold': 0.8
    },
    embedding_config={
        'model_name': 'all-mpnet-base-v2',
        'batch_size': 16,
        'max_length': 512
    }
)

# Memory-efficient configuration
pipeline = WebScrapingPipeline(
    output_dir="memory_efficient_output",
    chunk_config={
        'max_chunk_size': 600,
        'min_chunk_size': 80,
        'overlap_size': 30,
        'semantic_threshold': 0.5
    },
    embedding_config={
        'model_name': 'all-MiniLM-L6-v2',
        'batch_size': 8,
        'max_length': 256
    }
)
```

## 📊 Configuration Options

### Chunking Configuration

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `max_chunk_size` | Maximum characters per chunk | 1000 | 200-2000 |
| `min_chunk_size` | Minimum characters per chunk | 100 | 50-500 |
| `overlap_size` | Overlap between chunks | 100 | 0-200 |
| `semantic_threshold` | Minimum semantic coherence score | 0.7 | 0.0-1.0 |

### Embedding Configuration

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `model_name` | Sentence transformer model | 'all-MiniLM-L6-v2' | Various |
| `batch_size` | Batch size for processing | 32 | 1-64 |
| `max_length` | Maximum sequence length | 512 | 128-1024 |

### Recommended Models by Domain

| Domain | Recommended Model | Alternative |
|--------|------------------|-------------|
| Business | all-mpnet-base-v2 | all-MiniLM-L6-v2 |
| News | multi-qa-MiniLM-L6-cos-v1 | all-mpnet-base-v2 |
| Wikipedia | all-MiniLM-L6-v2 | all-mpnet-base-v2 |

## 🔧 Individual Component Usage

### Web Scraper

```python
from web_scraper import WebScraper

with WebScraper(headless=True) as scraper:
    content = scraper.scrape_url("https://example.com")
    if content:
        print(f"Title: {content.title}")
        print(f"Content: {len(content.content)} characters")
        print(f"Website type: {content.website_type}")
```

### Hybrid Chunker

```python
from chunking import HybridChunker

chunker = HybridChunker(max_chunk_size=800, min_chunk_size=100)
chunks = chunker.chunk_text(text, domain='business')

for chunk in chunks:
    print(f"Chunk: {chunk.text[:100]}...")
    print(f"Score: {chunk.semantic_score:.3f}")
```

### Embedding Generator

```python
from embeddings import EmbeddingGenerator

generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
embeddings = generator.generate_embeddings_batch(chunks, domains=['business'] * len(chunks))

# Find similar chunks
similar_chunks = generator.find_similar_chunks(
    embeddings[0], embeddings[1:], top_k=5
)
```

## 📁 Output Structure

```
output/
├── scraped/           # Raw scraped content
│   ├── scraped_https___example_com.json
│   └── ...
├── chunks/            # Text chunks
│   ├── chunks_https___example_com.json
│   └── ...
├── embeddings/        # Generated embeddings
│   ├── embeddings_https___example_com.json
│   └── ...
└── results/           # Complete pipeline results
    ├── result_https___example_com.json
    └── ...
```

## 🎯 Use Cases

### 1. Business Intelligence
- Scrape competitor websites
- Extract product information
- Analyze market trends

### 2. News Analysis
- Monitor news sources
- Extract key information
- Track trending topics

### 3. Research and Education
- Gather Wikipedia articles
- Create knowledge bases
- Support academic research

### 4. Content Analysis
- Analyze content quality
- Extract key insights
- Generate summaries

## 🔍 Performance Optimization

### GPU Acceleration
- The pipeline automatically detects and uses CUDA GPUs
- Optimized batch sizes for P100 GPU
- Memory-efficient processing

### Memory Management
- Configurable batch sizes
- Automatic cleanup of resources
- Intermediate result saving

### Speed Optimization
- Parallel processing where possible
- Optimized Chrome settings
- Efficient embedding generation

## 🛡️ Error Handling

The pipeline includes comprehensive error handling:

- **Network timeouts**: Automatic retry with exponential backoff
- **Website blocking**: User agent rotation and delays
- **Memory issues**: Configurable batch sizes and cleanup
- **Model loading**: Fallback to simpler models
- **Data corruption**: Validation and recovery mechanisms

## 📈 Monitoring and Logging

```python
from loguru import logger

# Configure logging
logger.add("pipeline.log", rotation="10 MB", level="INFO")

# Get pipeline statistics
stats = pipeline.get_pipeline_statistics()
print(f"Success rate: {stats['success_rate']:.2%}")
print(f"Average processing time: {stats['avg_processing_time']:.2f}s")
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Selenium**: Web automation framework
- **Sentence Transformers**: Embedding generation
- **spaCy**: NLP processing
- **NLTK**: Text processing utilities
- **BeautifulSoup**: HTML parsing

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Check the documentation
- Review the example scripts

---

**Built with ❤️ by a Full-Stack Developer with experience at Google, Amazon, and Microsoft**