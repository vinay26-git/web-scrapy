# 🚀 Advanced Web Scraper Pipeline

A professional-grade web scraping pipeline that extracts content from websites, applies intelligent text processing, performs hybrid chunking, and generates embeddings optimized for LLM applications.

## 🎯 Project Overview

This project provides a complete pipeline for:
- **Web Scraping**: Selenium-based scraping with site-specific optimizations
- **Text Processing**: Advanced cleaning and preprocessing
- **Hybrid Chunking**: Combines recursive and content-aware chunking techniques
- **Embedding Generation**: GPU-accelerated embedding creation using sentence transformers

### 🎪 Supported Website Types
- **Business Websites**: Corporate sites, product pages, service descriptions
- **News Websites**: Articles, breaking news, technology coverage
- **Wikipedia Pages**: Structured encyclopedia content

## ✨ Key Features

### 🔧 Advanced Web Scraping
- Site-specific content extraction strategies
- Automatic consent banner handling
- Stealth browsing capabilities
- Robust error handling and retries

### 📝 Intelligent Text Processing
- HTML artifact removal
- Unicode normalization
- Site-specific boilerplate removal
- Navigation and UI element filtering
- Key phrase extraction using NLP

### 🧩 Hybrid Chunking System
- **Recursive Chunking**: Size-based segmentation using LangChain
- **Content-Aware Chunking**: Semantic boundary detection
- **Intelligent Selection**: Automatically chooses the best method
- **Quality Evaluation**: Scores chunking approaches for optimization

### 🔢 Optimized Embedding Generation
- Batch processing for efficiency
- GPU acceleration support
- Intelligent caching system
- Content-type specific preprocessing
- Performance monitoring and analysis

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Chrome browser (for Selenium)
- CUDA-compatible GPU (optional, for faster embedding generation)

### Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd web-scraper
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download spaCy model (optional, for enhanced NLP features):**
```bash
python -m spacy download en_core_web_sm
```

4. **Install Chrome WebDriver:**
The webdriver-manager package will automatically download and manage ChromeDriver.

## 🚦 Quick Start

### Basic Usage

```python
from web_scraper_pipeline import WebScraperPipeline, PipelineConfig

# Define URLs to scrape
urls = [
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://www.microsoft.com/en-us/ai",
    "https://techcrunch.com/category/artificial-intelligence/"
]

# Create configuration
config = PipelineConfig(
    urls=urls,
    output_dir="my_output",
    save_intermediate=True,
    use_cache=True
)

# Run the pipeline
pipeline = WebScraperPipeline(config)
results = pipeline.run_pipeline()

print(f"Generated {results.total_embeddings} embeddings in {results.execution_time:.2f}s")
```

### Advanced Configuration

```python
from config import config

# Customize chunking parameters
config.chunking.chunk_size = 1500
config.chunking.chunk_overlap = 300
config.chunking.similarity_threshold = 0.6

# Customize embedding model
config.embedding.model_name = "sentence-transformers/all-mpnet-base-v2"
config.embedding.batch_size = 64

# Run with custom config
pipeline = WebScraperPipeline(pipeline_config)
results = pipeline.run_pipeline()
```

## 📊 Pipeline Stages

### 1. Web Scraping
- **Input**: List of URLs
- **Process**: Site-specific content extraction
- **Output**: Raw HTML content with metadata

### 2. Text Processing
- **Input**: Raw scraped content
- **Process**: Cleaning, normalization, NLP analysis
- **Output**: Clean, processed text with statistics

### 3. Hybrid Chunking
- **Input**: Processed text
- **Process**: Intelligent segmentation using multiple methods
- **Output**: Optimized text chunks with metadata

### 4. Embedding Generation
- **Input**: Text chunks
- **Process**: Batch embedding generation with caching
- **Output**: High-dimensional vectors ready for LLM use

## 📁 Output Structure

The pipeline generates several output files:

```
output_dir/
├── scraped_data.json          # Raw scraped content
├── processed_data.json        # Cleaned and processed text
├── chunked_data.json          # Text chunks with metadata
├── embeddings.pkl             # Binary embedding data
├── embeddings_info.json       # Embedding metadata
├── pipeline_results.json      # Complete execution results
└── execution_summary.txt      # Human-readable summary
```

## 🎮 Example Scripts

### Run Basic Example
```bash
python example_usage.py
```

### Wikipedia-Focused Scraping
```python
from example_usage import example_wikipedia_focused
results = example_wikipedia_focused()
```

### Mixed Content Types
```python
from example_usage import example_mixed_content
results = example_mixed_content()
```

## ⚙️ Configuration

### Scraping Configuration
```python
config.scraping.headless = True
config.scraping.implicit_wait = 10
config.scraping.page_load_timeout = 30
```

### Chunking Configuration
```python
config.chunking.chunk_size = 1000
config.chunking.chunk_overlap = 200
config.chunking.min_chunk_size = 100
config.chunking.max_chunk_size = 1500
```

### Embedding Configuration
```python
config.embedding.model_name = "sentence-transformers/all-MiniLM-L6-v2"
config.embedding.batch_size = 32
config.embedding.normalize_embeddings = True
```

## 📈 Performance Optimization

### For CPU-Only Environments
```python
config.embedding.device = "cpu"
config.embedding.batch_size = 16
```

### For GPU Acceleration
```python
config.embedding.device = "cuda"
config.embedding.batch_size = 64
```

### Memory Management
- Use caching for repeated content
- Adjust batch sizes based on available memory
- Enable intermediate file saving for large datasets

## 🔍 Site-Specific Features

### Wikipedia
- Section header recognition
- Reference removal
- Structured content preservation

### News Sites
- Dateline removal
- Article content extraction
- Advertisement filtering

### Business Sites
- Navigation removal
- Service description extraction
- Contact information filtering

## 📊 Performance Metrics

The pipeline provides comprehensive performance analytics:
- Processing speed (embeddings/second)
- Memory usage statistics
- Cache hit rates
- Stage-wise timing breakdown
- Content type distribution
- Quality metrics for chunking methods

## 🐛 Troubleshooting

### Common Issues

1. **ChromeDriver Issues**
   - Ensure Chrome browser is installed
   - webdriver-manager will auto-download compatible driver

2. **Memory Issues**
   - Reduce batch_size in config
   - Enable intermediate file saving
   - Use CPU instead of GPU for large datasets

3. **Scraping Failures**
   - Check internet connection
   - Verify URLs are accessible
   - Some sites may block automated access

4. **CUDA Issues**
   - Install PyTorch with CUDA support
   - Verify GPU compatibility
   - Fall back to CPU if needed

## 🚀 Advanced Usage

### Custom Content Processors
```python
from text_processor import TextProcessor

processor = TextProcessor()
# Add custom processing logic
```

### Custom Chunking Strategies
```python
from chunker import HybridChunker

chunker = HybridChunker(site_type="custom")
# Implement custom chunking logic
```

### Integration with Vector Databases
```python
# Example: Save to Pinecone
import pinecone

embeddings = pipeline.embedding_data
for emb in embeddings:
    pinecone.upsert(
        id=f"chunk_{emb.chunk_id}",
        values=emb.embedding.tolist(),
        metadata=emb.metadata
    )
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
- **LangChain**: Text splitting utilities
- **Sentence Transformers**: Embedding models
- **spaCy**: Natural language processing
- **Beautiful Soup**: HTML parsing
- **loguru**: Enhanced logging

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the troubleshooting section
- Review the example scripts

---

Built with ❤️ for the AI and ML community. Optimized for production use in Kaggle environments with P100 GPU support.