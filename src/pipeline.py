"""
Main pipeline module that orchestrates the entire web scraping process.
"""
import time
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import asdict
from loguru import logger
import sys

from .config import PipelineConfig, DEFAULT_CONFIG
from .scraper import WebScraper
from .chunker import HybridChunker, Chunk
from .embeddings import EmbeddingGenerator
from .utils import validate_url, normalize_url, get_text_statistics
import numpy as np


class WebScrapingPipeline:
    """
    Complete web scraping pipeline with chunking and embedding generation.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self._setup_logging()
        self._setup_components()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logger.remove()  # Remove default handler
        logger.add(
            sys.stdout,
            level=self.config.log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        )
        logger.add(
            Path(self.config.output_dir) / "pipeline.log",
            level="DEBUG",
            rotation="10 MB",
            retention="7 days"
        )
    
    def _setup_components(self):
        """Initialize pipeline components."""
        logger.info("Initializing web scraping pipeline components")
        
        # Initialize scraper
        self.scraper = WebScraper(
            config=self.config.scraping,
            website_config=self.config.website
        )
        
        # Initialize chunker
        self.chunker = HybridChunker(config=self.config.chunking)
        
        # Initialize embedding generator
        self.embedding_generator = EmbeddingGenerator(config=self.config.embedding)
        
        logger.info("Pipeline components initialized successfully")
    
    def process_url(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Process a single URL through the entire pipeline.
        
        Args:
            url: URL to process
            
        Returns:
            Dictionary containing processed results
        """
        try:
            logger.info(f"Starting pipeline processing for: {url}")
            
            # Validate and normalize URL
            if not validate_url(url):
                logger.error(f"Invalid URL: {url}")
                return None
            
            url = normalize_url(url)
            
            # Step 1: Scrape the webpage
            scraped_content = self.scraper.scrape_url(url)
            if not scraped_content:
                logger.error(f"Failed to scrape content from {url}")
                return None
            
            logger.info(f"Successfully scraped content from {url}")
            
            # Step 2: Chunk the content
            chunks = self.chunker.chunk_text(
                text=scraped_content['content'],
                metadata={
                    'url': url,
                    'title': scraped_content.get('title', ''),
                    'website_type': scraped_content.get('website_type', 'unknown'),
                    'scraped_at': scraped_content['metadata'].get('scraped_at', time.time()),
                    **scraped_content['metadata']
                }
            )
            
            if not chunks:
                logger.warning(f"No meaningful chunks created for {url}")
                return None
            
            logger.info(f"Created {len(chunks)} chunks from {url}")
            
            # Step 3: Generate embeddings
            chunks_with_embeddings = self.embedding_generator.generate_chunk_embeddings(chunks)
            
            logger.info(f"Generated embeddings for {len(chunks_with_embeddings)} chunks")
            
            # Step 4: Prepare results
            result = {
                'url': url,
                'scraped_content': scraped_content,
                'chunks': [self._chunk_to_dict(chunk) for chunk in chunks_with_embeddings],
                'statistics': self._get_processing_statistics(scraped_content, chunks_with_embeddings),
                'processing_info': {
                    'pipeline_version': '1.0.0',
                    'processed_at': time.time(),
                    'config': asdict(self.config)
                }
            }
            
            logger.info(f"Successfully processed {url} - {len(chunks_with_embeddings)} chunks with embeddings")
            return result
            
        except Exception as e:
            logger.error(f"Error processing {url}: {e}")
            return None
    
    def process_urls(self, urls: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple URLs through the pipeline.
        
        Args:
            urls: List of URLs to process
            
        Returns:
            List of processing results
        """
        results = []
        
        logger.info(f"Starting batch processing of {len(urls)} URLs")
        
        for i, url in enumerate(urls, 1):
            logger.info(f"Processing URL {i}/{len(urls)}: {url}")
            
            result = self.process_url(url)
            if result:
                results.append(result)
            
            # Add delay between requests to be respectful
            if i < len(urls):
                time.sleep(2)
        
        logger.info(f"Batch processing completed. Successfully processed {len(results)}/{len(urls)} URLs")
        return results
    
    def _chunk_to_dict(self, chunk: Chunk) -> Dict[str, Any]:
        """Convert Chunk object to dictionary for serialization."""
        return {
            'text': chunk.text,
            'chunk_id': chunk.chunk_id,
            'start_index': chunk.start_index,
            'end_index': chunk.end_index,
            'metadata': chunk.metadata,
            'embedding': chunk.embedding
        }
    
    def _get_processing_statistics(self, scraped_content: Dict[str, Any], chunks: List[Chunk]) -> Dict[str, Any]:
        """Generate comprehensive processing statistics."""
        # Text statistics
        text_stats = get_text_statistics(scraped_content['content'])
        
        # Chunk statistics
        chunk_stats = self.chunker.get_chunk_statistics(chunks)
        
        # Embedding statistics
        if chunks and chunks[0].embedding:
            embeddings = np.array([chunk.embedding for chunk in chunks])
            embedding_stats = self.embedding_generator.get_embedding_statistics(embeddings)
        else:
            embedding_stats = {'count': 0, 'dimension': 0}
        
        return {
            'text_statistics': text_stats,
            'chunk_statistics': chunk_stats,
            'embedding_statistics': embedding_stats,
            'processing_time': time.time() - scraped_content['metadata'].get('scraped_at', time.time())
        }
    
    def save_results(self, results: List[Dict[str, Any]], output_dir: Optional[str] = None):
        """
        Save processing results to files.
        
        Args:
            results: List of processing results
            output_dir: Output directory (uses config default if None)
        """
        if output_dir is None:
            output_dir = self.config.output_dir
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save individual results
        for i, result in enumerate(results):
            url_hash = result['url'].replace('://', '_').replace('/', '_').replace('.', '_')
            filename = f"result_{i}_{url_hash[:50]}.json"
            filepath = output_path / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved result to {filepath}")
        
        # Save summary
        summary = {
            'total_urls_processed': len(results),
            'successful_urls': len([r for r in results if r is not None]),
            'total_chunks': sum(len(r['chunks']) for r in results if r),
            'processing_timestamp': time.time(),
            'config': asdict(self.config)
        }
        
        summary_file = output_path / "processing_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved processing summary to {summary_file}")
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline configuration and components."""
        return {
            'scraping_config': asdict(self.config.scraping),
            'chunking_config': asdict(self.config.chunking),
            'embedding_config': asdict(self.config.embedding),
            'embedding_dimension': self.embedding_generator.get_embedding_dimension(),
            'device': self.embedding_generator.device,
            'model_name': self.config.embedding.model_name
        }
    
    def cleanup(self):
        """Clean up resources and close connections."""
        logger.info("Cleaning up pipeline resources")
        
        if hasattr(self, 'scraper'):
            self.scraper.close()
        
        logger.info("Pipeline cleanup completed")


def create_pipeline_from_config(config_path: str) -> WebScrapingPipeline:
    """
    Create a pipeline from a configuration file.
    
    Args:
        config_path: Path to configuration JSON file
        
    Returns:
        WebScrapingPipeline instance
    """
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Create config object from JSON
        config = PipelineConfig(**config_data)
        return WebScrapingPipeline(config)
        
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        logger.info("Using default configuration")
        return WebScrapingPipeline()


def run_pipeline_example():
    """Run an example pipeline with sample URLs."""
    # Sample URLs for different website types
    sample_urls = [
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://www.bbc.com/news/technology",
        "https://www.microsoft.com/en-us/about"
    ]
    
    # Create pipeline
    pipeline = WebScrapingPipeline()
    
    try:
        # Process URLs
        results = pipeline.process_urls(sample_urls)
        
        # Save results
        pipeline.save_results(results)
        
        # Print summary
        print(f"\nPipeline completed successfully!")
        print(f"Processed {len(results)} URLs")
        print(f"Total chunks created: {sum(len(r['chunks']) for r in results)}")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
    
    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    run_pipeline_example()