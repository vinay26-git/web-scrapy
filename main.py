#!/usr/bin/env python3
"""
Main entry point for the Web Scraper project.
Demonstrates usage of the complete pipeline from scraping to embedding generation.
"""
import os
import sys
from typing import List

# Add src to path for imports
sys.path.append('src')

from src.config import Config, WebsiteType
from src.pipeline import WebScrapingPipeline, run_web_scraping_pipeline
from loguru import logger


def main():
    """Main function demonstrating the web scraping pipeline."""
    
    # Configure logging
    logger.remove()  # Remove default logger
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    logger.info("Starting Web Scraper Demo")
    
    # Example URLs covering all three website types
    example_urls = [
        # Business websites
        "https://www.shopify.com/blog/what-is-ecommerce",
        "https://blog.hubspot.com/marketing/what-is-content-marketing",
        
        # News websites  
        "https://www.bbc.com/news/technology",
        "https://techcrunch.com/category/artificial-intelligence/",
        
        # Wikipedia pages
        "https://en.wikipedia.org/wiki/Machine_learning",
        "https://en.wikipedia.org/wiki/Web_scraping",
    ]
    
    logger.info(f"Processing {len(example_urls)} example URLs")
    
    try:
        # Option 1: Using the convenience function (simplest)
        logger.info("=" * 50)
        logger.info("OPTION 1: Using convenience function")
        logger.info("=" * 50)
        
        result = run_web_scraping_pipeline(
            urls=example_urls,
            min_quality_score=40.0,  # Lower threshold for demo
            output_dir="output/demo_run"
        )
        
        # Display results summary
        print_results_summary(result)
        
        # Option 2: Using the full pipeline class (more control)
        logger.info("\n" + "=" * 50)
        logger.info("OPTION 2: Using full pipeline class")
        logger.info("=" * 50)
        
        # Create custom configuration
        config = Config()
        config.chunking.chunk_size = 800  # Smaller chunks
        config.chunking.chunk_overlap = 150
        config.embedding.batch_size = 16
        
        # Initialize pipeline
        pipeline = WebScrapingPipeline(config)
        
        # Run with custom settings
        result2 = pipeline.run_pipeline(
            urls=example_urls[:3],  # Process fewer URLs for demo
            min_quality_score=30.0,
            save_results=True,
            output_dir="output/custom_run"
        )
        
        print_results_summary(result2)
        
        logger.success("Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        return 1
    
    return 0


def print_results_summary(result):
    """Print a formatted summary of pipeline results."""
    stats = result.statistics
    
    print("\n" + "=" * 60)
    print("PIPELINE RESULTS SUMMARY")
    print("=" * 60)
    print(f"Execution Time: {stats['execution_time_formatted']}")
    print(f"Success Rate: {stats['overall_success_rate']:.1%}")
    print()
    print("Processing Steps:")
    print(f"  • URLs Processed: {stats['input_urls']}")
    print(f"  • Content Scraped: {stats['scraped_contents']}")
    print(f"  • Content Processed: {stats['processed_contents']}")
    print(f"  • Chunks Created: {stats['total_chunks']}")
    print(f"  • Embeddings Generated: {stats['total_embeddings']}")
    print()
    
    if stats.get('avg_quality_score'):
        print(f"Average Quality Score: {stats['avg_quality_score']:.1f}/100")
    
    if stats.get('avg_chunk_size_tokens'):
        print(f"Average Chunk Size: {stats['avg_chunk_size_tokens']:.0f} tokens")
    
    print(f"Website Types: {dict(stats['website_types'])}")
    print(f"Languages: {dict(stats['languages'])}")
    print(f"Chunk Methods: {dict(stats['chunk_methods'])}")
    
    if result.embeddings:
        embedding_dim = result.embeddings[0].embedding_dim
        model_name = result.embeddings[0].model_name
        print(f"Embedding Model: {model_name}")
        print(f"Embedding Dimension: {embedding_dim}")
    
    print("=" * 60)


def demo_individual_components():
    """Demonstrate individual component usage."""
    logger.info("Running individual component demos...")
    
    from src.scraper import WebScraper
    from src.preprocessor import TextPreprocessor
    from src.chunker import HybridChunker
    from src.embeddings import EmbeddingGenerator
    
    config = Config()
    
    # Demo scraper
    logger.info("Demo: Web Scraper")
    scraper = WebScraper(config)
    scraped = scraper.scrape_url("https://en.wikipedia.org/wiki/Python_(programming_language)")
    if scraped:
        logger.info(f"Scraped: {scraped.title[:50]}...")
    
    # Demo preprocessor
    if scraped:
        logger.info("Demo: Text Preprocessor")
        preprocessor = TextPreprocessor(config)
        processed = preprocessor.process_content(scraped)
        if processed:
            logger.info(f"Processed {len(processed.sentences)} sentences")
    
    # Demo chunker
    if 'processed' in locals() and processed:
        logger.info("Demo: Hybrid Chunker")
        chunker = HybridChunker(config)
        chunks = chunker.chunk_content(processed)
        logger.info(f"Created {len(chunks)} chunks")
    
    # Demo embeddings
    if 'chunks' in locals() and chunks:
        logger.info("Demo: Embedding Generator")
        embedder = EmbeddingGenerator(config)
        embeddings = embedder.embed_chunks(chunks[:2])  # Just first 2 chunks for demo
        logger.info(f"Generated {len(embeddings)} embeddings")


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    os.makedirs("cache", exist_ok=True)
    
    # Run main demo
    exit_code = main()
    
    # Optionally run individual component demos
    if len(sys.argv) > 1 and sys.argv[1] == "--demo-components":
        demo_individual_components()
    
    sys.exit(exit_code)